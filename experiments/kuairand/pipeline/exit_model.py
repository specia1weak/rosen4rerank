"""当前KR离开预测模型：特征路由、威布尔预算与联合乘积BCE。

阅读顺序：kr_request_schema.py -> exit_model.py -> exit_training.py。
34字段预算参数网 = 30共享背景 + 4预算先验；48字段内容网 = 30共享背景 + 18内容。
两塔使用独立OmniEmbLayer，由EmbView显式选字段；共享背景包含用户、tab和开场时间。
当前plan13双塔只启用屏数时钟(req_idx_raw - 1)，分钟gamma固定1。
_pace配置另向内容塔追加log1p(elapsed_min_raw / max(req_idx_raw, 1))，放宽时钟排除。
历史零填充、sum标签聚合、未统一缩放的数值特征沿用v3，未声称消除进度代理。

混合预算以S_mix(t+1)/S_mix(t)计算gamma，后验权重随存活更新。
累计S不增；shape允许小于1，gamma与最终prediction均不承诺单调递减。
只有BCE(gamma * content_stay, y_continue)，两塔同时反传；没有辅助Loss或冻结。

此文件从plan12/13实现提炼，保持state_dict键、构造顺序、初始化和数学运算。
为精确复现保留原有未启用的head及prior_stay buffer，不将其误认作活跃训练分支。
历史time_budget_v*.py和run_plan13_ablation.py保持原样，供已有run源码哈希审计。
"""
import torch
from torch import nn
from torch.nn import functional as F
from rosen.models.msr.base import MSRModel
from rosen.emb.emblayer import OmniEmbLayer, EmbView
from experiments.kuairand.pipeline.kr_request_schema import (
    FEATURES, BUDGET_PARAMETER_FIELDS, CONTENT_TOWER_FIELDS,
)


def weibull_parameters(raw, capped):
    """原始输出 -> 正尺度lambda与shape；当前实验保留原有截断范围。"""
    log_scale = raw[:, 0].clamp(-2.0, 4.5) if capped else raw[:, 0]
    scale = torch.exp(log_scale+1.6)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise FloatingPointError('Weibull scale must be finite and positive')
    return scale, (.3+F.softplus(raw[:, 1])).clamp(max=5.)


def single_weibull_survival(raw, elapsed, delta, capped):
    """返回gamma=S(t+delta)/S(t)、S(t)、log S(t)，在float64中算风险增量。"""
    scale, shape = (x.double() for x in weibull_parameters(raw, capped))
    t, d = elapsed.double().clamp(min=0), delta.double().clamp(min=0)
    safe_t = t.clamp(min=1e-12)
    hazard = torch.where(t > 0, (safe_t/scale).pow(shape), torch.zeros_like(t))
    increment_positive = (safe_t/scale).pow(shape)*torch.expm1(shape*torch.log1p(d/safe_t))
    origin_increment = torch.where(d > 0, (d.clamp(min=1e-12)/scale).pow(shape), torch.zeros_like(d))
    increment = torch.where(t > 0, increment_positive, origin_increment)
    log_s = -hazard
    return (-increment).exp().to(raw.dtype), log_s.exp().to(raw.dtype), log_s


def pace_feature(batch):
    """Current-decision screen-end minutes/screen; not next-screen or future duration."""
    return torch.log1p(batch['elapsed_min_raw'].float().clamp(min=0)/batch['req_idx_raw'].float().clamp(min=1))


def mixture_parts(raw, elapsed):
    """K=2：前4列为两个分量参数，后2列为先验logits。

    posterior_k(t) = pi_k * S_k(t) / S_mix(t)
    gamma_mix(t) = sum_k posterior_k(t) * gamma_k(t)
    禁止直接用固定pi加权gamma_k；那不是混合分布的条件存活率。
    """
    t = elapsed.double().clamp(min=0)
    heads = raw[:, :4].reshape(-1, 2, 2)
    flat = heads.reshape(-1, 2)
    # Compute component log-gamma in float64 directly, avoiding float32 underflow/log(0).
    scale, shape = (x.double().reshape(-1, 2) for x in weibull_parameters(flat, True))
    safe_t = t[:, None].clamp(min=1e-12)
    log_s = -torch.where(t[:, None] > 0, (safe_t/scale).pow(shape), torch.zeros_like(scale))
    positive = (safe_t/scale).pow(shape)*torch.expm1(shape*torch.log1p(1/safe_t))
    increment = torch.where(t[:, None] > 0, positive, (1/scale).pow(shape))
    log_pi = torch.log_softmax(raw[:, 4:6].double(), dim=-1)
    log_mix_s = torch.logsumexp(log_pi+log_s, dim=-1)
    log_posterior = log_pi+log_s-log_mix_s[:, None]
    log_gamma = torch.logsumexp(log_posterior-increment, dim=-1)
    return dict(gamma=log_gamma.exp().to(raw.dtype), log_survival=log_mix_s,
                prior=log_pi.exp(), posterior=log_posterior.exp(),
                component_log_gamma=-increment, scale=scale, shape=shape)


def mixture_weibull_survival(raw, elapsed):
    parts = mixture_parts(raw, elapsed)
    log_s = parts['log_survival']
    return parts['gamma'], log_s.exp().to(raw.dtype), log_s


class KRExitModel(MSRModel):
    """Ours乘积模型与Fusion对照，共用明确的字段契约和构造顺序。

    mode='ours'：budget_view -> budget_net -> Weibull gamma；
                 content_view (+pace) -> content_net -> stay；prediction=gamma*stay。
    mode='fusion'：fusion_view (+pace) -> fusion_net -> prediction。
    """
    def __init__(self, manager, mode='ours', mixture=False, step_only=False, add_pace=False):
        if mode not in {'ours', 'fusion'}:
            raise ValueError(mode)
        if {s.field_name for s in manager.settings} != set(FEATURES):
            raise ValueError('KRExitModel requires the KR 54-field schema')
        super().__init__(manager, num_domains=5)
        self.mode, self.mixture, self.step_only, self.add_pace = mode, mixture, step_only, add_pace
        # stage=2仅兼容历史日志字段；本类从第一步起就是联合训练，没有阶段切换。
        self.training_strategy, self.stage = 'joint', 2

        # 1. 两套独立Omni：同名背景字段从同一初值出发，训练参数互不共享。
        budget_settings = self.omni_embedding.get_settings(include_fields=BUDGET_PARAMETER_FIELDS)
        self.budget_embedding = OmniEmbLayer(emb_settings=budget_settings)
        for name, table in self.budget_embedding.emb_modules.items():
            table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
        self.budget_view = EmbView(self.budget_embedding, include_fields=BUDGET_PARAMETER_FIELDS)
        self.content_view = EmbView(self.omni_embedding, include_fields=CONTENT_TOWER_FIELDS)
        self.fusion_view = self.omni_embedding.whole

        # 2. 先构造共有模块，再扩展head/pace，避免改变共有参数的随机初始化。
        self.budget_net = nn.Sequential(nn.Linear(self.budget_view.embedding_dim, 64), nn.SiLU(), nn.Linear(64, 4))
        self.content_net = nn.Sequential(nn.Linear(self.content_view.embedding_dim, 64), nn.SiLU(),
                                         nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        self.fusion_net = nn.Sequential(nn.Linear(self.fusion_view.embedding_dim, 64), nn.SiLU(),
                                        nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.constant_(self.content_net[-1].bias, 3.)
        self.register_buffer('prior_stay', torch.sigmoid(torch.tensor(3.)))
        # Build common modules before replacing any head: don't shift common RNG state.
        if mixture:
            original = self.budget_net[-1]
            with torch.random.fork_rng():
                expanded = nn.Linear(original.in_features, 12)
            with torch.no_grad():
                # Six outputs per clock: [scale0,shape0,scale1,shape1,gate0,gate1].
                for clock in range(2):
                    for component, shift in enumerate([-0.6931471805599453, 0.6931471805599453]):
                        target = slice(clock*6+component*2, clock*6+component*2+2)
                        expanded.weight[target].copy_(original.weight[clock*2:clock*2+2])
                        expanded.bias[target].copy_(original.bias[clock*2:clock*2+2])
                        expanded.bias[clock*6+component*2].add_(shift)
                    expanded.weight[clock*6+4:clock*6+6].zero_()
                    expanded.bias[clock*6+4:clock*6+6].zero_()
            self.budget_net[-1] = expanded
        if add_pace:
            # Zero new column preserves initial function; it receives a gradient immediately.
            for network in [self.content_net, self.fusion_net]:
                old = network[0]
                with torch.random.fork_rng():
                    expanded = nn.Linear(old.in_features+1, old.out_features)
                with torch.no_grad():
                    expanded.weight[:, :-1].copy_(old.weight)
                    expanded.weight[:, -1].zero_()
                    expanded.bias.copy_(old.bias)
                network[0] = expanded

    @property
    def budget_modules(self):
        return self.budget_embedding, self.budget_net

    def budget_raw(self, batch):
        return self.budget_net(self.budget_view(batch))

    def clock_components(self, batch):
        """分钟/屏数保持量纲分离；step_only将分钟分支替换为恒等存活率。"""
        raw = self.budget_raw(batch)
        time = batch['elapsed_min_raw'].float().clamp(min=0)
        step = (batch['req_idx_raw'].float()-1).clamp(min=0)
        if self.mixture:
            time_result = mixture_weibull_survival(raw[:, :6], time)
            step_result = mixture_weibull_survival(raw[:, 6:], step)
        else:
            time_result = single_weibull_survival(raw[:, :2], time, torch.ones_like(time), True)
            step_result = single_weibull_survival(raw[:, 2:], step, torch.ones_like(step), True)
        if self.step_only:
            time_result = (torch.ones_like(time), torch.ones_like(time), torch.zeros_like(time, dtype=torch.float64))
        return {'time': time_result, 'step': step_result}

    def network_view(self, batch, view):
        """pace仅追加到内容/Fusion网络，不进入预算参数网。"""
        x = view(batch)
        return torch.cat([x, pace_feature(batch).unsqueeze(-1)], dim=-1) if self.add_pace else x

    def budget_components(self, batch):
        clocks = self.clock_components(batch)
        gamma = clocks['time'][0]*clocks['step'][0]
        log_s = clocks['time'][2]+clocks['step'][2]
        return gamma, log_s.exp().to(gamma.dtype), log_s

    def components(self, batch):
        """返回prediction、gamma、stay、S、log S；保持两塔计算图相连。"""
        gamma, survival, log_s = self.budget_components(batch)
        stay = torch.sigmoid(self.content_net(self.network_view(batch, self.content_view)).squeeze(-1))
        return gamma*stay, gamma, stay, survival, log_s

    def predict(self, batch):
        if self.mode == 'fusion':
            return torch.sigmoid(self.fusion_net(self.network_view(batch, self.fusion_view)).squeeze(-1))
        return self.components(batch)[0]

    def calculate_loss(self, batch):
        return F.binary_cross_entropy(self.predict(batch).clamp(1e-6, 1-1e-6), batch['y_continue'].float())
