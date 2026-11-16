"""Argon 10：以Pooled AUC/LogLoss为主线的双时钟联合模型。

沿用v8 flexible_shape：34字段预算参数网、48字段内容塔、两套独立Omni；
30共享背景进入两塔，4预算先验仅进预算网，两个raw时钟只进各自公式，
18内容字段仅进内容塔。数据处理与schema不变。只优化gamma_time*gamma_step
*stay的一个BCE，不使用User-GAUC训练目标、辅助Loss、预热或冻结。

scale_caps='both'逐值复现v8 flexible_shape；'step'只保留屏数尺度截断；
'none'去掉两个尺度的上下硬截断。未截断的lambda=exp(raw+1.6)>0，
不再偷偷加另一个尺度上界；保留shape=clamp(0.3+softplus(raw),max=5)。
所有模式构造完全相同的模块，保证全state初始化逐值相同。
"""
import torch
from torch import nn
from torch.nn import functional as F
from rosen.models.msr.base import MSRModel
from rosen.emb.emblayer import OmniEmbLayer, EmbView
from workspace.kr_time_budget.pipeline.kr_request_schema import (
    FEATURES, BUDGET_PARAMETER_FIELDS, CONTENT_TOWER_FIELDS,
)


def weibull_parameters(raw, capped):
    log_scale = raw[:, 0].clamp(-2.0, 4.5) if capped else raw[:, 0]
    scale = torch.exp(log_scale+1.6)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise FloatingPointError('Nonfinite/zero uncapped scale; stop rather than silently cap it')
    return scale, (.3+F.softplus(raw[:, 1])).clamp(max=5.)


def single_weibull_survival(raw, elapsed, delta, capped):
    scale, shape = (x.double() for x in weibull_parameters(raw, capped))
    t, d = elapsed.double().clamp(min=0), delta.double().clamp(min=0)
    safe_t = t.clamp(min=1e-12)
    hazard = torch.where(t > 0, (safe_t/scale).pow(shape), torch.zeros_like(t))
    increment_positive = (safe_t/scale).pow(shape)*torch.expm1(shape*torch.log1p(d/safe_t))
    origin_increment = torch.where(d > 0, (d.clamp(min=1e-12)/scale).pow(shape), torch.zeros_like(d))
    increment = torch.where(t > 0, increment_positive, origin_increment)
    log_s = -hazard
    return (-increment).exp().to(raw.dtype), log_s.exp().to(raw.dtype), log_s


class PooledTimeBudgetModel(MSRModel):
    def __init__(self, manager, mode='ours', scale_caps='none'):
        if mode not in {'ours', 'fusion'} or scale_caps not in {'both', 'step', 'none'}:
            raise ValueError((mode, scale_caps))
        if {s.field_name for s in manager.settings} != set(FEATURES):
            raise ValueError('PooledTimeBudgetModel requires the KR 54-field schema')
        super().__init__(manager, num_domains=5)
        self.mode, self.scale_caps = mode, scale_caps
        self.training_strategy, self.stage = 'joint', 2
        budget_settings = self.omni_embedding.get_settings(include_fields=BUDGET_PARAMETER_FIELDS)
        self.budget_embedding = OmniEmbLayer(emb_settings=budget_settings)
        for name, table in self.budget_embedding.emb_modules.items():
            table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
        self.budget_view = EmbView(self.budget_embedding, include_fields=BUDGET_PARAMETER_FIELDS)
        self.content_view = EmbView(self.omni_embedding, include_fields=CONTENT_TOWER_FIELDS)
        self.fusion_view = self.omni_embedding.whole
        self.budget_net = nn.Sequential(nn.Linear(self.budget_view.embedding_dim, 64), nn.SiLU(), nn.Linear(64, 4))
        self.content_net = nn.Sequential(nn.Linear(self.content_view.embedding_dim, 64), nn.SiLU(),
                                         nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        self.fusion_net = nn.Sequential(nn.Linear(self.fusion_view.embedding_dim, 64), nn.SiLU(),
                                        nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.constant_(self.content_net[-1].bias, 3.)
        self.register_buffer('prior_stay', torch.sigmoid(torch.tensor(3.)))

    @property
    def budget_modules(self):
        return self.budget_embedding, self.budget_net

    def budget_raw(self, batch):
        return self.budget_net(self.budget_view(batch))

    def is_capped(self, clock):
        return self.scale_caps == 'both' or (self.scale_caps == 'step' and clock == 'step')

    def clock_components(self, batch):
        raw = self.budget_raw(batch)
        time = batch['elapsed_min_raw'].float().clamp(min=0)
        step = (batch['req_idx_raw'].float()-1).clamp(min=0)
        return {name: single_weibull_survival(head, t, torch.ones_like(t), self.is_capped(name))
                for name, head, t in [('time', raw[:, :2], time), ('step', raw[:, 2:], step)]}

    def budget_components(self, batch):
        clocks = self.clock_components(batch)
        gamma = clocks['time'][0]*clocks['step'][0]
        log_s = clocks['time'][2]+clocks['step'][2]
        return gamma, log_s.exp().to(gamma.dtype), log_s

    def components(self, batch):
        gamma, survival, log_s = self.budget_components(batch)
        stay = torch.sigmoid(self.content_net(self.content_view(batch)).squeeze(-1))
        return gamma*stay, gamma, stay, survival, log_s

    def predict(self, batch):
        if self.mode == 'fusion':
            return torch.sigmoid(self.fusion_net(self.fusion_view(batch)).squeeze(-1))
        return self.components(batch)[0]

    def calculate_loss(self, batch):
        return F.binary_cross_entropy(self.predict(batch).clamp(1e-6, 1-1e-6), batch['y_continue'].float())
