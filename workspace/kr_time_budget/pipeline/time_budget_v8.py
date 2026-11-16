"""Argon 9双分支：保持54字段路由，检验预算参数化及条件存活率形状。

30共享背景进入两塔；4预算先验仅进入34字段预算参数网；分钟/屏数raw时钟
只进入各自威布尔公式；18内容字段与30共享背景进入48字段内容视图。
两套Omni参数独立。预处理、sum聚合和log1p沿用v3，不在本轮修改。

只做联合训练：prediction=gamma_time*gamma_step*stay，一个BCE更新两塔。
不增加辅助Loss，不预热、不冻结、不detach，不给预算塔加入动态内容反馈。
shape_floor=1.05保留预算gamma递减约束；0.3允许shape<1，放弃该约束，
但两支的累计存活S依然不增。此约束不保证乘上变化的stay后预测也递减。

legacy参数化用于逐值复现v7；log参数化严格按Argon 9的exp(clamp(raw)+1.6)。
bounded_content限定stay=sigmoid(3+0.5*tanh(logit-3))；减3只撤掉末层
初始化bias，使有界/无界分支保持相同初始tensor。它会限制概率表达范围。
"""
import torch
from torch import nn
from torch.nn import functional as F
from rosen.models.msr.base import MSRModel
from rosen.emb.emblayer import OmniEmbLayer, EmbView
from workspace.kr_time_budget.pipeline.kr_request_schema import (
    FEATURES, BUDGET_PARAMETER_FIELDS, CONTENT_TOWER_FIELDS,
)
from rosen.utils.optimize import split_model_params_by_decay


def weibull_parameters(raw, scale_mapping='log', shape_floor=1.05):
    if scale_mapping == 'legacy':
        scale = F.softplus(raw[:, 0])*30.0+0.5
    elif scale_mapping == 'log':
        scale = torch.exp(raw[:, 0].clamp(-2.0, 4.5)+1.6)
    else:
        raise ValueError(scale_mapping)
    return scale, (shape_floor+F.softplus(raw[:, 1])).clamp(max=5.0)


def single_weibull_survival(raw, elapsed, delta, scale_mapping='log', shape_floor=1.05):
    """Stable float64 hazard increments, also valid for 0<shape<1 at t=0."""
    scale, shape = (x.double() for x in weibull_parameters(raw, scale_mapping, shape_floor))
    t, d = elapsed.double().clamp(min=0), delta.double().clamp(min=0)
    safe_t = t.clamp(min=1e-12)
    # When shape<1, differentiating (0/scale)**shape gives 0*inf in PyTorch.
    # Evaluate a positive base, then impose the exact zero boundary value.
    hazard = torch.where(t > 0, (safe_t/scale).pow(shape), torch.zeros_like(t))
    increment_positive = (safe_t/scale).pow(shape)*torch.expm1(shape*torch.log1p(d/safe_t))
    origin_increment = torch.where(d > 0, (d.clamp(min=1e-12)/scale).pow(shape), torch.zeros_like(d))
    increment = torch.where(t > 0, increment_positive, origin_increment)
    log_s = -hazard
    return (-increment).exp().to(raw.dtype), log_s.exp().to(raw.dtype), log_s


class DualBranchModel(MSRModel):
    def __init__(self, manager, mode='ours', scale_mapping='log', shape_floor=1.05,
                 content_bias=3.0, bounded_content=False):
        if mode not in {'ours', 'fusion'} or scale_mapping not in {'legacy', 'log'}:
            raise ValueError((mode, scale_mapping))
        if shape_floor not in {0.3, 1.05} or (bounded_content and content_bias != 3.0):
            raise ValueError((shape_floor, content_bias, bounded_content))
        if {s.field_name for s in manager.settings} != set(FEATURES):
            raise ValueError('DualBranchModel requires the KR 54-field schema')
        super().__init__(manager, num_domains=5)
        self.mode, self.scale_mapping, self.shape_floor = mode, scale_mapping, shape_floor
        self.content_bias, self.bounded_content = content_bias, bounded_content
        self.training_strategy, self.stage = 'joint', 2
        budget_settings = self.omni_embedding.get_settings(include_fields=BUDGET_PARAMETER_FIELDS)
        self.budget_embedding = OmniEmbLayer(emb_settings=budget_settings)
        for name, table in self.budget_embedding.emb_modules.items():
            table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
        self.budget_view = EmbView(self.budget_embedding, include_fields=BUDGET_PARAMETER_FIELDS)
        self.content_view = EmbView(self.omni_embedding, include_fields=CONTENT_TOWER_FIELDS)
        self.fusion_view = self.omni_embedding.whole
        # 与v7相同模块构造顺序：各方案只有明确的bias变化，其他初始tensor相同。
        self.budget_net = nn.Sequential(nn.Linear(self.budget_view.embedding_dim, 64), nn.SiLU(), nn.Linear(64, 4))
        self.content_net = nn.Sequential(nn.Linear(self.content_view.embedding_dim, 64), nn.SiLU(),
                                         nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        self.fusion_net = nn.Sequential(nn.Linear(self.fusion_view.embedding_dim, 64), nn.SiLU(),
                                        nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.constant_(self.content_net[-1].bias, content_bias)
        self.register_buffer('prior_stay', torch.sigmoid(torch.tensor(content_bias)))

    @property
    def budget_modules(self):
        return self.budget_embedding, self.budget_net

    def budget_raw(self, batch):
        return self.budget_net(self.budget_view(batch))

    def clock_components(self, batch):
        raw = self.budget_raw(batch)
        time = batch['elapsed_min_raw'].float().clamp(min=0)
        step = (batch['req_idx_raw'].float()-1).clamp(min=0)
        return {name: single_weibull_survival(head, t, torch.ones_like(t), self.scale_mapping, self.shape_floor)
                for name, head, t in [('time', raw[:, :2], time), ('step', raw[:, 2:], step)]}

    def budget_components(self, batch):
        clocks = self.clock_components(batch)
        gamma = clocks['time'][0]*clocks['step'][0]
        log_s = clocks['time'][2]+clocks['step'][2]
        return gamma, log_s.exp().to(gamma.dtype), log_s

    def content_stay(self, logit):
        if self.bounded_content:
            logit = 3.0+0.5*torch.tanh(logit-3.0)
        return torch.sigmoid(logit)

    def components(self, batch):
        gamma, survival, log_s = self.budget_components(batch)
        stay = self.content_stay(self.content_net(self.content_view(batch)).squeeze(-1))
        return gamma*stay, gamma, stay, survival, log_s

    def predict(self, batch):
        if self.mode == 'fusion':
            return torch.sigmoid(self.fusion_net(self.fusion_view(batch)).squeeze(-1))
        return self.components(batch)[0]

    def calculate_loss(self, batch):
        return F.binary_cross_entropy(self.predict(batch).clamp(1e-6, 1-1e-6), batch['y_continue'].float())


def joint_parameter_groups(model, budget_lr):
    """Native decay grouping, then disjoint budget/content LR groups; one optimizer."""
    budget_ids = {id(p) for module in model.budget_modules for p in module.parameters()}
    groups = []
    for decay_group in split_model_params_by_decay(model, weight_decay=1e-6):
        for scope in ['budget', 'other']:
            params = [p for p in decay_group['params'] if (id(p) in budget_ids) == (scope == 'budget')]
            if params:
                groups.append(dict(params=params, weight_decay=decay_group['weight_decay'],
                                   lr=budget_lr if scope == 'budget' else .001, scope=scope))
    ids = [id(p) for group in groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    return groups
