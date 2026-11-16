"""基于OmniEmbLayer/EmbView的30共享＋6预算＋18内容模型。

预算Omni独立持有34字段的表，内容视图从主Omni选择48字段；两塔共享
字段含义但不共享参数。Fusion视图读取54字段。视图原生提供维度、字段切片。

Ours默认training_strategy='joint'：gamma×stay的一个BCE同时更新两塔，
不detach、不冻结、不独立监督两个概率。'two_stage'显式保留原冻结预算对照。

clock='time'与'dual'使用完全相同的模块和初始化，预算末层固定4维；单时钟
仅用前两维。双时钟分别使用分钟和屏数，不把屏数换算成分钟。现有稳定
威布尔公式保持原始输出，仅在BCE入口做数值保护。

raw时钟必须由数据链路明确提供：elapsed_min_raw（分钟）、req_idx_raw（屏）。
本模型不隐式回退到可能已缩放的elapsed_min，也不新增数值变换。
"""
import torch
from torch import nn
from torch.nn import functional as F
from rosen.models.msr.base import MSRModel
from rosen.emb.emblayer import OmniEmbLayer, EmbView
from workspace.kr_time_budget.pipeline.kr_request_schema import (
    FEATURES, BUDGET_PARAMETER_FIELDS, CONTENT_TOWER_FIELDS,
)
from workspace.kr_time_budget.pipeline.time_budget_v4 import single_weibull_survival

PRIOR_STAY_LOGIT = 3.5


class TimeBudgetModel(MSRModel):
    def __init__(self, manager, mode='ours', clock='dual', training_strategy='joint'):
        if mode not in {'ours', 'budget', 'content', 'fusion'}:
            raise ValueError(f'Unknown model mode: {mode}')
        if clock not in {'time', 'dual'}:
            raise ValueError(f'Unknown clock: {clock}')
        if training_strategy not in {'joint', 'two_stage'}:
            raise ValueError(f'Unknown training strategy: {training_strategy}')
        if {s.field_name for s in manager.settings} != set(FEATURES):
            raise ValueError('TimeBudgetModel requires the KR 54-field schema')
        super().__init__(manager, num_domains=5)
        self.mode = mode
        self.clock = clock
        self.training_strategy = training_strategy
        self.stage = 1 if mode == 'ours' and training_strategy == 'two_stage' else 2

        # 从主Omni选择原生settings，不手写编码器，也不按FeatureSource猜业务路由。
        budget_settings = self.omni_embedding.get_settings(include_fields=BUDGET_PARAMETER_FIELDS)
        self.budget_embedding = OmniEmbLayer(emb_settings=budget_settings)
        for name, table in self.budget_embedding.emb_modules.items():
            table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
        self.budget_view = EmbView(self.budget_embedding, include_fields=BUDGET_PARAMETER_FIELDS)
        self.content_view = EmbView(self.omni_embedding, include_fields=CONTENT_TOWER_FIELDS)
        self.fusion_view = self.omni_embedding.whole

        # 两种时钟和四种mode都按相同顺序构造模块，保持共有初始tensor精确相同。
        self.budget_net = nn.Sequential(
            nn.Linear(self.budget_view.embedding_dim, 64), nn.SiLU(), nn.Linear(64, 4),
        )
        self.content_net = nn.Sequential(
            nn.Linear(self.content_view.embedding_dim, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1),
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(self.fusion_view.embedding_dim, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1),
        )
        nn.init.constant_(self.content_net[-1].bias, PRIOR_STAY_LOGIT)
        self.register_buffer('prior_stay', torch.sigmoid(torch.tensor(PRIOR_STAY_LOGIT)))

    @property
    def budget_modules(self):
        return self.budget_embedding, self.budget_net

    def set_stage(self, stage):
        if stage not in {1, 2}:
            raise ValueError('stage must be 1 or 2')
        if self.training_strategy == 'joint' and stage != 2:
            raise ValueError('Joint training has no budget-only warmup stage')
        self.stage = stage
        for module in self.budget_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(not (self.mode == 'ours' and self.training_strategy == 'two_stage' and stage == 2))
                parameter.grad = None

    def budget_raw(self, batch):
        return self.budget_net(self.budget_view(batch))

    def clock_components(self, batch):
        """返回各时钟的(gamma, S, logS)，便于断点观察和独立核对。"""
        raw = self.budget_raw(batch)
        minutes = batch['elapsed_min_raw'].float().clamp(min=0)
        clocks = {'time': single_weibull_survival(raw[:, :2], minutes, torch.ones_like(minutes))}
        if self.clock == 'dual':
            steps = (batch['req_idx_raw'].float() - 1).clamp(min=0)
            clocks['step'] = single_weibull_survival(raw[:, 2:], steps, torch.ones_like(steps))
        return clocks

    def budget_components(self, batch):
        clocks = self.clock_components(batch)
        gamma, survival, log_survival = clocks['time']
        if 'step' in clocks:
            gamma_step, _, log_step = clocks['step']
            gamma = gamma * gamma_step
            log_survival = log_survival + log_step
            survival = log_survival.exp().to(gamma.dtype)
        return gamma, survival, log_survival

    def content_x(self, batch):
        return self.content_view(batch)

    def components(self, batch):
        gamma, survival, log_survival = self.budget_components(batch)
        if self.mode == 'ours' and self.training_strategy == 'two_stage' and self.stage == 2:
            gamma = gamma.detach()
        stay = torch.sigmoid(self.content_net(self.content_view(batch)).squeeze(-1))
        return gamma * stay, gamma, stay, survival, log_survival

    def predict(self, batch):
        if self.mode == 'fusion':
            return torch.sigmoid(self.fusion_net(self.fusion_view(batch)).squeeze(-1))
        if self.mode == 'content':
            return torch.sigmoid(self.content_net(self.content_view(batch)).squeeze(-1))
        if self.mode == 'budget' or self.stage == 1:
            return self.budget_components(batch)[0] * self.prior_stay
        return self.components(batch)[0]

    def calculate_loss(self, batch):
        return F.binary_cross_entropy(
            self.predict(batch).clamp(1e-6, 1 - 1e-6), batch['y_continue'].float(),
        )

    def should_finish_stage_one(self, report):
        if self.training_strategy != 'two_stage':
            raise ValueError('Use an ordinary Trainer for joint training, not TwoStageRunner')
        return report.epoch >= 1

    def initialize_stage_two(self, snapshot):
        if self.training_strategy != 'two_stage':
            raise ValueError('Joint training does not initialize a frozen-budget second stage')
        with torch.random.fork_rng():
            model = TimeBudgetModel(self.manager, self.mode, self.clock, training_strategy='two_stage')
        model.load_state_dict(snapshot.model_state)
        nn.init.zeros_(model.content_net[-1].weight)
        nn.init.constant_(model.content_net[-1].bias, PRIOR_STAY_LOGIT)
        model.set_stage(2)
        return model
