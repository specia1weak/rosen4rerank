"""Argon 12: K=2 mixtures, then step-only budget plus average dwell pace.

30 shared background + 4 budget priors feed the independent 34-field budget Omni;
48-field content Omni remains unchanged. Stage-2 experiment add_pace explicitly adds
one derived network feature log1p(elapsed_min_raw/req_idx_raw) to content (or Fusion).
It is computed at the current screen-end decision, using no future duration; no schema
refit or parquet rewrite. This route deliberately relaxes the old content clock exclusion.

Mixture: two Weibulls per clock, original capped exp scale and shape in [0.3,5].
Priors depend only on opening background; posterior regime weights condition on survival.
gamma = S_mix(t+1)/S_mix(t), computed via logsumexp of survival-posterior*component ratios.
Never average conditional ratios with fixed prior weights. Independent clocks multiply.
One final product BCE jointly updates all active paths, no auxiliary loss or freezing.
"""
import torch
from torch import nn
from torch.nn import functional as F
from rosen.models.msr.base import MSRModel
from rosen.emb.emblayer import OmniEmbLayer, EmbView
from workspace.kr_time_budget.pipeline.kr_request_schema import (
    FEATURES, BUDGET_PARAMETER_FIELDS, CONTENT_TOWER_FIELDS,
)


from workspace.kr_time_budget.pipeline.time_budget_v9 import single_weibull_survival, weibull_parameters


def pace_feature(batch):
    """Current-decision screen-end minutes/screen; not next-screen or future duration."""
    return torch.log1p(batch['elapsed_min_raw'].float().clamp(min=0)/batch['req_idx_raw'].float().clamp(min=1))


def mixture_parts(raw, elapsed):
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


class MixtureTimeBudgetModel(MSRModel):
    def __init__(self, manager, mode='ours', mixture=False, step_only=False, add_pace=False):
        if mode not in {'ours', 'fusion'}:
            raise ValueError(mode)
        if {s.field_name for s in manager.settings} != set(FEATURES):
            raise ValueError('MixtureTimeBudgetModel requires the KR 54-field schema')
        super().__init__(manager, num_domains=5)
        self.mode, self.mixture, self.step_only, self.add_pace = mode, mixture, step_only, add_pace
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
        x = view(batch)
        return torch.cat([x, pace_feature(batch).unsqueeze(-1)], dim=-1) if self.add_pace else x

    def budget_components(self, batch):
        clocks = self.clock_components(batch)
        gamma = clocks['time'][0]*clocks['step'][0]
        log_s = clocks['time'][2]+clocks['step'][2]
        return gamma, log_s.exp().to(gamma.dtype), log_s

    def components(self, batch):
        gamma, survival, log_s = self.budget_components(batch)
        stay = torch.sigmoid(self.content_net(self.network_view(batch, self.content_view)).squeeze(-1))
        return gamma*stay, gamma, stay, survival, log_s

    def predict(self, batch):
        if self.mode == 'fusion':
            return torch.sigmoid(self.fusion_net(self.network_view(batch, self.fusion_view)).squeeze(-1))
        return self.components(batch)[0]

    def calculate_loss(self, batch):
        return F.binary_cross_entropy(self.predict(batch).clamp(1e-6, 1-1e-6), batch['y_continue'].float())
