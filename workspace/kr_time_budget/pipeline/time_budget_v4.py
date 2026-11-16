"""Argon 5: single IFR Weibull and an exactly continuous content transition."""
import torch
from torch import nn
from torch.nn import functional as F
from workspace.kr_time_budget.pipeline.time_budget_v3 import ExitModelV3

PRIOR_STAY_LOGIT = 2.6


def weibull_parameters(raw):
    return F.softplus(raw[:, 0]) * 30.0 + 0.5, (1.05 + F.softplus(raw[:, 1])).clamp(max=5.0)


def single_weibull_survival(raw, elapsed, delta):
    """Stable S(t+d)/S(t), without subtracting two large cumulative hazards.

    For t>0, H(t+d)-H(t) = H(t)*expm1(z*log1p(d/t)).
    Work in float64 for this small scalar path; model probabilities remain float32.
    """
    scale, shape = (x.double() for x in weibull_parameters(raw))
    t, d = elapsed.double().clamp(min=0), delta.double().clamp(min=0)
    safe_t = t.clamp(min=1e-12)
    hazard = (t / scale).pow(shape)
    increment_positive = (safe_t / scale).pow(shape) * torch.expm1(shape * torch.log1p(d / safe_t))
    increment = torch.where(t > 0, increment_positive, (d / scale).pow(shape))
    log_s = -hazard
    gamma = (-increment).exp().to(raw.dtype)
    return gamma, log_s.exp().to(raw.dtype), log_s


class ExitModelV4(ExitModelV3):
    def __init__(self, manager, mode='ours'):
        # Construct the original common modules first: unchanged baselines keep exact initial tensors.
        super().__init__(manager, mode)
        original = self.budget_net[-1]
        with torch.random.fork_rng():
            final = nn.Linear(original.in_features, 2)
        with torch.no_grad():
            final.weight.copy_(original.weight[:2])
            final.bias.copy_(original.bias[:2])
        self.budget_net[-1] = final
        self.register_buffer('prior_stay', torch.sigmoid(torch.tensor(PRIOR_STAY_LOGIT)))

    def budget_components(self, b):
        return single_weibull_survival(self.budget_raw(b), b['elapsed_min'], torch.ones_like(b['elapsed_min']))

    def predict(self, b):
        if self.mode == 'budget' or (self.mode == 'ours' and self.stage == 1):
            return self.budget_components(b)[0] * self.prior_stay
        return super().predict(b)

    def initialize_stage_two(self, snapshot):
        with torch.random.fork_rng():
            model = ExitModelV4(self.manager, self.mode)
        model.load_state_dict(snapshot.model_state)
        # This head was unused in warmup. At stage-two entry it must be exactly constant s0.
        nn.init.zeros_(model.content_net[-1].weight)
        nn.init.constant_(model.content_net[-1].bias, PRIOR_STAY_LOGIT)
        model.set_stage(2)
        return model
