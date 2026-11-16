"""Argon offline exit models. Both Ours towers consume the same encoded X_all."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from rosen.models.msr.base import MSRModel
from workspace.kr_time_budget.pipeline.features import B1, B2, B3, B4, CATS


def mixture_survival(raw, elapsed, delta):
    raw = raw.reshape(-1, 2, 3)
    log_pi = F.log_softmax(raw[:, :, 0], dim=-1)
    scale = F.softplus(raw[:, :, 1]) * 30 + .1
    shape = (F.softplus(raw[:, :, 2]) + .5).clamp(max=5)
    def log_s(time):
        t = time[:, None].clamp(min=0)
        log_hazard = (shape * (t.clamp(min=1e-8).log() - scale.log())).clamp(max=math.log(1e6))
        hazard = torch.where(t == 0, torch.zeros_like(log_hazard), log_hazard.exp())
        return torch.logsumexp(log_pi - hazard, dim=-1)
    current = log_s(elapsed)
    future = log_s(elapsed + delta.clamp(min=0))
    # Subtract log survivals before exponentiation; no independently floored S values.
    gamma = (future - current).clamp(max=0).exp()
    return gamma, current.exp(), current


class ExitModel(MSRModel):
    def __init__(self, manager, mode='ours'):
        super().__init__(manager, num_domains=1)
        self.mode = mode
        self.cat_dim = 4 * len(CATS)
        self.numeric = B1 + B2 + B3 + B4
        self.dim = len(self.numeric) + self.cat_dim + 8 * 4 + 6
        # Identical module construction in all modes makes initial common tensors exact.
        self.slate_attention = nn.Linear(16, 1)
        self.history_attention = nn.Linear(16, 1)
        self.budget_net = nn.Sequential(nn.Linear(self.dim, 64), nn.SiLU(), nn.Linear(64, 6))
        self.content_net = nn.Sequential(nn.Linear(self.dim, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        self.fusion_net = nn.Sequential(nn.Linear(self.dim, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        # rosen defaults are too small for these token paths; align with scaled dense inputs.
        for module in self.omni_embedding.emb_modules.values():
            module.reinitialize(init_std=.1)
        self.stage = 0

    def encode(self, b):
        emb = self.omni_embedding.emb_modules
        numeric = torch.stack([b[c].float() for c in self.numeric], -1)
        cats = torch.cat([emb[c](b[c]) for c in CATS], -1)
        def pool(prefix, scorer):
            item, tag = b[prefix + '_items'], b[prefix + '_tags']
            tokens = torch.cat([emb[prefix + '_items'](item), emb[prefix + '_tags'](tag)], -1)
            mask = item != 0
            weights = scorer(tokens).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1) * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
            return (tokens * weights[:, :, None]).sum(1)
        slate = pool('slate', self.slate_attention)
        hist = pool('history', self.history_attention)
        durations = b['slate_durations']
        if self.mode == 'content':
            numeric = numeric * numeric.new_tensor([0 if c in B1 + B2 else 1 for c in self.numeric])
            cats = cats * 0
        if self.mode == 'budget':
            numeric = numeric * numeric.new_tensor([1 if c in B1 + B2 else 0 for c in self.numeric])
            slate, hist, durations = slate * 0, hist * 0, durations * 0
        return torch.cat([numeric, cats, slate, hist, durations], -1)

    def components(self, b):
        x = self.encode(b)
        gamma, survival, log_survival = mixture_survival(self.budget_net(x), b['elapsed_minutes'], b['delta_minutes'])
        stay = torch.sigmoid(self.content_net(x).squeeze(-1))
        return gamma * stay, gamma, stay, survival, log_survival

    def predict(self, b):
        x = self.encode(b)
        if self.mode == 'fusion':
            return torch.sigmoid(self.fusion_net(x).squeeze(-1))
        if self.mode == 'content':
            return torch.sigmoid(self.content_net(x).squeeze(-1))
        gamma, _, _ = mixture_survival(self.budget_net(x), b['elapsed_minutes'], b['delta_minutes'])
        if self.mode == 'budget' or self.stage == 1:
            return gamma
        return gamma * torch.sigmoid(self.content_net(x).squeeze(-1))

    def calculate_loss(self, b):
        return F.binary_cross_entropy(self.predict(b).clamp(1e-6, 1 - 1e-6), b['y_continue'].float())
