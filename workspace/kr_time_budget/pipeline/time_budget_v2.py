"""V2 exit screening: stable context budget, tag-only content, fixed budget stage2."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from rosen.models.msr.base import MSRModel
from workspace.kr_time_budget.pipeline.features import B1, B2, NUMS, CATS
from workspace.kr_time_budget.pipeline.features_v2 import STATIC_BUDGET_NUMS
from workspace.kr_time_budget.pipeline.time_budget import mixture_survival


class TagEncoder(nn.Module):
    def __init__(self, manager, emb_modules):
        super().__init__()
        self.emb_modules = emb_modules
        self.slate_attention = nn.Linear(8, 1)
        self.history_attention = nn.Linear(8, 1)
        self.dim = len(NUMS) + len(CATS)*4 + 16 + 6
        for table in self.emb_modules.values():
            table.reinitialize(init_std=.1)

    def forward(self, b, budget=False):
        num = torch.stack([b[c].float() for c in NUMS], -1)
        cats = torch.cat([self.emb_modules[c](b[c]) for c in CATS], -1)
        if budget:
            # Only session-start background is active; no B3, no clock, no slate.
            num = num * num.new_tensor([int(c in STATIC_BUDGET_NUMS) for c in NUMS])
            return torch.cat([num, cats, num.new_zeros(len(num), 22)], -1)
        def pool(name, attention):
            ids = b[name]
            tokens = self.emb_modules[name](ids)
            mask = ids != 0
            w = attention(tokens).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1) * mask
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-8)
            return (tokens * w[:, :, None]).sum(1)
        return torch.cat([num, cats, pool('slate_tags', self.slate_attention), pool('history_tags', self.history_attention), b['slate_durations']], -1)


class ExitModelV2(MSRModel):
    def __init__(self, manager, mode='ours'):
        super().__init__(manager, num_domains=1)
        self.mode = mode
        # Explicit ownership: separate copy is not touched by content updates.
        self.content_encoder = TagEncoder(manager, self.omni_embedding.emb_modules)
        self.budget_encoder = deepcopy(self.content_encoder)
        dim = self.content_encoder.dim
        self.budget_net = nn.Sequential(nn.Linear(dim,64), nn.SiLU(), nn.Linear(64,6))
        self.content_net = nn.Sequential(nn.Linear(dim,64), nn.SiLU(), nn.Linear(64,32), nn.SiLU(), nn.Linear(32,1))
        self.fusion_net = nn.Sequential(nn.Linear(dim,64), nn.SiLU(), nn.Linear(64,32), nn.SiLU(), nn.Linear(32,1))
        nn.init.constant_(self.content_net[-1].bias, 2.6)
        self.stage = 0

    def set_stage(self, stage):
        self.stage = stage
        if self.mode == 'ours' and stage == 2:
            for module in [self.budget_encoder,self.budget_net]:
                for p in module.parameters():
                    p.requires_grad_(False)
                    p.grad = None

    def budget_raw(self, b):
        return self.budget_net(self.budget_encoder(b, budget=True))

    def budget_components(self, b):
        return mixture_survival(self.budget_raw(b), b['elapsed_minutes'], b['delta_minutes'])

    def content_x(self,b):
        x = self.content_encoder(b)
        if self.mode == 'content':
            mask = x.new_ones(self.content_encoder.dim)
            mask[:len(NUMS)] = x.new_tensor([0 if c in B1+B2 else 1 for c in NUMS])
            mask[len(NUMS):len(NUMS)+4*len(CATS)] = 0
            x = x * mask
        return x

    def components(self, b):
        gamma, survival, log_s = self.budget_components(b)
        if self.mode == 'ours' and self.stage == 2:
            gamma = gamma.detach()
        stay = torch.sigmoid(self.content_net(self.content_x(b)).squeeze(-1))
        return gamma * stay, gamma, stay, survival, log_s

    def predict(self, b):
        if self.mode == 'fusion':
            return torch.sigmoid(self.fusion_net(self.content_x(b)).squeeze(-1))
        if self.mode == 'content':
            return torch.sigmoid(self.content_net(self.content_x(b)).squeeze(-1))
        if self.mode == 'budget' or self.stage == 1:
            return self.budget_components(b)[0]
        return self.components(b)[0]

    def calculate_loss(self,b):
        return F.binary_cross_entropy(self.predict(b).clamp(1e-6,1-1e-6),b['y_continue'].float())
