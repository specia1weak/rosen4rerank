"""Argon 6: higher content prior, and Ours content excludes direct session clocks."""
import torch
from torch import nn
from workspace.kr_time_budget.pipeline.time_budget_v4 import ExitModelV4

PRIOR_STAY_LOGIT = 3.5
CONTENT_CLOCK_FIELDS = frozenset({'elapsed_min', 'req_idx'})


class ExitModelV5(ExitModelV4):
    def __init__(self, manager, mode='ours'):
        super().__init__(manager, mode)
        self.prior_stay.fill_(torch.sigmoid(torch.tensor(PRIOR_STAY_LOGIT)).item())

    def content_x(self, b):
        if self.mode != 'ours':
            return super().content_x(b)
        fields = self.omni_embedding(b, split_by='name')
        return torch.cat([torch.zeros_like(value) if name in CONTENT_CLOCK_FIELDS else value
                          for name, value in fields.items()], dim=-1)

    def initialize_stage_two(self, snapshot):
        with torch.random.fork_rng():
            model = ExitModelV5(self.manager, self.mode)
        model.load_state_dict(snapshot.model_state)
        nn.init.zeros_(model.content_net[-1].weight)
        nn.init.constant_(model.content_net[-1].bias, PRIOR_STAY_LOGIT)
        model.set_stage(2)
        return model
