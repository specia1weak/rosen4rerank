"""Argon 7: paired content-routing probes with the same frozen v5 budget."""
import torch
from torch import nn
from workspace.kr_time_budget.pipeline.time_budget_v5 import ExitModelV5, CONTENT_CLOCK_FIELDS, PRIOR_STAY_LOGIT

HISTORY_PROXY_FIELDS = frozenset({
    'is_first_req', 'sess_cum_click_rate', 'sess_cum_longview_rate',
    'sess_cum_pos_rate', 'sess_cum_hate_cnt', 'hist_tags',
})
CONTENT_MASKS = {
    'control': CONTENT_CLOCK_FIELDS,
    'restore_clocks': frozenset(),
    'mask_history': CONTENT_CLOCK_FIELDS | HISTORY_PROXY_FIELDS,
}


class ExitModelV6Probe(ExitModelV5):
    def __init__(self, manager, variant='control'):
        super().__init__(manager, mode='ours')
        self.variant = variant
        self.content_mask = CONTENT_MASKS[variant]

    def content_x(self, b):
        fields = self.omni_embedding(b, split_by='name')
        return torch.cat([torch.zeros_like(value) if name in self.content_mask else value
                          for name, value in fields.items()], dim=-1)

    def initialize_stage_two(self, snapshot):
        with torch.random.fork_rng():
            model = ExitModelV6Probe(self.manager, self.variant)
        model.load_state_dict(snapshot.model_state)
        nn.init.zeros_(model.content_net[-1].weight)
        nn.init.constant_(model.content_net[-1].bias, PRIOR_STAY_LOGIT)
        model.set_stage(2)
        return model
