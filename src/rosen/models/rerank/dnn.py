"""DNN baseline: independent per-item SiLU towers with a scalar prior."""
import torch
from torch import nn

from .slate import SlateRerankModel
from .components.generators import mlp


class DNNRerankModel(SlateRerankModel):
    def __init__(self, manager, *, mlp_a=(60, 32, 20), mlp_b=(128, 32), **kwargs):
        super().__init__(manager, **kwargs)
        self.encoder, width = mlp(self.item_dim, mlp_a, nn.SiLU)
        self.head, width = mlp(width + 1, mlp_b, nn.SiLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_items(self, cate, dense, mask):
        return self.encoder(cate.flatten(-2))

    def _score_encoded(self, items, dense, mask, context):
        return self.output(self.head(torch.cat((items, dense), -1))).squeeze(-1)
