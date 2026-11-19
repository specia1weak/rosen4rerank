"""PRM baseline: item encoding, one inter-item attention, then per-item scoring."""
import torch
from torch import nn

from .slate import SlateRerankModel
from .components.generators import mlp
from .components.attention import masked_softmax


class PRMRerankModel(SlateRerankModel):
    def __init__(self, manager, *, mlp_a=(60, 32, 10), mlp_b=(50, 20), attention_scale=8., **kwargs):
        super().__init__(manager, **kwargs)
        if attention_scale <= 0:
            raise ValueError("attention_scale must be positive.")
        self.attention_scale = attention_scale
        self.encoder, width = mlp(self.item_dim, mlp_a, nn.SiLU)
        width += 1
        self.q, self.k, self.v = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.head, width = mlp(width, mlp_b, nn.SiLU)
        self.output = nn.Linear(width, 1)
        for layer in (self.q, self.k, self.v, self.output):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _encode_items(self, cate, dense, mask):
        return self.encoder(cate.flatten(-2))

    def _score_encoded(self, items, dense, mask, context):
        x = torch.cat((items, dense), -1)
        valid = mask > 0
        weights = masked_softmax(self.q(x) @ self.k(x).transpose(-1, -2) / self.attention_scale, valid.unsqueeze(-2))
        context = (weights @ self.v(x)) * valid[..., None]
        return self.output(self.head(context)).squeeze(-1)
