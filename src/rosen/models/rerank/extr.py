"""EXTR: organic self-attention and causal ad-to-organic target attention."""
import math

import torch
from torch import nn

from .slate import SlateRerankModel
from .components.attention import masked_softmax
from .components.generators import mlp


class EXTRAttentionBlock(nn.Module):
    """Reference Q/K/V -> residual LayerNorm -> residual FFN (no output projection)."""
    def __init__(self, width, heads=4, attention_dropout=.1, ffn_dropout=.2):
        super().__init__()
        if heads <= 0 or width % heads:
            raise ValueError("EXTR item embedding width must be divisible by num_heads.")
        self.heads, self.head_width = heads, width // heads
        self.q, self.k, self.v = (nn.Linear(width, width) for _ in range(3))
        self.norm = nn.LayerNorm(width, eps=1e-8)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.ffn = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(ffn_dropout),
                                 nn.Linear(width, width), nn.Dropout(ffn_dropout))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, query, context, query_valid, context_valid, *, causal=False):
        b, t, width = query.shape
        s = context.shape[1]
        q = self.q(query).reshape(b, t, self.heads, self.head_width).transpose(1, 2)
        k = self.k(context).reshape(b, s, self.heads, self.head_width).transpose(1, 2)
        v = self.v(context).reshape(b, s, self.heads, self.head_width).transpose(1, 2)
        valid = context_valid[:, None, None, :]
        if causal:
            allowed = torch.arange(s, device=query.device)[None, :] <= torch.arange(t, device=query.device)[:, None]
            valid = valid & allowed[None, None]
        weights = masked_softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_width), valid)
        delta = (self.attention_dropout(weights) @ v).transpose(1, 2).reshape(b, t, width)
        x = self.norm(query + delta)
        return (x + self.ffn(x)) * query_valid[..., None]


class EXTRRerankModel(SlateRerankModel):
    requires_full_slate = True

    def __init__(self, manager, *, natural_indices=None, ad_indices=None, num_heads=4,
                 attention_dropout=.1, ffn_dropout=.2, mlp_a=(60, 32, 20), mlp_b=(128, 32), **kwargs):
        super().__init__(manager, **kwargs)
        natural = list(range(0, self.poi_num, 2)) if natural_indices is None else list(natural_indices)
        ads = list(range(1, self.poi_num, 2)) if ad_indices is None else list(ad_indices)
        if not natural or not ads or sorted(natural + ads) != list(range(self.poi_num)):
            raise ValueError("Non-empty natural_indices/ad_indices must partition every slate slot exactly once.")
        self.register_buffer("natural_indices", torch.tensor(natural, dtype=torch.long))
        self.register_buffer("ad_indices", torch.tensor(ads, dtype=torch.long))
        self.organic_attention = EXTRAttentionBlock(self.item_dim, num_heads, attention_dropout, ffn_dropout)
        self.ad_attention = EXTRAttentionBlock(self.item_dim, num_heads, attention_dropout, ffn_dropout)
        self.encoder, width = mlp(self.item_dim, mlp_a, nn.SiLU)
        self.head, width = mlp(len(ads) * self.item_dim + self.poi_num * (width + 1), mlp_b, nn.SiLU)
        self.output = nn.Linear(width, self.poi_num)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_items(self, cate, dense, mask):
        return cate.flatten(-2)

    def _score_encoded(self, items, dense, mask, context):
        b, m, n, width = items.shape
        valid = mask.flatten(0, 1) > 0
        dense = dense.flatten(0, 1)
        x = items.reshape(b * m, n, width) * valid[..., None]
        organic, ads = x[:, self.natural_indices], x[:, self.ad_indices]
        organic_valid, ad_valid = valid[:, self.natural_indices], valid[:, self.ad_indices]
        organic = self.organic_attention(organic, organic, organic_valid, organic_valid)
        target = self.ad_attention(ads, organic, ad_valid, organic_valid, causal=True)
        encoded = self.encoder(x) * valid[..., None]
        features = torch.cat((target.flatten(1), (dense * valid[..., None]).flatten(1), encoded.flatten(1)), -1)
        return self.output(self.head(features)).reshape(b, m, n)
