"""KuaiShou reranker: target-conditioned residual pooling of current and prior items."""
from dataclasses import dataclass

import torch
from torch import nn

from .slate import SlateRerankModel
from .components.attention import masked_softmax
from .components.generators import mlp


@dataclass(frozen=True)
class TargetContextEncoding:
    items: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor


class TargetContextPool(nn.Module):
    def __init__(self, slots, width, scale=8.):
        super().__init__()
        if scale <= 0:
            raise ValueError("attention_scale must be positive.")
        self.scale = scale
        self.q, self.k, self.v = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.pool = nn.Linear(slots * width, width, bias=False)
        for layer in (self.q, self.k, self.v, self.pool):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, target, context, valid):
        return self.score_encoded(target, self.encode_context(context, valid))

    def encode_context(self, context, valid):
        return TargetContextEncoding(context, self.k(context), self.v(context), valid)

    def score_encoded(self, target, context: TargetContextEncoding):
        items, keys, values, valid = context.items, context.keys, context.values, context.valid
        while keys.ndim < target.ndim:
            items, keys, values, valid = (value.unsqueeze(1) for value in (items, keys, values, valid))
        weights = masked_softmax(self.q(target) @ keys.transpose(-1, -2) / self.scale, valid.unsqueeze(-2))
        delta = weights @ values
        residual = (items.unsqueeze(-3) + delta.unsqueeze(-2)) * valid.unsqueeze(-2).unsqueeze(-1)
        return self.pool(residual.flatten(-2))


class KuaiShouRerankModel(SlateRerankModel):
    def __init__(self, manager, *, hist_dense_field=None, mlp_a=(60, 32, 10),
                 mlp_b=(50, 20), attention_scale=8., **kwargs):
        super().__init__(manager, **kwargs)
        self.require_history()
        self.hist_dense_field = hist_dense_field
        self.encoder, width = mlp(self.item_dim, mlp_a, nn.SiLU)
        width += 1
        self.order_context = TargetContextPool(self.poi_num, width, attention_scale)
        self.history_context = TargetContextPool(self.page_num * self.poi_num, width, attention_scale)
        self.head, width = mlp(3 * width, mlp_b, nn.SiLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_items(self, cate, dense, mask):
        return self.encoder(cate.flatten(-2))

    def _encode_context(self, interaction, cate, dense):
        history, pages = self._history_inputs(interaction, cate)
        history = history.flatten(1, 2)
        if self.hist_dense_field is None:
            hist_prior = dense.new_zeros(history.shape[:2] + (1,))
        else:
            hist_prior = interaction[self.hist_dense_field]
            if hist_prior.ndim == 2:
                hist_prior = hist_prior.unsqueeze(-1)
            if hist_prior.shape != history.shape[:2] + (1,):
                raise ValueError("Historical prior scores must be [B,P*K] or [B,P*K,1].")
            hist_prior = hist_prior.to(dense.dtype)
        previous = torch.cat((self.encoder(history.flatten(-2)), hist_prior), -1)
        hist_valid = (pages == 1).repeat_interleave(self.poi_num, -1)
        return self.history_context.encode_context(previous, hist_valid)

    def _score_encoded(self, items, dense, mask, context: TargetContextEncoding):
        current = torch.cat((items, dense), -1)
        valid = mask > 0
        order_context = self.order_context(current, current, valid)
        hist_context = self.history_context.score_encoded(current, context)
        features = torch.cat((current, order_context, hist_context), -1)
        return self.output(self.head(features)).squeeze(-1)
