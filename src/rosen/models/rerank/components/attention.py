"""Vectorized attention with the reference MGE residual structure and scale."""

import torch
from torch import nn


def masked_softmax(logits, valid):
    """Zero weights for empty rows as well as masked keys."""
    weights = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(-1) * valid
    return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)


class _Attention(nn.Module):
    def __init__(self, num_tokens, emb_dim, scale=8.0):
        super().__init__()
        if num_tokens <= 0 or emb_dim <= 0 or scale <= 0:
            raise ValueError("Attention dimensions and scale must be positive.")
        self.scale = scale
        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.pool = nn.Linear(num_tokens * emb_dim, emb_dim, bias=False)
        # All parameters are independent 2D linear weights: no hidden fan-in axis.
        for layer in (self.q, self.k, self.v, self.pool):
            nn.init.xavier_uniform_(layer.weight)

    def attend(self, x):
        weights = (self.q(x) @ self.k(x).transpose(-1, -2) / self.scale).softmax(-1)
        return weights @ self.v(x)


class IntraAttrSelfAttention(_Attention):
    def __init__(self, num_pois, num_attrs, emb_dim, scale=8.0):
        super().__init__(num_pois, emb_dim, scale)
        self.num_pois, self.num_attrs, self.emb_dim = num_pois, num_attrs, emb_dim

    def forward(self, x):
        if x.shape[-3:] != (self.num_pois, self.num_attrs, self.emb_dim):
            raise ValueError("Intra-attribute attention expects [..., K, F, D] matching its configuration.")
        per_attr = x.transpose(-3, -2)  # [..., F, K, D]
        delta = self.attend(per_attr)
        pooled = self.pool((per_attr + delta).flatten(-2))
        return pooled, delta.transpose(-3, -2)


class InterAttrSelfAttention(_Attention):
    def __init__(self, num_attrs, emb_dim, scale=8.0):
        super().__init__(num_attrs, emb_dim, scale)

    def forward(self, x):
        return self.pool((x + self.attend(x)).flatten(-2))


class OAMBlock(nn.Module):
    def __init__(self, num_pois, num_attrs, emb_dim, scale=8.0):
        super().__init__()
        self.intra = IntraAttrSelfAttention(num_pois, num_attrs, emb_dim, scale)
        self.inter = InterAttrSelfAttention(num_attrs, emb_dim, scale)

    def forward(self, x):
        pooled, per_poi = self.intra(x)
        return self.inter(pooled), per_poi


class IntraPOISelfAttention(InterAttrSelfAttention):
    """Uses the POI axis as a batch axis, attending only across its attributes."""


class PageTargetAttention(nn.Module):
    def forward(self, target, behavior, page_mask):
        if behavior.ndim != 3 or page_mask.shape != behavior.shape[:2]:
            raise ValueError("Expected history [B,P,D] and page_mask [B,P].")
        if target.shape[-1] != behavior.shape[-1]:
            raise ValueError("Target and history embedding widths must agree.")
        if target.ndim == 3:  # [B,M,D], preserve sample-major layout by broadcasting
            if target.shape[0] != behavior.shape[0]:
                raise ValueError("Structured targets and history must have the same batch size.")
            behavior = behavior[:, None]
            mask = page_mask[:, None]
        elif target.ndim == 2:
            if target.shape[0] % behavior.shape[0]:
                raise ValueError("Target batch must be a multiple of the history batch.")
            count = target.shape[0] // behavior.shape[0]
            behavior = behavior.repeat_interleave(count, dim=0)
            mask = page_mask.repeat_interleave(count, dim=0)
        else:
            raise ValueError("Expected targets [B,D], [B*M,D] or [B,M,D].")
        valid = mask == 1
        logits = (target.unsqueeze(-2) * behavior).sum(-1)
        weights = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(-1)
        weights = weights * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        return (weights.unsqueeze(-1) * behavior).sum(-2)
