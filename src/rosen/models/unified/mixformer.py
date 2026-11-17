from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rosen.core.enum_type import FeatureSource
from rosen.emb import SchemaManager
from rosen.emb.emblayer import EmbView
from rosen.emb.schema import SequenceSetting
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.utils.common import to_dims
from rosen.models.utils.general import MLP
from rosen.models.unified.rankmixer import MultiHeadTokenMixing, _make_norm
from rosen.models.unified.tokenmixer import (
    PerTokenSwiGLU,
    reset_per_token_linear_weight_,
)
from rosen.utils.time_bucket import RelativeTimeEmbedding, TIME_BUCKET_BOUNDARIES


def _resolve_timestamp_to_seconds_divisor(unit: str, time_field: Optional[str]) -> int:
    unit = str(unit or "s").lower()
    if unit == "auto":
        unit = "ms" if str(time_field or "").endswith("_ms") else "s"
    divisor_by_unit = {
        "s": 1,
        "sec": 1,
        "second": 1,
        "seconds": 1,
        "ms": 1000,
        "millisecond": 1000,
        "milliseconds": 1000,
    }
    if unit not in divisor_by_unit:
        raise ValueError("relative_time_unit must be one of: auto, s, sec, second, seconds, ms, millisecond, milliseconds.")
    return divisor_by_unit[unit]


def _to_seconds(ts: torch.Tensor, divisor: int) -> torch.Tensor:
    divisor = int(divisor)
    if divisor == 1:
        return ts
    if torch.is_floating_point(ts):
        return torch.floor(ts / float(divisor))
    return torch.div(ts, divisor, rounding_mode="floor")


def _infer_group_max_len(history_view) -> int:
    max_lens = {
        int(getattr(setting, "max_len"))
        for setting in history_view.selected_settings
        if getattr(setting, "max_len", None) is not None
    }
    if not max_lens:
        raise ValueError("Cannot infer max_len from sequence group settings.")
    if len(max_lens) != 1:
        raise ValueError(f"Sequence group has inconsistent max_len values: {sorted(max_lens)}")
    return max_lens.pop()


def _masked_history_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask4 = mask.to(device=logits.device, dtype=torch.bool).unsqueeze(1).unsqueeze(2)
    weights = torch.softmax(logits.masked_fill(~mask4, -1.0e4), dim=-1)
    weights = weights * mask4.to(dtype=weights.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    weights = weights / denom
    has_any = mask4.any(dim=-1, keepdim=True)
    return torch.where(has_any, weights, torch.zeros_like(weights))


def _double_softmax_attention(
    *,
    logits: torch.Tensor,
    values: torch.Tensor,
    same_mask: torch.Tensor,
    cross_mask: torch.Tensor,
    beta: float,
    mode: str = "blend",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    same_weights = _masked_history_softmax(logits, same_mask)
    cross_weights = _masked_history_softmax(logits, cross_mask)
    beta_value = float(beta)
    mode = str(mode or "blend").lower()
    if mode in {"sum", "same_plus_cross", "same_plus_beta_cross"}:
        weights = same_weights + beta_value * cross_weights
    else:
        same_coef = same_mask.any(dim=1).to(dtype=logits.dtype).view(-1, 1, 1, 1) * (1.0 - beta_value)
        cross_coef = cross_mask.any(dim=1).to(dtype=logits.dtype).view(-1, 1, 1, 1) * beta_value
        denom = (same_coef + cross_coef).clamp_min(1.0e-12)
        weights = (same_coef * same_weights + cross_coef * cross_weights) / denom
    output = torch.matmul(weights, values).squeeze(2)
    return output, weights, same_weights, cross_weights


class RelativeBucketedTimeAndPositionBasedBias(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int = 128,
        time_log_denom: float = 0.301,
        init_std: float = 1e-4,
    ):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.num_buckets = int(num_buckets)
        self.time_log_denom = float(time_log_denom)
        self.init_std = float(init_std)
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.num_buckets <= 0:
            raise ValueError("num_buckets must be positive.")
        if self.time_log_denom <= 0:
            raise ValueError("time_log_denom must be positive.")
        self.ts_w = nn.Parameter(torch.empty(self.num_buckets + 1))
        self.pos_w = nn.Parameter(torch.empty(2 * self.max_seq_len - 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ts_w, mean=0.0, std=self.init_std)
        nn.init.normal_(self.pos_w, mean=0.0, std=self.init_std)

    def _bucketize(self, time_difference: torch.Tensor) -> torch.Tensor:
        buckets = (torch.log(time_difference.abs().clamp(min=1)) / self.time_log_denom).long()
        return torch.clamp(buckets, min=0, max=self.num_buckets)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        if timestamps.ndim != 2:
            raise ValueError(f"Expected timestamps to have shape [B, L], got {tuple(timestamps.shape)}")
        batch_size, seq_len = timestamps.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.")
        device = timestamps.device
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.view(1, seq_len) - positions.view(seq_len, 1)
        rel_pos_index = (rel_pos + self.max_seq_len - 1).clamp(min=0, max=2 * self.max_seq_len - 2)
        rel_pos_bias = self.pos_w.to(device=device, dtype=timestamps.dtype)[rel_pos_index]

        time_difference = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)
        time_bucket = self._bucketize(time_difference)
        rel_ts_bias = self.ts_w.to(device=device, dtype=timestamps.dtype)[time_bucket]
        return rel_ts_bias + rel_pos_bias.unsqueeze(0).expand(batch_size, -1, -1)


class AbsoluteTimeEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        day_bucket_count: int = 512,
        init_std: float = 1e-4,
        include_hour: bool = True,
        include_weekday: bool = True,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.day_bucket_count = int(day_bucket_count)
        self.init_std = float(init_std)
        self.include_hour = bool(include_hour)
        self.include_weekday = bool(include_weekday)
        if self.day_bucket_count <= 0:
            raise ValueError("day_bucket_count must be positive.")
        self.day_embedding = nn.Embedding(self.day_bucket_count, self.embedding_dim)
        self.hour_embedding = nn.Embedding(24, self.embedding_dim) if self.include_hour else None
        self.weekday_embedding = nn.Embedding(7, self.embedding_dim) if self.include_weekday else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.day_embedding.weight, mean=0.0, std=self.init_std)
        if self.hour_embedding is not None:
            nn.init.normal_(self.hour_embedding.weight, mean=0.0, std=self.init_std)
        if self.weekday_embedding is not None:
            nn.init.normal_(self.weekday_embedding.weight, mean=0.0, std=self.init_std)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        if timestamps.ndim != 2:
            raise ValueError(f"Expected timestamps to have shape [B, L], got {tuple(timestamps.shape)}")
        ts = timestamps.long()
        valid = ts > 0
        day_index = torch.div(ts.clamp_min(0), 86400, rounding_mode="floor")
        emb = self.day_embedding(torch.remainder(day_index, self.day_bucket_count))
        if self.hour_embedding is not None:
            hour_index = torch.remainder(torch.div(ts.clamp_min(0), 3600, rounding_mode="floor"), 24)
            emb = emb + self.hour_embedding(hour_index)
        if self.weekday_embedding is not None:
            weekday_index = torch.remainder(day_index + 4, 7)
            emb = emb + self.weekday_embedding(weekday_index)
        return emb * valid.unsqueeze(-1).to(dtype=emb.dtype)


class HSTURelativeBiasSequenceAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_seq_len: int,
        num_heads: int = 1,
        num_buckets: int = 128,
        time_log_denom: float = 0.301,
        bias_init_std: float = 1e-4,
        output_init_std: float = 1e-4,
        dropout_rate: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.residual_scale = float(residual_scale)
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        self.head_dim = self.input_dim // self.num_heads
        self.norm = _make_norm(self.input_dim)
        self.q_proj = nn.Linear(self.input_dim, self.input_dim)
        self.k_proj = nn.Linear(self.input_dim, self.input_dim)
        self.v_proj = nn.Linear(self.input_dim, self.input_dim)
        self.out_proj = nn.Linear(self.input_dim, self.input_dim)
        self.rel_bias = RelativeBucketedTimeAndPositionBasedBias(
            max_seq_len=max_seq_len,
            num_buckets=num_buckets,
            time_log_denom=time_log_denom,
            init_std=bias_init_std,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        mask: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        if seq_tokens.ndim != 3:
            raise ValueError(f"Expected seq_tokens to have shape [B, L, D], got {tuple(seq_tokens.shape)}")
        batch_size, seq_len, _ = seq_tokens.shape
        mask = mask.to(device=seq_tokens.device, dtype=torch.bool)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Expected mask to have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}")
        timestamps = timestamps.to(device=seq_tokens.device, dtype=seq_tokens.dtype)
        if timestamps.shape != (batch_size, seq_len):
            raise ValueError(f"Expected timestamps to have shape {(batch_size, seq_len)}, got {tuple(timestamps.shape)}")

        normed = self.norm(seq_tokens)
        q = self._split_heads(self.q_proj(normed))
        k = self._split_heads(self.k_proj(normed))
        v = self._split_heads(self.v_proj(normed))
        logits = torch.matmul(q, k.transpose(-2, -1)) * (float(self.head_dim) ** -0.5)
        logits = logits + self.rel_bias(timestamps).unsqueeze(1)
        logits = logits.masked_fill(~mask.view(batch_size, 1, 1, seq_len), -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        delta = self.out_proj(context)
        delta = delta * mask.unsqueeze(-1).to(dtype=delta.dtype)
        return seq_tokens + self.residual_scale * delta


class SlidingWindowSequenceSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 2,
        window_size: int = 3,
        dropout_rate: float = 0.0,
        output_init_std: Optional[float] = None,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.residual_scale = float(residual_scale)
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.head_dim = self.input_dim // self.num_heads
        self.norm = _make_norm(self.input_dim)
        self.q_proj = nn.Linear(self.input_dim, self.input_dim)
        self.k_proj = nn.Linear(self.input_dim, self.input_dim)
        self.v_proj = nn.Linear(self.input_dim, self.input_dim)
        self.out_proj = nn.Linear(self.input_dim, self.input_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        if output_init_std is not None:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=float(output_init_std))
            nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, seq_tokens: torch.Tensor, mask: torch.Tensor):
        if seq_tokens.ndim != 3:
            raise ValueError(f"Expected seq_tokens to have shape [B, L, D], got {tuple(seq_tokens.shape)}")
        batch_size, seq_len, _ = seq_tokens.shape
        mask = mask.to(device=seq_tokens.device, dtype=torch.bool)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Expected mask to have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}")

        normed = self.norm(seq_tokens)
        q = self._split_heads(self.q_proj(normed))
        k = self._split_heads(self.k_proj(normed))
        v = self._split_heads(self.v_proj(normed))
        logits = torch.matmul(q, k.transpose(-2, -1)) * (float(self.head_dim) ** -0.5)

        positions = torch.arange(seq_len, device=seq_tokens.device)
        radius = int(self.window_size) // 2
        window_mask = (positions.view(1, -1) - positions.view(-1, 1)).abs() <= radius
        valid_mask = mask.view(batch_size, 1, 1, seq_len) & mask.view(batch_size, 1, seq_len, 1)
        attn_mask = valid_mask & window_mask.view(1, 1, seq_len, seq_len)
        logits = logits.masked_fill(~attn_mask, -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        weights = weights * attn_mask.to(dtype=weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        weights = weights / denom

        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        delta = self.out_proj(context)
        delta = delta * mask.unsqueeze(-1).to(dtype=delta.dtype)
        return (seq_tokens + self.residual_scale * delta) * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)


class LinearSequenceSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 2,
        dropout_rate: float = 0.0,
        output_init_std: Optional[float] = 1e-4,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.residual_scale = float(residual_scale)
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        self.head_dim = self.input_dim // self.num_heads
        self.norm = _make_norm(self.input_dim)
        self.q_proj = nn.Linear(self.input_dim, self.input_dim)
        self.k_proj = nn.Linear(self.input_dim, self.input_dim)
        self.v_proj = nn.Linear(self.input_dim, self.input_dim)
        self.out_proj = nn.Linear(self.input_dim, self.input_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        if output_init_std is not None:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=float(output_init_std))
            nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, seq_tokens: torch.Tensor, mask: torch.Tensor):
        if seq_tokens.ndim != 3:
            raise ValueError(f"Expected seq_tokens to have shape [B, L, D], got {tuple(seq_tokens.shape)}")
        batch_size, seq_len, _ = seq_tokens.shape
        mask = mask.to(device=seq_tokens.device, dtype=torch.bool)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Expected mask to have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}")

        normed = self.norm(seq_tokens)
        q = F.elu(self._split_heads(self.q_proj(normed))) + 1.0
        k = F.elu(self._split_heads(self.k_proj(normed))) + 1.0
        v = self._split_heads(self.v_proj(normed))
        mask_h = mask.view(batch_size, 1, seq_len, 1).to(dtype=seq_tokens.dtype)
        k = k * mask_h
        v = v * mask_h

        kv = torch.einsum("bhld,bhle->bhde", k, v)
        k_sum = k.sum(dim=2)
        numerator = torch.einsum("bhld,bhde->bhle", q, kv)
        denominator = torch.einsum("bhld,bhd->bhl", q, k_sum).unsqueeze(-1).clamp_min(1.0e-6)
        context = numerator / denominator
        context = self.dropout(context)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        delta = self.out_proj(context)
        delta = delta * mask.unsqueeze(-1).to(dtype=delta.dtype)
        return (seq_tokens + self.residual_scale * delta) * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)


class SequenceQueryNSTokenAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 2,
        dropout_rate: float = 0.0,
        output_init_std: Optional[float] = 1e-4,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.residual_scale = float(residual_scale)
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        self.head_dim = self.input_dim // self.num_heads
        self.seq_norm = _make_norm(self.input_dim)
        self.ns_norm = _make_norm(self.input_dim)
        self.q_proj = nn.Linear(self.input_dim, self.input_dim)
        self.k_proj = nn.Linear(self.input_dim, self.input_dim)
        self.v_proj = nn.Linear(self.input_dim, self.input_dim)
        self.out_proj = nn.Linear(self.input_dim, self.input_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        if output_init_std is not None:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=float(output_init_std))
            nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        return x.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        mask: torch.Tensor,
    ):
        if seq_tokens.ndim != 3:
            raise ValueError(f"Expected seq_tokens to have shape [B, L, D], got {tuple(seq_tokens.shape)}")
        if ns_tokens.ndim != 3:
            raise ValueError(f"Expected ns_tokens to have shape [B, T, D], got {tuple(ns_tokens.shape)}")
        batch_size, seq_len, _ = seq_tokens.shape
        if ns_tokens.shape[0] != batch_size or ns_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected ns_tokens to have shape [{batch_size}, T, {self.input_dim}], "
                f"got {tuple(ns_tokens.shape)}"
            )
        mask = mask.to(device=seq_tokens.device, dtype=torch.bool)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Expected mask to have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}")

        q = self._split_heads(self.q_proj(self.seq_norm(seq_tokens)))
        k = self._split_heads(self.k_proj(self.ns_norm(ns_tokens)))
        v = self._split_heads(self.v_proj(self.ns_norm(ns_tokens)))
        logits = torch.matmul(q, k.transpose(-2, -1)) * (float(self.head_dim) ** -0.5)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        delta = self.out_proj(context)
        delta = delta * mask.unsqueeze(-1).to(dtype=delta.dtype)
        return (seq_tokens + self.residual_scale * delta) * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)


class Embedding2Tokenization(nn.Module):
    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        num_tokens: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        output_dim = self.num_tokens * self.token_dim
        hidden_dim = max(output_dim * 2, self.token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        tokens = self.mlp(x)
        return tokens.view(x.size(0), self.num_tokens, self.token_dim)


class SwiGLU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expand: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(expand) * self.input_dim
        self.w1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.w2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.w3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.silu(self.w1(x)))
        return self.w3(h * self.w2(x))


class PerTokenSwiGLUInOut(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tokens: int,
        expand: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(expand) * self.input_dim
        self.w1 = nn.Parameter(torch.empty(self.num_tokens, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_tokens, self.input_dim, self.hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_tokens, self.hidden_dim, self.output_dim))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        reset_per_token_linear_weight_(self.w1, self.input_dim)
        reset_per_token_linear_weight_(self.w2, self.input_dim)
        reset_per_token_linear_weight_(self.w3, self.hidden_dim, scale=0.1)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.einsum("btd,tdk->btk", x, self.w1) + self.b1
        h1 = self.dropout(self.act(h1))
        h2 = torch.einsum("btd,tdk->btk", x, self.w2) + self.b2
        return torch.einsum("btk,tkd->btd", h1 * h2, self.w3)


class SlidingWindowTokenMixing(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        window_chunks: int = 2,
        global_last: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        if self.input_dim % self.num_tokens != 0:
            raise ValueError("input_dim must be divisible by num_tokens for SlidingWindowTokenMixing.")
        self.chunk_dim = self.input_dim // self.num_tokens
        self.window_chunks = max(1, min(int(window_chunks), self.num_tokens))
        self.global_last = bool(global_last)
        spans: list[tuple[int, int]] = []
        for token_idx in range(self.num_tokens):
            if self.global_last and token_idx == self.num_tokens - 1:
                spans.append((0, self.input_dim))
                continue
            start = min(token_idx * self.chunk_dim, self.input_dim - self.window_chunks * self.chunk_dim)
            end = start + self.window_chunks * self.chunk_dim
            spans.append((start, end))
        self.spans = tuple(spans)
        self.projs = nn.ModuleList(
            [
                nn.Linear(self.num_tokens * (end - start), self.input_dim)
                for start, end in self.spans
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = []
        for (start, end), proj in zip(self.spans, self.projs):
            window = x[..., start:end].flatten(start_dim=1)
            mixed.append(proj(window))
        return torch.stack(mixed, dim=1)


class MaskedMultiHeadTokenMixing(nn.Module):
    def __init__(self, input_dim: int, num_tokens: int, mode: str):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.mode = str(mode or "").lower()
        if self.input_dim % self.num_tokens != 0:
            raise ValueError("input_dim must be divisible by num_tokens for MaskedMultiHeadTokenMixing.")
        if self.num_tokens < 4:
            raise ValueError("Masked query-mix training slices require at least 4 tokens.")
        source_mask = torch.zeros(self.num_tokens, self.num_tokens, dtype=torch.float32)
        if self.mode == "pairmix_q12":
            source_mask[0:2, 0:2] = 1.0
        elif self.mode == "pairmix_q34":
            source_mask[2:4, 2:4] = 1.0
        elif self.mode == "pairmix_12_34":
            source_mask[0:2, 0:2] = 1.0
            source_mask[2:4, 2:4] = 1.0
        else:
            raise ValueError(f"Unsupported masked token-mixing mode: {mode!r}.")
        self.register_buffer("source_mask", source_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = torch.tensor_split(x, self.num_tokens, dim=-1)
        mixed = torch.stack(heads, dim=1)
        mixed = mixed * self.source_mask.to(device=x.device, dtype=x.dtype).view(1, self.num_tokens, self.num_tokens, 1)
        return mixed.flatten(start_dim=2)


class QueryMixer(nn.Module):
    MODE_CODES = {
        "full": 1,
        "none": 2,
        "headmix_only": 3,
        "headmix_replace_only": 4,
        "headmix_concat_ffn": 5,
        "headmix_concat_resffn": 6,
        "ffn_only": 7,
        "windowmix_only": 8,
        "windowmix_full": 9,
        "pairmix_q12": 21,
        "pairmix_q34": 22,
        "pairmix_12_34": 23,
    }

    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        expand: int = 2,
        dropout_rate: float = 0.0,
        mode: str = "full",
        window_chunks: int = 2,
        window_global_last: bool = True,
    ):
        super().__init__()
        self.mode = str(mode or "full").lower()
        allowed_modes = {
            "full",
            "none",
            "headmix_only",
            "headmix_replace_only",
            "headmix_concat_ffn",
            "headmix_concat_resffn",
            "ffn_only",
            "windowmix_only",
            "windowmix_full",
            "pairmix_q12",
            "pairmix_q34",
            "pairmix_12_34",
        }
        if self.mode not in allowed_modes:
            raise ValueError(f"query_mixer_mode must be one of {sorted(allowed_modes)}, got {mode!r}.")
        self.mode_code = int(self.MODE_CODES.get(self.mode, 0))
        self.use_token_mixing = self.mode in {
            "full",
            "headmix_only",
            "headmix_replace_only",
            "headmix_concat_ffn",
            "headmix_concat_resffn",
            "windowmix_only",
            "windowmix_full",
            "pairmix_q12",
            "pairmix_q34",
            "pairmix_12_34",
        }
        self.use_concat_ffn = self.mode in {"headmix_concat_ffn", "headmix_concat_resffn"}
        self.use_concat_residual = self.mode == "headmix_concat_resffn"
        self.use_mixing_residual = self.mode not in {"headmix_replace_only", "headmix_concat_ffn", "headmix_concat_resffn"}
        self.use_ffn = self.mode in {"full", "ffn_only", "windowmix_full", "pairmix_q12", "pairmix_q34", "pairmix_12_34"}
        self.norm1 = _make_norm(input_dim) if self.use_token_mixing else nn.Identity()
        self.norm2 = _make_norm(input_dim * 2) if self.use_concat_ffn else _make_norm(input_dim) if self.use_ffn else nn.Identity()
        self.token_mixing = (
            MultiHeadTokenMixing(input_dim=input_dim, num_tokens=num_tokens)
            if self.mode in {"full", "headmix_only", "headmix_replace_only", "headmix_concat_ffn", "headmix_concat_resffn"}
            else MaskedMultiHeadTokenMixing(
                input_dim=input_dim,
                num_tokens=num_tokens,
                mode=self.mode,
            )
            if self.mode in {"pairmix_q12", "pairmix_q34", "pairmix_12_34"}
            else SlidingWindowTokenMixing(
                input_dim=input_dim,
                num_tokens=num_tokens,
                window_chunks=window_chunks,
                global_last=window_global_last,
            )
            if self.mode in {"windowmix_only", "windowmix_full"}
            else nn.Identity()
        )
        self.per_head_ffn = (
            PerTokenSwiGLUInOut(
                input_dim=input_dim * 2,
                output_dim=input_dim,
                num_tokens=num_tokens,
                expand=expand,
                dropout_rate=dropout_rate,
            )
            if self.use_concat_ffn
            else
            PerTokenSwiGLU(
                input_dim=input_dim,
                num_tokens=num_tokens,
                expand=expand,
                dropout_rate=dropout_rate,
            )
            if self.use_ffn
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_token_mixing:
            normed = self.norm1(x)
            mixed = self.token_mixing(normed)
            if self.use_concat_ffn:
                concat = torch.cat([x, mixed], dim=-1)
                fused = self.per_head_ffn(self.norm2(concat))
                out = x + fused if self.use_concat_residual else fused
                return out
            x = mixed + x if self.use_mixing_residual else mixed
        if self.use_ffn:
            ffn_update = self.per_head_ffn(self.norm2(x))
            x = ffn_update + x
        return x


class KeyConditionedLowRankModulation(nn.Module):
    def __init__(
        self,
        token_dim: int,
        condition_dim: int,
        rank: int = 32,
        hidden_dim: int = 64,
        scale: float = 0.1,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.condition_dim = int(condition_dim)
        self.rank = max(1, min(int(rank), self.token_dim))
        self.scale = float(scale)
        hidden_dim = int(hidden_dim)
        self.down = nn.Linear(self.token_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.token_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(self.condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.rank),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        last = self.gate[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gate = torch.tanh(self.gate(condition)).unsqueeze(1)
        residual = self.up(self.down(tokens) * gate)
        return tokens + self.scale * residual


class DomainQueryLowRankModulation(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_domains: int,
        rank: int = 32,
        scale: float = 0.05,
        anchor_domain: int = -1,
        preserve_norm: bool = False,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_domains = int(num_domains)
        self.rank = max(1, min(int(rank), self.token_dim))
        self.scale = float(scale)
        self.anchor_domain = int(anchor_domain)
        self.preserve_norm = bool(preserve_norm)
        self.down = nn.Linear(self.token_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.token_dim, bias=False)
        self.domain_gate = nn.Embedding(self.num_domains, self.rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.domain_gate.weight)

    def forward(self, query_tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        domain_ids = domain_ids.long().view(-1).clamp(min=0, max=self.num_domains - 1)
        gate = torch.tanh(self.domain_gate(domain_ids))
        if 0 <= self.anchor_domain < self.num_domains:
            anchor_mask = domain_ids == self.anchor_domain
            gate = torch.where(anchor_mask.unsqueeze(-1), torch.zeros_like(gate), gate)
        residual = self.up(self.down(query_tokens) * gate.unsqueeze(1))
        modulated = query_tokens + self.scale * residual
        if self.preserve_norm:
            source_norm = query_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            modulated_norm = modulated.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            modulated = modulated * (source_norm / modulated_norm)
        return modulated


class ScoreConditionedLowRankRerank(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        condition_dim: int,
        rank: int = 32,
        hidden_dim: int = 64,
        scale: float = 0.1,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.condition_dim = int(condition_dim)
        self.rank = max(1, min(int(rank), self.token_dim))
        self.scale = float(scale)
        hidden_dim = int(hidden_dim)
        self.q_low_proj = nn.Linear(self.token_dim, self.rank, bias=False)
        self.k_low_proj = nn.Linear(self.token_dim, self.num_tokens * self.rank, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(self.condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.rank),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_low_proj.weight)
        nn.init.xavier_uniform_(self.k_low_proj.weight)
        last = self.gate[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = seq_tokens.shape
        q_low = self.q_low_proj(ns_tokens)
        k_low = self.k_low_proj(seq_tokens).view(batch_size, seq_len, self.num_tokens, self.rank)
        k_low = k_low.permute(0, 2, 1, 3)
        gate = torch.tanh(self.gate(condition)).view(batch_size, 1, 1, self.rank)
        score = (q_low.unsqueeze(2) * k_low * gate).sum(dim=-1)
        score = score * (self.scale / (float(self.rank) ** 0.5))
        if mask is not None:
            score = score * mask.unsqueeze(1).to(dtype=score.dtype)
        return score


class CrossAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        expand: int = 2,
        dropout_rate: float = 0.0,
        key_modulation_condition_dim: Optional[int] = None,
        key_modulation_rank: int = 32,
        key_modulation_hidden_dim: int = 64,
        key_modulation_scale: float = 0.1,
        score_rerank_condition_dim: Optional[int] = None,
        score_rerank_rank: int = 32,
        score_rerank_hidden_dim: int = 64,
        score_rerank_scale: float = 0.1,
        query_modulation_num_domains: Optional[int] = None,
        query_modulation_rank: int = 32,
        query_modulation_scale: float = 0.05,
        query_modulation_anchor_domain: int = -1,
        query_modulation_preserve_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.seq_norm = _make_norm(input_dim)
        self.seq_ffn = SwiGLU(input_dim=input_dim, expand=expand, dropout_rate=dropout_rate)
        self.k_proj = nn.Linear(input_dim, input_dim * self.num_tokens)
        self.v_proj = nn.Linear(input_dim, input_dim * self.num_tokens)
        self.query_modulator = (
            DomainQueryLowRankModulation(
                token_dim=input_dim,
                num_domains=int(query_modulation_num_domains),
                rank=query_modulation_rank,
                scale=query_modulation_scale,
                anchor_domain=query_modulation_anchor_domain,
                preserve_norm=query_modulation_preserve_norm,
            )
            if query_modulation_num_domains is not None
            else None
        )
        self.key_modulator = (
            KeyConditionedLowRankModulation(
                token_dim=input_dim,
                condition_dim=int(key_modulation_condition_dim),
                rank=key_modulation_rank,
                hidden_dim=key_modulation_hidden_dim,
                scale=key_modulation_scale,
            )
            if key_modulation_condition_dim is not None
            else None
        )
        self.score_reranker = (
            ScoreConditionedLowRankRerank(
                token_dim=input_dim,
                num_tokens=self.num_tokens,
                condition_dim=int(score_rerank_condition_dim),
                rank=score_rerank_rank,
                hidden_dim=score_rerank_hidden_dim,
                scale=score_rerank_scale,
            )
            if score_rerank_condition_dim is not None
            else None
        )

    def forward(
        self,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        key_modulation_condition: Optional[torch.Tensor] = None,
        score_rerank_condition: Optional[torch.Tensor] = None,
        query_modulation_domain_ids: Optional[torch.Tensor] = None,
        attention_double_softmax_beta: Optional[float] = None,
        attention_double_softmax_mode: str = "blend",
        attention_same_mask: Optional[torch.Tensor] = None,
        attention_cross_mask: Optional[torch.Tensor] = None,
        attention_output_gate: Optional[torch.Tensor] = None,
    ):
        seq_tokens = self.seq_ffn(self.seq_norm(seq_tokens)) + seq_tokens
        if mask is not None:
            seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        key_tokens = seq_tokens
        if self.key_modulator is not None:
            if key_modulation_condition is None:
                raise ValueError("key_modulation_condition is required when key modulation is enabled.")
            key_tokens = self.key_modulator(seq_tokens, key_modulation_condition)
            if mask is not None:
                key_tokens = key_tokens * mask.unsqueeze(-1).to(dtype=key_tokens.dtype)

        keys = self.k_proj(key_tokens).chunk(self.num_tokens, dim=-1)
        values = self.v_proj(seq_tokens).chunk(self.num_tokens, dim=-1)
        k = torch.stack(keys, dim=1)
        v = torch.stack(values, dim=1)
        query_tokens = ns_tokens
        if self.query_modulator is not None:
            if query_modulation_domain_ids is None:
                raise ValueError("query_modulation_domain_ids is required when query modulation is enabled.")
            query_tokens = self.query_modulator(query_tokens, query_modulation_domain_ids)
        q = query_tokens.unsqueeze(2)
        bias = None
        if attention_bias is not None:
            bias = attention_bias.to(device=q.device, dtype=q.dtype)
            if bias.dim() == 2:
                bias = bias.unsqueeze(1).unsqueeze(2)
            elif bias.dim() == 3:
                bias = bias.unsqueeze(2)
        if self.score_reranker is not None:
            if score_rerank_condition is None:
                raise ValueError("score_rerank_condition is required when score rerank is enabled.")
            rerank_bias = self.score_reranker(seq_tokens, ns_tokens, score_rerank_condition, mask)
            rerank_bias = rerank_bias.to(device=q.device, dtype=q.dtype).unsqueeze(2)
            bias = rerank_bias if bias is None else bias + rerank_bias
        use_double_softmax = (
            attention_double_softmax_beta is not None
            and attention_same_mask is not None
            and attention_cross_mask is not None
        )
        if use_double_softmax:
            logits = torch.matmul(q, k.transpose(-2, -1)) * (float(self.input_dim) ** -0.5)
            if bias is not None:
                logits = logits + bias
            attention_output, _, _, _ = _double_softmax_attention(
                logits=logits,
                values=v,
                same_mask=attention_same_mask,
                cross_mask=attention_cross_mask,
                beta=float(attention_double_softmax_beta),
                mode=attention_double_softmax_mode,
            )
        elif bias is None:
            attention_output = F.scaled_dot_product_attention(q, k, v).squeeze(2)
        else:
            attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).squeeze(2)
        if attention_output_gate is not None:
            gate = attention_output_gate.to(device=attention_output.device, dtype=attention_output.dtype)
            if gate.dim() != 2:
                raise ValueError(
                    "attention_output_gate must have shape [batch_size, num_tokens], "
                    f"got {tuple(gate.shape)}."
                )
            attention_output = attention_output * gate.unsqueeze(-1)
        ns_tokens = attention_output + ns_tokens
        return seq_tokens, ns_tokens


class OutputFusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        expand: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.norm = _make_norm(input_dim)
        self.per_head_ffn = PerTokenSwiGLU(
            input_dim=input_dim,
            num_tokens=num_tokens,
            expand=expand,
            dropout_rate=dropout_rate,
        )

    def forward(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        return self.per_head_ffn(self.norm(ns_tokens)) + ns_tokens


class MixFormerBlocks(nn.Module):
    QUERYMIX_EVAL_NORMAL = 0
    QUERYMIX_EVAL_OFF = 1
    QUERYMIX_EVAL_PERMUTE_SLOTS = 2

    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        num_layers: int,
        expand: int = 2,
        dropout_rate: float = 0.0,
        key_modulation_condition_dim: Optional[int] = None,
        key_modulation_rank: int = 32,
        key_modulation_hidden_dim: int = 64,
        key_modulation_scale: float = 0.1,
        key_modulation_layers: str = "all",
        score_rerank_condition_dim: Optional[int] = None,
        score_rerank_rank: int = 32,
        score_rerank_hidden_dim: int = 64,
        score_rerank_scale: float = 0.1,
        score_rerank_layers: str = "all",
        query_modulation_num_domains: Optional[int] = None,
        query_modulation_rank: int = 32,
        query_modulation_scale: float = 0.05,
        query_modulation_anchor_domain: int = -1,
        query_modulation_preserve_norm: bool = False,
        query_modulation_layers: str = "tail",
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.query_mixer_mode = str(query_mixer_mode or "full").lower()
        self.query_mixer_window_chunks = int(query_mixer_window_chunks)
        self.query_mixer_window_global_last = bool(query_mixer_window_global_last)
        self.query_mixers = nn.ModuleList(
            [
                QueryMixer(
                    input_dim=input_dim,
                    num_tokens=num_tokens,
                    expand=expand,
                    dropout_rate=dropout_rate,
                    mode=self.query_mixer_mode,
                    window_chunks=self.query_mixer_window_chunks,
                    window_global_last=self.query_mixer_window_global_last,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.cross_attentions = nn.ModuleList(
            [
                CrossAttention(
                    input_dim=input_dim,
                    num_tokens=num_tokens,
                    expand=expand,
                    dropout_rate=dropout_rate,
                    key_modulation_condition_dim=(
                        key_modulation_condition_dim
                        if self._layer_enabled(layer_idx, key_modulation_layers)
                        else None
                    ),
                    key_modulation_rank=key_modulation_rank,
                    key_modulation_hidden_dim=key_modulation_hidden_dim,
                    key_modulation_scale=key_modulation_scale,
                    score_rerank_condition_dim=(
                        score_rerank_condition_dim
                        if self._layer_enabled(layer_idx, score_rerank_layers)
                        else None
                    ),
                    score_rerank_rank=score_rerank_rank,
                    score_rerank_hidden_dim=score_rerank_hidden_dim,
                    score_rerank_scale=score_rerank_scale,
                    query_modulation_num_domains=(
                        query_modulation_num_domains
                        if self._layer_enabled(layer_idx, query_modulation_layers)
                        else None
                    ),
                    query_modulation_rank=query_modulation_rank,
                    query_modulation_scale=query_modulation_scale,
                    query_modulation_anchor_domain=query_modulation_anchor_domain,
                    query_modulation_preserve_norm=query_modulation_preserve_norm,
                )
                for layer_idx in range(self.num_layers)
            ]
        )
        self.output_fusions = nn.ModuleList(
            [
                OutputFusion(
                    input_dim=input_dim,
                    num_tokens=num_tokens,
                    expand=expand,
                    dropout_rate=dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _layer_enabled(self, layer_idx: int, layers: str) -> bool:
        layers = str(layers or "all").lower()
        if layers == "all":
            return True
        if layers == "last":
            return layer_idx == self.num_layers - 1
        if layers == "tail":
            return layer_idx > 0
        if layers == "none":
            return False
        raise ValueError("layers must be one of: all, last, tail, none.")

    def _run_query_mixer_step(
        self,
        query_mixer: QueryMixer,
        ns_tokens: torch.Tensor,
        *,
        querymix_eval_code: int = 0,
    ) -> torch.Tensor:
        code = int(querymix_eval_code)
        if code == self.QUERYMIX_EVAL_OFF:
            return ns_tokens

        mixed = query_mixer(ns_tokens)
        if code != self.QUERYMIX_EVAL_PERMUTE_SLOTS:
            return mixed

        num_slots = int(mixed.shape[1])
        order = torch.arange(num_slots, device=mixed.device, dtype=torch.long).roll(shifts=-1)
        return mixed.index_select(1, order)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        key_modulation_condition: Optional[torch.Tensor] = None,
        score_rerank_condition: Optional[torch.Tensor] = None,
        query_modulation_domain_ids: Optional[torch.Tensor] = None,
        attention_double_softmax_beta: Optional[float] = None,
        attention_double_softmax_mode: str = "blend",
        attention_same_mask: Optional[torch.Tensor] = None,
        attention_cross_mask: Optional[torch.Tensor] = None,
        attention_output_gate: Optional[torch.Tensor] = None,
        querymix_eval_code: int = 0,
    ):
        for layer_idx, (query_mixer, cross_attention, output_fusion) in enumerate(zip(
            self.query_mixers,
            self.cross_attentions,
            self.output_fusions,
        )):
            ns_tokens = self._run_query_mixer_step(
                query_mixer,
                ns_tokens,
                querymix_eval_code=querymix_eval_code,
            )
            seq_tokens, ns_tokens = cross_attention(
                seq_tokens,
                ns_tokens,
                mask,
                attention_bias,
                key_modulation_condition,
                score_rerank_condition,
                query_modulation_domain_ids,
                attention_double_softmax_beta=attention_double_softmax_beta,
                attention_double_softmax_mode=attention_double_softmax_mode,
                attention_same_mask=attention_same_mask,
                attention_cross_mask=attention_cross_mask,
                attention_output_gate=attention_output_gate,
            )
            ns_tokens = output_fusion(ns_tokens)
        return seq_tokens, ns_tokens


class MixFormerModel(MSRModel):
    ATTENTION_SAME_BIAS_FIELD = "__mixformer_attention_same_bias__"
    ATTENTION_CROSS_BIAS_FIELD = "__mixformer_attention_cross_bias__"
    ATTENTION_OFF_FIELD = "__mixformer_attention_off_code__"
    ATTENTION_SLOT_CATEGORY_OFF_CODE_FIELD = "__mixformer_attention_slot_category_off_code__"
    ATTENTION_SLOT_CATEGORY_OFF_SLOT_FIELD = "__mixformer_attention_slot_category_off_slot__"
    ATTENTION_SLOT_CATEGORY_OFF_LAYER_FIELD = "__mixformer_attention_slot_category_off_layer__"
    ATTENTION_SLOT_OUTPUT_OFF_SLOT_FIELD = "__mixformer_attention_slot_output_off_slot__"
    ATTENTION_SLOT_OUTPUT_OFF_LAYER_FIELD = "__mixformer_attention_slot_output_off_layer__"
    FINAL_SLOT_ABLATION_FIELD = "__mixformer_final_slot_ablation_code__"
    FINAL_SLOT_PERMUTE_FIELD = "__mixformer_final_slot_permute_code__"
    FINAL_SLOT_WEIGHT_FIELD = "__mixformer_final_slot_weight__"
    QUERYMIX_EVAL_FIELD = "__mixformer_querymix_eval_code__"
    ATTENTION_OFF_CODES = {
        "off_prev_last1": 1,
        "off_prev_segment_all": 2,
        "off_same_domain_history": 3,
        "off_cross_before_target_entry": 4,
        "off_all_cross": 5,
        "off_target_entry_token": 6,
        "off_target_tail": 7,
        "off_target_bucket_0_20": 10,
        "off_target_bucket_20_40": 11,
        "off_target_bucket_40_60": 12,
        "off_target_bucket_60_80": 13,
        "off_target_bucket_80_100": 14,
    }
    FINAL_SLOT_ABLATION_CODES = {
        "final_q1_only": 1,
        "final_q2_only": 2,
        "final_q3_only": 3,
        "final_q4_only": 4,
        "final_q12_only": 12,
        "final_q13_only": 13,
        "final_q14_only": 14,
        "final_q23_only": 23,
        "final_q24_only": 24,
        "final_q34_only": 34,
        "final_q123_only": 123,
        "final_q124_only": 124,
        "final_q134_only": 134,
        "final_q234_only": 234,
        "final_q1_off": 101,
        "final_q2_off": 102,
        "final_q3_off": 103,
        "final_q4_off": 104,
    }
    FINAL_SLOT_PERMUTE_CODES = {
        "final_q_permute_slots_eval": 1,
    }
    QUERYMIX_EVAL_CODES = {
        "querymix_off_eval": MixFormerBlocks.QUERYMIX_EVAL_OFF,
        "querymix_permute_slots_eval": MixFormerBlocks.QUERYMIX_EVAL_PERMUTE_SLOTS,
    }

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        attention_same_protect: bool = False,
        cross_context_mode: str = "none",
        cross_context_beta: float = 0.0,
        attention_double_softmax_beta: Optional[float] = None,
        attention_double_softmax_mode: str = "blend",
        max_len: Optional[int] = None,
        use_absolute_position: bool = False,
        absolute_position_init_std: float = 1e-3,
        use_absolute_time: bool = False,
        absolute_time_init_std: float = 1e-4,
        absolute_time_day_buckets: int = 512,
        absolute_time_include_hour: bool = True,
        absolute_time_include_weekday: bool = True,
        use_segment_relative_position: bool = False,
        segment_relative_position_buckets: int = 5,
        segment_relative_position_init_std: float = 1e-3,
        use_boundary_position: bool = False,
        boundary_position_init_std: float = 1e-3,
        use_hstu_relative_attention: bool = False,
        hstu_relative_attention_num_heads: int = 1,
        hstu_relative_time_num_buckets: int = 128,
        hstu_relative_time_log_denom: float = 0.301,
        hstu_relative_bias_init_std: float = 1e-4,
        hstu_relative_output_init_std: float = 1e-4,
        hstu_relative_attention_scale: float = 1.0,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = manager.domain_field
        self.LABEL = manager.label_field
        if seq_group_name not in self.omni_embedding.seq_groups:
            raise ValueError(f"Unknown sequence group: {seq_group_name}")
        self.history_view = self.omni_embedding.seq_groups[seq_group_name]
        self.target_feature_keep = (
            frozenset(str(field_name) for field_name in target_feature_fields)
            if target_feature_fields is not None
            else None
        )
        target_settings_by_field = {
            setting.field_name: setting
            for setting in self.history_view.target_settings
        }
        self.late_target_feature_fields = tuple(str(field_name) for field_name in (late_target_feature_fields or ()))
        unknown_late_target_fields = sorted(set(self.late_target_feature_fields).difference(target_settings_by_field))
        if unknown_late_target_fields:
            raise ValueError(f"Unknown late target fields: {unknown_late_target_fields}")
        self.late_target_embedding_dim = sum(
            int(target_settings_by_field[field_name].embedding_dim)
            for field_name in self.late_target_feature_fields
        )

        target_field_names = {setting.field_name for setting in self.history_view.target_settings}
        label_field_names = set(manager.label_fields)
        excluded_current_fields = set(exclude_current_fields or ())
        current_fields = [
            setting.field_name
            for setting in manager.settings
            if not isinstance(setting, SequenceSetting)
            and setting.field_name not in label_field_names
            and setting.field_name not in excluded_current_fields
            and (
                setting.field_name not in target_field_names
                or setting.field_name == self.DOMAIN
            )
        ]
        self.current_view = EmbView(self.omni_embedding, include_fields=current_fields)
        settings_by_field = {setting.field_name: setting for setting in manager.settings}
        self.current_component_groups = self._make_current_component_groups(current_fields, settings_by_field)
        self.target_component_groups = self._make_target_component_groups(self.history_view.target_settings)
        self.current_embedding_dim = self.current_view.embedding_dim
        self.sequence_embedding_dim = self.history_view.embedding_dim
        self.token_dim = int(token_dim)
        self.num_ns_token = int(num_ns_token)
        self.query_mixer_mode = str(query_mixer_mode or "full").lower()
        self.query_mixer_window_chunks = int(query_mixer_window_chunks)
        self.query_mixer_window_global_last = bool(query_mixer_window_global_last)
        self.max_len = int(max_len) if max_len is not None else _infer_group_max_len(self.history_view)
        self.use_absolute_position = bool(use_absolute_position)
        self.use_absolute_time = bool(use_absolute_time)
        self.use_segment_relative_position = bool(use_segment_relative_position)
        self.use_boundary_position = bool(use_boundary_position)
        self.use_hstu_relative_attention = bool(use_hstu_relative_attention)
        self.segment_relative_position_buckets = max(1, int(segment_relative_position_buckets))

        self.use_relative_time = bool(use_relative_time)
        self.time_field = manager.time_field
        if self.use_relative_time or self.use_absolute_time or self.use_hstu_relative_attention:
            if not self.time_field:
                raise ValueError(
                    "Relative-time, absolute-time, or HSTU relative attention requires manager.time_field to be set."
                )
            if not self.history_view.time_field:
                raise ValueError(
                    f"Relative-time, absolute-time, or HSTU relative attention requires seq group "
                    f"'{seq_group_name}' to define time_field_name."
                )
            self.relative_time_to_seconds_divisor = _resolve_timestamp_to_seconds_divisor(
                relative_time_unit,
                self.time_field,
            )
        else:
            self.relative_time_to_seconds_divisor = 1
        if self.use_relative_time:
            self.relative_time_embedding = RelativeTimeEmbedding(
                embedding_dim=self.sequence_embedding_dim,
                boundaries=tuple(relative_time_boundaries or TIME_BUCKET_BOUNDARIES),
                init_std=float(relative_time_init_std),
            )
        else:
            self.relative_time_embedding = None

        self.seq_token_proj = (
            nn.Linear(self.sequence_embedding_dim, self.token_dim)
            if self.sequence_embedding_dim != self.token_dim
            else nn.Identity()
        )
        self.absolute_position_embedding = None
        if self.use_absolute_position:
            self.absolute_position_embedding = nn.Embedding(self.max_len, self.token_dim)
            nn.init.normal_(
                self.absolute_position_embedding.weight,
                mean=0.0,
                std=float(absolute_position_init_std),
            )
        self.absolute_time_embedding = None
        if self.use_absolute_time:
            self.absolute_time_embedding = AbsoluteTimeEmbedding(
                embedding_dim=self.token_dim,
                day_bucket_count=int(absolute_time_day_buckets),
                init_std=float(absolute_time_init_std),
                include_hour=bool(absolute_time_include_hour),
                include_weekday=bool(absolute_time_include_weekday),
            )
        self.segment_relative_position_embedding = None
        if self.use_segment_relative_position:
            self.segment_relative_position_embedding = nn.Embedding(
                self.segment_relative_position_buckets,
                self.token_dim,
            )
            nn.init.normal_(
                self.segment_relative_position_embedding.weight,
                mean=0.0,
                std=float(segment_relative_position_init_std),
            )
        self.boundary_position_role_embedding = None
        self.boundary_position_type_embedding = None
        if self.use_boundary_position:
            self.boundary_position_role_embedding = nn.Embedding(4, self.token_dim)
            self.boundary_position_type_embedding = nn.Embedding(3, self.token_dim)
            nn.init.normal_(
                self.boundary_position_role_embedding.weight,
                mean=0.0,
                std=float(boundary_position_init_std),
            )
            nn.init.normal_(
                self.boundary_position_type_embedding.weight,
                mean=0.0,
                std=float(boundary_position_init_std),
            )
            with torch.no_grad():
                self.boundary_position_role_embedding.weight[0].zero_()
                self.boundary_position_type_embedding.weight[0].zero_()
        self.hstu_relative_attention = None
        if self.use_hstu_relative_attention:
            self.hstu_relative_attention = HSTURelativeBiasSequenceAttention(
                input_dim=self.token_dim,
                max_seq_len=self.max_len,
                num_heads=int(hstu_relative_attention_num_heads),
                num_buckets=int(hstu_relative_time_num_buckets),
                time_log_denom=float(hstu_relative_time_log_denom),
                bias_init_std=float(hstu_relative_bias_init_std),
                output_init_std=float(hstu_relative_output_init_std),
                dropout_rate=dropout_rate,
                residual_scale=float(hstu_relative_attention_scale),
            )
        self.tokenizer = Embedding2Tokenization(
            input_dim=self.current_embedding_dim + self.sequence_embedding_dim,
            token_dim=self.token_dim,
            num_tokens=self.num_ns_token,
            dropout_rate=dropout_rate,
        )
        self.mixformer = MixFormerBlocks(
            input_dim=self.token_dim,
            num_tokens=self.num_ns_token,
            num_layers=num_layers,
            expand=expansion_factor,
            dropout_rate=dropout_rate,
            query_mixer_mode=self.query_mixer_mode,
            query_mixer_window_chunks=self.query_mixer_window_chunks,
            query_mixer_window_global_last=self.query_mixer_window_global_last,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.token_dim + self.late_target_embedding_dim,
            hidden_dims=to_dims(tower_hidden_dims, default=(128, 64)),
            dropout_rate=tower_dropout_rate,
            activation="relu",
            batch_norm=tower_batch_norm,
        )

        domain_setting = manager.get_setting(self.DOMAIN)
        self.domain_shift = 1 if getattr(domain_setting, "padding_zero", False) else 0
        self.domain_seq_field = str(domain_seq_field)
        self.attention_same_protect = bool(attention_same_protect)
        self.cross_context_mode = str(cross_context_mode or "none").lower()
        self.cross_context_beta = float(cross_context_beta)
        self.attention_double_softmax_beta = (
            None
            if attention_double_softmax_beta is None
            else float(attention_double_softmax_beta)
        )
        self.attention_double_softmax_mode = str(attention_double_softmax_mode or "blend").lower()
        self.cross_context_patch_count = self._parse_patch_linear_count(self.cross_context_mode)
        self.cross_context_patch_target_embedding = None
        self.cross_context_patch_q_proj = None
        self.cross_context_patch_k_proj = None
        self.cross_context_patch_v_proj = None
        self.cross_context_patch_out_proj = None
        if self.cross_context_patch_count > 0:
            self.cross_context_patch_target_embedding = nn.Embedding(int(num_domains), self.token_dim)
            self.cross_context_patch_q_proj = nn.Linear(self.token_dim, self.token_dim)
            self.cross_context_patch_k_proj = nn.Linear(self.token_dim, self.token_dim)
            self.cross_context_patch_v_proj = nn.Linear(self.token_dim, self.token_dim)
            self.cross_context_patch_out_proj = nn.Linear(self.token_dim, self.token_dim)

    def _append_component_group(
        self,
        groups: list[list[object]],
        group_name: str,
        field_name: str,
    ) -> None:
        if groups and groups[-1][0] == group_name:
            groups[-1][1].append(field_name)
            return
        groups.append([group_name, [field_name]])

    def _make_current_component_groups(
        self,
        current_fields: Iterable[str],
        settings_by_field: dict[str, object],
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        groups: list[list[object]] = []
        for field_name in current_fields:
            setting = settings_by_field.get(field_name)
            source = getattr(setting, "source", None)
            if field_name == self.DOMAIN:
                group_name = "domain"
            elif source == FeatureSource.USER_ID:
                group_name = "user_id"
            elif source == FeatureSource.USER:
                group_name = "user_side"
            elif source == FeatureSource.INTERACTION:
                group_name = "interaction"
            else:
                group_name = "other"
            self._append_component_group(groups, group_name, field_name)
        return tuple((str(name), tuple(fields)) for name, fields in groups)

    def _make_target_component_groups(
        self,
        target_settings: Iterable[object],
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        groups: list[list[object]] = []
        high_card_item_fields = {"author_id", "music_id"}
        for setting in target_settings:
            field_name = setting.field_name
            source = getattr(setting, "source", None)
            if field_name == self.DOMAIN:
                group_name = "domain"
            elif source == FeatureSource.ITEM_ID:
                group_name = "item_id"
            elif field_name in high_card_item_fields:
                group_name = "item_high_card"
            elif source == FeatureSource.ITEM:
                group_name = "item_low_card"
            elif source == FeatureSource.INTERACTION:
                group_name = "interaction"
            else:
                group_name = "other"
            self._append_component_group(groups, group_name, field_name)
        return tuple((str(name), tuple(fields)) for name, fields in groups)


    def _add_absolute_position_embedding(self, seq_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.absolute_position_embedding is None:
            return seq_tokens
        batch_size, seq_len, _ = seq_tokens.shape
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds model max_len {self.max_len}.")
        position_ids = torch.arange(seq_len, device=seq_tokens.device, dtype=torch.long).unsqueeze(0)
        pos = self.absolute_position_embedding(position_ids.expand(batch_size, -1))
        seq_tokens = seq_tokens + pos.to(dtype=seq_tokens.dtype)
        return seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

    def _add_absolute_time_embedding(
        self,
        seq_tokens: torch.Tensor,
        mask: torch.Tensor,
        hist_ts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.absolute_time_embedding is None:
            return seq_tokens
        if hist_ts is None:
            raise ValueError("use_absolute_time=True requires history timestamps.")
        hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
        abs_time = self.absolute_time_embedding(hist_ts.to(device=seq_tokens.device))
        seq_tokens = seq_tokens + abs_time.to(dtype=seq_tokens.dtype)
        return seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

    @staticmethod
    def _same_target_segment_edges(same_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev_same = torch.zeros_like(same_target, dtype=torch.bool)
        next_same = torch.zeros_like(same_target, dtype=torch.bool)
        if same_target.shape[1] > 1:
            prev_same[:, 1:] = same_target[:, :-1]
            next_same[:, :-1] = same_target[:, 1:]
        return same_target & (~prev_same), same_target & (~next_same)

    def _add_segment_relative_position_embedding(
        self,
        seq_tokens: torch.Tensor,
        mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.segment_relative_position_embedding is None:
            return seq_tokens
        batch_size, seq_len, _ = seq_tokens.shape
        if seq_len <= 0:
            return seq_tokens
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_target = known_domain_mask & (hist_domain_ids == target_domain)
        segment_start, segment_end = self._same_target_segment_edges(same_target)
        pos = torch.arange(seq_len, device=seq_tokens.device, dtype=torch.long).view(1, -1)
        start_pos = torch.where(segment_start, pos, torch.full_like(pos, -1)).expand(batch_size, -1)
        current_start = torch.cummax(start_pos, dim=1).values
        end_pos = torch.where(segment_end, pos, torch.full_like(pos, seq_len)).expand(batch_size, -1)
        current_end = torch.cummin(end_pos.flip(1), dim=1).values.flip(1)
        seg_len = (current_end - current_start + 1).clamp_min(1)
        offset = (pos.expand(batch_size, -1) - current_start).clamp_min(0)
        bucket = torch.div(
            offset * int(self.segment_relative_position_buckets),
            seg_len,
            rounding_mode="floor",
        ).clamp(min=0, max=int(self.segment_relative_position_buckets) - 1)
        delta = self.segment_relative_position_embedding(bucket.long()).to(dtype=seq_tokens.dtype)
        delta = delta * same_target.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        bucket_obs = torch.where(
            same_target,
            bucket.to(dtype=seq_tokens.dtype),
            torch.full_like(bucket, -1).to(dtype=seq_tokens.dtype),
        )
        seq_tokens = seq_tokens + delta
        return seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

    def _add_boundary_position_embedding(
        self,
        seq_tokens: torch.Tensor,
        mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.boundary_position_role_embedding is None or self.boundary_position_type_embedding is None:
            return seq_tokens
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_target = known_domain_mask & (hist_domain_ids == target_domain)
        cross_target = known_domain_mask & (hist_domain_ids != target_domain)
        target_entry, target_tail = self._same_target_segment_edges(same_target)
        target_middle = same_target & (~target_entry) & (~target_tail)
        entry_switch = torch.zeros_like(known_domain_mask, dtype=torch.bool)
        pre_switch = torch.zeros_like(known_domain_mask, dtype=torch.bool)
        if known_domain_mask.shape[1] > 1:
            adjacent_known = known_domain_mask[:, :-1] & known_domain_mask[:, 1:]
            adjacent_cross = adjacent_known & (hist_domain_ids[:, :-1] != hist_domain_ids[:, 1:])
            entry_switch[:, 1:] = adjacent_cross
            pre_switch[:, :-1] = adjacent_cross
        boundary = entry_switch | pre_switch
        role_code = torch.zeros_like(hist_domain_ids, dtype=torch.long)
        role_code = torch.where(target_entry, torch.ones_like(role_code), role_code)
        role_code = torch.where(target_middle, torch.full_like(role_code, 2), role_code)
        role_code = torch.where(target_tail, torch.full_like(role_code, 3), role_code)
        type_code = torch.zeros_like(hist_domain_ids, dtype=torch.long)
        type_code = torch.where(boundary & same_target, torch.ones_like(type_code), type_code)
        type_code = torch.where(boundary & cross_target, torch.full_like(type_code, 2), type_code)
        role_active = role_code > 0
        type_active = type_code > 0
        role_delta = self.boundary_position_role_embedding(role_code).to(dtype=seq_tokens.dtype)
        type_delta = self.boundary_position_type_embedding(type_code).to(dtype=seq_tokens.dtype)
        delta = (
            role_delta * role_active.unsqueeze(-1).to(dtype=seq_tokens.dtype)
            + type_delta * type_active.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        )
        delta = delta * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        seq_tokens = seq_tokens + delta
        return seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

    def _apply_hstu_relative_attention(
        self,
        seq_tokens: torch.Tensor,
        mask: torch.Tensor,
        hist_ts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.hstu_relative_attention is None:
            return seq_tokens
        if hist_ts is None:
            raise ValueError("use_hstu_relative_attention=True requires history timestamps.")
        hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
        seq_tokens = self.hstu_relative_attention(seq_tokens, mask, hist_ts)
        return seq_tokens


    def _flatten_grouped_embeddings(
        self,
        embeddings_by_name: dict[str, torch.Tensor],
        groups: tuple[tuple[str, tuple[str, ...]], ...],
        prefix: str,
    ) -> torch.Tensor:
        chunks = []
        for group_name, field_names in groups:
            group_chunks = [embeddings_by_name[field_name] for field_name in field_names if field_name in embeddings_by_name]
            if not group_chunks:
                continue
            group_x = torch.cat(group_chunks, dim=-1) if len(group_chunks) > 1 else group_chunks[0]
            group_x = torch.flatten(group_x, start_dim=1)
            chunks.append(group_x)
        if not chunks:
            raise ValueError(f"No embeddings available for {prefix} groups.")
        return torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]

    def _encode_current_components(self, interaction) -> torch.Tensor:
        current_by_name = self.current_view(interaction, split_by="name")
        current_x = self._flatten_grouped_embeddings(current_by_name, self.current_component_groups, "current")
        return current_x

    def _encode_target_components(self, interaction) -> torch.Tensor:
        target_by_name = self.history_view.forward_target(interaction, split_by="name")
        if self.target_feature_keep is not None:
            target_by_name = {
                field_name: (
                    value
                    if field_name in self.target_feature_keep
                    else torch.zeros_like(value)
                )
                for field_name, value in target_by_name.items()
            }
        target_emb = self._flatten_grouped_embeddings(target_by_name, self.target_component_groups, "target")
        return target_emb

    def _encode_late_target_components(self, interaction) -> Optional[torch.Tensor]:
        if not self.late_target_feature_fields:
            return None
        target_by_name = self.history_view.forward_target(interaction, split_by="name")
        chunks = [
            torch.flatten(target_by_name[field_name], start_dim=1)
            for field_name in self.late_target_feature_fields
        ]
        late_target_x = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
        return late_target_x

    def _history_domain_ids(
        self,
        interaction,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_field = self.domain_seq_field if self.domain_seq_field in interaction else "tab_seq"
        if source_field not in interaction:
            empty_ids = torch.zeros_like(hist_mask, dtype=torch.long)
            return empty_ids, torch.zeros_like(hist_mask, dtype=torch.bool)
        raw_domain = interaction[source_field].long().to(device=hist_mask.device)
        if raw_domain.shape[:2] != hist_mask.shape:
            raise ValueError(
                f"{source_field} shape {tuple(raw_domain.shape)} does not match history mask {tuple(hist_mask.shape)}"
            )
        if self.domain_shift > 0:
            known = hist_mask & (raw_domain > 0)
        else:
            known = hist_mask & (raw_domain >= 0)
        domain_ids = (raw_domain - int(self.domain_shift)).clamp(min=0, max=int(self.num_domains) - 1)
        return domain_ids, known

    def _interaction_scalar_field(
        self,
        interaction,
        field_name: str,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if field_name not in interaction:
            return None
        value = interaction[field_name]
        if not isinstance(value, torch.Tensor) or value.numel() <= 0:
            return None
        value = value.to(device=device, dtype=dtype).view(-1)
        if value.numel() == 1:
            return value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError(f"{field_name} must be scalar or shape [B], got {tuple(value.shape)} for B={batch_size}.")
        return value

    @staticmethod
    def _parse_patch_linear_count(mode: str) -> int:
        mode = str(mode or "none").lower()
        if "patch_linear" not in mode and "patchlinear" not in mode:
            return 0
        for marker in ("_p", "patches"):
            if marker not in mode:
                continue
            suffix = mode.rsplit(marker, 1)[-1]
            digits = []
            for char in suffix:
                if char.isdigit():
                    digits.append(char)
                else:
                    break
            if digits:
                return max(1, int("".join(digits)))
        return 4

    @classmethod
    def attention_off_code_for_variant(cls, variant: str) -> int:
        return int(cls.ATTENTION_OFF_CODES[str(variant)])

    @classmethod
    def final_slot_ablation_code_for_variant(cls, variant: str) -> int:
        return int(cls.FINAL_SLOT_ABLATION_CODES[str(variant)])

    @classmethod
    def final_slot_permute_code_for_variant(cls, variant: str) -> int:
        return int(cls.FINAL_SLOT_PERMUTE_CODES[str(variant)])

    @classmethod
    def querymix_eval_code_for_variant(cls, variant: str) -> int:
        return int(cls.QUERYMIX_EVAL_CODES[str(variant)])

    def _attention_off_code(self, interaction) -> int:
        if self.ATTENTION_OFF_FIELD not in interaction:
            return 0
        value = interaction[self.ATTENTION_OFF_FIELD]
        if not isinstance(value, torch.Tensor) or value.numel() <= 0:
            return 0
        return int(value.detach().long().max().item())

    def _interaction_int_code(self, interaction, field_name: str, default: int = 0) -> int:
        if field_name not in interaction:
            return int(default)
        value = interaction[field_name]
        if not isinstance(value, torch.Tensor) or value.numel() <= 0:
            return int(default)
        return int(value.detach().long().max().item())

    def _final_slot_ablation_code(self, interaction) -> int:
        if self.FINAL_SLOT_ABLATION_FIELD not in interaction:
            return 0
        value = interaction[self.FINAL_SLOT_ABLATION_FIELD]
        if not isinstance(value, torch.Tensor) or value.numel() <= 0:
            return 0
        return int(value.detach().long().max().item())

    def _final_slot_permute_code(self, interaction) -> int:
        if self.FINAL_SLOT_PERMUTE_FIELD not in interaction:
            return 0
        value = interaction[self.FINAL_SLOT_PERMUTE_FIELD]
        if not isinstance(value, torch.Tensor) or value.numel() <= 0:
            return 0
        return int(value.detach().long().max().item())

    def _querymix_eval_code(self, interaction) -> int:
        if self.QUERYMIX_EVAL_FIELD not in interaction:
            return MixFormerBlocks.QUERYMIX_EVAL_NORMAL
        value = interaction[self.QUERYMIX_EVAL_FIELD]
        if not isinstance(value, torch.Tensor) or value.numel() <= 0:
            return MixFormerBlocks.QUERYMIX_EVAL_NORMAL
        return int(value.detach().long().max().item())


    @staticmethod
    def _final_slot_keep_indices(code: int, num_slots: int) -> list[int]:
        code = int(code)
        num_slots = int(num_slots)
        if code <= 0:
            return list(range(num_slots))
        if 100 < code <= 100 + num_slots:
            drop_idx = code - 101
            return [idx for idx in range(num_slots) if idx != drop_idx]
        keep: list[int] = []
        for char in str(code):
            if not char.isdigit():
                continue
            slot_idx = int(char) - 1
            if 0 <= slot_idx < num_slots and slot_idx not in keep:
                keep.append(slot_idx)
        return keep

    def _final_slot_weights(self, interaction, ns_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_slots = int(ns_tokens.shape[0]), int(ns_tokens.shape[1])
        if self.FINAL_SLOT_WEIGHT_FIELD in interaction:
            custom_weights = interaction[self.FINAL_SLOT_WEIGHT_FIELD]
            if not isinstance(custom_weights, torch.Tensor) or custom_weights.numel() <= 0:
                raise ValueError(f"{self.FINAL_SLOT_WEIGHT_FIELD} must be a non-empty tensor.")
            custom_weights = custom_weights.to(device=ns_tokens.device, dtype=ns_tokens.dtype)
            if custom_weights.dim() == 1:
                custom_weights = custom_weights.unsqueeze(0).expand(batch_size, -1)
            if custom_weights.dim() != 2 or custom_weights.shape != (batch_size, num_slots):
                raise ValueError(
                    f"{self.FINAL_SLOT_WEIGHT_FIELD} must have shape "
                    f"[batch_size, num_slots]={batch_size, num_slots}, "
                    f"got {tuple(custom_weights.shape)}."
                )
            weights = custom_weights.clamp_min(0.0)
            return weights
        code = self._final_slot_ablation_code(interaction)
        weights = torch.zeros(
            (batch_size, num_slots),
            device=ns_tokens.device,
            dtype=ns_tokens.dtype,
        )
        keep_indices = self._final_slot_keep_indices(code, num_slots)
        if keep_indices:
            weights[:, keep_indices] = 1.0
        code_values = torch.full(
            (batch_size,),
            float(code),
            device=ns_tokens.device,
            dtype=ns_tokens.dtype,
        )
        return weights

    def _maybe_permute_final_slots(self, interaction, ns_tokens: torch.Tensor) -> torch.Tensor:
        code = self._final_slot_permute_code(interaction)
        batch_size, num_slots = int(ns_tokens.shape[0]), int(ns_tokens.shape[1])
        code_values = torch.full(
            (batch_size,),
            float(code),
            device=ns_tokens.device,
            dtype=ns_tokens.dtype,
        )
        if code <= 0 or num_slots <= 1:
            return ns_tokens
        order = torch.arange(num_slots, device=ns_tokens.device, dtype=torch.long).roll(shifts=-1)
        permuted = ns_tokens.index_select(1, order)
        order_obs = order.to(dtype=ns_tokens.dtype).view(1, -1).expand(batch_size, -1)
        return permuted

    def _pool_final_slots(self, interaction, ns_tokens: torch.Tensor) -> torch.Tensor:
        ns_tokens = self._maybe_permute_final_slots(interaction, ns_tokens)
        slot_norm = ns_tokens.detach().float().norm(dim=-1).to(device=ns_tokens.device, dtype=ns_tokens.dtype)
        slot_weights = self._final_slot_weights(interaction, ns_tokens)
        denom = slot_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized_weights = slot_weights / denom
        contributions = ns_tokens * normalized_weights.unsqueeze(-1)
        pooled = contributions.sum(dim=1)
        return pooled

    @staticmethod
    def _target_bucket_weight(
        *,
        same_target: torch.Tensor,
        bucket_idx: int,
    ) -> torch.Tensor:
        batch_size, seq_len = same_target.shape
        if seq_len <= 0:
            return same_target.to(dtype=torch.float32)
        pos = torch.arange(seq_len, device=same_target.device).view(1, -1)
        prev_same = torch.zeros_like(same_target, dtype=torch.bool)
        next_same = torch.zeros_like(same_target, dtype=torch.bool)
        if seq_len > 1:
            prev_same[:, 1:] = same_target[:, :-1]
            next_same[:, :-1] = same_target[:, 1:]
        segment_start = same_target & (~prev_same)
        segment_end = same_target & (~next_same)

        start_pos = torch.where(segment_start, pos, torch.full_like(pos, -1)).expand(batch_size, -1)
        current_start = torch.cummax(start_pos, dim=1).values
        end_pos = torch.where(segment_end, pos, torch.full_like(pos, seq_len)).expand(batch_size, -1)
        current_end = torch.cummin(end_pos.flip(1), dim=1).values.flip(1)
        seg_len = (current_end - current_start + 1).clamp_min(1).to(dtype=torch.float32)
        offset = (pos.expand(batch_size, -1) - current_start).clamp_min(0).to(dtype=torch.float32)

        token_start = offset / seg_len
        token_end = (offset + 1.0) / seg_len
        bucket_width = 1.0 / 5.0
        bucket_start = float(bucket_idx) * bucket_width
        bucket_end = float(bucket_idx + 1) * bucket_width
        overlap = torch.minimum(token_end, torch.full_like(token_end, bucket_end)) - torch.maximum(
            token_start,
            torch.full_like(token_start, bucket_start),
        )
        weight = (overlap.clamp_min(0.0) / bucket_width).clamp_min(0.0)
        return weight * same_target.to(dtype=weight.dtype)

    @staticmethod
    def _prev_segment_before_target(
        *,
        hist_domain_ids: torch.Tensor,
        cross_target: torch.Tensor,
        target_entry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = cross_target.shape
        prev_tail = torch.zeros_like(cross_target, dtype=torch.bool)
        if seq_len > 1:
            prev_tail[:, :-1] = cross_target[:, :-1] & target_entry[:, 1:]
        if seq_len <= 0:
            return torch.zeros_like(cross_target, dtype=torch.bool), prev_tail
        pos = torch.arange(seq_len, device=cross_target.device).view(1, -1)
        next_same_cross_domain = torch.zeros_like(cross_target, dtype=torch.bool)
        if seq_len > 1:
            next_same_cross_domain[:, :-1] = (
                cross_target[:, :-1]
                & cross_target[:, 1:]
                & (hist_domain_ids[:, :-1] == hist_domain_ids[:, 1:])
            )
        cross_segment_end = cross_target & (~next_same_cross_domain)
        end_pos = torch.where(cross_segment_end, pos, torch.full_like(pos, seq_len)).expand(batch_size, -1)
        current_end = torch.cummin(end_pos.flip(1), dim=1).values.flip(1)
        gather_idx = current_end.clamp(min=0, max=max(seq_len - 1, 0))
        tail_at_end = prev_tail.gather(1, gather_idx)
        prev_segment = cross_target & tail_at_end
        return prev_segment, prev_tail

    def _attention_category_off_weight(
        self,
        *,
        code: int,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        code = int(code)
        if code <= 0:
            return None
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_target = known_domain_mask & (hist_domain_ids == target_domain)
        cross_target = known_domain_mask & (hist_domain_ids != target_domain)
        prev_same = torch.zeros_like(same_target, dtype=torch.bool)
        next_same = torch.zeros_like(same_target, dtype=torch.bool)
        if same_target.shape[1] > 1:
            prev_same[:, 1:] = same_target[:, :-1]
            next_same[:, :-1] = same_target[:, 1:]
        target_entry = same_target & (~prev_same)
        target_tail = same_target & (~next_same)
        prev_segment, prev_tail = self._prev_segment_before_target(
            hist_domain_ids=hist_domain_ids,
            cross_target=cross_target,
            target_entry=target_entry,
        )
        if code == 1:
            return prev_tail.to(dtype=torch.float32)
        if code == 2 or code == 4:
            return prev_segment.to(dtype=torch.float32)
        if code == 3:
            return same_target.to(dtype=torch.float32)
        if code == 5:
            return cross_target.to(dtype=torch.float32)
        if code == 6:
            return target_entry.to(dtype=torch.float32)
        if code == 7:
            return target_tail.to(dtype=torch.float32)
        if 10 <= code <= 14:
            return self._target_bucket_weight(same_target=same_target, bucket_idx=code - 10)
        return None

    @staticmethod
    def _slot_indices_from_code(slot_code: int, num_tokens: int) -> list[int]:
        slot_code = int(slot_code)
        num_tokens = int(num_tokens)
        if slot_code <= 0:
            return list(range(num_tokens))
        indices: list[int] = []
        for char in str(slot_code):
            if not char.isdigit():
                continue
            slot_idx = int(char) - 1
            if 0 <= slot_idx < num_tokens and slot_idx not in indices:
                indices.append(slot_idx)
        if not indices and 1 <= slot_code <= num_tokens:
            indices.append(slot_code - 1)
        return indices

    @staticmethod
    def _layer_code_matches(layer_code: int, layer_idx: int) -> bool:
        layer_code = int(layer_code)
        layer_idx = int(layer_idx)
        if layer_code < 0:
            return True
        if layer_code < 10:
            return layer_code == layer_idx
        return layer_idx in {int(char) for char in str(layer_code) if char.isdigit()}

    def _attention_slot_category_off_bias(
        self,
        interaction,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        layer_idx: int,
        num_tokens: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        code = self._interaction_int_code(interaction, self.ATTENTION_SLOT_CATEGORY_OFF_CODE_FIELD, 0)
        if code <= 0:
            return None
        target_layer = self._interaction_int_code(interaction, self.ATTENTION_SLOT_CATEGORY_OFF_LAYER_FIELD, -1)
        if not self._layer_code_matches(target_layer, layer_idx):
            return None
        target_slot = self._interaction_int_code(interaction, self.ATTENTION_SLOT_CATEGORY_OFF_SLOT_FIELD, 0)
        slot_indices = self._slot_indices_from_code(target_slot, num_tokens)
        if not slot_indices:
            return None
        off_weight = self._attention_category_off_weight(
            code=code,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=target_domain_ids,
        )
        if off_weight is None:
            return None
        bias = torch.zeros(
            (int(hist_domain_ids.shape[0]), int(num_tokens), int(hist_domain_ids.shape[1])),
            device=hist_domain_ids.device,
            dtype=dtype,
        )
        off_bias = (-1.0e4) * off_weight.to(device=hist_domain_ids.device, dtype=dtype)
        if int(target_slot) <= 0:
            bias = bias + off_bias.unsqueeze(1)
        else:
            bias[:, slot_indices, :] = off_bias.unsqueeze(1)
        return bias

    def _attention_slot_output_off_gate(
        self,
        interaction,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        layer_idx: int,
        num_tokens: int,
    ) -> Optional[torch.Tensor]:
        target_slot = self._interaction_int_code(interaction, self.ATTENTION_SLOT_OUTPUT_OFF_SLOT_FIELD, 0)
        if target_slot <= 0:
            return None
        target_layer = self._interaction_int_code(interaction, self.ATTENTION_SLOT_OUTPUT_OFF_LAYER_FIELD, -1)
        if not self._layer_code_matches(target_layer, layer_idx):
            return None
        slot_indices = self._slot_indices_from_code(target_slot, num_tokens)
        if not slot_indices:
            return None
        gate = torch.ones((int(batch_size), int(num_tokens)), device=device, dtype=dtype)
        gate[:, slot_indices] = 0.0
        return gate

    @staticmethod
    def _masked_mean_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    @staticmethod
    def _masked_last_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, token_dim = tokens.shape
        if seq_len <= 0:
            return tokens.new_zeros((batch_size, token_dim))
        pos = torch.arange(seq_len, device=tokens.device).view(1, -1)
        last_pos = torch.where(mask, pos, torch.full_like(pos, -1)).max(dim=1).values
        has_any = last_pos >= 0
        gather_idx = last_pos.clamp(min=0, max=max(seq_len - 1, 0)).view(-1, 1, 1).expand(-1, 1, token_dim)
        gathered = tokens.gather(1, gather_idx).squeeze(1)
        return torch.where(has_any.view(-1, 1), gathered, torch.zeros_like(gathered))

    def _patch_linear_context_tokens(
        self,
        *,
        seq_tokens: torch.Tensor,
        mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        patch_count = int(self.cross_context_patch_count)
        if patch_count <= 0:
            return seq_tokens.new_zeros((seq_tokens.shape[0], seq_tokens.shape[-1]))
        if (
            self.cross_context_patch_target_embedding is None
            or self.cross_context_patch_q_proj is None
            or self.cross_context_patch_k_proj is None
            or self.cross_context_patch_v_proj is None
            or self.cross_context_patch_out_proj is None
        ):
            raise RuntimeError("Patch linear context modules are not initialized.")
        batch_size, seq_len, _ = seq_tokens.shape
        target_domain_ids = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        target_emb = self.cross_context_patch_target_embedding(target_domain_ids).to(dtype=seq_tokens.dtype)
        q_phi = F.elu(self.cross_context_patch_q_proj(target_emb)) + 1.0
        k_phi = F.elu(self.cross_context_patch_k_proj(seq_tokens)) + 1.0
        values = self.cross_context_patch_v_proj(seq_tokens)

        pos = torch.arange(seq_len, device=seq_tokens.device)
        patch_ids = torch.div(pos * patch_count, max(1, seq_len), rounding_mode="floor").clamp(max=patch_count - 1)
        patch_contexts = []
        patch_has_any = []
        for patch_idx in range(patch_count):
            patch_mask = mask & (patch_ids.view(1, -1) == int(patch_idx))
            patch_weight = patch_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
            k_sum = (k_phi * patch_weight).sum(dim=1)
            kv_sum = torch.einsum("bld,ble->bde", k_phi * patch_weight, values)
            numerator = torch.einsum("bd,bde->be", q_phi, kv_sum)
            denominator = (q_phi * k_sum).sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            patch_context = numerator / denominator
            has_any = patch_mask.any(dim=1)
            patch_contexts.append(torch.where(has_any.view(-1, 1), patch_context, torch.zeros_like(patch_context)))
            patch_has_any.append(has_any)
        contexts = torch.stack(patch_contexts, dim=1)
        has_any = torch.stack(patch_has_any, dim=1)
        context = self._masked_mean_tokens(contexts, has_any)
        context = self.cross_context_patch_out_proj(context)
        return context

    def _cross_context_tokens(
        self,
        *,
        seq_tokens: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        mode = str(self.cross_context_mode or "none").lower()
        if mode in {"none", "off", ""} or float(self.cross_context_beta) == 0.0:
            return None
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_target = known_domain_mask & (hist_domain_ids == target_domain)
        cross_target = known_domain_mask & (hist_domain_ids != target_domain)
        prev_same = torch.zeros_like(same_target, dtype=torch.bool)
        if same_target.shape[1] > 1:
            prev_same[:, 1:] = same_target[:, :-1]
        target_entry = same_target & (~prev_same)
        prev_segment, prev_tail = self._prev_segment_before_target(
            hist_domain_ids=hist_domain_ids,
            cross_target=cross_target,
            target_entry=target_entry,
        )
        if mode in {"cross_mean", "sourcedomainmean", "source_domain_mean"}:
            mask = cross_target
            context = self._masked_mean_tokens(seq_tokens, mask)
        elif mode in {"cross_last", "last1"}:
            mask = cross_target
            context = self._masked_last_tokens(seq_tokens, mask)
        elif mode in {"prev_last1", "last1_before_entry"}:
            mask = prev_tail
            context = self._masked_last_tokens(seq_tokens, mask)
        elif mode in {"prev_segment_mean", "segment_mean", "prevseg"}:
            mask = prev_segment
            context = self._masked_mean_tokens(seq_tokens, mask)
        elif "patch_linear" in mode or "patchlinear" in mode:
            mask = cross_target
            context = self._patch_linear_context_tokens(
                seq_tokens=seq_tokens,
                mask=mask,
                target_domain_ids=target_domain_ids,
            )
        else:
            raise ValueError(f"Unsupported cross_context_mode: {self.cross_context_mode!r}.")
        return context

    def _apply_cross_context_residual(
        self,
        *,
        features: torch.Tensor,
        seq_tokens: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        context = self._cross_context_tokens(
            seq_tokens=seq_tokens,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=target_domain_ids,
        )
        if context is None:
            return features
        beta = float(self.cross_context_beta)
        delta = beta * context.to(dtype=features.dtype)
        return features + delta

    def _attention_domain_bias(
        self,
        interaction,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        batch_size = int(hist_domain_ids.shape[0])
        same_bias = self._interaction_scalar_field(
            interaction,
            self.ATTENTION_SAME_BIAS_FIELD,
            batch_size=batch_size,
            device=hist_domain_ids.device,
            dtype=dtype,
        )
        cross_bias = self._interaction_scalar_field(
            interaction,
            self.ATTENTION_CROSS_BIAS_FIELD,
            batch_size=batch_size,
            device=hist_domain_ids.device,
            dtype=dtype,
        )
        off_code = self._attention_off_code(interaction)
        if same_bias is None and cross_bias is None and off_code <= 0 and not self.attention_same_protect:
            return None
        if same_bias is None:
            same_bias = torch.zeros(batch_size, device=hist_domain_ids.device, dtype=dtype)
        if cross_bias is None:
            cross_bias = torch.zeros(batch_size, device=hist_domain_ids.device, dtype=dtype)
        if self.attention_same_protect:
            cross_bias = cross_bias + torch.full_like(cross_bias, -1.0e4)
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_target = known_domain_mask & (hist_domain_ids == target_domain)
        cross_target = known_domain_mask & (hist_domain_ids != target_domain)
        bias = (
            same_bias.view(-1, 1) * same_target.to(dtype=dtype)
            + cross_bias.view(-1, 1) * cross_target.to(dtype=dtype)
        )
        bias = bias * known_domain_mask.to(dtype=dtype)
        off_weight = self._attention_category_off_weight(
            code=off_code,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=target_domain_ids,
        )
        if off_weight is not None:
            off_weight = off_weight.to(device=hist_domain_ids.device, dtype=dtype)
            bias = bias + (-1.0e4) * off_weight
        return bias

    def _attention_double_softmax_masks(
        self,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.attention_double_softmax_beta is None:
            return None, None
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_target = known_domain_mask & (hist_domain_ids == target_domain)
        cross_target = known_domain_mask & (hist_domain_ids != target_domain)
        return same_target, cross_target

    @staticmethod
    def _merge_attention_bias(
        base_bias: Optional[torch.Tensor],
        slot_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if slot_bias is None:
            return base_bias
        if base_bias is None:
            return slot_bias
        if base_bias.dim() == 2:
            return base_bias.unsqueeze(1) + slot_bias
        if base_bias.dim() == 3:
            return base_bias + slot_bias
        raise ValueError(f"Unsupported attention bias shape: {tuple(base_bias.shape)}")

    def _attention_slot_bias_and_output_gate(
        self,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        dtype: torch.dtype,
        num_tokens: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None

    def _fuse_sequence_relative_time(
        self,
        seq_emb: torch.Tensor,
        relative_time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse candidate-relative time before sequence-token projection.

        The generic MixFormer contract remains additive.  Dataset-specific
        models may override this hook when individual fields must remain
        explicit features until a learned token-fusion projection.
        """
        if seq_emb.shape != relative_time_emb.shape:
            raise ValueError(
                "Additive relative-time fusion requires matching tensors, got "
                f"sequence={tuple(seq_emb.shape)} and time={tuple(relative_time_emb.shape)}."
            )
        return seq_emb + relative_time_emb

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        self._last_m6_qnorm_aux_terms = []
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)
        attention_bias = self._attention_domain_bias(
            interaction,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            dtype=seq_emb.dtype,
        )
        attention_slot_bias, attention_output_gate = self._attention_slot_bias_and_output_gate(
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            dtype=seq_emb.dtype,
            num_tokens=self.num_ns_token,
        )
        attention_bias = self._merge_attention_bias(attention_bias, attention_slot_bias)
        attention_same_mask, attention_cross_mask = self._attention_double_softmax_masks(
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
        )

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts_seconds = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            relative_time_emb = self.relative_time_embedding(
                curr_ts,
                hist_ts_seconds,
                seq_lens=seq_len,
            )
            seq_emb = self._fuse_sequence_relative_time(seq_emb, relative_time_emb)
        if (self.absolute_time_embedding is not None or self.hstu_relative_attention is not None) and hist_ts is None:
            hist_ts = interaction[self.history_view.time_field]
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        seq_tokens = self._add_absolute_time_embedding(seq_tokens, mask, hist_ts)
        seq_tokens = self._add_absolute_position_embedding(seq_tokens, mask)
        seq_tokens = self._add_segment_relative_position_embedding(
            seq_tokens,
            mask,
            hist_domain_ids,
            known_domain_mask,
            domain_ids,
        )
        seq_tokens = self._add_boundary_position_embedding(
            seq_tokens,
            mask,
            hist_domain_ids,
            known_domain_mask,
            domain_ids,
        )
        seq_tokens = self._apply_hstu_relative_attention(seq_tokens, mask, hist_ts)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        querymix_eval_code = self._querymix_eval_code(interaction)
        _, ns_tokens = self.mixformer(
            seq_tokens,
            ns_tokens,
            mask,
            attention_bias=attention_bias,
            query_modulation_domain_ids=domain_ids,
            attention_double_softmax_beta=self.attention_double_softmax_beta,
            attention_double_softmax_mode=self.attention_double_softmax_mode,
            attention_same_mask=attention_same_mask,
            attention_cross_mask=attention_cross_mask,
            attention_output_gate=attention_output_gate,
            querymix_eval_code=querymix_eval_code,
        )
        features = self.dropout(self._pool_final_slots(interaction, ns_tokens))
        features = self._apply_cross_context_residual(
            features=features,
            seq_tokens=seq_tokens,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
        )
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(x, domain_ids)

    def logits(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        logits = self.forward(x, domain_ids)
        return logits

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.logits(interaction)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if self.lsp.active:
            self.lsp.add("task", loss)
        return loss


class M6DomainIdMixFormerModel(MixFormerModel):
    """MixFormer baseline for domain-id injection order around shift/FFN/KV."""

    M6_MODE_CODES = {
        "postffn_domainid_kv": 1,
        "shift_pre_domainid_kv": 2,
        "domainid_pre_shift_kv": 3,
        "shift_pre_no_domainid_kv": 4,
        "layered_l0_all_l1_domainid_all_l2_same_kv": 5,
        "shift_pre_no_domainid_kv_q4off": 6,
        "shift_pre_no_domainid_kv_segq4": 7,
        "shift_pre_no_domainid_kv_segq4_cap": 8,
    }
    M6_SHIFT_MODE_CODES = {
        "none": 0,
        "all": 1,
        "same": 2,
        "cross": 3,
    }

    def __init__(
        self,
        *args,
        m6_domainid_mode: str = "postffn_domainid_kv",
        m6_domainid_init_std: float = 1e-4,
        m6_event_shift_mode: str = "same",
        m6_segment_slot_cap: int = 100,
        m6_qnorm_aux_weight: float = 0.0,
        m6_qnorm_aux_layers: str = "2",
        m6_qnorm_aux_max_ratio: float = 1.25,
        m6_q4_anchor_aux_weight: float = 0.0,
        m6_q4_anchor_aux_mode: str = "off",
        m6_q4_anchor_aux_layers: str = "2",
        m6_q4_anchor_aux_margin: float = 0.2,
        m6_q4_anchor_aux_max_cos: float = 0.25,
        m6_q4_anchor_aux_detach_others: bool = True,
        m6_qslot_cohesion_aux_weight: float = 0.0,
        m6_qslot_div_aux_weight: float = 0.0,
        m6_qslot_aux_layers: str = "2",
        m6_qslot_aux_margin: float = 0.2,
        m6_q4_complement_aux_weight: float = 0.0,
        m6_q4_complement_aux_layers: str = "2",
        m6_q4_complement_detach_main: bool = True,
        m6_q4_complement_power: float = 1.0,
        m6_querymix_identity_aux_weight: float = 0.0,
        m6_querymix_identity_aux_layers: str = "0,1",
        m6_querymix_identity_aux_margin: float = 0.05,
        m6_querymix_identity_aux_detach_before: bool = True,
        m6_q34_survival_aux_weight: float = 0.0,
        m6_q4_survival_aux_weight: float = 0.0,
        m6_qtheme_window_aux_weight: float = 0.0,
        m6_qtheme_window_aux_layers: str = "2",
        m6_qtheme_window_size: int = 8,
        m6_qtheme_window_stride: int = 4,
        m6_qtheme_entropy_low: float = 0.45,
        m6_qtheme_entropy_high: float = 0.75,
        m6_qtheme_load_top_cap: float = 0.78,
        m6_qtheme_load_entropy_floor: float = 0.55,
        m6_qtheme_segment_aux_weight: float = 0.0,
        m6_qtheme_segment_aux_layers: str = "2",
        m6_qtheme_segment_entropy_low: float = 0.45,
        m6_qtheme_segment_entropy_high: float = 0.75,
        m6_qtheme_segment_min_len: int = 2,
        m6_qtheme_segment_include_prev: bool = True,
        m6_qtheme_segment_include_target: bool = True,
        m6_qtheme_segment_include_boundary: bool = True,
        m6_final_q_gate: bool = False,
        m6_final_q_gate_floor: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        mode = str(m6_domainid_mode or "postffn_domainid_kv").lower()
        if mode not in self.M6_MODE_CODES:
            raise ValueError(
                f"m6_domainid_mode must be one of {sorted(self.M6_MODE_CODES)}, "
                f"got {m6_domainid_mode!r}."
            )
        if int(self.token_dim) % 4 != 0:
            raise ValueError("M6DomainIdMixFormerModel requires token_dim divisible by 4.")
        self.m6_domainid_mode = mode
        self.m6_domainid_init_std = float(m6_domainid_init_std)
        event_shift_mode = str(m6_event_shift_mode or "same").lower()
        if event_shift_mode != "same":
            raise ValueError(
                "M6 event shift is domain-local by design; "
                f"m6_event_shift_mode must be 'same', got {m6_event_shift_mode!r}."
            )
        self.m6_event_shift_mode = event_shift_mode
        self.m6_segment_slot_cap = max(0, int(m6_segment_slot_cap))
        self.m6_qnorm_aux_weight = float(m6_qnorm_aux_weight)
        self.m6_qnorm_aux_layers = self._parse_m6_qnorm_aux_layers(m6_qnorm_aux_layers)
        self.m6_qnorm_aux_max_ratio = float(m6_qnorm_aux_max_ratio)
        self._last_m6_qnorm_aux_terms: list[torch.Tensor] = []
        q4_anchor_mode = str(m6_q4_anchor_aux_mode or "off").lower()
        allowed_q4_anchor_modes = {"off", "cross_rank", "cos_div", "cross_rank_cos"}
        if q4_anchor_mode not in allowed_q4_anchor_modes:
            raise ValueError(
                f"m6_q4_anchor_aux_mode must be one of {sorted(allowed_q4_anchor_modes)}, "
                f"got {m6_q4_anchor_aux_mode!r}."
            )
        self.m6_q4_anchor_aux_weight = float(m6_q4_anchor_aux_weight)
        self.m6_q4_anchor_aux_mode = q4_anchor_mode
        self.m6_q4_anchor_aux_layers = self._parse_m6_qnorm_aux_layers(m6_q4_anchor_aux_layers)
        self.m6_q4_anchor_aux_margin = float(m6_q4_anchor_aux_margin)
        self.m6_q4_anchor_aux_max_cos = float(m6_q4_anchor_aux_max_cos)
        self.m6_q4_anchor_aux_detach_others = bool(m6_q4_anchor_aux_detach_others)
        self._last_m6_q4_anchor_aux_terms: list[torch.Tensor] = []
        self.m6_qslot_cohesion_aux_weight = float(m6_qslot_cohesion_aux_weight)
        self.m6_qslot_div_aux_weight = float(m6_qslot_div_aux_weight)
        self.m6_qslot_aux_layers = self._parse_m6_qnorm_aux_layers(m6_qslot_aux_layers)
        self.m6_qslot_aux_margin = float(m6_qslot_aux_margin)
        self._last_m6_qslot_cohesion_aux_terms: list[torch.Tensor] = []
        self._last_m6_qslot_div_aux_terms: list[torch.Tensor] = []
        self.m6_q4_complement_aux_weight = float(m6_q4_complement_aux_weight)
        self.m6_q4_complement_aux_layers = self._parse_m6_qnorm_aux_layers(m6_q4_complement_aux_layers)
        self.m6_q4_complement_detach_main = bool(m6_q4_complement_detach_main)
        self.m6_q4_complement_power = float(m6_q4_complement_power)
        self._last_m6_q4_complement_aux_terms: list[torch.Tensor] = []
        self.m6_querymix_identity_aux_weight = float(m6_querymix_identity_aux_weight)
        self.m6_querymix_identity_aux_layers = self._parse_m6_qnorm_aux_layers(m6_querymix_identity_aux_layers)
        self.m6_querymix_identity_aux_margin = float(m6_querymix_identity_aux_margin)
        self.m6_querymix_identity_aux_detach_before = bool(m6_querymix_identity_aux_detach_before)
        self._last_m6_querymix_identity_aux_terms: list[torch.Tensor] = []
        self.m6_q34_survival_aux_weight = float(m6_q34_survival_aux_weight)
        self.m6_q4_survival_aux_weight = float(m6_q4_survival_aux_weight)
        self.m6_qtheme_window_aux_weight = float(m6_qtheme_window_aux_weight)
        self.m6_qtheme_window_aux_layers = self._parse_m6_qnorm_aux_layers(m6_qtheme_window_aux_layers)
        self.m6_qtheme_window_size = max(int(m6_qtheme_window_size), 1)
        self.m6_qtheme_window_stride = max(int(m6_qtheme_window_stride), 1)
        self.m6_qtheme_entropy_low = float(m6_qtheme_entropy_low)
        self.m6_qtheme_entropy_high = float(m6_qtheme_entropy_high)
        self.m6_qtheme_load_top_cap = float(m6_qtheme_load_top_cap)
        self.m6_qtheme_load_entropy_floor = float(m6_qtheme_load_entropy_floor)
        self._last_m6_qtheme_window_aux_terms: list[torch.Tensor] = []
        self.m6_qtheme_segment_aux_weight = float(m6_qtheme_segment_aux_weight)
        self.m6_qtheme_segment_aux_layers = self._parse_m6_qnorm_aux_layers(m6_qtheme_segment_aux_layers)
        self.m6_qtheme_segment_entropy_low = float(m6_qtheme_segment_entropy_low)
        self.m6_qtheme_segment_entropy_high = float(m6_qtheme_segment_entropy_high)
        self.m6_qtheme_segment_min_len = max(int(m6_qtheme_segment_min_len), 1)
        self.m6_qtheme_segment_include_prev = bool(m6_qtheme_segment_include_prev)
        self.m6_qtheme_segment_include_target = bool(m6_qtheme_segment_include_target)
        self.m6_qtheme_segment_include_boundary = bool(m6_qtheme_segment_include_boundary)
        self._last_m6_qtheme_segment_aux_terms: list[torch.Tensor] = []
        self.m6_q34_survival_head = (
            nn.Linear(int(self.token_dim), 1)
            if self.m6_q34_survival_aux_weight > 0.0
            else None
        )
        self.m6_q4_survival_head = (
            nn.Linear(int(self.token_dim), 1)
            if self.m6_q4_survival_aux_weight > 0.0
            else None
        )
        self._last_m6_q34_survival_logits: Optional[torch.Tensor] = None
        self._last_m6_q4_survival_logits: Optional[torch.Tensor] = None
        self._last_dvc_aux_loss_stats: dict[str, float] = {}
        self.m6_final_q_gate_enabled = bool(m6_final_q_gate)
        self.m6_final_q_gate_floor = float(m6_final_q_gate_floor)
        self.m6_final_q_gate_logits = None
        if self.m6_final_q_gate_enabled:
            self.m6_final_q_gate_logits = nn.Embedding(int(self.num_domains), int(self.num_ns_token))
            nn.init.zeros_(self.m6_final_q_gate_logits.weight)
        self.m6_domain_embedding = nn.Embedding(int(self.num_domains), int(self.token_dim))
        nn.init.normal_(self.m6_domain_embedding.weight, mean=0.0, std=self.m6_domainid_init_std)
        self.m6_domain_inject_mlp = nn.Sequential(
            nn.Linear(int(self.token_dim) * 2, int(self.token_dim)),
            nn.SiLU(),
            nn.Linear(int(self.token_dim), int(self.token_dim)),
        )
        nn.init.normal_(self.m6_domain_inject_mlp[-1].weight, mean=0.0, std=self.m6_domainid_init_std)
        nn.init.zeros_(self.m6_domain_inject_mlp[-1].bias)
        num_mix_layers = int(len(self.mixformer.query_mixers))
        self.m6_shift_begin = nn.Parameter(torch.zeros(num_mix_layers, int(self.token_dim)))
        self.m6_shift_end = nn.Parameter(torch.zeros(num_mix_layers, int(self.token_dim)))
        self.m6_segment_sidecar_capped = mode == "shift_pre_no_domainid_kv_segq4_cap"
        self.m6_segment_sidecar_enabled = mode in {
            "shift_pre_no_domainid_kv_segq4",
            "shift_pre_no_domainid_kv_segq4_cap",
        }
        if self.m6_segment_sidecar_enabled:
            dim = int(self.token_dim)
            # Parameters exist only for the explicit SEG model.  The historical
            # shift_pre_no_domainid_kv state_dict and RNG stream stay unchanged.
            self.m6_segment_seed = nn.Parameter(torch.empty(dim))
            nn.init.normal_(self.m6_segment_seed, mean=0.0, std=self.m6_domainid_init_std)
            self.m6_segment_tail_time = RelativeTimeEmbedding(
                embedding_dim=dim,
                boundaries=TIME_BUCKET_BOUNDARIES,
                init_std=self.m6_domainid_init_std,
            )
            self.m6_segment_candidate_time = RelativeTimeEmbedding(
                embedding_dim=dim,
                boundaries=TIME_BUCKET_BOUNDARIES,
                init_std=self.m6_domainid_init_std,
            )
            self.m6_segment_q_norms = nn.ModuleList([_make_norm(dim) for _ in range(num_mix_layers)])
            self.m6_segment_event_norms = nn.ModuleList([_make_norm(dim) for _ in range(num_mix_layers)])
            self.m6_segment_q_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_mix_layers)])
            self.m6_segment_k_projs = nn.ModuleList([nn.Linear(dim * 2, dim, bias=False) for _ in range(num_mix_layers)])
            self.m6_segment_v_projs = nn.ModuleList([nn.Linear(dim * 2, dim, bias=False) for _ in range(num_mix_layers)])
            self.m6_segment_out_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_mix_layers)])
            self.m6_segment_ffn_norms = nn.ModuleList([_make_norm(dim) for _ in range(num_mix_layers)])
            self.m6_segment_ffns = nn.ModuleList([
                SwiGLU(input_dim=dim, expand=2, dropout_rate=0.0)
                for _ in range(num_mix_layers)
            ])
            self.m6_cross_segment_norms = nn.ModuleList([_make_norm(dim) for _ in range(num_mix_layers)])
            self.m6_cross_segment_k_projs = nn.ModuleList([nn.Linear(dim * 2, dim, bias=False) for _ in range(num_mix_layers)])
            self.m6_cross_segment_v_projs = nn.ModuleList([nn.Linear(dim * 2, dim, bias=False) for _ in range(num_mix_layers)])
        self._last_m6_segment_observations: list[dict[str, torch.Tensor]] = []

    @staticmethod
    def _parse_m6_qnorm_aux_layers(layers: str | Iterable[int]) -> tuple[int, ...]:
        if isinstance(layers, str):
            text = layers.strip().lower()
            if text in {"", "none", "off"}:
                return ()
            if text == "all":
                return (0, 1, 2)
            values = []
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                values.append(int(part))
            return tuple(sorted(set(values)))
        return tuple(sorted({int(layer) for layer in layers}))

    def _m6_qnorm_aux_enabled(self, layer_idx: int) -> bool:
        return (
            self.training
            and float(self.m6_qnorm_aux_weight) > 0.0
            and int(layer_idx) in set(self.m6_qnorm_aux_layers)
        )

    def _m6_q4_anchor_aux_enabled(self, layer_idx: int) -> bool:
        return (
            self.training
            and float(self.m6_q4_anchor_aux_weight) > 0.0
            and str(self.m6_q4_anchor_aux_mode) != "off"
            and int(layer_idx) in set(self.m6_q4_anchor_aux_layers)
        )

    def _m6_qslot_aux_enabled(self, layer_idx: int) -> bool:
        return (
            self.training
            and (
                float(self.m6_qslot_cohesion_aux_weight) > 0.0
                or float(self.m6_qslot_div_aux_weight) > 0.0
            )
            and int(layer_idx) in set(self.m6_qslot_aux_layers)
        )

    def _m6_q4_complement_aux_enabled(self, layer_idx: int) -> bool:
        return (
            self.training
            and float(self.m6_q4_complement_aux_weight) > 0.0
            and int(layer_idx) in set(self.m6_q4_complement_aux_layers)
        )

    def _m6_querymix_identity_aux_enabled(self, layer_idx: int) -> bool:
        return (
            self.training
            and float(self.m6_querymix_identity_aux_weight) > 0.0
            and int(layer_idx) in set(self.m6_querymix_identity_aux_layers)
        )

    def _m6_qtheme_window_aux_enabled(self, layer_idx: int) -> bool:
        return (
            self.training
            and float(self.m6_qtheme_window_aux_weight) > 0.0
            and int(layer_idx) in set(self.m6_qtheme_window_aux_layers)
        )

    def _m6_qtheme_segment_aux_enabled(self, layer_idx: int) -> bool:
        return (
            self.training
            and float(self.m6_qtheme_segment_aux_weight) > 0.0
            and int(layer_idx) in set(self.m6_qtheme_segment_aux_layers)
        )

    def _m6_attention_aux_enabled(self, layer_idx: int) -> bool:
        return (
            self._m6_qnorm_aux_enabled(layer_idx)
            or self._m6_q4_anchor_aux_enabled(layer_idx)
            or self._m6_qslot_aux_enabled(layer_idx)
            or self._m6_q4_complement_aux_enabled(layer_idx)
            or self._m6_qtheme_window_aux_enabled(layer_idx)
            or self._m6_qtheme_segment_aux_enabled(layer_idx)
        )

    def _m6_record_qnorm_aux(self, logits: torch.Tensor, layer_idx: int) -> None:
        if not self._m6_qnorm_aux_enabled(layer_idx):
            return
        if logits.ndim != 3:
            raise ValueError(f"Expected M6 attention logits [B, Q, L], got {tuple(logits.shape)}")
        norms = logits.float().norm(dim=-1)
        mean_norm = norms.mean(dim=1, keepdim=True).clamp_min(1.0e-6)
        ratio = norms / mean_norm
        max_ratio = max(float(self.m6_qnorm_aux_max_ratio), 1.0)
        excess = torch.relu(ratio - max_ratio)
        term = excess.pow(2).mean()
        self._last_m6_qnorm_aux_terms.append(term)

    def _m6_record_q4_anchor_aux(
        self,
        weights: torch.Tensor,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        layer_idx: int,
    ) -> None:
        if not self._m6_q4_anchor_aux_enabled(layer_idx):
            return
        if weights.ndim != 3:
            raise ValueError(f"Expected M6 attention weights [B, Q, L], got {tuple(weights.shape)}")
        batch_size, num_slots, seq_len = int(weights.shape[0]), int(weights.shape[1]), int(weights.shape[2])
        if num_slots <= 1 or seq_len <= 0:
            return
        anchor_idx = min(3, num_slots - 1)
        other_indices = [idx for idx in range(num_slots) if idx != anchor_idx]
        if not other_indices:
            return

        device = weights.device
        hist_len = min(seq_len, int(known_domain_mask.shape[1]))
        weight_f = weights[..., :hist_len].float()
        hist_domains = hist_domain_ids.to(device=device).long()[:, :hist_len]
        known = known_domain_mask.to(device=device, dtype=torch.bool)[:, :hist_len]
        target = target_domain_ids.to(device=device).long().view(-1, 1)
        cross = known & (hist_domains != target)
        same = known & (hist_domains == target)
        valid_sample = cross.any(dim=1) & same.any(dim=1)
        if not bool(valid_sample.any().item()):
            return

        eps = 1.0e-6
        known_f = known.to(dtype=weight_f.dtype)
        cross_f = cross.to(dtype=weight_f.dtype)
        same_f = same.to(dtype=weight_f.dtype)
        valid_mass = (weight_f * known_f.unsqueeze(1)).sum(dim=-1)
        valid_slot = valid_mass > eps
        valid_pair_base = valid_sample.view(batch_size, 1) & valid_slot[:, [anchor_idx]] & valid_slot[:, other_indices]

        parts: list[torch.Tensor] = []
        mode = str(self.m6_q4_anchor_aux_mode)
        if mode in {"cross_rank", "cross_rank_cos"}:
            cross_mass = (weight_f * cross_f.unsqueeze(1)).sum(dim=-1)
            same_mass = (weight_f * same_f.unsqueeze(1)).sum(dim=-1)
            score = torch.log(cross_mass + eps) - torch.log(same_mass + eps)
            anchor_score = score[:, anchor_idx].unsqueeze(1)
            other_score = score[:, other_indices]
            if self.m6_q4_anchor_aux_detach_others:
                other_score = other_score.detach()
            gap = anchor_score - other_score
            valid_pair = valid_pair_base.to(dtype=gap.dtype)
            if bool(valid_pair.bool().any().item()):
                cross_rank_loss = (
                    F.softplus(float(self.m6_q4_anchor_aux_margin) - gap) * valid_pair
                ).sum() / valid_pair.sum().clamp_min(1.0)
                parts.append(cross_rank_loss)

        if mode in {"cos_div", "cross_rank_cos"}:
            masked = weight_f * known_f.unsqueeze(1)
            anchor_vec = masked[:, anchor_idx, :]
            other_vec = masked[:, other_indices, :]
            if self.m6_q4_anchor_aux_detach_others:
                other_vec = other_vec.detach()
            dot = (anchor_vec.unsqueeze(1) * other_vec).sum(dim=-1)
            anchor_norm = anchor_vec.norm(dim=-1, keepdim=True).clamp_min(eps)
            other_norm = other_vec.norm(dim=-1).clamp_min(eps)
            cos = dot / (anchor_norm * other_norm)
            valid_pair = valid_pair_base.to(dtype=cos.dtype)
            if bool(valid_pair.bool().any().item()):
                cos_loss = (torch.relu(cos - float(self.m6_q4_anchor_aux_max_cos)).pow(2) * valid_pair).sum()
                cos_loss = cos_loss / valid_pair.sum().clamp_min(1.0)
                parts.append(cos_loss)

        if not parts:
            return
        term = torch.stack(parts).mean()
        self._last_m6_q4_anchor_aux_terms.append(term)

    def _m6_record_qslot_aux(
        self,
        weights: torch.Tensor,
        token_reprs: torch.Tensor,
        *,
        known_domain_mask: torch.Tensor,
        layer_idx: int,
    ) -> None:
        if not self._m6_qslot_aux_enabled(layer_idx):
            return
        if weights.ndim != 3:
            raise ValueError(f"Expected M6 attention weights [B, Q, L], got {tuple(weights.shape)}")
        if token_reprs.ndim != 3:
            raise ValueError(f"Expected M6 qslot token representations [B, L, D], got {tuple(token_reprs.shape)}")
        batch_size, num_slots, seq_len = int(weights.shape[0]), int(weights.shape[1]), int(weights.shape[2])
        if batch_size <= 0 or num_slots <= 1 or seq_len <= 0:
            return
        hist_len = min(seq_len, int(token_reprs.shape[1]), int(known_domain_mask.shape[1]))
        if hist_len <= 0:
            return
        eps = 1.0e-6
        weight_f = weights[..., :hist_len].float()
        z = token_reprs[:, :hist_len, :].float()
        valid = known_domain_mask.to(device=weights.device, dtype=torch.bool)[:, :hist_len]
        valid_f = valid.to(dtype=weight_f.dtype)
        masked_weight = weight_f * valid_f.unsqueeze(1)
        mass = masked_weight.sum(dim=-1).clamp_min(eps)
        slot_valid = (masked_weight.sum(dim=-1) > eps)
        if not bool(slot_valid.any().item()):
            return

        centers = torch.einsum("bql,bld->bqd", masked_weight, z) / mass.unsqueeze(-1)
        centers_norm = F.normalize(centers, dim=-1, eps=eps)
        z_norm = F.normalize(z, dim=-1, eps=eps)

        if float(self.m6_qslot_cohesion_aux_weight) > 0.0:
            token_cos = torch.einsum("bqd,bld->bql", centers_norm, z_norm)
            cohesion_per_slot = (masked_weight * (1.0 - token_cos)).sum(dim=-1) / mass
            cohesion_mask = slot_valid.to(dtype=cohesion_per_slot.dtype)
            cohesion_loss = (cohesion_per_slot * cohesion_mask).sum() / cohesion_mask.sum().clamp_min(1.0)
            self._last_m6_qslot_cohesion_aux_terms.append(cohesion_loss)

        if float(self.m6_qslot_div_aux_weight) > 0.0:
            center_cos = torch.matmul(centers_norm, centers_norm.transpose(1, 2))
            eye = torch.eye(num_slots, device=weights.device, dtype=torch.bool).unsqueeze(0)
            pair_valid = slot_valid.unsqueeze(1) & slot_valid.unsqueeze(2) & ~eye
            if bool(pair_valid.any().item()):
                margin = float(self.m6_qslot_aux_margin)
                div_loss = torch.relu(center_cos - margin).pow(2)
                div_mask = pair_valid.to(dtype=div_loss.dtype)
                div_loss = (div_loss * div_mask).sum() / div_mask.sum().clamp_min(1.0)
                self._last_m6_qslot_div_aux_terms.append(div_loss)

    def _m6_record_q4_complement_aux(
        self,
        weights: torch.Tensor,
        *,
        known_domain_mask: torch.Tensor,
        layer_idx: int,
    ) -> None:
        if not self._m6_q4_complement_aux_enabled(layer_idx):
            return
        if weights.ndim != 3:
            raise ValueError(f"Expected M6 attention weights [B, Q, L], got {tuple(weights.shape)}")
        batch_size, num_slots, seq_len = int(weights.shape[0]), int(weights.shape[1]), int(weights.shape[2])
        if batch_size <= 0 or num_slots <= 1 or seq_len <= 0:
            return
        anchor_idx = min(3, num_slots - 1)
        main_indices = [idx for idx in range(num_slots) if idx != anchor_idx]
        if not main_indices:
            return

        hist_len = min(seq_len, int(known_domain_mask.shape[1]))
        if hist_len <= 0:
            return
        eps = 1.0e-6
        weight_f = weights[..., :hist_len].float()
        valid = known_domain_mask.to(device=weights.device, dtype=torch.bool)[:, :hist_len]
        valid_f = valid.to(dtype=weight_f.dtype)
        sample_valid = valid.any(dim=1)
        if not bool(sample_valid.any().item()):
            return

        main_cover = weight_f[:, main_indices, :].max(dim=1).values * valid_f
        if self.m6_q4_complement_detach_main:
            main_cover = main_cover.detach()
        main_max = main_cover.max(dim=1, keepdim=True).values.clamp_min(eps)
        main_cover_norm = (main_cover / main_max).clamp(min=0.0, max=1.0)
        target = (1.0 - main_cover_norm) * valid_f
        power = max(float(self.m6_q4_complement_power), 0.0)
        if abs(power - 1.0) > 1.0e-8:
            target = target.clamp_min(0.0).pow(power)
        target_sum = target.sum(dim=1, keepdim=True)
        sample_mask = sample_valid & (target_sum.view(-1) > eps)
        if not bool(sample_mask.any().item()):
            return
        target = target / target_sum.clamp_min(eps)
        if self.m6_q4_complement_detach_main:
            target = target.detach()

        q4_attn = weight_f[:, anchor_idx, :].clamp_min(eps)
        loss_per_sample = -(target * q4_attn.log()).sum(dim=1)
        sample_mask_f = sample_mask.to(dtype=loss_per_sample.dtype)
        comp_loss = (loss_per_sample * sample_mask_f).sum() / sample_mask_f.sum().clamp_min(1.0)
        target_entropy = -(target.clamp_min(eps) * target.clamp_min(eps).log()).sum(dim=1)
        self._last_m6_q4_complement_aux_terms.append(comp_loss)

    def _m6_record_qtheme_window_aux(
        self,
        weights: torch.Tensor,
        *,
        known_domain_mask: torch.Tensor,
        layer_idx: int,
    ) -> None:
        if not self._m6_qtheme_window_aux_enabled(layer_idx):
            return
        if weights.ndim != 3:
            raise ValueError(f"Expected M6 attention weights [B, Q, L], got {tuple(weights.shape)}")
        batch_size, num_slots, seq_len = int(weights.shape[0]), int(weights.shape[1]), int(weights.shape[2])
        if batch_size <= 0 or num_slots <= 1 or seq_len <= 0:
            return
        hist_len = min(seq_len, int(known_domain_mask.shape[1]))
        if hist_len <= 0:
            return

        eps = 1.0e-6
        weight_f = weights[..., :hist_len].float()
        valid = known_domain_mask.to(device=weights.device, dtype=torch.bool)[:, :hist_len]
        valid_f = valid.to(dtype=weight_f.dtype)
        window_size = min(max(int(self.m6_qtheme_window_size), 1), hist_len)
        stride = max(int(self.m6_qtheme_window_stride), 1)
        entropy_low = max(float(self.m6_qtheme_entropy_low), 0.0)
        entropy_high = min(max(float(self.m6_qtheme_entropy_high), entropy_low), 1.0)

        group_losses: list[torch.Tensor] = []
        group_entropy_values: list[torch.Tensor] = []
        load_accum = weight_f.new_zeros(num_slots)
        load_seen = False
        log_q = math.log(float(num_slots))
        for start in range(0, hist_len, stride):
            end = min(start + window_size, hist_len)
            if end <= start:
                continue
            win_valid = valid[:, start:end]
            sample_valid = win_valid.any(dim=1)
            if not bool(sample_valid.any().item()):
                continue
            win_weight = weight_f[:, :, start:end] * valid_f[:, start:end].unsqueeze(1)
            q_mass = win_weight.sum(dim=-1)
            total_mass = q_mass.sum(dim=1, keepdim=True)
            sample_valid = sample_valid & (total_mass.view(-1) > eps)
            if not bool(sample_valid.any().item()):
                continue
            p = q_mass / total_mass.clamp_min(eps)
            entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1) / log_q
            band_loss = torch.relu(entropy_low - entropy).pow(2) + torch.relu(entropy - entropy_high).pow(2)
            sample_mask = sample_valid.to(dtype=band_loss.dtype)
            group_losses.append((band_loss * sample_mask).sum() / sample_mask.sum().clamp_min(1.0))
            group_entropy_values.append((entropy.detach() * sample_mask).sum() / sample_mask.sum().clamp_min(1.0))
            load_accum = load_accum + q_mass[sample_valid].sum(dim=0)
            load_seen = True

        if not group_losses or not load_seen:
            return
        theme_band_loss = torch.stack(group_losses).mean()
        load_p = load_accum / load_accum.sum().clamp_min(eps)
        load_entropy = -(load_p.clamp_min(eps) * load_p.clamp_min(eps).log()).sum() / log_q
        load_top = load_p.max()
        load_loss = torch.relu(load_top - float(self.m6_qtheme_load_top_cap)).pow(2)
        load_loss = load_loss + torch.relu(float(self.m6_qtheme_load_entropy_floor) - load_entropy).pow(2)
        total_loss = theme_band_loss + load_loss
        self._last_m6_qtheme_window_aux_terms.append(total_loss)

    def _m6_record_qtheme_segment_aux(
        self,
        weights: torch.Tensor,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        layer_idx: int,
    ) -> None:
        if not self._m6_qtheme_segment_aux_enabled(layer_idx):
            return
        if weights.ndim != 3:
            raise ValueError(f"Expected M6 attention weights [B, Q, L], got {tuple(weights.shape)}")
        batch_size, num_slots, seq_len = int(weights.shape[0]), int(weights.shape[1]), int(weights.shape[2])
        if batch_size <= 0 or num_slots <= 1 or seq_len <= 0:
            return
        hist_len = min(seq_len, int(hist_domain_ids.shape[1]), int(known_domain_mask.shape[1]))
        if hist_len <= 0:
            return

        eps = 1.0e-6
        device = weights.device
        weight_f = weights[..., :hist_len].float()
        hist_domains = hist_domain_ids.to(device=device).long()[:, :hist_len]
        known = known_domain_mask.to(device=device, dtype=torch.bool)[:, :hist_len]
        target = target_domain_ids.to(device=device).long().view(-1, 1)
        same_target = known & (hist_domains == target)

        prev_known = torch.zeros_like(known)
        prev_domain = hist_domains
        if hist_len > 1:
            prev_known[:, 1:] = known[:, :-1]
            prev_domain = torch.cat([hist_domains[:, :1], hist_domains[:, :-1]], dim=1)
        segment_start = known & ((~prev_known) | (hist_domains != prev_domain))
        segment_id = segment_start.long().cumsum(dim=1) - 1
        segment_id = torch.where(known, segment_id, torch.full_like(segment_id, -1))

        boundary_token = torch.zeros_like(known)
        if hist_len > 1:
            adjacent_known = known[:, :-1] & known[:, 1:]
            adjacent_cross = adjacent_known & (hist_domains[:, :-1] != hist_domains[:, 1:])
            boundary_token[:, :-1] = boundary_token[:, :-1] | adjacent_cross
            boundary_token[:, 1:] = boundary_token[:, 1:] | adjacent_cross

        entropy_low = max(float(self.m6_qtheme_segment_entropy_low), 0.0)
        entropy_high = min(max(float(self.m6_qtheme_segment_entropy_high), entropy_low), 1.0)
        min_len = max(int(self.m6_qtheme_segment_min_len), 1)
        log_q = math.log(float(num_slots))
        max_segments = int(segment_start.long().sum(dim=1).max().detach().cpu().item()) if bool(segment_start.any().item()) else 0
        if max_segments <= 0:
            return

        group_losses: list[torch.Tensor] = []
        group_entropy_values: list[torch.Tensor] = []
        group_counts: dict[str, int] = {"target": 0, "prev": 0, "boundary": 0}

        def add_group(seg_mask: torch.Tensor, sample_valid: torch.Tensor, group_name: str) -> None:
            if not bool(sample_valid.any().item()):
                return
            group_weight = weight_f * seg_mask.unsqueeze(1).to(dtype=weight_f.dtype)
            q_mass = group_weight.sum(dim=-1)
            total_mass = q_mass.sum(dim=1, keepdim=True)
            sample_valid_mass = sample_valid & (total_mass.view(-1) > eps)
            if not bool(sample_valid_mass.any().item()):
                return
            p = q_mass / total_mass.clamp_min(eps)
            entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1) / log_q
            band_loss = torch.relu(entropy_low - entropy).pow(2) + torch.relu(entropy - entropy_high).pow(2)
            sample_mask = sample_valid_mass.to(dtype=band_loss.dtype)
            group_losses.append((band_loss * sample_mask).sum() / sample_mask.sum().clamp_min(1.0))
            group_entropy_values.append((entropy.detach() * sample_mask).sum() / sample_mask.sum().clamp_min(1.0))
            group_counts[group_name] = int(group_counts[group_name]) + int(sample_valid_mass.long().sum().detach().cpu().item())

        for seg_idx in range(max_segments):
            seg_mask = (segment_id == int(seg_idx)) & known
            seg_len = seg_mask.long().sum(dim=1)
            base_valid = seg_len >= min_len
            if not bool(base_valid.any().item()):
                continue
            same_count = (seg_mask & same_target).long().sum(dim=1)
            target_seg = base_valid & (same_count == seg_len)
            next_target_seg = ((segment_id == int(seg_idx) + 1) & same_target).any(dim=1)
            prev_seg = base_valid & next_target_seg & (~target_seg)
            boundary_seg = base_valid & (seg_mask & boundary_token).any(dim=1)

            if bool(self.m6_qtheme_segment_include_target):
                add_group(seg_mask, target_seg, "target")
            if bool(self.m6_qtheme_segment_include_prev):
                add_group(seg_mask, prev_seg, "prev")
            if bool(self.m6_qtheme_segment_include_boundary):
                add_group(seg_mask, boundary_seg, "boundary")

        if not group_losses:
            return
        segment_loss = torch.stack(group_losses).mean()
        entropy_mean = torch.stack(group_entropy_values).mean()
        self._last_m6_qtheme_segment_aux_terms.append(segment_loss)

    def _m6_record_querymix_identity_aux(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        layer_idx: int,
    ) -> None:
        if not self._m6_querymix_identity_aux_enabled(layer_idx):
            return
        if before.ndim != 3 or after.ndim != 3:
            raise ValueError(
                "Expected M6 querymix identity tensors [B, Q, D], "
                f"got before={tuple(before.shape)} after={tuple(after.shape)}"
            )
        if before.shape != after.shape:
            raise ValueError(
                "M6 querymix identity tensors must share shape, "
                f"got before={tuple(before.shape)} after={tuple(after.shape)}"
            )
        batch_size, num_slots = int(before.shape[0]), int(before.shape[1])
        if batch_size <= 0 or num_slots <= 1:
            return

        eps = 1.0e-6
        before_ref = before.detach() if self.m6_querymix_identity_aux_detach_before else before
        before_norm = F.normalize(before_ref.float(), dim=-1, eps=eps)
        after_norm = F.normalize(after.float(), dim=-1, eps=eps)
        sim = torch.matmul(after_norm, before_norm.transpose(1, 2))
        pos = sim.diagonal(dim1=1, dim2=2)
        eye = torch.eye(num_slots, device=sim.device, dtype=torch.bool).unsqueeze(0)
        neg = sim.masked_fill(eye, -1.0e4).max(dim=-1).values
        margin = float(self.m6_querymix_identity_aux_margin)
        loss = F.softplus(margin + neg - pos).mean()
        gap = pos - neg
        self._last_m6_querymix_identity_aux_terms.append(loss)

    def _m6_record_branch_survival_aux(self, ns_tokens: torch.Tensor) -> None:
        self._last_m6_q34_survival_logits = None
        self._last_m6_q4_survival_logits = None
        if not self.training:
            return
        if ns_tokens.ndim != 3:
            raise ValueError(f"Expected M6 final Q slots [B, Q, D], got {tuple(ns_tokens.shape)}")
        num_slots = int(ns_tokens.shape[1])
        if num_slots <= 0:
            return

        if (
            float(self.m6_q34_survival_aux_weight) > 0.0
            and self.m6_q34_survival_head is not None
            and num_slots >= 3
        ):
            q34_start = 2
            q34_end = min(4, num_slots)
            q34_feature = ns_tokens[:, q34_start:q34_end, :].mean(dim=1)
            q34_logits = self.m6_q34_survival_head(q34_feature).view(-1)
            self._last_m6_q34_survival_logits = q34_logits

        if (
            float(self.m6_q4_survival_aux_weight) > 0.0
            and self.m6_q4_survival_head is not None
            and num_slots >= 4
        ):
            q4_feature = ns_tokens[:, 3, :]
            q4_logits = self.m6_q4_survival_head(q4_feature).view(-1)
            self._last_m6_q4_survival_logits = q4_logits

    def dvc_auxiliary_loss(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        parts: list[torch.Tensor] = []
        stats: dict[str, float] = {}
        if float(self.m6_qnorm_aux_weight) > 0.0 and self._last_m6_qnorm_aux_terms:
            qnorm_terms = self._last_m6_qnorm_aux_terms
            qnorm_loss = torch.stack(qnorm_terms).mean()
            qnorm_raw = float(qnorm_loss.detach().cpu().item())
            stats["m6_qnorm_aux_raw"] = qnorm_raw
            stats["m6_qnorm_aux_weighted"] = float(self.m6_qnorm_aux_weight) * qnorm_raw
            parts.append(float(self.m6_qnorm_aux_weight) * qnorm_loss)
        if float(self.m6_q4_anchor_aux_weight) > 0.0 and self._last_m6_q4_anchor_aux_terms:
            q4_anchor_terms = self._last_m6_q4_anchor_aux_terms
            q4_anchor_loss = torch.stack(q4_anchor_terms).mean()
            q4_anchor_raw = float(q4_anchor_loss.detach().cpu().item())
            stats["m6_q4_anchor_aux_raw"] = q4_anchor_raw
            stats["m6_q4_anchor_aux_weighted"] = float(self.m6_q4_anchor_aux_weight) * q4_anchor_raw
            parts.append(float(self.m6_q4_anchor_aux_weight) * q4_anchor_loss)
        if float(self.m6_qslot_cohesion_aux_weight) > 0.0 and self._last_m6_qslot_cohesion_aux_terms:
            qslot_cohesion_loss = torch.stack(self._last_m6_qslot_cohesion_aux_terms).mean()
            qslot_cohesion_raw = float(qslot_cohesion_loss.detach().cpu().item())
            stats["m6_qslot_cohesion_aux_raw"] = qslot_cohesion_raw
            stats["m6_qslot_cohesion_aux_weighted"] = float(self.m6_qslot_cohesion_aux_weight) * qslot_cohesion_raw
            parts.append(float(self.m6_qslot_cohesion_aux_weight) * qslot_cohesion_loss)
        if float(self.m6_qslot_div_aux_weight) > 0.0 and self._last_m6_qslot_div_aux_terms:
            qslot_div_loss = torch.stack(self._last_m6_qslot_div_aux_terms).mean()
            qslot_div_raw = float(qslot_div_loss.detach().cpu().item())
            stats["m6_qslot_div_aux_raw"] = qslot_div_raw
            stats["m6_qslot_div_aux_weighted"] = float(self.m6_qslot_div_aux_weight) * qslot_div_raw
            parts.append(float(self.m6_qslot_div_aux_weight) * qslot_div_loss)
        if float(self.m6_q4_complement_aux_weight) > 0.0 and self._last_m6_q4_complement_aux_terms:
            q4_complement_loss = torch.stack(self._last_m6_q4_complement_aux_terms).mean()
            q4_complement_raw = float(q4_complement_loss.detach().cpu().item())
            stats["m6_q4_complement_aux_raw"] = q4_complement_raw
            stats["m6_q4_complement_aux_weighted"] = float(self.m6_q4_complement_aux_weight) * q4_complement_raw
            parts.append(float(self.m6_q4_complement_aux_weight) * q4_complement_loss)
        if float(self.m6_querymix_identity_aux_weight) > 0.0 and self._last_m6_querymix_identity_aux_terms:
            qmix_identity_loss = torch.stack(self._last_m6_querymix_identity_aux_terms).mean()
            qmix_identity_raw = float(qmix_identity_loss.detach().cpu().item())
            stats["m6_querymix_identity_aux_raw"] = qmix_identity_raw
            stats["m6_querymix_identity_aux_weighted"] = (
                float(self.m6_querymix_identity_aux_weight) * qmix_identity_raw
            )
            parts.append(float(self.m6_querymix_identity_aux_weight) * qmix_identity_loss)
        if float(self.m6_qtheme_window_aux_weight) > 0.0 and self._last_m6_qtheme_window_aux_terms:
            qtheme_window_loss = torch.stack(self._last_m6_qtheme_window_aux_terms).mean()
            qtheme_window_raw = float(qtheme_window_loss.detach().cpu().item())
            stats["m6_qtheme_window_aux_raw"] = qtheme_window_raw
            stats["m6_qtheme_window_aux_weighted"] = float(self.m6_qtheme_window_aux_weight) * qtheme_window_raw
            parts.append(float(self.m6_qtheme_window_aux_weight) * qtheme_window_loss)
        if float(self.m6_qtheme_segment_aux_weight) > 0.0 and self._last_m6_qtheme_segment_aux_terms:
            qtheme_segment_loss = torch.stack(self._last_m6_qtheme_segment_aux_terms).mean()
            qtheme_segment_raw = float(qtheme_segment_loss.detach().cpu().item())
            stats["m6_qtheme_segment_aux_raw"] = qtheme_segment_raw
            stats["m6_qtheme_segment_aux_weighted"] = (
                float(self.m6_qtheme_segment_aux_weight) * qtheme_segment_raw
            )
            parts.append(float(self.m6_qtheme_segment_aux_weight) * qtheme_segment_loss)
        labels_f = labels.float().view(-1)
        if float(self.m6_q34_survival_aux_weight) > 0.0 and self._last_m6_q34_survival_logits is not None:
            q34_logits = self._last_m6_q34_survival_logits.view(-1)
            q34_loss = F.binary_cross_entropy_with_logits(q34_logits, labels_f)
            q34_raw = float(q34_loss.detach().cpu().item())
            stats["m6_q34_survival_aux_raw"] = q34_raw
            stats["m6_q34_survival_aux_weighted"] = float(self.m6_q34_survival_aux_weight) * q34_raw
            parts.append(float(self.m6_q34_survival_aux_weight) * q34_loss)
        if float(self.m6_q4_survival_aux_weight) > 0.0 and self._last_m6_q4_survival_logits is not None:
            q4_logits = self._last_m6_q4_survival_logits.view(-1)
            q4_loss = F.binary_cross_entropy_with_logits(q4_logits, labels_f)
            q4_raw = float(q4_loss.detach().cpu().item())
            stats["m6_q4_survival_aux_raw"] = q4_raw
            stats["m6_q4_survival_aux_weighted"] = float(self.m6_q4_survival_aux_weight) * q4_raw
            parts.append(float(self.m6_q4_survival_aux_weight) * q4_loss)
        self._last_dvc_aux_loss_stats = stats
        self._last_m6_qnorm_aux_terms = []
        self._last_m6_q4_anchor_aux_terms = []
        self._last_m6_qslot_cohesion_aux_terms = []
        self._last_m6_qslot_div_aux_terms = []
        self._last_m6_q4_complement_aux_terms = []
        self._last_m6_querymix_identity_aux_terms = []
        self._last_m6_qtheme_window_aux_terms = []
        self._last_m6_qtheme_segment_aux_terms = []
        self._last_m6_q34_survival_logits = None
        self._last_m6_q4_survival_logits = None
        if not parts:
            return None
        return torch.stack(parts).sum()

    def _m6_domain_q_gate(self, interaction, ns_tokens: torch.Tensor) -> torch.Tensor:
        if self.m6_final_q_gate_logits is None:
            raise RuntimeError("M6 final Q gate is not enabled.")
        batch_size, num_slots = int(ns_tokens.shape[0]), int(ns_tokens.shape[1])
        domain_ids = interaction[self.DOMAIN].long().view(-1) - int(self.domain_shift)
        domain_ids = domain_ids.clamp(min=0, max=int(self.num_domains) - 1)
        logits = self.m6_final_q_gate_logits(domain_ids).to(device=ns_tokens.device, dtype=ns_tokens.dtype)
        if logits.shape != (batch_size, num_slots):
            raise ValueError(f"M6 final Q gate logits must have shape {(batch_size, num_slots)}, got {tuple(logits.shape)}")
        soft_gate = torch.softmax(logits, dim=-1)
        floor = min(max(float(self.m6_final_q_gate_floor), 0.0), 1.0)
        gate = (1.0 - floor) * soft_gate + floor / float(max(num_slots, 1))
        entropy = -(gate.clamp_min(1.0e-12) * gate.clamp_min(1.0e-12).log()).sum(dim=-1)
        return gate

    def _final_slot_weights(self, interaction, ns_tokens: torch.Tensor) -> torch.Tensor:
        if self.m6_final_q_gate_logits is None or self.FINAL_SLOT_WEIGHT_FIELD in interaction:
            return super()._final_slot_weights(interaction, ns_tokens)
        batch_size, num_slots = int(ns_tokens.shape[0]), int(ns_tokens.shape[1])
        gate = self._m6_domain_q_gate(interaction, ns_tokens)
        code = self._final_slot_ablation_code(interaction)
        keep_indices = self._final_slot_keep_indices(code, num_slots)
        keep_mask = torch.zeros((batch_size, num_slots), device=ns_tokens.device, dtype=ns_tokens.dtype)
        if keep_indices:
            keep_mask[:, keep_indices] = 1.0
            weights = gate * keep_mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        else:
            weights = keep_mask
        code_values = torch.full(
            (batch_size,),
            float(code),
            device=ns_tokens.device,
            dtype=ns_tokens.dtype,
        )
        return weights

    def _m6_mode_code(self) -> int:
        return int(self.M6_MODE_CODES[str(self.m6_domainid_mode)])


    def _m6_shift_sequence(
        self,
        *,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        layer_idx: int,
        shift_mode: str = "same",
    ) -> torch.Tensor:
        shift_mode = str(shift_mode or "same").lower()
        configured_mode = str(getattr(self, "m6_event_shift_mode", "same"))
        if shift_mode != "same" or configured_mode != "same":
            raise ValueError(
                "M6 event shift cannot cross domains; "
                f"expected shift_mode='same', got call={shift_mode!r}, "
                f"configured={configured_mode!r}."
            )
        batch_size, seq_len, _ = seq_tokens.shape
        begin = self.m6_shift_begin[int(layer_idx)].to(dtype=seq_tokens.dtype, device=seq_tokens.device).view(1, 1, -1)
        end = self.m6_shift_end[int(layer_idx)].to(dtype=seq_tokens.dtype, device=seq_tokens.device).view(1, 1, -1)
        begin = begin.expand(batch_size, 1, -1)
        end = end.expand(batch_size, 1, -1)
        padded_tokens = torch.cat([begin, seq_tokens, end], dim=1)
        prev_tokens = padded_tokens[:, :-2, :]
        next_tokens = padded_tokens[:, 2:, :]

        prev_valid = torch.zeros_like(hist_mask, dtype=torch.bool)
        next_valid = torch.zeros_like(hist_mask, dtype=torch.bool)
        prev_cross = torch.zeros_like(hist_mask, dtype=torch.bool)
        next_cross = torch.zeros_like(hist_mask, dtype=torch.bool)
        prev_same = torch.zeros_like(hist_mask, dtype=torch.bool)
        next_same = torch.zeros_like(hist_mask, dtype=torch.bool)
        if seq_len > 1:
            adjacent_known = known_domain_mask[:, :-1] & known_domain_mask[:, 1:]
            adjacent_cross = adjacent_known & (hist_domain_ids[:, :-1] != hist_domain_ids[:, 1:])
            adjacent_same = adjacent_known & (hist_domain_ids[:, :-1] == hist_domain_ids[:, 1:])
            prev_valid[:, 1:] = hist_mask[:, :-1] & hist_mask[:, 1:]
            next_valid[:, :-1] = hist_mask[:, :-1] & hist_mask[:, 1:]
            prev_cross[:, 1:] = adjacent_cross
            next_cross[:, :-1] = adjacent_cross
            prev_same[:, 1:] = adjacent_same
            next_same[:, :-1] = adjacent_same

        prev_use = prev_valid & prev_same
        next_use = next_valid & next_same
        curr_chunks = torch.chunk(seq_tokens, 4, dim=-1)
        prev_chunks = torch.chunk(prev_tokens, 4, dim=-1)
        next_chunks = torch.chunk(next_tokens, 4, dim=-1)
        prev_mid = torch.where(
            prev_use.unsqueeze(-1),
            prev_chunks[2],
            torch.where(prev_valid.unsqueeze(-1), curr_chunks[1], prev_chunks[2]),
        )
        next_mid = torch.where(
            next_use.unsqueeze(-1),
            next_chunks[1],
            torch.where(next_valid.unsqueeze(-1), curr_chunks[2], next_chunks[1]),
        )
        shifted = torch.cat([curr_chunks[0], prev_mid, next_mid, curr_chunks[3]], dim=-1)
        shifted = shifted * hist_mask.unsqueeze(-1).to(dtype=shifted.dtype)
        return shifted

    def _m6_domain_delta(
        self,
        *,
        seq_tokens: torch.Tensor,
        known_domain_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        layer_idx: int,
        stage: str,
    ) -> torch.Tensor:
        domain_ids = hist_domain_ids.long().clamp(min=0, max=max(0, int(self.num_domains) - 1))
        delta = self.m6_domain_embedding(domain_ids).to(dtype=seq_tokens.dtype)
        delta = delta * known_domain_mask.unsqueeze(-1).to(dtype=delta.dtype)
        return delta

    def _m6_domain_mlp_delta(
        self,
        *,
        seq_tokens: torch.Tensor,
        known_domain_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        layer_idx: int,
        stage: str,
    ) -> torch.Tensor:
        domain_ids = hist_domain_ids.long().clamp(min=0, max=max(0, int(self.num_domains) - 1))
        domain_emb = self.m6_domain_embedding(domain_ids).to(dtype=seq_tokens.dtype)
        domain_emb = domain_emb * known_domain_mask.unsqueeze(-1).to(dtype=domain_emb.dtype)
        mlp_input = torch.cat([seq_tokens, domain_emb], dim=-1)
        delta = self.m6_domain_inject_mlp(mlp_input)
        delta = delta * known_domain_mask.unsqueeze(-1).to(dtype=delta.dtype)
        return delta

    def _m6_project_delta_obs(
        self,
        *,
        cross_attention: CrossAttention,
        base_tokens: torch.Tensor,
        kv_source: torch.Tensor,
        hist_mask: torch.Tensor,
        layer_idx: int,
    ) -> None:
        base_tokens = base_tokens * hist_mask.unsqueeze(-1).to(dtype=base_tokens.dtype)
        kv_source = kv_source * hist_mask.unsqueeze(-1).to(dtype=kv_source.dtype)
        base_k = torch.stack(cross_attention.k_proj(base_tokens).chunk(cross_attention.num_tokens, dim=-1), dim=1)
        base_v = torch.stack(cross_attention.v_proj(base_tokens).chunk(cross_attention.num_tokens, dim=-1), dim=1)
        kv_k = torch.stack(cross_attention.k_proj(kv_source).chunk(cross_attention.num_tokens, dim=-1), dim=1)
        kv_v = torch.stack(cross_attention.v_proj(kv_source).chunk(cross_attention.num_tokens, dim=-1), dim=1)
        key_delta_norm = (kv_k - base_k).norm(dim=-1).mean(dim=1)
        value_delta_norm = (kv_v - base_v).norm(dim=-1).mean(dim=1)

    @staticmethod
    def _m6_masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(masked.float(), dim=-1).to(dtype=logits.dtype)
        weights = weights * mask.to(dtype=weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)

    def _m6_segment_layout(
        self,
        *,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        hist_ts_seconds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build a fixed-width sidecar: one SEG at each continuous run tail."""
        valid = hist_mask & known_domain_mask
        prev_valid = F.pad(valid[:, :-1], (1, 0), value=False)
        next_valid = F.pad(valid[:, 1:], (0, 1), value=False)
        prev_domain = F.pad(hist_domain_ids[:, :-1], (1, 0), value=0)
        next_domain = F.pad(hist_domain_ids[:, 1:], (0, 1), value=0)
        run_start = valid & (~prev_valid | hist_domain_ids.ne(prev_domain))
        run_end = valid & (~next_valid | hist_domain_ids.ne(next_domain))
        run_id = torch.cumsum(run_start.long(), dim=1)
        own_run = (
            run_end.unsqueeze(2)
            & valid.unsqueeze(1)
            & run_id.unsqueeze(2).eq(run_id.unsqueeze(1))
        )

        length = int(valid.shape[1])
        positions = torch.arange(length, device=valid.device).view(1, length)
        end_candidates = torch.where(run_end, positions, torch.full_like(positions, length))
        # Earliest run tail at or after each event is its owning SEG tail.
        tail_index = torch.flip(
            torch.cummin(torch.flip(end_candidates, dims=[1]), dim=1).values,
            dims=[1],
        ).clamp(max=max(length - 1, 0))
        tail_time_for_event = hist_ts_seconds.gather(1, tail_index)
        segment_tail_time = torch.where(run_end, hist_ts_seconds, torch.zeros_like(hist_ts_seconds))
        return {
            "segment_mask": run_end,
            "segment_event_mask": own_run,
            "tail_time_for_event": tail_time_for_event,
            "segment_tail_time": segment_tail_time,
        }

    def _m6_capped_segment_layout(
        self,
        *,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        hist_ts_seconds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compact the newest continuous runs into a fixed number of SEG slots.

        The event stream remains untouched.  Only the SEG sidecar drops runs:
        if a sample has more than ``m6_segment_slot_cap`` runs, its oldest
        (temporally farthest) runs receive no SEG slot.  Kept slots are ordered
        from oldest to newest and left-aligned, so their width is independent
        of the event sequence width.
        """
        valid = hist_mask & known_domain_mask
        previous_valid = F.pad(valid[:, :-1], (1, 0), value=False)
        next_valid = F.pad(valid[:, 1:], (0, 1), value=False)
        previous_domain = F.pad(hist_domain_ids[:, :-1], (1, 0), value=0)
        next_domain = F.pad(hist_domain_ids[:, 1:], (0, 1), value=0)
        run_start = valid & (~previous_valid | hist_domain_ids.ne(previous_domain))
        run_end = valid & (~next_valid | hist_domain_ids.ne(next_domain))
        run_id = torch.cumsum(run_start.long(), dim=1)
        run_count = run_id.max(dim=1).values

        slot_count = min(int(self.m6_segment_slot_cap), int(valid.shape[1]))
        first_kept_run = (run_count - slot_count).clamp_min(0)
        kept_event = valid & run_id.gt(first_kept_run.unsqueeze(1))
        event_slot = run_id - first_kept_run.unsqueeze(1) - 1
        slot_positions = torch.arange(slot_count, device=valid.device).view(1, slot_count)
        kept_count = run_count.clamp(max=slot_count)
        segment_mask = slot_positions < kept_count.unsqueeze(1)
        own_run = (
            segment_mask.unsqueeze(2)
            & kept_event.unsqueeze(1)
            & event_slot.unsqueeze(1).eq(slot_positions.unsqueeze(2))
        )

        length = int(valid.shape[1])
        positions = torch.arange(length, device=valid.device).view(1, length)
        end_candidates = torch.where(run_end, positions, torch.full_like(positions, length))
        tail_index = torch.flip(
            torch.cummin(torch.flip(end_candidates, dims=[1]), dim=1).values,
            dims=[1],
        ).clamp(max=max(length - 1, 0))
        tail_time_for_event = hist_ts_seconds.gather(1, tail_index)
        segment_tail_time = torch.where(
            own_run & run_end.unsqueeze(1),
            hist_ts_seconds.unsqueeze(1),
            torch.zeros_like(hist_ts_seconds).unsqueeze(1),
        ).sum(dim=2)
        return {
            "segment_mask": segment_mask,
            "segment_event_mask": own_run,
            "tail_time_for_event": tail_time_for_event,
            "segment_tail_time": segment_tail_time,
            "segment_count": run_count,
            "dropped_segment_count": (run_count - slot_count).clamp_min(0),
        }

    def _m6_tail_event_time_embedding(
        self,
        *,
        tail_time_for_event: torch.Tensor,
        hist_ts_seconds: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> torch.Tensor:
        delta = (tail_time_for_event - hist_ts_seconds).clamp_min(0)
        boundaries = torch.as_tensor(
            self.m6_segment_tail_time.boundaries,
            device=delta.device,
            dtype=delta.dtype,
        )
        bucket = torch.bucketize(delta, boundaries, right=False)
        bucket = bucket.clamp(max=boundaries.numel() - 1).long() + 1
        bucket = torch.where(hist_mask, bucket, torch.zeros_like(bucket))
        return self.m6_segment_tail_time.embedding(bucket)

    def _m6_segment_layer_update(
        self,
        *,
        layer_idx: int,
        event_tokens: torch.Tensor,
        segment_tokens: torch.Tensor,
        segment_mask: torch.Tensor,
        segment_event_mask: torch.Tensor,
        tail_event_time_embedding: torch.Tensor,
        candidate_segment_time_embedding: torch.Tensor,
        q4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(layer_idx)
        if int(segment_tokens.shape[1]) == 0:
            # cap=0 is an explicit fixed-weight ablation: preserve Q4's own
            # QueryMixer/OutputFusion residual stream, but remove every
            # Segment-to-Q4 attention update.
            self._last_m6_segment_observations.append({
                "segment_attention": event_tokens.new_empty(
                    int(event_tokens.shape[0]), 0, int(event_tokens.shape[1])
                ),
                "cross_attention": event_tokens.new_empty(int(event_tokens.shape[0]), 0),
                "segment_mask": segment_mask.detach(),
                "segment_event_mask": segment_event_mask.detach(),
                "attention_delta": segment_tokens.detach(),
                "ffn_delta": segment_tokens.detach(),
            })
            return segment_tokens, torch.zeros_like(q4)
        q = self.m6_segment_q_projs[idx](self.m6_segment_q_norms[idx](segment_tokens))
        event_view = torch.cat(
            [self.m6_segment_event_norms[idx](event_tokens), tail_event_time_embedding],
            dim=-1,
        )
        k = self.m6_segment_k_projs[idx](event_view)
        v = self.m6_segment_v_projs[idx](event_view)
        logits = torch.matmul(q, k.transpose(1, 2)) * (float(self.token_dim) ** -0.5)
        weights = self._m6_masked_softmax(logits, segment_event_mask)
        context = torch.matmul(weights, v)
        mask_f = segment_mask.unsqueeze(-1).to(dtype=event_tokens.dtype)
        attention_delta = self.m6_segment_out_projs[idx](context) * mask_f
        segment_tokens = segment_tokens + attention_delta
        ffn_delta = self.m6_segment_ffns[idx](
            self.m6_segment_ffn_norms[idx](segment_tokens)
        ) * mask_f
        segment_tokens = (segment_tokens + ffn_delta) * mask_f

        segment_view = torch.cat(
            [self.m6_cross_segment_norms[idx](segment_tokens), candidate_segment_time_embedding],
            dim=-1,
        )
        cross_k = self.m6_cross_segment_k_projs[idx](segment_view)
        cross_v = self.m6_cross_segment_v_projs[idx](segment_view)
        cross_logits = torch.einsum("bd,bld->bl", q4, cross_k) * (float(self.token_dim) ** -0.5)
        cross_weights = self._m6_masked_softmax(cross_logits, segment_mask)
        q4_update = torch.einsum("bl,bld->bd", cross_weights, cross_v)
        self._last_m6_segment_observations.append({
            "segment_attention": weights.detach(),
            "cross_attention": cross_weights.detach(),
            "segment_mask": segment_mask.detach(),
            "segment_event_mask": segment_event_mask.detach(),
            "attention_delta": attention_delta.detach(),
            "ffn_delta": ffn_delta.detach(),
        })
        return segment_tokens, q4_update

    def _m6_event_ffn_residual(
        self,
        *,
        cross_attention: CrossAttention,
        shifted: torch.Tensor,
        hist_mask: torch.Tensor,
        layer_idx: int,
        segment_tokens: Optional[torch.Tensor] = None,
        segment_layout: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply the ordinary post-Shift Event FFN residual.

        Experimental subclasses may condition this update on the Episode
        state that entered the current layer.  The default expression is kept
        deliberately identical to the historical implementation.
        """

        output = cross_attention.seq_ffn(cross_attention.seq_norm(shifted)) + shifted
        return output * hist_mask.unsqueeze(-1).to(dtype=output.dtype)

    def _m6_event_kv_inputs(
        self,
        *,
        kv_source: torch.Tensor,
        hist_mask: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current layer's Event Key/Value input tensors.

        The base implementation is deliberately an identity hook.  Controlled
        experimental subclasses may construct an ephemeral read-view here,
        while ``kv_source`` remains the persistent Event state returned to the
        next layer.
        """

        return kv_source, kv_source

    def _m6_project_event_kv(
        self,
        *,
        cross_attention: CrossAttention,
        key_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project Event K/V through the historical, user-independent path."""

        return cross_attention.k_proj(key_tokens), cross_attention.v_proj(value_tokens)

    def _m6_cross_attention_layer(
        self,
        *,
        cross_attention: CrossAttention,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
        layer_idx: int,
        attention_same_mask: Optional[torch.Tensor] = None,
        attention_cross_mask: Optional[torch.Tensor] = None,
        attention_output_gate: Optional[torch.Tensor] = None,
        segment_tokens: Optional[torch.Tensor] = None,
        segment_layout: Optional[dict[str, torch.Tensor]] = None,
        tail_event_time_embedding: Optional[torch.Tensor] = None,
        candidate_segment_time_embedding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mode = str(self.m6_domainid_mode)
        ctxbank_runtime = None
        shifted = seq_tokens
        base_reference_for_kv = None
        domain_delta = torch.zeros_like(seq_tokens)

        if mode == "postffn_domainid_kv":
            seq_after_ffn = cross_attention.seq_ffn(cross_attention.seq_norm(seq_tokens)) + seq_tokens
            seq_after_ffn = seq_after_ffn * hist_mask.unsqueeze(-1).to(dtype=seq_after_ffn.dtype)
            next_seq_tokens = seq_after_ffn
            base_reference_for_kv = seq_after_ffn
            domain_delta = self._m6_domain_delta(
                seq_tokens=seq_after_ffn,
                known_domain_mask=known_domain_mask,
                hist_domain_ids=hist_domain_ids,
                layer_idx=layer_idx,
                stage="postffn",
            )
            kv_source = (seq_after_ffn + domain_delta) * hist_mask.unsqueeze(-1).to(dtype=seq_after_ffn.dtype)
        elif mode == "shift_pre_domainid_kv":
            shifted = self._m6_shift_sequence(
                seq_tokens=seq_tokens,
                hist_mask=hist_mask,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
            seq_after_ffn = cross_attention.seq_ffn(cross_attention.seq_norm(shifted)) + shifted
            seq_after_ffn = seq_after_ffn * hist_mask.unsqueeze(-1).to(dtype=seq_after_ffn.dtype)
            next_seq_tokens = seq_after_ffn
            base_reference_for_kv = seq_after_ffn
            domain_delta = self._m6_domain_delta(
                seq_tokens=seq_after_ffn,
                known_domain_mask=known_domain_mask,
                hist_domain_ids=hist_domain_ids,
                layer_idx=layer_idx,
                stage="post_shift_ffn",
            )
            kv_source = (seq_after_ffn + domain_delta) * hist_mask.unsqueeze(-1).to(dtype=seq_after_ffn.dtype)
        elif mode == "domainid_pre_shift_kv":
            domain_delta = self._m6_domain_delta(
                seq_tokens=seq_tokens,
                known_domain_mask=known_domain_mask,
                hist_domain_ids=hist_domain_ids,
                layer_idx=layer_idx,
                stage="pre_shift",
            )
            pre_shift = (seq_tokens + domain_delta) * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
            shifted = self._m6_shift_sequence(
                seq_tokens=pre_shift,
                hist_mask=hist_mask,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
            seq_after_ffn = cross_attention.seq_ffn(cross_attention.seq_norm(shifted)) + shifted
            seq_after_ffn = seq_after_ffn * hist_mask.unsqueeze(-1).to(dtype=seq_after_ffn.dtype)
            next_seq_tokens = seq_after_ffn
            kv_source = seq_after_ffn
            base_reference_for_kv = kv_source
        elif mode in {
            "shift_pre_no_domainid_kv",
            "shift_pre_no_domainid_kv_q4off",
            "shift_pre_no_domainid_kv_segq4",
            "shift_pre_no_domainid_kv_segq4_cap",
        }:
            shifted = self._m6_shift_sequence(
                seq_tokens=seq_tokens,
                hist_mask=hist_mask,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
            seq_after_ffn = self._m6_event_ffn_residual(
                cross_attention=cross_attention,
                shifted=shifted,
                hist_mask=hist_mask,
                layer_idx=layer_idx,
                segment_tokens=segment_tokens,
                segment_layout=segment_layout,
            )
            next_seq_tokens = seq_after_ffn
            base_reference_for_kv = seq_after_ffn
            kv_source = seq_after_ffn
        elif mode == "layered_l0_all_l1_domainid_all_l2_same_kv":
            if int(layer_idx) == 1:
                domain_delta = self._m6_domain_mlp_delta(
                    seq_tokens=seq_tokens,
                    known_domain_mask=known_domain_mask,
                    hist_domain_ids=hist_domain_ids,
                    layer_idx=layer_idx,
                    stage="layered_mlp_pre_shift",
                )
                shift_input = (seq_tokens + domain_delta) * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
                shift_mode = "same"
            else:
                shift_input = seq_tokens
                shift_mode = "same"
            shifted = self._m6_shift_sequence(
                seq_tokens=shift_input,
                hist_mask=hist_mask,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
                shift_mode=shift_mode,
            )
            seq_after_ffn = cross_attention.seq_ffn(cross_attention.seq_norm(shifted)) + shifted
            seq_after_ffn = seq_after_ffn * hist_mask.unsqueeze(-1).to(dtype=seq_after_ffn.dtype)
            next_seq_tokens = seq_after_ffn
            kv_source = seq_after_ffn
            base_reference_for_kv = kv_source
        else:
            raise RuntimeError(f"Unsupported M6 mode: {mode}")

        next_norm = next_seq_tokens.norm(dim=-1).clamp_min(1e-12)
        kv_norm = kv_source.norm(dim=-1).clamp_min(1e-12)
        next_kv_cos = (next_seq_tokens * kv_source).sum(dim=-1) / (next_norm * kv_norm)

        key_tokens, value_tokens = self._m6_event_kv_inputs(
            kv_source=kv_source,
            hist_mask=hist_mask,
            layer_idx=layer_idx,
        )
        if cross_attention.key_modulator is not None:
            raise ValueError("M6DomainIdMixFormerModel does not support key_modulator in the first-batch baseline.")
        projected_keys, projected_values = self._m6_project_event_kv(
            cross_attention=cross_attention,
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            hist_mask=hist_mask,
            layer_idx=layer_idx,
        )
        keys = projected_keys.chunk(cross_attention.num_tokens, dim=-1)
        values = projected_values.chunk(cross_attention.num_tokens, dim=-1)
        k = torch.stack(keys, dim=1)
        v = torch.stack(values, dim=1)

        query_tokens = ns_tokens
        if cross_attention.query_modulator is not None:
            raise ValueError("M6DomainIdMixFormerModel does not support query_modulator in the first-batch baseline.")
        q = query_tokens.unsqueeze(2)

        bias = None
        if attention_bias is not None:
            bias = attention_bias.to(device=q.device, dtype=q.dtype)
            if bias.dim() == 2:
                bias = bias.unsqueeze(1).unsqueeze(2)
            elif bias.dim() == 3:
                bias = bias.unsqueeze(2)

        use_double_softmax = (
            self.attention_double_softmax_beta is not None
            and attention_same_mask is not None
            and attention_cross_mask is not None
        )
        if use_double_softmax:
            logits = torch.matmul(q, k.transpose(-2, -1)) * (float(cross_attention.input_dim) ** -0.5)
            if bias is not None:
                logits = logits + bias
            self._m6_record_qnorm_aux(logits.squeeze(2), layer_idx)
            attention_output, weights, _, _ = _double_softmax_attention(
                logits=logits,
                values=v,
                same_mask=attention_same_mask,
                cross_mask=attention_cross_mask,
                beta=float(self.attention_double_softmax_beta),
                mode=self.attention_double_softmax_mode,
            )
            self._m6_record_q4_anchor_aux(
                weights.squeeze(2),
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                target_domain_ids=target_domain_ids,
                layer_idx=layer_idx,
            )
            self._m6_record_qslot_aux(
                weights.squeeze(2),
                kv_source,
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
            self._m6_record_q4_complement_aux(
                weights.squeeze(2),
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
            self._m6_record_qtheme_window_aux(
                weights.squeeze(2),
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
            self._m6_record_qtheme_segment_aux(
                weights.squeeze(2),
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                target_domain_ids=target_domain_ids,
                layer_idx=layer_idx,
            )
        elif self._m6_attention_aux_enabled(layer_idx) or bool(
            getattr(self, "_collect_m6_event_attention_obs", False)
        ):
            logits = torch.matmul(q, k.transpose(-2, -1)) * (float(cross_attention.input_dim) ** -0.5)
            if bias is not None:
                logits = logits + bias
            self._m6_record_qnorm_aux(logits.squeeze(2), layer_idx)
            weights = torch.softmax(logits, dim=-1)
            self._m6_record_q4_anchor_aux(
                weights.squeeze(2),
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                target_domain_ids=target_domain_ids,
                layer_idx=layer_idx,
            )
            self._m6_record_qslot_aux(
                weights.squeeze(2),
                kv_source,
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
            self._m6_record_q4_complement_aux(
                weights.squeeze(2),
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
            self._m6_record_qtheme_window_aux(
                weights.squeeze(2),
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
            self._m6_record_qtheme_segment_aux(
                weights.squeeze(2),
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                target_domain_ids=target_domain_ids,
                layer_idx=layer_idx,
            )
            attention_output = torch.matmul(weights, v).squeeze(2)
        elif bias is None:
            attention_output = F.scaled_dot_product_attention(q, k, v).squeeze(2)
        else:
            attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).squeeze(2)
        if bool(getattr(self, "_collect_m6_event_attention_obs", False)):
            # OBS is opt-in and eval-only in the caller.  Store detached
            # tensors for one bounded batch, including availability metadata;
            # Q4 is explicitly labelled as the event-path output because the
            # Segment sidecar replaces it immediately below.
            self._last_m6_event_attention_observations.append({
                "layer_idx": int(layer_idx),
                "attention": weights.squeeze(2).detach(),
                "event_output_pre_residual": attention_output.detach(),
                "query_tokens_pre_attention": ns_tokens.detach(),
                "hist_mask": hist_mask.detach(),
                "hist_domain_ids": hist_domain_ids.detach(),
                "known_domain_mask": known_domain_mask.detach(),
                "target_domain_ids": target_domain_ids.detach(),
            })
        if attention_output_gate is not None:
            gate = attention_output_gate.to(device=attention_output.device, dtype=attention_output.dtype)
            attention_output = attention_output * gate.unsqueeze(-1)
        else:
            gate = attention_output.new_ones(attention_output.shape[:2])
        if bool(getattr(self, "_collect_m6_event_attention_obs", False)):
            effective_attention = (
                self._last_m6_event_attention_observations[-1]["attention"]
                * gate.unsqueeze(-1)
            )
            self._last_m6_event_attention_observations[-1].update({
                "attention": effective_attention.detach(),
                "attention_output_gate": gate.detach(),
                "event_output_post_gate": attention_output.detach(),
            })
        if mode == "shift_pre_no_domainid_kv_q4off":
            attention_output = attention_output.clone()
            attention_output[:, 3] = 0
        elif self.m6_segment_sidecar_enabled:
            if (
                segment_tokens is None
                or segment_layout is None
                or tail_event_time_embedding is None
                or candidate_segment_time_embedding is None
            ):
                raise ValueError("SEG-Q4 mode requires initialized sidecar tensors and time embeddings.")
            segment_tokens, q4_update = self._m6_segment_layer_update(
                layer_idx=layer_idx,
                event_tokens=next_seq_tokens,
                segment_tokens=segment_tokens,
                segment_mask=segment_layout["segment_mask"],
                segment_event_mask=segment_layout["segment_event_mask"],
                tail_event_time_embedding=tail_event_time_embedding,
                candidate_segment_time_embedding=candidate_segment_time_embedding,
                q4=ns_tokens[:, 3],
            )
            attention_output = attention_output.clone()
            attention_output[:, 3] = q4_update
        ns_tokens = attention_output + ns_tokens
        return next_seq_tokens, ns_tokens, segment_tokens

    def _mixformer_with_m6_domainid(
        self,
        *,
        interaction,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        attention_output_gate: Optional[torch.Tensor] = None,
        querymix_eval_code: int = 0,
        curr_ts_seconds: Optional[torch.Tensor] = None,
        hist_ts_seconds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_same_mask, attention_cross_mask = self._attention_double_softmax_masks(
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=target_domain_ids,
        )
        segment_tokens = None
        segment_layout = None
        tail_event_time_embedding = None
        candidate_segment_time_embedding = None
        self._last_m6_segment_observations = []
        self._last_m6_event_attention_observations = []
        if self.m6_segment_sidecar_enabled:
            if curr_ts_seconds is None or hist_ts_seconds is None:
                raise ValueError("SEG-Q4 mode requires candidate and history timestamps in seconds.")
            layout_fn = (
                self._m6_capped_segment_layout
                if self.m6_segment_sidecar_capped
                else self._m6_segment_layout
            )
            segment_layout = layout_fn(
                hist_mask=hist_mask,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                hist_ts_seconds=hist_ts_seconds,
            )
            segment_mask = segment_layout["segment_mask"]
            segment_tokens = self.m6_segment_seed.view(1, 1, -1).expand(
                int(seq_tokens.shape[0]), int(segment_mask.shape[1]), -1,
            ) * segment_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
            tail_event_time_embedding = self._m6_tail_event_time_embedding(
                tail_time_for_event=segment_layout["tail_time_for_event"],
                hist_ts_seconds=hist_ts_seconds,
                hist_mask=hist_mask,
            ).to(dtype=seq_tokens.dtype)
            candidate_segment_time_embedding = self.m6_segment_candidate_time(
                curr_ts_seconds,
                segment_layout["segment_tail_time"],
                padding_mask=~segment_mask,
            ).to(dtype=seq_tokens.dtype)
        for layer_idx, (query_mixer, cross_attention, output_fusion) in enumerate(zip(
            self.mixformer.query_mixers,
            self.mixformer.cross_attentions,
            self.mixformer.output_fusions,
        )):
            before_querymix = ns_tokens
            ns_tokens = self.mixformer._run_query_mixer_step(
                query_mixer,
                ns_tokens,
                querymix_eval_code=querymix_eval_code,
            )
            self._m6_record_querymix_identity_aux(before_querymix, ns_tokens, layer_idx=layer_idx)
            slot_category_bias = self._attention_slot_category_off_bias(
                interaction,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                target_domain_ids=target_domain_ids,
                layer_idx=layer_idx,
                num_tokens=self.num_ns_token,
                dtype=seq_tokens.dtype,
            )
            layer_attention_bias = self._merge_attention_bias(attention_bias, slot_category_bias)
            slot_output_gate = self._attention_slot_output_off_gate(
                interaction,
                batch_size=int(ns_tokens.shape[0]),
                device=ns_tokens.device,
                dtype=ns_tokens.dtype,
                layer_idx=layer_idx,
                num_tokens=self.num_ns_token,
            )
            layer_attention_output_gate = attention_output_gate
            if slot_output_gate is not None:
                layer_attention_output_gate = (
                    slot_output_gate
                    if layer_attention_output_gate is None
                    else layer_attention_output_gate.to(device=ns_tokens.device, dtype=ns_tokens.dtype) * slot_output_gate
                )
            seq_tokens, ns_tokens, segment_tokens = self._m6_cross_attention_layer(
                cross_attention=cross_attention,
                seq_tokens=seq_tokens,
                ns_tokens=ns_tokens,
                hist_mask=hist_mask,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                target_domain_ids=target_domain_ids,
                attention_bias=layer_attention_bias,
                layer_idx=layer_idx,
                attention_same_mask=attention_same_mask,
                attention_cross_mask=attention_cross_mask,
                attention_output_gate=layer_attention_output_gate,
                segment_tokens=segment_tokens,
                segment_layout=segment_layout,
                tail_event_time_embedding=tail_event_time_embedding,
                candidate_segment_time_embedding=candidate_segment_time_embedding,
            )
            ns_tokens = output_fusion(ns_tokens)
        return ns_tokens

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        self._last_m6_q34_survival_logits = None
        self._last_m6_q4_survival_logits = None
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)

        attention_bias = self._attention_domain_bias(
            interaction,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            dtype=seq_emb.dtype,
        )
        attention_slot_bias, attention_output_gate = self._attention_slot_bias_and_output_gate(
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            dtype=seq_emb.dtype,
            num_tokens=self.num_ns_token,
        )
        attention_bias = self._merge_attention_bias(attention_bias, attention_slot_bias)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts_seconds = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            relative_time_emb = self.relative_time_embedding(
                curr_ts,
                hist_ts_seconds,
                seq_lens=seq_len,
            )
            seq_emb = self._fuse_sequence_relative_time(seq_emb, relative_time_emb)
        if (self.absolute_time_embedding is not None or self.hstu_relative_attention is not None) and hist_ts is None:
            hist_ts = interaction[self.history_view.time_field]
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        seq_tokens = self._add_absolute_time_embedding(seq_tokens, mask, hist_ts)
        seq_tokens = self._add_absolute_position_embedding(seq_tokens, mask)
        seq_tokens = self._add_segment_relative_position_embedding(
            seq_tokens,
            mask,
            hist_domain_ids,
            known_domain_mask,
            domain_ids,
        )
        seq_tokens = self._add_boundary_position_embedding(
            seq_tokens,
            mask,
            hist_domain_ids,
            known_domain_mask,
            domain_ids,
        )
        seq_tokens = self._apply_hstu_relative_attention(seq_tokens, mask, hist_ts)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        querymix_eval_code = self._querymix_eval_code(interaction)
        ns_tokens = self._mixformer_with_m6_domainid(
            interaction=interaction,
            seq_tokens=seq_tokens,
            ns_tokens=ns_tokens,
            hist_mask=mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            attention_bias=attention_bias,
            attention_output_gate=attention_output_gate,
            querymix_eval_code=querymix_eval_code,
            curr_ts_seconds=(
                _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
                if self.m6_segment_sidecar_enabled else None
            ),
            hist_ts_seconds=(
                _to_seconds(interaction[self.history_view.time_field], self.relative_time_to_seconds_divisor)
                if self.m6_segment_sidecar_enabled else None
            ),
        )
        self._m6_record_branch_survival_aux(ns_tokens)
        features = self.dropout(self._pool_final_slots(interaction, ns_tokens))
        features = self._apply_cross_context_residual(
            features=features,
            seq_tokens=seq_tokens,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
        )
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class DomainSplitReadoutMixFormerModel(MixFormerModel):
    """MixFormer with two normal readout slots and two domain-restricted readout slots."""

    def __init__(
        self,
        *args,
        domain_split_order: str = "cross_then_same",
        domain_split_penalty: float = -1.0e4,
        domain_split_zero_empty: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if int(self.num_ns_token) < 2:
            raise ValueError("DomainSplitReadoutMixFormerModel requires at least two query tokens.")
        order = str(domain_split_order or "cross_then_same").lower()
        aliases = {
            "cross_then_same": "cross_then_same",
            "cross_same": "cross_then_same",
            "cs": "cross_then_same",
            "same_then_cross": "same_then_cross",
            "same_cross": "same_then_cross",
            "sc": "same_then_cross",
        }
        if order not in aliases:
            raise ValueError(
                "domain_split_order must be one of "
                f"{sorted(aliases)}, got {domain_split_order!r}."
            )
        self.domain_split_order = aliases[order]
        self.domain_split_penalty = float(domain_split_penalty)
        self.domain_split_zero_empty = bool(domain_split_zero_empty)

    def _domain_split_slot_indices(self, num_tokens: int) -> tuple[int, int]:
        first = max(0, int(num_tokens) - 2)
        second = max(0, int(num_tokens) - 1)
        if self.domain_split_order == "same_then_cross":
            return second, first
        return first, second

    def _domain_split_order_code(self) -> int:
        return 1 if self.domain_split_order == "same_then_cross" else 0

    def _attention_slot_bias_and_output_gate(
        self,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        dtype: torch.dtype,
        num_tokens: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_tokens = int(num_tokens)
        if num_tokens < 2:
            return None, None
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_mask = known_domain_mask & (hist_domain_ids == target_domain)
        cross_mask = known_domain_mask & (hist_domain_ids != target_domain)
        cross_slot, same_slot = self._domain_split_slot_indices(num_tokens)
        batch_size, seq_len = hist_domain_ids.shape
        penalty = torch.full(
            (batch_size, num_tokens, seq_len),
            0.0,
            device=hist_domain_ids.device,
            dtype=dtype,
        )
        penalty[:, cross_slot, :] = torch.where(
            cross_mask,
            torch.zeros((), device=hist_domain_ids.device, dtype=dtype),
            torch.full((), float(self.domain_split_penalty), device=hist_domain_ids.device, dtype=dtype),
        )
        penalty[:, same_slot, :] = torch.where(
            same_mask,
            torch.zeros((), device=hist_domain_ids.device, dtype=dtype),
            torch.full((), float(self.domain_split_penalty), device=hist_domain_ids.device, dtype=dtype),
        )
        output_gate = torch.ones((batch_size, num_tokens), device=hist_domain_ids.device, dtype=dtype)
        if self.domain_split_zero_empty:
            output_gate[:, cross_slot] = cross_mask.any(dim=1).to(dtype=dtype)
            output_gate[:, same_slot] = same_mask.any(dim=1).to(dtype=dtype)
        slot_kind = torch.zeros((batch_size, num_tokens), device=hist_domain_ids.device, dtype=dtype)
        slot_kind[:, cross_slot] = 1.0
        slot_kind[:, same_slot] = 2.0
        return penalty, output_gate


class FinalSlotReadoutDomainSplitMixFormerModel(DomainSplitReadoutMixFormerModel):
    """DomainSplitReadout with alternative final Q-slot readout before DomainTower."""

    FINAL_READOUT_MODE_CODES = {
        "mean": 0,
        "domain_weighted_sum": 1,
        "concat": 2,
        "domain_weighted_sum_concat_mean": 3,
    }

    def __init__(
        self,
        *args,
        final_slot_readout_mode: str = "domain_weighted_sum",
        final_slot_pre_tower_relu: bool = False,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            **kwargs,
        )
        mode = str(final_slot_readout_mode or "domain_weighted_sum").lower()
        aliases = {
            "mean": "mean",
            "domain_weighted_sum": "domain_weighted_sum",
            "dsw": "domain_weighted_sum",
            "concat": "concat",
            "domain_weighted_sum_concat_mean": "domain_weighted_sum_concat_mean",
            "dsw_concat_mean": "domain_weighted_sum_concat_mean",
            "concat_mean": "domain_weighted_sum_concat_mean",
        }
        if mode not in aliases:
            raise ValueError(f"Unknown final_slot_readout_mode={final_slot_readout_mode!r}.")
        self.final_slot_readout_mode = aliases[mode]
        self.final_slot_pre_tower_relu = bool(final_slot_pre_tower_relu)
        self.final_slot_gate_logits = None
        if self.final_slot_readout_mode in {"domain_weighted_sum", "domain_weighted_sum_concat_mean"}:
            self.final_slot_gate_logits = nn.Embedding(int(self.num_domains), int(self.num_ns_token))
            nn.init.zeros_(self.final_slot_gate_logits.weight)

        final_dim = self._final_slot_readout_dim()
        self.head = DomainTowerHead(
            num_domains=int(self.num_domains),
            input_dim=final_dim + self.late_target_embedding_dim,
            hidden_dims=to_dims(tower_hidden_dims, default=(128, 64)),
            dropout_rate=tower_dropout_rate,
            activation="relu",
            batch_norm=tower_batch_norm,
        )

    def _final_slot_readout_dim(self) -> int:
        if self.final_slot_readout_mode == "concat":
            return int(self.token_dim) * int(self.num_ns_token)
        if self.final_slot_readout_mode == "domain_weighted_sum_concat_mean":
            return int(self.token_dim) * 2
        return int(self.token_dim)

    def _domain_slot_gate(self, interaction, ns_tokens: torch.Tensor) -> torch.Tensor:
        if self.final_slot_gate_logits is None:
            raise RuntimeError("final_slot_gate_logits is not enabled for this readout mode.")
        domain_ids = interaction[self.DOMAIN].long().view(-1) - int(self.domain_shift)
        domain_ids = domain_ids.clamp(min=0, max=int(self.num_domains) - 1)
        logits = self.final_slot_gate_logits(domain_ids).to(dtype=ns_tokens.dtype)
        gate = torch.softmax(logits, dim=-1)
        entropy = -(gate.clamp_min(1e-12) * gate.clamp_min(1e-12).log()).sum(dim=-1)
        cross_slot, same_slot = self._domain_split_slot_indices(int(ns_tokens.shape[1]))
        normal_mask = torch.ones(int(ns_tokens.shape[1]), device=ns_tokens.device, dtype=torch.bool)
        normal_mask[same_slot] = False
        normal_mask[cross_slot] = False
        return gate

    def _pool_final_slots(self, interaction, ns_tokens: torch.Tensor) -> torch.Tensor:
        ns_tokens = self._maybe_permute_final_slots(interaction, ns_tokens)
        mode_code = self.FINAL_READOUT_MODE_CODES[self.final_slot_readout_mode]
        slot_norm = ns_tokens.detach().float().norm(dim=-1).to(device=ns_tokens.device, dtype=ns_tokens.dtype)

        mean_feature = ns_tokens.mean(dim=1)
        if self.final_slot_readout_mode == "mean":
            feature = mean_feature
        elif self.final_slot_readout_mode == "domain_weighted_sum":
            gate = self._domain_slot_gate(interaction, ns_tokens)
            feature = (ns_tokens * gate.unsqueeze(-1)).sum(dim=1)
        elif self.final_slot_readout_mode == "concat":
            feature = ns_tokens.flatten(start_dim=1)
        elif self.final_slot_readout_mode == "domain_weighted_sum_concat_mean":
            gate = self._domain_slot_gate(interaction, ns_tokens)
            gated_feature = (ns_tokens * gate.unsqueeze(-1)).sum(dim=1)
            feature = torch.cat([mean_feature, gated_feature], dim=-1)
        else:
            raise RuntimeError(f"Unsupported final slot readout mode: {self.final_slot_readout_mode}")

        if self.final_slot_pre_tower_relu:
            feature = torch.relu(feature)
        return feature


class CrossDomainPositionMixFormerModel(MixFormerModel):
    """MixFormer with target-domain-aware recency encoding on history tokens."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_domain_position_max_len: int = 100,
        cross_domain_position_init_std: float = 1e-4,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
        )
        self.domain_seq_field = str(domain_seq_field)
        self.cross_domain_position_max_len = max(1, int(cross_domain_position_max_len))
        self.cross_domain_position_init_std = float(cross_domain_position_init_std)
        self.cross_domain_position_embedding = nn.Embedding(
            int(num_domains) * int(num_domains) * (self.cross_domain_position_max_len + 1),
            self.token_dim,
        )
        nn.init.normal_(
            self.cross_domain_position_embedding.weight,
            mean=0.0,
            std=self.cross_domain_position_init_std,
        )

    def _history_domain_ids(
        self,
        interaction,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_field = self.domain_seq_field if self.domain_seq_field in interaction else "tab_seq"
        if source_field not in interaction:
            empty_ids = torch.zeros_like(hist_mask, dtype=torch.long)
            return empty_ids, torch.zeros_like(hist_mask, dtype=torch.bool)
        raw_domain = interaction[source_field].long().to(device=hist_mask.device)
        if raw_domain.shape[:2] != hist_mask.shape:
            raise ValueError(
                f"{source_field} shape {tuple(raw_domain.shape)} does not match history mask {tuple(hist_mask.shape)}"
            )
        if self.domain_shift > 0:
            known = hist_mask & (raw_domain > 0)
        else:
            known = hist_mask & (raw_domain >= 0)
        domain_ids = (raw_domain - int(self.domain_shift)).clamp(min=0, max=int(self.num_domains) - 1)
        return domain_ids, known

    def _cross_domain_position_delta(
        self,
        interaction,
        *,
        hist_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        hist_domain_ids, known = self._history_domain_ids(interaction, hist_mask)
        batch_size, seq_len = hist_domain_ids.shape
        positions = torch.arange(seq_len, device=hist_domain_ids.device).unsqueeze(0)
        hist_len = hist_mask.long().sum(dim=1, keepdim=True)
        recency_ids = (hist_len - 1 - positions).clamp(min=0, max=self.cross_domain_position_max_len)
        target_domain_ids = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        pair_ids = target_domain_ids.unsqueeze(1) * int(self.num_domains) + hist_domain_ids.long()
        embedding_ids = pair_ids * (self.cross_domain_position_max_len + 1) + recency_ids.long()
        delta = self.cross_domain_position_embedding(embedding_ids)
        delta = delta * known.unsqueeze(-1).to(dtype=delta.dtype)
        if batch_size == 0:
            return delta
        return delta

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        cross_pos_delta = self._cross_domain_position_delta(
            interaction,
            hist_mask=mask,
            target_domain_ids=domain_ids,
        ).to(device=seq_tokens.device, dtype=seq_tokens.dtype)
        seq_tokens = seq_tokens + cross_pos_delta
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        _, ns_tokens = self.mixformer(
            seq_tokens,
            ns_tokens,
            mask,
            query_modulation_domain_ids=domain_ids,
        )
        features = self.dropout(ns_tokens.mean(dim=1))
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class FactorizedCrossDomainPositionMixFormerModel(CrossDomainPositionMixFormerModel):
    """Cross-domain recency encoding factorized as +source-in -target-out."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_domain_position_max_len: int = 100,
        cross_domain_position_init_std: float = 1e-4,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_domain_position_max_len=cross_domain_position_max_len,
            cross_domain_position_init_std=cross_domain_position_init_std,
        )
        del self.cross_domain_position_embedding
        table_size = int(num_domains) * (self.cross_domain_position_max_len + 1)
        self.cross_domain_position_in_embedding = nn.Embedding(table_size, self.token_dim)
        self.cross_domain_position_out_embedding = nn.Embedding(table_size, self.token_dim)
        nn.init.normal_(
            self.cross_domain_position_in_embedding.weight,
            mean=0.0,
            std=self.cross_domain_position_init_std,
        )
        nn.init.normal_(
            self.cross_domain_position_out_embedding.weight,
            mean=0.0,
            std=self.cross_domain_position_init_std,
        )

    def _cross_domain_position_delta(
        self,
        interaction,
        *,
        hist_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        hist_domain_ids, known = self._history_domain_ids(interaction, hist_mask)
        _, seq_len = hist_domain_ids.shape
        positions = torch.arange(seq_len, device=hist_domain_ids.device).unsqueeze(0)
        hist_len = hist_mask.long().sum(dim=1, keepdim=True)
        recency_ids = (hist_len - 1 - positions).clamp(min=0, max=self.cross_domain_position_max_len)
        recency_ids = recency_ids.long()
        target_domain_ids = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        table_width = self.cross_domain_position_max_len + 1
        in_ids = hist_domain_ids.long() * table_width + recency_ids
        out_ids = target_domain_ids.unsqueeze(1) * table_width + recency_ids
        in_delta = self.cross_domain_position_in_embedding(in_ids)
        out_delta = self.cross_domain_position_out_embedding(out_ids)
        mask = known.unsqueeze(-1).to(dtype=in_delta.dtype)
        in_delta = in_delta * mask
        out_delta = out_delta * mask
        delta = in_delta - out_delta

        pair_ids = target_domain_ids.unsqueeze(1) * int(self.num_domains) + hist_domain_ids.long()
        target_domain_matrix = target_domain_ids.unsqueeze(1).expand_as(hist_domain_ids)
        return delta


class EntryUpdateMixFormerModel(MixFormerModel):
    """MixFormer with sequence-side entry token message passing."""

    ENTRY_UPDATE_OFF_FIELD = "__mixformer_entry_update_off__"

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        entry_update_mode: str = "last1",
        entry_update_target_only: bool = False,
        entry_update_scale: float = 0.1,
        entry_update_init_std: float = 1e-3,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
        )
        mode = str(entry_update_mode or "last1").lower()
        if mode in {"segment", "prev_segment", "prev_segment_mean"}:
            mode = "segment_mean"
        if mode in {"segment_encoder", "segment_encoder_k50", "prev_segment_encoder", "segment_gru"}:
            mode = "segment_encoder"
        if mode in {"domain_subseq_encoder", "domain_subseq_encoder_k100", "source_domain_subseq_encoder", "domain_subseq_gru"}:
            mode = "domain_subseq_encoder"
        allowed_modes = {"last1", "segment_mean", "segment_encoder", "domain_subseq_encoder"}
        if mode not in allowed_modes:
            raise ValueError(f"entry_update_mode must be one of {sorted(allowed_modes)}, got {entry_update_mode!r}.")
        self.domain_seq_field = str(domain_seq_field)
        self.entry_update_mode = mode
        self.entry_update_target_only = bool(entry_update_target_only)
        self.entry_update_scale = float(entry_update_scale)
        self.entry_update_pair_embedding = nn.Embedding(int(num_domains) * int(num_domains), self.token_dim)
        self.entry_update_target_embedding = nn.Embedding(int(num_domains), self.token_dim)
        self.entry_update_context_norm = nn.LayerNorm(self.token_dim)
        self.entry_update_context_input = nn.Linear(self.token_dim, self.token_dim)
        self.entry_update_context_state = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.entry_update_context_gate = nn.Linear(self.token_dim * 2, self.token_dim)
        entry_input_dim = self.token_dim * 6
        hidden_dim = self.token_dim * 2
        self.entry_update_message_mlp = nn.Sequential(
            nn.LayerNorm(entry_input_dim),
            nn.Linear(entry_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.entry_update_gate_mlp = nn.Sequential(
            nn.LayerNorm(entry_input_dim),
            nn.Linear(entry_input_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )
        nn.init.normal_(self.entry_update_pair_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.entry_update_target_embedding.weight, mean=0.0, std=0.02)
        message_last = self.entry_update_message_mlp[-1]
        gate_last = self.entry_update_gate_mlp[-1]
        nn.init.normal_(message_last.weight, mean=0.0, std=float(entry_update_init_std))
        nn.init.zeros_(message_last.bias)
        nn.init.zeros_(gate_last.weight)
        nn.init.constant_(gate_last.bias, -1.0)

    def _history_domain_ids(
        self,
        interaction,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_field = self.domain_seq_field if self.domain_seq_field in interaction else "tab_seq"
        if source_field not in interaction:
            empty_ids = torch.zeros_like(hist_mask, dtype=torch.long)
            return empty_ids, torch.zeros_like(hist_mask, dtype=torch.bool)
        raw_domain = interaction[source_field].long().to(device=hist_mask.device)
        if raw_domain.shape[:2] != hist_mask.shape:
            raise ValueError(
                f"{source_field} shape {tuple(raw_domain.shape)} does not match history mask {tuple(hist_mask.shape)}"
            )
        if self.domain_shift > 0:
            known = hist_mask & (raw_domain > 0)
        else:
            known = hist_mask & (raw_domain >= 0)
        domain_ids = (raw_domain - int(self.domain_shift)).clamp(min=0, max=int(self.num_domains) - 1)
        return domain_ids, known

    def _entry_update_off(self, interaction) -> bool:
        if self.ENTRY_UPDATE_OFF_FIELD not in interaction:
            return False
        flag = interaction[self.ENTRY_UPDATE_OFF_FIELD]
        return bool(flag.detach().float().max().item() > 0.5)

    def _previous_last_token_context(
        self,
        seq_tokens: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero_tokens = torch.zeros_like(seq_tokens[:, :1, :])
        prev_tokens = torch.cat([zero_tokens, seq_tokens[:, :-1, :]], dim=1)
        zero_domain = torch.zeros_like(hist_domain_ids[:, :1])
        prev_domain_ids = torch.cat([zero_domain, hist_domain_ids[:, :-1]], dim=1)
        prev_known = torch.cat([torch.zeros_like(known_domain_mask[:, :1]), known_domain_mask[:, :-1]], dim=1)
        return prev_tokens, prev_domain_ids, prev_known

    def _previous_segment_mean_context(
        self,
        seq_tokens: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = seq_tokens.shape
        running_sum = seq_tokens.new_zeros(batch_size, dim)
        running_count = seq_tokens.new_zeros(batch_size, 1)
        prev_domain = torch.zeros(batch_size, device=seq_tokens.device, dtype=torch.long)
        prev_known = torch.zeros(batch_size, device=seq_tokens.device, dtype=torch.bool)
        contexts: list[torch.Tensor] = []
        for pos in range(seq_len):
            contexts.append(running_sum / running_count.clamp_min(1.0))
            token = seq_tokens[:, pos, :]
            token_known = known_domain_mask[:, pos]
            curr_domain = hist_domain_ids[:, pos]
            same_segment = prev_known & token_known & (curr_domain == prev_domain)
            reset_segment = token_known & ~same_segment
            running_sum = torch.where(
                same_segment.unsqueeze(-1),
                running_sum + token,
                torch.where(reset_segment.unsqueeze(-1), token, torch.zeros_like(running_sum)),
            )
            running_count = torch.where(
                same_segment.unsqueeze(-1),
                running_count + 1.0,
                torch.where(reset_segment.unsqueeze(-1), torch.ones_like(running_count), torch.zeros_like(running_count)),
            )
            prev_domain = torch.where(token_known, curr_domain, prev_domain)
            prev_known = token_known
        if not contexts:
            return torch.zeros_like(seq_tokens)
        return torch.stack(contexts, dim=1)

    def _entry_context_encoder_step(self, token: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        token = self.entry_update_context_norm(token)
        candidate = torch.tanh(self.entry_update_context_input(token) + self.entry_update_context_state(state))
        gate = torch.sigmoid(self.entry_update_context_gate(torch.cat([token, state], dim=-1)))
        return gate * candidate + (1.0 - gate) * state

    def _previous_segment_encoder_context(
        self,
        seq_tokens: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = seq_tokens.shape
        running_state = seq_tokens.new_zeros(batch_size, dim)
        prev_domain = torch.zeros(batch_size, device=seq_tokens.device, dtype=torch.long)
        prev_known = torch.zeros(batch_size, device=seq_tokens.device, dtype=torch.bool)
        contexts: list[torch.Tensor] = []
        for pos in range(seq_len):
            contexts.append(running_state)
            token = seq_tokens[:, pos, :]
            token_known = known_domain_mask[:, pos]
            curr_domain = hist_domain_ids[:, pos]
            same_segment = prev_known & token_known & (curr_domain == prev_domain)
            base_state = torch.where(same_segment.unsqueeze(-1), running_state, torch.zeros_like(running_state))
            updated_state = self._entry_context_encoder_step(token, base_state)
            running_state = torch.where(token_known.unsqueeze(-1), updated_state, torch.zeros_like(running_state))
            prev_domain = torch.where(token_known, curr_domain, prev_domain)
            prev_known = token_known
        if not contexts:
            return torch.zeros_like(seq_tokens)
        return torch.stack(contexts, dim=1)

    def _previous_domain_subseq_encoder_context(
        self,
        seq_tokens: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        source_domain_ids: torch.Tensor,
        source_known_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = seq_tokens.shape
        num_domains = int(self.num_domains)
        states = seq_tokens.new_zeros(batch_size, num_domains, dim)
        rows = torch.arange(batch_size, device=seq_tokens.device)
        contexts: list[torch.Tensor] = []
        zero_context = seq_tokens.new_zeros(batch_size, dim)
        for pos in range(seq_len):
            source_ids = source_domain_ids[:, pos].long().clamp(min=0, max=max(0, num_domains - 1))
            context = states[rows, source_ids]
            context = torch.where(source_known_mask[:, pos].unsqueeze(-1), context, zero_context)
            contexts.append(context)

            token_known = known_domain_mask[:, pos]
            curr_domain = hist_domain_ids[:, pos].long().clamp(min=0, max=max(0, num_domains - 1))
            prev_state = states[rows, curr_domain]
            updated_state = self._entry_context_encoder_step(seq_tokens[:, pos, :], prev_state)
            next_state = torch.where(token_known.unsqueeze(-1), updated_state, prev_state)
            states = states.clone()
            states[rows, curr_domain] = next_state
        if not contexts:
            return torch.zeros_like(seq_tokens)
        return torch.stack(contexts, dim=1)

    def _entry_update_sequence(
        self,
        *,
        interaction,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        hist_domain_ids, known = self._history_domain_ids(interaction, hist_mask)
        prev_last_tokens, prev_domain_ids, prev_known = self._previous_last_token_context(
            seq_tokens,
            hist_domain_ids,
            known,
        )
        if self.entry_update_mode == "segment_mean":
            context_tokens = self._previous_segment_mean_context(seq_tokens, hist_domain_ids, known)
        elif self.entry_update_mode == "segment_encoder":
            context_tokens = self._previous_segment_encoder_context(seq_tokens, hist_domain_ids, known)
        elif self.entry_update_mode == "domain_subseq_encoder":
            context_tokens = self._previous_domain_subseq_encoder_context(
                seq_tokens,
                hist_domain_ids,
                known,
                prev_domain_ids,
                prev_known,
            )
        else:
            context_tokens = prev_last_tokens
        entry_mask = known & prev_known & (hist_domain_ids != prev_domain_ids)
        target_domain_ids = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        if self.entry_update_target_only:
            entry_mask = entry_mask & (hist_domain_ids == target_domain_ids.unsqueeze(1))
        pair_ids = (prev_domain_ids.long() * int(self.num_domains) + hist_domain_ids.long()).clamp(
            min=0,
            max=int(self.num_domains) * int(self.num_domains) - 1,
        )
        target_emb = self.entry_update_target_embedding(target_domain_ids).unsqueeze(1).expand_as(seq_tokens)
        pair_emb = self.entry_update_pair_embedding(pair_ids)
        update_input = torch.cat(
            [
                seq_tokens,
                context_tokens,
                seq_tokens * context_tokens,
                seq_tokens - context_tokens,
                pair_emb,
                target_emb,
            ],
            dim=-1,
        )
        message = self.entry_update_message_mlp(update_input)
        gate = torch.sigmoid(self.entry_update_gate_mlp(update_input).squeeze(-1))
        if self._entry_update_off(interaction):
            gate = torch.zeros_like(gate)
        else:
            pass
        gate = gate * entry_mask.to(dtype=gate.dtype)
        delta = self.entry_update_scale * gate.unsqueeze(-1).to(dtype=message.dtype) * message
        updated = (seq_tokens + delta) * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        return updated

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        seq_tokens = self._entry_update_sequence(
            interaction=interaction,
            seq_tokens=seq_tokens,
            hist_mask=mask,
            target_domain_ids=domain_ids,
        )

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        _, ns_tokens = self.mixformer(
            seq_tokens,
            ns_tokens,
            mask,
            query_modulation_domain_ids=domain_ids,
        )
        features = self.dropout(ns_tokens.mean(dim=1))
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class DomainFlowCrossDomainPositionMixFormerModel(CrossDomainPositionMixFormerModel):
    """CrossPos plus directed domain-flow transition states.

    The flow branch keeps the original 4-query MixFormer topology intact. It adds
    small directed adjacent-transition deltas to history tokens and lets the final
    feature read a compact latest-state memory from other domains.
    """

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_domain_position_max_len: int = 100,
        cross_domain_position_init_std: float = 1e-4,
        domain_flow_init_std: float = 1e-4,
        domain_flow_state_scale: float = 0.1,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_domain_position_max_len=cross_domain_position_max_len,
            cross_domain_position_init_std=cross_domain_position_init_std,
        )
        self.domain_flow_init_std = float(domain_flow_init_std)
        self.domain_flow_state_scale = float(domain_flow_state_scale)
        pair_count = int(num_domains) * int(num_domains)
        self.domain_flow_prev_transition_embedding = nn.Embedding(pair_count, self.token_dim)
        self.domain_flow_next_transition_embedding = nn.Embedding(pair_count, self.token_dim)
        nn.init.normal_(self.domain_flow_prev_transition_embedding.weight, mean=0.0, std=self.domain_flow_init_std)
        nn.init.normal_(self.domain_flow_next_transition_embedding.weight, mean=0.0, std=self.domain_flow_init_std)

        self.domain_flow_target_embedding = nn.Embedding(int(num_domains), self.token_dim)
        nn.init.normal_(self.domain_flow_target_embedding.weight, mean=0.0, std=0.02)
        self.domain_flow_pair_bias = nn.Embedding(pair_count, 1)
        nn.init.zeros_(self.domain_flow_pair_bias.weight)
        self.domain_flow_state_norm = _make_norm(self.token_dim)
        self.domain_flow_query_mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim * 2),
            nn.Linear(self.token_dim * 2, self.token_dim * 2),
            nn.GELU(),
            nn.Linear(self.token_dim * 2, self.token_dim),
        )
        state_input_dim = self.token_dim * 3 + 4
        self.domain_flow_state_mlp = nn.Sequential(
            nn.LayerNorm(state_input_dim),
            nn.Linear(state_input_dim, self.token_dim * 2),
            nn.GELU(),
            nn.Linear(self.token_dim * 2, self.token_dim),
        )
        self.domain_flow_gate_mlp = nn.Sequential(
            nn.LayerNorm(state_input_dim),
            nn.Linear(state_input_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )
        last_state = self.domain_flow_state_mlp[-1]
        if isinstance(last_state, nn.Linear):
            nn.init.zeros_(last_state.weight)
            nn.init.zeros_(last_state.bias)
        last_gate = self.domain_flow_gate_mlp[-1]
        if isinstance(last_gate, nn.Linear):
            nn.init.zeros_(last_gate.weight)
            nn.init.constant_(last_gate.bias, -2.0)

    @staticmethod
    def _domain_flow_masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        masked_logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(masked_logits, dim=-1) * mask.to(dtype=logits.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return weights / denom

    def _domain_flow_transition_delta(
        self,
        *,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = seq_tokens.shape
        prev_cross = torch.zeros_like(hist_mask, dtype=torch.bool)
        next_cross = torch.zeros_like(hist_mask, dtype=torch.bool)
        prev_pair_ids = torch.zeros_like(hist_domain_ids, dtype=torch.long)
        next_pair_ids = torch.zeros_like(hist_domain_ids, dtype=torch.long)
        if seq_len > 1:
            adjacent_known = known_domain_mask[:, :-1] & known_domain_mask[:, 1:]
            adjacent_cross = adjacent_known & (hist_domain_ids[:, :-1] != hist_domain_ids[:, 1:])
            prev_cross[:, 1:] = adjacent_cross
            next_cross[:, :-1] = adjacent_cross
            prev_pair_ids[:, 1:] = hist_domain_ids[:, :-1] * int(self.num_domains) + hist_domain_ids[:, 1:]
            next_pair_ids[:, :-1] = hist_domain_ids[:, :-1] * int(self.num_domains) + hist_domain_ids[:, 1:]
        prev_delta = self.domain_flow_prev_transition_embedding(prev_pair_ids.clamp(min=0))
        next_delta = self.domain_flow_next_transition_embedding(next_pair_ids.clamp(min=0))
        delta = (
            prev_delta * prev_cross.unsqueeze(-1).to(dtype=prev_delta.dtype)
            + next_delta * next_cross.unsqueeze(-1).to(dtype=next_delta.dtype)
        )
        delta = delta * hist_mask.unsqueeze(-1).to(dtype=delta.dtype)
        return delta

    def _domain_flow_transition_to_target_counts(
        self,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = hist_domain_ids.shape
        counts = torch.zeros(batch_size, int(self.num_domains), device=hist_domain_ids.device, dtype=torch.float32)
        if seq_len > 1:
            source = hist_domain_ids[:, :-1].long().clamp(min=0, max=int(self.num_domains) - 1)
            dest = hist_domain_ids[:, 1:].long().clamp(min=0, max=int(self.num_domains) - 1)
            valid = known_domain_mask[:, :-1] & known_domain_mask[:, 1:]
            target = target_domain_ids.long().view(-1, 1).clamp(min=0, max=int(self.num_domains) - 1)
            to_target = valid & (dest == target)
            one_hot = F.one_hot(source, num_classes=int(self.num_domains)).to(dtype=counts.dtype)
            counts = (one_hot * to_target.unsqueeze(-1).to(dtype=counts.dtype)).sum(dim=1)
        last_source = torch.zeros_like(counts)
        positions = torch.arange(seq_len, device=hist_domain_ids.device).view(1, -1)
        score = torch.where(known_domain_mask, positions, torch.full_like(positions, -1))
        last_idx = score.argmax(dim=1)
        has_history = score.max(dim=1).values >= 0
        last_domain = hist_domain_ids.gather(1, last_idx.view(-1, 1)).squeeze(1).clamp(min=0, max=int(self.num_domains) - 1)
        last_source.scatter_(1, last_domain.view(-1, 1), has_history.to(dtype=counts.dtype).view(-1, 1))
        return counts, last_source

    def _domain_flow_latest_states(
        self,
        *,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = seq_tokens.shape
        domain_range = torch.arange(int(self.num_domains), device=seq_tokens.device).view(1, 1, -1)
        positions = torch.arange(seq_len, device=seq_tokens.device).view(1, seq_len, 1)
        domain_match = hist_domain_ids.long().unsqueeze(-1) == domain_range
        present_at_pos = known_domain_mask.unsqueeze(-1) & domain_match
        score = torch.where(present_at_pos, positions, torch.full_like(positions, -1))
        last_idx = score.argmax(dim=1)
        present = score.max(dim=1).values >= 0
        gather_idx = last_idx.clamp(min=0).unsqueeze(-1).expand(batch_size, int(self.num_domains), dim)
        states = seq_tokens.gather(1, gather_idx) * present.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        hist_len = hist_mask.long().sum(dim=1, keepdim=True)
        recency = (hist_len - 1 - last_idx).clamp(min=0).to(dtype=seq_tokens.dtype)
        recency = recency * present.to(dtype=seq_tokens.dtype)
        return states, present, recency

    def _domain_flow_state_residual(
        self,
        *,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        base_feature: torch.Tensor,
    ) -> torch.Tensor:
        states, present, recency = self._domain_flow_latest_states(
            seq_tokens=seq_tokens,
            hist_mask=hist_mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
        )
        target_domain_ids = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        target_emb = self.domain_flow_target_embedding(target_domain_ids).to(dtype=base_feature.dtype)
        query = self.domain_flow_query_mlp(torch.cat([base_feature, target_emb], dim=-1))
        norm_states = self.domain_flow_state_norm(states)
        logits = torch.einsum("bd,bkd->bk", query, norm_states) * (float(self.token_dim) ** -0.5)
        source_domain = torch.arange(int(self.num_domains), device=seq_tokens.device).view(1, -1)
        source_is_cross = source_domain != target_domain_ids.view(-1, 1)
        pair_ids = target_domain_ids.view(-1, 1) * int(self.num_domains) + source_domain
        logits = logits + self.domain_flow_pair_bias(pair_ids).squeeze(-1).to(dtype=logits.dtype)
        transition_counts, last_source = self._domain_flow_transition_to_target_counts(
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=target_domain_ids,
        )
        logits = logits + torch.log1p(transition_counts.to(dtype=logits.dtype)) + 0.5 * last_source.to(dtype=logits.dtype)
        source_mask = present & source_is_cross
        weights = self._domain_flow_masked_softmax(logits, source_mask)
        summary = torch.einsum("bk,bkd->bd", weights, states)
        cross_count = source_mask.long().sum(dim=1, keepdim=True).to(dtype=base_feature.dtype)
        total_count = present.long().sum(dim=1, keepdim=True).clamp_min(1).to(dtype=base_feature.dtype)
        stats = torch.cat(
            [
                (cross_count > 0).to(dtype=base_feature.dtype),
                cross_count / float(max(1, int(self.num_domains) - 1)),
                transition_counts.sum(dim=1, keepdim=True).to(dtype=base_feature.dtype) / total_count,
                (last_source * source_is_cross.to(dtype=last_source.dtype)).sum(dim=1, keepdim=True).to(dtype=base_feature.dtype),
            ],
            dim=-1,
        )
        state_input = torch.cat([summary, base_feature, target_emb, stats], dim=-1)
        gate = torch.sigmoid(self.domain_flow_gate_mlp(state_input)).squeeze(-1)
        residual = self.domain_flow_state_mlp(state_input)
        residual = float(self.domain_flow_state_scale) * gate.unsqueeze(-1) * residual
        return residual

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(curr_ts, hist_ts, seq_lens=seq_len)
        seq_tokens = self.seq_token_proj(seq_emb)
        cross_pos_delta = self._cross_domain_position_delta(
            interaction,
            hist_mask=mask,
            target_domain_ids=domain_ids,
        ).to(device=seq_tokens.device, dtype=seq_tokens.dtype)
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)
        flow_delta = self._domain_flow_transition_delta(
            seq_tokens=seq_tokens,
            hist_mask=mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
        ).to(device=seq_tokens.device, dtype=seq_tokens.dtype)
        seq_tokens = seq_tokens + cross_pos_delta + flow_delta
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        _, ns_tokens = self.mixformer(
            seq_tokens,
            ns_tokens,
            mask,
            query_modulation_domain_ids=domain_ids,
        )
        features = self.dropout(ns_tokens.mean(dim=1))
        flow_residual = self._domain_flow_state_residual(
            seq_tokens=seq_tokens,
            hist_mask=mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            base_feature=features,
        )
        features = features + flow_residual
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class DomainFlowFactorizedCrossDomainPositionMixFormerModel(DomainFlowCrossDomainPositionMixFormerModel):
    """Factorized CrossPos plus directed domain-flow transition states."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_domain_position_max_len: int = 100,
        cross_domain_position_init_std: float = 1e-4,
        domain_flow_init_std: float = 1e-4,
        domain_flow_state_scale: float = 0.1,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_domain_position_max_len=cross_domain_position_max_len,
            cross_domain_position_init_std=cross_domain_position_init_std,
            domain_flow_init_std=domain_flow_init_std,
            domain_flow_state_scale=domain_flow_state_scale,
        )
        del self.cross_domain_position_embedding
        table_size = int(num_domains) * (self.cross_domain_position_max_len + 1)
        self.cross_domain_position_in_embedding = nn.Embedding(table_size, self.token_dim)
        self.cross_domain_position_out_embedding = nn.Embedding(table_size, self.token_dim)
        nn.init.normal_(
            self.cross_domain_position_in_embedding.weight,
            mean=0.0,
            std=self.cross_domain_position_init_std,
        )
        nn.init.normal_(
            self.cross_domain_position_out_embedding.weight,
            mean=0.0,
            std=self.cross_domain_position_init_std,
        )

    def _cross_domain_position_delta(
        self,
        interaction,
        *,
        hist_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        return FactorizedCrossDomainPositionMixFormerModel._cross_domain_position_delta(
            self,
            interaction,
            hist_mask=hist_mask,
            target_domain_ids=target_domain_ids,
        )


class CrossEventReaderMixFormerModel(MixFormerModel):
    """MixFormer with a side branch that contextualizes cross-domain events."""

    CROSS_EVENT_OFF_FIELD = "mixformer_cross_event_off"

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_event_logit_scale: float = 0.05,
        cross_event_learnable_scale: bool = False,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
        )
        self.domain_seq_field = str(domain_seq_field)
        self.cross_event_learnable_scale = bool(cross_event_learnable_scale)
        if self.cross_event_learnable_scale:
            self.cross_event_logit_scale_param = nn.Parameter(
                torch.tensor(float(cross_event_logit_scale), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "cross_event_logit_scale_buffer",
                torch.tensor(float(cross_event_logit_scale), dtype=torch.float32),
                persistent=False,
            )

        self.cross_event_q_norm = _make_norm(self.token_dim)
        self.cross_event_k_norm = _make_norm(self.token_dim)
        self.cross_event_q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.cross_event_k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.cross_event_v_proj = nn.Linear(self.token_dim, self.token_dim)
        self.cross_event_context_proj = nn.Linear(self.token_dim, self.token_dim)
        self.cross_event_summary_k_norm = _make_norm(self.token_dim)
        self.cross_event_summary_k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.cross_event_target_embedding = nn.Embedding(int(num_domains), self.token_dim)
        q5_input_dim = self.token_dim * 2
        self.cross_event_q5_mlp = nn.Sequential(
            nn.LayerNorm(q5_input_dim),
            nn.Linear(q5_input_dim, self.token_dim * 2),
            nn.GELU(),
            nn.Linear(self.token_dim * 2, self.token_dim),
        )
        logit_input_dim = self.token_dim * 3
        self.cross_event_logit_mlp = nn.Sequential(
            nn.LayerNorm(logit_input_dim),
            nn.Linear(logit_input_dim, self.token_dim * 2),
            nn.GELU(),
            nn.Linear(self.token_dim * 2, 1),
        )
        nn.init.normal_(self.cross_event_target_embedding.weight, mean=0.0, std=0.02)

    def _history_domain_ids(
        self,
        interaction,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_field = self.domain_seq_field if self.domain_seq_field in interaction else "tab_seq"
        if source_field not in interaction:
            empty_ids = torch.zeros_like(hist_mask, dtype=torch.long)
            return empty_ids, torch.zeros_like(hist_mask, dtype=torch.bool)
        raw_domain = interaction[source_field].long().to(device=hist_mask.device)
        if raw_domain.shape[:2] != hist_mask.shape:
            raise ValueError(
                f"{source_field} shape {tuple(raw_domain.shape)} does not match history mask {tuple(hist_mask.shape)}"
            )
        if self.domain_shift > 0:
            known = hist_mask & (raw_domain > 0)
        else:
            known = hist_mask & (raw_domain >= 0)
        domain_ids = (raw_domain - int(self.domain_shift)).clamp(min=0, max=int(self.num_domains) - 1)
        return domain_ids, known

    def _cross_event_off(self, interaction) -> bool:
        if self.CROSS_EVENT_OFF_FIELD not in interaction:
            return False
        flag = interaction[self.CROSS_EVENT_OFF_FIELD]
        return bool(flag.detach().float().max().item() > 0.5)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        masked_logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(masked_logits, dim=dim) * mask.to(dtype=logits.dtype)
        denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        return weights / denom

    def _cross_event_scale(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.cross_event_learnable_scale:
            return self.cross_event_logit_scale_param.to(dtype=dtype, device=device)
        return self.cross_event_logit_scale_buffer.to(dtype=dtype, device=device)

    def _zero_cross_event_branch(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        zero = torch.zeros(batch_size, device=device, dtype=dtype)
        return zero

    def _cross_event_branch(
        self,
        *,
        interaction,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        base_feature: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = seq_tokens.shape
        hist_domain_ids, known = self._history_domain_ids(interaction, hist_mask)
        target_domain_ids = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        cross_mask = known & (hist_domain_ids != target_domain_ids.unsqueeze(1))
        cross_count = cross_mask.long().sum(dim=1)
        raw_cross_tokens = seq_tokens * cross_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        positions = torch.arange(seq_len, device=seq_tokens.device)
        causal = positions.view(1, seq_len, 1) >= positions.view(1, 1, seq_len)
        event_attn_mask = causal & hist_mask.unsqueeze(1).to(dtype=torch.bool)
        q = self.cross_event_q_proj(self.cross_event_q_norm(seq_tokens))
        k = self.cross_event_k_proj(self.cross_event_k_norm(seq_tokens))
        v = self.cross_event_v_proj(seq_tokens)
        event_logits = torch.matmul(q, k.transpose(-2, -1)) * (float(self.token_dim) ** -0.5)
        event_attn = self._masked_softmax(event_logits, event_attn_mask, dim=-1)
        event_attn = event_attn * cross_mask.unsqueeze(-1).to(dtype=event_attn.dtype)
        event_context = torch.matmul(event_attn, v)
        event_enhanced = seq_tokens + self.cross_event_context_proj(event_context)
        event_enhanced = event_enhanced * cross_mask.unsqueeze(-1).to(dtype=event_enhanced.dtype)
        enhance_delta = event_enhanced - raw_cross_tokens

        target_emb = self.cross_event_target_embedding(target_domain_ids).to(dtype=seq_tokens.dtype)
        q5_input = torch.cat([base_feature, target_emb], dim=-1)
        q5 = self.cross_event_q5_mlp(q5_input)
        summary_k = self.cross_event_summary_k_proj(self.cross_event_summary_k_norm(event_enhanced))
        q5_logits = torch.einsum("bd,bld->bl", q5, summary_k) * (float(self.token_dim) ** -0.5)
        q5_attn = self._masked_softmax(q5_logits, cross_mask, dim=-1)
        cross_summary = torch.einsum("bl,bld->bd", q5_attn, event_enhanced)
        logit_input = torch.cat([cross_summary, base_feature, target_emb], dim=-1)
        cross_logit = self.cross_event_logit_mlp(logit_input).squeeze(-1)
        scale = self._cross_event_scale(dtype=cross_logit.dtype, device=cross_logit.device)
        scaled_logit = scale * cross_logit
        return scaled_logit

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        _, ns_tokens = self.mixformer(
            seq_tokens,
            ns_tokens,
            mask,
            query_modulation_domain_ids=domain_ids,
        )
        base_feature = self.dropout(ns_tokens.mean(dim=1))
        if self._cross_event_off(interaction):
            self._last_cross_event_scaled_logit = self._zero_cross_event_branch(
                batch_size=int(base_feature.shape[0]),
                dtype=base_feature.dtype,
                device=base_feature.device,
            )
        else:
            self._last_cross_event_scaled_logit = self._cross_event_branch(
                interaction=interaction,
                seq_tokens=seq_tokens,
                hist_mask=mask,
                base_feature=base_feature,
                target_domain_ids=domain_ids,
            )
        features = base_feature
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids

    def logits(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        logits = self.forward(x, domain_ids)
        scaled_logit = getattr(self, "_last_cross_event_scaled_logit", None)
        if isinstance(scaled_logit, torch.Tensor):
            logits = logits + scaled_logit
        return logits


class CrossQ5QueryMixFormerModel(MixFormerModel):
    """MixFormer where an explicit cross-domain Q5 token joins QueryMixer."""

    CROSS_EVENT_OFF_FIELD = CrossEventReaderMixFormerModel.CROSS_EVENT_OFF_FIELD

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "cross_switch",
        q5_decision_scale: float = 1.0,
        attention_same_protect: bool = False,
        cross_context_mode: str = "none",
        cross_context_beta: float = 0.0,
        attention_double_softmax_beta: Optional[float] = None,
        attention_double_softmax_mode: str = "blend",
    ):
        self.base_num_ns_token = int(num_ns_token)
        if self.base_num_ns_token < 2:
            raise ValueError("CrossQ5QueryMixFormerModel requires num_ns_token >= 2.")
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=self.base_num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            attention_same_protect=attention_same_protect,
            cross_context_mode=cross_context_mode,
            cross_context_beta=cross_context_beta,
            attention_double_softmax_beta=attention_double_softmax_beta,
            attention_double_softmax_mode=attention_double_softmax_mode,
        )
        mode = str(cross_q5_mode or "cross_switch").lower()
        allowed_modes = {"cross_switch", "target_only"}
        if mode not in allowed_modes:
            raise ValueError(f"cross_q5_mode must be one of {sorted(allowed_modes)}, got {cross_q5_mode!r}.")
        self.cross_q5_mode = mode
        self.domain_seq_field = str(domain_seq_field)
        self.q5_decision_scale = float(q5_decision_scale)
        self.cross_q5_target_embedding = nn.Embedding(int(num_domains), self.token_dim)
        self.cross_q5_stats_proj = nn.Linear(4, self.token_dim)
        q5_input_dim = self.token_dim * 4
        self.cross_q5_init_mlp = nn.Sequential(
            nn.LayerNorm(q5_input_dim),
            nn.Linear(q5_input_dim, self.token_dim * 2),
            nn.GELU(),
            nn.Linear(self.token_dim * 2, self.token_dim),
        )
        self.cross_q5_final_norm = _make_norm(self.token_dim)
        nn.init.normal_(self.cross_q5_target_embedding.weight, mean=0.0, std=0.02)

    def _history_domain_ids(
        self,
        interaction,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_field = self.domain_seq_field if self.domain_seq_field in interaction else "tab_seq"
        if source_field not in interaction:
            empty_ids = torch.zeros_like(hist_mask, dtype=torch.long)
            return empty_ids, torch.zeros_like(hist_mask, dtype=torch.bool)
        raw_domain = interaction[source_field].long().to(device=hist_mask.device)
        if raw_domain.shape[:2] != hist_mask.shape:
            raise ValueError(
                f"{source_field} shape {tuple(raw_domain.shape)} does not match history mask {tuple(hist_mask.shape)}"
            )
        if self.domain_shift > 0:
            known = hist_mask & (raw_domain > 0)
        else:
            known = hist_mask & (raw_domain >= 0)
        domain_ids = (raw_domain - int(self.domain_shift)).clamp(min=0, max=int(self.num_domains) - 1)
        return domain_ids, known

    def _cross_q5_off(self, interaction) -> bool:
        if self.CROSS_EVENT_OFF_FIELD not in interaction:
            return False
        flag = interaction[self.CROSS_EVENT_OFF_FIELD]
        return bool(flag.detach().float().max().item() > 0.5)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    def _make_cross_q5_token(
        self,
        *,
        interaction,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        hist_domain_ids, known = self._history_domain_ids(interaction, hist_mask)
        target_domain_ids = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        target_domain = target_domain_ids.unsqueeze(1)
        cross_mask = known & (hist_domain_ids != target_domain)
        same_mask = known & (hist_domain_ids == target_domain)
        switch_mask = torch.zeros_like(hist_mask, dtype=torch.bool)
        if seq_tokens.shape[1] > 1:
            prev_known = known[:, :-1] & known[:, 1:]
            switched = hist_domain_ids[:, :-1] != hist_domain_ids[:, 1:]
            switch_mask[:, 1:] = prev_known & switched
        cross_switch_mask = cross_mask & switch_mask
        valid_count = known.long().sum(dim=1).clamp_min(1).to(dtype=seq_tokens.dtype)
        cross_count = cross_mask.long().sum(dim=1).to(dtype=seq_tokens.dtype)
        switch_count = switch_mask.long().sum(dim=1).to(dtype=seq_tokens.dtype)
        cross_switch_count = cross_switch_mask.long().sum(dim=1).to(dtype=seq_tokens.dtype)
        stats = torch.stack(
            [
                cross_count / valid_count,
                switch_count / valid_count,
                cross_switch_count / valid_count,
                (cross_count > 0).to(dtype=seq_tokens.dtype),
            ],
            dim=-1,
        )

        all_pool = self._masked_mean(seq_tokens, known)
        cross_pool = self._masked_mean(seq_tokens, cross_mask)
        switch_pool = self._masked_mean(seq_tokens, cross_switch_mask)
        target_emb = self.cross_q5_target_embedding(target_domain_ids).to(dtype=seq_tokens.dtype)
        if self.cross_q5_mode == "target_only":
            cross_pool = all_pool
            switch_pool = torch.zeros_like(switch_pool)
            stats = torch.zeros_like(stats)
        stats_emb = self.cross_q5_stats_proj(stats.to(dtype=seq_tokens.dtype))
        q5_input = torch.cat([cross_pool, switch_pool, target_emb, stats_emb], dim=-1)
        q5_init = self.cross_q5_init_mlp(q5_input)
        return q5_init

    def _pool_cross_q5_final_slots(
        self,
        interaction,
        ns_tokens: torch.Tensor,
        *,
        q5_active: bool = True,
    ) -> torch.Tensor:
        slot_norm = ns_tokens.detach().float().norm(dim=-1)
        raw_weights = self._final_slot_weights(interaction, ns_tokens)
        effective_weights = raw_weights.clone()
        q5_index = min(max(0, int(self.base_num_ns_token) - 1), int(ns_tokens.shape[1]) - 1)
        if not bool(q5_active):
            effective_weights[:, q5_index] = 0.0
        else:
            effective_weights[:, q5_index] = effective_weights[:, q5_index] * float(self.q5_decision_scale)
        q5_scale = torch.ones_like(raw_weights)
        q5_scale[:, q5_index] = float(self.q5_decision_scale) if bool(q5_active) else 0.0
        denom = effective_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (ns_tokens * effective_weights.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        tokenizer_q5_slot = ns_tokens[:, self.base_num_ns_token - 1, :]
        q5_init = self._make_cross_q5_token(
            interaction=interaction,
            seq_tokens=seq_tokens,
            hist_mask=mask,
            target_domain_ids=domain_ids,
        )
        if self._cross_q5_off(interaction):
            q5_init = torch.zeros_like(q5_init)
        else:
            pass
        ns_tokens = torch.cat([ns_tokens[:, : self.base_num_ns_token - 1, :], q5_init.unsqueeze(1)], dim=1)
        _, ns_tokens = self.mixformer(
            seq_tokens,
            ns_tokens,
            mask,
            query_modulation_domain_ids=domain_ids,
        )
        base_tokens = ns_tokens[:, : self.base_num_ns_token - 1, :]
        q5_final = ns_tokens[:, self.base_num_ns_token - 1, :]
        base_feature = self.dropout(base_tokens.mean(dim=1))
        q5_feature = q5_final
        cross_q5_off = self._cross_q5_off(interaction)
        if cross_q5_off:
            q5_feature = torch.zeros_like(q5_feature)
        readout_tokens = torch.cat([base_tokens, q5_feature.unsqueeze(1)], dim=1)
        features = self.dropout(
            self._pool_cross_q5_final_slots(
                interaction,
                readout_tokens,
                q5_active=not cross_q5_off,
            )
        )
        q5_residual = features - base_feature
        features = self._apply_cross_context_residual(
            features=features,
            seq_tokens=seq_tokens,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
        )
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class TransitionShiftQ5MixFormerModel(CrossQ5QueryMixFormerModel):
    """CrossQ5 variant where the Q5 slot reads a transition-mixed sequence view."""

    FOCUS_ABLATION_FIELD = "mixformer_focus_ablation_code"
    TRANSITION_EVAL_FIELD = "__mixformer_q5_transition_eval_code__"
    FOCUS_ABLATION_CODES = {
        "none": 0,
        "sall_entry_same_target_off": 1,
        "sall_entry_cross_target_off": 2,
        "sall_pre_same_target_off": 3,
        "sall_boundary_same_target_off": 4,
        "sall_random_entry_count_off": 5,
        "n_entry_same_target_off": 6,
    }
    TRANSITION_EVAL_CODES = {
        "q5_transition_off_eval": 1,
        "q5_transition_k_off_eval": 2,
        "q5_transition_v_off_eval": 3,
        "q5_transition_kv_off_eval": 4,
        "q5_transition_persist_off_eval": 5,
        "q5_transition_layer0_off_eval": 10,
        "q5_transition_layer1_off_eval": 11,
        "q5_transition_layer2_off_eval": 12,
    }

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "target_only",
        q5_decision_scale: float = 1.0,
        q5_transition_mode: str = "cross",
        q5_transition_slots: str = "q5",
        q5_transition_kv_mode: str = "kv",
        q5_transition_persist: bool = False,
        q5_transition_persist_scale: float = 1.0,
        q5_transition_persist_source: str = "s",
        q5_transition_fuse_mode: str = "none",
        q5_transition_window_size: int = 3,
        q5_transition_window_num_heads: int = 2,
        q5_transition_window_output_init_std: Optional[float] = None,
        q5_transition_window_residual_scale: float = 1.0,
        q5_transition_linear_num_heads: int = 2,
        q5_transition_linear_output_init_std: Optional[float] = 1e-4,
        q5_transition_linear_residual_scale: float = 0.1,
        q5_transition_linear_swap_chunk: str = "same_position",
        q5_transition_nqns_num_heads: int = 2,
        q5_transition_nqns_output_init_std: Optional[float] = 1e-4,
        q5_transition_nqns_residual_scale: float = 0.1,
        q5_transition_boundary_flag_init_std: float = 1e-4,
        q5_transition_conv_output_init_std: Optional[float] = 0.0,
        q5_transition_conv_residual_scale: float = 1.0,
        q5_transition_off_layers: Optional[Iterable[int]] = None,
        attention_same_protect: bool = False,
        cross_context_mode: str = "none",
        cross_context_beta: float = 0.0,
        attention_double_softmax_beta: Optional[float] = None,
        attention_double_softmax_mode: str = "blend",
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_q5_mode=cross_q5_mode,
            q5_decision_scale=q5_decision_scale,
            attention_same_protect=attention_same_protect,
            cross_context_mode=cross_context_mode,
            cross_context_beta=cross_context_beta,
            attention_double_softmax_beta=attention_double_softmax_beta,
            attention_double_softmax_mode=attention_double_softmax_mode,
        )
        mode = str(q5_transition_mode or "cross").lower()
        allowed_modes = {
            "all",
            "cross",
            "domain_subseq",
            "same",
            "boundary_flag",
            "window_attn",
            "linear_n_swap",
            "n_query_ns",
            "conv_cross_boundary",
            "none",
        }
        if mode not in allowed_modes:
            raise ValueError(f"q5_transition_mode must be one of {sorted(allowed_modes)}, got {q5_transition_mode!r}.")
        if int(self.token_dim) % 4 != 0:
            raise ValueError("TransitionShiftQ5MixFormerModel requires token_dim divisible by 4.")
        self.q5_transition_mode = mode
        self.q5_transition_persist = bool(q5_transition_persist)
        self.q5_transition_persist_scale = float(q5_transition_persist_scale)
        persist_source = str(q5_transition_persist_source or "s").lower()
        persist_source_aliases = {
            "s": "s",
            "shift": "s",
            "shift_swapped": "s",
            "q5": "s",
            "n": "n_pre_swap",
            "linear": "n_pre_swap",
            "linear_n": "n_pre_swap",
            "n_pre": "n_pre_swap",
            "n_pre_swap": "n_pre_swap",
            "linear_pre_swap": "n_pre_swap",
            "n_swapped": "n_swapped",
            "linear_swapped": "n_swapped",
        }
        if persist_source not in persist_source_aliases:
            raise ValueError(
                "q5_transition_persist_source must be one of "
                f"{sorted(persist_source_aliases)}, got {q5_transition_persist_source!r}."
            )
        self.q5_transition_persist_source = persist_source_aliases[persist_source]
        self.q5_transition_window_size = int(q5_transition_window_size)
        self.q5_transition_window_num_heads = int(q5_transition_window_num_heads)
        self.q5_transition_window_residual_scale = float(q5_transition_window_residual_scale)
        self.q5_transition_linear_num_heads = int(q5_transition_linear_num_heads)
        self.q5_transition_linear_residual_scale = float(q5_transition_linear_residual_scale)
        self.q5_transition_nqns_num_heads = int(q5_transition_nqns_num_heads)
        self.q5_transition_nqns_residual_scale = float(q5_transition_nqns_residual_scale)
        self.q5_transition_boundary_flag_init_std = float(q5_transition_boundary_flag_init_std)
        self.q5_transition_conv_residual_scale = float(q5_transition_conv_residual_scale)
        swap_chunk = str(q5_transition_linear_swap_chunk or "same_position").lower()
        if swap_chunk not in {"same_position", "samepos", "same", "first", "first_quarter", "q0", "chunk0"}:
            raise ValueError(
                "q5_transition_linear_swap_chunk must describe the same-position fixed-quarter swap "
                "(same_position/first_quarter/chunk0)."
            )
        self.q5_transition_linear_swap_chunk = "same_position"
        self.q5_transition_linear_swap_index = 0
        fuse_mode = str(q5_transition_fuse_mode or "none").lower()
        allowed_fuse_modes = {"none", "avg", "concat_reduce", "concat_res", "half_ns"}
        if fuse_mode not in allowed_fuse_modes:
            raise ValueError(
                f"q5_transition_fuse_mode must be one of {sorted(allowed_fuse_modes)}, "
                f"got {q5_transition_fuse_mode!r}."
            )
        self.q5_transition_fuse_mode = fuse_mode
        slot_mode = str(q5_transition_slots or "q5").lower()
        if slot_mode in {"2n2s", "last2"}:
            slot_mode = "tail2"
        allowed_slot_modes = {"q5", "all", "tail2"}
        if slot_mode not in allowed_slot_modes:
            raise ValueError(f"q5_transition_slots must be one of {sorted(allowed_slot_modes)}, got {q5_transition_slots!r}.")
        self.q5_transition_slots = slot_mode
        kv_mode = str(q5_transition_kv_mode or "kv").lower()
        kv_mode_aliases = {
            "kv": "kv",
            "both": "kv",
            "all": "kv",
            "k_only": "k_only",
            "key_only": "k_only",
            "keys_only": "k_only",
            "k": "k_only",
            "v_only": "v_only",
            "value_only": "v_only",
            "values_only": "v_only",
            "v": "v_only",
        }
        if kv_mode not in kv_mode_aliases:
            raise ValueError(
                "q5_transition_kv_mode must be one of "
                f"{sorted(kv_mode_aliases)}, got {q5_transition_kv_mode!r}."
            )
        self.q5_transition_kv_mode = kv_mode_aliases[kv_mode]
        num_mix_layers = int(len(self.mixformer.query_mixers))
        off_layers = tuple(sorted({int(layer) for layer in (q5_transition_off_layers or ())}))
        invalid_off_layers = [layer for layer in off_layers if layer < 0 or layer >= num_mix_layers]
        if invalid_off_layers:
            raise ValueError(
                "q5_transition_off_layers must be within "
                f"[0, {max(0, num_mix_layers - 1)}], got {invalid_off_layers}."
            )
        self.q5_transition_off_layers = off_layers
        self.q5_transition_begin = nn.Parameter(torch.zeros(num_mix_layers, self.token_dim))
        self.q5_transition_end = nn.Parameter(torch.zeros(num_mix_layers, self.token_dim))
        self.q5_transition_boundary_flag = nn.Parameter(torch.empty(num_mix_layers, 4, self.token_dim))
        nn.init.normal_(
            self.q5_transition_boundary_flag,
            mean=0.0,
            std=self.q5_transition_boundary_flag_init_std,
        )
        self.q5_transition_window_attns = nn.ModuleList()
        self.q5_transition_linear_attns = nn.ModuleList()
        self.q5_transition_nqns_attns = nn.ModuleList()
        self.q5_transition_conv_projs = nn.ModuleList()
        if self.q5_transition_mode == "window_attn":
            self.q5_transition_window_attns = nn.ModuleList(
                [
                    SlidingWindowSequenceSelfAttention(
                        input_dim=self.token_dim,
                        num_heads=self.q5_transition_window_num_heads,
                        window_size=self.q5_transition_window_size,
                        dropout_rate=dropout_rate,
                        output_init_std=q5_transition_window_output_init_std,
                        residual_scale=self.q5_transition_window_residual_scale,
                    )
                    for _ in range(num_mix_layers)
                ]
            )
        if self.q5_transition_mode == "linear_n_swap":
            self.q5_transition_linear_attns = nn.ModuleList(
                [
                    LinearSequenceSelfAttention(
                        input_dim=self.token_dim,
                        num_heads=self.q5_transition_linear_num_heads,
                        dropout_rate=dropout_rate,
                        output_init_std=q5_transition_linear_output_init_std,
                        residual_scale=self.q5_transition_linear_residual_scale,
                    )
                    for _ in range(num_mix_layers)
                ]
            )
        if self.q5_transition_mode == "n_query_ns":
            self.q5_transition_nqns_attns = nn.ModuleList(
                [
                    SequenceQueryNSTokenAttention(
                        input_dim=self.token_dim,
                        num_heads=self.q5_transition_nqns_num_heads,
                        dropout_rate=dropout_rate,
                        output_init_std=q5_transition_nqns_output_init_std,
                        residual_scale=self.q5_transition_nqns_residual_scale,
                    )
                    for _ in range(num_mix_layers)
                ]
            )
        if self.q5_transition_mode == "conv_cross_boundary":
            self.q5_transition_conv_projs = nn.ModuleList(
                [nn.Linear(self.token_dim * 3, self.token_dim) for _ in range(num_mix_layers)]
            )
            for proj in self.q5_transition_conv_projs:
                if q5_transition_conv_output_init_std is None:
                    continue
                if float(q5_transition_conv_output_init_std) == 0.0:
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                else:
                    nn.init.normal_(proj.weight, mean=0.0, std=float(q5_transition_conv_output_init_std))
                    nn.init.zeros_(proj.bias)
        if self.q5_transition_fuse_mode in {"concat_reduce", "concat_res"}:
            self.q5_transition_fuse_norms = nn.ModuleList(
                [_make_norm(self.token_dim * 2) for _ in range(num_mix_layers)]
            )
            self.q5_transition_fuse_projs = nn.ModuleList(
                [nn.Linear(self.token_dim * 2, self.token_dim) for _ in range(num_mix_layers)]
            )
            if self.q5_transition_fuse_mode == "concat_res":
                for proj in self.q5_transition_fuse_projs:
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)

    @classmethod
    def focus_ablation_code_for_variant(cls, variant: str) -> int:
        return int(cls.FOCUS_ABLATION_CODES[str(variant)])

    @classmethod
    def transition_eval_code_for_variant(cls, variant: str) -> int:
        return int(cls.TRANSITION_EVAL_CODES[str(variant)])

    def _focus_ablation_code(self, interaction) -> int:
        if self.FOCUS_ABLATION_FIELD not in interaction:
            return 0
        code = interaction[self.FOCUS_ABLATION_FIELD]
        if not isinstance(code, torch.Tensor) or code.numel() <= 0:
            return 0
        return int(code.detach().long().max().item())

    def _transition_eval_code(self, interaction) -> int:
        if self.TRANSITION_EVAL_FIELD not in interaction:
            return 0
        code = interaction[self.TRANSITION_EVAL_FIELD]
        if not isinstance(code, torch.Tensor) or code.numel() <= 0:
            return 0
        return int(code.detach().long().max().item())

    @staticmethod
    def _transition_eval_layer_off(eval_code: int, layer_idx: int) -> bool:
        code = int(eval_code)
        if code == 1:
            return True
        return code in {10, 11, 12} and int(layer_idx) == code - 10

    def _transition_train_layer_off(self, layer_idx: int) -> bool:
        return int(layer_idx) in set(getattr(self, "q5_transition_off_layers", ()))

    def _transition_slot_mask(self, num_slots: int, device: torch.device) -> torch.Tensor:
        slot_mask = torch.zeros(int(num_slots), device=device, dtype=torch.bool)
        if self.q5_transition_slots == "all":
            slot_mask[:] = True
        elif self.q5_transition_slots == "tail2":
            slot_mask[max(0, int(num_slots) - 2):] = True
        else:
            q5_index = min(max(0, int(self.base_num_ns_token) - 1), int(num_slots) - 1)
            slot_mask[q5_index] = True
        return slot_mask

    def _q5_transition_fuse_mode_code(self) -> int:
        return {
            "none": 0,
            "avg": 1,
            "concat_reduce": 2,
            "concat_res": 3,
            "half_ns": 4,
        }.get(str(self.q5_transition_fuse_mode), 0)

    def _q5_transition_persist_source_code(self) -> int:
        return {
            "s": 0,
            "n_pre_swap": 1,
            "n_swapped": 2,
        }.get(str(self.q5_transition_persist_source), 0)

    def _q5_transition_kv_mode_code(self) -> int:
        return {
            "kv": 0,
            "k_only": 1,
            "v_only": 2,
        }.get(str(self.q5_transition_kv_mode), 0)

    def _linear_n_sequence(
        self,
        *,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if self.q5_transition_mode != "linear_n_swap":
            return seq_tokens
        if int(layer_idx) >= len(self.q5_transition_linear_attns):
            raise RuntimeError("Linear-N transition attention modules are not initialized.")
        return self.q5_transition_linear_attns[int(layer_idx)](
            seq_tokens,
            hist_mask,
        )

    def _n_query_ns_sequence(
        self,
        *,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if self.q5_transition_mode != "n_query_ns":
            return seq_tokens
        if int(layer_idx) >= len(self.q5_transition_nqns_attns):
            raise RuntimeError("N-query-NS transition attention modules are not initialized.")
        return self.q5_transition_nqns_attns[int(layer_idx)](
            seq_tokens,
            ns_tokens,
            hist_mask,
        )

    def _linear_n_swap_sequences(
        self,
        *,
        n_seq_tokens: torch.Tensor,
        s_seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q5_transition_mode != "linear_n_swap":
            return n_seq_tokens, s_seq_tokens
        n_chunks = list(n_seq_tokens.chunk(4, dim=-1))
        s_chunks = list(s_seq_tokens.chunk(4, dim=-1))
        swap_index = int(self.q5_transition_linear_swap_index)
        n_chunks[swap_index], s_chunks[swap_index] = s_chunks[swap_index], n_chunks[swap_index]
        n_swapped = torch.cat(n_chunks, dim=-1)
        s_swapped = torch.cat(s_chunks, dim=-1)
        mask = hist_mask.unsqueeze(-1).to(dtype=n_swapped.dtype)
        n_swapped = n_swapped * mask
        s_swapped = s_swapped * mask
        return n_swapped, s_swapped

    def _next_q5_transition_seq(
        self,
        *,
        seq_tokens: torch.Tensor,
        q5_seq_tokens: torch.Tensor,
        persist_source_tokens: Optional[torch.Tensor] = None,
        hist_mask: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        mode = str(self.q5_transition_fuse_mode)
        write_back = bool(self.q5_transition_persist or mode != "none")
        if not write_back:
            return seq_tokens

        if persist_source_tokens is not None and mode == "none":
            next_seq_tokens = persist_source_tokens
        elif mode in {"none", "avg"}:
            scale = float(self.q5_transition_persist_scale)
            next_seq_tokens = seq_tokens + scale * (q5_seq_tokens - seq_tokens)
        elif mode == "concat_reduce":
            fuse_input = torch.cat([seq_tokens, q5_seq_tokens], dim=-1)
            fuse_input = self.q5_transition_fuse_norms[int(layer_idx)](fuse_input)
            next_seq_tokens = self.q5_transition_fuse_projs[int(layer_idx)](fuse_input)
        elif mode == "concat_res":
            fuse_input = torch.cat([seq_tokens, q5_seq_tokens], dim=-1)
            fuse_input = self.q5_transition_fuse_norms[int(layer_idx)](fuse_input)
            delta = self.q5_transition_fuse_projs[int(layer_idx)](fuse_input)
            next_seq_tokens = seq_tokens + delta
        elif mode == "half_ns":
            n_chunks = seq_tokens.chunk(4, dim=-1)
            s_chunks = q5_seq_tokens.chunk(4, dim=-1)
            next_seq_tokens = torch.cat([n_chunks[0], n_chunks[1], s_chunks[2], s_chunks[3]], dim=-1)
        else:
            raise ValueError(f"Unsupported q5_transition_fuse_mode: {mode!r}.")

        next_seq_tokens = next_seq_tokens * hist_mask.unsqueeze(-1).to(dtype=next_seq_tokens.dtype)
        if persist_source_tokens is not None and mode == "none":
            pass
        if self.q5_transition_persist:
            pass
        return next_seq_tokens

    @staticmethod
    def _adjacent_domain_masks(
        *,
        known_domain_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry_switch = torch.zeros_like(known_domain_mask, dtype=torch.bool)
        pre_switch = torch.zeros_like(known_domain_mask, dtype=torch.bool)
        if hist_domain_ids.shape[1] > 1:
            adjacent_known = known_domain_mask[:, :-1] & known_domain_mask[:, 1:]
            adjacent_cross = adjacent_known & (hist_domain_ids[:, :-1] != hist_domain_ids[:, 1:])
            entry_switch[:, 1:] = adjacent_cross
            pre_switch[:, :-1] = adjacent_cross
        return entry_switch, pre_switch, entry_switch | pre_switch

    @staticmethod
    def _deterministic_same_count_mask(
        *,
        candidate_mask: torch.Tensor,
        count_like_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        batch_size, seq_len = candidate_mask.shape
        counts = count_like_mask.long().sum(dim=1)
        candidate_counts = candidate_mask.long().sum(dim=1)
        counts = torch.minimum(counts, candidate_counts)
        if int(counts.max().item()) <= 0:
            return torch.zeros_like(candidate_mask, dtype=torch.bool)
        positions = torch.arange(seq_len, device=candidate_mask.device, dtype=torch.float32).view(1, -1)
        domains = target_domain_ids.to(device=candidate_mask.device, dtype=torch.float32).view(-1, 1)
        scores = torch.frac(
            torch.sin((positions + 1.0) * 12.9898 + (domains + 1.0) * 78.233 + float(layer_idx + 1) * 37.719)
            * 43758.5453
        )
        scores = scores.masked_fill(~candidate_mask, -1.0)
        order = torch.argsort(scores, dim=1, descending=True)
        sorted_pick = torch.arange(seq_len, device=candidate_mask.device).view(1, -1) < counts.view(-1, 1)
        selected = torch.zeros((batch_size, seq_len), device=candidate_mask.device, dtype=torch.bool)
        selected.scatter_(1, order, sorted_pick)
        return selected & candidate_mask

    def _focus_ablation_position_mask(
        self,
        *,
        focus_ablation_code: int,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        code = int(focus_ablation_code)
        if code <= 0:
            return torch.zeros_like(known_domain_mask, dtype=torch.bool)
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_target = known_domain_mask & (hist_domain_ids == target_domain)
        cross_target = known_domain_mask & (hist_domain_ids != target_domain)
        entry_switch, pre_switch, boundary = self._adjacent_domain_masks(
            known_domain_mask=known_domain_mask,
            hist_domain_ids=hist_domain_ids,
        )
        if code in {1, 6}:
            return entry_switch & same_target
        if code == 2:
            return entry_switch & cross_target
        if code == 3:
            return pre_switch & same_target
        if code == 4:
            return boundary & same_target
        if code == 5:
            entry_same_target = entry_switch & same_target
            candidates = known_domain_mask & (~entry_same_target)
            return self._deterministic_same_count_mask(
                candidate_mask=candidates,
                count_like_mask=entry_same_target,
                target_domain_ids=target_domain_ids,
                layer_idx=layer_idx,
            )
        return torch.zeros_like(known_domain_mask, dtype=torch.bool)

    def _apply_focus_ablation_bias(
        self,
        *,
        bias: Optional[torch.Tensor],
        focus_ablation_code: int,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        num_slots: int,
        dtype: torch.dtype,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        code = int(focus_ablation_code)
        if code <= 0:
            return bias
        position_mask = self._focus_ablation_position_mask(
            focus_ablation_code=code,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=target_domain_ids,
            layer_idx=layer_idx,
        )
        if int(position_mask.long().sum().item()) <= 0:
            return bias
        s_slots = self._transition_slot_mask(num_slots, hist_domain_ids.device)
        slot_mask = ~s_slots if code == 6 else s_slots
        if int(slot_mask.long().sum().item()) <= 0:
            return bias
        penalty_mask = slot_mask.view(1, int(num_slots), 1, 1) & position_mask.view(
            position_mask.shape[0],
            1,
            1,
            position_mask.shape[1],
        )
        penalty = torch.zeros(
            (position_mask.shape[0], int(num_slots), 1, position_mask.shape[1]),
            device=hist_domain_ids.device,
            dtype=dtype,
        )
        penalty = penalty.masked_fill(penalty_mask, -1.0e4)
        code_tensor = torch.full((position_mask.shape[0],), float(code), device=hist_domain_ids.device, dtype=dtype)
        return penalty if bias is None else bias + penalty

    def _transition_shift_sequence(
        self,
        *,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if self.q5_transition_mode == "none":
            return seq_tokens
        if self.q5_transition_mode == "window_attn":
            if int(layer_idx) >= len(self.q5_transition_window_attns):
                raise RuntimeError("Sliding-window transition attention modules are not initialized.")
            windowed = self.q5_transition_window_attns[int(layer_idx)](seq_tokens, hist_mask)
            delta_norm = (windowed - seq_tokens).norm(dim=-1)
            return windowed
        batch_size, seq_len, _ = seq_tokens.shape
        begin = self.q5_transition_begin[int(layer_idx)].to(dtype=seq_tokens.dtype, device=seq_tokens.device).view(1, 1, -1)
        end = self.q5_transition_end[int(layer_idx)].to(dtype=seq_tokens.dtype, device=seq_tokens.device).view(1, 1, -1)
        begin = begin.expand(batch_size, 1, -1)
        end = end.expand(batch_size, 1, -1)
        padded_tokens = torch.cat([begin, seq_tokens, end], dim=1)
        prev_tokens = padded_tokens[:, :-2, :]
        next_tokens = padded_tokens[:, 2:, :]

        prev_valid = torch.zeros_like(hist_mask, dtype=torch.bool)
        next_valid = torch.zeros_like(hist_mask, dtype=torch.bool)
        prev_cross = torch.zeros_like(hist_mask, dtype=torch.bool)
        next_cross = torch.zeros_like(hist_mask, dtype=torch.bool)
        prev_same = torch.zeros_like(hist_mask, dtype=torch.bool)
        next_same = torch.zeros_like(hist_mask, dtype=torch.bool)
        if seq_len > 1:
            adjacent_known = known_domain_mask[:, :-1] & known_domain_mask[:, 1:]
            adjacent_cross = adjacent_known & (hist_domain_ids[:, :-1] != hist_domain_ids[:, 1:])
            adjacent_same = adjacent_known & (hist_domain_ids[:, :-1] == hist_domain_ids[:, 1:])
            prev_valid[:, 1:] = hist_mask[:, :-1] & hist_mask[:, 1:]
            next_valid[:, :-1] = hist_mask[:, :-1] & hist_mask[:, 1:]
            prev_cross[:, 1:] = adjacent_cross
            next_cross[:, :-1] = adjacent_cross
            prev_same[:, 1:] = adjacent_same
            next_same[:, :-1] = adjacent_same

        if self.q5_transition_mode == "domain_subseq":
            num_domains = int(self.num_domains)
            positions = torch.arange(seq_len, device=seq_tokens.device, dtype=torch.long).view(1, seq_len, 1)
            domain_ids = hist_domain_ids.long().clamp(min=0, max=max(0, num_domains - 1))
            domain_one_hot = F.one_hot(domain_ids, num_classes=num_domains).to(dtype=torch.bool)
            domain_one_hot = domain_one_hot & known_domain_mask.unsqueeze(-1)

            prev_domain_positions = torch.where(domain_one_hot, positions, torch.full_like(positions, -1))
            prev_inclusive = torch.cummax(prev_domain_positions, dim=1).values
            prev_before = torch.cat([torch.full_like(prev_inclusive[:, :1, :], -1), prev_inclusive[:, :-1, :]], dim=1)
            prev_indices = prev_before.gather(2, domain_ids.unsqueeze(-1)).squeeze(-1)
            prev_use = known_domain_mask
            prev_has_same_domain = known_domain_mask & (prev_indices >= 0)
            prev_gather_indices = prev_indices.clamp(min=0, max=max(0, seq_len - 1)).unsqueeze(-1).expand(-1, -1, self.token_dim)
            prev_tokens = torch.gather(seq_tokens, 1, prev_gather_indices)
            prev_tokens = torch.where(prev_has_same_domain.unsqueeze(-1), prev_tokens, begin.expand(batch_size, seq_len, -1))

            next_domain_positions = torch.where(domain_one_hot, positions, torch.full_like(positions, seq_len))
            next_inclusive = torch.cummin(next_domain_positions.flip(1), dim=1).values.flip(1)
            next_after = torch.cat([next_inclusive[:, 1:, :], torch.full_like(next_inclusive[:, :1, :], seq_len)], dim=1)
            next_indices = next_after.gather(2, domain_ids.unsqueeze(-1)).squeeze(-1)
            next_use = known_domain_mask
            next_has_same_domain = known_domain_mask & (next_indices < seq_len)
            next_gather_indices = next_indices.clamp(min=0, max=max(0, seq_len - 1)).unsqueeze(-1).expand(-1, -1, self.token_dim)
            next_tokens = torch.gather(seq_tokens, 1, next_gather_indices)
            next_tokens = torch.where(next_has_same_domain.unsqueeze(-1), next_tokens, end.expand(batch_size, seq_len, -1))
        elif self.q5_transition_mode in {"all", "linear_n_swap", "n_query_ns"}:
            prev_use = prev_valid
            next_use = next_valid
        elif self.q5_transition_mode == "same":
            prev_use = prev_same
            next_use = next_same
        else:
            prev_use = prev_cross
            next_use = next_cross

        if self.q5_transition_mode == "boundary_flag":
            boundary_code = prev_cross.to(dtype=torch.long) + next_cross.to(dtype=torch.long) * 2
            flag_table = self.q5_transition_boundary_flag[int(layer_idx)].to(dtype=seq_tokens.dtype, device=seq_tokens.device)
            boundary_flag = F.embedding(boundary_code.clamp(min=0, max=3), flag_table)
            boundary_flag = torch.where(
                (boundary_code > 0).unsqueeze(-1),
                boundary_flag,
                torch.zeros_like(boundary_flag),
            )
            boundary_flag = boundary_flag * hist_mask.unsqueeze(-1).to(dtype=boundary_flag.dtype)
            shifted = seq_tokens + boundary_flag
        elif self.q5_transition_mode == "conv_cross_boundary":
            if int(layer_idx) >= len(self.q5_transition_conv_projs):
                raise RuntimeError("Transition local-conv modules are not initialized.")
            conv_mask = (prev_use | next_use) & hist_mask
            conv_input = torch.cat([prev_tokens, seq_tokens, next_tokens], dim=-1)
            conv_delta = self.q5_transition_conv_projs[int(layer_idx)](conv_input)
            conv_delta = conv_delta * conv_mask.unsqueeze(-1).to(dtype=conv_delta.dtype)
            conv_delta = conv_delta * float(self.q5_transition_conv_residual_scale)
            shifted = seq_tokens + conv_delta
        else:
            curr_chunks = torch.chunk(seq_tokens, 4, dim=-1)
            prev_chunks = torch.chunk(prev_tokens, 4, dim=-1)
            next_chunks = torch.chunk(next_tokens, 4, dim=-1)
            if self.q5_transition_mode == "domain_subseq":
                prev_mid = torch.where(prev_use.unsqueeze(-1), prev_chunks[2], curr_chunks[1])
                next_mid = torch.where(next_use.unsqueeze(-1), next_chunks[1], curr_chunks[2])
            else:
                prev_mid = torch.where(
                    prev_use.unsqueeze(-1),
                    prev_chunks[2],
                    torch.where(prev_valid.unsqueeze(-1), curr_chunks[1], prev_chunks[2]),
                )
                next_mid = torch.where(
                    next_use.unsqueeze(-1),
                    next_chunks[1],
                    torch.where(next_valid.unsqueeze(-1), curr_chunks[2], next_chunks[1]),
                )
            shifted = torch.cat([curr_chunks[0], prev_mid, next_mid, curr_chunks[3]], dim=-1)
        shifted = shifted * hist_mask.unsqueeze(-1).to(dtype=shifted.dtype)
        shift_delta = shifted - seq_tokens
        shift_delta_norm = shift_delta.norm(dim=-1)
        input_norm = seq_tokens.norm(dim=-1).clamp_min(1e-12)
        shifted_norm = shifted.norm(dim=-1).clamp_min(1e-12)
        shift_cos = (seq_tokens * shifted).sum(dim=-1) / (input_norm * shifted_norm)
        return shifted

    def _transition_contextbank_extend_attention_kv(
        self,
        *,
        cross_attention: CrossAttention,
        q_tokens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor],
        seq_tokens: torch.Tensor,
        q5_seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: Optional[torch.Tensor],
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict[str, torch.Tensor]]]:
        return k, v, bias, None

    def _transition_contextbank_apply_attention_output(
        self,
        *,
        attention_output: torch.Tensor,
        q_tokens: torch.Tensor,
        ctxbank_runtime: Optional[dict[str, torch.Tensor]],
        layer_idx: int,
    ) -> torch.Tensor:
        return attention_output

    def _cross_attention_with_q5_transition_view(
        self,
        *,
        cross_attention: CrossAttention,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
        key_modulation_condition: Optional[torch.Tensor],
        score_rerank_condition: Optional[torch.Tensor],
        query_modulation_domain_ids: Optional[torch.Tensor],
        focus_ablation_code: int,
        transition_eval_code: int,
        layer_idx: int,
        attention_same_mask: Optional[torch.Tensor] = None,
        attention_cross_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition_eval_code = int(transition_eval_code)
        train_layer_off = self._transition_train_layer_off(layer_idx)
        eval_layer_off = self._transition_eval_layer_off(transition_eval_code, layer_idx)
        layer_transition_off = train_layer_off or eval_layer_off
        if layer_transition_off:
            q5_seq_input = seq_tokens
            zero_mask_values = torch.zeros_like(hist_mask, dtype=seq_tokens.dtype)
            zero_delta = torch.zeros_like(seq_tokens)
            if eval_layer_off:
                pass
            if train_layer_off:
                pass
        else:
            q5_seq_input = self._transition_shift_sequence(
                seq_tokens=seq_tokens,
                hist_mask=hist_mask,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                layer_idx=layer_idx,
            )
        n_seq_input = self._linear_n_sequence(
            seq_tokens=seq_tokens,
            hist_mask=hist_mask,
            layer_idx=layer_idx,
        )
        n_seq_input = self._n_query_ns_sequence(
            seq_tokens=n_seq_input,
            ns_tokens=ns_tokens,
            hist_mask=hist_mask,
            layer_idx=layer_idx,
        )
        q5_seq_tokens = cross_attention.seq_ffn(cross_attention.seq_norm(q5_seq_input)) + q5_seq_input
        q5_seq_tokens = q5_seq_tokens * hist_mask.unsqueeze(-1).to(dtype=q5_seq_tokens.dtype)

        seq_tokens = cross_attention.seq_ffn(cross_attention.seq_norm(n_seq_input)) + n_seq_input
        seq_tokens = seq_tokens * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        n_pre_swap_seq_tokens = seq_tokens
        seq_tokens, q5_seq_tokens = self._linear_n_swap_sequences(
            n_seq_tokens=seq_tokens,
            s_seq_tokens=q5_seq_tokens,
            hist_mask=hist_mask,
            layer_idx=layer_idx,
        )
        q5_delta = q5_seq_tokens - seq_tokens

        key_tokens = seq_tokens
        q5_key_tokens = q5_seq_tokens
        if cross_attention.key_modulator is not None:
            if key_modulation_condition is None:
                raise ValueError("key_modulation_condition is required when key modulation is enabled.")
            key_tokens = cross_attention.key_modulator(seq_tokens, key_modulation_condition)
            q5_key_tokens = cross_attention.key_modulator(q5_seq_tokens, key_modulation_condition)
            key_tokens = key_tokens * hist_mask.unsqueeze(-1).to(dtype=key_tokens.dtype)
            q5_key_tokens = q5_key_tokens * hist_mask.unsqueeze(-1).to(dtype=q5_key_tokens.dtype)

        keys = list(cross_attention.k_proj(key_tokens).chunk(cross_attention.num_tokens, dim=-1))
        values = list(cross_attention.v_proj(seq_tokens).chunk(cross_attention.num_tokens, dim=-1))
        q5_keys = cross_attention.k_proj(q5_key_tokens).chunk(cross_attention.num_tokens, dim=-1)
        q5_values = cross_attention.v_proj(q5_seq_tokens).chunk(cross_attention.num_tokens, dim=-1)
        q5_index = int(self.base_num_ns_token) - 1
        original_k = torch.stack(keys, dim=1)
        original_v = torch.stack(values, dim=1)
        candidate_k = torch.stack(list(q5_keys), dim=1)
        candidate_v = torch.stack(list(q5_values), dim=1)
        slot_mask = self._transition_slot_mask(int(cross_attention.num_tokens), device=original_k.device)
        selected = slot_mask.view(1, -1, 1).to(dtype=original_k.dtype)
        selected_count = selected.sum(dim=1).clamp_min(1.0)
        candidate_key_delta = ((candidate_k - original_k).norm(dim=-1) * selected).sum(dim=1) / selected_count
        candidate_value_delta = ((candidate_v - original_v).norm(dim=-1) * selected).sum(dim=1) / selected_count
        replace_keys = self.q5_transition_kv_mode in {"kv", "k_only"}
        replace_values = self.q5_transition_kv_mode in {"kv", "v_only"}
        if transition_eval_code == 2:
            replace_keys = False
        elif transition_eval_code == 3:
            replace_values = False
        elif transition_eval_code == 4:
            replace_keys = False
            replace_values = False
        for slot_idx in range(int(cross_attention.num_tokens)):
            if not bool(slot_mask[slot_idx].item()):
                continue
            if replace_keys:
                keys[slot_idx] = q5_keys[slot_idx]
            if replace_values:
                values[slot_idx] = q5_values[slot_idx]
        k = torch.stack(keys, dim=1)
        v = torch.stack(values, dim=1)
        actual_key_delta = ((k - original_k).norm(dim=-1) * selected).sum(dim=1) / selected_count
        actual_value_delta = ((v - original_v).norm(dim=-1) * selected).sum(dim=1) / selected_count

        query_tokens = ns_tokens
        if cross_attention.query_modulator is not None:
            if query_modulation_domain_ids is None:
                raise ValueError("query_modulation_domain_ids is required when query modulation is enabled.")
            query_tokens = cross_attention.query_modulator(query_tokens, query_modulation_domain_ids)
        q = query_tokens.unsqueeze(2)
        bias = None
        if attention_bias is not None:
            bias = attention_bias.to(device=q.device, dtype=q.dtype)
            if bias.dim() == 2:
                bias = bias.unsqueeze(1).unsqueeze(2)
            elif bias.dim() == 3:
                bias = bias.unsqueeze(2)
        if cross_attention.score_reranker is not None:
            if score_rerank_condition is None:
                raise ValueError("score_rerank_condition is required when score rerank is enabled.")
            rerank_bias = cross_attention.score_reranker(seq_tokens, ns_tokens, score_rerank_condition, hist_mask)
            rerank_bias = rerank_bias.to(device=q.device, dtype=q.dtype).unsqueeze(2)
            bias = rerank_bias if bias is None else bias + rerank_bias
        if query_modulation_domain_ids is not None:
            bias = self._apply_focus_ablation_bias(
                bias=bias,
                focus_ablation_code=focus_ablation_code,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                target_domain_ids=query_modulation_domain_ids,
                num_slots=int(cross_attention.num_tokens),
                dtype=q.dtype,
                layer_idx=layer_idx,
            )
        k, v, bias, ctxbank_runtime = self._transition_contextbank_extend_attention_kv(
            cross_attention=cross_attention,
            q_tokens=query_tokens,
            k=k,
            v=v,
            bias=bias,
            seq_tokens=seq_tokens,
            q5_seq_tokens=q5_seq_tokens,
            hist_mask=hist_mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=query_modulation_domain_ids,
            layer_idx=layer_idx,
        )

        use_double_softmax = (
            self.attention_double_softmax_beta is not None
            and attention_same_mask is not None
            and attention_cross_mask is not None
            and attention_same_mask.shape[-1] == k.shape[-2]
            and attention_cross_mask.shape[-1] == k.shape[-2]
        )
        if use_double_softmax:
            logits = torch.matmul(q, k.transpose(-2, -1)) * (float(cross_attention.input_dim) ** -0.5)
            if bias is not None:
                logits = logits + bias
            attention_output, _, _, _ = _double_softmax_attention(
                logits=logits,
                values=v,
                same_mask=attention_same_mask,
                cross_mask=attention_cross_mask,
                beta=float(self.attention_double_softmax_beta),
                mode=self.attention_double_softmax_mode,
            )
        elif bias is None:
            attention_output = F.scaled_dot_product_attention(q, k, v).squeeze(2)
        else:
            attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).squeeze(2)
        attention_output = self._transition_contextbank_apply_attention_output(
            attention_output=attention_output,
            q_tokens=query_tokens,
            ctxbank_runtime=ctxbank_runtime,
            layer_idx=layer_idx,
        )
        ns_tokens = attention_output + ns_tokens
        persist_source_tokens = None
        if self.q5_transition_persist_source == "n_pre_swap":
            persist_source_tokens = n_pre_swap_seq_tokens
        elif self.q5_transition_persist_source == "n_swapped":
            persist_source_tokens = seq_tokens
        if transition_eval_code == 5:
            next_seq_tokens = seq_tokens
        else:
            next_seq_tokens = self._next_q5_transition_seq(
                seq_tokens=seq_tokens,
                q5_seq_tokens=q5_seq_tokens,
                persist_source_tokens=persist_source_tokens,
                hist_mask=hist_mask,
                layer_idx=layer_idx,
            )
        return next_seq_tokens, ns_tokens

    def _mixformer_with_q5_transition_view(
        self,
        *,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        query_modulation_domain_ids: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        focus_ablation_code: int = 0,
        transition_eval_code: int = 0,
        querymix_eval_code: int = 0,
    ) -> torch.Tensor:
        attention_same_mask, attention_cross_mask = self._attention_double_softmax_masks(
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=query_modulation_domain_ids,
        )
        for layer_idx, (query_mixer, cross_attention, output_fusion) in enumerate(zip(
            self.mixformer.query_mixers,
            self.mixformer.cross_attentions,
            self.mixformer.output_fusions,
        )):
            ns_tokens = self.mixformer._run_query_mixer_step(
                query_mixer,
                ns_tokens,
                querymix_eval_code=querymix_eval_code,
            )
            seq_tokens, ns_tokens = self._cross_attention_with_q5_transition_view(
                cross_attention=cross_attention,
                seq_tokens=seq_tokens,
                ns_tokens=ns_tokens,
                hist_mask=hist_mask,
                hist_domain_ids=hist_domain_ids,
                known_domain_mask=known_domain_mask,
                attention_bias=attention_bias,
                key_modulation_condition=None,
                score_rerank_condition=None,
                query_modulation_domain_ids=query_modulation_domain_ids,
                focus_ablation_code=focus_ablation_code,
                transition_eval_code=transition_eval_code,
                layer_idx=layer_idx,
                attention_same_mask=attention_same_mask,
                attention_cross_mask=attention_cross_mask,
            )
            ns_tokens = output_fusion(ns_tokens)
        return ns_tokens

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)
        if getattr(self, "q5_transition_off_layers", ()):
            off_layer_mask = torch.zeros(
                (seq_tokens.shape[0], int(len(self.mixformer.query_mixers))),
                device=seq_tokens.device,
                dtype=seq_tokens.dtype,
            )
            for layer_idx in self.q5_transition_off_layers:
                off_layer_mask[:, int(layer_idx)] = 1.0
        transition_eval_code = self._transition_eval_code(interaction)
        if self.q5_transition_mode == "linear_n_swap":
            swap_code = int(self.q5_transition_linear_swap_index)
        if self.q5_transition_mode == "n_query_ns":
            pass
        if self.q5_transition_mode == "conv_cross_boundary":
            pass
        attention_bias = self._attention_domain_bias(
            interaction,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            dtype=seq_tokens.dtype,
        )

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        tokenizer_q5_slot = ns_tokens[:, self.base_num_ns_token - 1, :]
        q5_init = self._make_cross_q5_token(
            interaction=interaction,
            seq_tokens=seq_tokens,
            hist_mask=mask,
            target_domain_ids=domain_ids,
        )
        if self._cross_q5_off(interaction):
            q5_init = torch.zeros_like(q5_init)
        else:
            pass
        ns_tokens = torch.cat([ns_tokens[:, : self.base_num_ns_token - 1, :], q5_init.unsqueeze(1)], dim=1)
        querymix_eval_code = self._querymix_eval_code(interaction)
        ns_tokens = self._mixformer_with_q5_transition_view(
            seq_tokens=seq_tokens,
            ns_tokens=ns_tokens,
            hist_mask=mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            query_modulation_domain_ids=domain_ids,
            attention_bias=attention_bias,
            focus_ablation_code=self._focus_ablation_code(interaction),
            transition_eval_code=transition_eval_code,
            querymix_eval_code=querymix_eval_code,
        )
        base_tokens = ns_tokens[:, : self.base_num_ns_token - 1, :]
        q5_final = ns_tokens[:, self.base_num_ns_token - 1, :]
        base_feature = self.dropout(base_tokens.mean(dim=1))
        q5_feature = q5_final
        cross_q5_off = self._cross_q5_off(interaction)
        if cross_q5_off:
            q5_feature = torch.zeros_like(q5_feature)
        readout_tokens = torch.cat([base_tokens, q5_feature.unsqueeze(1)], dim=1)
        features = self.dropout(
            self._pool_cross_q5_final_slots(
                interaction,
                readout_tokens,
                q5_active=not cross_q5_off,
            )
        )
        q5_residual = features - base_feature
        features = self._apply_cross_context_residual(
            features=features,
            seq_tokens=seq_tokens,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
        )
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class Q4ContextBankTransitionShiftQ5MixFormerModel(TransitionShiftQ5MixFormerModel):
    """V6 shift-persist trunk plus a compact q-slot ContextBank side path."""

    CTXBANK_OFF_FIELD = "__mixformer_q4_ctxbank_off__"
    CTXBANK_BOUNDARY_OFF_FIELD = "__mixformer_q4_ctxbank_boundary_off__"
    CTXBANK_CROSS_RECENT_OFF_FIELD = "__mixformer_q4_ctxbank_cross_recent_off__"
    CTXBANK_SHUFFLE_USER_FIELD = "__mixformer_q4_ctxbank_shuffle_user__"

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "target_only",
        q5_decision_scale: float = 1.0,
        q5_transition_mode: str = "all",
        q5_transition_slots: str = "tail2",
        q5_transition_kv_mode: str = "kv",
        q5_transition_persist: bool = True,
        q5_transition_persist_scale: float = 1.0,
        q5_transition_persist_source: str = "s",
        q5_transition_fuse_mode: str = "none",
        q5_transition_window_size: int = 3,
        q5_transition_window_num_heads: int = 2,
        q5_transition_window_output_init_std: Optional[float] = None,
        q5_transition_window_residual_scale: float = 1.0,
        q5_transition_linear_num_heads: int = 2,
        q5_transition_linear_output_init_std: Optional[float] = 1e-4,
        q5_transition_linear_residual_scale: float = 0.1,
        q5_transition_linear_swap_chunk: str = "same_position",
        q5_transition_nqns_num_heads: int = 2,
        q5_transition_nqns_output_init_std: Optional[float] = 1e-4,
        q5_transition_nqns_residual_scale: float = 0.1,
        q5_transition_boundary_flag_init_std: float = 1e-4,
        q5_transition_conv_output_init_std: Optional[float] = 0.0,
        q5_transition_conv_residual_scale: float = 1.0,
        q5_transition_off_layers: Optional[Iterable[int]] = None,
        q4_ctxbank_mode: str = "bcr",
        q4_ctxbank_layer: int = 2,
        q4_ctxbank_scale: float = 0.5,
        q4_ctxbank_cross_recent_k: int = 100,
        q4_ctxbank_boundary_pre_window: int = 3,
        q4_ctxbank_boundary_post_window: int = 3,
        q4_ctxbank_gate_zero: bool = False,
        q4_ctxbank_shuffle_user: bool = False,
        q4_ctxbank_raw_concat_control: bool = False,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_q5_mode=cross_q5_mode,
            q5_decision_scale=q5_decision_scale,
            q5_transition_mode=q5_transition_mode,
            q5_transition_slots=q5_transition_slots,
            q5_transition_kv_mode=q5_transition_kv_mode,
            q5_transition_persist=q5_transition_persist,
            q5_transition_persist_scale=q5_transition_persist_scale,
            q5_transition_persist_source=q5_transition_persist_source,
            q5_transition_fuse_mode=q5_transition_fuse_mode,
            q5_transition_window_size=q5_transition_window_size,
            q5_transition_window_num_heads=q5_transition_window_num_heads,
            q5_transition_window_output_init_std=q5_transition_window_output_init_std,
            q5_transition_window_residual_scale=q5_transition_window_residual_scale,
            q5_transition_linear_num_heads=q5_transition_linear_num_heads,
            q5_transition_linear_output_init_std=q5_transition_linear_output_init_std,
            q5_transition_linear_residual_scale=q5_transition_linear_residual_scale,
            q5_transition_linear_swap_chunk=q5_transition_linear_swap_chunk,
            q5_transition_nqns_num_heads=q5_transition_nqns_num_heads,
            q5_transition_nqns_output_init_std=q5_transition_nqns_output_init_std,
            q5_transition_nqns_residual_scale=q5_transition_nqns_residual_scale,
            q5_transition_boundary_flag_init_std=q5_transition_boundary_flag_init_std,
            q5_transition_conv_output_init_std=q5_transition_conv_output_init_std,
            q5_transition_conv_residual_scale=q5_transition_conv_residual_scale,
            q5_transition_off_layers=q5_transition_off_layers,
        )
        mode = str(q4_ctxbank_mode or "bcr").lower()
        mode_aliases = {
            "q4_bcr": "bcr",
            "bcr": "bcr",
            "boundary": "boundary",
            "q4_boundary": "boundary",
            "crossrecent": "crossrecent",
            "cross_recent": "crossrecent",
            "q4_crossrecent": "crossrecent",
            "q2_boundary_q4_cross": "q2_boundary_q4_cross",
        }
        if mode not in mode_aliases:
            raise ValueError(f"Unsupported q4_ctxbank_mode: {q4_ctxbank_mode!r}.")
        self.q4_ctxbank_mode = mode_aliases[mode]
        self.q4_ctxbank_layer = int(q4_ctxbank_layer)
        self.q4_ctxbank_scale = float(q4_ctxbank_scale)
        self.q4_ctxbank_cross_recent_k = int(q4_ctxbank_cross_recent_k)
        self.q4_ctxbank_boundary_pre_window = int(q4_ctxbank_boundary_pre_window)
        self.q4_ctxbank_boundary_post_window = int(q4_ctxbank_boundary_post_window)
        self.q4_ctxbank_gate_zero = bool(q4_ctxbank_gate_zero)
        self.q4_ctxbank_shuffle_user = bool(q4_ctxbank_shuffle_user)
        self.q4_ctxbank_raw_concat_control = bool(q4_ctxbank_raw_concat_control)
        self.q4_ctxbank_gate = nn.Linear(self.token_dim, 1)
        self.q4_ctxbank_ctx_norm = _make_norm(self.token_dim)
        nn.init.zeros_(self.q4_ctxbank_gate.weight)
        nn.init.zeros_(self.q4_ctxbank_gate.bias)

    @staticmethod
    def _flag_from_interaction(interaction, field: str) -> bool:
        if field not in interaction:
            return False
        value = interaction[field]
        return bool(isinstance(value, torch.Tensor) and value.detach().float().max().item() > 0.5)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        self._q4_ctxbank_off_active = self._flag_from_interaction(interaction, self.CTXBANK_OFF_FIELD)
        self._q4_ctxbank_boundary_off_active = self._flag_from_interaction(interaction, self.CTXBANK_BOUNDARY_OFF_FIELD)
        self._q4_ctxbank_cross_recent_off_active = self._flag_from_interaction(interaction, self.CTXBANK_CROSS_RECENT_OFF_FIELD)
        self._q4_ctxbank_shuffle_eval_active = self._flag_from_interaction(interaction, self.CTXBANK_SHUFFLE_USER_FIELD)
        try:
            return super().encode_features(interaction)
        finally:
            self._q4_ctxbank_off_active = False
            self._q4_ctxbank_boundary_off_active = False
            self._q4_ctxbank_cross_recent_off_active = False
            self._q4_ctxbank_shuffle_eval_active = False

    def _q4_ctxbank_enabled_for_layer(self, layer_idx: int) -> bool:
        if self.q4_ctxbank_layer < 0:
            return True
        return int(layer_idx) == int(self.q4_ctxbank_layer)

    @staticmethod
    def _ctxbank_masked_attention_pool(
        *,
        query: torch.Tensor,
        source_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.einsum("bd,bld->bl", query, source_tokens) * (float(source_tokens.shape[-1]) ** -0.5)
        logits = logits.masked_fill(~mask, -1.0e4)
        has_any = mask.any(dim=1)
        logits = torch.where(has_any.unsqueeze(-1), logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        weights = weights / denom
        pooled = torch.einsum("bl,bld->bd", weights, source_tokens)
        pooled = torch.where(has_any.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        return pooled, weights

    def _q4_ctxbank_masks(
        self,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = hist_domain_ids.shape
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_target = known_domain_mask & (hist_domain_ids == target_domain)
        cross_target = known_domain_mask & (hist_domain_ids != target_domain)
        prev_known = torch.zeros_like(known_domain_mask, dtype=torch.bool)
        prev_domain = hist_domain_ids
        if seq_len > 1:
            prev_known[:, 1:] = known_domain_mask[:, :-1]
            prev_domain = torch.cat([hist_domain_ids[:, :1], hist_domain_ids[:, :-1]], dim=1)
        segment_start = known_domain_mask & ((~prev_known) | (hist_domain_ids != prev_domain))
        target_entry = segment_start & same_target

        positions = torch.arange(seq_len, device=hist_domain_ids.device, dtype=torch.long)
        pos_as_token = positions.view(1, seq_len, 1)
        pos_as_entry = positions.view(1, 1, seq_len)
        dist_to_entry = pos_as_token - pos_as_entry
        boundary_mask = (
            target_entry.unsqueeze(1)
            & (dist_to_entry >= -int(self.q4_ctxbank_boundary_pre_window))
            & (dist_to_entry <= int(self.q4_ctxbank_boundary_post_window))
        ).any(dim=2)
        boundary_mask = boundary_mask & known_domain_mask

        if int(self.q4_ctxbank_cross_recent_k) > 0:
            rank_from_tail = cross_target.long().flip(1).cumsum(dim=1).flip(1)
            cross_recent_mask = cross_target & (rank_from_tail <= int(self.q4_ctxbank_cross_recent_k))
        else:
            cross_recent_mask = cross_target
        if bool(getattr(self, "_q4_ctxbank_boundary_off_active", False)):
            boundary_mask = torch.zeros_like(boundary_mask)
        if bool(getattr(self, "_q4_ctxbank_cross_recent_off_active", False)):
            cross_recent_mask = torch.zeros_like(cross_recent_mask)
        return boundary_mask, cross_recent_mask

    def _q4_ctxbank_make_bank(
        self,
        *,
        q_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        layer_idx: int,
    ) -> dict[str, torch.Tensor]:
        num_slots = int(q_tokens.shape[1])
        q4_index = min(max(0, num_slots - 1), num_slots - 1)
        q4_query = q_tokens[:, q4_index, :]
        boundary_mask, cross_recent_mask = self._q4_ctxbank_masks(
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=target_domain_ids,
        )
        boundary_ctx, boundary_pool_weights = self._ctxbank_masked_attention_pool(
            query=q4_query,
            source_tokens=source_tokens,
            mask=boundary_mask,
        )
        cross_recent_ctx, cross_recent_pool_weights = self._ctxbank_masked_attention_pool(
            query=q4_query,
            source_tokens=source_tokens,
            mask=cross_recent_mask,
        )
        bank = torch.stack([boundary_ctx, cross_recent_ctx], dim=1)
        bank_mask = torch.stack([boundary_mask.any(dim=1), cross_recent_mask.any(dim=1)], dim=1)

        shuffle_active = bool(self.q4_ctxbank_shuffle_user) or bool(getattr(self, "_q4_ctxbank_shuffle_eval_active", False))
        if shuffle_active and bank.shape[0] > 1:
            perm = torch.randperm(bank.shape[0], device=bank.device)
            bank = bank.index_select(0, perm)
            bank_mask = bank_mask.index_select(0, perm)
            boundary_ctx = boundary_ctx.index_select(0, perm)
            cross_recent_ctx = cross_recent_ctx.index_select(0, perm)

        return {
            "bank": bank,
            "bank_mask": bank_mask,
            "q4_index": torch.tensor(q4_index, device=q_tokens.device, dtype=torch.long),
        }

    def _transition_contextbank_extend_attention_kv(
        self,
        *,
        cross_attention: CrossAttention,
        q_tokens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor],
        seq_tokens: torch.Tensor,
        q5_seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: Optional[torch.Tensor],
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict[str, torch.Tensor]]]:
        if target_domain_ids is None or not self._q4_ctxbank_enabled_for_layer(layer_idx):
            return k, v, bias, None
        runtime = self._q4_ctxbank_make_bank(
            q_tokens=q_tokens,
            source_tokens=q5_seq_tokens,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=target_domain_ids,
            layer_idx=layer_idx,
        )
        if not bool(self.q4_ctxbank_raw_concat_control):
            return k, v, bias, runtime

        bank = runtime["bank"]
        bank_mask = runtime["bank_mask"]
        bank_keys = torch.stack(list(cross_attention.k_proj(bank).chunk(cross_attention.num_tokens, dim=-1)), dim=1)
        bank_values = torch.stack(list(cross_attention.v_proj(bank).chunk(cross_attention.num_tokens, dim=-1)), dim=1)
        hist_len = int(k.shape[-2])
        k = torch.cat([k, bank_keys], dim=2)
        v = torch.cat([v, bank_values], dim=2)
        extra_bias = torch.zeros(
            (bank.shape[0], int(cross_attention.num_tokens), 1, bank.shape[1]),
            device=bank.device,
            dtype=k.dtype,
        )
        extra_bias = extra_bias.masked_fill(~bank_mask.view(bank.shape[0], 1, 1, bank.shape[1]), -1.0e4)
        if bias is None:
            hist_bias = torch.zeros(
                (bank.shape[0], int(cross_attention.num_tokens), 1, hist_len),
                device=bank.device,
                dtype=k.dtype,
            )
            bias = torch.cat([hist_bias, extra_bias], dim=-1)
        else:
            if bias.shape[1] == 1 and int(cross_attention.num_tokens) > 1:
                bias = bias.expand(-1, int(cross_attention.num_tokens), -1, -1)
            bias = torch.cat([bias, extra_bias], dim=-1)
        runtime["rawconcat_hist_len"] = hist_len
        return k, v, bias, runtime

    def _slot_bank_mask(self, slot_idx: int, bank_mask: torch.Tensor) -> torch.Tensor:
        mode = str(self.q4_ctxbank_mode)
        if mode == "boundary":
            return bank_mask & torch.tensor([True, False], device=bank_mask.device).view(1, 2)
        if mode == "crossrecent":
            return bank_mask & torch.tensor([False, True], device=bank_mask.device).view(1, 2)
        if mode == "q2_boundary_q4_cross":
            if int(slot_idx) == 1:
                return bank_mask & torch.tensor([True, False], device=bank_mask.device).view(1, 2)
            return bank_mask & torch.tensor([False, True], device=bank_mask.device).view(1, 2)
        return bank_mask

    def _ctxbank_slots(self, num_slots: int) -> tuple[int, ...]:
        q4_idx = max(0, int(num_slots) - 1)
        if self.q4_ctxbank_mode == "q2_boundary_q4_cross" and num_slots >= 2:
            return (1, q4_idx)
        return (q4_idx,)

    def _transition_contextbank_apply_attention_output(
        self,
        *,
        attention_output: torch.Tensor,
        q_tokens: torch.Tensor,
        ctxbank_runtime: Optional[dict[str, torch.Tensor]],
        layer_idx: int,
    ) -> torch.Tensor:
        if not isinstance(ctxbank_runtime, dict) or bool(self.q4_ctxbank_raw_concat_control):
            return attention_output
        bank = ctxbank_runtime.get("bank")
        bank_mask = ctxbank_runtime.get("bank_mask")
        if not isinstance(bank, torch.Tensor) or not isinstance(bank_mask, torch.Tensor):
            return attention_output
        batch_size, num_slots, dim = attention_output.shape
        out = attention_output
        slot_mask_obs = torch.zeros((batch_size, num_slots), device=out.device, dtype=out.dtype)
        gate_obs = torch.zeros_like(slot_mask_obs)
        boundary_share_obs = torch.zeros_like(slot_mask_obs)
        cross_share_obs = torch.zeros_like(slot_mask_obs)
        entropy_obs = torch.zeros_like(slot_mask_obs)
        delta_norm_obs = torch.zeros_like(slot_mask_obs)
        delta_ratio_obs = torch.zeros_like(slot_mask_obs)
        before_after_cos_obs = torch.ones_like(slot_mask_obs)
        q_main_norm_obs = torch.zeros_like(slot_mask_obs)
        q_ctx_norm_obs = torch.zeros_like(slot_mask_obs)
        q_out_norm_obs = torch.zeros_like(slot_mask_obs)
        off_active = bool(getattr(self, "_q4_ctxbank_off_active", False))

        for slot_idx in self._ctxbank_slots(num_slots):
            slot_idx = int(slot_idx)
            slot_mask_obs[:, slot_idx] = 1.0
            q_slot = q_tokens[:, slot_idx, :]
            slot_bank_mask = self._slot_bank_mask(slot_idx, bank_mask)
            logits = torch.einsum("bd,bkd->bk", q_slot, bank) * (float(dim) ** -0.5)
            logits = logits.masked_fill(~slot_bank_mask, -1.0e4)
            has_bank = slot_bank_mask.any(dim=1)
            logits = torch.where(has_bank.unsqueeze(-1), logits, torch.zeros_like(logits))
            weights = torch.softmax(logits, dim=-1)
            weights = torch.where(slot_bank_mask, weights, torch.zeros_like(weights))
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            weights = weights / denom
            ctx = torch.einsum("bk,bkd->bd", weights, bank)
            ctx = torch.where(has_bank.unsqueeze(-1), ctx, torch.zeros_like(ctx))
            ctx = self.q4_ctxbank_ctx_norm(ctx)
            raw_gate = torch.sigmoid(self.q4_ctxbank_gate(q_slot)).squeeze(-1)
            if self.q4_ctxbank_gate_zero or off_active:
                gate = torch.zeros_like(raw_gate)
            else:
                gate = raw_gate * has_bank.to(dtype=raw_gate.dtype)
            delta = float(self.q4_ctxbank_scale) * gate.unsqueeze(-1) * ctx
            before = out[:, slot_idx, :]
            after = before + delta
            out = out.clone()
            out[:, slot_idx, :] = after

            before_norm = before.norm(dim=-1).clamp_min(1.0e-12)
            after_norm = after.norm(dim=-1).clamp_min(1.0e-12)
            delta_norm = delta.norm(dim=-1)
            gate_obs[:, slot_idx] = gate
            boundary_share_obs[:, slot_idx] = weights[:, 0]
            cross_share_obs[:, slot_idx] = weights[:, 1]
            entropy_obs[:, slot_idx] = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1)
            delta_norm_obs[:, slot_idx] = delta_norm
            delta_ratio_obs[:, slot_idx] = delta_norm / before_norm
            before_after_cos_obs[:, slot_idx] = (before * after).sum(dim=-1) / (before_norm * after_norm)
            q_main_norm_obs[:, slot_idx] = before_norm
            q_ctx_norm_obs[:, slot_idx] = (float(self.q4_ctxbank_scale) * gate.unsqueeze(-1) * ctx).norm(dim=-1)
            q_out_norm_obs[:, slot_idx] = after.norm(dim=-1)

        return out


class MultiStreamExchangeMixFormerModel(MixFormerModel):
    """Four-stream MixFormer with shared pre-FFN chunk exchange.

    Streams are homogeneous and share the original MixFormer sequence FFN/K/V
    parameters.  Only the target stream is read by the query tokens; P/S/G can
    affect the prediction only through exchange into the target stream.
    """

    EXCHANGE_OFF_FIELD = "__mixformer_ms4_exchange_off__"
    P_STREAM_ZERO_FIELD = "__mixformer_ms4_p_stream_zero__"
    S_STREAM_ZERO_FIELD = "__mixformer_ms4_s_stream_zero__"
    G_STREAM_ZERO_FIELD = "__mixformer_ms4_g_stream_zero__"
    TARGET_WINDOW_OFF_FIELD = "__mixformer_ms4_target_window_off_code__"

    STREAM_INDEX = {"T": 0, "P": 1, "S": 2, "G": 3}
    STREAM_NAMES = ("T", "P", "S", "G")
    TARGET_WINDOW_OFF_CODES = {
        "none": 0,
        "target_entry_w1_off": 1,
        "target_entry_w2_off": 2,
        "target_entry_w3_off": 3,
        "target_segment_after_entry_off": 4,
    }

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        ms4_exchange_schedule: str = "none",
        ms4_exchange_persist: bool = False,
        ms4_prev_mode: str = "segment",
        ms4_source_mode: str = "tail",
        ms4_source_tail_k: int = 100,
    ):
        if int(token_dim) % 4 != 0:
            raise ValueError("MultiStreamExchangeMixFormerModel requires token_dim divisible by 4.")
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
        )
        schedule = str(ms4_exchange_schedule or "none").lower()
        schedule = schedule.replace("-", "_")
        aliases = {
            "off": "none",
            "no": "none",
            "no_exchange": "none",
            "same_tp_sg": "tp_sg",
            "same_ts_pg": "ts_pg",
            "same_tg_ps": "tg_ps",
            "round_robin": "rr",
            "target_round_robin": "rr",
            "source_first": "sourcefirst",
            "boundary_first": "boundaryfirst",
        }
        schedule = aliases.get(schedule, schedule)
        allowed = {"none", "tp_sg", "ts_pg", "tg_ps", "rr", "sourcefirst", "boundaryfirst"}
        if schedule not in allowed:
            raise ValueError(f"ms4_exchange_schedule must be one of {sorted(allowed)}, got {ms4_exchange_schedule!r}.")
        prev_mode = str(ms4_prev_mode or "segment").lower().replace("-", "_")
        prev_mode = {
            "prev_segment": "segment",
            "p_segment": "segment",
            "last": "last1",
            "prev_last1": "last1",
            "p_last1": "last1",
            "zero": "off",
            "none": "off",
        }.get(prev_mode, prev_mode)
        allowed_prev_modes = {"segment", "last1", "off"}
        if prev_mode not in allowed_prev_modes:
            raise ValueError(f"ms4_prev_mode must be one of {sorted(allowed_prev_modes)}, got {ms4_prev_mode!r}.")
        source_mode = str(ms4_source_mode or "tail").lower().replace("-", "_")
        source_mode = {
            "source_tail": "tail",
            "s_tail": "tail",
            "balanced": "balanced_tail",
            "domain_balanced": "balanced_tail",
            "source_balanced": "balanced_tail",
            "zero": "off",
            "none": "off",
        }.get(source_mode, source_mode)
        allowed_source_modes = {"tail", "balanced_tail", "off"}
        if source_mode not in allowed_source_modes:
            raise ValueError(f"ms4_source_mode must be one of {sorted(allowed_source_modes)}, got {ms4_source_mode!r}.")
        self.domain_seq_field = str(domain_seq_field)
        self.ms4_exchange_schedule = schedule
        self.ms4_exchange_persist = bool(ms4_exchange_persist)
        self.ms4_prev_mode = prev_mode
        self.ms4_source_mode = source_mode
        self.ms4_source_tail_k = max(0, int(ms4_source_tail_k))

    @classmethod
    def target_window_off_code_for_variant(cls, variant: str) -> int:
        return int(cls.TARGET_WINDOW_OFF_CODES[str(variant)])

    def _history_domain_ids(
        self,
        interaction,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_field = self.domain_seq_field if self.domain_seq_field in interaction else "tab_seq"
        if source_field not in interaction:
            empty_ids = torch.zeros_like(hist_mask, dtype=torch.long)
            return empty_ids, torch.zeros_like(hist_mask, dtype=torch.bool)
        raw_domain = interaction[source_field].long().to(device=hist_mask.device)
        if raw_domain.shape[:2] != hist_mask.shape:
            raise ValueError(
                f"{source_field} shape {tuple(raw_domain.shape)} does not match history mask {tuple(hist_mask.shape)}"
            )
        if self.domain_shift > 0:
            known = hist_mask & (raw_domain > 0)
        else:
            known = hist_mask & (raw_domain >= 0)
        domain_ids = (raw_domain - int(self.domain_shift)).clamp(min=0, max=int(self.num_domains) - 1)
        return domain_ids, known

    def _flag_enabled(self, interaction, field_name: str) -> bool:
        if field_name not in interaction:
            return False
        flag = interaction[field_name]
        if not isinstance(flag, torch.Tensor) or flag.numel() <= 0:
            return False
        return bool(flag.detach().float().max().item() > 0.5)

    def _target_window_off_code(self, interaction) -> int:
        if self.TARGET_WINDOW_OFF_FIELD not in interaction:
            return 0
        code = interaction[self.TARGET_WINDOW_OFF_FIELD]
        if not isinstance(code, torch.Tensor) or code.numel() <= 0:
            return 0
        return int(code.detach().long().max().item())

    def _exchange_off(self, interaction) -> bool:
        return self.ms4_exchange_schedule == "none" or self._flag_enabled(interaction, self.EXCHANGE_OFF_FIELD)

    @staticmethod
    def _pack_tokens_by_mask(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, token_dim = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).view(1, seq_len).expand(batch_size, -1)
        sort_key = torch.where(mask, positions, positions + seq_len)
        order = torch.argsort(sort_key, dim=1)
        packed = torch.gather(tokens, 1, order.unsqueeze(-1).expand(-1, -1, token_dim))
        counts = mask.long().sum(dim=1).clamp(min=0, max=seq_len)
        packed_mask = torch.arange(seq_len, device=tokens.device).view(1, seq_len) < counts.view(-1, 1)
        packed = packed * packed_mask.unsqueeze(-1).to(dtype=packed.dtype)
        return packed, packed_mask

    @staticmethod
    def _previous_source_segment_mask(
        *,
        same_target: torch.Tensor,
        source: torch.Tensor,
        known: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, width = same_target.shape
        device = same_target.device
        positions = torch.arange(width, device=device).view(1, width).expand(batch_size, -1)
        no_pos = torch.full_like(positions, -1)
        safe_last = (seq_len.view(-1).long() - 1).clamp(min=0, max=max(0, width - 1))
        has_history = seq_len.view(-1).long() > 0
        tail_same = same_target.gather(1, safe_last.view(-1, 1)).squeeze(1) & has_history

        before_tail = positions <= safe_last.view(-1, 1)
        last_non_target_before_tail = torch.where(known & (~same_target) & before_tail, positions, no_pos).max(dim=1).values
        target_run_start = last_non_target_before_tail + 1
        p_end_from_tail = target_run_start - 1

        last_source = torch.where(source, positions, no_pos).max(dim=1).values
        p_end = torch.where(tail_same, p_end_from_tail, last_source)
        has_p_end = p_end >= 0
        before_p_end = positions <= p_end.view(-1, 1)
        last_non_source = torch.where(known & (~source) & before_p_end, positions, no_pos).max(dim=1).values
        p_start = last_non_source + 1
        return source & has_p_end.view(-1, 1) & (positions >= p_start.view(-1, 1)) & before_p_end

    @staticmethod
    def _latest_k_mask(mask: torch.Tensor, k: int) -> torch.Tensor:
        k = int(k)
        if k <= 0:
            return torch.zeros_like(mask, dtype=torch.bool)
        if k >= mask.shape[1]:
            return mask
        rev_rank = torch.flip(torch.cumsum(torch.flip(mask.long(), dims=[1]), dim=1), dims=[1])
        return mask & (rev_rank <= k)

    def _domain_balanced_source_tail_mask(
        self,
        *,
        source: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        k = int(k)
        if k <= 0:
            return torch.zeros_like(source, dtype=torch.bool)
        if int(self.num_domains) <= 1:
            return self._latest_k_mask(source, k)
        per_domain_k = max(1, math.ceil(k / max(1, int(self.num_domains) - 1)))
        balanced = torch.zeros_like(source, dtype=torch.bool)
        for domain_idx in range(int(self.num_domains)):
            domain_source = source & (hist_domain_ids == int(domain_idx))
            balanced = balanced | self._latest_k_mask(domain_source, per_domain_k)
        return balanced

    @staticmethod
    def _latest_target_segment_mask(
        *,
        same_target: torch.Tensor,
        known: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, width = same_target.shape
        device = same_target.device
        positions = torch.arange(width, device=device).view(1, width).expand(batch_size, -1)
        no_pos = torch.full_like(positions, -1)
        latest_same = torch.where(same_target, positions, no_pos).max(dim=1).values
        has_same = latest_same >= 0
        before_latest = positions <= latest_same.view(-1, 1)
        last_non_same = torch.where(known & (~same_target) & before_latest, positions, no_pos).max(dim=1).values
        run_start = last_non_same + 1
        segment = (
            has_same.view(-1, 1)
            & (positions >= run_start.view(-1, 1))
            & (positions <= latest_same.view(-1, 1))
            & same_target
        )
        return segment, run_start, latest_same

    def _target_window_ablation_mask(
        self,
        *,
        code: int,
        same_target: torch.Tensor,
        known: torch.Tensor,
    ) -> torch.Tensor:
        code = int(code)
        if code <= 0:
            return torch.zeros_like(same_target, dtype=torch.bool)
        segment, run_start, run_end = self._latest_target_segment_mask(same_target=same_target, known=known)
        positions = torch.arange(same_target.shape[1], device=same_target.device).view(1, -1)
        if code in {1, 2, 3}:
            return segment & (positions < (run_start.view(-1, 1) + code)) & (positions <= run_end.view(-1, 1))
        if code == 4:
            return segment
        return torch.zeros_like(same_target, dtype=torch.bool)

    @staticmethod
    def _masked_stream_mean(streams: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        weights = masks.unsqueeze(-1).to(dtype=streams.dtype)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (streams * weights).sum(dim=2) / denom


    def _build_streams(
        self,
        *,
        interaction,
        seq_tokens: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        same_target = known_domain_mask & (hist_domain_ids == target_domain)
        source = known_domain_mask & (hist_domain_ids != target_domain)
        prev_segment = self._previous_source_segment_mask(
            same_target=same_target,
            source=source,
            known=known_domain_mask,
            seq_len=seq_len,
        )
        raw_source = source
        raw_prev_segment = prev_segment
        if self.ms4_prev_mode == "last1":
            prev_segment = self._latest_k_mask(prev_segment, 1)
        elif self.ms4_prev_mode == "off":
            prev_segment = torch.zeros_like(prev_segment)
        if self.ms4_source_mode == "tail":
            source = self._latest_k_mask(source, self.ms4_source_tail_k)
        elif self.ms4_source_mode == "balanced_tail":
            source = self._domain_balanced_source_tail_mask(
                source=source,
                hist_domain_ids=hist_domain_ids,
                k=self.ms4_source_tail_k,
            )
        elif self.ms4_source_mode == "off":
            source = torch.zeros_like(source)
        global_recent = known_domain_mask

        off_code = self._target_window_off_code(interaction)
        target_off_mask = self._target_window_ablation_mask(
            code=off_code,
            same_target=same_target,
            known=known_domain_mask,
        )
        same_target = same_target & (~target_off_mask)
        if self._flag_enabled(interaction, self.P_STREAM_ZERO_FIELD):
            prev_segment = torch.zeros_like(prev_segment)
        if self._flag_enabled(interaction, self.S_STREAM_ZERO_FIELD):
            source = torch.zeros_like(source)
        if self._flag_enabled(interaction, self.G_STREAM_ZERO_FIELD):
            global_recent = torch.zeros_like(global_recent)

        stream_source_masks = [same_target, prev_segment, source, global_recent]
        packed_tokens = []
        packed_masks = []
        for stream_mask in stream_source_masks:
            stream_tokens, packed_mask = self._pack_tokens_by_mask(seq_tokens, stream_mask)
            packed_tokens.append(stream_tokens)
            packed_masks.append(packed_mask)
        streams = torch.stack(packed_tokens, dim=1)
        masks = torch.stack(packed_masks, dim=1)
        prev_mode_code = {"off": 0.0, "segment": 1.0, "last1": 2.0}[self.ms4_prev_mode]
        source_mode_code = {"off": 0.0, "tail": 1.0, "balanced_tail": 2.0}[self.ms4_source_mode]
        return streams, masks

    def _schedule_group(self, layer_idx: int) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
        tp_sg = ((0, 1), (2, 3))
        ts_pg = ((0, 2), (1, 3))
        tg_ps = ((0, 3), (1, 2))
        code_by_pairs = {tp_sg: 1, ts_pg: 2, tg_ps: 3}
        if self.ms4_exchange_schedule == "tp_sg":
            pairs = tp_sg
        elif self.ms4_exchange_schedule == "ts_pg":
            pairs = ts_pg
        elif self.ms4_exchange_schedule == "tg_ps":
            pairs = tg_ps
        elif self.ms4_exchange_schedule == "rr":
            pairs = (tp_sg, ts_pg, tg_ps)[int(layer_idx) % 3]
        elif self.ms4_exchange_schedule == "sourcefirst":
            pairs = (tg_ps, tp_sg, ts_pg)[int(layer_idx) % 3]
        elif self.ms4_exchange_schedule == "boundaryfirst":
            pairs = (tp_sg, tg_ps, ts_pg)[int(layer_idx) % 3]
        else:
            pairs = tp_sg
        return int(code_by_pairs[pairs]), pairs

    def _exchange_streams(
        self,
        *,
        streams: torch.Tensor,
        masks: torch.Tensor,
        layer_idx: int,
        exchange_off: bool,
    ) -> torch.Tensor:
        if exchange_off:
            return streams
        exchanged = streams.clone()
        _, pairs = self._schedule_group(layer_idx)
        for left_idx, right_idx in pairs:
            left = streams[:, left_idx, :, :]
            right = streams[:, right_idx, :, :]
            left_mask = masks[:, left_idx, :].unsqueeze(-1)
            right_mask = masks[:, right_idx, :].unsqueeze(-1)
            l0, l1, l2, l3 = torch.chunk(left, 4, dim=-1)
            r0, r1, r2, r3 = torch.chunk(right, 4, dim=-1)
            l1_new = torch.where(right_mask, r1, l1)
            l2_new = torch.where(right_mask, r2, l2)
            r1_new = torch.where(left_mask, l1, r1)
            r2_new = torch.where(left_mask, l2, r2)
            exchanged[:, left_idx, :, :] = torch.cat([l0, l1_new, l2_new, l3], dim=-1)
            exchanged[:, right_idx, :, :] = torch.cat([r0, r1_new, r2_new, r3], dim=-1)
        exchanged = exchanged * masks.unsqueeze(-1).to(dtype=exchanged.dtype)
        return exchanged

    @staticmethod
    def _stream_seq_ffn(
        *,
        cross_attention: CrossAttention,
        streams: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_streams, seq_len, token_dim = streams.shape
        flat = streams.reshape(batch_size * num_streams, seq_len, token_dim)
        flat_mask = masks.reshape(batch_size * num_streams, seq_len)
        flat = cross_attention.seq_ffn(cross_attention.seq_norm(flat)) + flat
        flat = flat * flat_mask.unsqueeze(-1).to(dtype=flat.dtype)
        return flat.reshape(batch_size, num_streams, seq_len, token_dim)

    def _cross_attention_from_target_stream(
        self,
        *,
        cross_attention: CrossAttention,
        target_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        target_mask: torch.Tensor,
        query_modulation_domain_ids: Optional[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        key_tokens = target_tokens
        if cross_attention.key_modulator is not None:
            raise ValueError("MultiStreamExchangeMixFormerModel does not support key_modulator yet.")

        keys = cross_attention.k_proj(key_tokens).chunk(cross_attention.num_tokens, dim=-1)
        values = cross_attention.v_proj(target_tokens).chunk(cross_attention.num_tokens, dim=-1)
        k = torch.stack(keys, dim=1)
        v = torch.stack(values, dim=1)
        query_tokens = ns_tokens
        if cross_attention.query_modulator is not None:
            if query_modulation_domain_ids is None:
                raise ValueError("query_modulation_domain_ids is required when query modulation is enabled.")
            query_tokens = cross_attention.query_modulator(query_tokens, query_modulation_domain_ids)
        q = query_tokens.unsqueeze(2)

        bias = torch.zeros(
            (target_tokens.shape[0], 1, 1, target_tokens.shape[1]),
            device=target_tokens.device,
            dtype=q.dtype,
        )
        bias = bias.masked_fill(~target_mask.view(target_tokens.shape[0], 1, 1, target_tokens.shape[1]), -1.0e4)
        if cross_attention.score_reranker is not None:
            raise ValueError("MultiStreamExchangeMixFormerModel does not support score_reranker yet.")
        attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).squeeze(2)
        return attention_output + ns_tokens

    def _mixformer_with_multistream_exchange(
        self,
        *,
        streams: torch.Tensor,
        stream_masks: torch.Tensor,
        ns_tokens: torch.Tensor,
        query_modulation_domain_ids: torch.Tensor,
        exchange_off: bool,
        querymix_eval_code: int = 0,
    ) -> torch.Tensor:
        batch_size = streams.shape[0]
        persist_value = 1.0 if self.ms4_exchange_persist and not exchange_off else 0.0
        for layer_idx, (query_mixer, cross_attention, output_fusion) in enumerate(zip(
            self.mixformer.query_mixers,
            self.mixformer.cross_attentions,
            self.mixformer.output_fusions,
        )):
            ns_tokens = self.mixformer._run_query_mixer_step(
                query_mixer,
                ns_tokens,
                querymix_eval_code=querymix_eval_code,
            )

            group_code, _ = self._schedule_group(layer_idx)
            if exchange_off:
                next_streams = self._stream_seq_ffn(
                    cross_attention=cross_attention,
                    streams=streams,
                    masks=stream_masks,
                )
                attention_streams = next_streams
                exchange_delta = torch.zeros_like(next_streams)
            else:
                exchange_view = self._exchange_streams(
                    streams=streams,
                    masks=stream_masks,
                    layer_idx=layer_idx,
                    exchange_off=False,
                )
                exchanged_after = self._stream_seq_ffn(
                    cross_attention=cross_attention,
                    streams=exchange_view,
                    masks=stream_masks,
                )
                normal_after = self._stream_seq_ffn(
                    cross_attention=cross_attention,
                    streams=streams,
                    masks=stream_masks,
                )
                if self.ms4_exchange_persist:
                    next_streams = exchanged_after
                    attention_streams = next_streams
                else:
                    next_streams = normal_after
                    attention_streams = exchanged_after
                exchange_delta = exchanged_after - normal_after
            target_tokens = attention_streams[:, 0, :, :]
            target_mask = stream_masks[:, 0, :]
            ns_tokens = self._cross_attention_from_target_stream(
                cross_attention=cross_attention,
                target_tokens=target_tokens,
                ns_tokens=ns_tokens,
                target_mask=target_mask,
                query_modulation_domain_ids=query_modulation_domain_ids,
                layer_idx=layer_idx,
            )
            streams = next_streams
            ns_tokens = output_fusion(ns_tokens)
        return ns_tokens

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)
        streams, stream_masks = self._build_streams(
            interaction=interaction,
            seq_tokens=seq_tokens,
            hist_mask=mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            seq_len=seq_len,
        )

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        querymix_eval_code = self._querymix_eval_code(interaction)
        ns_tokens = self._mixformer_with_multistream_exchange(
            streams=streams,
            stream_masks=stream_masks,
            ns_tokens=ns_tokens,
            query_modulation_domain_ids=domain_ids,
            exchange_off=self._exchange_off(interaction),
            querymix_eval_code=querymix_eval_code,
        )
        features = self.dropout(ns_tokens.mean(dim=1))
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class SlotStreamExchangeMixFormerModel(MixFormerModel):
    """Full-history MixFormer with same-position exchange between the 4 native slots."""

    EXCHANGE_OFF_FIELD = "__mixformer_slotx_exchange_off__"

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        slotx_exchange_schedule: str = "none",
        slotx_exchange_persist: bool = False,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
        )
        if int(num_ns_token) != 4:
            raise ValueError("SlotStreamExchangeMixFormerModel currently expects num_ns_token=4.")
        if int(token_dim) % 4 != 0:
            raise ValueError("SlotStreamExchangeMixFormerModel requires token_dim divisible by 4.")
        schedule = str(slotx_exchange_schedule or "none").lower()
        aliases = {
            "no_exchange": "none",
            "off": "none",
            "tp_sg": "01_23",
            "ts_pg": "02_13",
            "tg_ps": "03_12",
        }
        schedule = aliases.get(schedule, schedule)
        allowed = {"none", "01_23", "02_13", "03_12"}
        if schedule not in allowed:
            raise ValueError(f"slotx_exchange_schedule must be one of {sorted(allowed)}, got {slotx_exchange_schedule!r}.")
        self.slotx_exchange_schedule = schedule
        self.slotx_exchange_persist = bool(slotx_exchange_persist)
        self.slotx_stream_bias = nn.Parameter(torch.zeros(int(num_ns_token), self.token_dim))

    def _slotx_flag_enabled(self, interaction, field_name: str) -> bool:
        if field_name not in interaction:
            return False
        flag = interaction[field_name]
        if not isinstance(flag, torch.Tensor) or flag.numel() <= 0:
            return False
        return bool(flag.detach().float().max().item() > 0.5)

    def _slotx_exchange_off(self, interaction) -> bool:
        return self.slotx_exchange_schedule == "none" or self._slotx_flag_enabled(interaction, self.EXCHANGE_OFF_FIELD)

    def _slotx_schedule_group(self) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
        pairs_01_23 = ((0, 1), (2, 3))
        pairs_02_13 = ((0, 2), (1, 3))
        pairs_03_12 = ((0, 3), (1, 2))
        if self.slotx_exchange_schedule == "01_23":
            return 1, pairs_01_23
        if self.slotx_exchange_schedule == "02_13":
            return 2, pairs_02_13
        if self.slotx_exchange_schedule == "03_12":
            return 3, pairs_03_12
        return 0, pairs_01_23

    @staticmethod
    def _slotx_exchange_streams(
        *,
        streams: torch.Tensor,
        mask: torch.Tensor,
        pairs: tuple[tuple[int, int], tuple[int, int]],
    ) -> torch.Tensor:
        exchanged = streams.clone()
        token_mask = mask.unsqueeze(-1)
        for left_idx, right_idx in pairs:
            left = streams[:, left_idx, :, :]
            right = streams[:, right_idx, :, :]
            l0, l1, l2, l3 = torch.chunk(left, 4, dim=-1)
            r0, r1, r2, r3 = torch.chunk(right, 4, dim=-1)
            valid = token_mask.to(dtype=torch.bool)
            exchanged[:, left_idx, :, :] = torch.cat(
                [l0, torch.where(valid, r1, l1), torch.where(valid, r2, l2), l3],
                dim=-1,
            )
            exchanged[:, right_idx, :, :] = torch.cat(
                [r0, torch.where(valid, l1, r1), torch.where(valid, l2, r2), r3],
                dim=-1,
            )
        return exchanged * mask.unsqueeze(1).unsqueeze(-1).to(dtype=exchanged.dtype)

    @staticmethod
    def _slotx_stream_ffn(
        *,
        cross_attention: CrossAttention,
        streams: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_slots, seq_len, token_dim = streams.shape
        flat = streams.reshape(batch_size * num_slots, seq_len, token_dim)
        flat_mask = mask.unsqueeze(1).expand(batch_size, num_slots, seq_len).reshape(batch_size * num_slots, seq_len)
        flat = cross_attention.seq_ffn(cross_attention.seq_norm(flat)) + flat
        flat = flat * flat_mask.unsqueeze(-1).to(dtype=flat.dtype)
        return flat.reshape(batch_size, num_slots, seq_len, token_dim)

    @staticmethod
    def _slotx_single_ffn(
        *,
        cross_attention: CrossAttention,
        seq_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_tokens = cross_attention.seq_ffn(cross_attention.seq_norm(seq_tokens)) + seq_tokens
        return seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

    def _slotx_cross_attention(
        self,
        *,
        cross_attention: CrossAttention,
        slot_streams: torch.Tensor,
        ns_tokens: torch.Tensor,
        mask: torch.Tensor,
        query_modulation_domain_ids: Optional[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        if cross_attention.key_modulator is not None:
            raise ValueError("SlotStreamExchangeMixFormerModel does not support key_modulator yet.")
        if cross_attention.score_reranker is not None:
            raise ValueError("SlotStreamExchangeMixFormerModel does not support score_reranker yet.")
        num_slots = int(cross_attention.num_tokens)
        key_streams = slot_streams
        value_streams = slot_streams
        keys = []
        values = []
        for slot_idx in range(num_slots):
            slot_key_chunks = cross_attention.k_proj(key_streams[:, slot_idx, :, :]).chunk(num_slots, dim=-1)
            slot_value_chunks = cross_attention.v_proj(value_streams[:, slot_idx, :, :]).chunk(num_slots, dim=-1)
            keys.append(slot_key_chunks[slot_idx])
            values.append(slot_value_chunks[slot_idx])
        k = torch.stack(keys, dim=1)
        v = torch.stack(values, dim=1)

        query_tokens = ns_tokens
        if cross_attention.query_modulator is not None:
            if query_modulation_domain_ids is None:
                raise ValueError("query_modulation_domain_ids is required when query modulation is enabled.")
            query_tokens = cross_attention.query_modulator(query_tokens, query_modulation_domain_ids)
        q = query_tokens.unsqueeze(2)

        bias = torch.zeros(
            (slot_streams.shape[0], 1, 1, slot_streams.shape[2]),
            device=slot_streams.device,
            dtype=q.dtype,
        )
        bias = bias.masked_fill(~mask.view(slot_streams.shape[0], 1, 1, slot_streams.shape[2]), -1.0e4)
        attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).squeeze(2)
        return attention_output + ns_tokens

    def _mixformer_with_slot_exchange(
        self,
        *,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        mask: torch.Tensor,
        query_modulation_domain_ids: torch.Tensor,
        exchange_off: bool,
        querymix_eval_code: int = 0,
    ) -> torch.Tensor:
        batch_size = seq_tokens.shape[0]
        slot_bias = self.slotx_stream_bias.to(device=seq_tokens.device, dtype=seq_tokens.dtype).view(
            1,
            int(self.num_ns_token),
            1,
            self.token_dim,
        )
        slot_streams = seq_tokens.unsqueeze(1) + slot_bias
        slot_streams = slot_streams * mask.unsqueeze(1).unsqueeze(-1).to(dtype=slot_streams.dtype)
        seq_state = seq_tokens
        group_code, pairs = self._slotx_schedule_group()
        for layer_idx, (query_mixer, cross_attention, output_fusion) in enumerate(zip(
            self.mixformer.query_mixers,
            self.mixformer.cross_attentions,
            self.mixformer.output_fusions,
        )):
            ns_tokens = self.mixformer._run_query_mixer_step(
                query_mixer,
                ns_tokens,
                querymix_eval_code=querymix_eval_code,
            )
            normal_slot_after = self._slotx_stream_ffn(
                cross_attention=cross_attention,
                streams=slot_streams,
                mask=mask,
            )
            if exchange_off:
                attention_streams = normal_slot_after
                exchange_delta = torch.zeros_like(normal_slot_after)
            else:
                exchange_view = self._slotx_exchange_streams(streams=slot_streams, mask=mask, pairs=pairs)
                attention_streams = self._slotx_stream_ffn(
                    cross_attention=cross_attention,
                    streams=exchange_view,
                    mask=mask,
                )
                exchange_delta = attention_streams - normal_slot_after
            seq_state = self._slotx_single_ffn(cross_attention=cross_attention, seq_tokens=seq_state, mask=mask)
            ns_tokens = self._slotx_cross_attention(
                cross_attention=cross_attention,
                slot_streams=attention_streams,
                ns_tokens=ns_tokens,
                mask=mask,
                query_modulation_domain_ids=query_modulation_domain_ids,
                layer_idx=layer_idx,
            )
            if self.slotx_exchange_persist:
                slot_streams = attention_streams
            else:
                slot_streams = seq_state.unsqueeze(1) + slot_bias
                slot_streams = slot_streams * mask.unsqueeze(1).unsqueeze(-1).to(dtype=slot_streams.dtype)
            ns_tokens = output_fusion(ns_tokens)
        return ns_tokens

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        querymix_eval_code = self._querymix_eval_code(interaction)
        ns_tokens = self._mixformer_with_slot_exchange(
            seq_tokens=seq_tokens,
            ns_tokens=ns_tokens,
            mask=mask,
            query_modulation_domain_ids=domain_ids,
            exchange_off=self._slotx_exchange_off(interaction),
            querymix_eval_code=querymix_eval_code,
        )
        features = self.dropout(ns_tokens.mean(dim=1))
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class ContextBankTransitionShiftQ5MixFormerModel(TransitionShiftQ5MixFormerModel):
    """Target-only main path plus source-domain context bank side reception."""

    CONTEXTBANK_OFF_FIELD = "__mixformer_contextbank_off__"
    CONTEXTBANK_SOURCE_ZERO_FIELD = "__mixformer_contextbank_source_zero__"

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "target_only",
        q5_decision_scale: float = 1.0,
        q5_transition_mode: str = "all",
        q5_transition_slots: str = "tail2",
        contextbank_tail_k: int = 20,
        contextbank_injection_mode: str = "side_attention",
        contextbank_target_only_main: bool = True,
        contextbank_scale: float = 0.1,
        contextbank_num_heads: int = 4,
        contextbank_init_std: float = 1e-3,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_q5_mode=cross_q5_mode,
            q5_decision_scale=q5_decision_scale,
            q5_transition_mode=q5_transition_mode,
            q5_transition_slots=q5_transition_slots,
        )
        mode = str(contextbank_injection_mode or "side_attention").lower()
        if mode in {"sideattn", "side_attn", "side_attention_bank"}:
            mode = "side_attention"
        if mode in {"readout_side_channel", "readout_residual"}:
            mode = "readout"
        if mode in {"target_tokens", "all_target_tokens", "all_target_tokens_side_residual"}:
            mode = "target_tokens"
        allowed_modes = {"readout", "side_attention", "target_tokens"}
        if mode not in allowed_modes:
            raise ValueError(f"contextbank_injection_mode must be one of {sorted(allowed_modes)}, got {contextbank_injection_mode!r}.")
        if int(token_dim) % int(contextbank_num_heads) != 0:
            raise ValueError("contextbank_num_heads must divide token_dim.")
        self.contextbank_tail_k = int(contextbank_tail_k)
        self.contextbank_injection_mode = mode
        self.contextbank_target_only_main = bool(contextbank_target_only_main)
        self.contextbank_scale = float(contextbank_scale)
        self.contextbank_num_heads = int(contextbank_num_heads)
        self.contextbank_head_dim = int(token_dim) // int(contextbank_num_heads)

        self.contextbank_source_norm = nn.LayerNorm(self.token_dim)
        self.contextbank_qkv = nn.Linear(self.token_dim, self.token_dim * 3)
        self.contextbank_out = nn.Linear(self.token_dim, self.token_dim)
        self.contextbank_ffn_norm = nn.LayerNorm(self.token_dim)
        self.contextbank_ffn = SwiGLU(input_dim=self.token_dim, expand=2, dropout_rate=dropout_rate)
        self.contextbank_pair_embedding = nn.Embedding(int(num_domains) * int(num_domains), self.token_dim)
        self.contextbank_target_embedding = nn.Embedding(int(num_domains), self.token_dim)
        entry_input_dim = self.token_dim * 6
        hidden_dim = self.token_dim * 2
        self.contextbank_message_mlp = nn.Sequential(
            nn.LayerNorm(entry_input_dim),
            nn.Linear(entry_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.contextbank_gate_mlp = nn.Sequential(
            nn.LayerNorm(entry_input_dim),
            nn.Linear(entry_input_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )
        self.contextbank_query_proj = nn.Linear(self.token_dim, self.token_dim)
        self.contextbank_key_proj = nn.Linear(self.token_dim, self.token_dim)
        self.contextbank_value_proj = nn.Linear(self.token_dim, self.token_dim)
        nn.init.normal_(self.contextbank_pair_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.contextbank_target_embedding.weight, mean=0.0, std=0.02)
        message_last = self.contextbank_message_mlp[-1]
        gate_last = self.contextbank_gate_mlp[-1]
        if isinstance(message_last, nn.Linear):
            nn.init.normal_(message_last.weight, mean=0.0, std=float(contextbank_init_std))
            nn.init.zeros_(message_last.bias)
        if isinstance(gate_last, nn.Linear):
            nn.init.zeros_(gate_last.weight)
            nn.init.constant_(gate_last.bias, -1.0)

    def _contextbank_off(self, interaction) -> bool:
        if self.CONTEXTBANK_OFF_FIELD not in interaction:
            return False
        flag = interaction[self.CONTEXTBANK_OFF_FIELD]
        return bool(flag.detach().float().max().item() > 0.5)

    def _contextbank_source_zero(self, interaction) -> bool:
        if self.CONTEXTBANK_SOURCE_ZERO_FIELD not in interaction:
            return False
        flag = interaction[self.CONTEXTBANK_SOURCE_ZERO_FIELD]
        return bool(flag.detach().float().max().item() > 0.5)

    def _contextbank_tail_mask(
        self,
        *,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = hist_domain_ids.shape
        num_domains = int(self.num_domains)
        target_domain = target_domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        source_valid = known_domain_mask & (hist_domain_ids != target_domain)
        domain_ids = hist_domain_ids.long().clamp(min=0, max=max(0, num_domains - 1))
        domain_one_hot = F.one_hot(domain_ids, num_classes=num_domains).to(dtype=torch.bool)
        domain_one_hot = domain_one_hot & source_valid.unsqueeze(-1)
        source_len = domain_one_hot.long().sum(dim=1)
        if self.contextbank_tail_k > 0 and seq_len > 0:
            cum_count = domain_one_hot.long().cumsum(dim=1)
            token_rank = cum_count.gather(2, domain_ids.unsqueeze(-1)).squeeze(-1)
            domain_total = source_len.gather(1, domain_ids)
            from_tail = domain_total - token_rank
            source_valid = source_valid & (from_tail < int(self.contextbank_tail_k))
            domain_one_hot = F.one_hot(domain_ids, num_classes=num_domains).to(dtype=torch.bool)
            domain_one_hot = domain_one_hot & source_valid.unsqueeze(-1)
            source_len = domain_one_hot.long().sum(dim=1)
        return source_valid, source_len

    def _contextbank_encode_sources(
        self,
        *,
        seq_tokens: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        known_domain_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
        interaction,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_valid, source_len = self._contextbank_tail_mask(
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=target_domain_ids,
        )
        batch_size, seq_len, dim = seq_tokens.shape
        num_domains = int(self.num_domains)
        if self._contextbank_source_zero(interaction):
            source_valid = torch.zeros_like(source_valid)
        else:
            pass
        if seq_len <= 0:
            bank = seq_tokens.new_zeros(batch_size, num_domains, dim)
            has_source = torch.zeros(batch_size, num_domains, device=seq_tokens.device, dtype=torch.bool)
            return bank, has_source, source_len

        normed = self.contextbank_source_norm(seq_tokens)
        qkv = self.contextbank_qkv(normed)
        q, k, v = qkv.chunk(3, dim=-1)
        heads = int(self.contextbank_num_heads)
        head_dim = int(self.contextbank_head_dim)
        q = q.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
        same_domain = hist_domain_ids.unsqueeze(2) == hist_domain_ids.unsqueeze(1)
        causal = torch.arange(seq_len, device=seq_tokens.device).view(1, seq_len, 1) >= torch.arange(
            seq_len,
            device=seq_tokens.device,
        ).view(1, 1, seq_len)
        allowed = (
            source_valid.unsqueeze(2)
            & source_valid.unsqueeze(1)
            & same_domain
            & causal
        )
        attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=seq_tokens.device, dtype=seq_tokens.dtype)
        attn_mask = attn_mask.masked_fill(~allowed.unsqueeze(1), -1.0e4)
        source_ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        source_ctx = source_ctx.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        source_ctx = self.contextbank_out(source_ctx)
        source_ctx = seq_tokens + source_ctx
        source_ctx = self.contextbank_ffn(self.contextbank_ffn_norm(source_ctx)) + source_ctx
        source_ctx = source_ctx * source_valid.unsqueeze(-1).to(dtype=source_ctx.dtype)

        positions = torch.arange(seq_len, device=seq_tokens.device).view(1, seq_len)
        bank_parts = []
        has_parts = []
        for domain in range(num_domains):
            domain_mask = source_valid & (hist_domain_ids == int(domain))
            has_domain = domain_mask.any(dim=1)
            last_pos = torch.where(domain_mask, positions, torch.full_like(positions, -1)).max(dim=1).values
            gather_idx = last_pos.clamp(min=0, max=max(0, seq_len - 1)).view(batch_size, 1, 1).expand(-1, 1, dim)
            domain_context = source_ctx.gather(1, gather_idx).squeeze(1)
            domain_context = torch.where(has_domain.unsqueeze(-1), domain_context, torch.zeros_like(domain_context))
            bank_parts.append(domain_context)
            has_parts.append(has_domain)
        bank = torch.stack(bank_parts, dim=1)
        has_source = torch.stack(has_parts, dim=1)
        return bank, has_source, source_len

    def _contextbank_message(
        self,
        *,
        source_bank: torch.Tensor,
        source_has: torch.Tensor,
        target_state: torch.Tensor,
        target_domain_ids: torch.Tensor,
        interaction,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_domains, dim = source_bank.shape
        target_domain_ids = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        source_ids = torch.arange(num_domains, device=source_bank.device).view(1, num_domains).expand(batch_size, -1)
        pair_ids = (source_ids * int(self.num_domains) + target_domain_ids.view(-1, 1)).clamp(
            min=0,
            max=int(self.num_domains) * int(self.num_domains) - 1,
        )
        target_state_expanded = target_state.unsqueeze(1).expand_as(source_bank)
        pair_emb = self.contextbank_pair_embedding(pair_ids)
        target_emb = self.contextbank_target_embedding(target_domain_ids).unsqueeze(1).expand_as(source_bank)
        transfer_input = torch.cat(
            [
                source_bank,
                target_state_expanded,
                source_bank * target_state_expanded,
                source_bank - target_state_expanded,
                pair_emb,
                target_emb,
            ],
            dim=-1,
        )
        message = self.contextbank_message_mlp(transfer_input)
        gate = torch.sigmoid(self.contextbank_gate_mlp(transfer_input).squeeze(-1))
        source_mask = source_has & (source_ids != target_domain_ids.view(-1, 1))
        contextbank_off = self._contextbank_off(interaction)
        if contextbank_off:
            gate = torch.zeros_like(gate)
        else:
            pass
        gate = gate * source_mask.to(dtype=gate.dtype)
        message = message * source_mask.unsqueeze(-1).to(dtype=message.dtype)
        if contextbank_off:
            delta = source_bank.new_zeros(batch_size, dim)
        elif self.contextbank_injection_mode == "side_attention":
            query = self.contextbank_query_proj(target_state).unsqueeze(1)
            key = self.contextbank_key_proj(message)
            value = self.contextbank_value_proj(message)
            logits = (query * key).sum(dim=-1) * (float(dim) ** -0.5)
            logits = logits + gate.clamp_min(1.0e-6).log()
            logits = logits.masked_fill(~source_mask, -1.0e4)
            no_source = ~source_mask.any(dim=1)
            logits = torch.where(no_source.unsqueeze(-1), torch.zeros_like(logits), logits)
            weights = torch.softmax(logits, dim=-1)
            weights = torch.where(source_mask, weights, torch.zeros_like(weights))
            delta = (weights.unsqueeze(-1) * value).sum(dim=1)
        else:
            denom = source_mask.long().sum(dim=1, keepdim=True).clamp_min(1).to(dtype=message.dtype)
            delta = (gate.unsqueeze(-1) * message).sum(dim=1) / denom
        delta = float(self.contextbank_scale) * delta
        return delta, gate, message

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        full_mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        full_seq_tokens = self.seq_token_proj(seq_emb)
        full_seq_tokens = full_seq_tokens * full_mask.unsqueeze(-1).to(dtype=full_seq_tokens.dtype)
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, full_mask)
        source_bank, source_has, _ = self._contextbank_encode_sources(
            seq_tokens=full_seq_tokens,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            interaction=interaction,
        )

        target_domain = domain_ids.long().view(-1, 1).to(device=hist_domain_ids.device)
        target_mask = known_domain_mask & (hist_domain_ids == target_domain)
        main_mask = target_mask if self.contextbank_target_only_main else full_mask
        seq_tokens = full_seq_tokens * main_mask.unsqueeze(-1).to(dtype=full_seq_tokens.dtype)
        if self.contextbank_injection_mode == "target_tokens":
            target_pool = self._masked_mean(seq_tokens, main_mask)
            token_delta, _, _ = self._contextbank_message(
                source_bank=source_bank,
                source_has=source_has,
                target_state=target_pool,
                target_domain_ids=domain_ids,
                interaction=interaction,
            )
            seq_tokens = seq_tokens + token_delta.unsqueeze(1) * main_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        tokenizer_q5_slot = ns_tokens[:, self.base_num_ns_token - 1, :]
        q5_init = self._make_cross_q5_token(
            interaction=interaction,
            seq_tokens=seq_tokens,
            hist_mask=main_mask,
            target_domain_ids=domain_ids,
        )
        if self._cross_q5_off(interaction):
            q5_init = torch.zeros_like(q5_init)
        else:
            pass
        ns_tokens = torch.cat([ns_tokens[:, : self.base_num_ns_token - 1, :], q5_init.unsqueeze(1)], dim=1)
        ns_tokens = self._mixformer_with_q5_transition_view(
            seq_tokens=seq_tokens,
            ns_tokens=ns_tokens,
            hist_mask=main_mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=main_mask,
            query_modulation_domain_ids=domain_ids,
            focus_ablation_code=self._focus_ablation_code(interaction),
        )
        base_tokens = ns_tokens[:, : self.base_num_ns_token - 1, :]
        q5_final = ns_tokens[:, self.base_num_ns_token - 1, :]
        base_feature = self.dropout(base_tokens.mean(dim=1))
        q5_feature = q5_final
        if self._cross_q5_off(interaction):
            features = base_feature
            q5_feature = torch.zeros_like(q5_feature)
        else:
            generic_count = float(max(1, self.base_num_ns_token - 1))
            features = (base_tokens.sum(dim=1) + float(self.q5_decision_scale) * q5_feature) / (
                generic_count + float(self.q5_decision_scale)
            )
            features = self.dropout(features)
        q5_residual = features - base_feature
        features_before_contextbank = features
        if self.contextbank_injection_mode == "target_tokens":
            features = features_before_contextbank
        else:
            context_delta, _, _ = self._contextbank_message(
                source_bank=source_bank,
                source_has=source_has,
                target_state=features_before_contextbank,
                target_domain_ids=domain_ids,
                interaction=interaction,
            )
            features = features_before_contextbank + context_delta
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class ScenarioSlotGateTransitionShiftQ5MixFormerModel(TransitionShiftQ5MixFormerModel):
    """Domain-conditioned late readout over shared MixFormer evidence slots."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "target_only",
        q5_decision_scale: float = 1.0,
        q5_transition_mode: str = "same",
        q5_transition_slots: str = "tail2",
        slot_gate_temperature: float = 1.0,
        slot_gate_init_std: float = 0.0,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_q5_mode=cross_q5_mode,
            q5_decision_scale=q5_decision_scale,
            q5_transition_mode=q5_transition_mode,
            q5_transition_slots=q5_transition_slots,
        )
        self.slot_gate_temperature = max(float(slot_gate_temperature), 1e-6)
        self.slot_gate_logits = nn.Embedding(int(num_domains), int(self.base_num_ns_token))
        if float(slot_gate_init_std) > 0.0:
            nn.init.normal_(self.slot_gate_logits.weight, mean=0.0, std=float(slot_gate_init_std))
        else:
            nn.init.zeros_(self.slot_gate_logits.weight)

    def _safe_domain_ids(self, domain_ids: torch.Tensor) -> torch.Tensor:
        return domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)

    def _encode_transition_slot_tokens(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(curr_ts, hist_ts, seq_lens=seq_len)
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        tokenizer_q5_slot = ns_tokens[:, self.base_num_ns_token - 1, :]
        q5_init = self._make_cross_q5_token(
            interaction=interaction,
            seq_tokens=seq_tokens,
            hist_mask=mask,
            target_domain_ids=domain_ids,
        )
        if self._cross_q5_off(interaction):
            q5_init = torch.zeros_like(q5_init)
        else:
            pass
        ns_tokens = torch.cat([ns_tokens[:, : self.base_num_ns_token - 1, :], q5_init.unsqueeze(1)], dim=1)
        ns_tokens = self._mixformer_with_q5_transition_view(
            seq_tokens=seq_tokens,
            ns_tokens=ns_tokens,
            hist_mask=mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            query_modulation_domain_ids=domain_ids,
            focus_ablation_code=self._focus_ablation_code(interaction),
        )
        return ns_tokens, domain_ids

    def _slot_gate(
        self,
        *,
        ns_tokens: torch.Tensor,
        domain_ids: torch.Tensor,
        candidate_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_slots, _ = ns_tokens.shape
        safe_domain_ids = self._safe_domain_ids(domain_ids).to(device=ns_tokens.device)
        logits_all = self.slot_gate_logits(safe_domain_ids).to(dtype=ns_tokens.dtype)
        logits_all = logits_all[:, :num_slots]
        if candidate_indices is None:
            candidate_indices = torch.arange(num_slots, device=ns_tokens.device, dtype=torch.long)
        else:
            candidate_indices = candidate_indices.to(device=ns_tokens.device, dtype=torch.long)
        candidate_logits = logits_all.index_select(1, candidate_indices)
        gate = torch.softmax(candidate_logits / float(self.slot_gate_temperature), dim=-1)
        gate_full = logits_all.new_zeros((batch_size, num_slots))
        gate_full.scatter_(1, candidate_indices.view(1, -1).expand(batch_size, -1), gate)
        candidate_mask = logits_all.new_zeros((batch_size, num_slots))
        candidate_mask.scatter_(1, candidate_indices.view(1, -1).expand(batch_size, -1), 1.0)
        masked_logits = logits_all.masked_fill(candidate_mask <= 0.0, -1.0e4)
        entropy = -(gate.clamp_min(1e-12) * gate.clamp_min(1e-12).log()).sum(dim=-1)
        denom = torch.log(torch.full_like(entropy, float(max(2, int(candidate_indices.numel())))))
        norm_entropy = entropy / denom.clamp_min(1e-12)
        top1 = gate_full.argmax(dim=-1).to(dtype=gate_full.dtype)
        s_slot_mask = self._transition_slot_mask(num_slots, ns_tokens.device).to(dtype=gate_full.dtype)
        s_share = (gate_full * s_slot_mask.view(1, -1)).sum(dim=-1)
        n_share = (gate_full * (1.0 - s_slot_mask.view(1, -1))).sum(dim=-1)
        return gate_full, candidate_mask

    def _gated_slot_feature(
        self,
        *,
        ns_tokens: torch.Tensor,
        domain_ids: torch.Tensor,
        candidate_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate, _ = self._slot_gate(
            ns_tokens=ns_tokens,
            domain_ids=domain_ids,
            candidate_indices=candidate_indices,
        )
        feature = torch.einsum("bs,bsd->bd", gate, ns_tokens)
        return feature

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        ns_tokens, domain_ids = self._encode_transition_slot_tokens(interaction)
        feature = self._gated_slot_feature(ns_tokens=ns_tokens, domain_ids=domain_ids)
        feature = self.dropout(feature)
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            feature = torch.cat([feature, late_target_x], dim=-1)
        return feature, domain_ids


class ProtectedSharedQScenarioSlotGateTransitionShiftQ5MixFormerModel(ScenarioSlotGateTransitionShiftQ5MixFormerModel):
    """Keep one shared slot outside the domain gate and gate only the remaining slots."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "target_only",
        q5_decision_scale: float = 1.0,
        q5_transition_mode: str = "same",
        q5_transition_slots: str = "tail2",
        slot_gate_temperature: float = 1.0,
        slot_gate_init_std: float = 0.0,
        protected_slot_index: int = 0,
        protected_branch_scale: float = 1.0,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_q5_mode=cross_q5_mode,
            q5_decision_scale=q5_decision_scale,
            q5_transition_mode=q5_transition_mode,
            q5_transition_slots=q5_transition_slots,
            slot_gate_temperature=slot_gate_temperature,
            slot_gate_init_std=slot_gate_init_std,
        )
        if int(num_ns_token) < 2:
            raise ValueError("ProtectedSharedQScenarioSlotGateTransitionShiftQ5MixFormerModel requires num_ns_token >= 2.")
        self.protected_slot_index = min(max(0, int(protected_slot_index)), int(num_ns_token) - 1)
        self.protected_branch_scale = float(protected_branch_scale)
        feature_dim = int(self.token_dim) + int(self.late_target_embedding_dim)
        hidden_dims = to_dims(tower_hidden_dims, default=(128, 64))
        self.shared_head = MLP(
            feature_dim,
            *hidden_dims,
            1,
            dropout_rate=tower_dropout_rate,
            activation="relu",
            batch_norm=tower_batch_norm,
        )
        self.protected_feature_dim = feature_dim

    def _candidate_indices(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [idx for idx in range(int(ns_tokens.shape[1])) if idx != int(self.protected_slot_index)],
            device=ns_tokens.device,
            dtype=torch.long,
        )

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        ns_tokens, domain_ids = self._encode_transition_slot_tokens(interaction)
        shared_feature = ns_tokens[:, int(self.protected_slot_index), :]
        gated_feature = self._gated_slot_feature(
            ns_tokens=ns_tokens,
            domain_ids=domain_ids,
            candidate_indices=self._candidate_indices(ns_tokens),
        )
        gated_feature = self.dropout(gated_feature)
        shared_feature = self.dropout(shared_feature)
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            shared_feature = torch.cat([shared_feature, late_target_x], dim=-1)
            gated_feature = torch.cat([gated_feature, late_target_x], dim=-1)
        features = torch.cat([shared_feature, gated_feature], dim=-1)
        return features, domain_ids

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        split = int(self.protected_feature_dim)
        shared_feature = x[:, :split]
        gated_feature = x[:, split:]
        shared_logit = self.shared_head(shared_feature).squeeze(-1)
        gated_logit = self.head(gated_feature, domain_ids)
        final_logit = shared_logit + float(self.protected_branch_scale) * gated_logit
        return final_logit


class ProtectedSharedQZeroOptionScenarioSlotGateTransitionShiftQ5MixFormerModel(
    ProtectedSharedQScenarioSlotGateTransitionShiftQ5MixFormerModel
):
    """Protected shared slot plus a non-learnable zero abstention option for the domain branch."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "target_only",
        q5_decision_scale: float = 1.0,
        q5_transition_mode: str = "same",
        q5_transition_slots: str = "tail2",
        slot_gate_temperature: float = 1.0,
        slot_gate_init_std: float = 0.0,
        protected_slot_index: int = 0,
        protected_branch_scale: float = 1.0,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_q5_mode=cross_q5_mode,
            q5_decision_scale=q5_decision_scale,
            q5_transition_mode=q5_transition_mode,
            q5_transition_slots=q5_transition_slots,
            slot_gate_temperature=slot_gate_temperature,
            slot_gate_init_std=slot_gate_init_std,
            protected_slot_index=protected_slot_index,
            protected_branch_scale=protected_branch_scale,
        )
        self.slot_gate_zero_option = True
        self.slot_gate_zero_index = int(self.base_num_ns_token)
        self.slot_gate_logits = nn.Embedding(int(num_domains), int(self.base_num_ns_token) + 1)
        if float(slot_gate_init_std) > 0.0:
            nn.init.normal_(self.slot_gate_logits.weight, mean=0.0, std=float(slot_gate_init_std))
        else:
            nn.init.zeros_(self.slot_gate_logits.weight)

    def _zero_option_slot_gate(self, *, ns_tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        batch_size, num_real_slots, _ = ns_tokens.shape
        extended_slots = num_real_slots + 1
        safe_domain_ids = self._safe_domain_ids(domain_ids).to(device=ns_tokens.device)
        logits_all = self.slot_gate_logits(safe_domain_ids).to(dtype=ns_tokens.dtype)
        logits_all = logits_all[:, :extended_slots]
        real_candidates = self._candidate_indices(ns_tokens)
        zero_index = torch.tensor([num_real_slots], device=ns_tokens.device, dtype=torch.long)
        candidate_indices = torch.cat([real_candidates, zero_index], dim=0)
        candidate_logits = logits_all.index_select(1, candidate_indices)
        gate = torch.softmax(candidate_logits / float(self.slot_gate_temperature), dim=-1)
        gate_full = logits_all.new_zeros((batch_size, extended_slots))
        gate_full.scatter_(1, candidate_indices.view(1, -1).expand(batch_size, -1), gate)
        candidate_mask = logits_all.new_zeros((batch_size, extended_slots))
        candidate_mask.scatter_(1, candidate_indices.view(1, -1).expand(batch_size, -1), 1.0)
        masked_logits = logits_all.masked_fill(candidate_mask <= 0.0, -1.0e4)
        entropy = -(gate.clamp_min(1e-12) * gate.clamp_min(1e-12).log()).sum(dim=-1)
        denom = torch.log(torch.full_like(entropy, float(max(2, int(candidate_indices.numel())))))
        norm_entropy = entropy / denom.clamp_min(1e-12)
        top1 = gate_full.argmax(dim=-1).to(dtype=gate_full.dtype)
        s_slot_mask = self._transition_slot_mask(num_real_slots, ns_tokens.device).to(dtype=gate_full.dtype)
        s_slot_mask = torch.cat([s_slot_mask, s_slot_mask.new_zeros(1)], dim=0)
        zero_mask = gate_full.new_zeros(extended_slots)
        zero_mask[num_real_slots] = 1.0
        zero_share = gate_full[:, num_real_slots]
        real_share = 1.0 - zero_share
        s_share = (gate_full * s_slot_mask.view(1, -1)).sum(dim=-1)
        n_share = (gate_full * (1.0 - s_slot_mask.view(1, -1) - zero_mask.view(1, -1))).sum(dim=-1)
        return gate_full

    def _gated_slot_feature_with_zero(self, *, ns_tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        gate = self._zero_option_slot_gate(ns_tokens=ns_tokens, domain_ids=domain_ids)
        feature = torch.einsum("bs,bsd->bd", gate[:, : ns_tokens.shape[1]], ns_tokens)
        return feature

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        ns_tokens, domain_ids = self._encode_transition_slot_tokens(interaction)
        shared_feature = ns_tokens[:, int(self.protected_slot_index), :]
        gated_feature = self._gated_slot_feature_with_zero(ns_tokens=ns_tokens, domain_ids=domain_ids)
        gated_feature = self.dropout(gated_feature)
        shared_feature = self.dropout(shared_feature)
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            shared_feature = torch.cat([shared_feature, late_target_x], dim=-1)
            gated_feature = torch.cat([gated_feature, late_target_x], dim=-1)
        features = torch.cat([shared_feature, gated_feature], dim=-1)
        return features, domain_ids

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        split = int(self.protected_feature_dim)
        shared_feature = x[:, :split]
        gated_feature = x[:, split:]
        shared_logit = self.shared_head(shared_feature).squeeze(-1)
        raw_domain_logit = self.head(gated_feature, domain_ids)
        zero_feature = torch.zeros_like(gated_feature)
        zero_baseline_logit = self.head(zero_feature, domain_ids)
        gated_logit = raw_domain_logit - zero_baseline_logit
        final_logit = shared_logit + float(self.protected_branch_scale) * gated_logit
        return final_logit


class ProtectedSharedQSharedQGateZeroOptionScenarioSlotGateTransitionShiftQ5MixFormerModel(
    ProtectedSharedQZeroOptionScenarioSlotGateTransitionShiftQ5MixFormerModel
):
    """Zero-option protected gate whose logits are conditioned on the protected shared slot."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "target_only",
        q5_decision_scale: float = 1.0,
        q5_transition_mode: str = "same",
        q5_transition_slots: str = "tail2",
        slot_gate_temperature: float = 1.0,
        slot_gate_init_std: float = 0.0,
        protected_slot_index: int = 0,
        protected_branch_scale: float = 1.0,
        slot_gate_context_hidden_dim: Optional[int] = None,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_q5_mode=cross_q5_mode,
            q5_decision_scale=q5_decision_scale,
            q5_transition_mode=q5_transition_mode,
            q5_transition_slots=q5_transition_slots,
            slot_gate_temperature=slot_gate_temperature,
            slot_gate_init_std=slot_gate_init_std,
            protected_slot_index=protected_slot_index,
            protected_branch_scale=protected_branch_scale,
        )
        self.slot_gate_condition_on_shared_q = True
        hidden_dim = int(slot_gate_context_hidden_dim or token_dim)
        output_dim = int(self.base_num_ns_token) + 1
        self.slot_gate_sharedq_context = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _zero_option_slot_gate(self, *, ns_tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        batch_size, num_real_slots, _ = ns_tokens.shape
        extended_slots = num_real_slots + 1
        safe_domain_ids = self._safe_domain_ids(domain_ids).to(device=ns_tokens.device)
        domain_logits = self.slot_gate_logits(safe_domain_ids).to(dtype=ns_tokens.dtype)
        domain_logits = domain_logits[:, :extended_slots]
        shared_q = ns_tokens[:, int(self.protected_slot_index), :]
        context_logits = self.slot_gate_sharedq_context(shared_q).to(dtype=ns_tokens.dtype)
        context_logits = context_logits[:, :extended_slots]
        logits_all = domain_logits + context_logits
        real_candidates = self._candidate_indices(ns_tokens)
        zero_index = torch.tensor([num_real_slots], device=ns_tokens.device, dtype=torch.long)
        candidate_indices = torch.cat([real_candidates, zero_index], dim=0)
        candidate_logits = logits_all.index_select(1, candidate_indices)
        gate = torch.softmax(candidate_logits / float(self.slot_gate_temperature), dim=-1)
        gate_full = logits_all.new_zeros((batch_size, extended_slots))
        gate_full.scatter_(1, candidate_indices.view(1, -1).expand(batch_size, -1), gate)
        candidate_mask = logits_all.new_zeros((batch_size, extended_slots))
        candidate_mask.scatter_(1, candidate_indices.view(1, -1).expand(batch_size, -1), 1.0)
        masked_logits = logits_all.masked_fill(candidate_mask <= 0.0, -1.0e4)
        entropy = -(gate.clamp_min(1e-12) * gate.clamp_min(1e-12).log()).sum(dim=-1)
        denom = torch.log(torch.full_like(entropy, float(max(2, int(candidate_indices.numel())))))
        norm_entropy = entropy / denom.clamp_min(1e-12)
        top1 = gate_full.argmax(dim=-1).to(dtype=gate_full.dtype)
        s_slot_mask = self._transition_slot_mask(num_real_slots, ns_tokens.device).to(dtype=gate_full.dtype)
        s_slot_mask = torch.cat([s_slot_mask, s_slot_mask.new_zeros(1)], dim=0)
        zero_mask = gate_full.new_zeros(extended_slots)
        zero_mask[num_real_slots] = 1.0
        zero_share = gate_full[:, num_real_slots]
        real_share = 1.0 - zero_share
        s_share = (gate_full * s_slot_mask.view(1, -1)).sum(dim=-1)
        n_share = (gate_full * (1.0 - s_slot_mask.view(1, -1) - zero_mask.view(1, -1))).sum(dim=-1)
        return gate_full


class DomainFlowTransitionShiftQ5MixFormerModel(TransitionShiftQ5MixFormerModel):
    """Best transition-shift Q5 backbone plus compact directed domain-flow memory."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "target_only",
        q5_decision_scale: float = 1.0,
        q5_transition_mode: str = "all",
        q5_transition_slots: str = "all",
        domain_flow_init_std: float = 1e-4,
        domain_flow_state_scale: float = 0.1,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_q5_mode=cross_q5_mode,
            q5_decision_scale=q5_decision_scale,
            q5_transition_mode=q5_transition_mode,
            q5_transition_slots=q5_transition_slots,
        )
        self.domain_flow_init_std = float(domain_flow_init_std)
        self.domain_flow_state_scale = float(domain_flow_state_scale)
        pair_count = int(num_domains) * int(num_domains)
        self.domain_flow_target_embedding = nn.Embedding(int(num_domains), self.token_dim)
        nn.init.normal_(self.domain_flow_target_embedding.weight, mean=0.0, std=0.02)
        self.domain_flow_pair_bias = nn.Embedding(pair_count, 1)
        nn.init.zeros_(self.domain_flow_pair_bias.weight)
        self.domain_flow_state_norm = _make_norm(self.token_dim)
        self.domain_flow_query_mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim * 2),
            nn.Linear(self.token_dim * 2, self.token_dim * 2),
            nn.GELU(),
            nn.Linear(self.token_dim * 2, self.token_dim),
        )
        state_input_dim = self.token_dim * 3 + 4
        self.domain_flow_state_mlp = nn.Sequential(
            nn.LayerNorm(state_input_dim),
            nn.Linear(state_input_dim, self.token_dim * 2),
            nn.GELU(),
            nn.Linear(self.token_dim * 2, self.token_dim),
        )
        self.domain_flow_gate_mlp = nn.Sequential(
            nn.LayerNorm(state_input_dim),
            nn.Linear(state_input_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )
        last_state = self.domain_flow_state_mlp[-1]
        if isinstance(last_state, nn.Linear):
            nn.init.zeros_(last_state.weight)
            nn.init.zeros_(last_state.bias)
        last_gate = self.domain_flow_gate_mlp[-1]
        if isinstance(last_gate, nn.Linear):
            nn.init.zeros_(last_gate.weight)
            nn.init.constant_(last_gate.bias, -2.0)

    def _domain_flow_latest_states(self, **kwargs):
        return DomainFlowCrossDomainPositionMixFormerModel._domain_flow_latest_states(self, **kwargs)

    def _domain_flow_transition_to_target_counts(self, **kwargs):
        return DomainFlowCrossDomainPositionMixFormerModel._domain_flow_transition_to_target_counts(self, **kwargs)

    def _domain_flow_state_residual(self, **kwargs):
        return DomainFlowCrossDomainPositionMixFormerModel._domain_flow_state_residual(self, **kwargs)

    @staticmethod
    def _domain_flow_masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return DomainFlowCrossDomainPositionMixFormerModel._domain_flow_masked_softmax(logits, mask)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(curr_ts, hist_ts, seq_lens=seq_len)
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        tokenizer_q5_slot = ns_tokens[:, self.base_num_ns_token - 1, :]
        q5_init = self._make_cross_q5_token(
            interaction=interaction,
            seq_tokens=seq_tokens,
            hist_mask=mask,
            target_domain_ids=domain_ids,
        )
        if self._cross_q5_off(interaction):
            q5_init = torch.zeros_like(q5_init)
        else:
            pass
        ns_tokens = torch.cat([ns_tokens[:, : self.base_num_ns_token - 1, :], q5_init.unsqueeze(1)], dim=1)
        ns_tokens = self._mixformer_with_q5_transition_view(
            seq_tokens=seq_tokens,
            ns_tokens=ns_tokens,
            hist_mask=mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            query_modulation_domain_ids=domain_ids,
            focus_ablation_code=self._focus_ablation_code(interaction),
        )
        base_tokens = ns_tokens[:, : self.base_num_ns_token - 1, :]
        q5_final = ns_tokens[:, self.base_num_ns_token - 1, :]
        base_feature = self.dropout(base_tokens.mean(dim=1))
        q5_feature = q5_final
        if self._cross_q5_off(interaction):
            features = base_feature
            q5_feature = torch.zeros_like(q5_feature)
        else:
            generic_count = float(max(1, self.base_num_ns_token - 1))
            features = (base_tokens.sum(dim=1) + float(self.q5_decision_scale) * q5_feature) / (
                generic_count + float(self.q5_decision_scale)
            )
            features = self.dropout(features)
        q5_residual = features - base_feature
        flow_residual = self._domain_flow_state_residual(
            seq_tokens=seq_tokens,
            hist_mask=mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            target_domain_ids=domain_ids,
            base_feature=features,
        )
        features = features + flow_residual
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class EntryUpdateTransitionShiftQ5MixFormerModel(TransitionShiftQ5MixFormerModel):
    """Transition-shift Q5 backbone with directed entry-token updates."""

    ENTRY_UPDATE_OFF_FIELD = EntryUpdateMixFormerModel.ENTRY_UPDATE_OFF_FIELD

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        cross_q5_mode: str = "target_only",
        q5_decision_scale: float = 1.0,
        q5_transition_mode: str = "all",
        q5_transition_slots: str = "all",
        entry_update_mode: str = "last1",
        entry_update_target_only: bool = False,
        entry_update_scale: float = 0.1,
        entry_update_init_std: float = 1e-3,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
            domain_seq_field=domain_seq_field,
            cross_q5_mode=cross_q5_mode,
            q5_decision_scale=q5_decision_scale,
            q5_transition_mode=q5_transition_mode,
            q5_transition_slots=q5_transition_slots,
        )
        mode = str(entry_update_mode or "last1").lower()
        if mode in {"segment", "prev_segment", "prev_segment_mean"}:
            mode = "segment_mean"
        if mode in {"segment_encoder", "segment_encoder_k50", "prev_segment_encoder", "segment_gru"}:
            mode = "segment_encoder"
        if mode in {"domain_subseq_encoder", "domain_subseq_encoder_k100", "source_domain_subseq_encoder", "domain_subseq_gru"}:
            mode = "domain_subseq_encoder"
        allowed_modes = {"last1", "segment_mean", "segment_encoder", "domain_subseq_encoder"}
        if mode not in allowed_modes:
            raise ValueError(f"entry_update_mode must be one of {sorted(allowed_modes)}, got {entry_update_mode!r}.")
        self.entry_update_mode = mode
        self.entry_update_target_only = bool(entry_update_target_only)
        self.entry_update_scale = float(entry_update_scale)
        self.entry_update_pair_embedding = nn.Embedding(int(num_domains) * int(num_domains), self.token_dim)
        self.entry_update_target_embedding = nn.Embedding(int(num_domains), self.token_dim)
        self.entry_update_context_norm = nn.LayerNorm(self.token_dim)
        self.entry_update_context_input = nn.Linear(self.token_dim, self.token_dim)
        self.entry_update_context_state = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.entry_update_context_gate = nn.Linear(self.token_dim * 2, self.token_dim)
        entry_input_dim = self.token_dim * 6
        hidden_dim = self.token_dim * 2
        self.entry_update_message_mlp = nn.Sequential(
            nn.LayerNorm(entry_input_dim),
            nn.Linear(entry_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.entry_update_gate_mlp = nn.Sequential(
            nn.LayerNorm(entry_input_dim),
            nn.Linear(entry_input_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )
        nn.init.normal_(self.entry_update_pair_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.entry_update_target_embedding.weight, mean=0.0, std=0.02)
        message_last = self.entry_update_message_mlp[-1]
        gate_last = self.entry_update_gate_mlp[-1]
        if isinstance(message_last, nn.Linear):
            nn.init.normal_(message_last.weight, mean=0.0, std=float(entry_update_init_std))
            nn.init.zeros_(message_last.bias)
        if isinstance(gate_last, nn.Linear):
            nn.init.zeros_(gate_last.weight)
            nn.init.constant_(gate_last.bias, -1.0)

    def _entry_update_off(self, interaction) -> bool:
        return EntryUpdateMixFormerModel._entry_update_off(self, interaction)

    def _previous_last_token_context(self, *args, **kwargs):
        return EntryUpdateMixFormerModel._previous_last_token_context(self, *args, **kwargs)

    def _previous_segment_mean_context(self, *args, **kwargs):
        return EntryUpdateMixFormerModel._previous_segment_mean_context(self, *args, **kwargs)

    def _entry_context_encoder_step(self, *args, **kwargs):
        return EntryUpdateMixFormerModel._entry_context_encoder_step(self, *args, **kwargs)

    def _previous_segment_encoder_context(self, *args, **kwargs):
        return EntryUpdateMixFormerModel._previous_segment_encoder_context(self, *args, **kwargs)

    def _previous_domain_subseq_encoder_context(self, *args, **kwargs):
        return EntryUpdateMixFormerModel._previous_domain_subseq_encoder_context(self, *args, **kwargs)

    def _entry_update_sequence(self, **kwargs):
        return EntryUpdateMixFormerModel._entry_update_sequence(self, **kwargs)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(curr_ts, hist_ts, seq_lens=seq_len)
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        seq_tokens = self._entry_update_sequence(
            interaction=interaction,
            seq_tokens=seq_tokens,
            hist_mask=mask,
            target_domain_ids=domain_ids,
        )
        hist_domain_ids, known_domain_mask = self._history_domain_ids(interaction, mask)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        tokenizer_q5_slot = ns_tokens[:, self.base_num_ns_token - 1, :]
        q5_init = self._make_cross_q5_token(
            interaction=interaction,
            seq_tokens=seq_tokens,
            hist_mask=mask,
            target_domain_ids=domain_ids,
        )
        if self._cross_q5_off(interaction):
            q5_init = torch.zeros_like(q5_init)
        else:
            pass
        ns_tokens = torch.cat([ns_tokens[:, : self.base_num_ns_token - 1, :], q5_init.unsqueeze(1)], dim=1)
        ns_tokens = self._mixformer_with_q5_transition_view(
            seq_tokens=seq_tokens,
            ns_tokens=ns_tokens,
            hist_mask=mask,
            hist_domain_ids=hist_domain_ids,
            known_domain_mask=known_domain_mask,
            query_modulation_domain_ids=domain_ids,
            focus_ablation_code=self._focus_ablation_code(interaction),
        )
        base_tokens = ns_tokens[:, : self.base_num_ns_token - 1, :]
        q5_final = ns_tokens[:, self.base_num_ns_token - 1, :]
        base_feature = self.dropout(base_tokens.mean(dim=1))
        q5_feature = q5_final
        if self._cross_q5_off(interaction):
            features = base_feature
            q5_feature = torch.zeros_like(q5_feature)
        else:
            generic_count = float(max(1, self.base_num_ns_token - 1))
            features = (base_tokens.sum(dim=1) + float(self.q5_decision_scale) * q5_feature) / (
                generic_count + float(self.q5_decision_scale)
            )
            features = self.dropout(features)
        q5_residual = features - base_feature
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids


class DomainViewLogitMixFormerModel(MixFormerModel):
    """MixFormer plus per-domain token opinions at the logit outlet."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_mixer_mode: str = "full",
        query_mixer_window_chunks: int = 2,
        query_mixer_window_global_last: bool = True,
        domain_seq_field: str = "domain_tab_seq",
        view_logit_scale: float = 1.0,
        support_aware_gate: bool = True,
        gate_allow_target_fallback: bool = True,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
            query_mixer_mode=query_mixer_mode,
            query_mixer_window_chunks=query_mixer_window_chunks,
            query_mixer_window_global_last=query_mixer_window_global_last,
        )
        self.domain_seq_field = str(domain_seq_field)
        self.view_logit_scale = float(view_logit_scale)
        self.support_aware_gate = bool(support_aware_gate)
        self.gate_allow_target_fallback = bool(gate_allow_target_fallback)
        self.domain_view_query = nn.Embedding(int(num_domains), self.token_dim)
        self.domain_view_target_embedding = nn.Embedding(int(num_domains), self.token_dim)
        hidden_dim = max(self.token_dim * 2, 64)
        gate_input_dim = self.token_dim * 3
        self.domain_view_gate_mlp = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.domain_view_logit_mlp = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.domain_view_query.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.domain_view_target_embedding.weight, mean=0.0, std=0.02)
        self._zero_last_linear(self.domain_view_gate_mlp)
        self._small_last_linear(self.domain_view_logit_mlp)
        self._last_domain_view_logit_agg: Optional[torch.Tensor] = None

    @staticmethod
    def _zero_last_linear(module: nn.Module) -> None:
        last_linear = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                last_linear = child
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    @staticmethod
    def _small_last_linear(module: nn.Module, std: float = 1e-3) -> None:
        last_linear = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                last_linear = child
        if last_linear is not None:
            nn.init.normal_(last_linear.weight, mean=0.0, std=float(std))
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def _history_domain_counts(
        self,
        interaction,
        *,
        mask: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        source_field = self.domain_seq_field if self.domain_seq_field in interaction else "tab_seq"
        if source_field not in interaction:
            return torch.ones(
                domain_ids.shape[0],
                int(self.num_domains),
                device=domain_ids.device,
                dtype=torch.float32,
            )
        hist_domain_ids = interaction[source_field].long().to(device=domain_ids.device) - int(self.domain_shift)
        known = (hist_domain_ids >= 0) & (hist_domain_ids < int(self.num_domains)) & mask.to(dtype=torch.bool)
        one_hot = F.one_hot(
            hist_domain_ids.clamp(min=0, max=int(self.num_domains) - 1),
            num_classes=int(self.num_domains),
        ).to(dtype=torch.float32)
        counts = torch.einsum("bl,blk->bk", known.to(dtype=torch.float32), one_hot)
        return counts

    def _domain_view_parts(
        self,
        *,
        ns_tokens: torch.Tensor,
        base_features: torch.Tensor,
        domain_ids: torch.Tensor,
        interaction,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(ns_tokens.shape[0])
        safe_domain_ids = domain_ids.long().clamp(min=0, max=int(self.num_domains) - 1)
        source_queries = self.domain_view_query.weight.to(device=ns_tokens.device, dtype=ns_tokens.dtype)
        source_scores = torch.einsum("btd,kd->bkt", ns_tokens, source_queries)
        source_scores = source_scores * (float(self.token_dim) ** -0.5)
        source_weights = torch.softmax(source_scores, dim=-1)
        domain_views = torch.einsum("bkt,btd->bkd", source_weights, ns_tokens)

        shared_expand = base_features.unsqueeze(1).expand(-1, int(self.num_domains), -1)
        target_emb = self.domain_view_target_embedding(safe_domain_ids)
        target_expand = target_emb.unsqueeze(1).expand(-1, int(self.num_domains), -1)
        gate_input = torch.cat([domain_views, shared_expand, target_expand], dim=-1)
        gate_logits = self.domain_view_gate_mlp(gate_input).squeeze(-1)
        if self.support_aware_gate:
            hist_counts = self._history_domain_counts(interaction, mask=mask, domain_ids=safe_domain_ids)
            support_mask = hist_counts > 0
            if self.gate_allow_target_fallback:
                target_support = F.one_hot(
                    safe_domain_ids,
                    num_classes=int(self.num_domains),
                ).to(dtype=torch.bool)
                support_mask = support_mask | target_support
            gate_logits = gate_logits.masked_fill(~support_mask, -1e4)
        gate = torch.softmax(gate_logits, dim=-1)
        entropy = -(gate * torch.log(gate.clamp_min(1e-12))).sum(dim=-1)

        view_logits = self.domain_view_logit_mlp(gate_input).squeeze(-1)
        view_logit_agg = float(self.view_logit_scale) * torch.einsum("bk,bk->b", gate, view_logits)
        if batch_size == 0:
            return view_logit_agg
        return view_logit_agg

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = self._encode_current_components(interaction)
        if self.use_relative_time:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = interaction[self.history_view.time_field]
        else:
            seq_emb = self.history_view.forward(interaction, split_by="none")
            seq_len = interaction[self.history_view.seq_len_field]
            hist_ts = None
        target_emb = self._encode_target_components(interaction)
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        _, ns_tokens = self.mixformer(
            seq_tokens,
            ns_tokens,
            mask,
            query_modulation_domain_ids=domain_ids,
        )
        features = self.dropout(ns_tokens.mean(dim=1))
        self._last_domain_view_logit_agg = self._domain_view_parts(
            ns_tokens=ns_tokens,
            base_features=features,
            domain_ids=domain_ids,
            interaction=interaction,
            mask=mask,
        )
        late_target_x = self._encode_late_target_components(interaction)
        if late_target_x is not None:
            features = torch.cat([features, late_target_x], dim=-1)
        return features, domain_ids

    def logits(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        logits = self.forward(x, domain_ids)
        if isinstance(self._last_domain_view_logit_agg, torch.Tensor):
            logits = logits + self._last_domain_view_logit_agg
        return logits


class QueryModulatedMixFormerModel(MixFormerModel):
    """MixFormer with domain-conditioned low-rank modulation on the Query path only."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        target_feature_fields: Optional[Iterable[str]] = None,
        late_target_feature_fields: Optional[Iterable[str]] = None,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        query_modulation_rank: int = 32,
        query_modulation_scale: float = 0.05,
        query_modulation_anchor_domain: int = -1,
        query_modulation_preserve_norm: bool = False,
        query_modulation_layers: str = "tail",
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            target_feature_fields=target_feature_fields,
            late_target_feature_fields=late_target_feature_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.query_modulation_rank = int(query_modulation_rank)
        self.query_modulation_scale = float(query_modulation_scale)
        self.query_modulation_anchor_domain = int(query_modulation_anchor_domain)
        self.query_modulation_preserve_norm = bool(query_modulation_preserve_norm)
        self.query_modulation_layers = str(query_modulation_layers)
        self.mixformer = MixFormerBlocks(
            input_dim=self.token_dim,
            num_tokens=self.num_ns_token,
            num_layers=num_layers,
            expand=expansion_factor,
            dropout_rate=dropout_rate,
            query_modulation_num_domains=num_domains,
            query_modulation_rank=self.query_modulation_rank,
            query_modulation_scale=self.query_modulation_scale,
            query_modulation_anchor_domain=self.query_modulation_anchor_domain,
            query_modulation_preserve_norm=self.query_modulation_preserve_norm,
            query_modulation_layers=self.query_modulation_layers,
        )


class KeyModulatedMixFormerModel(MixFormerModel):
    """MixFormer with target-conditioned low-rank modulation on the Key path only."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        key_modulation_rank: int = 32,
        key_modulation_hidden_dim: int = 64,
        key_modulation_scale: float = 0.1,
        key_modulation_layers: str = "all",
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.key_modulation_rank = int(key_modulation_rank)
        self.key_modulation_hidden_dim = int(key_modulation_hidden_dim)
        self.key_modulation_scale = float(key_modulation_scale)
        self.key_modulation_layers = str(key_modulation_layers)
        self.mixformer = MixFormerBlocks(
            input_dim=self.token_dim,
            num_tokens=self.num_ns_token,
            num_layers=num_layers,
            expand=expansion_factor,
            dropout_rate=dropout_rate,
            key_modulation_condition_dim=self.current_embedding_dim + self.sequence_embedding_dim,
            key_modulation_rank=self.key_modulation_rank,
            key_modulation_hidden_dim=self.key_modulation_hidden_dim,
            key_modulation_scale=self.key_modulation_scale,
            key_modulation_layers=self.key_modulation_layers,
        )

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = torch.flatten(self.current_view(interaction), start_dim=1)
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        _, ns_tokens = self.mixformer(
            seq_tokens,
            ns_tokens,
            mask,
            key_modulation_condition=ns_input,
        )
        features = self.dropout(ns_tokens.mean(dim=1))
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        return features, domain_ids


class ScoreRerankMixFormerModel(MixFormerModel):
    """MixFormer with target-conditioned low-rank reranking on attention scores."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        score_rerank_rank: int = 32,
        score_rerank_hidden_dim: int = 64,
        score_rerank_scale: float = 0.1,
        score_rerank_layers: str = "all",
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.score_rerank_rank = int(score_rerank_rank)
        self.score_rerank_hidden_dim = int(score_rerank_hidden_dim)
        self.score_rerank_scale = float(score_rerank_scale)
        self.score_rerank_layers = str(score_rerank_layers)
        self.mixformer = MixFormerBlocks(
            input_dim=self.token_dim,
            num_tokens=self.num_ns_token,
            num_layers=num_layers,
            expand=expansion_factor,
            dropout_rate=dropout_rate,
            score_rerank_condition_dim=self.current_embedding_dim + self.sequence_embedding_dim,
            score_rerank_rank=self.score_rerank_rank,
            score_rerank_hidden_dim=self.score_rerank_hidden_dim,
            score_rerank_scale=self.score_rerank_scale,
            score_rerank_layers=self.score_rerank_layers,
        )

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = torch.flatten(self.current_view(interaction), start_dim=1)
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        _, ns_tokens = self.mixformer(
            seq_tokens,
            ns_tokens,
            mask,
            score_rerank_condition=ns_input,
        )
        features = self.dropout(ns_tokens.mean(dim=1))
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        return features, domain_ids


class AdaptiveDomainTokenScaling(nn.Module):
    def __init__(
        self,
        token_dim: int,
        condition_dim: int,
        hidden_dim: int = 64,
        eta: float = 2.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.eta = float(eta)
        self.shared_weight = nn.Parameter(torch.empty(self.token_dim, self.token_dim))
        self.weight_gen = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.token_dim * self.token_dim),
        )
        self.bias_gen = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.token_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.eye_(self.shared_weight)
        for module in (self.weight_gen, self.bias_gen):
            last = module[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = int(tokens.shape[0])
        private_weight = torch.sigmoid(self.weight_gen(condition)).view(
            batch_size,
            self.token_dim,
            self.token_dim,
        )
        generated_weight = self.eta * self.shared_weight.unsqueeze(0) * private_weight
        generated_bias = self.bias_gen(condition).unsqueeze(1)
        return torch.bmm(tokens, generated_weight.transpose(1, 2)) + generated_bias


class ADSMixFormerModel(MixFormerModel):
    """MixFormer with ADS-style domain-conditioned sequence/query tokens."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        ads_condition_fields: Iterable[str] | str = (),
        ads_hidden_dim: int = 64,
        ads_eta: float = 2.0,
        ads_query_init_std: float = 0.0,
        ads_use_psrg: bool = True,
        ads_use_pcrg: bool = True,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        if isinstance(ads_condition_fields, str):
            condition_fields = tuple(field.strip() for field in ads_condition_fields.split(",") if field.strip())
        else:
            condition_fields = tuple(ads_condition_fields or ())
        if not condition_fields:
            condition_fields = (self.DOMAIN,)
        self.ads_condition_view = EmbView(self.omni_embedding, include_fields=condition_fields)
        self.ads_use_psrg = bool(ads_use_psrg)
        self.ads_use_pcrg = bool(ads_use_pcrg)
        self.ads_condition_fields = condition_fields
        self.ads_seq_scaling = AdaptiveDomainTokenScaling(
            token_dim=self.token_dim,
            condition_dim=self.ads_condition_view.embedding_dim,
            hidden_dim=int(ads_hidden_dim),
            eta=float(ads_eta),
        )
        self.ads_query_gen = nn.Sequential(
            nn.Linear(self.ads_condition_view.embedding_dim + self.sequence_embedding_dim, int(ads_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(ads_hidden_dim), self.num_ns_token * self.token_dim),
        )
        last = self.ads_query_gen[-1]
        if isinstance(last, nn.Linear):
            std = float(ads_query_init_std)
            if std > 0:
                nn.init.normal_(last.weight, mean=0.0, std=std)
            else:
                nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _ads_condition(self, interaction) -> torch.Tensor:
        return torch.flatten(self.ads_condition_view(interaction), start_dim=1)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = torch.flatten(self.current_view(interaction), start_dim=1)
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        condition = self._ads_condition(interaction)
        seq_tokens = self.seq_token_proj(seq_emb)
        if self.ads_use_psrg:
            seq_tokens = self.ads_seq_scaling(seq_tokens, condition)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        if self.ads_use_pcrg:
            query_delta = self.ads_query_gen(torch.cat([condition, target_emb], dim=-1))
            ns_tokens = ns_tokens + query_delta.view(target_emb.shape[0], self.num_ns_token, self.token_dim)
        _, ns_tokens = self.mixformer(seq_tokens, ns_tokens, mask)
        features = self.dropout(ns_tokens.mean(dim=1))
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        return features, domain_ids


class SeqUniXMixFormerModel(MixFormerModel):
    """MixFormer with target-domain-aware sequence tokenization and readout."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        use_target_domain_adapter: bool = True,
        use_domain_readout: bool = True,
        seq_adapter_rank: int = 32,
        seq_adapter_scale: float = 0.1,
        domain_readout_temperature: float = 1.0,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.use_target_domain_adapter = bool(use_target_domain_adapter)
        self.use_domain_readout = bool(use_domain_readout)
        self.seq_adapter_scale = float(seq_adapter_scale)
        self.domain_readout_temperature = max(float(domain_readout_temperature), 1e-6)
        self.seq_adapter_rank = max(1, min(int(seq_adapter_rank), self.token_dim))

        if self.use_target_domain_adapter:
            self.seq_adapter_down = nn.Linear(self.token_dim, self.seq_adapter_rank, bias=False)
            self.seq_adapter_up = nn.Parameter(torch.empty(int(num_domains), self.seq_adapter_rank, self.token_dim))
            self.target_domain_token_bias = nn.Embedding(int(num_domains), self.token_dim)
            nn.init.xavier_uniform_(self.seq_adapter_down.weight)
            nn.init.normal_(self.seq_adapter_up, mean=0.0, std=1e-3)
            nn.init.normal_(self.target_domain_token_bias.weight, mean=0.0, std=1e-3)
        else:
            self.seq_adapter_down = None
            self.seq_adapter_up = None
            self.target_domain_token_bias = None

        if self.use_domain_readout:
            self.domain_readout_query = nn.Embedding(int(num_domains), self.token_dim)
            nn.init.normal_(self.domain_readout_query.weight, mean=0.0, std=1e-3)
        else:
            self.domain_readout_query = None

    def _domain_ids(self, interaction) -> torch.Tensor:
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        return domain_ids.clamp(min=0, max=int(self.num_domains) - 1)

    def _adapt_sequence_tokens(self, seq_tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_target_domain_adapter:
            return seq_tokens
        low_rank = self.seq_adapter_down(seq_tokens)
        up_weight = self.seq_adapter_up[domain_ids]
        residual = torch.einsum("blr,brt->blt", low_rank, up_weight)
        residual = residual + self.target_domain_token_bias(domain_ids).unsqueeze(1)
        return seq_tokens + self.seq_adapter_scale * residual

    def _readout_ns_tokens(self, ns_tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_domain_readout:
            return ns_tokens.mean(dim=1)
        query = self.domain_readout_query(domain_ids)
        scores = (ns_tokens * query.unsqueeze(1)).sum(dim=-1)
        scores = scores / (float(self.token_dim) ** 0.5 * self.domain_readout_temperature)
        weights = torch.softmax(scores, dim=1)
        return (ns_tokens * weights.unsqueeze(-1)).sum(dim=1)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = torch.flatten(self.current_view(interaction), start_dim=1)
        domain_ids = self._domain_ids(interaction)
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = self._adapt_sequence_tokens(seq_tokens, domain_ids)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        _, ns_tokens = self.mixformer(seq_tokens, ns_tokens, mask)
        features = self.dropout(self._readout_ns_tokens(ns_tokens, domain_ids))
        return features, domain_ids


class TargetEventBiasMixFormerModel(MixFormerModel):
    """MixFormer with target/event-domain relation bias in sequence cross-attention."""

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        event_tab_seq_field: str = "tab_seq",
        target_event_bias_scale: float = 1.0,
        target_event_bias_by_ns_token: bool = False,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            num_ns_token=num_ns_token,
            dropout_rate=dropout_rate,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.event_tab_seq_field = str(event_tab_seq_field)
        self.target_event_bias_scale = float(target_event_bias_scale)
        self.target_event_bias_by_ns_token = bool(target_event_bias_by_ns_token)
        if self.target_event_bias_by_ns_token:
            self.target_event_bias = nn.Parameter(torch.zeros(int(num_domains), int(num_domains), self.num_ns_token))
        else:
            self.target_event_bias = nn.Parameter(torch.zeros(int(num_domains), int(num_domains)))

    def _target_event_attention_bias(
        self,
        interaction,
        *,
        domain_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.event_tab_seq_field not in interaction:
            raise ValueError(f"TargetEventBiasMixFormerModel requires field {self.event_tab_seq_field!r}.")
        event_tab = interaction[self.event_tab_seq_field].long()
        if event_tab.dim() != 2:
            raise ValueError(f"{self.event_tab_seq_field!r} must have shape [batch, seq_len].")
        event_domain_ids = (event_tab - int(self.domain_shift)).clamp(min=0, max=int(self.num_domains) - 1)
        target_domain_ids = domain_ids.clamp(min=0, max=int(self.num_domains) - 1)
        if self.target_event_bias_by_ns_token:
            bias = self.target_event_bias[target_domain_ids.unsqueeze(1), event_domain_ids]
            bias = bias.permute(0, 2, 1)
        else:
            bias = self.target_event_bias[target_domain_ids.unsqueeze(1), event_domain_ids]
        bias = bias * float(self.target_event_bias_scale)
        if bias.dim() == 2:
            return bias * mask.to(dtype=bias.dtype)
        return bias * mask.unsqueeze(1).to(dtype=bias.dtype)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_x = torch.flatten(self.current_view(interaction), start_dim=1)
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)

        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = self.seq_token_proj(seq_emb)
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        ns_input = torch.cat([current_x, target_emb], dim=-1)
        ns_tokens = self.tokenizer(ns_input)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        domain_ids = domain_ids.clamp(min=0, max=int(self.num_domains) - 1)
        attention_bias = self._target_event_attention_bias(interaction, domain_ids=domain_ids, mask=mask)
        _, ns_tokens = self.mixformer(seq_tokens, ns_tokens, mask, attention_bias=attention_bias)
        features = self.dropout(ns_tokens.mean(dim=1))
        return features, domain_ids
