from __future__ import annotations

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
from rosen.models.utils.activation import activation_layer
from rosen.models.utils.common import to_dims
from rosen.models.utils.sequence import AttentionSequencePoolingLayer
from rosen.models.unified.rankmixer import PerTokenFeedForward
from rosen.utils.time_bucket import RelativeTimeEmbedding, TIME_BUCKET_BOUNDARIES


def _make_norm(dim: int) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    return nn.LayerNorm(dim)


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
        raise ValueError(
            "relative_time_unit must be one of: auto, s, sec, second, seconds, ms, millisecond, milliseconds."
        )
    return divisor_by_unit[unit]


def _to_seconds(ts: torch.Tensor, divisor: int) -> torch.Tensor:
    divisor = int(divisor)
    if divisor == 1:
        return ts
    if torch.is_floating_point(ts):
        return torch.floor(ts / float(divisor))
    return torch.div(ts, divisor, rounding_mode="floor")


def _valid_mask(seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
    seq_len = seq_len.view(-1).long().clamp(min=0, max=int(max_len))
    pos = torch.arange(int(max_len), device=seq_len.device).unsqueeze(0)
    return pos < seq_len.unsqueeze(1)


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (tokens * mask_f).sum(dim=1) / denom


def _scaled_dot_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:
    score = torch.matmul(q, k.transpose(-2, -1)) * (float(q.shape[-1]) ** -0.5)
    if causal:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        offset = k_len - q_len
        q_pos = torch.arange(q_len, device=q.device).view(q_len, 1) + offset
        k_pos = torch.arange(k_len, device=q.device).view(1, k_len)
        causal_mask = k_pos <= q_pos
        score = score.masked_fill(~causal_mask.view(1, 1, q_len, k_len), -1e4)
    if attn_mask is not None:
        score = score.masked_fill(~attn_mask.to(device=score.device, dtype=torch.bool), -1e4)
    return torch.matmul(torch.softmax(score, dim=-1), v)


def _gather_last_k(tokens: torch.Tensor, seq_len: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, max_len, dim = tokens.shape
    k = max(1, int(k))
    seq_len = seq_len.view(-1).long().clamp(min=0, max=max_len)
    offsets = torch.arange(k, device=tokens.device).unsqueeze(0)
    start = (seq_len - k).clamp_min(0).unsqueeze(1)
    index = start + offsets
    valid = offsets < seq_len.clamp_max(k).unsqueeze(1)
    index = index.clamp(min=0, max=max_len - 1)
    gathered = tokens.gather(1, index.unsqueeze(-1).expand(batch_size, k, dim))
    gathered = gathered * valid.unsqueeze(-1).to(dtype=tokens.dtype)
    return gathered, valid


class MLPTokenization(nn.Module):
    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        num_tokens: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        output_dim = self.num_tokens * self.token_dim
        hidden_dim = max(output_dim * 2, self.token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            activation_layer(activation),
            nn.Dropout(float(dropout_rate)) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        out = self.mlp(x)
        return out.view(x.shape[0], self.num_tokens, self.token_dim)


class FieldTokenization(nn.Module):
    def __init__(
        self,
        field_dims: Iterable[int],
        token_dim: int,
        num_readout_tokens: int = 1,
    ):
        super().__init__()
        self.field_dims = tuple(int(dim) for dim in field_dims)
        self.token_dim = int(token_dim)
        self.projections = nn.ModuleList([nn.Linear(dim, self.token_dim, bias=False) for dim in self.field_dims])
        self.readout_tokens = nn.Parameter(torch.empty(1, int(num_readout_tokens), self.token_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.readout_tokens)

    def forward(self, field_values: Iterable[torch.Tensor]) -> torch.Tensor:
        values = list(field_values)
        if len(values) != len(self.projections):
            raise ValueError("field_values and projections have different lengths.")
        tokens = [proj(value) for proj, value in zip(self.projections, values)]
        field_tokens = torch.stack(tokens, dim=1) if tokens else self.readout_tokens.new_zeros((0, 0, self.token_dim))
        readout = self.readout_tokens.expand(field_tokens.shape[0], -1, -1)
        return torch.cat([field_tokens, readout], dim=1)


class HeteroTokenMixing(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        num_heads: int = 4,
        low_rank_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        self.head_dim = self.input_dim // self.num_heads
        self.mix_dim = self.num_tokens * self.head_dim
        self.low_rank_dim = int(low_rank_dim or max(16, self.mix_dim // 8))
        self.w_l = nn.Parameter(torch.empty(self.num_heads, self.mix_dim, self.low_rank_dim))
        self.b_l = nn.Parameter(torch.zeros(self.num_heads, self.low_rank_dim))
        self.w_r = nn.Parameter(torch.empty(self.num_heads, self.low_rank_dim, self.mix_dim))
        self.b_r = nn.Parameter(torch.zeros(self.num_heads, self.mix_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_l)
        nn.init.xavier_uniform_(self.w_r)
        nn.init.zeros_(self.b_l)
        nn.init.zeros_(self.b_r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        if num_tokens != self.num_tokens or dim != self.input_dim:
            raise ValueError(f"Expected [B,{self.num_tokens},{self.input_dim}], got {tuple(x.shape)}.")
        mixed = x.view(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        mixed = mixed.reshape(batch_size, self.num_heads, self.mix_dim)
        hidden = torch.einsum("bhf,hfr->bhr", mixed, self.w_l) + self.b_l.unsqueeze(0)
        hidden = F.relu(hidden)
        out = torch.einsum("bhr,hrf->bhf", hidden, self.w_r) + self.b_r.unsqueeze(0)
        out = out.view(batch_size, self.num_heads, num_tokens, self.head_dim).permute(0, 2, 1, 3)
        return out.contiguous().view(batch_size, num_tokens, dim)


class HeteroMixer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        num_layers: int,
        expand: int = 2,
        num_mix_heads: int = 4,
        low_rank_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.mixer_norms = nn.ModuleList([_make_norm(input_dim) for _ in range(self.num_layers)])
        self.ffn_norms = nn.ModuleList([_make_norm(input_dim) for _ in range(self.num_layers)])
        self.mixers = nn.ModuleList([
            HeteroTokenMixing(input_dim, num_tokens, num_heads=num_mix_heads, low_rank_dim=low_rank_dim)
            for _ in range(self.num_layers)
        ])
        self.ffns = nn.ModuleList([
            PerTokenFeedForward(input_dim, num_tokens, expand=expand, dropout_rate=dropout_rate)
            for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mixer_norm, mixer, ffn_norm, ffn in zip(self.mixer_norms, self.mixers, self.ffn_norms, self.ffns):
            x = x + mixer(mixer_norm(x))
            x = x + ffn(ffn_norm(x))
        return x


class MixedHeteroAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_query_tokens: int,
        attention_masking: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.num_query_tokens = int(num_query_tokens)
        self.attention_masking = bool(attention_masking)
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        self.head_dim = self.input_dim // self.num_heads
        self.w_q = nn.Parameter(torch.empty(self.num_query_tokens, self.input_dim, self.input_dim))
        self.w_k = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_v = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_o = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(self, seq_tokens: torch.Tensor, query_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, dim = seq_tokens.shape
        query_len = query_tokens.shape[1]
        if query_len > self.num_query_tokens:
            raise ValueError("query token count exceeds configured num_query_tokens.")
        if seq_len == 0:
            return query_tokens
        if mask is not None and self.attention_masking:
            mask = mask.bool()
            seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        q = torch.einsum("bqd,qdh->bqh", query_tokens, self.w_q[:query_len])
        k = self.w_k(seq_tokens)
        v = self.w_v(seq_tokens)
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        valid_rows = None
        if mask is not None and self.attention_masking:
            valid_rows = mask.any(dim=1).view(batch_size, 1, 1, 1)
            safe_mask = torch.where(mask.any(dim=1, keepdim=True), mask, torch.ones_like(mask))
            attn_mask = safe_mask.view(batch_size, 1, 1, seq_len)
        out = _scaled_dot_attention(q, k, v, attn_mask=attn_mask)
        if valid_rows is not None:
            out = out * valid_rows.to(dtype=out.dtype)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, dim)
        return self.w_o(out)


class HeMixTokenization(nn.Module):
    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        num_global_tokens: int,
        num_real_tokens: int,
        num_heads: int = 2,
        attention_masking: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_global_tokens = int(num_global_tokens)
        self.num_real_tokens = int(num_real_tokens)
        self.num_total_ns_tokens = self.num_global_tokens + self.num_real_tokens
        self.token_dim = int(token_dim)
        self.mlp = MLPTokenization(
            input_dim=input_dim,
            token_dim=token_dim,
            num_tokens=self.num_total_ns_tokens,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.global_fixed_tokens = nn.Parameter(torch.empty(1, self.num_global_tokens, self.token_dim))
        self.real_fixed_tokens = nn.Parameter(torch.empty(1, self.num_real_tokens, self.token_dim))
        self.global_attention = MixedHeteroAttention(
            input_dim=self.token_dim,
            num_heads=num_heads,
            num_query_tokens=self.num_global_tokens * 2,
            attention_masking=attention_masking,
        )
        self.real_attention = MixedHeteroAttention(
            input_dim=self.token_dim,
            num_heads=num_heads,
            num_query_tokens=self.num_real_tokens * 2,
            attention_masking=attention_masking,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.global_fixed_tokens)
        nn.init.xavier_uniform_(self.real_fixed_tokens)

    def forward(
        self,
        context: torch.Tensor,
        real_sequence_tokens: torch.Tensor,
        global_sequence_tokens: torch.Tensor,
        real_mask: Optional[torch.Tensor] = None,
        global_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        ns_tokens = self.mlp(context)
        global_ns = ns_tokens[:, :self.num_global_tokens, :]
        real_ns = ns_tokens[:, self.num_global_tokens:, :]
        global_queries = torch.cat([global_ns, self.global_fixed_tokens.expand(batch_size, -1, -1)], dim=1)
        real_queries = torch.cat([real_ns, self.real_fixed_tokens.expand(batch_size, -1, -1)], dim=1)
        global_tokens = self.global_attention(global_sequence_tokens, global_queries, global_mask)
        real_tokens = self.real_attention(real_sequence_tokens, real_queries, real_mask)
        return torch.cat([real_tokens, global_tokens, ns_tokens], dim=1)


class HiFormerAttentionLayer(nn.Module):
    def __init__(self, token_dim: int, num_tokens: int, num_heads: int = 2):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        if self.token_dim % self.num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")
        self.head_dim = self.token_dim // self.num_heads
        flat_dim = self.num_tokens * self.token_dim
        self.w_q = nn.Linear(flat_dim, flat_dim, bias=False)
        self.w_k = nn.Linear(flat_dim, flat_dim, bias=False)
        self.w_v = nn.Linear(flat_dim, flat_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        if num_tokens != self.num_tokens or dim != self.token_dim:
            raise ValueError(f"Expected [B,{self.num_tokens},{self.token_dim}], got {tuple(x.shape)}.")
        flat = x.flatten(start_dim=1)
        q = self.w_q(flat).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(flat).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(flat).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-2, -1)) * (float(self.head_dim) ** -0.5)
        out = torch.matmul(torch.softmax(score, dim=-1), v)
        return out.transpose(1, 2).contiguous().view(batch_size, num_tokens, dim)


class HiFormerBlocks(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        num_layers: int,
        expand: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.mixer_norms = nn.ModuleList([_make_norm(token_dim) for _ in range(self.num_layers)])
        self.ffn_norms = nn.ModuleList([_make_norm(token_dim) for _ in range(self.num_layers)])
        self.attention_layers = nn.ModuleList([
            HiFormerAttentionLayer(token_dim=token_dim, num_tokens=num_tokens, num_heads=num_heads)
            for _ in range(self.num_layers)
        ])
        self.ffns = nn.ModuleList([
            PerTokenFeedForward(token_dim, num_tokens, expand=expand, dropout_rate=dropout_rate)
            for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mixer_norm, attention, ffn_norm, ffn in zip(self.mixer_norms, self.attention_layers, self.ffn_norms, self.ffns):
            x = x + attention(mixer_norm(x))
            x = x + ffn(ffn_norm(x))
        return x


class CrossCausalAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, expansion_factor: int = 2, activation: str = "relu"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        self.head_dim = self.input_dim // self.num_heads
        self.w_q = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_k = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_v = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * int(expansion_factor)),
            activation_layer(activation),
            nn.Linear(self.input_dim * int(expansion_factor), self.input_dim),
        )

    def forward(self, query_tokens: torch.Tensor, key_value_tokens: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        batch_size, query_len, dim = query_tokens.shape
        key_len = key_value_tokens.shape[1]
        q = self.w_q(query_tokens).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key_value_tokens).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(key_value_tokens).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        out = _scaled_dot_attention(q, k, v, causal=bool(causal))
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, dim)
        return self.ffn(out)


class InnerTrans(nn.Module):
    def __init__(self, input_dim: int, num_groups: int = 4, expansion_factor: int = 2, activation: str = "relu"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_groups = int(num_groups)
        hidden_dim = self.input_dim * self.num_groups
        self.w_q = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_k = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_v = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * int(expansion_factor)),
            activation_layer(activation),
            nn.Linear(hidden_dim * int(expansion_factor), self.input_dim),
        )

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = seq_tokens.shape
        if seq_len % self.num_groups != 0:
            raise ValueError(f"sequence length {seq_len} must be divisible by num_groups={self.num_groups}.")
        merged_len = seq_len // self.num_groups
        grouped = seq_tokens.reshape(batch_size, merged_len, self.num_groups, dim).reshape(
            batch_size * merged_len,
            self.num_groups,
            dim,
        )
        q = self.w_q(grouped)
        k = self.w_k(grouped)
        v = self.w_v(grouped)
        out = _scaled_dot_attention(q, k, v)
        out = out.reshape(batch_size, merged_len, self.num_groups * dim)
        return self.ffn(out)


class LongerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_groups: int,
        num_layers: int,
        query_size: int,
        expansion_factor: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.query_size = int(query_size)
        self.inner_norm = _make_norm(input_dim)
        self.cross_norm = _make_norm(input_dim)
        self.self_norms = nn.ModuleList([_make_norm(input_dim) for _ in range(int(num_layers))])
        self.inner_trans = InnerTrans(input_dim, num_groups=num_groups, expansion_factor=expansion_factor, activation=activation)
        self.cross_attention = CrossCausalAttention(input_dim, num_heads=num_heads, expansion_factor=expansion_factor, activation=activation)
        self.self_attentions = nn.ModuleList([
            CrossCausalAttention(input_dim, num_heads=num_heads, expansion_factor=expansion_factor, activation=activation)
            for _ in range(int(num_layers))
        ])

    def forward(self, seq_tokens: torch.Tensor, ns_tokens: torch.Tensor, seq_len: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        merged_seq = self.inner_trans(self.inner_norm(seq_tokens))
        recent_tokens, _ = _gather_last_k(seq_tokens, seq_len, self.query_size)
        query_tokens = torch.cat([recent_tokens, ns_tokens], dim=1)
        kv_tokens = torch.cat([merged_seq, ns_tokens], dim=1)
        tokens = query_tokens + self.cross_attention(self.cross_norm(query_tokens), self.cross_norm(kv_tokens), causal=False)
        for norm, attention in zip(self.self_norms, self.self_attentions):
            tokens = tokens + attention(norm(tokens), norm(tokens), causal=True)
        return tokens


class MixedMHA(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, num_ns_tokens: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.num_ns_tokens = int(num_ns_tokens)
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        self.head_dim = self.input_dim // self.num_heads
        self.w_q_s = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_k_s = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_v_s = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_q_ns = nn.Parameter(torch.empty(self.num_ns_tokens, self.input_dim, self.input_dim))
        self.w_k_ns = nn.Parameter(torch.empty(self.num_ns_tokens, self.input_dim, self.input_dim))
        self.w_v_ns = nn.Parameter(torch.empty(self.num_ns_tokens, self.input_dim, self.input_dim))
        self.w_o = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_q_ns)
        nn.init.xavier_uniform_(self.w_k_ns)
        nn.init.xavier_uniform_(self.w_v_ns)

    def build_mask(
        self,
        batch_size: int,
        full_seq_len: int,
        query_seq_len: int,
        ns_len: int,
        device: torch.device,
        kv_mask: Optional[torch.Tensor],
        q_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        full_k_mask = (
            kv_mask.bool()
            if kv_mask is not None
            else torch.ones(batch_size, full_seq_len, dtype=torch.bool, device=device)
        )
        query_mask = (
            q_mask.bool()
            if q_mask is not None
            else torch.ones(batch_size, query_seq_len, dtype=torch.bool, device=device)
        )
        ns_k_mask = torch.ones(batch_size, ns_len, dtype=torch.bool, device=device)
        ns_q_mask = torch.ones(batch_size, ns_len, dtype=torch.bool, device=device)
        key_valid = torch.cat([full_k_mask, ns_k_mask], dim=1)
        query_valid = torch.cat([query_mask, ns_q_mask], dim=1)
        q_start = full_seq_len - query_seq_len
        q_pos = torch.cat([
            torch.arange(q_start, full_seq_len, device=device),
            torch.arange(full_seq_len, full_seq_len + ns_len, device=device),
        ], dim=0).unsqueeze(0).expand(batch_size, -1)
        k_pos = torch.cat([
            torch.arange(full_seq_len, device=device),
            torch.arange(full_seq_len, full_seq_len + ns_len, device=device),
        ], dim=0).unsqueeze(0).expand(batch_size, -1)
        causal = k_pos.unsqueeze(1) <= q_pos.unsqueeze(2)
        return (causal & query_valid.unsqueeze(-1) & key_valid.unsqueeze(1)).unsqueeze(1)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        pruned_seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
        q_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, full_seq_len, dim = seq_tokens.shape
        query_seq_len = pruned_seq_tokens.shape[1]
        ns_len = ns_tokens.shape[1]
        if kv_mask is not None:
            seq_tokens = seq_tokens * kv_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        if q_mask is not None:
            pruned_seq_tokens = pruned_seq_tokens * q_mask.unsqueeze(-1).to(dtype=pruned_seq_tokens.dtype)

        q_s = self.w_q_s(pruned_seq_tokens)
        k_s = self.w_k_s(seq_tokens)
        v_s = self.w_v_s(seq_tokens)
        q_ns = torch.einsum("bnd,ndh->bnh", ns_tokens, self.w_q_ns[:ns_len])
        k_ns = torch.einsum("bnd,ndh->bnh", ns_tokens, self.w_k_ns[:ns_len])
        v_ns = torch.einsum("bnd,ndh->bnh", ns_tokens, self.w_v_ns[:ns_len])
        q = torch.cat([q_s, q_ns], dim=1)
        k = torch.cat([k_s, k_ns], dim=1)
        v = torch.cat([v_s, v_ns], dim=1)
        total_q_len = query_seq_len + ns_len
        total_k_len = full_seq_len + ns_len
        q = q.view(batch_size, total_q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, total_k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, total_k_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = self.build_mask(batch_size, full_seq_len, query_seq_len, ns_len, seq_tokens.device, kv_mask, q_mask)
        out = _scaled_dot_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, total_q_len, dim)
        out = self.w_o(out)
        seq_out = out[:, :query_seq_len, :]
        ns_out = out[:, query_seq_len:, :]
        if q_mask is not None:
            seq_out = seq_out * q_mask.unsqueeze(-1).to(dtype=seq_out.dtype)
        return seq_out, ns_out


class MixedFFN(nn.Module):
    def __init__(self, input_dim: int, num_ns_tokens: int, activation: str = "relu", expansion_factor: int = 2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_ns_tokens = int(num_ns_tokens)
        hidden_dim = self.input_dim * int(expansion_factor)
        self.ffn_s = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            activation_layer(activation),
            nn.Linear(hidden_dim, self.input_dim),
        )
        self.w1_ns = nn.Parameter(torch.empty(self.num_ns_tokens, self.input_dim, hidden_dim))
        self.b1_ns = nn.Parameter(torch.zeros(self.num_ns_tokens, hidden_dim))
        self.w2_ns = nn.Parameter(torch.empty(self.num_ns_tokens, hidden_dim, self.input_dim))
        self.b2_ns = nn.Parameter(torch.zeros(self.num_ns_tokens, self.input_dim))
        self.activation = activation_layer(activation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w1_ns)
        nn.init.xavier_uniform_(self.w2_ns)

    def forward(self, seq_tokens: torch.Tensor, ns_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq_out = self.ffn_s(seq_tokens)
        hidden = torch.einsum("bnd,ndh->bnh", ns_tokens, self.w1_ns[:ns_tokens.shape[1]]) + self.b1_ns[:ns_tokens.shape[1]].unsqueeze(0)
        hidden = self.activation(hidden)
        ns_out = torch.einsum("bnh,nhd->bnd", hidden, self.w2_ns[:ns_tokens.shape[1]]) + self.b2_ns[:ns_tokens.shape[1]].unsqueeze(0)
        if mask is not None:
            seq_out = seq_out * mask.unsqueeze(-1).to(dtype=seq_out.dtype)
        return seq_out, ns_out


class OneTransBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        num_ns_tokens: int,
        activation: str = "relu",
        expansion_factor: int = 2,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.norm1 = nn.ModuleList([_make_norm(input_dim) for _ in range(self.num_layers)])
        self.norm2 = nn.ModuleList([_make_norm(input_dim) for _ in range(self.num_layers)])
        self.mha_layers = nn.ModuleList([
            MixedMHA(input_dim, num_heads=num_heads, num_ns_tokens=num_ns_tokens)
            for _ in range(self.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            MixedFFN(input_dim, num_ns_tokens=num_ns_tokens, activation=activation, expansion_factor=expansion_factor)
            for _ in range(self.num_layers)
        ])

    def forward(self, seq_tokens: torch.Tensor, ns_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            seq_tokens = seq_tokens * mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
            query_mask = mask
        else:
            query_mask = None
        pruned_seq = seq_tokens
        for norm1, norm2, mha, ffn in zip(self.norm1, self.norm2, self.mha_layers, self.ffn_layers):
            start = pruned_seq.shape[1] // 2
            pruned_seq = pruned_seq[:, start:, :]
            if query_mask is not None:
                query_mask = query_mask[:, start:]
            seq_norm = norm1(seq_tokens)
            pruned_norm = norm1(pruned_seq)
            ns_norm = norm1(ns_tokens)
            delta_seq, delta_ns = mha(seq_norm, pruned_norm, ns_norm, kv_mask=mask, q_mask=query_mask)
            pruned_seq = pruned_seq + delta_seq
            ns_tokens = ns_tokens + delta_ns
            pruned_norm = norm2(pruned_seq)
            ns_norm = norm2(ns_tokens)
            delta_seq, delta_ns = ffn(pruned_norm, ns_norm, query_mask)
            pruned_seq = pruned_seq + delta_seq
            ns_tokens = ns_tokens + delta_ns
            seq_tokens = pruned_seq
            mask = query_mask
        return ns_tokens


class SORTEncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        expansion_factor: int = 2,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")
        self.head_dim = self.input_dim // self.num_heads
        hidden_dim = self.input_dim * int(expansion_factor)
        self.norm1 = _make_norm(self.input_dim)
        self.norm2 = _make_norm(self.input_dim)
        self.w_q = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_k = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_v = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.w_o = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            activation_layer(activation),
            nn.Dropout(float(dropout_rate)) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.input_dim),
        )
        self.dropout = nn.Dropout(float(dropout_rate)) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, key_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        h = self.norm1(x)
        q = self.w_q(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_valid is not None:
            attn_mask = key_valid.bool().view(batch_size, 1, 1, seq_len)
        delta = _scaled_dot_attention(q, k, v, attn_mask=attn_mask)
        delta = delta.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        x = x + self.dropout(self.w_o(delta))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SORTEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        expansion_factor: int = 2,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SORTEncoderBlock(
                input_dim=input_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                activation=activation,
                dropout_rate=dropout_rate,
            )
            for _ in range(int(num_layers))
        ])
        self.norm = _make_norm(input_dim)

    def forward(self, x: torch.Tensor, key_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_valid=key_valid)
        return self.norm(x)


class UniRankSequenceBaseModel(MSRModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = manager.domain_field
        self.LABEL = manager.label_field
        if seq_group_name not in self.omni_embedding.seq_groups:
            raise ValueError(f"Unknown sequence group: {seq_group_name}")
        self.history_view = self.omni_embedding.seq_groups[seq_group_name]
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
        self.current_embedding_dim = int(self.current_view.embedding_dim)
        self.current_field_slices = self.current_view.field_slices
        self.current_field_names = tuple(field for field, _, _ in self.current_view.field_segments)
        self.sequence_embedding_dim = int(self.history_view.embedding_dim)
        self.target_field_segments = tuple(
            (setting.field_name, int(setting.embedding_dim))
            for setting in self.history_view.target_settings
        )
        self.token_dim = int(token_dim)
        self.use_relative_time = bool(use_relative_time)
        self.time_field = manager.time_field
        if self.use_relative_time:
            if not self.time_field:
                raise ValueError("use_relative_time=True requires manager.time_field to be set.")
            if not self.history_view.time_field:
                raise ValueError(
                    f"use_relative_time=True requires seq group '{seq_group_name}' to define time_field_name."
                )
            self.relative_time_to_seconds_divisor = _resolve_timestamp_to_seconds_divisor(relative_time_unit, self.time_field)
            self.relative_time_embedding = RelativeTimeEmbedding(
                embedding_dim=self.sequence_embedding_dim,
                boundaries=tuple(relative_time_boundaries or TIME_BUCKET_BOUNDARIES),
                init_std=float(relative_time_init_std),
            )
        else:
            self.relative_time_to_seconds_divisor = 1
            self.relative_time_embedding = None

        domain_setting = manager.get_setting(self.DOMAIN)
        self.domain_shift = 1 if getattr(domain_setting, "padding_zero", False) else 0

    def _current_features(self, interaction) -> torch.Tensor:
        return torch.flatten(self.current_view(interaction), start_dim=1)

    def _current_field_values(self, current_x: torch.Tensor) -> list[torch.Tensor]:
        return [current_x[:, self.current_field_slices[field_name]] for field_name in self.current_field_names]

    def _target_field_values(self, target_emb: torch.Tensor) -> list[torch.Tensor]:
        values = []
        cursor = 0
        for _, dim in self.target_field_segments:
            values.append(target_emb[:, cursor:cursor + dim])
            cursor += dim
        return values

    def _history(self, interaction) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        batch_size, max_len, _ = seq_emb.shape
        seq_len = seq_len.view(-1).long().clamp(min=0, max=max_len)
        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(curr_ts, hist_ts, seq_lens=seq_len)
        mask = _valid_mask(seq_len, max_len)
        seq_emb = seq_emb * mask.unsqueeze(-1).to(dtype=seq_emb.dtype)
        return seq_emb, target_emb, seq_len, mask

    def _domain_ids(self, interaction) -> torch.Tensor:
        return interaction[self.DOMAIN].long().view(-1) - self.domain_shift

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(x, domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


class UniRankOneTransModel(UniRankSequenceBaseModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 2,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
    ):
        super().__init__(
            manager,
            num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.item_token_proj = nn.Linear(self.sequence_embedding_dim, self.token_dim) if self.sequence_embedding_dim != self.token_dim else nn.Identity()
        self.tokenizer = MLPTokenization(
            input_dim=self.current_embedding_dim + self.sequence_embedding_dim,
            token_dim=self.token_dim,
            num_tokens=num_ns_token,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.block = OneTransBlock(
            input_dim=self.token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_ns_tokens=num_ns_token,
            activation=activation,
            expansion_factor=expansion_factor,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.token_dim * int(num_ns_token),
            hidden_dims=to_dims(tower_hidden_dims, default=(128, 64)),
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=tower_batch_norm,
        )

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        seq_emb, target_emb, _, mask = self._history(interaction)
        current_x = self._current_features(interaction)
        seq_tokens = self.item_token_proj(seq_emb)
        ns_tokens = self.tokenizer(torch.cat([current_x, target_emb], dim=-1))
        ns_tokens = self.block(seq_tokens, ns_tokens, mask)
        return self.dropout(ns_tokens.flatten(start_dim=1)), self._domain_ids(interaction)


class UniRankSORTModel(UniRankSequenceBaseModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        expansion_factor: int = 2,
        item_id_field: str = "video_id",
        item_seq_field: str = "video_id_seq",
        item_id_emb_dim: int = 64,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
    ):
        super().__init__(
            manager,
            num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.item_id_field = str(item_id_field)
        self.item_seq_field = str(item_seq_field)
        item_setting = manager.get_setting(self.item_id_field)
        if item_setting is None:
            raise ValueError(f"Unknown item_id_field: {self.item_id_field}")
        if manager.get_setting(self.item_seq_field) is None:
            raise ValueError(f"Unknown item_seq_field: {self.item_seq_field}")
        self.item_padding_idx = 0 if getattr(item_setting, "padding_zero", False) else -1
        padding_idx = self.item_padding_idx if self.item_padding_idx >= 0 else None
        num_item_embeddings = int(getattr(item_setting, "num_embeddings", -1))
        if num_item_embeddings <= 0:
            raise ValueError(f"item_id_field={self.item_id_field} is not fitted.")
        self.item_id_embedding = nn.Embedding(
            num_item_embeddings,
            int(item_id_emb_dim),
            padding_idx=padding_idx,
        )
        self.user_profile_proj = nn.Linear(self.current_embedding_dim, self.token_dim)
        self.item_aux_proj = nn.Linear(self.sequence_embedding_dim, self.token_dim)
        self.history_proj = nn.Linear(int(item_id_emb_dim) + self.token_dim, self.token_dim)
        self.candidate_proj = nn.Linear(int(item_id_emb_dim) + self.token_dim, self.token_dim)
        self.bos_token = nn.Parameter(torch.empty(1, 1, self.token_dim))
        self.sep_token = nn.Parameter(torch.empty(1, 1, self.token_dim))
        self.encoder = SORTEncoder(
            input_dim=self.token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.token_dim,
            hidden_dims=to_dims(tower_hidden_dims, default=(128, 64)),
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=tower_batch_norm,
        )
        self.reset_sort_parameters()

    def reset_sort_parameters(self) -> None:
        nn.init.xavier_uniform_(self.bos_token)
        nn.init.xavier_uniform_(self.sep_token)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        seq_emb, target_emb, seq_len, mask = self._history(interaction)
        current_x = self._current_features(interaction)
        batch_size, hist_len, _ = seq_emb.shape
        history_ids = interaction[self.item_seq_field].long().clamp_min(0)
        candidate_ids = interaction[self.item_id_field].long().view(batch_size, 1).clamp_min(0)
        if history_ids.shape[1] != hist_len:
            raise ValueError(
                f"{self.item_seq_field} length {history_ids.shape[1]} does not match history length {hist_len}."
            )
        history_ids = history_ids * mask.to(dtype=history_ids.dtype)
        history_id_emb = self.item_id_embedding(history_ids)
        candidate_id_emb = self.item_id_embedding(candidate_ids)
        history_aux = self.item_aux_proj(seq_emb)
        candidate_aux = self.item_aux_proj(target_emb).unsqueeze(1)
        history_tokens = self.history_proj(torch.cat([history_id_emb, history_aux], dim=-1))
        candidate_tokens = self.candidate_proj(torch.cat([candidate_id_emb, candidate_aux], dim=-1))
        user_token = self.user_profile_proj(current_x).unsqueeze(1)
        bos = self.bos_token.expand(batch_size, -1, -1)
        sep = self.sep_token.expand(batch_size, -1, -1)
        tokens = torch.cat([bos, history_tokens, sep, user_token, sep, candidate_tokens], dim=1)
        always_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
        key_valid = torch.cat([always_valid, mask.bool(), always_valid, always_valid, always_valid, always_valid], dim=1)
        encoded = self.encoder(tokens, key_valid=key_valid)
        candidate_out = encoded[:, -1, :]
        return self.dropout(candidate_out), self._domain_ids(interaction)


class UniRankLONGERModel(UniRankSequenceBaseModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 2,
        expansion_factor: int = 2,
        num_ns_token: int = 4,
        num_groups: int = 4,
        query_size: int = 20,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
    ):
        super().__init__(
            manager,
            num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.item_token_proj = nn.Linear(self.sequence_embedding_dim, self.token_dim) if self.sequence_embedding_dim != self.token_dim else nn.Identity()
        self.tokenizer = MLPTokenization(
            input_dim=self.current_embedding_dim + self.sequence_embedding_dim,
            token_dim=self.token_dim,
            num_tokens=num_ns_token,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.block = LongerBlock(
            input_dim=self.token_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            num_layers=num_layers,
            query_size=query_size,
            expansion_factor=expansion_factor,
            activation=activation,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.token_dim * (int(num_ns_token) + int(query_size)),
            hidden_dims=to_dims(tower_hidden_dims, default=(128, 64)),
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=tower_batch_norm,
        )

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        seq_emb, target_emb, seq_len, mask = self._history(interaction)
        current_x = self._current_features(interaction)
        seq_tokens = self.item_token_proj(seq_emb)
        ns_tokens = self.tokenizer(torch.cat([current_x, target_emb], dim=-1))
        tokens = self.block(seq_tokens, ns_tokens, seq_len, mask)
        return self.dropout(tokens.flatten(start_dim=1)), self._domain_ids(interaction)


class UniRankHeMixModel(UniRankSequenceBaseModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 2,
        expansion_factor: int = 2,
        num_global_token: int = 4,
        num_real_token: int = 4,
        num_mix_heads: int = 4,
        real_seq_len: int = 20,
        low_rank_dim: Optional[int] = None,
        attention_masking: bool = True,
        split_mode: str = "valid_last_k",
        activation: str = "relu",
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
    ):
        super().__init__(
            manager,
            num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.real_seq_len = int(real_seq_len)
        self.split_mode = str(split_mode)
        if self.split_mode not in {"valid_last_k", "tail_slice"}:
            raise ValueError("split_mode must be one of: valid_last_k, tail_slice.")
        self.item_token_proj = nn.Linear(self.sequence_embedding_dim, self.token_dim) if self.sequence_embedding_dim != self.token_dim else nn.Identity()
        self.tokenizer = HeMixTokenization(
            input_dim=self.current_embedding_dim + self.sequence_embedding_dim,
            token_dim=self.token_dim,
            num_global_tokens=num_global_token,
            num_real_tokens=num_real_token,
            num_heads=num_heads,
            attention_masking=attention_masking,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        total_ns = int(num_global_token) + int(num_real_token)
        self.block = HeteroMixer(
            input_dim=self.token_dim,
            num_tokens=total_ns * 3,
            num_layers=num_layers,
            expand=expansion_factor,
            num_mix_heads=num_mix_heads,
            low_rank_dim=low_rank_dim,
            dropout_rate=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.token_dim,
            hidden_dims=to_dims(tower_hidden_dims, default=(128, 64)),
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=tower_batch_norm,
        )

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        seq_emb, target_emb, seq_len, mask = self._history(interaction)
        current_x = self._current_features(interaction)
        seq_tokens = self.item_token_proj(seq_emb)
        if self.split_mode == "tail_slice":
            real_tokens = seq_tokens[:, -self.real_seq_len:, :]
            global_tokens = seq_tokens[:, :-self.real_seq_len, :]
            real_mask = mask[:, -self.real_seq_len:]
            global_mask = mask[:, :-self.real_seq_len]
        else:
            real_tokens, real_mask = _gather_last_k(seq_tokens, seq_len, self.real_seq_len)
            global_len = max(seq_tokens.shape[1] - self.real_seq_len, 1)
            global_tokens = seq_tokens[:, :global_len, :]
            global_valid_len = (seq_len - self.real_seq_len).clamp(min=0, max=global_len)
            global_mask = _valid_mask(global_valid_len, global_len)
        tokens = self.tokenizer(
            torch.cat([current_x, target_emb], dim=-1),
            real_tokens,
            global_tokens,
            real_mask=real_mask,
            global_mask=global_mask,
        )
        tokens = self.block(tokens)
        return self.dropout(tokens.mean(dim=1)), self._domain_ids(interaction)


class UniRankHiFormerModel(UniRankSequenceBaseModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 2,
        expansion_factor: int = 2,
        num_readout_tokens: int = 4,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        att_hidden_units: Iterable[int] = (128, 32),
        exclude_current_fields: Iterable[str] = ("action",),
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
    ):
        super().__init__(
            manager,
            num_domains,
            seq_group_name=seq_group_name,
            token_dim=token_dim,
            exclude_current_fields=exclude_current_fields,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.att_pool = AttentionSequencePoolingLayer(
            embedding_dim=self.sequence_embedding_dim,
            att_hidden_units=tuple(int(v) for v in att_hidden_units),
            att_activation="sigmoid",
        )
        field_dims = [stop - start for _, start, stop in self.current_view.field_segments]
        field_dims.extend(dim for _, dim in self.target_field_segments)
        field_dims.append(self.sequence_embedding_dim)
        self.num_context_tokens = len(field_dims) + int(num_readout_tokens)
        self.tokenizer = FieldTokenization(field_dims, token_dim=self.token_dim, num_readout_tokens=num_readout_tokens)
        self.block = HiFormerBlocks(
            token_dim=self.token_dim,
            num_tokens=self.num_context_tokens,
            num_layers=num_layers,
            expand=expansion_factor,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.num_readout_tokens = int(num_readout_tokens)
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.token_dim,
            hidden_dims=to_dims(tower_hidden_dims, default=(128, 64)),
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=tower_batch_norm,
        )

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        seq_emb, target_emb, seq_len, _ = self._history(interaction)
        current_x = self._current_features(interaction)
        pooled_history = self.att_pool(target_emb, seq_emb, seq_len)
        values = self._current_field_values(current_x)
        values.extend(self._target_field_values(target_emb))
        values.append(pooled_history)
        tokens = self.tokenizer(values)
        tokens = self.block(tokens)
        readout_tokens = tokens[:, -self.num_readout_tokens:, :]
        return self.dropout(readout_tokens.mean(dim=1)), self._domain_ids(interaction)
