from __future__ import annotations

from collections import defaultdict
import math
from typing import Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from rosen.emb import SchemaManager
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.utils.activation import activation_layer
from rosen.models.utils.common import build_mlp, to_dims


def _new_value_stats() -> Dict[str, float]:
    return {
        "batches": 0.0,
        "sample_count": 0.0,
        "value_count": 0.0,
        "sum": 0.0,
        "sq_sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "abs_dev_sum": 0.0,
        "low_count": 0.0,
        "high_count": 0.0,
        "sat_low_count": 0.0,
        "sat_high_count": 0.0,
    }


def _new_norm_stats() -> Dict[str, float]:
    return {
        "batches": 0.0,
        "sample_count": 0.0,
        "sum": 0.0,
        "sq_sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def _accumulate_values(stats: Dict[str, float], values: torch.Tensor, *, sample_count: int) -> None:
    if values.numel() == 0:
        return
    values = values.detach().float().reshape(-1)
    stats["batches"] += 1.0
    stats["sample_count"] += float(sample_count)
    stats["value_count"] += float(values.numel())
    stats["sum"] += float(values.sum().item())
    stats["sq_sum"] += float(values.pow(2).sum().item())
    stats["min"] = min(stats["min"], float(values.min().item()))
    stats["max"] = max(stats["max"], float(values.max().item()))
    stats["abs_dev_sum"] += float((values - 1.0).abs().sum().item())
    stats["low_count"] += float((values < 0.5).sum().item())
    stats["high_count"] += float((values > 1.5).sum().item())
    stats["sat_low_count"] += float((values < 0.1).sum().item())
    stats["sat_high_count"] += float((values > 1.9).sum().item())


def _accumulate_norms(stats: Dict[str, float], values: torch.Tensor, *, sample_count: int) -> None:
    if values.numel() == 0:
        return
    values = values.detach().float().reshape(-1)
    stats["batches"] += 1.0
    stats["sample_count"] += float(sample_count)
    stats["sum"] += float(values.sum().item())
    stats["sq_sum"] += float(values.pow(2).sum().item())
    stats["min"] = min(stats["min"], float(values.min().item()))
    stats["max"] = max(stats["max"], float(values.max().item()))


def _stats_mean_std(stats: Dict[str, float], count_key: str = "sample_count") -> tuple[float, float]:
    count = max(1.0, float(stats[count_key]))
    mean = float(stats["sum"]) / count
    var = max(0.0, float(stats["sq_sum"]) / count - mean ** 2)
    return mean, var ** 0.5


def _reset_last_linear_to_zero(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


class DomainInputMask(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims: Optional[Iterable[int]] = (128,),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            mode: str = "sigmoid2",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_domains = int(num_domains)
        self.mode = str(mode)
        dims = tuple(to_dims(hidden_dims, (128,))) + (self.input_dim,)
        self.gates = nn.ModuleList([
            build_mlp(
                self.input_dim,
                dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_domains)
        ])
        for gate in self.gates:
            _reset_last_linear_to_zero(gate)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        domain_ids = domain_ids.long().view(-1)
        mask = x.new_ones(x.shape)
        for domain_idx, gate in enumerate(self.gates):
            selected = domain_ids == int(domain_idx)
            if not bool(selected.any().item()):
                continue
            logits = gate(x[selected])
            if self.mode == "sigmoid2":
                values = 2.0 * torch.sigmoid(logits)
            elif self.mode == "tanh05":
                values = 1.0 + 0.5 * torch.tanh(logits)
            else:
                raise ValueError(f"Unknown mask mode={self.mode!r}.")
            mask[selected] = values
        return mask


class DomainSigmoidQuery(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_domains: int,
            hidden_dims: Optional[Iterable[int]] = (128,),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            sharpness: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_domains = int(num_domains)
        self.sharpness = float(sharpness)
        if self.sharpness <= 0.0:
            raise ValueError("DomainSigmoidQuery.sharpness must be positive.")
        dims = tuple(to_dims(hidden_dims, (128,))) + (self.output_dim,)
        self.queries = nn.ModuleList([
            build_mlp(
                self.input_dim,
                dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_domains)
        ])
        for query in self.queries:
            _reset_last_linear_to_zero(query)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        domain_ids = domain_ids.long().view(-1)
        logits = x.new_zeros(x.size(0), self.output_dim)
        for domain_idx, query in enumerate(self.queries):
            selected = domain_ids == int(domain_idx)
            if not bool(selected.any().item()):
                continue
            logits[selected] = query(x[selected])
        gate = 2.0 * torch.sigmoid(float(self.sharpness) * logits)
        return gate, logits


class SharedTowerHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: Iterable[int] = (128, 64),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        dims = tuple(to_dims(hidden_dims, (max(1, int(input_dim) // 2),))) + (1,)
        self.net = build_mlp(
            int(input_dim),
            dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor | None = None) -> torch.Tensor:
        del domain_ids
        return self.net(x).squeeze(-1)


class ExpertTokenMixerBlock(nn.Module):
    def __init__(
            self,
            num_tokens: int,
            token_dim: int,
            token_hidden_dim: Optional[int] = None,
            channel_hidden_dim: Optional[int] = None,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            zero_init_residual: bool = True,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        token_hidden_dim = int(token_hidden_dim or self.num_tokens)
        channel_hidden_dim = int(channel_hidden_dim or max(self.token_dim, self.token_dim * 2))
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.num_tokens, token_hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(token_hidden_dim, self.num_tokens),
        )
        self.channel_norm = nn.LayerNorm(self.token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.token_dim, channel_hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(channel_hidden_dim, self.token_dim),
        )
        if zero_init_residual:
            _reset_last_linear_to_zero(self.token_mlp)
            _reset_last_linear_to_zero(self.channel_mlp)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mixed = self.token_norm(tokens).transpose(1, 2)
        tokens = tokens + self.token_mlp(mixed).transpose(1, 2)
        tokens = tokens + self.channel_mlp(self.channel_norm(tokens))
        return tokens


class ExpertTokenMixer(nn.Module):
    def __init__(
            self,
            num_tokens: int,
            token_dim: int,
            num_layers: int = 1,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            zero_init_residual: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ExpertTokenMixerBlock(
                num_tokens=num_tokens,
                token_dim=token_dim,
                activation=activation,
                dropout_rate=dropout_rate,
                zero_init_residual=zero_init_residual,
            )
            for _ in range(max(1, int(num_layers)))
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


class SceneUniMixBlockLayer(nn.Module):
    def __init__(
            self,
            flat_dim: int,
            block_size: int = 8,
            temperature: float = 1.0,
            sinkhorn_iters: int = 4,
            dropout_rate: float = 0.0,
            init_diag: float = 2.0,
            init_noise_std: float = 1e-3,
    ):
        super().__init__()
        self.flat_dim = int(flat_dim)
        self.block_size = int(block_size)
        if self.flat_dim <= 0 or self.block_size <= 0:
            raise ValueError("SceneUniMixBlockLayer requires positive flat_dim and block_size.")
        if self.flat_dim % self.block_size != 0:
            raise ValueError(f"flat_dim={self.flat_dim} must be divisible by block_size={self.block_size}.")
        self.num_blocks = self.flat_dim // self.block_size
        self.temperature = max(float(temperature), 1e-4)
        self.sinkhorn_iters = max(1, int(sinkhorn_iters))
        self.local_logits = nn.Parameter(torch.empty(self.num_blocks, self.block_size, self.block_size))
        self.global_logits = nn.Parameter(torch.empty(self.num_blocks, self.num_blocks))
        self.dropout = nn.Dropout(float(dropout_rate)) if float(dropout_rate) > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(self.flat_dim)
        self.reset_parameters(init_diag=float(init_diag), init_noise_std=float(init_noise_std))

    def reset_parameters(self, *, init_diag: float = 2.0, init_noise_std: float = 1e-3) -> None:
        with torch.no_grad():
            self.local_logits.normal_(mean=0.0, std=float(init_noise_std))
            local_eye = torch.eye(self.block_size, dtype=self.local_logits.dtype, device=self.local_logits.device)
            self.local_logits.add_(float(init_diag) * local_eye.view(1, self.block_size, self.block_size))
            self.global_logits.normal_(mean=0.0, std=float(init_noise_std))
            global_eye = torch.eye(self.num_blocks, dtype=self.global_logits.dtype, device=self.global_logits.device)
            self.global_logits.add_(float(init_diag) * global_eye)

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        sym = 0.5 * (logits + logits.transpose(-1, -2))
        scaled = sym / logits.new_tensor(float(self.temperature)).clamp_min(1e-4)
        scaled = scaled - scaled.amax(dim=-1, keepdim=True)
        weights = torch.exp(scaled).clamp_min(1e-12)
        for _ in range(self.sinkhorn_iters):
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            weights = weights / weights.sum(dim=-2, keepdim=True).clamp_min(1e-12)
        return weights

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        blocks = flat.view(flat.size(0), self.num_blocks, self.block_size)
        local_w = self._sinkhorn(self.local_logits)
        global_w = self._sinkhorn(self.global_logits)
        local = torch.einsum("bni,nio->bno", blocks, local_w)
        mixed_blocks = torch.einsum("ij,bjd->bid", global_w, local)
        mixed = mixed_blocks.reshape(flat.size(0), self.flat_dim)
        return self.norm(flat + self.dropout(mixed))


class SceneUniMixBlockSource(nn.Module):
    def __init__(
            self,
            num_streams: int,
            stream_dim: int,
            output_dim: int,
            block_size: int = 8,
            num_layers: int = 1,
            temperature: float = 1.0,
            sinkhorn_iters: int = 4,
            activation: str = "relu",
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_streams = int(num_streams)
        self.stream_dim = int(stream_dim)
        self.output_dim = int(output_dim)
        self.block_size = int(block_size)
        self.flat_dim = self.num_streams * self.stream_dim
        if self.flat_dim % self.block_size != 0:
            raise ValueError(
                f"num_streams * stream_dim = {self.flat_dim} must be divisible by block_size={self.block_size}."
            )
        self.num_blocks = self.flat_dim // self.block_size
        self.layers = nn.ModuleList([
            SceneUniMixBlockLayer(
                self.flat_dim,
                block_size=self.block_size,
                temperature=temperature,
                sinkhorn_iters=sinkhorn_iters,
                dropout_rate=dropout_rate,
            )
            for _ in range(max(1, int(num_layers)))
        ])
        self.block_norm = nn.LayerNorm(self.block_size)
        self.block_projection = nn.Sequential(
            nn.Linear(self.block_size, self.output_dim),
            activation_layer(activation),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, stream_tokens: torch.Tensor) -> torch.Tensor:
        flat = stream_tokens.reshape(stream_tokens.size(0), self.flat_dim)
        for layer in self.layers:
            flat = layer(flat)
        blocks = flat.view(stream_tokens.size(0), self.num_blocks, self.block_size)
        return self.block_projection(self.block_norm(blocks))


class SceneUniMixBlockSelector(nn.Module):
    def __init__(
            self,
            num_streams: int,
            stream_dim: int,
            scene_dim: int,
            readout_dim: int,
            block_size: int = 8,
            num_layers: int = 1,
            topk: int = 8,
            temperature: float = 1.0,
            sinkhorn_iters: int = 4,
            hard_select: bool = True,
            context_readout: bool = True,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_streams = int(num_streams)
        self.stream_dim = int(stream_dim)
        self.scene_dim = int(scene_dim)
        self.readout_dim = int(readout_dim)
        self.block_size = int(block_size)
        self.flat_dim = self.num_streams * self.stream_dim
        if self.flat_dim % self.block_size != 0:
            raise ValueError(
                f"num_streams * stream_dim = {self.flat_dim} must be divisible by block_size={self.block_size}."
            )
        if self.readout_dim != self.flat_dim:
            raise ValueError(
                "SceneUniMixBlockSelector currently requires readout_dim == num_streams * stream_dim "
                f"({self.readout_dim} != {self.flat_dim})."
            )
        self.num_blocks = self.flat_dim // self.block_size
        self.topk = min(max(1, int(topk)), self.num_blocks)
        self.temperature = max(1e-4, float(temperature))
        self.hard_select = bool(hard_select)
        self.context_readout = bool(context_readout)
        self.layers = nn.ModuleList([
            SceneUniMixBlockLayer(
                self.flat_dim,
                block_size=self.block_size,
                temperature=self.temperature,
                sinkhorn_iters=sinkhorn_iters,
                dropout_rate=dropout_rate,
            )
            for _ in range(max(1, int(num_layers)))
        ])
        self.block_norm = nn.LayerNorm(self.block_size)
        self.scene_norm = nn.LayerNorm(self.scene_dim)
        self.query_proj = nn.Linear(self.scene_dim, self.block_size)
        self.key_proj = nn.Linear(self.block_size, self.block_size, bias=False)
        self.context_norm = nn.LayerNorm(self.readout_dim)
        self.context_proj = nn.Linear(self.readout_dim, self.scene_dim)
        self.dropout = nn.Dropout(float(dropout_rate)) if float(dropout_rate) > 0.0 else nn.Identity()

    def forward(
            self,
            scene_tokens: torch.Tensor,
            stream_tokens: torch.Tensor,
            dense_readout: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat = stream_tokens.reshape(stream_tokens.size(0), self.flat_dim)
        for layer in self.layers:
            flat = layer(flat)
        blocks = flat.view(stream_tokens.size(0), self.num_blocks, self.block_size)
        query_tokens = scene_tokens
        if self.context_readout and dense_readout is not None:
            context = self.context_proj(self.context_norm(dense_readout)).unsqueeze(1)
            query_tokens = query_tokens + context.to(dtype=query_tokens.dtype, device=query_tokens.device)
        query = self.query_proj(self.scene_norm(query_tokens))
        key = self.key_proj(self.block_norm(blocks))
        logits = torch.einsum("bqd,bnd->bqn", query, key) / math.sqrt(float(self.block_size))
        block_logits = logits.mean(dim=1)
        soft_prob = torch.softmax((block_logits / self.temperature).float(), dim=-1).to(dtype=block_logits.dtype)
        soft_gate = (soft_prob * float(self.topk)).clamp(max=1.0)
        if self.hard_select:
            topk_idx = torch.topk(block_logits, k=self.topk, dim=-1).indices
            hard_gate = torch.zeros_like(soft_gate)
            hard_gate.scatter_(dim=-1, index=topk_idx, value=1.0)
            gate = hard_gate + soft_gate - soft_gate.detach()
        else:
            gate = soft_gate
        selected_blocks = blocks * self.dropout(gate).unsqueeze(-1)
        selected_flat = selected_blocks.reshape(stream_tokens.size(0), self.flat_dim)
        return selected_flat, {
            "blocks": blocks,
            "gate": gate,
            "soft_gate": soft_gate,
            "soft_prob": soft_prob,
            "block_logits": block_logits,
        }


class SceneAttentionReadBlock(nn.Module):
    def __init__(
            self,
            token_dim: int,
            num_heads: int = 4,
            ffn_hidden_dim: Optional[int] = None,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            zero_init_residual: bool = True,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        if self.token_dim % self.num_heads != 0:
            raise ValueError(f"scene attention token_dim={self.token_dim} must be divisible by num_heads={self.num_heads}.")
        ffn_hidden_dim = int(ffn_hidden_dim or self.token_dim * 2)
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.source_norm = nn.LayerNorm(self.token_dim)
        self.head_dim = self.token_dim // self.num_heads
        self.attn_scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)
        self.attn_dropout = nn.Dropout(float(dropout_rate)) if float(dropout_rate) > 0.0 else nn.Identity()
        self.dropout = nn.Dropout(float(dropout_rate)) if float(dropout_rate) > 0.0 else nn.Identity()
        self.ffn_norm = nn.LayerNorm(self.token_dim)
        self.ffn = build_mlp(
            self.token_dim,
            (ffn_hidden_dim, self.token_dim),
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=False,
        )
        if zero_init_residual:
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)
            _reset_last_linear_to_zero(self.ffn)

    def forward(
            self,
            scene_tokens: torch.Tensor,
            source_tokens: torch.Tensor,
            *,
            need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = self.query_norm(scene_tokens)
        source = self.source_norm(source_tokens)
        batch_size = int(query.shape[0])
        query_len = int(query.shape[1])
        source_len = int(source.shape[1])
        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(source).view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(source).view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        read = torch.matmul(self.attn_dropout(weights), v)
        read = read.transpose(1, 2).contiguous().view(batch_size, query_len, self.token_dim)
        read = self.out_proj(read)
        scene_tokens = scene_tokens + self.dropout(read)
        scene_tokens = scene_tokens + self.dropout(self.ffn(self.ffn_norm(scene_tokens)))
        return scene_tokens, weights if need_weights else None


class FixedScaleResidual(nn.Module):
    def __init__(self, block: nn.Module, scale: float):
        super().__init__()
        self.block = block
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.block(x)


class StreamFinalMLPBasis(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: Iterable[int] = (256, 128, 64),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        dims = to_dims(hidden_dims, (256, 128, self.output_dim))
        if int(dims[-1]) != self.output_dim:
            dims = tuple(dims) + (self.output_dim,)
        self.selector1 = nn.Linear(self.input_dim, self.input_dim)
        self.selector2 = nn.Linear(self.input_dim, self.input_dim)
        nn.init.zeros_(self.selector1.weight)
        nn.init.zeros_(self.selector1.bias)
        nn.init.zeros_(self.selector2.weight)
        nn.init.zeros_(self.selector2.bias)
        self.stream1 = build_mlp(
            self.input_dim,
            dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.stream2 = build_mlp(
            self.input_dim,
            dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.fusion_norm = nn.LayerNorm(self.output_dim)

    def forward(
            self,
            x: torch.Tensor,
            *,
            reception_gate: torch.Tensor | None = None,
            input_mask: torch.Tensor | None = None,
            pre_fusion_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if input_mask is not None:
            x = x * input_mask
        gate1 = 2.0 * torch.sigmoid(self.selector1(x))
        gate2 = 2.0 * torch.sigmoid(self.selector2(x))
        stream1 = self.stream1(x * gate1)
        stream2 = self.stream2(x * gate2)
        if pre_fusion_mask is not None:
            stream1 = stream1 * pre_fusion_mask
        fused = self.fusion_norm(stream1 * stream2)
        if reception_gate is not None:
            fused = fused * reception_gate
        return fused, {
            "stream1": stream1,
            "stream2": stream2,
            "fused": fused,
            "selection_gate_1": gate1,
            "selection_gate_2": gate2,
        }


class StreamLowRankQuadraticBasis(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            cross_rank: int = 64,
            hidden_dims: Iterable[int] = (128, 64),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            use_linear_lowrank: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.cross_rank = int(cross_rank)
        self.use_linear_lowrank = bool(use_linear_lowrank)
        self.left = nn.Linear(self.input_dim, self.cross_rank, bias=False)
        self.right = nn.Linear(self.input_dim, self.cross_rank, bias=False)
        self.linear_lowrank = (
            nn.Linear(self.input_dim, self.cross_rank, bias=False)
            if self.use_linear_lowrank
            else None
        )
        self.cross_norm = nn.LayerNorm(self.cross_rank)
        cross_input_dim = self.cross_rank * (2 if self.use_linear_lowrank else 1)
        dims = to_dims(hidden_dims, (128, self.output_dim))
        if int(dims[-1]) != self.output_dim:
            dims = tuple(dims) + (self.output_dim,)
        self.cross_mlp = build_mlp(
            cross_input_dim,
            dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def forward(
            self,
            x: torch.Tensor,
            *,
            reception_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        left = self.left(x)
        right = self.right(x)
        quadratic = self.cross_norm(left * right)
        gated_quadratic = quadratic if reception_gate is None else quadratic * reception_gate
        if self.linear_lowrank is None:
            cross_input = gated_quadratic
            linear = x.new_empty(0)
        else:
            linear = self.linear_lowrank(x)
            cross_input = torch.cat([linear, gated_quadratic], dim=-1)
        out = self.cross_mlp(cross_input)
        return out, {
            "stream1": left,
            "stream2": right,
            "fused": gated_quadratic,
            "linear_factor": linear,
            "quadratic_factor": quadratic,
        }


class StreamReceptionGate(nn.Module):
    def __init__(
            self,
            mode: str,
            *,
            num_domains: int,
            num_streams: int,
            dim: int,
            hard_threshold: float = 0.5,
            init_logit: float = 0.05,
            hard_topk: int = 0,
            init_noise_std: float = 0.0,
    ):
        super().__init__()
        self.mode = str(mode).strip().lower()
        self.num_domains = int(num_domains)
        self.num_streams = int(num_streams)
        self.dim = int(dim)
        self.hard_threshold = float(hard_threshold)
        self.hard_topk = max(1, min(self.dim, int(hard_topk or max(1, self.dim // 2))))
        if self.mode in {"", "none", "off", "false"}:
            self.mode = "none"
            self.logits = None
        elif self.mode == "shared_soft_gate":
            self.logits = nn.Parameter(torch.zeros(self.dim))
        elif self.mode == "stream_sep_soft_gate":
            self.logits = nn.Parameter(torch.zeros(self.num_streams, self.dim))
        elif self.mode == "domain_sep_soft_gate":
            self.logits = nn.Parameter(torch.zeros(self.num_domains, self.dim))
        elif self.mode == "domain_hard_rank_gate_sqrt":
            self.logits = nn.Parameter(torch.full((self.num_domains, self.dim), float(init_logit)))
        elif self.mode == "domain_hard_topk_gate":
            initial = torch.full((self.num_domains, self.dim), float(init_logit))
            if float(init_noise_std) > 0.0:
                initial = initial + torch.randn_like(initial) * float(init_noise_std)
            self.logits = nn.Parameter(initial)
        else:
            raise ValueError(f"Unknown gate_reception={mode!r}.")

    @property
    def is_hard(self) -> bool:
        return self.mode in {"domain_hard_rank_gate_sqrt", "domain_hard_topk_gate"}

    def _hard_mask_from_soft(self, soft: torch.Tensor) -> torch.Tensor:
        if self.mode == "domain_hard_topk_gate":
            k = max(1, min(int(soft.shape[-1]), int(self.hard_topk)))
            _, indices = torch.topk(soft, k=k, dim=-1)
            hard = torch.zeros_like(soft)
            return hard.scatter(dim=-1, index=indices, value=1.0)
        return (soft >= self.hard_threshold).to(soft.dtype)

    def _select_logits(self, domain_ids: torch.Tensor, stream_idx: int, batch_size: int) -> torch.Tensor | None:
        if self.logits is None:
            return None
        if self.mode == "shared_soft_gate":
            return self.logits.view(1, -1).expand(batch_size, -1)
        if self.mode == "stream_sep_soft_gate":
            return self.logits[int(stream_idx)].view(1, -1).expand(batch_size, -1)
        domain_ids = domain_ids.long().view(-1).to(self.logits.device)
        return self.logits[domain_ids]

    def forward(
            self,
            domain_ids: torch.Tensor,
            *,
            stream_idx: int,
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
            structure_ste: bool = True,
            detach_structure: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        logits = self._select_logits(domain_ids, stream_idx, batch_size)
        if logits is None:
            return None, None, None
        logits = logits.to(device=device, dtype=dtype)
        soft = torch.sigmoid(logits)
        if self.is_hard:
            hard = self._hard_mask_from_soft(soft).to(dtype)
            if structure_ste and not detach_structure:
                gate = hard + soft - soft.detach()
            else:
                gate = hard.detach()
            return gate, hard, soft
        gate = 2.0 * soft
        return gate, None, soft

    def static_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if self.logits is None:
            return rows
        with torch.no_grad():
            logits = self.logits.detach().float()
            if self.mode == "shared_soft_gate":
                soft = torch.sigmoid(logits)
                gate = 2.0 * soft
                rows.append({
                    "stat_group": "reception_gate",
                    "gate_reception": self.mode,
                    "domain": -1,
                    "stream": -1,
                    "rank_dim": int(self.dim),
                    "gate_mean": float(gate.mean().item()),
                    "gate_std": float(gate.std(unbiased=False).item()),
                    "gate_min": float(gate.min().item()),
                    "gate_max": float(gate.max().item()),
                    "gate_abs_dev_from_1": float((gate - 1.0).abs().mean().item()),
                })
            elif self.mode == "stream_sep_soft_gate":
                gate = 2.0 * torch.sigmoid(logits)
                for stream_idx, values in enumerate(gate):
                    rows.append({
                        "stat_group": "reception_gate",
                        "gate_reception": self.mode,
                        "domain": -1,
                        "stream": int(stream_idx),
                        "rank_dim": int(self.dim),
                        "gate_mean": float(values.mean().item()),
                        "gate_std": float(values.std(unbiased=False).item()),
                        "gate_min": float(values.min().item()),
                        "gate_max": float(values.max().item()),
                        "gate_abs_dev_from_1": float((values - 1.0).abs().mean().item()),
                    })
                for left in range(int(gate.shape[0])):
                    for right in range(left + 1, int(gate.shape[0])):
                        left_values = gate[left].view(-1)
                        right_values = gate[right].view(-1)
                        rows.append({
                            "stat_group": "reception_gate_pair",
                            "gate_reception": self.mode,
                            "stream_i": int(left),
                            "stream_j": int(right),
                            "gate_cosine": float(F.cosine_similarity(
                                left_values.view(1, -1),
                                right_values.view(1, -1),
                                dim=-1,
                            ).item()),
                            "gate_l1_distance": float((left_values - right_values).abs().mean().item()),
                            "gate_l2_distance": float((left_values - right_values).pow(2).mean().sqrt().item()),
                        })
            elif self.mode == "domain_sep_soft_gate":
                gate = 2.0 * torch.sigmoid(logits)
                for domain_idx, values in enumerate(gate):
                    rows.append({
                        "stat_group": "reception_gate",
                        "gate_reception": self.mode,
                        "domain": int(domain_idx),
                        "stream": -1,
                        "rank_dim": int(self.dim),
                        "gate_mean": float(values.mean().item()),
                        "gate_std": float(values.std(unbiased=False).item()),
                        "gate_min": float(values.min().item()),
                        "gate_max": float(values.max().item()),
                        "gate_abs_dev_from_1": float((values - 1.0).abs().mean().item()),
                    })
                for left in range(int(gate.shape[0])):
                    for right in range(left + 1, int(gate.shape[0])):
                        left_values = gate[left].view(-1)
                        right_values = gate[right].view(-1)
                        rows.append({
                            "stat_group": "reception_gate_pair",
                            "gate_reception": self.mode,
                            "domain_i": int(left),
                            "domain_j": int(right),
                            "gate_cosine": float(F.cosine_similarity(
                                left_values.view(1, -1),
                                right_values.view(1, -1),
                                dim=-1,
                            ).item()),
                            "gate_l1_distance": float((left_values - right_values).abs().mean().item()),
                            "gate_l2_distance": float((left_values - right_values).pow(2).mean().sqrt().item()),
                        })
            elif self.mode in {"domain_hard_rank_gate_sqrt", "domain_hard_topk_gate"}:
                soft = torch.sigmoid(logits)
                hard = self._hard_mask_from_soft(soft)
                for domain_idx, values in enumerate(hard):
                    soft_values = soft[domain_idx]
                    rows.append({
                        "stat_group": "hard_rank_gate",
                        "gate_reception": self.mode,
                        "domain": int(domain_idx),
                        "stream": -1,
                        "rank_dim": int(self.dim),
                        "topk": int(self.hard_topk) if self.mode == "domain_hard_topk_gate" else -1,
                        "keep_count": int(values.sum().item()),
                        "keep_frac": float(values.mean().item()),
                        "flip_rate": 0.0,
                        "surrogate_prob_mean": float(soft_values.mean().item()),
                        "surrogate_prob_std": float(soft_values.std(unbiased=False).item()),
                        "hard_soft_gap": float((values - soft_values).abs().mean().item()),
                        "sqrt_grad_scale": 1.0,
                    })
                for left in range(self.num_domains):
                    for right in range(left + 1, self.num_domains):
                        left_hard = hard[left].bool()
                        right_hard = hard[right].bool()
                        overlap = int((left_hard & right_hard).sum().item())
                        union = int((left_hard | right_hard).sum().item())
                        rows.append({
                            "stat_group": "hard_rank_gate_pair",
                            "gate_reception": self.mode,
                            "domain_i": int(left),
                            "domain_j": int(right),
                            "topk": int(self.hard_topk) if self.mode == "domain_hard_topk_gate" else -1,
                            "jaccard": float(overlap / max(1, union)),
                            "overlap_count": overlap,
                            "exclusive_i_count": int((left_hard & ~right_hard).sum().item()),
                            "exclusive_j_count": int((right_hard & ~left_hard).sum().item()),
                        })
        return rows


class AllDomainExpertMixerModel(MSRModel):
    """
    All-domain basis experts with optional domain input masks and expert-token mixing.

    Every expert receives every sample. Domain-specific parameters act as an input
    lens, not as an expert router.
    """

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (256, 128, 64),
            num_experts: int = 5,
            architecture: str = "all_expert",
            cross_basis: str = "mlp",
            gate_reception: str = "none",
            cross_rank: int = 64,
            use_linear_lowrank: bool = True,
            hard_gate_init_logit: float = 0.05,
            hard_gate_threshold: float = 0.5,
            hard_gate_topk: int = 0,
            hard_gate_init_noise_std: float = 0.0,
            readout_mode: str = "flatten",
            readout_projection_dim: int = 0,
            residual_side_mode: str = "none",
            residual_readout_projection_dim: int = 0,
            residual_beta_init: float = 0.2,
            residual_beta_max: float = 1.0,
            residual_beta_mode: str = "domain",
            residual_side_aux_weight: float = 0.0,
            residual_side_aux_mode: str = "side_bce",
            residual_side_combine: str = "raw",
            use_domain_mask: bool = True,
            use_token_mixer: bool = True,
            shared_lower_dims: Optional[Iterable[int]] = None,
            private_lower_dims: Optional[Iterable[int]] = None,
            shared_middle_dims: Optional[Iterable[int]] = None,
            shared_middle_mode: str = "mlp",
            shared_middle_residual_scale: float = 0.05,
            pre_shared_merge_mode: str = "none",
            post_shared_interest_mode: str = "identity",
            post_shared_interest_dims: Optional[Iterable[int]] = None,
            post_shared_interest_residual_scale: float = 0.1,
            finalmlp_domain_mask_mode: str = "none",
            finalmlp_domain_mask_sharpness: float = 1.0,
            finalmlp_domain_mask_reg_weight: float = 0.0,
            scene_query_mode: str = "none",
            scene_query_hidden_dims: Optional[Iterable[int]] = (128,),
            scene_query_fusion_mode: str = "add_delta",
            scene_query_sharpness: float = 1.0,
            scene_query_detach_backbone: bool = True,
            scene_token_mode: str = "none",
            scene_token_stages: str | Iterable[str] = "before_shared,after_shared,after_interest",
            scene_num_global_tokens: int = 1,
            scene_num_domain_tokens: int = 2,
            scene_token_num_heads: int = 4,
            scene_token_mixer_layers: int = 1,
            scene_token_source_mode: str = "tokenmix",
            scene_token_unimix_block_size: int = 8,
            scene_token_unimix_temperature: float = 1.0,
            scene_token_unimix_sinkhorn_iters: int = 4,
            scene_token_detach_backbone: bool = True,
            scene_token_block_select_topk: int = 8,
            scene_token_block_select_temperature: float = 1.0,
            scene_token_block_select_hard: bool = True,
            scene_token_block_select_context: bool = True,
            scene_token_fusion_mode: str = "add_readout",
            scene_token_fusion_scale: float = 0.1,
            scene_token_zero_init_residual: bool = True,
            scene_token_zero_init_fusion: bool = True,
            scene_token_fusion_init_std: float = 0.0,
            scene_token_context_mode: str = "none",
            scene_token_context_hidden_dims: Optional[Iterable[int]] = (128,),
            scene_token_context_scale: float = 0.1,
            scene_token_context_detach_input: bool = False,
            prediction_head_mode: str = "domain_tower",
            mask_hidden_dims: Optional[Iterable[int]] = (128,),
            mask_mode: str = "sigmoid2",
            mask_reg_weight: float = 0.0,
            expert_aux_weight: float = 0.05,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            mixer_layers: int = 1,
            mixer_dropout_rate: float = 0.0,
            tower_hidden_dims: Iterable[int] = (128, 64),
            tower_dropout_rate: float = 0.0,
            include_domain_in_input: bool = True,
            monitor_step_interval: int = 0,
            record_step_interval: int = 0,
            cross_gate_ste_update_interval: int = 0,
            cross_gate_ste_lr: float = 0.0,
            hard_gate_twopass_sqrt_count: bool = False,
            hard_gate_update_interval: int = 0,
            hard_gate_structure_lr: float = 0.0,
    ):
        del (
            monitor_step_interval,
            cross_gate_ste_update_interval,
            cross_gate_ste_lr,
            hard_gate_twopass_sqrt_count,
            hard_gate_update_interval,
            hard_gate_structure_lr,
        )
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.include_domain_in_input = bool(include_domain_in_input)
        self.input_view = (
            self.omni_embedding.whole if self.include_domain_in_input
            else self.omni_embedding.whole_without_domain
        )
        self.input_dim = self.input_view.embedding_dim
        self.num_experts = int(num_experts)
        self.architecture = str(architecture).lower()
        if self.architecture not in {"all_expert", "mid_private", "xshape"}:
            raise ValueError(f"Unknown all-expert architecture={architecture!r}.")
        self.cross_basis = str(cross_basis).strip().lower()
        if self.cross_basis in {"", "plain"}:
            self.cross_basis = "mlp"
        if self.cross_basis not in {"mlp", "finalmlp_swiglu", "lq_rank64"}:
            raise ValueError(f"Unknown all-expert cross_basis={cross_basis!r}.")
        self.gate_reception = str(gate_reception).strip().lower()
        if self.gate_reception in {"", "off", "false"}:
            self.gate_reception = "none"
        readout_mode = str(readout_mode).lower()
        if readout_mode in {"concat", "flat"}:
            readout_mode = "flatten"
        if readout_mode in {"mean_pool", "pool", "avg"}:
            readout_mode = "mean"
        if readout_mode in {"split_mean", "stream_head_mean"}:
            readout_mode = "split_head_mean"
        if readout_mode in {"split_domain_weight", "stream_head_domain_weight"}:
            readout_mode = "split_head_domain_weight"
        if readout_mode not in {"flatten", "mean", "split_head_mean", "split_head_domain_weight"}:
            raise ValueError(f"Unknown all-expert readout_mode={readout_mode!r}.")
        self.readout_mode = readout_mode
        self.readout_projection_dim = int(readout_projection_dim or 0)
        self.residual_side_mode = str(residual_side_mode).strip().lower()
        if self.residual_side_mode in {"", "off", "false"}:
            self.residual_side_mode = "none"
        if self.residual_side_mode in {"readout64", "readout_projection64", "lowrank_readout"}:
            self.residual_side_mode = "readout_projection"
            if int(residual_readout_projection_dim or 0) <= 0:
                residual_readout_projection_dim = 64
        if self.residual_side_mode not in {"none", "readout_projection"}:
            raise ValueError(f"Unknown residual_side_mode={residual_side_mode!r}.")
        self.residual_readout_projection_dim = int(residual_readout_projection_dim or 0)
        self.residual_beta_init = float(residual_beta_init)
        self.residual_beta_max = float(residual_beta_max)
        self.residual_beta_mode = str(residual_beta_mode).strip().lower()
        if self.residual_beta_mode in {"", "static", "domain_static", "domain_scalar"}:
            self.residual_beta_mode = "domain"
        if self.residual_beta_mode in {"sample", "sample_readout", "sample_dense_readout"}:
            self.residual_beta_mode = "sample_dense"
        if self.residual_beta_mode not in {"domain", "sample_dense"}:
            raise ValueError(f"Unknown residual_beta_mode={residual_beta_mode!r}.")
        self.residual_side_aux_weight = float(residual_side_aux_weight)
        self.residual_side_aux_mode = str(residual_side_aux_mode).strip().lower()
        if self.residual_side_aux_mode in {"", "side", "side_logit", "side_logits"}:
            self.residual_side_aux_mode = "side_bce"
        if self.residual_side_aux_mode in {
            "dense_residual",
            "dense_residual_bce",
            "dense_stopgrad_residual",
            "dense_stopgrad_residual_bce",
        }:
            self.residual_side_aux_mode = "dense_residual_bce"
        if self.residual_side_aux_mode not in {"side_bce", "dense_residual_bce"}:
            raise ValueError(f"Unknown residual_side_aux_mode={residual_side_aux_mode!r}.")
        self.residual_side_combine = str(residual_side_combine).strip().lower()
        if self.residual_side_combine in {"", "side", "side_logit", "side_logits"}:
            self.residual_side_combine = "raw"
        if self.residual_side_combine in {"center", "centered_side", "side_centered"}:
            self.residual_side_combine = "centered"
        if self.residual_side_combine in {"side_minus_dense", "side_minus_dense_detach", "smdetach"}:
            self.residual_side_combine = "side_minus_dense_stopgrad"
        if self.residual_side_combine not in {"raw", "centered", "side_minus_dense_stopgrad"}:
            raise ValueError(f"Unknown residual_side_combine={residual_side_combine!r}.")
        self.use_domain_mask = bool(use_domain_mask)
        self.use_token_mixer = bool(use_token_mixer)
        self.shared_middle_mode = str(shared_middle_mode).strip().lower()
        if self.shared_middle_mode in {"", "default"}:
            self.shared_middle_mode = "mlp"
        if self.shared_middle_mode in {"none", "off", "no_shared", "no_shared_identity"}:
            self.shared_middle_mode = "identity"
        if self.shared_middle_mode in {"residual", "small_residual"}:
            self.shared_middle_mode = "residual_small"
        if self.shared_middle_mode not in {"mlp", "linear", "identity", "residual_small"}:
            raise ValueError(f"Unknown shared_middle_mode={shared_middle_mode!r}.")
        self.shared_middle_residual_scale = float(shared_middle_residual_scale)
        self.pre_shared_merge_mode = str(pre_shared_merge_mode).strip().lower()
        if self.pre_shared_merge_mode in {"", "off", "false", "no", "none"}:
            self.pre_shared_merge_mode = "none"
        if self.pre_shared_merge_mode in {"avg", "average"}:
            self.pre_shared_merge_mode = "mean"
        if self.pre_shared_merge_mode in {"sum_sqrt5", "sqrt_sum", "sqrt"}:
            self.pre_shared_merge_mode = "sum_sqrt"
        if self.pre_shared_merge_mode not in {"none", "mean", "sum_sqrt"}:
            raise ValueError(f"Unknown pre_shared_merge_mode={pre_shared_merge_mode!r}.")
        if self.pre_shared_merge_mode != "none" and self.architecture != "xshape":
            raise ValueError("pre_shared_merge_mode is only supported for architecture='xshape'.")
        self.post_shared_interest_mode = str(post_shared_interest_mode).strip().lower()
        if self.post_shared_interest_mode in {"", "none", "off", "false", "identity"}:
            self.post_shared_interest_mode = "identity"
        if self.post_shared_interest_mode in {"private", "private_mlp", "private_ffn"}:
            self.post_shared_interest_mode = "private_ffn"
        if self.post_shared_interest_mode in {"private_residual", "private_residual_ffn", "residual_ffn"}:
            self.post_shared_interest_mode = "private_residual_ffn"
        if self.post_shared_interest_mode in {"shared", "shared_mlp", "shared_ffn"}:
            self.post_shared_interest_mode = "shared_ffn"
        if self.post_shared_interest_mode not in {"identity", "private_ffn", "private_residual_ffn", "shared_ffn"}:
            raise ValueError(f"Unknown post_shared_interest_mode={post_shared_interest_mode!r}.")
        if self.post_shared_interest_mode != "identity" and self.architecture != "xshape":
            raise ValueError("post_shared_interest_mode is only supported for architecture='xshape'.")
        if self.post_shared_interest_mode != "identity" and self.pre_shared_merge_mode != "none":
            raise ValueError("post_shared_interest_mode currently requires pre_shared_merge_mode='none'.")
        self.post_shared_interest_residual_scale = float(post_shared_interest_residual_scale)
        self.finalmlp_domain_mask_mode = str(finalmlp_domain_mask_mode).strip().lower()
        if self.finalmlp_domain_mask_mode in {"", "off", "false", "no", "none"}:
            self.finalmlp_domain_mask_mode = "none"
        if self.finalmlp_domain_mask_mode in {"input", "feature", "input_feature", "feature_input"}:
            self.finalmlp_domain_mask_mode = "input_feature"
        if self.finalmlp_domain_mask_mode in {
            "fusion",
            "pre_fusion",
            "prefusion",
            "pre_product",
            "product",
            "fusion_operand",
            "operand",
        }:
            self.finalmlp_domain_mask_mode = "pre_fusion"
        if self.finalmlp_domain_mask_mode in {
            "both",
            "input_fusion",
            "input_prefusion",
            "input_pre_fusion",
            "input_feature_pre_fusion",
        }:
            self.finalmlp_domain_mask_mode = "input_feature_pre_fusion"
        if self.finalmlp_domain_mask_mode not in {"none", "input_feature", "pre_fusion", "input_feature_pre_fusion"}:
            raise ValueError(f"Unknown finalmlp_domain_mask_mode={finalmlp_domain_mask_mode!r}.")
        if self.finalmlp_domain_mask_mode != "none" and self.cross_basis != "finalmlp_swiglu":
            raise ValueError("finalmlp_domain_mask_mode requires cross_basis='finalmlp_swiglu'.")
        self.finalmlp_domain_mask_sharpness = float(finalmlp_domain_mask_sharpness)
        if self.finalmlp_domain_mask_sharpness <= 0.0:
            raise ValueError("finalmlp_domain_mask_sharpness must be positive.")
        self.finalmlp_domain_mask_reg_weight = float(finalmlp_domain_mask_reg_weight)
        self.scene_query_mode = str(scene_query_mode).strip().lower()
        if self.scene_query_mode in {"", "off", "false", "no", "none"}:
            self.scene_query_mode = "none"
        if self.scene_query_mode in {"sigmoid", "sigmoid_query", "scene_sigmoid", "scene_sigmoid_query"}:
            self.scene_query_mode = "sigmoid_query"
        if self.scene_query_mode not in {"none", "sigmoid_query"}:
            raise ValueError(f"Unknown scene_query_mode={scene_query_mode!r}.")
        self.scene_query_fusion_mode = str(scene_query_fusion_mode).strip().lower()
        if self.scene_query_fusion_mode in {"", "add", "residual", "additive"}:
            self.scene_query_fusion_mode = "add_delta"
        if self.scene_query_fusion_mode in {"concat", "cat"}:
            self.scene_query_fusion_mode = "concat_delta"
        if self.scene_query_fusion_mode not in {"add_delta", "concat_delta"}:
            raise ValueError(f"Unknown scene_query_fusion_mode={scene_query_fusion_mode!r}.")
        self.scene_query_sharpness = float(scene_query_sharpness)
        if self.scene_query_sharpness <= 0.0:
            raise ValueError("scene_query_sharpness must be positive.")
        self.scene_query_detach_backbone = bool(scene_query_detach_backbone)
        self.scene_token_mode = str(scene_token_mode).strip().lower()
        if self.scene_token_mode in {"", "off", "false", "no", "none"}:
            self.scene_token_mode = "none"
        if self.scene_token_mode in {"attn", "attention", "attn_multitoken", "multi_token_attention"}:
            self.scene_token_mode = "attn_multitoken"
        if self.scene_token_mode not in {"none", "attn_multitoken"}:
            raise ValueError(f"Unknown scene_token_mode={scene_token_mode!r}.")
        if self.scene_token_mode != "none" and self.architecture != "xshape":
            raise ValueError("scene_token_mode currently requires architecture='xshape'.")
        explicit_no_scene_read = False
        if isinstance(scene_token_stages, str):
            stage_text = scene_token_stages.strip().lower()
            explicit_no_scene_read = stage_text in {"", "none", "off", "false", "no", "no_read"}
            raw_stages = (
                tuple()
                if explicit_no_scene_read
                else tuple(part.strip().lower() for part in scene_token_stages.split(",") if part.strip())
            )
        else:
            raw_stages = tuple(str(part).strip().lower() for part in scene_token_stages if str(part).strip())
        if not raw_stages and not explicit_no_scene_read:
            raw_stages = ("after_interest",)
        valid_scene_stages = {"before_shared", "after_shared", "after_interest", "before_readout"}
        unknown_scene_stages = sorted(set(raw_stages) - valid_scene_stages)
        if unknown_scene_stages:
            raise ValueError(f"Unknown scene_token_stages={unknown_scene_stages!r}.")
        self.scene_token_stages = raw_stages
        self.scene_num_global_tokens = max(0, int(scene_num_global_tokens))
        self.scene_num_domain_tokens = max(0, int(scene_num_domain_tokens))
        if self.scene_token_mode != "none" and self.scene_num_global_tokens + self.scene_num_domain_tokens <= 0:
            raise ValueError("scene_token_mode requires at least one global or domain scene token.")
        self.scene_token_num_heads = int(scene_token_num_heads)
        self.scene_token_mixer_layers = max(1, int(scene_token_mixer_layers))
        self.scene_token_source_mode = str(scene_token_source_mode).strip().lower()
        if self.scene_token_source_mode in {"", "tokenmix", "stream_tokenmix", "stream", "expert_tokenmix"}:
            self.scene_token_source_mode = "tokenmix"
        if self.scene_token_source_mode in {"unimix", "unimix_block", "unimix_blocks", "block_unimix", "blocks"}:
            self.scene_token_source_mode = "unimix_blocks"
        if self.scene_token_source_mode not in {"tokenmix", "unimix_blocks"}:
            raise ValueError(f"Unknown scene_token_source_mode={scene_token_source_mode!r}.")
        self.scene_token_unimix_block_size = max(1, int(scene_token_unimix_block_size))
        self.scene_token_unimix_temperature = max(1e-4, float(scene_token_unimix_temperature))
        self.scene_token_unimix_sinkhorn_iters = max(1, int(scene_token_unimix_sinkhorn_iters))
        self.scene_token_detach_backbone = bool(scene_token_detach_backbone)
        self.scene_token_block_select_topk = max(1, int(scene_token_block_select_topk))
        self.scene_token_block_select_temperature = max(1e-4, float(scene_token_block_select_temperature))
        self.scene_token_block_select_hard = bool(scene_token_block_select_hard)
        self.scene_token_block_select_context = bool(scene_token_block_select_context)
        self.scene_token_fusion_mode = str(scene_token_fusion_mode).strip().lower()
        if self.scene_token_fusion_mode in {"", "add", "residual", "additive", "add_readout"}:
            self.scene_token_fusion_mode = "add_readout"
        if self.scene_token_fusion_mode in {"concat", "cat", "concat_readout"}:
            self.scene_token_fusion_mode = "concat_readout"
        if self.scene_token_fusion_mode in {"block_select", "block_select_readout", "select_blocks"}:
            self.scene_token_fusion_mode = "block_select_readout"
        if self.scene_token_fusion_mode not in {"add_readout", "concat_readout", "block_select_readout"}:
            raise ValueError(f"Unknown scene_token_fusion_mode={scene_token_fusion_mode!r}.")
        self.scene_token_fusion_scale = float(scene_token_fusion_scale)
        self.scene_token_zero_init_residual = bool(scene_token_zero_init_residual)
        self.scene_token_zero_init_fusion = bool(scene_token_zero_init_fusion)
        self.scene_token_fusion_init_std = max(0.0, float(scene_token_fusion_init_std))
        self.scene_token_context_mode = str(scene_token_context_mode).strip().lower()
        if self.scene_token_context_mode in {"", "off", "false", "no", "none"}:
            self.scene_token_context_mode = "none"
        if self.scene_token_context_mode in {"raw", "input", "raw_input", "raw_input_add", "x", "x_add"}:
            self.scene_token_context_mode = "raw_input_add"
        if self.scene_token_context_mode not in {"none", "raw_input_add"}:
            raise ValueError(f"Unknown scene_token_context_mode={scene_token_context_mode!r}.")
        if self.scene_token_context_mode != "none" and self.scene_token_mode == "none":
            raise ValueError("scene_token_context_mode requires scene_token_mode to be enabled.")
        if self.scene_token_context_mode != "none" and self.scene_num_domain_tokens <= 0:
            raise ValueError("scene_token_context_mode currently conditions domain scene tokens, so scene_num_domain_tokens must be positive.")
        self.scene_token_context_scale = float(scene_token_context_scale)
        self.scene_token_context_detach_input = bool(scene_token_context_detach_input)
        self.prediction_head_mode = str(prediction_head_mode).strip().lower()
        if self.prediction_head_mode in {"", "domain", "domain_tower", "domain_head"}:
            self.prediction_head_mode = "domain_tower"
        if self.prediction_head_mode in {"shared", "shared_tower", "single_tower", "single"}:
            self.prediction_head_mode = "shared_tower"
        if self.prediction_head_mode not in {"domain_tower", "shared_tower"}:
            raise ValueError(f"Unknown prediction_head_mode={prediction_head_mode!r}.")
        self.readout_num_tokens = (
            1
            if self.architecture == "xshape" and self.pre_shared_merge_mode != "none"
            else self.num_experts
        )
        self.mask_reg_weight = float(mask_reg_weight)
        self.expert_aux_weight = float(expert_aux_weight)
        self.record_step_interval = max(1, int(record_step_interval or 1))
        self._forward_count = 0
        self._record_this_step = False
        self._structure_ste = True
        self._detach_structure = False
        expert_dims = to_dims(expert_dims, (256, 128, 64))
        self.output_dim = int(expert_dims[-1])
        self.cross_rank = int(cross_rank)
        self.finalmlp_domain_input_mask_logits: Optional[nn.Parameter] = None
        self.finalmlp_domain_prefusion_mask_logits: Optional[nn.Parameter] = None
        if self.finalmlp_domain_mask_mode in {"input_feature", "input_feature_pre_fusion"}:
            self.finalmlp_domain_input_mask_logits = nn.Parameter(
                torch.zeros(int(num_domains), self.num_experts, self.input_dim)
            )
        if self.finalmlp_domain_mask_mode in {"pre_fusion", "input_feature_pre_fusion"}:
            self.finalmlp_domain_prefusion_mask_logits = nn.Parameter(
                torch.zeros(int(num_domains), self.num_experts, self.output_dim)
            )
        gate_dim = self.cross_rank if self.cross_basis == "lq_rank64" else self.output_dim
        self.cross_hard_gate = StreamReceptionGate(
            self.gate_reception,
            num_domains=num_domains,
            num_streams=self.num_experts,
            dim=gate_dim,
            hard_threshold=hard_gate_threshold,
            init_logit=hard_gate_init_logit,
            hard_topk=hard_gate_topk,
            init_noise_std=hard_gate_init_noise_std,
        )

        self.domain_mask = (
            DomainInputMask(
                input_dim=self.input_dim,
                num_domains=num_domains,
                hidden_dims=mask_hidden_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
                mode=mask_mode,
            )
            if self.use_domain_mask
            else None
        )
        self.experts = nn.ModuleList()
        self.shared_lower = nn.Identity()
        self.private_lowers = nn.ModuleList()
        self.shared_middle = nn.Identity()
        self.post_shared_interest = nn.Identity()
        self.post_shared_interest_heads = nn.ModuleList()

        def ensure_final_dim(dims: Iterable[int], final_dim: int) -> tuple[int, ...]:
            out = to_dims(dims, (final_dim,))
            if int(out[-1]) != int(final_dim):
                out = tuple(out) + (int(final_dim),)
            return tuple(out)

        def make_stream_basis() -> nn.Module:
            if self.cross_basis == "finalmlp_swiglu":
                return StreamFinalMLPBasis(
                    self.input_dim,
                    self.output_dim,
                    hidden_dims=expert_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
            if self.cross_basis == "lq_rank64":
                return StreamLowRankQuadraticBasis(
                    self.input_dim,
                    self.output_dim,
                    cross_rank=self.cross_rank,
                    hidden_dims=expert_dims[1:] if len(expert_dims) > 1 else (self.output_dim,),
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                    use_linear_lowrank=use_linear_lowrank,
                )
            return build_mlp(
                self.input_dim,
                ensure_final_dim(expert_dims, self.output_dim),
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )

        if self.architecture == "all_expert":
            self.experts = nn.ModuleList([
                make_stream_basis()
                for _ in range(self.num_experts)
            ])
        elif self.architecture == "mid_private":
            default_lower = tuple(expert_dims[:-1]) if len(expert_dims) > 1 else (max(1, self.input_dim // 2),)
            lower_dims = to_dims(shared_lower_dims, default_lower)
            self.shared_lower = build_mlp(
                self.input_dim,
                lower_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            private_dims = ensure_final_dim(private_lower_dims or (self.output_dim,), self.output_dim)
            self.experts = nn.ModuleList([
                build_mlp(
                    int(lower_dims[-1]),
                    private_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                for _ in range(self.num_experts)
            ])
        else:
            lower_dims = ensure_final_dim(private_lower_dims or expert_dims, self.output_dim)
            self.private_lowers = nn.ModuleList([
                make_stream_basis()
                if self.cross_basis != "mlp"
                else build_mlp(
                    self.input_dim,
                    lower_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                for _ in range(self.num_experts)
            ])
            shared_input_dim = int(lower_dims[-1])
            if shared_input_dim != self.output_dim and self.shared_middle_mode in {"identity", "residual_small"}:
                raise ValueError(
                    f"shared_middle_mode={self.shared_middle_mode!r} requires "
                    f"input_dim == output_dim, got {shared_input_dim} and {self.output_dim}."
                )
            if self.shared_middle_mode == "identity":
                self.shared_middle = nn.Identity()
            elif self.shared_middle_mode == "linear":
                self.shared_middle = nn.Linear(shared_input_dim, self.output_dim)
            else:
                middle_dims = ensure_final_dim(shared_middle_dims or (self.output_dim, self.output_dim), self.output_dim)
                middle = build_mlp(
                    shared_input_dim,
                    middle_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                if self.shared_middle_mode == "residual_small":
                    self.shared_middle = FixedScaleResidual(middle, self.shared_middle_residual_scale)
                else:
                    self.shared_middle = middle
            interest_dims = ensure_final_dim(post_shared_interest_dims or (self.output_dim, self.output_dim), self.output_dim)
            if self.post_shared_interest_mode == "shared_ffn":
                self.post_shared_interest = build_mlp(
                    self.output_dim,
                    interest_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
            elif self.post_shared_interest_mode in {"private_ffn", "private_residual_ffn"}:
                self.post_shared_interest_heads = nn.ModuleList([
                    build_mlp(
                        self.output_dim,
                        interest_dims,
                        dropout_rate=dropout_rate,
                        activation=activation,
                        batch_norm=batch_norm,
                    )
                    for _ in range(self.num_experts)
                ])
        self.mixer = (
            ExpertTokenMixer(
                num_tokens=self.num_experts,
                token_dim=self.output_dim,
                num_layers=mixer_layers,
                activation=activation,
                dropout_rate=mixer_dropout_rate,
            )
            if self.use_token_mixer
            else nn.Identity()
        )
        self.scene_global_tokens: Optional[nn.Parameter] = None
        self.scene_domain_tokens: Optional[nn.Parameter] = None
        self.scene_read_mixers = nn.ModuleDict()
        self.scene_read_sources = nn.ModuleDict()
        self.scene_read_blocks = nn.ModuleDict()
        self.scene_fusion_projection = None
        self.scene_block_selector = None
        self.scene_token_context_norm = None
        self.scene_domain_context = None
        if self.scene_token_mode == "attn_multitoken":
            if self.scene_num_global_tokens > 0:
                self.scene_global_tokens = nn.Parameter(torch.empty(self.scene_num_global_tokens, self.output_dim))
                nn.init.normal_(self.scene_global_tokens, mean=0.0, std=0.02)
            if self.scene_num_domain_tokens > 0:
                self.scene_domain_tokens = nn.Parameter(
                    torch.empty(int(num_domains), self.scene_num_domain_tokens, self.output_dim)
                )
                nn.init.normal_(self.scene_domain_tokens, mean=0.0, std=0.02)
            if self.scene_token_context_mode == "raw_input_add":
                self.scene_token_context_norm = nn.LayerNorm(self.input_dim)
                context_dims = tuple(to_dims(scene_token_context_hidden_dims, (128,))) + (
                    self.scene_num_domain_tokens * self.output_dim,
                )
                self.scene_domain_context = build_mlp(
                    self.input_dim,
                    context_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
            for stage in self.scene_token_stages:
                if self.scene_token_source_mode == "unimix_blocks":
                    self.scene_read_sources[stage] = SceneUniMixBlockSource(
                        num_streams=self.num_experts,
                        stream_dim=self.output_dim,
                        output_dim=self.output_dim,
                        block_size=self.scene_token_unimix_block_size,
                        num_layers=self.scene_token_mixer_layers,
                        temperature=self.scene_token_unimix_temperature,
                        sinkhorn_iters=self.scene_token_unimix_sinkhorn_iters,
                        activation=activation,
                        dropout_rate=mixer_dropout_rate,
                    )
                else:
                    self.scene_read_mixers[stage] = ExpertTokenMixer(
                        num_tokens=self.num_experts,
                        token_dim=self.output_dim,
                        num_layers=self.scene_token_mixer_layers,
                        activation=activation,
                        dropout_rate=mixer_dropout_rate,
                        zero_init_residual=self.scene_token_zero_init_residual,
                    )
                self.scene_read_blocks[stage] = SceneAttentionReadBlock(
                    token_dim=self.output_dim,
                    num_heads=self.scene_token_num_heads,
                    activation=activation,
                    dropout_rate=mixer_dropout_rate,
                    zero_init_residual=self.scene_token_zero_init_residual,
                )
        split_head_readout = self.readout_mode in {"split_head_mean", "split_head_domain_weight"}
        if split_head_readout and self.readout_num_tokens != self.num_experts:
            raise ValueError("pre_shared_merge_mode is not supported with split-head readout modes.")
        self.readout_base_dim = (
            self.output_dim
            if self.readout_mode == "mean"
            else self.readout_num_tokens * self.output_dim
        )
        self.readout_projection = None
        self.stream_heads = nn.ModuleList()
        self.domain_stream_logit_weight = None
        self.residual_side_projection = None
        self.residual_side_head = None
        self.residual_beta_logits = None
        self.residual_beta_head = None
        self.residual_beta_domain_bias = None
        self.scene_query = None
        if self.readout_mode == "flatten" and self.readout_projection_dim > 0:
            self.readout_projection = nn.Linear(self.readout_base_dim, self.readout_projection_dim, bias=False)
        self.dense_readout_dim = (
            self.readout_projection_dim if self.readout_projection is not None else self.readout_base_dim
        )
        if split_head_readout:
            if self.scene_query_mode != "none" or self.scene_token_mode != "none":
                raise ValueError("scene query/token modes are only supported for dense readout modes.")
            self.head = None
            self.stream_heads = nn.ModuleList([
                DomainTowerHead(
                    num_domains=num_domains,
                    input_dim=self.output_dim,
                    hidden_dims=tower_hidden_dims,
                    dropout_rate=tower_dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                for _ in range(self.num_experts)
            ])
            if self.readout_mode == "split_head_domain_weight":
                self.domain_stream_logit_weight = nn.Parameter(torch.zeros(num_domains, self.num_experts))
        else:
            readout_dim = self.dense_readout_dim
            if self.scene_query_mode == "sigmoid_query":
                self.scene_query = DomainSigmoidQuery(
                    input_dim=readout_dim,
                    output_dim=readout_dim,
                    num_domains=num_domains,
                    hidden_dims=scene_query_hidden_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                    sharpness=self.scene_query_sharpness,
                )
            if self.scene_token_mode != "none" and self.scene_token_fusion_mode == "block_select_readout":
                self.scene_block_selector = SceneUniMixBlockSelector(
                    num_streams=self.readout_num_tokens,
                    stream_dim=self.output_dim,
                    scene_dim=self.output_dim,
                    readout_dim=readout_dim,
                    block_size=self.scene_token_unimix_block_size,
                    num_layers=self.scene_token_mixer_layers,
                    topk=self.scene_token_block_select_topk,
                    temperature=self.scene_token_block_select_temperature,
                    sinkhorn_iters=self.scene_token_unimix_sinkhorn_iters,
                    hard_select=self.scene_token_block_select_hard,
                    context_readout=self.scene_token_block_select_context,
                    dropout_rate=mixer_dropout_rate,
                )
            if self.scene_token_mode != "none" and self.scene_token_fusion_mode == "add_readout":
                self.scene_fusion_projection = nn.Linear(self.output_dim, readout_dim)
                if self.scene_token_zero_init_fusion:
                    nn.init.zeros_(self.scene_fusion_projection.weight)
                    if self.scene_fusion_projection.bias is not None:
                        nn.init.zeros_(self.scene_fusion_projection.bias)
                elif self.scene_token_fusion_init_std > 0.0:
                    nn.init.normal_(self.scene_fusion_projection.weight, mean=0.0, std=self.scene_token_fusion_init_std)
                    if self.scene_fusion_projection.bias is not None:
                        nn.init.zeros_(self.scene_fusion_projection.bias)
            head_input_dim = (
                readout_dim * 2
                if self.scene_query_mode != "none" and self.scene_query_fusion_mode == "concat_delta"
                else readout_dim
            )
            if self.scene_token_mode != "none" and self.scene_token_fusion_mode == "concat_readout":
                head_input_dim += self.output_dim
            if self.prediction_head_mode == "shared_tower":
                self.head = SharedTowerHead(
                    input_dim=head_input_dim,
                    hidden_dims=tower_hidden_dims,
                    dropout_rate=tower_dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
            else:
                self.head = DomainTowerHead(
                    num_domains=num_domains,
                    input_dim=head_input_dim,
                    hidden_dims=tower_hidden_dims,
                    dropout_rate=tower_dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
        if (self.scene_query_mode != "none" or self.scene_token_mode != "none") and self.residual_side_mode != "none":
            raise ValueError("scene query/token modes are not combined with residual_side_mode in this implementation.")
        if self.residual_side_mode == "readout_projection":
            if split_head_readout:
                raise ValueError("residual_side_mode=readout_projection is only supported for dense readout modes.")
            side_dim = int(self.residual_readout_projection_dim or 64)
            self.residual_readout_projection_dim = side_dim
            self.residual_side_projection = nn.Linear(self.readout_base_dim, side_dim, bias=False)
            self.residual_side_head = DomainTowerHead(
                num_domains=num_domains,
                input_dim=side_dim,
                hidden_dims=tower_hidden_dims,
                dropout_rate=tower_dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            beta_max = max(float(self.residual_beta_max), 1e-6)
            beta_init = min(max(float(self.residual_beta_init), 1e-6), beta_max - 1e-6)
            beta_prob = beta_init / beta_max
            beta_logit = math.log(beta_prob / max(1e-12, 1.0 - beta_prob))
            if self.residual_beta_mode == "sample_dense":
                self.residual_beta_head = nn.Linear(readout_dim, 1)
                nn.init.zeros_(self.residual_beta_head.weight)
                nn.init.constant_(self.residual_beta_head.bias, float(beta_logit))
                self.residual_beta_domain_bias = nn.Parameter(torch.zeros(num_domains))
            else:
                self.residual_beta_logits = nn.Parameter(torch.full((num_domains,), float(beta_logit)))
        self.aux_heads = nn.ModuleList([nn.Linear(self.output_dim, 1) for _ in range(self.num_experts)])

        self._latest_aux_logits: Optional[torch.Tensor] = None
        self._latest_logit_components: dict[str, torch.Tensor] = {}
        self._latest_mask: Optional[torch.Tensor] = None
        self._mask_stats: Dict[int, Dict[str, float]] = defaultdict(_new_value_stats)
        self._token_norm_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._token_grad_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._mixer_delta_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._mixer_ratio_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._stream_norm_stats: Dict[tuple[str, int, int], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._stream_grad_stats: Dict[tuple[str, int, int], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._stream_pair_cos_stats: Dict[tuple[str, int, int, int], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._shared_middle_stats: Dict[tuple[int, int, str], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._post_shared_interest_stats: Dict[tuple[int, int, str], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._aux_train_logloss_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._cross_basis_signal_stats: Dict[tuple[int, int, str], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._finalmlp_gate_stats: Dict[tuple[int, int, str, str], Dict[str, float]] = defaultdict(_new_value_stats)
        self._finalmlp_domain_mask_stats: Dict[tuple[int, int, str, str], Dict[str, float]] = defaultdict(_new_value_stats)
        self._reception_gate_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(_new_value_stats)
        self._scene_query_gate_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(_new_value_stats)
        self._scene_query_fusion_stats: Dict[tuple[int, str], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._scene_token_attention_stats: Dict[tuple[str, int, str, int, int], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._scene_token_delta_stats: Dict[tuple[str, int, str], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._scene_token_fusion_stats: Dict[tuple[int, str], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._scene_token_block_select_gate_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(_new_value_stats)
        self._scene_token_block_select_summary_stats: Dict[tuple[int, str], Dict[str, float]] = defaultdict(_new_norm_stats)
        self._latest_scene_tokens: Optional[torch.Tensor] = None
        self._latest_scene_summary: Optional[torch.Tensor] = None
        self._latest_scene_block_tokens: Optional[torch.Tensor] = None

    def reset_train_observations(self) -> None:
        self._forward_count = 0
        self._record_this_step = False
        self._mask_stats.clear()
        self._token_norm_stats.clear()
        self._token_grad_stats.clear()
        self._mixer_delta_stats.clear()
        self._mixer_ratio_stats.clear()
        self._stream_norm_stats.clear()
        self._stream_grad_stats.clear()
        self._stream_pair_cos_stats.clear()
        self._shared_middle_stats.clear()
        self._post_shared_interest_stats.clear()
        self._aux_train_logloss_stats.clear()
        self._cross_basis_signal_stats.clear()
        self._finalmlp_gate_stats.clear()
        self._finalmlp_domain_mask_stats.clear()
        self._reception_gate_stats.clear()
        self._scene_query_gate_stats.clear()
        self._scene_query_fusion_stats.clear()
        self._scene_token_attention_stats.clear()
        self._scene_token_delta_stats.clear()
        self._scene_token_fusion_stats.clear()
        self._scene_token_block_select_gate_stats.clear()
        self._scene_token_block_select_summary_stats.clear()
        self._latest_scene_tokens = None
        self._latest_scene_summary = None
        self._latest_scene_block_tokens = None

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def _record_mask(self, mask: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or mask.numel() == 0:
            return
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(mask.device)
            values = mask.detach().float()
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                _accumulate_values(self._mask_stats[domain_idx], values[selected], sample_count=count)

    def _record_token_norms(self, tokens: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or tokens.numel() == 0:
            return
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(tokens.device)
            norms = torch.linalg.vector_norm(tokens.detach().float(), ord=2, dim=-1)
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for expert_idx in range(int(tokens.shape[1])):
                    _accumulate_norms(
                        self._token_norm_stats[(domain_idx, expert_idx)],
                        norms[selected, expert_idx],
                        sample_count=count,
                    )

    def _record_stream_norms(self, stage: str, tokens: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or tokens.numel() == 0:
            return
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(tokens.device)
            norms = torch.linalg.vector_norm(tokens.detach().float(), ord=2, dim=-1)
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for stream_idx in range(int(tokens.shape[1])):
                    _accumulate_norms(
                        self._stream_norm_stats[(stage, domain_idx, stream_idx)],
                        norms[selected, stream_idx],
                        sample_count=count,
                    )

    def _record_stream_pair_cosine(self, stage: str, tokens: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or tokens.numel() == 0 or self.num_experts < 2:
            return
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(tokens.device)
            values = tokens.detach().float()
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                selected_tokens = values[selected]
                num_streams = int(selected_tokens.shape[1])
                for left in range(num_streams):
                    for right in range(left + 1, num_streams):
                        cosine = F.cosine_similarity(
                            selected_tokens[:, left],
                            selected_tokens[:, right],
                            dim=-1,
                            eps=1e-12,
                        )
                        _accumulate_norms(
                            self._stream_pair_cos_stats[(stage, domain_idx, left, right)],
                            cosine,
                            sample_count=count,
                        )

    def _record_mixer_delta(self, before: torch.Tensor, after: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or before.numel() == 0:
            return
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(before.device)
            before_norm = torch.linalg.vector_norm(before.detach().float(), ord=2, dim=-1)
            delta_norm = torch.linalg.vector_norm((after - before).detach().float(), ord=2, dim=-1)
            ratio = delta_norm / before_norm.clamp_min(1e-12)
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for expert_idx in range(int(before.shape[1])):
                    _accumulate_norms(
                        self._mixer_delta_stats[(domain_idx, expert_idx)],
                        delta_norm[selected, expert_idx],
                        sample_count=count,
                    )
                    _accumulate_norms(
                        self._mixer_ratio_stats[(domain_idx, expert_idx)],
                        ratio[selected, expert_idx],
                        sample_count=count,
                    )

    def _record_shared_middle_delta(self, before: torch.Tensor, after: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or before.numel() == 0 or before.shape != after.shape:
            return
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(before.device)
            before_float = before.detach().float()
            after_float = after.detach().float()
            input_norm = torch.linalg.vector_norm(before_float, ord=2, dim=-1)
            output_norm = torch.linalg.vector_norm(after_float, ord=2, dim=-1)
            delta_norm = torch.linalg.vector_norm(after_float - before_float, ord=2, dim=-1)
            delta_ratio = delta_norm / input_norm.clamp_min(1e-12)
            cosine = F.cosine_similarity(before_float, after_float, dim=-1, eps=1e-12)
            values_by_name = {
                "input_norm": input_norm,
                "output_norm": output_norm,
                "delta_norm": delta_norm,
                "delta_ratio": delta_ratio,
                "before_after_cosine": cosine,
            }
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for stream_idx in range(int(before.shape[1])):
                    for name, values in values_by_name.items():
                        _accumulate_norms(
                            self._shared_middle_stats[(domain_idx, stream_idx, name)],
                            values[selected, stream_idx],
                            sample_count=count,
                        )

    def _record_post_shared_interest_delta(
            self,
            before: torch.Tensor,
            after: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step or before.numel() == 0 or before.shape != after.shape:
            return
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(before.device)
            before_float = before.detach().float()
            after_float = after.detach().float()
            input_norm = torch.linalg.vector_norm(before_float, ord=2, dim=-1)
            output_norm = torch.linalg.vector_norm(after_float, ord=2, dim=-1)
            delta_norm = torch.linalg.vector_norm(after_float - before_float, ord=2, dim=-1)
            delta_ratio = delta_norm / input_norm.clamp_min(1e-12)
            cosine = F.cosine_similarity(before_float, after_float, dim=-1, eps=1e-12)
            values_by_name = {
                "input_norm": input_norm,
                "output_norm": output_norm,
                "delta_norm": delta_norm,
                "delta_ratio": delta_ratio,
                "before_after_cosine": cosine,
            }
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for stream_idx in range(int(before.shape[1])):
                    for name, values in values_by_name.items():
                        _accumulate_norms(
                            self._post_shared_interest_stats[(domain_idx, stream_idx, name)],
                            values[selected, stream_idx],
                            sample_count=count,
                        )

    def _apply_post_shared_interest(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.post_shared_interest_mode == "identity":
            return tokens
        if self.post_shared_interest_mode == "shared_ffn":
            return torch.stack(
                [self.post_shared_interest(tokens[:, stream_idx]) for stream_idx in range(int(tokens.shape[1]))],
                dim=1,
            )
        if self.post_shared_interest_mode == "private_ffn":
            return torch.stack(
                [head(tokens[:, stream_idx]) for stream_idx, head in enumerate(self.post_shared_interest_heads)],
                dim=1,
            )
        if self.post_shared_interest_mode == "private_residual_ffn":
            delta = torch.stack(
                [head(tokens[:, stream_idx]) for stream_idx, head in enumerate(self.post_shared_interest_heads)],
                dim=1,
            )
            return tokens + float(self.post_shared_interest_residual_scale) * delta
        raise ValueError(f"Unknown post_shared_interest_mode={self.post_shared_interest_mode!r}.")

    def _tap_token_gradients(self, tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._record_this_step or not tokens.requires_grad:
            return tokens
        detached_domains = domain_ids.detach()

        def hook(grad: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                domain_values = detached_domains.long().view(-1).to(grad.device)
                for domain_idx in range(int(self.num_domains)):
                    selected = domain_values == domain_idx
                    count = int(selected.sum().item())
                    if count <= 0:
                        continue
                    for expert_idx in range(int(grad.shape[1])):
                        value = torch.linalg.vector_norm(
                            grad[selected, expert_idx].detach().float(),
                            ord=2,
                            dim=-1,
                        )
                        _accumulate_norms(
                            self._token_grad_stats[(domain_idx, expert_idx)],
                            value,
                            sample_count=count,
                        )
            return grad

        tokens.register_hook(hook)
        return tokens

    def _tap_stream_gradients(self, stage: str, tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._record_this_step or not tokens.requires_grad:
            return tokens
        detached_domains = domain_ids.detach()

        def hook(grad: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                domain_values = detached_domains.long().view(-1).to(grad.device)
                for domain_idx in range(int(self.num_domains)):
                    selected = domain_values == domain_idx
                    count = int(selected.sum().item())
                    if count <= 0:
                        continue
                    for stream_idx in range(int(grad.shape[1])):
                        values = torch.linalg.vector_norm(
                            grad[selected, stream_idx].detach().float(),
                            ord=2,
                            dim=-1,
                        )
                        _accumulate_norms(
                            self._stream_grad_stats[(stage, domain_idx, stream_idx)],
                            values,
                            sample_count=count,
                        )
            return grad

        tokens.register_hook(hook)
        return tokens

    def _record_stage(self, stage: str, tokens: torch.Tensor, domain_ids: torch.Tensor) -> None:
        self._record_stream_norms(stage, tokens, domain_ids)
        self._record_stream_pair_cosine(stage, tokens, domain_ids)

    def _compute_aux_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [head(tokens[:, stream_idx]).view(-1, 1) for stream_idx, head in enumerate(self.aux_heads)],
            dim=1,
        )

    def _merge_pre_shared_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pre_shared_merge_mode == "mean":
            return tokens.mean(dim=1)
        if self.pre_shared_merge_mode == "sum_sqrt":
            return tokens.sum(dim=1) / math.sqrt(max(1, int(tokens.shape[1])))
        raise ValueError(f"Unknown pre_shared_merge_mode={self.pre_shared_merge_mode!r}.")

    def _record_aux_train_logloss(self, logits: torch.Tensor, labels: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or logits.numel() == 0:
            return
        with torch.no_grad():
            label_matrix = labels.float().view(-1, 1).expand_as(logits)
            losses = F.binary_cross_entropy_with_logits(logits.detach().float(), label_matrix, reduction="none")
            domain_values = domain_ids.detach().long().view(-1).to(logits.device)
            for domain_idx in range(int(self.num_domains)):
                selected = domain_values == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for stream_idx in range(self.num_experts):
                    _accumulate_norms(
                        self._aux_train_logloss_stats[(domain_idx, stream_idx)],
                        losses[selected, stream_idx],
                        sample_count=count,
                    )

    def _stream_reception_gate(
            self,
            domain_ids: torch.Tensor,
            stream_idx: int,
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor | None:
        gate, hard, soft = self.cross_hard_gate(
            domain_ids,
            stream_idx=stream_idx,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            structure_ste=self._structure_ste,
            detach_structure=self._detach_structure,
        )
        observed = gate if gate is not None else None
        if observed is not None:
            self._record_reception_gate(stream_idx, observed, domain_ids)
        return gate

    def _record_reception_gate(self, stream_idx: int, gate: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or gate.numel() == 0:
            return
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(gate.device)
            values = gate.detach().float()
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                _accumulate_values(
                    self._reception_gate_stats[(domain_idx, int(stream_idx))],
                    values[selected],
                    sample_count=count,
                )

    def _record_cross_basis_signal(
            self,
            stream_idx: int,
            trace: dict[str, torch.Tensor],
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step or not trace:
            return
        stream1 = trace.get("stream1")
        stream2 = trace.get("stream2")
        fused = trace.get("fused")
        if stream1 is None or stream2 is None or fused is None or fused.numel() == 0:
            return
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(fused.device)
            stream1 = stream1.detach().float()
            stream2 = stream2.detach().float()
            fused = fused.detach().float()
            stream1_norm = torch.linalg.vector_norm(stream1, ord=2, dim=-1)
            stream2_norm = torch.linalg.vector_norm(stream2, ord=2, dim=-1)
            fused_norm = torch.linalg.vector_norm(fused, ord=2, dim=-1)
            base_norm = 0.5 * (stream1_norm + stream2_norm)
            ratio = fused_norm / base_norm.clamp_min(1e-12)
            cosine = F.cosine_similarity(stream1, stream2, dim=-1, eps=1e-12)
            fusion_abs = fused.abs().mean(dim=-1)
            fusion_sparse = (fused.abs() < 1e-6).float().mean(dim=-1)
            values_by_name = {
                "stream1_norm": stream1_norm,
                "stream2_norm": stream2_norm,
                "fused_norm": fused_norm,
                "fused_to_base_norm_ratio": ratio,
                "stream1_stream2_cosine": cosine,
                "fusion_abs_mean": fusion_abs,
                "fusion_sparsity_frac": fusion_sparse,
            }
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for name, values in values_by_name.items():
                    _accumulate_norms(
                        self._cross_basis_signal_stats[(domain_idx, int(stream_idx), name)],
                        values[selected],
                        sample_count=count,
                    )

    def _record_finalmlp_gates(
            self,
            stream_idx: int,
            trace: dict[str, torch.Tensor],
            domain_ids: torch.Tensor,
            labels: torch.Tensor | None = None,
    ) -> None:
        del labels
        if not self.training or not self._record_this_step or self.cross_basis != "finalmlp_swiglu":
            return
        gate_by_name = {
            "gate1": trace.get("selection_gate_1"),
            "gate2": trace.get("selection_gate_2"),
        }
        field_segments = tuple(getattr(self.input_view, "field_segments", ()))
        if not field_segments:
            field_segments = (("all", 0, self.input_dim),)
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1)
            for gate_name, gate in gate_by_name.items():
                if gate is None or gate.numel() == 0:
                    continue
                gate = gate.detach().float()
                domain_values = domain_ids.to(gate.device)
                for domain_idx in range(int(self.num_domains)):
                    selected = domain_values == domain_idx
                    count = int(selected.sum().item())
                    if count <= 0:
                        continue
                    for field_name, start, stop in field_segments:
                        values = gate[selected, int(start):int(stop)]
                        if values.numel() == 0:
                            continue
                        _accumulate_values(
                            self._finalmlp_gate_stats[(domain_idx, int(stream_idx), gate_name, str(field_name))],
                            values,
                            sample_count=count,
                        )

    def _domain_stream_mask_from_logits(
            self,
            logits: nn.Parameter | None,
            domain_ids: torch.Tensor,
            stream_idx: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor | None:
        if logits is None:
            return None
        domain_values = domain_ids.long().view(-1).to(device=logits.device)
        stream_logits = logits[domain_values, int(stream_idx)]
        mask = 2.0 * torch.sigmoid(float(self.finalmlp_domain_mask_sharpness) * stream_logits)
        return mask.to(dtype=dtype, device=device)

    def _record_finalmlp_domain_mask(
            self,
            stream_idx: int,
            stage: str,
            mask: torch.Tensor | None,
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step or mask is None or mask.numel() == 0:
            return
        if stage == "input_feature":
            field_segments = tuple(getattr(self.input_view, "field_segments", ()))
            if not field_segments:
                field_segments = (("all", 0, int(mask.shape[-1])),)
        else:
            field_segments = (("cross_hidden", 0, int(mask.shape[-1])),)
        with torch.no_grad():
            domain_ids = domain_ids.detach().long().view(-1).to(mask.device)
            mask = mask.detach().float()
            for domain_idx in range(int(self.num_domains)):
                selected = domain_ids == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for field_name, start, stop in field_segments:
                    values = mask[selected, int(start):int(stop)]
                    if values.numel() == 0:
                        continue
                    _accumulate_values(
                        self._finalmlp_domain_mask_stats[
                            (domain_idx, int(stream_idx), str(stage), str(field_name))
                        ],
                        values,
                        sample_count=count,
                    )

    def _run_stream_module(
            self,
            module: nn.Module,
            x: torch.Tensor,
            domain_ids: torch.Tensor,
            stream_idx: int,
    ) -> torch.Tensor:
        gate = self._stream_reception_gate(
            domain_ids,
            stream_idx,
            int(x.size(0)),
            x.dtype,
            x.device,
        )
        if isinstance(module, StreamFinalMLPBasis):
            input_mask = None
            pre_fusion_mask = None
            input_mask = self._domain_stream_mask_from_logits(
                self.finalmlp_domain_input_mask_logits,
                domain_ids,
                stream_idx,
                x.dtype,
                x.device,
            )
            pre_fusion_mask = self._domain_stream_mask_from_logits(
                self.finalmlp_domain_prefusion_mask_logits,
                domain_ids,
                stream_idx,
                x.dtype,
                x.device,
            )
            self._record_finalmlp_domain_mask(stream_idx, "input_feature", input_mask, domain_ids)
            self._record_finalmlp_domain_mask(stream_idx, "pre_fusion", pre_fusion_mask, domain_ids)
            out, trace = module(
                x,
                reception_gate=gate,
                input_mask=input_mask,
                pre_fusion_mask=pre_fusion_mask,
            )
            self._record_cross_basis_signal(stream_idx, trace, domain_ids)
            self._record_finalmlp_gates(stream_idx, trace, domain_ids)
            return out
        if isinstance(module, StreamLowRankQuadraticBasis):
            out, trace = module(x, reception_gate=gate)
            self._record_cross_basis_signal(stream_idx, trace, domain_ids)
            return out
        out = module(x)
        if gate is not None:
            out = out * gate
        return out

    def _init_scene_tokens(
            self,
            domain_ids: torch.Tensor,
            raw_input: torch.Tensor | None = None,
            *,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor | None:
        if self.scene_token_mode == "none":
            return None
        batch_size = int(domain_ids.numel())
        pieces: list[torch.Tensor] = []
        if self.scene_global_tokens is not None:
            global_tokens = self.scene_global_tokens.to(dtype=dtype, device=device)
            pieces.append(global_tokens.unsqueeze(0).expand(batch_size, -1, -1))
        if self.scene_domain_tokens is not None:
            domain_values = domain_ids.long().view(-1).to(device=self.scene_domain_tokens.device)
            domain_tokens = self.scene_domain_tokens[domain_values].to(dtype=dtype, device=device)
            if (
                    self.scene_token_context_mode == "raw_input_add"
                    and self.scene_domain_context is not None
                    and self.scene_token_context_norm is not None
                    and raw_input is not None
            ):
                context_source = raw_input.detach() if self.scene_token_context_detach_input else raw_input
                context_source = context_source.to(device=device, dtype=dtype)
                context_delta = self.scene_domain_context(self.scene_token_context_norm(context_source))
                context_delta = context_delta.view(batch_size, self.scene_num_domain_tokens, self.output_dim)
                domain_tokens = domain_tokens + float(self.scene_token_context_scale) * context_delta
            pieces.append(domain_tokens)
        if not pieces:
            return None
        return torch.cat(pieces, dim=1)

    def _scene_summary(self, scene_tokens: torch.Tensor | None) -> torch.Tensor | None:
        if scene_tokens is None:
            return None
        return scene_tokens.mean(dim=1)

    def _record_scene_token_delta(
            self,
            stage: str,
            before: torch.Tensor,
            after: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step or before.numel() == 0:
            return
        with torch.no_grad():
            domain_values = domain_ids.detach().long().view(-1).to(before.device)
            before_float = before.detach().float()
            after_float = after.detach().float()
            before_norm = torch.linalg.vector_norm(before_float, ord=2, dim=-1)
            after_norm = torch.linalg.vector_norm(after_float, ord=2, dim=-1)
            delta_norm = torch.linalg.vector_norm(after_float - before_float, ord=2, dim=-1)
            delta_ratio = delta_norm / before_norm.clamp_min(1e-12)
            cosine = F.cosine_similarity(before_float, after_float, dim=-1, eps=1e-12)
            values_by_name = {
                "before_norm": before_norm,
                "after_norm": after_norm,
                "delta_norm": delta_norm,
                "delta_ratio": delta_ratio,
                "before_after_cosine": cosine,
            }
            for domain_idx in range(int(self.num_domains)):
                selected = domain_values == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for name, values in values_by_name.items():
                    per_sample_values = values[selected]
                    if per_sample_values.dim() > 1:
                        per_sample_values = per_sample_values.mean(dim=1)
                    _accumulate_norms(
                        self._scene_token_delta_stats[(str(stage), domain_idx, str(name))],
                        per_sample_values,
                        sample_count=count,
                    )

    def _record_scene_token_attention(
            self,
            stage: str,
            weights: torch.Tensor | None,
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step or weights is None or weights.numel() == 0:
            return
        with torch.no_grad():
            domain_values = domain_ids.detach().long().view(-1).to(weights.device)
            attn = weights.detach().float().mean(dim=1)
            global_count = int(self.scene_num_global_tokens)
            for domain_idx in range(int(self.num_domains)):
                selected = domain_values == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for query_idx in range(int(attn.shape[1])):
                    query_scope = "global" if query_idx < global_count else "domain"
                    local_query_idx = query_idx if query_scope == "global" else query_idx - global_count
                    for stream_idx in range(int(attn.shape[2])):
                        _accumulate_norms(
                            self._scene_token_attention_stats[
                                (str(stage), domain_idx, query_scope, int(local_query_idx), int(stream_idx))
                            ],
                            attn[selected, query_idx, stream_idx],
                            sample_count=count,
                        )

    def _record_scene_token_block_select(
            self,
            selector_stats: dict[str, torch.Tensor],
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step or not selector_stats:
            return
        gate = selector_stats.get("gate")
        soft_prob = selector_stats.get("soft_prob")
        if gate is None or soft_prob is None or gate.numel() == 0:
            return
        with torch.no_grad():
            domain_values = domain_ids.detach().long().view(-1).to(gate.device)
            gate_values = gate.detach().float()
            prob_values = soft_prob.detach().float()
            selected_count = gate_values.sum(dim=-1)
            prob_hhi = prob_values.pow(2).sum(dim=-1).clamp_min(1e-12)
            effective_blocks = 1.0 / prob_hhi
            max_prob = prob_values.max(dim=-1).values
            for domain_idx in range(int(self.num_domains)):
                selected = domain_values == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for block_idx in range(int(gate_values.shape[1])):
                    _accumulate_values(
                        self._scene_token_block_select_gate_stats[(domain_idx, block_idx)],
                        gate_values[selected, block_idx],
                        sample_count=count,
                    )
                summary_values = {
                    "selected_count": selected_count,
                    "effective_blocks": effective_blocks,
                    "max_prob": max_prob,
                }
                for name, values in summary_values.items():
                    _accumulate_norms(
                        self._scene_token_block_select_summary_stats[(domain_idx, str(name))],
                        values[selected],
                        sample_count=count,
                    )

    def _update_scene_tokens(
            self,
            stage: str,
            scene_tokens: torch.Tensor | None,
            source_tokens: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
                self.scene_token_mode == "none"
                or scene_tokens is None
                or str(stage) not in self.scene_read_blocks
                or source_tokens.numel() == 0
        ):
            return scene_tokens
        source = source_tokens.detach() if self.scene_token_detach_backbone else source_tokens
        if str(stage) in self.scene_read_sources:
            mixed_source = self.scene_read_sources[str(stage)](source)
        else:
            mixed_source = self.scene_read_mixers[str(stage)](source)
        before = scene_tokens
        need_weights = bool(self.training and self._record_this_step)
        after, weights = self.scene_read_blocks[str(stage)](
            scene_tokens,
            mixed_source,
            need_weights=need_weights,
        )
        self._record_scene_token_delta(stage, before, after, domain_ids)
        self._record_scene_token_attention(stage, weights, domain_ids)
        return after

    def _build_stream_tokens(self, x: torch.Tensor, domain_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            self._forward_count += 1
            self._record_this_step = (
                self._forward_count == 1
                or (self._forward_count % self.record_step_interval) == 0
            )
        else:
            self._record_this_step = False
        self._latest_mask = None
        self._latest_scene_tokens = None
        self._latest_scene_summary = None
        self._latest_scene_block_tokens = None
        scene_tokens = self._init_scene_tokens(domain_ids, raw_input=x, dtype=x.dtype, device=x.device)
        if self.domain_mask is not None:
            mask = self.domain_mask(x, domain_ids)
            self._latest_mask = mask
            self._record_mask(mask, domain_ids)
            x = x * mask

        if self.architecture == "all_expert":
            tokens = torch.stack(
                [
                    self._run_stream_module(expert, x, domain_ids, stream_idx)
                    for stream_idx, expert in enumerate(self.experts)
                ],
                dim=1,
            )
            self._record_stage("after_expert", tokens, domain_ids)
        elif self.architecture == "mid_private":
            shared = self.shared_lower(x)
            tokens = torch.stack(
                [
                    self._run_stream_module(expert, shared, domain_ids, stream_idx)
                    for stream_idx, expert in enumerate(self.experts)
                ],
                dim=1,
            )
            self._record_stage("after_private_mid", tokens, domain_ids)
        else:
            before_shared = torch.stack(
                [
                    self._run_stream_module(expert, x, domain_ids, stream_idx)
                    for stream_idx, expert in enumerate(self.private_lowers)
                ],
                dim=1,
            )
            before_shared = self._tap_stream_gradients("before_shared", before_shared, domain_ids)
            self._record_stage("before_shared", before_shared, domain_ids)
            scene_tokens = self._update_scene_tokens("before_shared", scene_tokens, before_shared, domain_ids)
            if self.pre_shared_merge_mode == "none":
                tokens = torch.stack(
                    [self.shared_middle(before_shared[:, stream_idx]) for stream_idx in range(self.num_experts)],
                    dim=1,
                )
                aux_tokens = tokens
                self._record_shared_middle_delta(before_shared, tokens, domain_ids)
            else:
                merged = self._merge_pre_shared_tokens(before_shared)
                merged_tokens = merged.unsqueeze(1)
                self._record_stage("merged_before_shared", merged_tokens, domain_ids)
                tokens = self.shared_middle(merged).unsqueeze(1)
                aux_tokens = before_shared
                self._record_shared_middle_delta(merged_tokens, tokens, domain_ids)
            self._record_stage("after_shared", tokens, domain_ids)
            scene_tokens = self._update_scene_tokens("after_shared", scene_tokens, tokens, domain_ids)
            if self.post_shared_interest_mode != "identity":
                before_interest = tokens
                tokens = self._apply_post_shared_interest(before_interest)
                aux_tokens = tokens
                self._record_post_shared_interest_delta(before_interest, tokens, domain_ids)
                self._record_stage("after_interest", tokens, domain_ids)
                scene_tokens = self._update_scene_tokens("after_interest", scene_tokens, tokens, domain_ids)
        if self.architecture != "xshape":
            aux_tokens = tokens
        tokens = self._tap_token_gradients(tokens, domain_ids)
        if self.use_token_mixer:
            tokens = self._tap_stream_gradients("before_mixer", tokens, domain_ids)
            self._record_stage("before_mixer", tokens, domain_ids)
            mixed_tokens = self.mixer(tokens)
            self._record_mixer_delta(tokens, mixed_tokens, domain_ids)
            tokens = mixed_tokens

        tokens = self._tap_stream_gradients("before_readout", tokens, domain_ids)
        self._record_stage("before_readout", tokens, domain_ids)
        scene_tokens = self._update_scene_tokens("before_readout", scene_tokens, tokens, domain_ids)
        self._latest_scene_tokens = scene_tokens
        self._latest_scene_summary = self._scene_summary(scene_tokens)
        self._latest_scene_block_tokens = tokens
        self._record_token_norms(tokens, domain_ids)
        return tokens, aux_tokens

    def _tokens_to_readout(self, tokens: torch.Tensor) -> torch.Tensor:
        readout = self._tokens_to_base_readout(tokens)
        if self.readout_projection is not None:
            readout = self.readout_projection(readout)
        return readout

    def _tokens_to_base_readout(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.readout_mode == "mean":
            return tokens.mean(dim=1)
        return tokens.reshape(tokens.size(0), -1)

    def _scene_query_blocks(self, values: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        if (
                self.readout_mode == "flatten"
                and self.readout_projection is None
                and int(values.shape[-1]) == int(self.readout_num_tokens) * int(self.output_dim)
        ):
            blocks = values.view(values.size(0), int(self.readout_num_tokens), int(self.output_dim))
            return [(stream_idx, blocks[:, stream_idx]) for stream_idx in range(int(blocks.shape[1]))]
        return [(-1, values)]

    def _record_scene_query_gate(self, gate: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or gate.numel() == 0:
            return
        with torch.no_grad():
            domain_values = domain_ids.detach().long().view(-1).to(gate.device)
            gate_values = gate.detach().float()
            blocks = [(-1, gate_values)]
            per_stream = self._scene_query_blocks(gate_values)
            if per_stream and per_stream[0][0] != -1:
                blocks.extend(per_stream)
            for domain_idx in range(int(self.num_domains)):
                selected = domain_values == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for stream_idx, block in blocks:
                    _accumulate_values(
                        self._scene_query_gate_stats[(domain_idx, int(stream_idx))],
                        block[selected],
                        sample_count=count,
                    )

    def _record_scene_query_fusion(
            self,
            task_readout: torch.Tensor,
            scene_delta: torch.Tensor,
            fused_readout: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step or task_readout.numel() == 0:
            return
        if self.scene_query_fusion_mode == "concat_delta":
            comparable_fused = task_readout + scene_delta
        else:
            comparable_fused = fused_readout
        with torch.no_grad():
            domain_values = domain_ids.detach().long().view(-1).to(task_readout.device)
            task = task_readout.detach().float()
            delta = scene_delta.detach().float()
            fused = comparable_fused.detach().float()
            task_norm = torch.linalg.vector_norm(task, ord=2, dim=-1)
            delta_norm = torch.linalg.vector_norm(delta, ord=2, dim=-1)
            fused_norm = torch.linalg.vector_norm(fused, ord=2, dim=-1)
            delta_ratio = delta_norm / task_norm.clamp_min(1e-12)
            task_delta_cosine = F.cosine_similarity(task, delta, dim=-1, eps=1e-12)
            task_fused_cosine = F.cosine_similarity(task, fused, dim=-1, eps=1e-12)
            values_by_name = {
                "task_norm": task_norm,
                "scene_delta_norm": delta_norm,
                "fused_norm": fused_norm,
                "scene_delta_ratio": delta_ratio,
                "task_scene_delta_cosine": task_delta_cosine,
                "task_fused_cosine": task_fused_cosine,
            }
            for domain_idx in range(int(self.num_domains)):
                selected = domain_values == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for name, values in values_by_name.items():
                    _accumulate_norms(
                        self._scene_query_fusion_stats[(domain_idx, str(name))],
                        values[selected],
                        sample_count=count,
                    )

    def _apply_scene_query_readout(
            self,
            dense_readout: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.scene_query_mode == "none":
            return dense_readout
        if self.scene_query is None:
            raise RuntimeError("scene_query_mode is enabled, but scene_query is not initialized.")
        query_source = dense_readout.detach() if self.scene_query_detach_backbone else dense_readout
        gate, _ = self.scene_query(query_source, domain_ids)
        self._record_scene_query_gate(gate, domain_ids)
        value_source = dense_readout.detach() if self.scene_query_detach_backbone else dense_readout
        scene_delta = (gate - 1.0) * value_source
        if self.scene_query_fusion_mode == "add_delta":
            fused = dense_readout + scene_delta
        elif self.scene_query_fusion_mode == "concat_delta":
            fused = torch.cat([dense_readout, scene_delta], dim=-1)
        else:
            raise ValueError(f"Unknown scene_query_fusion_mode={self.scene_query_fusion_mode!r}.")
        self._record_scene_query_fusion(dense_readout, scene_delta, fused, domain_ids)
        return fused

    def _record_scene_token_fusion(
            self,
            task_readout: torch.Tensor,
            scene_delta: torch.Tensor,
            fused_readout: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step or task_readout.numel() == 0:
            return
        comparable_fused = fused_readout
        with torch.no_grad():
            domain_values = domain_ids.detach().long().view(-1).to(task_readout.device)
            task = task_readout.detach().float()
            delta = scene_delta.detach().float()
            fused = comparable_fused.detach().float()
            task_norm = torch.linalg.vector_norm(task, ord=2, dim=-1)
            delta_norm = torch.linalg.vector_norm(delta, ord=2, dim=-1)
            fused_norm = torch.linalg.vector_norm(fused, ord=2, dim=-1)
            delta_ratio = delta_norm / task_norm.clamp_min(1e-12)
            if task.shape[-1] == delta.shape[-1]:
                task_delta_cosine = F.cosine_similarity(task, delta, dim=-1, eps=1e-12)
            else:
                task_delta_cosine = torch.zeros_like(delta_norm)
            if task.shape[-1] == fused.shape[-1]:
                task_fused_cosine = F.cosine_similarity(task, fused, dim=-1, eps=1e-12)
            else:
                task_fused_cosine = torch.zeros_like(delta_norm)
            values_by_name = {
                "task_norm": task_norm,
                "scene_delta_norm": delta_norm,
                "fused_norm": fused_norm,
                "scene_delta_ratio": delta_ratio,
                "task_scene_delta_cosine": task_delta_cosine,
                "task_fused_cosine": task_fused_cosine,
            }
            for domain_idx in range(int(self.num_domains)):
                selected = domain_values == domain_idx
                count = int(selected.sum().item())
                if count <= 0:
                    continue
                for name, values in values_by_name.items():
                    _accumulate_norms(
                        self._scene_token_fusion_stats[(domain_idx, str(name))],
                        values[selected],
                        sample_count=count,
                    )

    def _apply_scene_token_readout(
            self,
            dense_readout: torch.Tensor,
            domain_ids: torch.Tensor,
            *,
            use_scene_context: bool = True,
    ) -> torch.Tensor:
        if self.scene_token_mode == "none" or not use_scene_context:
            return dense_readout
        scene_summary = self._latest_scene_summary
        if scene_summary is None or int(scene_summary.shape[0]) != int(dense_readout.shape[0]):
            return dense_readout
        scene_summary = scene_summary.to(dtype=dense_readout.dtype, device=dense_readout.device)
        if self.scene_token_fusion_mode == "block_select_readout":
            scene_tokens = self._latest_scene_tokens
            source_tokens = self._latest_scene_block_tokens
            if scene_tokens is None or source_tokens is None or self.scene_block_selector is None:
                return dense_readout
            source = source_tokens.detach() if self.scene_token_detach_backbone else source_tokens
            selected_flat, selector_stats = self.scene_block_selector(
                scene_tokens.to(dtype=dense_readout.dtype, device=dense_readout.device),
                source.to(dtype=dense_readout.dtype, device=dense_readout.device),
                dense_readout,
            )
            scene_delta = float(self.scene_token_fusion_scale) * selected_flat
            fused = dense_readout + scene_delta
            self._record_scene_token_block_select(selector_stats, domain_ids)
        elif self.scene_token_fusion_mode == "add_readout":
            if self.scene_fusion_projection is None:
                raise RuntimeError("scene_token_fusion_mode='add_readout' requires scene_fusion_projection.")
            scene_delta = float(self.scene_token_fusion_scale) * self.scene_fusion_projection(scene_summary)
            fused = dense_readout + scene_delta
        elif self.scene_token_fusion_mode == "concat_readout":
            scene_delta = float(self.scene_token_fusion_scale) * scene_summary
            fused = torch.cat([dense_readout, scene_delta], dim=-1)
        else:
            raise ValueError(f"Unknown scene_token_fusion_mode={self.scene_token_fusion_mode!r}.")
        self._record_scene_token_fusion(dense_readout, scene_delta, fused, domain_ids)
        return fused

    def _residual_beta(
            self,
            domain_ids: torch.Tensor,
            beta_input: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.residual_beta_mode == "sample_dense":
            if self.residual_beta_head is None or beta_input is None:
                return None
            logits = self.residual_beta_head(beta_input).view(-1)
            if self.residual_beta_domain_bias is not None:
                logits = logits + self.residual_beta_domain_bias[domain_ids.long().view(-1)].to(
                    dtype=logits.dtype,
                    device=logits.device,
                )
            return float(self.residual_beta_max) * torch.sigmoid(logits)
        if self.residual_beta_logits is None:
            return None
        logits = self.residual_beta_logits[domain_ids.long().view(-1)]
        return float(self.residual_beta_max) * torch.sigmoid(logits)

    def _tokens_to_logit_components(
            self,
            tokens: torch.Tensor,
            domain_ids: torch.Tensor,
            *,
            use_scene_context: bool = True,
    ) -> dict[str, torch.Tensor]:
        if self.readout_mode in {"split_head_mean", "split_head_domain_weight"}:
            stream_logits = torch.stack(
                [
                    head(tokens[:, stream_idx], domain_ids).view(-1)
                    for stream_idx, head in enumerate(self.stream_heads)
                ],
                dim=1,
            )
            if self.readout_mode == "split_head_mean" or self.domain_stream_logit_weight is None:
                logits = stream_logits.mean(dim=1)
                return {
                    "logits": logits,
                    "dense_logits": logits,
                    "stream_logits": stream_logits,
                }
            weights = torch.softmax(self.domain_stream_logit_weight[domain_ids.long().view(-1)], dim=-1)
            logits = torch.sum(stream_logits * weights.to(dtype=stream_logits.dtype, device=stream_logits.device), dim=1)
            return {
                "logits": logits,
                "dense_logits": logits,
                "stream_logits": stream_logits,
                "stream_logit_weights": weights,
            }
        if self.head is None:
            raise RuntimeError("AllDomainExpertMixerModel.head is not initialized for this readout mode.")
        base_readout = self._tokens_to_base_readout(tokens)
        dense_readout = self.readout_projection(base_readout) if self.readout_projection is not None else base_readout
        task_readout = self._apply_scene_token_readout(
            dense_readout,
            domain_ids,
            use_scene_context=use_scene_context,
        )
        task_readout = self._apply_scene_query_readout(task_readout, domain_ids)
        dense_logits = self.head(task_readout, domain_ids).view(-1)
        if self.residual_side_mode == "readout_projection":
            if self.residual_side_projection is None or self.residual_side_head is None:
                raise RuntimeError("Residual side path is not initialized.")
            side_readout = self.residual_side_projection(base_readout)
            side_logits = self.residual_side_head(side_readout, domain_ids).view(-1)
            beta = self._residual_beta(domain_ids, dense_readout)
            if beta is None:
                beta = torch.zeros_like(side_logits)
            side_delta_logits = side_logits
            if self.residual_side_combine == "centered":
                side_delta_logits = side_logits - side_logits.detach().mean()
            elif self.residual_side_combine == "side_minus_dense_stopgrad":
                side_delta_logits = side_logits - dense_logits.detach()
            residual_logits = beta.to(dtype=side_delta_logits.dtype, device=side_delta_logits.device) * side_delta_logits
            logits = dense_logits + residual_logits
            return {
                "logits": logits,
                "dense_logits": dense_logits,
                "side_logits": side_logits,
                "side_delta_logits": side_delta_logits,
                "residual_logits": residual_logits,
                "residual_beta": beta.to(dtype=side_delta_logits.dtype, device=side_delta_logits.device),
            }
        return {
            "logits": dense_logits,
            "dense_logits": dense_logits,
            "task_readout": task_readout,
        }

    def _tokens_to_logits(self, tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._tokens_to_logit_components(tokens, domain_ids)["logits"].view(-1)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        tokens, aux_tokens = self._build_stream_tokens(x, domain_ids)
        self._latest_aux_logits = self._compute_aux_logits(aux_tokens)
        self._latest_logit_components = self._tokens_to_logit_components(tokens, domain_ids)
        return self._latest_logit_components["logits"].view(-1)

    def logits(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def trace_interaction(
            self,
            interaction,
            *,
            structure_ste: bool = True,
            detach_structure: bool = False,
    ) -> dict[str, torch.Tensor]:
        prev_structure_ste = self._structure_ste
        prev_detach_structure = self._detach_structure
        self._structure_ste = bool(structure_ste)
        self._detach_structure = bool(detach_structure)
        try:
            x, domain_ids = self.encode_features(interaction)
            logits = self.forward(x, domain_ids).view(-1)
            return {
                "logits": logits,
                "domain_ids": domain_ids,
            }
        finally:
            self._structure_ste = prev_structure_ste
            self._detach_structure = prev_detach_structure

    def calculate_loss(
            self,
            interaction,
            *,
            structure_ste: bool = True,
            detach_structure: bool = False,
    ):
        prev_structure_ste = self._structure_ste
        prev_detach_structure = self._detach_structure
        self._structure_ste = bool(structure_ste)
        self._detach_structure = bool(detach_structure)
        labels = interaction[self.LABEL].float().view(-1)
        domain_ids = interaction[self.DOMAIN].long().view(-1)
        try:
            logits = self.logits(interaction).view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if self.lsp.active:
                self.lsp.add("task", loss)
            if self._latest_aux_logits is not None:
                self._record_aux_train_logloss(self._latest_aux_logits, labels, domain_ids)
            if self.expert_aux_weight > 0.0 and self._latest_aux_logits is not None:
                aux_labels = labels.view(-1, 1).expand_as(self._latest_aux_logits)
                aux_loss = F.binary_cross_entropy_with_logits(self._latest_aux_logits, aux_labels)
                loss = loss + self.expert_aux_weight * aux_loss
                if self.lsp.active:
                    self.lsp.add("auxiliary.expert", aux_loss, weight=self.expert_aux_weight)
            if self.residual_side_aux_weight > 0.0:
                side_logits = self._latest_logit_components.get("side_logits")
                if side_logits is not None:
                    if self.residual_side_aux_mode == "dense_residual_bce":
                        dense_logits = self._latest_logit_components.get("dense_logits")
                        side_delta_logits = self._latest_logit_components.get("side_delta_logits", side_logits)
                        if dense_logits is None:
                            side_loss = F.binary_cross_entropy_with_logits(side_logits.view(-1), labels)
                        else:
                            residual_target_logits = dense_logits.detach().view(-1) + side_delta_logits.view(-1)
                            side_loss = F.binary_cross_entropy_with_logits(residual_target_logits, labels)
                    else:
                        side_loss = F.binary_cross_entropy_with_logits(side_logits.view(-1), labels)
                    loss = loss + self.residual_side_aux_weight * side_loss
                    if self.lsp.active:
                        self.lsp.add(
                            "auxiliary.residual_side",
                            side_loss,
                            weight=self.residual_side_aux_weight,
                        )
            if self.mask_reg_weight > 0.0 and self._latest_mask is not None:
                mask_loss = (self._latest_mask - 1.0).pow(2).mean()
                loss = loss + self.mask_reg_weight * mask_loss
                if self.lsp.active:
                    self.lsp.add("regularization.mask", mask_loss, weight=self.mask_reg_weight)
            if self.finalmlp_domain_mask_reg_weight > 0.0:
                mask_regs = []
                for mask_logits in (
                    self.finalmlp_domain_input_mask_logits,
                    self.finalmlp_domain_prefusion_mask_logits,
                ):
                    if mask_logits is not None:
                        mask = 2.0 * torch.sigmoid(float(self.finalmlp_domain_mask_sharpness) * mask_logits)
                        mask_regs.append((mask - 1.0).pow(2).mean())
                if mask_regs:
                    finalmlp_mask_loss = torch.stack(mask_regs).mean()
                    loss = loss + self.finalmlp_domain_mask_reg_weight * finalmlp_mask_loss
                    if self.lsp.active:
                        self.lsp.add(
                            "regularization.finalmlp_domain_mask",
                            finalmlp_mask_loss,
                            weight=self.finalmlp_domain_mask_reg_weight,
                        )
            return loss
        finally:
            self._structure_ste = prev_structure_ste
            self._detach_structure = prev_detach_structure

    def predict_aux_logits(self, interaction) -> torch.Tensor:
        x, domain_ids = self.encode_features(interaction)
        _, aux_tokens = self._build_stream_tokens(x, domain_ids)
        return self._compute_aux_logits(aux_tokens)

    def logits_from_tokens(self, tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._tokens_to_logit_components(tokens, domain_ids, use_scene_context=False)["logits"].view(-1)

    def trace_streams(self, interaction) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        x, domain_ids = self.encode_features(interaction)
        if self.domain_mask is not None:
            x = x * self.domain_mask(x, domain_ids)
        stages: dict[str, torch.Tensor] = {}
        if self.architecture == "all_expert":
            tokens = torch.stack(
                [
                    self._run_stream_module(expert, x, domain_ids, stream_idx)
                    for stream_idx, expert in enumerate(self.experts)
                ],
                dim=1,
            )
            stages["after_expert"] = tokens
        elif self.architecture == "mid_private":
            shared = self.shared_lower(x)
            tokens = torch.stack(
                [
                    self._run_stream_module(expert, shared, domain_ids, stream_idx)
                    for stream_idx, expert in enumerate(self.experts)
                ],
                dim=1,
            )
            stages["after_private_mid"] = tokens
        else:
            before_shared = torch.stack(
                [
                    self._run_stream_module(expert, x, domain_ids, stream_idx)
                    for stream_idx, expert in enumerate(self.private_lowers)
                ],
                dim=1,
            )
            stages["before_shared"] = before_shared
            if self.pre_shared_merge_mode == "none":
                tokens = torch.stack(
                    [self.shared_middle(before_shared[:, stream_idx]) for stream_idx in range(self.num_experts)],
                    dim=1,
                )
            else:
                merged = self._merge_pre_shared_tokens(before_shared)
                stages["merged_before_shared"] = merged.unsqueeze(1)
                tokens = self.shared_middle(merged).unsqueeze(1)
            stages["after_shared"] = tokens
            if self.post_shared_interest_mode != "identity":
                tokens = self._apply_post_shared_interest(tokens)
                stages["after_interest"] = tokens
        if self.use_token_mixer:
            stages["before_mixer"] = tokens
            tokens = self.mixer(tokens)
            stages["after_mixer"] = tokens
        stages["before_readout"] = tokens
        components = self._tokens_to_logit_components(tokens, domain_ids, use_scene_context=False)
        return {
            "domain_ids": domain_ids,
            "logits": components["logits"],
            "tokens": tokens,
            "stages": stages,
            **components,
        }

    def export_train_observation_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for domain_idx, stats in sorted(self._mask_stats.items()):
            value_count = max(1.0, float(stats["value_count"]))
            mean, std = _stats_mean_std(stats, "value_count")
            rows.append({
                "stat_group": "all_expert_mask",
                "domain": int(domain_idx),
                "expert_index": -1,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "value_count": int(stats["value_count"]),
                "mask_mean": mean,
                "mask_std": std,
                "mask_min": float(stats["min"]) if stats["min"] != float("inf") else 1.0,
                "mask_max": float(stats["max"]) if stats["max"] != float("-inf") else 1.0,
                "mask_abs_dev_mean": float(stats["abs_dev_sum"]) / value_count,
                "mask_low_frac": float(stats["low_count"]) / value_count,
                "mask_high_frac": float(stats["high_count"]) / value_count,
                "mask_sat_low_frac": float(stats["sat_low_count"]) / value_count,
                "mask_sat_high_frac": float(stats["sat_high_count"]) / value_count,
            })
            rows.append({
                "stat_group": "stream_mask",
                "mask_position": "input",
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream": -1,
                "expert_index": -1,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "value_count": int(stats["value_count"]),
                "mask_mean": mean,
                "mask_std": std,
                "mask_min": float(stats["min"]) if stats["min"] != float("inf") else 1.0,
                "mask_max": float(stats["max"]) if stats["max"] != float("-inf") else 1.0,
                "mask_abs_dev_mean": float(stats["abs_dev_sum"]) / value_count,
                "mask_low_frac": float(stats["low_count"]) / value_count,
                "mask_high_frac": float(stats["high_count"]) / value_count,
                "mask_sat_low_frac": float(stats["sat_low_count"]) / value_count,
                "mask_sat_high_frac": float(stats["sat_high_count"]) / value_count,
            })
        rows.extend(self._export_norm_rows("all_expert_token_norm", self._token_norm_stats, "token_norm"))
        rows.extend(self._export_norm_rows("all_expert_token_grad", self._token_grad_stats, "token_grad_l2"))
        rows.extend(self._export_norm_rows("all_expert_mixer_delta", self._mixer_delta_stats, "mixer_delta_norm"))
        rows.extend(self._export_norm_rows("all_expert_mixer_ratio", self._mixer_ratio_stats, "mixer_delta_ratio"))
        rows.extend(self._export_stream_norm_rows())
        rows.extend(self._export_stream_grad_rows())
        rows.extend(self._export_stream_pair_rows())
        rows.extend(self._export_shared_middle_rows())
        rows.extend(self._export_post_shared_interest_rows())
        rows.extend(self._export_aux_train_rows())
        rows.extend(self._export_cross_basis_signal_rows())
        rows.extend(self._export_finalmlp_gate_rows())
        rows.extend(self._export_finalmlp_domain_mask_rows())
        rows.extend(self._export_finalmlp_domain_mask_param_rows())
        rows.extend(self._export_reception_gate_rows())
        rows.extend(self._export_scene_query_gate_rows())
        rows.extend(self._export_scene_query_fusion_rows())
        rows.extend(self._export_scene_token_attention_rows())
        rows.extend(self._export_scene_token_delta_rows())
        rows.extend(self._export_scene_token_fusion_rows())
        rows.extend(self._export_scene_token_block_select_rows())
        rows.extend(self._export_scene_token_block_select_summary_rows())
        return rows

    def _parameter_count_rows(self) -> list[dict[str, object]]:
        def counts_from_parameters(parameters) -> tuple[int, int, int]:
            total = 0
            requires_grad = 0
            for parameter in parameters:
                size = int(parameter.numel())
                total += size
                if parameter.requires_grad:
                    requires_grad += size
            optimizable = requires_grad if requires_grad > 0 else total
            return total, optimizable, requires_grad

        rows: list[dict[str, object]] = []
        total, trainable, requires_grad = counts_from_parameters(self.parameters())
        common = {
            "architecture": self.architecture,
            "cross_basis": self.cross_basis,
            "gate_reception": self.gate_reception,
            "readout_mode": self.readout_mode,
            "shared_middle_mode": self.shared_middle_mode,
            "post_shared_interest_mode": self.post_shared_interest_mode,
            "finalmlp_domain_mask_mode": self.finalmlp_domain_mask_mode,
            "finalmlp_domain_mask_sharpness": float(self.finalmlp_domain_mask_sharpness),
            "scene_query_mode": self.scene_query_mode,
            "scene_query_fusion_mode": self.scene_query_fusion_mode,
            "scene_query_sharpness": float(self.scene_query_sharpness),
            "scene_query_detach_backbone": int(self.scene_query_detach_backbone),
            "scene_token_mode": self.scene_token_mode,
            "scene_token_stages": ",".join(str(stage) for stage in self.scene_token_stages),
            "scene_num_global_tokens": int(self.scene_num_global_tokens),
            "scene_num_domain_tokens": int(self.scene_num_domain_tokens),
            "scene_token_num_heads": int(self.scene_token_num_heads),
            "scene_token_mixer_layers": int(self.scene_token_mixer_layers),
            "scene_token_source_mode": self.scene_token_source_mode,
            "scene_token_unimix_block_size": int(self.scene_token_unimix_block_size),
            "scene_token_unimix_temperature": float(self.scene_token_unimix_temperature),
            "scene_token_unimix_sinkhorn_iters": int(self.scene_token_unimix_sinkhorn_iters),
            "scene_token_detach_backbone": int(self.scene_token_detach_backbone),
            "scene_token_fusion_mode": self.scene_token_fusion_mode,
            "scene_token_fusion_scale": float(self.scene_token_fusion_scale),
            "scene_token_zero_init_residual": int(self.scene_token_zero_init_residual),
            "scene_token_zero_init_fusion": int(self.scene_token_zero_init_fusion),
            "scene_token_fusion_init_std": float(self.scene_token_fusion_init_std),
            "scene_token_context_mode": self.scene_token_context_mode,
            "scene_token_context_scale": float(self.scene_token_context_scale),
            "scene_token_context_detach_input": int(self.scene_token_context_detach_input),
            "prediction_head_mode": self.prediction_head_mode,
            "pre_shared_merge_mode": self.pre_shared_merge_mode,
            "readout_projection_dim": int(self.readout_projection_dim),
            "residual_side_mode": self.residual_side_mode,
            "residual_beta_mode": self.residual_beta_mode,
            "residual_side_aux_mode": self.residual_side_aux_mode,
        }
        rows.append({
            "stat_group": "model_parameter_count",
            "param_scope": "all",
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "requires_grad_parameters": int(requires_grad),
            **common,
        })
        module_scopes: list[tuple[str, Optional[nn.Module]]] = [
            ("domain_mask", self.domain_mask),
            ("experts", self.experts),
            ("shared_lower", self.shared_lower),
            ("private_lowers", self.private_lowers),
            ("shared_middle", self.shared_middle),
            ("post_shared_interest", self.post_shared_interest),
            ("post_shared_interest_heads", self.post_shared_interest_heads),
            ("cross_hard_gate", self.cross_hard_gate),
            ("mixer", self.mixer),
            ("scene_read_mixers", self.scene_read_mixers),
            ("scene_read_blocks", self.scene_read_blocks),
            ("scene_block_selector", self.scene_block_selector),
            ("scene_fusion_projection", self.scene_fusion_projection),
            ("readout_projection", self.readout_projection),
            ("scene_query", self.scene_query),
            ("head", self.head),
            ("stream_heads", self.stream_heads),
            ("aux_heads", self.aux_heads),
            ("residual_side_projection", self.residual_side_projection),
            ("residual_side_head", self.residual_side_head),
            ("residual_beta_head", self.residual_beta_head),
        ]
        for scope, module in module_scopes:
            if module is None:
                continue
            scope_total, scope_trainable, scope_requires_grad = counts_from_parameters(module.parameters())
            rows.append({
                "stat_group": "model_parameter_count",
                "param_scope": scope,
                "total_parameters": int(scope_total),
                "trainable_parameters": int(scope_trainable),
                "requires_grad_parameters": int(scope_requires_grad),
                **common,
            })
        if self.domain_stream_logit_weight is not None:
            scope_total, scope_trainable, scope_requires_grad = counts_from_parameters([self.domain_stream_logit_weight])
            rows.append({
                "stat_group": "model_parameter_count",
                "param_scope": "domain_stream_logit_weight",
                "total_parameters": int(scope_total),
                "trainable_parameters": int(scope_trainable),
                "requires_grad_parameters": int(scope_requires_grad),
                **common,
            })
        for scope, parameter in (
            ("finalmlp_domain_input_mask", self.finalmlp_domain_input_mask_logits),
            ("finalmlp_domain_prefusion_mask", self.finalmlp_domain_prefusion_mask_logits),
            ("scene_global_tokens", self.scene_global_tokens),
            ("scene_domain_tokens", self.scene_domain_tokens),
        ):
            if parameter is None:
                continue
            scope_total, scope_trainable, scope_requires_grad = counts_from_parameters([parameter])
            rows.append({
                "stat_group": "model_parameter_count",
                "param_scope": scope,
                "total_parameters": int(scope_total),
                "trainable_parameters": int(scope_trainable),
                "requires_grad_parameters": int(scope_requires_grad),
                **common,
            })
        if self.residual_beta_logits is not None:
            scope_total, scope_trainable, scope_requires_grad = counts_from_parameters([self.residual_beta_logits])
            rows.append({
                "stat_group": "model_parameter_count",
                "param_scope": "residual_beta_logits",
                "total_parameters": int(scope_total),
                "trainable_parameters": int(scope_trainable),
                "requires_grad_parameters": int(scope_requires_grad),
                **common,
            })
        return rows

    def export_static_observation_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = [
            {
                "stat_group": "factorial_design",
                "architecture": self.architecture,
                "topology": f"{self.architecture}_{self.readout_mode}_e{self.num_experts}",
                "cross_basis": self.cross_basis,
                "gate_reception": self.gate_reception,
                "gate_position": "cross_channel" if self.cross_basis == "lq_rank64" else "stream_output",
                "hard_or_soft": "hard" if self.cross_hard_gate.is_hard else "soft" if self.gate_reception != "none" else "none",
                "shared_or_separated": (
                    "shared"
                    if self.gate_reception == "shared_soft_gate"
                    else "stream"
                    if self.gate_reception == "stream_sep_soft_gate"
                    else "domain"
                    if self.gate_reception in {
                        "domain_sep_soft_gate",
                        "domain_hard_rank_gate_sqrt",
                        "domain_hard_topk_gate",
                    }
                    else "none"
                ),
                "num_experts": int(self.num_experts),
                "readout_num_tokens": int(self.readout_num_tokens),
                "rank_dim": int(self.cross_hard_gate.dim),
                "readout_projection_dim": int(self.readout_projection_dim),
                "shared_middle_mode": self.shared_middle_mode,
                "shared_middle_residual_scale": float(self.shared_middle_residual_scale),
                "post_shared_interest_mode": self.post_shared_interest_mode,
                "post_shared_interest_residual_scale": float(self.post_shared_interest_residual_scale),
                "pre_shared_merge_mode": self.pre_shared_merge_mode,
                "finalmlp_domain_mask_mode": self.finalmlp_domain_mask_mode,
                "finalmlp_domain_mask_sharpness": float(self.finalmlp_domain_mask_sharpness),
                "finalmlp_domain_mask_reg_weight": float(self.finalmlp_domain_mask_reg_weight),
                "scene_query_mode": self.scene_query_mode,
                "scene_query_fusion_mode": self.scene_query_fusion_mode,
                "scene_query_sharpness": float(self.scene_query_sharpness),
                "scene_query_detach_backbone": int(self.scene_query_detach_backbone),
                "scene_token_mode": self.scene_token_mode,
                "scene_token_stages": ",".join(str(stage) for stage in self.scene_token_stages),
                "scene_num_global_tokens": int(self.scene_num_global_tokens),
                "scene_num_domain_tokens": int(self.scene_num_domain_tokens),
                "scene_token_num_heads": int(self.scene_token_num_heads),
                "scene_token_mixer_layers": int(self.scene_token_mixer_layers),
                "scene_token_source_mode": self.scene_token_source_mode,
                "scene_token_unimix_block_size": int(self.scene_token_unimix_block_size),
                "scene_token_unimix_temperature": float(self.scene_token_unimix_temperature),
                "scene_token_unimix_sinkhorn_iters": int(self.scene_token_unimix_sinkhorn_iters),
                "scene_token_detach_backbone": int(self.scene_token_detach_backbone),
                "scene_token_fusion_mode": self.scene_token_fusion_mode,
                "scene_token_fusion_scale": float(self.scene_token_fusion_scale),
                "scene_token_zero_init_residual": int(self.scene_token_zero_init_residual),
                "scene_token_zero_init_fusion": int(self.scene_token_zero_init_fusion),
                "scene_token_fusion_init_std": float(self.scene_token_fusion_init_std),
                "scene_token_context_mode": self.scene_token_context_mode,
                "scene_token_context_scale": float(self.scene_token_context_scale),
                "scene_token_context_detach_input": int(self.scene_token_context_detach_input),
                "prediction_head_mode": self.prediction_head_mode,
                "residual_side_mode": self.residual_side_mode,
                "residual_readout_projection_dim": int(self.residual_readout_projection_dim),
                "residual_beta_max": float(self.residual_beta_max),
                "residual_side_aux_weight": float(self.residual_side_aux_weight),
                "residual_side_aux_mode": self.residual_side_aux_mode,
                "residual_side_combine": self.residual_side_combine,
            }
        ]
        rows.extend(self._parameter_count_rows())
        rows.extend(self.cross_hard_gate.static_rows())
        rows.extend(self._residual_side_static_rows())
        if self.readout_mode in {"split_head_mean", "split_head_domain_weight"}:
            rows.extend(self._split_head_static_rows())
            return rows
        domain_networks = getattr(getattr(self.head, "head", None), "domain_networks", None)
        if domain_networks is None:
            return rows
        for domain_idx, tower in enumerate(domain_networks):
            first_linear = None
            for module in tower.modules():
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
            if first_linear is None:
                continue
            weight = first_linear.weight.detach().float()
            effective_readout_projection = False
            if self.readout_projection is not None and self.readout_mode == "flatten":
                weight = torch.matmul(weight, self.readout_projection.weight.detach().float())
                effective_readout_projection = True
            if self.readout_mode == "mean":
                l2 = float(torch.linalg.vector_norm(weight, ord=2).item())
                rows.append({
                    "stat_group": "tower_readout_block_norm",
                    "domain": int(domain_idx),
                    "stream": -1,
                    "expert_index": -1,
                    "readout_mode": self.readout_mode,
                    "architecture": self.architecture,
                    "readout_projection_dim": int(self.readout_projection_dim),
                    "effective_readout_projection": int(effective_readout_projection),
                    "first_linear_weight_l2": l2,
                    "first_linear_weight_abs_mean": float(weight.abs().mean().item()),
                    "block_share": 1.0,
                })
                continue
            expected = self.readout_num_tokens * self.output_dim
            if weight.shape[1] != expected:
                continue
            blocks = weight.view(weight.shape[0], self.readout_num_tokens, self.output_dim)
            block_l2 = torch.linalg.vector_norm(blocks, ord=2, dim=(0, 2))
            total = float(block_l2.sum().clamp_min(1e-12).item())
            for stream_idx in range(self.readout_num_tokens):
                block = blocks[:, stream_idx, :]
                l2 = float(block_l2[stream_idx].item())
                rows.append({
                    "stat_group": "tower_readout_block_norm",
                    "domain": int(domain_idx),
                    "stream": int(stream_idx),
                    "expert_index": int(stream_idx),
                    "readout_mode": self.readout_mode,
                    "architecture": self.architecture,
                    "readout_projection_dim": int(self.readout_projection_dim),
                    "effective_readout_projection": int(effective_readout_projection),
                    "first_linear_weight_l2": l2,
                    "first_linear_weight_abs_mean": float(block.abs().mean().item()),
                    "block_share": l2 / total,
                })
        if self.readout_projection is not None and self.readout_mode == "flatten":
            projection_weight = self.readout_projection.weight.detach().float()
            expected = self.readout_num_tokens * self.output_dim
            if projection_weight.shape[1] == expected:
                blocks = projection_weight.view(projection_weight.shape[0], self.readout_num_tokens, self.output_dim)
                block_l2 = torch.linalg.vector_norm(blocks, ord=2, dim=(0, 2))
                total = float(block_l2.sum().clamp_min(1e-12).item())
                for stream_idx in range(self.readout_num_tokens):
                    block = blocks[:, stream_idx, :]
                    l2 = float(block_l2[stream_idx].item())
                    rows.append({
                        "stat_group": "readout_projection_block_norm",
                        "domain": -1,
                        "stream": int(stream_idx),
                        "expert_index": int(stream_idx),
                        "readout_mode": self.readout_mode,
                        "architecture": self.architecture,
                        "readout_projection_dim": int(self.readout_projection_dim),
                        "projection_weight_l2": l2,
                        "projection_weight_abs_mean": float(block.abs().mean().item()),
                        "block_share": l2 / total,
                    })
        return rows

    def _residual_side_static_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if self.residual_beta_logits is None:
            return rows
        with torch.no_grad():
            beta = float(self.residual_beta_max) * torch.sigmoid(self.residual_beta_logits.detach().float())
            for domain_idx, value in enumerate(beta.view(-1)):
                rows.append({
                    "stat_group": "residual_beta_by_domain",
                    "domain": int(domain_idx),
                    "stream": -1,
                    "readout_mode": self.readout_mode,
                    "architecture": self.architecture,
                    "residual_side_mode": self.residual_side_mode,
                    "residual_readout_projection_dim": int(self.residual_readout_projection_dim),
                    "residual_beta": float(value.item()),
                    "residual_beta_max": float(self.residual_beta_max),
                    "residual_side_aux_weight": float(self.residual_side_aux_weight),
                    "residual_side_combine": self.residual_side_combine,
                })
        if self.residual_side_projection is not None:
            projection_weight = self.residual_side_projection.weight.detach().float()
            expected = self.readout_num_tokens * self.output_dim
            if self.readout_mode == "mean":
                rows.append({
                    "stat_group": "residual_readout_projection_block_norm",
                    "domain": -1,
                    "stream": -1,
                    "expert_index": -1,
                    "readout_mode": self.readout_mode,
                    "architecture": self.architecture,
                    "residual_side_mode": self.residual_side_mode,
                    "residual_side_combine": self.residual_side_combine,
                    "residual_readout_projection_dim": int(self.residual_readout_projection_dim),
                    "projection_weight_l2": float(torch.linalg.vector_norm(projection_weight, ord=2).item()),
                    "projection_weight_abs_mean": float(projection_weight.abs().mean().item()),
                    "block_share": 1.0,
                })
            elif projection_weight.shape[1] == expected:
                blocks = projection_weight.view(projection_weight.shape[0], self.readout_num_tokens, self.output_dim)
                block_l2 = torch.linalg.vector_norm(blocks, ord=2, dim=(0, 2))
                total = float(block_l2.sum().clamp_min(1e-12).item())
                for stream_idx in range(self.readout_num_tokens):
                    block = blocks[:, stream_idx, :]
                    l2 = float(block_l2[stream_idx].item())
                    rows.append({
                        "stat_group": "residual_readout_projection_block_norm",
                        "domain": -1,
                        "stream": int(stream_idx),
                        "expert_index": int(stream_idx),
                        "readout_mode": self.readout_mode,
                        "architecture": self.architecture,
                        "residual_side_mode": self.residual_side_mode,
                        "residual_side_combine": self.residual_side_combine,
                        "residual_readout_projection_dim": int(self.residual_readout_projection_dim),
                        "projection_weight_l2": l2,
                        "projection_weight_abs_mean": float(block.abs().mean().item()),
                        "block_share": l2 / total,
                    })
        domain_networks = getattr(getattr(self.residual_side_head, "head", None), "domain_networks", None)
        if domain_networks is None or self.residual_side_projection is None:
            return rows
        projection_weight = self.residual_side_projection.weight.detach().float()
        expected = self.readout_num_tokens * self.output_dim
        for domain_idx, tower in enumerate(domain_networks):
            first_linear = None
            for module in tower.modules():
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
            if first_linear is None:
                continue
            weight = first_linear.weight.detach().float()
            effective = torch.matmul(weight, projection_weight)
            if self.readout_mode == "mean":
                rows.append({
                    "stat_group": "residual_tower_readout_block_norm",
                    "domain": int(domain_idx),
                    "stream": -1,
                    "expert_index": -1,
                    "readout_mode": self.readout_mode,
                    "architecture": self.architecture,
                    "residual_side_mode": self.residual_side_mode,
                    "residual_side_combine": self.residual_side_combine,
                    "residual_readout_projection_dim": int(self.residual_readout_projection_dim),
                    "first_linear_weight_l2": float(torch.linalg.vector_norm(effective, ord=2).item()),
                    "first_linear_weight_abs_mean": float(effective.abs().mean().item()),
                    "block_share": 1.0,
                })
                continue
            if effective.shape[1] != expected:
                continue
            blocks = effective.view(effective.shape[0], self.readout_num_tokens, self.output_dim)
            block_l2 = torch.linalg.vector_norm(blocks, ord=2, dim=(0, 2))
            total = float(block_l2.sum().clamp_min(1e-12).item())
            for stream_idx in range(self.readout_num_tokens):
                block = blocks[:, stream_idx, :]
                l2 = float(block_l2[stream_idx].item())
                rows.append({
                    "stat_group": "residual_tower_readout_block_norm",
                    "domain": int(domain_idx),
                    "stream": int(stream_idx),
                    "expert_index": int(stream_idx),
                    "readout_mode": self.readout_mode,
                    "architecture": self.architecture,
                    "residual_side_mode": self.residual_side_mode,
                    "residual_side_combine": self.residual_side_combine,
                    "residual_readout_projection_dim": int(self.residual_readout_projection_dim),
                    "first_linear_weight_l2": l2,
                    "first_linear_weight_abs_mean": float(block.abs().mean().item()),
                    "block_share": l2 / total,
                })
        return rows

    def _split_head_static_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        weights = None
        if self.domain_stream_logit_weight is not None:
            weights = torch.softmax(self.domain_stream_logit_weight.detach().float(), dim=-1)
            for domain_idx in range(int(weights.shape[0])):
                for stream_idx in range(int(weights.shape[1])):
                    rows.append({
                        "stat_group": "split_head_domain_weight",
                        "domain": int(domain_idx),
                        "stream": int(stream_idx),
                        "expert_index": int(stream_idx),
                        "readout_mode": self.readout_mode,
                        "architecture": self.architecture,
                        "weight": float(weights[domain_idx, stream_idx].item()),
                    })
        per_domain_l2: dict[int, list[tuple[int, torch.Tensor, float]]] = defaultdict(list)
        for stream_idx, head in enumerate(self.stream_heads):
            domain_networks = getattr(getattr(head, "head", None), "domain_networks", None)
            if domain_networks is None:
                continue
            for domain_idx, tower in enumerate(domain_networks):
                first_linear = None
                for module in tower.modules():
                    if isinstance(module, nn.Linear):
                        first_linear = module
                        break
                if first_linear is None:
                    continue
                weight_tensor = first_linear.weight.detach().float()
                l2 = float(torch.linalg.vector_norm(weight_tensor, ord=2).item())
                per_domain_l2[int(domain_idx)].append((int(stream_idx), weight_tensor, l2))
        for domain_idx, values in sorted(per_domain_l2.items()):
            total = sum(l2 for _, _, l2 in values) or 1.0
            for stream_idx, weight_tensor, l2 in values:
                rows.append({
                    "stat_group": "tower_readout_block_norm",
                    "domain": int(domain_idx),
                    "stream": int(stream_idx),
                    "expert_index": int(stream_idx),
                    "readout_mode": self.readout_mode,
                    "architecture": self.architecture,
                    "readout_projection_dim": int(self.readout_projection_dim),
                    "effective_readout_projection": 0,
                    "first_linear_weight_l2": l2,
                    "first_linear_weight_abs_mean": float(weight_tensor.abs().mean().item()),
                    "block_share": float(l2 / total),
                    "domain_logit_weight": (
                        float(weights[domain_idx, stream_idx].item())
                        if weights is not None
                        else 1.0 / max(1, self.num_experts)
                    ),
                })
        return rows

    def _export_cross_basis_signal_rows(self) -> list[dict[str, object]]:
        rows_by_key: Dict[tuple[int, int], dict[str, object]] = {}
        for (domain_idx, stream_idx, name), stats in sorted(self._cross_basis_signal_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            row = rows_by_key.setdefault(
                (domain_idx, stream_idx),
                {
                    "stat_group": "cross_basis_signal",
                    "cross_basis": self.cross_basis,
                    "gate_reception": self.gate_reception,
                    "domain": int(domain_idx),
                    "source_domain": int(domain_idx),
                    "stream": int(stream_idx),
                    "expert_index": int(stream_idx),
                    "architecture": self.architecture,
                    "readout_mode": self.readout_mode,
                    "batches": int(stats["batches"]),
                    "sample_count": int(stats["sample_count"]),
                },
            )
            row[name] = mean
            row[f"{name}_std"] = std
        return list(rows_by_key.values())

    def _export_finalmlp_gate_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (domain_idx, stream_idx, gate_id, field), stats in sorted(self._finalmlp_gate_stats.items()):
            value_count = max(1.0, float(stats["value_count"]))
            mean, std = _stats_mean_std(stats, "value_count")
            prob = min(max(mean / 2.0, 1e-12), 1.0 - 1e-12)
            entropy = -(
                prob * torch.log(torch.tensor(prob))
                + (1.0 - prob) * torch.log(torch.tensor(1.0 - prob))
            )
            rows.append({
                "stat_group": "finalmlp_gate",
                "cross_basis": self.cross_basis,
                "gate_reception": self.gate_reception,
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream": int(stream_idx),
                "expert_index": int(stream_idx),
                "field": str(field),
                "gate_id": str(gate_id),
                "architecture": self.architecture,
                "readout_mode": self.readout_mode,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "value_count": int(stats["value_count"]),
                "gate_mean": mean,
                "gate_std": std,
                "gate_entropy": float(entropy.item()),
                "gate_abs_dev_from_1": float(stats["abs_dev_sum"]) / value_count,
                "gate_low_frac": float(stats["low_count"]) / value_count,
                "gate_high_frac": float(stats["high_count"]) / value_count,
                "gate_sat_low_frac": float(stats["sat_low_count"]) / value_count,
                "gate_sat_high_frac": float(stats["sat_high_count"]) / value_count,
            })
        return rows

    def _export_finalmlp_domain_mask_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (domain_idx, stream_idx, stage, field), stats in sorted(self._finalmlp_domain_mask_stats.items()):
            value_count = max(1.0, float(stats["value_count"]))
            mean, std = _stats_mean_std(stats, "value_count")
            prob = min(max(mean / 2.0, 1e-12), 1.0 - 1e-12)
            entropy = -(
                prob * torch.log(torch.tensor(prob))
                + (1.0 - prob) * torch.log(torch.tensor(1.0 - prob))
            )
            rows.append({
                "stat_group": "finalmlp_domain_stream_mask",
                "cross_basis": self.cross_basis,
                "gate_reception": self.gate_reception,
                "finalmlp_domain_mask_mode": self.finalmlp_domain_mask_mode,
                "finalmlp_domain_mask_sharpness": float(self.finalmlp_domain_mask_sharpness),
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream": int(stream_idx),
                "expert_index": int(stream_idx),
                "field": str(field),
                "mask_stage": str(stage),
                "architecture": self.architecture,
                "readout_mode": self.readout_mode,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "value_count": int(stats["value_count"]),
                "mask_mean": mean,
                "mask_std": std,
                "mask_entropy": float(entropy.item()),
                "mask_abs_dev_from_1": float(stats["abs_dev_sum"]) / value_count,
                "mask_low_frac": float(stats["low_count"]) / value_count,
                "mask_high_frac": float(stats["high_count"]) / value_count,
                "mask_sat_low_frac": float(stats["sat_low_count"]) / value_count,
                "mask_sat_high_frac": float(stats["sat_high_count"]) / value_count,
            })
        return rows

    def _export_finalmlp_domain_mask_param_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []

        def append_rows(logits: nn.Parameter | None, stage: str) -> None:
            if logits is None:
                return
            mask = (2.0 * torch.sigmoid(float(self.finalmlp_domain_mask_sharpness) * logits.detach())).float()
            if stage == "input_feature":
                field_segments = tuple(getattr(self.input_view, "field_segments", ()))
                if not field_segments:
                    field_segments = (("all", 0, int(mask.shape[-1])),)
            else:
                field_segments = (("cross_hidden", 0, int(mask.shape[-1])),)
            for domain_idx in range(int(mask.shape[0])):
                for stream_idx in range(int(mask.shape[1])):
                    for field_name, start, stop in field_segments:
                        values = mask[domain_idx, stream_idx, int(start):int(stop)].reshape(-1)
                        if values.numel() == 0:
                            continue
                        value_count = max(1.0, float(values.numel()))
                        mean = float(values.mean().item())
                        sq_mean = float(values.pow(2).mean().item())
                        std = max(0.0, sq_mean - mean ** 2) ** 0.5
                        prob = min(max(mean / 2.0, 1e-12), 1.0 - 1e-12)
                        entropy = -(
                            prob * torch.log(torch.tensor(prob))
                            + (1.0 - prob) * torch.log(torch.tensor(1.0 - prob))
                        )
                        rows.append({
                            "stat_group": "finalmlp_domain_stream_mask_param",
                            "cross_basis": self.cross_basis,
                            "gate_reception": self.gate_reception,
                            "finalmlp_domain_mask_mode": self.finalmlp_domain_mask_mode,
                            "finalmlp_domain_mask_sharpness": float(self.finalmlp_domain_mask_sharpness),
                            "domain": int(domain_idx),
                            "source_domain": int(domain_idx),
                            "stream": int(stream_idx),
                            "expert_index": int(stream_idx),
                            "field": str(field_name),
                            "mask_stage": str(stage),
                            "architecture": self.architecture,
                            "readout_mode": self.readout_mode,
                            "batches": 0,
                            "sample_count": 0,
                            "value_count": int(values.numel()),
                            "mask_mean": mean,
                            "mask_std": std,
                            "mask_entropy": float(entropy.item()),
                            "mask_abs_dev_from_1": float((values - 1.0).abs().mean().item()),
                            "mask_low_frac": float((values < 0.5).float().mean().item()),
                            "mask_high_frac": float((values > 1.5).float().mean().item()),
                            "mask_sat_low_frac": float((values < 0.1).float().mean().item()),
                            "mask_sat_high_frac": float((values > 1.9).float().mean().item()),
                        })

        append_rows(self.finalmlp_domain_input_mask_logits, "input_feature")
        append_rows(self.finalmlp_domain_prefusion_mask_logits, "pre_fusion")
        return rows

    def _export_reception_gate_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (domain_idx, stream_idx), stats in sorted(self._reception_gate_stats.items()):
            value_count = max(1.0, float(stats["value_count"]))
            mean, std = _stats_mean_std(stats, "value_count")
            rows.append({
                "stat_group": "stream_input_gate",
                "cross_basis": self.cross_basis,
                "gate_reception": self.gate_reception,
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream": int(stream_idx),
                "expert_index": int(stream_idx),
                "field": "cross_channel" if self.cross_basis == "lq_rank64" else "stream_output",
                "architecture": self.architecture,
                "readout_mode": self.readout_mode,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "value_count": int(stats["value_count"]),
                "gate_mean": mean,
                "gate_std": std,
                "gate_abs_dev_from_1": float(stats["abs_dev_sum"]) / value_count,
                "gate_low_frac": float(stats["low_count"]) / value_count,
                "gate_high_frac": float(stats["high_count"]) / value_count,
                "gate_sat_low_frac": float(stats["sat_low_count"]) / value_count,
                "gate_sat_high_frac": float(stats["sat_high_count"]) / value_count,
            })
        return rows

    def _export_scene_query_gate_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (domain_idx, stream_idx), stats in sorted(self._scene_query_gate_stats.items()):
            value_count = max(1.0, float(stats["value_count"]))
            mean, std = _stats_mean_std(stats, "value_count")
            prob = min(max(mean / 2.0, 1e-12), 1.0 - 1e-12)
            entropy = -(
                prob * torch.log(torch.tensor(prob))
                + (1.0 - prob) * torch.log(torch.tensor(1.0 - prob))
            )
            rows.append({
                "stat_group": "scene_query_gate",
                "cross_basis": self.cross_basis,
                "scene_query_mode": self.scene_query_mode,
                "scene_query_fusion_mode": self.scene_query_fusion_mode,
                "scene_query_sharpness": float(self.scene_query_sharpness),
                "scene_query_detach_backbone": int(self.scene_query_detach_backbone),
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream": int(stream_idx),
                "expert_index": int(stream_idx),
                "field": "readout" if int(stream_idx) < 0 else "readout_stream",
                "architecture": self.architecture,
                "readout_mode": self.readout_mode,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "value_count": int(stats["value_count"]),
                "gate_mean": mean,
                "gate_std": std,
                "gate_entropy": float(entropy.item()),
                "gate_abs_dev_from_1": float(stats["abs_dev_sum"]) / value_count,
                "gate_low_frac": float(stats["low_count"]) / value_count,
                "gate_high_frac": float(stats["high_count"]) / value_count,
                "gate_sat_low_frac": float(stats["sat_low_count"]) / value_count,
                "gate_sat_high_frac": float(stats["sat_high_count"]) / value_count,
            })
        return rows

    def _export_scene_query_fusion_rows(self) -> list[dict[str, object]]:
        rows_by_key: Dict[int, dict[str, object]] = {}
        for (domain_idx, name), stats in sorted(self._scene_query_fusion_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            row = rows_by_key.setdefault(
                int(domain_idx),
                {
                    "stat_group": "scene_query_fusion",
                    "domain": int(domain_idx),
                    "source_domain": int(domain_idx),
                    "stream": -1,
                    "expert_index": -1,
                    "architecture": self.architecture,
                    "cross_basis": self.cross_basis,
                    "readout_mode": self.readout_mode,
                    "scene_query_mode": self.scene_query_mode,
                    "scene_query_fusion_mode": self.scene_query_fusion_mode,
                    "scene_query_sharpness": float(self.scene_query_sharpness),
                    "scene_query_detach_backbone": int(self.scene_query_detach_backbone),
                    "batches": int(stats["batches"]),
                    "sample_count": int(stats["sample_count"]),
                },
            )
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        return list(rows_by_key.values())

    def _export_scene_token_attention_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (stage, domain_idx, query_scope, query_idx, stream_idx), stats in sorted(
                self._scene_token_attention_stats.items()
        ):
            mean, std = _stats_mean_std(stats, "sample_count")
            rows.append({
                "stat_group": "scene_token_attention",
                "stage": str(stage),
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "query_scope": str(query_scope),
                "query_index": int(query_idx),
                "stream": int(stream_idx),
                "expert_index": int(stream_idx),
                "architecture": self.architecture,
                "cross_basis": self.cross_basis,
                "readout_mode": self.readout_mode,
                "scene_token_mode": self.scene_token_mode,
                "scene_num_global_tokens": int(self.scene_num_global_tokens),
                "scene_num_domain_tokens": int(self.scene_num_domain_tokens),
                "scene_token_source_mode": self.scene_token_source_mode,
                "scene_token_unimix_block_size": int(self.scene_token_unimix_block_size),
                "scene_token_detach_backbone": int(self.scene_token_detach_backbone),
                "scene_token_zero_init_residual": int(self.scene_token_zero_init_residual),
                "scene_token_zero_init_fusion": int(self.scene_token_zero_init_fusion),
                "scene_token_fusion_init_std": float(self.scene_token_fusion_init_std),
                "scene_token_context_mode": self.scene_token_context_mode,
                "scene_token_context_scale": float(self.scene_token_context_scale),
                "scene_token_context_detach_input": int(self.scene_token_context_detach_input),
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "attention_mean": mean,
                "attention_std": std,
                "attention_min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "attention_max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        return rows

    def _export_scene_token_delta_rows(self) -> list[dict[str, object]]:
        rows_by_key: Dict[tuple[str, int], dict[str, object]] = {}
        for (stage, domain_idx, name), stats in sorted(self._scene_token_delta_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            row = rows_by_key.setdefault(
                (str(stage), int(domain_idx)),
                {
                    "stat_group": "scene_token_delta",
                    "stage": str(stage),
                    "domain": int(domain_idx),
                    "source_domain": int(domain_idx),
                    "stream": -1,
                    "expert_index": -1,
                    "architecture": self.architecture,
                    "cross_basis": self.cross_basis,
                    "readout_mode": self.readout_mode,
                    "scene_token_mode": self.scene_token_mode,
                    "scene_num_global_tokens": int(self.scene_num_global_tokens),
                    "scene_num_domain_tokens": int(self.scene_num_domain_tokens),
                    "scene_token_source_mode": self.scene_token_source_mode,
                    "scene_token_unimix_block_size": int(self.scene_token_unimix_block_size),
                    "scene_token_detach_backbone": int(self.scene_token_detach_backbone),
                    "scene_token_zero_init_residual": int(self.scene_token_zero_init_residual),
                    "scene_token_zero_init_fusion": int(self.scene_token_zero_init_fusion),
                    "scene_token_fusion_init_std": float(self.scene_token_fusion_init_std),
                    "scene_token_context_mode": self.scene_token_context_mode,
                    "scene_token_context_scale": float(self.scene_token_context_scale),
                    "scene_token_context_detach_input": int(self.scene_token_context_detach_input),
                    "batches": int(stats["batches"]),
                    "sample_count": int(stats["sample_count"]),
                },
            )
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        return list(rows_by_key.values())

    def _export_scene_token_fusion_rows(self) -> list[dict[str, object]]:
        rows_by_key: Dict[int, dict[str, object]] = {}
        for (domain_idx, name), stats in sorted(self._scene_token_fusion_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            row = rows_by_key.setdefault(
                int(domain_idx),
                {
                    "stat_group": "scene_token_fusion",
                    "domain": int(domain_idx),
                    "source_domain": int(domain_idx),
                    "stream": -1,
                    "expert_index": -1,
                    "architecture": self.architecture,
                    "cross_basis": self.cross_basis,
                    "readout_mode": self.readout_mode,
                    "scene_token_mode": self.scene_token_mode,
                    "scene_token_fusion_mode": self.scene_token_fusion_mode,
                    "scene_token_fusion_scale": float(self.scene_token_fusion_scale),
                    "scene_token_source_mode": self.scene_token_source_mode,
                    "scene_token_unimix_block_size": int(self.scene_token_unimix_block_size),
                    "scene_token_detach_backbone": int(self.scene_token_detach_backbone),
                    "scene_token_zero_init_residual": int(self.scene_token_zero_init_residual),
                    "scene_token_zero_init_fusion": int(self.scene_token_zero_init_fusion),
                    "scene_token_fusion_init_std": float(self.scene_token_fusion_init_std),
                    "scene_token_context_mode": self.scene_token_context_mode,
                    "scene_token_context_scale": float(self.scene_token_context_scale),
                    "scene_token_context_detach_input": int(self.scene_token_context_detach_input),
                    "prediction_head_mode": self.prediction_head_mode,
                    "batches": int(stats["batches"]),
                    "sample_count": int(stats["sample_count"]),
                },
            )
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        return list(rows_by_key.values())

    def _export_scene_token_block_select_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (domain_idx, block_idx), stats in sorted(self._scene_token_block_select_gate_stats.items()):
            value_count = max(1.0, float(stats["value_count"]))
            mean, std = _stats_mean_std(stats, "value_count")
            rows.append({
                "stat_group": "scene_token_block_select",
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream": int(block_idx),
                "expert_index": int(block_idx),
                "architecture": self.architecture,
                "cross_basis": self.cross_basis,
                "readout_mode": self.readout_mode,
                "scene_token_mode": self.scene_token_mode,
                "scene_token_fusion_mode": self.scene_token_fusion_mode,
                "scene_token_fusion_scale": float(self.scene_token_fusion_scale),
                "scene_token_source_mode": self.scene_token_source_mode,
                "scene_token_unimix_block_size": int(self.scene_token_unimix_block_size),
                "scene_token_detach_backbone": int(self.scene_token_detach_backbone),
                "scene_token_block_select_topk": int(self.scene_token_block_select_topk),
                "scene_token_block_select_temperature": float(self.scene_token_block_select_temperature),
                "scene_token_block_select_hard": int(self.scene_token_block_select_hard),
                "scene_token_block_select_context": int(self.scene_token_block_select_context),
                "prediction_head_mode": self.prediction_head_mode,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "value_count": int(stats["value_count"]),
                "gate_mean": mean,
                "gate_std": std,
                "gate_min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "gate_max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
                "gate_abs_dev_from_1": float(stats["abs_dev_sum"]) / value_count,
                "gate_low_frac": float(stats["low_count"]) / value_count,
                "gate_high_frac": float(stats["high_count"]) / value_count,
                "gate_sat_low_frac": float(stats["sat_low_count"]) / value_count,
                "gate_sat_high_frac": float(stats["sat_high_count"]) / value_count,
            })
        return rows

    def _export_scene_token_block_select_summary_rows(self) -> list[dict[str, object]]:
        rows_by_key: Dict[int, dict[str, object]] = {}
        for (domain_idx, name), stats in sorted(self._scene_token_block_select_summary_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            row = rows_by_key.setdefault(
                int(domain_idx),
                {
                    "stat_group": "scene_token_block_select_summary",
                    "domain": int(domain_idx),
                    "source_domain": int(domain_idx),
                    "stream": -1,
                    "expert_index": -1,
                    "architecture": self.architecture,
                    "cross_basis": self.cross_basis,
                    "readout_mode": self.readout_mode,
                    "scene_token_mode": self.scene_token_mode,
                    "scene_token_fusion_mode": self.scene_token_fusion_mode,
                    "scene_token_fusion_scale": float(self.scene_token_fusion_scale),
                    "scene_token_source_mode": self.scene_token_source_mode,
                    "scene_token_unimix_block_size": int(self.scene_token_unimix_block_size),
                    "scene_token_detach_backbone": int(self.scene_token_detach_backbone),
                    "scene_token_block_select_topk": int(self.scene_token_block_select_topk),
                    "scene_token_block_select_temperature": float(self.scene_token_block_select_temperature),
                    "scene_token_block_select_hard": int(self.scene_token_block_select_hard),
                    "scene_token_block_select_context": int(self.scene_token_block_select_context),
                    "prediction_head_mode": self.prediction_head_mode,
                    "batches": int(stats["batches"]),
                    "sample_count": int(stats["sample_count"]),
                },
            )
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        return list(rows_by_key.values())

    @staticmethod
    def _export_norm_rows(
            stat_group: str,
            source: Dict[tuple[int, int], Dict[str, float]],
            prefix: str,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (domain_idx, expert_idx), stats in sorted(source.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            rows.append({
                "stat_group": stat_group,
                "domain": int(domain_idx),
                "expert_index": int(expert_idx),
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                f"{prefix}_mean": mean,
                f"{prefix}_std": std,
                f"{prefix}_min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                f"{prefix}_max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        return rows

    def _export_stream_norm_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (stage, domain_idx, stream_idx), stats in sorted(self._stream_norm_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            rows.append({
                "stat_group": "stream_token_norm",
                "stage": stage,
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream": int(stream_idx),
                "expert_index": int(stream_idx),
                "architecture": self.architecture,
                "readout_mode": self.readout_mode,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "token_norm_mean": mean,
                "token_norm_std": std,
                "token_norm_min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "token_norm_max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        return rows

    def _export_stream_grad_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (stage, domain_idx, stream_idx), stats in sorted(self._stream_grad_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            rows.append({
                "stat_group": "stream_grad_l2",
                "stage": stage,
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream": int(stream_idx),
                "expert_index": int(stream_idx),
                "architecture": self.architecture,
                "readout_mode": self.readout_mode,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "grad_l2_mean": mean,
                "grad_l2_std": std,
                "grad_l2_cv": std / max(abs(mean), 1e-12),
                "grad_l2_min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "grad_l2_max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        return rows

    def _export_stream_pair_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (stage, domain_idx, left, right), stats in sorted(self._stream_pair_cos_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            rows.append({
                "stat_group": "stream_pair_cosine",
                "stage": stage,
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream_i": int(left),
                "stream_j": int(right),
                "architecture": self.architecture,
                "readout_mode": self.readout_mode,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "cosine_mean": mean,
                "cosine_std": std,
                "cosine_min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "cosine_max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        return rows

    def _export_shared_middle_rows(self) -> list[dict[str, object]]:
        rows_by_key: Dict[tuple[int, int], dict[str, object]] = {}
        for (domain_idx, stream_idx, name), stats in sorted(self._shared_middle_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            row = rows_by_key.setdefault(
                (domain_idx, stream_idx),
                {
                    "stat_group": "shared_middle_delta",
                    "domain": int(domain_idx),
                    "source_domain": int(domain_idx),
                    "stream": int(stream_idx),
                    "expert_index": int(stream_idx),
                    "architecture": self.architecture,
                    "readout_mode": self.readout_mode,
                    "batches": int(stats["batches"]),
                    "sample_count": int(stats["sample_count"]),
                },
            )
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        return list(rows_by_key.values())

    def _export_post_shared_interest_rows(self) -> list[dict[str, object]]:
        rows_by_key: Dict[tuple[int, int], dict[str, object]] = {}
        for (domain_idx, stream_idx, name), stats in sorted(self._post_shared_interest_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            row = rows_by_key.setdefault(
                (domain_idx, stream_idx),
                {
                    "stat_group": "post_shared_interest_delta",
                    "domain": int(domain_idx),
                    "source_domain": int(domain_idx),
                    "stream": int(stream_idx),
                    "expert_index": int(stream_idx),
                    "architecture": self.architecture,
                    "readout_mode": self.readout_mode,
                    "post_shared_interest_mode": self.post_shared_interest_mode,
                    "batches": int(stats["batches"]),
                    "sample_count": int(stats["sample_count"]),
                },
            )
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        return list(rows_by_key.values())

    def _export_aux_train_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (domain_idx, stream_idx), stats in sorted(self._aux_train_logloss_stats.items()):
            mean, std = _stats_mean_std(stats, "sample_count")
            rows.append({
                "stat_group": "aux_head_train_logloss",
                "domain": int(domain_idx),
                "source_domain": int(domain_idx),
                "stream": int(stream_idx),
                "expert_index": int(stream_idx),
                "architecture": self.architecture,
                "readout_mode": self.readout_mode,
                "expert_aux_weight": self.expert_aux_weight,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "logloss_mean": mean,
                "logloss_std": std,
                "logloss_min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "logloss_max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        if not rows:
            rows.append({
                "stat_group": "aux_head_train_logloss",
                "domain": -1,
                "source_domain": -1,
                "stream": -1,
                "expert_index": -1,
                "architecture": self.architecture,
                "readout_mode": self.readout_mode,
                "expert_aux_weight": self.expert_aux_weight,
                "sample_count": 0,
                "status": "disabled_or_not_observed",
            })
        return rows
