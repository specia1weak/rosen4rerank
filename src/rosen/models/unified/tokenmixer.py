from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from torch import nn

from rosen.emb import SchemaManager
from rosen.emb.emblayer import EmbView
from rosen.emb.schema import SequenceSetting
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.utils.common import to_dims
from rosen.models.utils.sequence import AttentionSequencePoolingLayer
from rosen.utils.time_bucket import RelativeTimeEmbedding, TIME_BUCKET_BOUNDARIES
from rosen.models.unified.rankmixer import MultiHeadTokenMixing, _make_norm


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


def reset_per_token_linear_weight_(
    weight: torch.Tensor,
    fan_in: int,
    *,
    scale: float = 1.0,
) -> None:
    """Initialize packed per-token matrices like independent ``nn.Linear`` weights.

    ``torch.nn.init`` interprets dimensions beyond the first two as convolutional
    receptive fields.  Applying Xavier directly to ``[tokens, in, out]`` therefore
    makes the token count part of fan calculation and initializes these matrices
    much smaller than their independent ``[in, out]`` semantics require.
    """

    fan_in = int(fan_in)
    if fan_in <= 0:
        raise ValueError("fan_in must be positive.")
    if weight.ndim != 3 or int(weight.shape[-2]) != fan_in:
        raise ValueError(
            "Expected a packed per-token weight with shape [tokens, fan_in, fan_out], "
            f"got {tuple(weight.shape)} for fan_in={fan_in}."
        )
    scale = float(scale)
    if scale < 0.0:
        raise ValueError("scale must be non-negative.")
    bound = scale / math.sqrt(fan_in)
    nn.init.uniform_(weight, -bound, bound)


class PerTokenSwiGLU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        expand: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(expand) * self.input_dim
        self.w1 = nn.Parameter(torch.empty(self.num_tokens, self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_tokens, self.input_dim, self.hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_tokens, self.hidden_dim, self.input_dim))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        reset_per_token_linear_weight_(self.w1, self.input_dim)
        reset_per_token_linear_weight_(self.w2, self.input_dim)
        # Keep the two SwiGLU branches healthy while starting the residual
        # write-back conservatively.  Six of these FFNs are stacked in the
        # three-layer MixFormer used by the main experiment.
        reset_per_token_linear_weight_(self.w3, self.hidden_dim, scale=0.1)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.einsum("btd,tdk->btk", x, self.w1) + self.b1
        h1 = self.dropout(self.act(h1))
        h2 = torch.einsum("btd,tdk->btk", x, self.w2) + self.b2
        return torch.einsum("btk,tkd->btd", h1 * h2, self.w3)


class TokenMixerBlocks(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        num_layers: int,
        expand: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.mixer_norms = nn.ModuleList([_make_norm(input_dim) for _ in range(self.num_layers)])
        self.ffn_norms = nn.ModuleList([_make_norm(input_dim) for _ in range(self.num_layers)])
        self.mixer_layers = nn.ModuleList(
            [MultiHeadTokenMixing(input_dim=input_dim, num_tokens=num_tokens) for _ in range(self.num_layers)]
        )
        self.revert_layers = nn.ModuleList(
            [MultiHeadTokenMixing(input_dim=input_dim, num_tokens=num_tokens) for _ in range(self.num_layers)]
        )
        self.ffn_layers1 = nn.ModuleList(
            [PerTokenSwiGLU(input_dim, num_tokens, expand=expand, dropout_rate=dropout_rate) for _ in range(self.num_layers)]
        )
        self.ffn_layers2 = nn.ModuleList(
            [PerTokenSwiGLU(input_dim, num_tokens, expand=expand, dropout_rate=dropout_rate) for _ in range(self.num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mixer_norm, mixer, revert, ffn1, ffn_norm, ffn2 in zip(
            self.mixer_norms,
            self.mixer_layers,
            self.revert_layers,
            self.ffn_layers1,
            self.ffn_norms,
            self.ffn_layers2,
        ):
            mixed = mixer(mixer_norm(x))
            mixed = mixed + ffn1(mixed)
            x = x + revert(mixed)
            x = x + ffn2(ffn_norm(x))
        return x


class TokenMixerModel(MSRModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 144,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_global_token: int = 1,
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
        self.current_fields = tuple(field_name for field_name, _, _ in self.current_view.field_segments)
        self.current_slices = self.current_view.field_slices
        self.sequence_embedding_dim = self.history_view.embedding_dim
        self.token_dim = int(token_dim)
        self.num_global_token = int(num_global_token)
        self.num_tokens = self.num_global_token + len(self.current_fields) + 2
        if self.token_dim % self.num_tokens != 0:
            raise ValueError(
                f"TokenMixer requires token_dim % num_tokens == 0, got token_dim={self.token_dim}, "
                f"num_tokens={self.num_tokens}. Choose a divisible token_dim."
            )

        self.use_relative_time = bool(use_relative_time)
        self.time_field = manager.time_field
        if self.use_relative_time:
            if not self.time_field:
                raise ValueError("use_relative_time=True requires manager.time_field to be set.")
            if not self.history_view.time_field:
                raise ValueError(
                    f"use_relative_time=True requires seq group '{seq_group_name}' to define time_field_name."
                )
            self.relative_time_to_seconds_divisor = _resolve_timestamp_to_seconds_divisor(
                relative_time_unit,
                self.time_field,
            )
            self.relative_time_embedding = RelativeTimeEmbedding(
                embedding_dim=self.sequence_embedding_dim,
                boundaries=tuple(relative_time_boundaries or TIME_BUCKET_BOUNDARIES),
                init_std=float(relative_time_init_std),
            )
        else:
            self.relative_time_to_seconds_divisor = 1
            self.relative_time_embedding = None

        self.field_projs = nn.ModuleDict()
        for field_name, start, stop in self.current_view.field_segments:
            dim = int(stop - start)
            self.field_projs[field_name] = (
                nn.Linear(dim, self.token_dim)
                if dim != self.token_dim
                else nn.Identity()
            )
        self.target_proj = (
            nn.Linear(self.sequence_embedding_dim, self.token_dim)
            if self.sequence_embedding_dim != self.token_dim
            else nn.Identity()
        )
        self.pool_proj = (
            nn.Linear(self.sequence_embedding_dim, self.token_dim)
            if self.sequence_embedding_dim != self.token_dim
            else nn.Identity()
        )
        self.global_mlp = nn.Sequential(
            nn.Linear((len(self.current_fields) + 2) * self.token_dim, self.num_global_token * self.token_dim * 2),
            nn.ReLU(),
            nn.Linear(self.num_global_token * self.token_dim * 2, self.num_global_token * self.token_dim),
        )
        self.att_pool = AttentionSequencePoolingLayer(
            embedding_dim=self.sequence_embedding_dim,
            att_hidden_units=tuple(int(v) for v in att_hidden_units),
            att_activation="sigmoid",
        )
        self.tokenmixer = TokenMixerBlocks(
            input_dim=self.token_dim,
            num_tokens=self.num_tokens,
            num_layers=num_layers,
            expand=expansion_factor,
            dropout_rate=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.token_dim,
            hidden_dims=to_dims(tower_hidden_dims, default=(128, 64)),
            dropout_rate=tower_dropout_rate,
            activation="relu",
            batch_norm=tower_batch_norm,
        )

        domain_setting = manager.get_setting(self.DOMAIN)
        self.domain_shift = 1 if getattr(domain_setting, "padding_zero", False) else 0

    def _current_tokens(self, interaction) -> torch.Tensor:
        current_x = self.current_view(interaction)
        tokens = []
        for field_name in self.current_fields:
            tokens.append(self.field_projs[field_name](current_x[:, self.current_slices[field_name]]))
        return torch.stack(tokens, dim=1)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        current_tokens = self._current_tokens(interaction)
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        seq_len = seq_len.view(-1).long().clamp(min=0, max=seq_emb.size(1))
        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_emb = seq_emb + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        history_x = self.att_pool(target_emb, seq_emb, seq_len)
        target_token = self.target_proj(target_emb).unsqueeze(1)
        pool_token = self.pool_proj(history_x).unsqueeze(1)
        ns_tokens = torch.cat([current_tokens, target_token, pool_token], dim=1)
        global_tokens = self.global_mlp(torch.flatten(ns_tokens, start_dim=1)).view(
            ns_tokens.size(0),
            self.num_global_token,
            self.token_dim,
        )
        tokens = torch.cat([global_tokens, ns_tokens], dim=1)
        tokens = self.tokenmixer(tokens)
        features = self.dropout(tokens.mean(dim=1))
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        return features, domain_ids

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(x, domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
