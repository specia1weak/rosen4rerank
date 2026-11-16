from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rosen.emb import SchemaManager
from rosen.emb.emblayer import EmbView
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.utils.common import build_mlp, to_dims
from rosen.utils.time_bucket import (
    TIME_BUCKET_BOUNDARIES,
    AdjacentTimeEmbedding,
    RelativeTimeEmbedding,
)


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


class DomainInternalAttentionBranch(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        feedforward_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        if self.token_dim % self.num_heads != 0:
            raise ValueError(f"token_dim={self.token_dim} must be divisible by num_heads={self.num_heads}")
        self.norm1 = nn.LayerNorm(self.token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=self.num_heads,
            dropout=float(dropout_rate),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.token_dim)
        hidden_dim = int(feedforward_dim or self.token_dim * 4)
        if str(activation).lower() == "gelu":
            activation_layer: nn.Module = nn.GELU()
        else:
            activation_layer = nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            activation_layer,
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(
        self,
        tokens: torch.Tensor,
        domain_ids: torch.Tensor,
        domain_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B, L, D], got {tuple(tokens.shape)}")
        batch_size, seq_len, _ = tokens.shape
        if seq_len <= 0:
            return tokens
        domain_mask = domain_mask.bool()
        tokens = tokens * domain_mask.unsqueeze(-1).to(dtype=tokens.dtype)

        same_domain = domain_ids.unsqueeze(2) == domain_ids.unsqueeze(1)
        attn_mask = ~same_domain
        attn_mask = torch.where(domain_mask.unsqueeze(2), attn_mask, torch.zeros_like(attn_mask))
        empty_rows = ~domain_mask.any(dim=1)
        key_padding_mask = ~domain_mask
        if bool(empty_rows.any().detach().cpu().item()):
            key_padding_mask = key_padding_mask.clone()
            attn_mask = attn_mask.clone()
            key_padding_mask[empty_rows] = False
            attn_mask[empty_rows] = False
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        normed = self.norm1(tokens)
        attended, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(attended)
        tokens = tokens * domain_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        return tokens * domain_mask.unsqueeze(-1).to(dtype=tokens.dtype)


class DomainTokenQueryMixer(nn.Module):
    """Small token-axis mixer for 5 domain queries.

    Unlike MixFormer head-mixing, this does not require token_dim to be
    divisible by the number of domain tokens. It only produces a mixed query;
    history values are still read through the domain mask.
    """

    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        expand: int = 2,
        dropout_rate: float = 0.0,
        init_std: float = 1e-3,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        hidden_tokens = max(self.num_tokens, int(expand) * self.num_tokens)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.num_tokens, hidden_tokens),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_tokens, self.num_tokens),
        )
        self.ffn_norm = nn.LayerNorm(self.token_dim)
        hidden_dim = max(self.token_dim, int(expand) * self.token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self._small_last_linear(self.token_mlp, std=init_std)
        self._small_last_linear(self.ffn, std=init_std)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.token_norm(x)
        mixed = self.token_mlp(normed.transpose(1, 2)).transpose(1, 2)
        x = x + mixed
        ffn_update = self.ffn(self.ffn_norm(x))
        x = x + ffn_update
        return x


class SASRecRankerModel(MSRModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        max_len: Optional[int] = None,
        token_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: Optional[int] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        position_init_std: float = 1e-3,
        position_mode: str = "pre",
        use_causal_mask: bool = True,
        bottom_hidden_dims: Iterable[int] = (256, 128, 64),
        bottom_dropout_rate: float = 0.0,
        bottom_batch_norm: bool = False,
        seq_delta_init_std: float = 1e-3,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        include_target_fields_in_current: bool = True,
        norm_first: bool = True,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        use_adjacent_time: bool = False,
        adjacent_time_boundaries: Optional[Iterable[int]] = None,
        adjacent_time_init_std: float = 1e-4,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = manager.domain_field
        self.LABEL = manager.label_field
        if seq_group_name not in self.omni_embedding.seq_groups:
            raise ValueError(f"Unknown sequence group: {seq_group_name}")

        self.history_view = self.omni_embedding.seq_groups[seq_group_name]
        self.max_len = int(max_len) if max_len is not None else _infer_group_max_len(self.history_view)
        self.token_dim = int(token_dim)
        self.use_causal_mask = bool(use_causal_mask)
        self.position_mode = str(position_mode).lower()
        if self.position_mode not in {"pre", "post"}:
            raise ValueError("position_mode must be 'pre' or 'post'.")
        if self.token_dim % int(num_heads) != 0:
            raise ValueError(f"token_dim={self.token_dim} must be divisible by num_heads={num_heads}")

        target_field_names = {setting.field_name for setting in self.history_view.target_settings}
        label_field_names = set(manager.label_fields)
        excluded_current_fields = set(exclude_current_fields or ())
        current_fields = [
            setting.field_name
            for setting in manager.settings
            if not setting.is_sequence_setting
            and setting.field_name not in label_field_names
            and setting.field_name not in excluded_current_fields
            and (
                include_target_fields_in_current
                or setting.field_name not in target_field_names
                or setting.field_name == self.DOMAIN
            )
        ]
        self.current_view = (
            EmbView(self.omni_embedding, include_fields=current_fields)
            if current_fields
            else None
        )
        self.current_embedding_dim = (
            self.current_view.embedding_dim
            if self.current_view is not None
            else 0
        )
        if self.current_embedding_dim <= 0:
            raise ValueError("SASRecRankerModel residual fusion requires current fields.")
        self.sequence_embedding_dim = self.history_view.embedding_dim
        self.use_relative_time = bool(use_relative_time)
        self.time_field = manager.time_field
        if self.use_relative_time:
            if not self.time_field:
                raise ValueError("use_relative_time=True requires manager.time_field to be set.")
            if not self.history_view.time_field:
                raise ValueError(
                    f"use_relative_time=True requires seq group '{seq_group_name}' to define time_field_name."
                )
            second_boundaries = tuple(relative_time_boundaries or TIME_BUCKET_BOUNDARIES)
            self.relative_time_to_seconds_divisor = _resolve_timestamp_to_seconds_divisor(
                relative_time_unit,
                self.time_field,
            )
            self.relative_time_embedding = RelativeTimeEmbedding(
                embedding_dim=self.token_dim,
                boundaries=second_boundaries,
                init_std=float(relative_time_init_std),
            )
        else:
            self.relative_time_to_seconds_divisor = 1
            self.relative_time_embedding = None
        self.use_adjacent_time = bool(use_adjacent_time)
        if self.use_adjacent_time:
            if not self.time_field:
                raise ValueError("use_adjacent_time=True requires manager.time_field to be set.")
            if not self.history_view.time_field:
                raise ValueError(
                    f"use_adjacent_time=True requires seq group '{seq_group_name}' "
                    "to define time_field_name."
                )
            self.adjacent_time_to_seconds_divisor = (
                self.relative_time_to_seconds_divisor
                if self.use_relative_time
                else _resolve_timestamp_to_seconds_divisor(
                    relative_time_unit,
                    self.time_field,
                )
            )
            self.adjacent_time_embedding = AdjacentTimeEmbedding(
                embedding_dim=self.token_dim,
                boundaries=tuple(adjacent_time_boundaries or TIME_BUCKET_BOUNDARIES),
                init_std=float(adjacent_time_init_std),
            )
        else:
            self.adjacent_time_to_seconds_divisor = 1
            self.adjacent_time_embedding = None

        self.item_token_proj = (
            nn.Linear(self.sequence_embedding_dim, self.token_dim)
            if self.sequence_embedding_dim != self.token_dim
            else nn.Identity()
        )
        self.target_token_proj = (
            nn.Linear(self.sequence_embedding_dim, self.token_dim)
            if self.sequence_embedding_dim != self.token_dim
            else nn.Identity()
        )
        self.start_token = nn.Parameter(torch.empty(1, 1, self.token_dim))
        nn.init.normal_(self.start_token, mean=0.0, std=float(position_init_std))
        self.position_embedding = nn.Embedding(self.max_len + 2, self.token_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=float(position_init_std))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim or self.token_dim * 4),
            dropout=float(dropout_rate),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.input_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.output_norm = nn.LayerNorm(self.token_dim)

        causal_mask = torch.triu(
            torch.ones(self.max_len + 2, self.max_len + 2, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        resolved_bottom_dims = to_dims(bottom_hidden_dims, default=(256, 128, 64))
        self.bottom_output_dim = int(resolved_bottom_dims[-1])
        self.current_bottom = build_mlp(
            self.current_embedding_dim,
            resolved_bottom_dims,
            dropout_rate=bottom_dropout_rate,
            activation=activation,
            batch_norm=bottom_batch_norm,
        )
        self.seq_delta = nn.Linear(self.token_dim, self.bottom_output_dim)
        nn.init.normal_(self.seq_delta.weight, mean=0.0, std=float(seq_delta_init_std))
        nn.init.zeros_(self.seq_delta.bias)
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.bottom_output_dim,
            hidden_dims=to_dims(tower_hidden_dims, default=(128,)),
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=tower_batch_norm,
        )

        domain_setting = manager.get_setting(self.DOMAIN)
        self.domain_shift = 1 if getattr(domain_setting, "padding_zero", False) else 0


    def _current_features(self, interaction, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.current_view is None or self.current_embedding_dim == 0:
            return torch.empty(batch_size, 0, device=device)
        return torch.flatten(self.current_view(interaction), start_dim=1)

    def _position_ids(self, hist_mask: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        batch_size, max_len = hist_mask.shape
        device = hist_mask.device
        start_pos = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        history_offset = 1
        if self.position_mode == "pre":
            history_offset = (max_len - seq_len).clamp(min=0).unsqueeze(1) + 1
        history_pos = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1) + history_offset
        history_pos = history_pos.masked_fill(~hist_mask, 0)
        if self.position_mode == "pre":
            target_pos = torch.full((batch_size, 1), max_len + 1, dtype=torch.long, device=device)
        else:
            target_pos = seq_len.clamp(min=0, max=max_len).unsqueeze(1) + 1
        return torch.cat([start_pos, history_pos, target_pos], dim=1)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_relative_time or self.use_adjacent_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        batch_size, max_len, _ = seq_emb.shape
        if max_len > self.max_len:
            raise ValueError(f"Input sequence length {max_len} exceeds model max_len {self.max_len}.")

        device = seq_emb.device
        if seq_len is None:
            seq_len = torch.full((batch_size,), max_len, dtype=torch.long, device=device)
        else:
            seq_len = seq_len.view(-1).long().clamp(min=0, max=max_len)

        hist_mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        seq_tokens = self.item_token_proj(seq_emb)
        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            relative_hist_ts = _to_seconds(
                hist_ts,
                self.relative_time_to_seconds_divisor,
            )
            seq_tokens = seq_tokens + self.relative_time_embedding(
                curr_ts,
                relative_hist_ts,
                seq_lens=seq_len,
            )
        if self.adjacent_time_embedding is not None:
            adjacent_hist_ts = _to_seconds(
                hist_ts,
                self.adjacent_time_to_seconds_divisor,
            )
            seq_tokens = seq_tokens + self.adjacent_time_embedding(
                adjacent_hist_ts,
                seq_lens=seq_len,
            )
        seq_tokens = seq_tokens * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        target_token = self.target_token_proj(target_emb).unsqueeze(1)
        start_token = self.start_token.expand(batch_size, -1, -1)
        tokens = torch.cat([start_token, seq_tokens, target_token], dim=1)

        start_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        target_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        valid_mask = torch.cat([start_mask, hist_mask, target_mask], dim=1)
        position_ids = self._position_ids(hist_mask, seq_len)
        tokens = self.input_dropout(tokens + self.position_embedding(position_ids))

        attn_mask = None
        if self.use_causal_mask:
            total_len = max_len + 2
            attn_mask = self.causal_mask[:total_len, :total_len]
        encoded = self.encoder(
            tokens,
            mask=attn_mask,
            src_key_padding_mask=~valid_mask,
        )
        last_token = self.output_norm(encoded[:, -1, :])

        current_x = self._current_features(interaction, batch_size, device)
        features = self.current_bottom(current_x) + self.seq_delta(last_token)
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


class DomainJumpSASRecRankerModel(SASRecRankerModel):
    """SASRec with domain transition encoding on sequence and target tokens.

    A transition x -> y is encoded as leave[x] + enter[y]. Tokens without a
    known domain switch share one domain-stay token.
    """

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        max_len: Optional[int] = None,
        token_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: Optional[int] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        position_init_std: float = 1e-3,
        position_mode: str = "pre",
        use_causal_mask: bool = True,
        bottom_hidden_dims: Iterable[int] = (256, 128, 64),
        bottom_dropout_rate: float = 0.0,
        bottom_batch_norm: bool = False,
        seq_delta_init_std: float = 1e-3,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        include_target_fields_in_current: bool = True,
        norm_first: bool = True,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        domain_seq_field: str = "domain_tab_seq",
        domain_jump_init_std: float = 1e-4,
        domain_jump_apply_target: bool = True,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            max_len=max_len,
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            position_init_std=position_init_std,
            position_mode=position_mode,
            use_causal_mask=use_causal_mask,
            bottom_hidden_dims=bottom_hidden_dims,
            bottom_dropout_rate=bottom_dropout_rate,
            bottom_batch_norm=bottom_batch_norm,
            seq_delta_init_std=seq_delta_init_std,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            include_target_fields_in_current=include_target_fields_in_current,
            norm_first=norm_first,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.domain_seq_field = str(domain_seq_field)
        self.domain_jump_init_std = float(domain_jump_init_std)
        self.domain_jump_apply_target = bool(domain_jump_apply_target)
        self.domain_leave_embedding = nn.Embedding(int(num_domains), self.token_dim)
        self.domain_enter_embedding = nn.Embedding(int(num_domains), self.token_dim)
        self.domain_stay_token = nn.Parameter(torch.empty(self.token_dim))
        nn.init.normal_(self.domain_leave_embedding.weight, mean=0.0, std=self.domain_jump_init_std)
        nn.init.normal_(self.domain_enter_embedding.weight, mean=0.0, std=self.domain_jump_init_std)
        nn.init.normal_(self.domain_stay_token, mean=0.0, std=self.domain_jump_init_std)
        self.feature_output_dim = self.bottom_output_dim

    def _history_domain_ids(
        self,
        interaction,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_field = self.domain_seq_field if self.domain_seq_field in interaction else "tab_seq"
        if source_field not in interaction:
            empty_ids = torch.zeros_like(hist_mask, dtype=torch.long)
            return empty_ids, torch.zeros_like(hist_mask, dtype=torch.bool)
        raw_domain = interaction[source_field].long()
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

    def _domain_jump_delta(
        self,
        hist_domain_ids: torch.Tensor,
        domain_known_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_domain_ids = torch.zeros_like(hist_domain_ids)
        prev_known_mask = torch.zeros_like(domain_known_mask)
        if hist_domain_ids.shape[1] > 1:
            prev_domain_ids[:, 1:] = hist_domain_ids[:, :-1]
            prev_known_mask[:, 1:] = domain_known_mask[:, :-1]

        switch_mask = domain_known_mask & prev_known_mask & (hist_domain_ids != prev_domain_ids)
        leave_delta = self.domain_leave_embedding(prev_domain_ids.long())
        enter_delta = self.domain_enter_embedding(hist_domain_ids.long())
        jump_delta = leave_delta + enter_delta
        stay_delta = self.domain_stay_token.view(1, 1, -1).expand_as(jump_delta)
        seq_delta = torch.where(switch_mask.unsqueeze(-1), jump_delta, stay_delta)
        seq_delta = seq_delta * domain_known_mask.unsqueeze(-1).to(dtype=seq_delta.dtype)

        batch_size, max_len = hist_domain_ids.shape
        idx = torch.arange(max_len, device=hist_domain_ids.device).unsqueeze(0).expand(batch_size, -1)
        last_idx = torch.where(domain_known_mask, idx, torch.full_like(idx, -1)).max(dim=1).values
        has_last = last_idx >= 0
        last_domain_ids = hist_domain_ids.gather(1, last_idx.clamp(min=0).unsqueeze(1)).squeeze(1)
        safe_target = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        target_switch = has_last & (last_domain_ids != safe_target)
        target_jump_delta = self.domain_leave_embedding(last_domain_ids.long()) + self.domain_enter_embedding(safe_target)
        target_stay_delta = self.domain_stay_token.view(1, -1).expand(batch_size, -1)
        target_delta = torch.where(target_switch.unsqueeze(-1), target_jump_delta, target_stay_delta)
        if not self.domain_jump_apply_target:
            target_delta = torch.zeros_like(target_delta)

        return seq_delta, target_delta

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        batch_size, max_len, _ = seq_emb.shape
        if max_len > self.max_len:
            raise ValueError(f"Input sequence length {max_len} exceeds model max_len {self.max_len}.")

        device = seq_emb.device
        if seq_len is None:
            seq_len = torch.full((batch_size,), max_len, dtype=torch.long, device=device)
        else:
            seq_len = seq_len.view(-1).long().clamp(min=0, max=max_len)

        hist_mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        hist_domain_ids, domain_known_mask = self._history_domain_ids(interaction, hist_mask)
        seq_tokens = self.item_token_proj(seq_emb)
        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_tokens = seq_tokens + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        jump_delta, target_jump_delta = self._domain_jump_delta(hist_domain_ids, domain_known_mask, domain_ids)
        seq_tokens = seq_tokens + jump_delta
        seq_tokens = seq_tokens * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)
        target_token = self.target_token_proj(target_emb).unsqueeze(1)
        target_token = target_token + target_jump_delta.unsqueeze(1)
        start_token = self.start_token.expand(batch_size, -1, -1)
        tokens = torch.cat([start_token, seq_tokens, target_token], dim=1)

        start_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        target_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        valid_mask = torch.cat([start_mask, hist_mask, target_mask], dim=1)
        position_ids = self._position_ids(hist_mask, seq_len)
        tokens = self.input_dropout(tokens + self.position_embedding(position_ids))

        attn_mask = None
        if self.use_causal_mask:
            total_len = max_len + 2
            attn_mask = self.causal_mask[:total_len, :total_len]

        encoded = self.encoder(
            tokens,
            mask=attn_mask,
            src_key_padding_mask=~valid_mask,
        )
        encoded = self.output_norm(encoded)
        history_encoded = encoded[:, 1 : max_len + 1, :]
        history_encoded = history_encoded * hist_mask.unsqueeze(-1).to(dtype=history_encoded.dtype)
        last_token = encoded[:, -1, :]

        current_x = self._current_features(interaction, batch_size, device)
        features = self.current_bottom(current_x) + self.seq_delta(last_token)
        return features, domain_ids


class DomainPairBranchSASRecRankerModel(SASRecRankerModel):
    """SASRec plus a small cross-domain encoding and a post-SASRec domain branch.

    The main SASRec path is still present. The branch first lets history tokens
    attend only to tokens from the same history domain, then mean-pools outputs
    by domain. Missing domains are represented by exact zeros.
    """

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        max_len: Optional[int] = None,
        token_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: Optional[int] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        position_init_std: float = 1e-3,
        position_mode: str = "pre",
        use_causal_mask: bool = True,
        bottom_hidden_dims: Iterable[int] = (256, 128, 64),
        bottom_dropout_rate: float = 0.0,
        bottom_batch_norm: bool = False,
        seq_delta_init_std: float = 1e-3,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
        exclude_current_fields: Iterable[str] = ("action",),
        include_target_fields_in_current: bool = True,
        norm_first: bool = True,
        use_relative_time: bool = False,
        relative_time_unit: str = "auto",
        relative_time_boundaries: Optional[Iterable[int]] = None,
        relative_time_init_std: float = 1e-4,
        domain_seq_field: str = "domain_tab_seq",
        domain_pair_init_std: float = 1e-4,
        domain_branch_mode: str = "concat",
        domain_branch_num_heads: Optional[int] = None,
        domain_branch_logit_scale: float = 1.0,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            seq_group_name=seq_group_name,
            max_len=max_len,
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            position_init_std=position_init_std,
            position_mode=position_mode,
            use_causal_mask=use_causal_mask,
            bottom_hidden_dims=bottom_hidden_dims,
            bottom_dropout_rate=bottom_dropout_rate,
            bottom_batch_norm=bottom_batch_norm,
            seq_delta_init_std=seq_delta_init_std,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            tower_batch_norm=tower_batch_norm,
            exclude_current_fields=exclude_current_fields,
            include_target_fields_in_current=include_target_fields_in_current,
            norm_first=norm_first,
            use_relative_time=use_relative_time,
            relative_time_unit=relative_time_unit,
            relative_time_boundaries=relative_time_boundaries,
            relative_time_init_std=relative_time_init_std,
        )
        self.domain_seq_field = str(domain_seq_field)
        self.domain_pair_init_std = float(domain_pair_init_std)
        self.domain_branch_mode = str(domain_branch_mode or "concat").lower()
        allowed_modes = {"pair_only", "concat", "logit_add", "concat_logit_add"}
        if self.domain_branch_mode not in allowed_modes:
            raise ValueError(f"domain_branch_mode must be one of {sorted(allowed_modes)}, got {domain_branch_mode!r}.")
        self.domain_branch_logit_scale = float(domain_branch_logit_scale)
        self.domain_pair_embedding = nn.Embedding(int(num_domains) * int(num_domains), self.token_dim)
        nn.init.normal_(self.domain_pair_embedding.weight, mean=0.0, std=self.domain_pair_init_std)
        branch_heads = int(domain_branch_num_heads or num_heads)
        self.use_domain_branch = self.domain_branch_mode != "pair_only"
        self.domain_branch = (
            DomainInternalAttentionBranch(
                token_dim=self.token_dim,
                num_heads=branch_heads,
                feedforward_dim=feedforward_dim,
                dropout_rate=dropout_rate,
                activation=activation,
            )
            if self.use_domain_branch
            else None
        )
        self.domain_branch_flat_dim = int(num_domains) * self.token_dim
        self.use_domain_branch_concat = self.domain_branch_mode in {"concat", "concat_logit_add"}
        self.use_domain_branch_logit = self.domain_branch_mode in {"logit_add", "concat_logit_add"}
        self.feature_output_dim = self.bottom_output_dim + (
            self.domain_branch_flat_dim
            if self.use_domain_branch_concat
            else 0
        )
        if self.use_domain_branch_concat:
            self.head = DomainTowerHead(
                num_domains=num_domains,
                input_dim=self.feature_output_dim,
                hidden_dims=to_dims(tower_hidden_dims, default=(128,)),
                dropout_rate=tower_dropout_rate,
                activation=activation,
                batch_norm=tower_batch_norm,
            )
        if self.use_domain_branch_logit:
            logit_hidden_dims = to_dims(tower_hidden_dims, default=(128, 64))
            self.domain_branch_logit_mlp = build_mlp(
                self.domain_branch_flat_dim,
                (*logit_hidden_dims, 1),
                dropout_rate=tower_dropout_rate,
                activation=activation,
                batch_norm=tower_batch_norm,
            )
            last_linear = None
            for module in self.domain_branch_logit_mlp.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)
        else:
            self.domain_branch_logit_mlp = None
        self._last_domain_branch_flat: Optional[torch.Tensor] = None

    def _history_domain_ids(
        self,
        interaction,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_field = self.domain_seq_field if self.domain_seq_field in interaction else "tab_seq"
        if source_field not in interaction:
            empty_ids = torch.zeros_like(hist_mask, dtype=torch.long)
            return empty_ids, torch.zeros_like(hist_mask, dtype=torch.bool)
        raw_domain = interaction[source_field].long()
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

    def _domain_pair_delta(
        self,
        hist_domain_ids: torch.Tensor,
        domain_known_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        safe_target = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        pair_ids = hist_domain_ids.long() * int(self.num_domains) + safe_target.unsqueeze(1)
        pair_delta = self.domain_pair_embedding(pair_ids)
        return pair_delta * domain_known_mask.unsqueeze(-1).to(dtype=pair_delta.dtype)

    def _pool_domain_tokens(
        self,
        tokens: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        domain_known_mask: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = F.one_hot(
            hist_domain_ids.long().clamp(min=0, max=int(self.num_domains) - 1),
            num_classes=int(self.num_domains),
        ).to(dtype=tokens.dtype)
        one_hot = one_hot * domain_known_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        counts = one_hot.sum(dim=1)
        pooled = torch.einsum("bld,blk->bkd", tokens, one_hot)
        pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
        return pooled * (counts > 0).unsqueeze(-1).to(dtype=tokens.dtype)

    def encode_parts(self, interaction) -> dict[str, torch.Tensor]:
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        batch_size, max_len, _ = seq_emb.shape
        if max_len > self.max_len:
            raise ValueError(f"Input sequence length {max_len} exceeds model max_len {self.max_len}.")

        device = seq_emb.device
        if seq_len is None:
            seq_len = torch.full((batch_size,), max_len, dtype=torch.long, device=device)
        else:
            seq_len = seq_len.view(-1).long().clamp(min=0, max=max_len)
        hist_mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        hist_domain_ids, domain_known_mask = self._history_domain_ids(interaction, hist_mask)

        seq_tokens = self.item_token_proj(seq_emb)
        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_tokens = seq_tokens + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        pair_delta = self._domain_pair_delta(hist_domain_ids, domain_known_mask, domain_ids)
        seq_tokens = seq_tokens + pair_delta
        seq_tokens = seq_tokens * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        target_token = self.target_token_proj(target_emb).unsqueeze(1)
        start_token = self.start_token.expand(batch_size, -1, -1)
        tokens = torch.cat([start_token, seq_tokens, target_token], dim=1)

        start_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        target_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        valid_mask = torch.cat([start_mask, hist_mask, target_mask], dim=1)
        position_ids = self._position_ids(hist_mask, seq_len)
        tokens = self.input_dropout(tokens + self.position_embedding(position_ids))

        attn_mask = None
        if self.use_causal_mask:
            total_len = max_len + 2
            attn_mask = self.causal_mask[:total_len, :total_len]

        encoded = self.encoder(
            tokens,
            mask=attn_mask,
            src_key_padding_mask=~valid_mask,
        )
        encoded = self.output_norm(encoded)
        history_encoded = encoded[:, 1 : max_len + 1, :]
        history_encoded = history_encoded * hist_mask.unsqueeze(-1).to(dtype=history_encoded.dtype)
        target_encoded = encoded[:, -1, :]

        if self.domain_branch is None:
            domain_branch_tokens = torch.zeros_like(history_encoded)
        else:
            domain_branch_tokens = self.domain_branch(history_encoded, hist_domain_ids, domain_known_mask)
        domain_branch_pooled = self._pool_domain_tokens(domain_branch_tokens, hist_domain_ids, domain_known_mask)
        domain_branch_flat = torch.flatten(domain_branch_pooled, start_dim=1)

        current_x = self._current_features(interaction, batch_size, device)
        base_features = self.current_bottom(current_x) + self.seq_delta(target_encoded)
        if self.use_domain_branch_concat:
            features = torch.cat([base_features, domain_branch_flat], dim=-1)
        else:
            features = base_features
        self._last_domain_branch_flat = domain_branch_flat
        return {
            "features": features,
            "domain_ids": domain_ids,
            "domain_branch_flat": domain_branch_flat,
            "domain_branch_pooled": domain_branch_pooled,
            "domain_branch_tokens": domain_branch_tokens,
            "history_encoded": history_encoded,
            "target_encoded": target_encoded,
            "domain_known_mask": domain_known_mask,
            "hist_domain_ids": hist_domain_ids,
        }

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.encode_parts(interaction)
        return parts["features"], parts["domain_ids"]

    def domain_branch_logits(self, domain_branch_flat: torch.Tensor) -> torch.Tensor:
        if self.domain_branch_logit_mlp is None:
            return torch.zeros(domain_branch_flat.shape[0], device=domain_branch_flat.device, dtype=domain_branch_flat.dtype)
        logits = self.domain_branch_logit_mlp(domain_branch_flat).squeeze(-1)
        return float(self.domain_branch_logit_scale) * logits

    def predict_logits_with_domains(self, interaction) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = self.encode_parts(interaction)
        logits = self.forward(parts["features"], parts["domain_ids"])
        if self.use_domain_branch_logit:
            branch_logits = self.domain_branch_logits(parts["domain_branch_flat"])
            logits = logits + branch_logits
        return logits, parts["domain_ids"], parts["features"]

    def predict(self, interaction):
        logits, _, _ = self.predict_logits_with_domains(interaction)
        return logits


class DVCSeqStage2SASRecRankerModel(DomainPairBranchSASRecRankerModel):
    """Stage 2 DVCSeq: shared SASRec token plus all-domain cross views.

    This keeps the Stage 1 pair encoding, but replaces the static readout with
    sample-dependent cross-domain views. All cross views participate in every
    forward pass through a soft aggregate, so their reader/gate parameters are
    trained by every target domain instead of only their own domain.
    """

    def __init__(
        self,
        *args,
        dvc_view_hidden_dim: Optional[int] = None,
        dvc_anchor_residual_scale: float = 0.0,
        dvc_orthogonal_anchor: bool = False,
        dvc_per_view_logits: bool = False,
        dvc_view_logit_scale: float = 1.0,
        dvc_private_enabled: bool = False,
        dvc_private_guided_gate: bool = False,
        dvc_cross_aux_weight: float = 0.0,
        dvc_private_aux_weight: float = 0.0,
        dvc_diversity_weight: float = 0.0,
        dvc_query_anchor_scale: float = 0.0,
        dvc_cross_domain_mask: bool = False,
        dvc_transition_enabled: bool = False,
        dvc_transition_init_std: float = 1e-4,
        dvc_layerwise_domain_tokens: bool = False,
        dvc_layerwise_update_scale: float = 0.1,
        dvc_querymix_enabled: bool = False,
        dvc_querymix_mode: str = "full",
        dvc_querymix_window_chunks: int = 2,
        dvc_querymix_window_global_last: bool = True,
        dvc_support_aware_gate: bool = False,
        dvc_gate_allow_target_fallback: bool = True,
        seq_delta_init_std: float = 1e-3,
        **kwargs,
    ):
        querymix_dropout_rate = float(kwargs.get("dropout_rate", 0.0))
        kwargs["domain_branch_mode"] = "pair_only"
        super().__init__(*args, seq_delta_init_std=seq_delta_init_std, **kwargs)
        hidden_dim = int(dvc_view_hidden_dim or self.token_dim * 2)
        self.dvc_anchor_residual_scale = float(dvc_anchor_residual_scale)
        self.dvc_orthogonal_anchor = bool(dvc_orthogonal_anchor)
        self.dvc_per_view_logits = bool(dvc_per_view_logits)
        self.dvc_view_logit_scale = float(dvc_view_logit_scale)
        self.dvc_private_guided_gate = bool(dvc_private_guided_gate)
        self.dvc_cross_aux_weight = float(dvc_cross_aux_weight)
        self.dvc_private_aux_weight = float(dvc_private_aux_weight)
        self.dvc_diversity_weight = float(dvc_diversity_weight)
        self.dvc_query_anchor_scale = float(dvc_query_anchor_scale)
        self.dvc_cross_domain_mask = bool(dvc_cross_domain_mask)
        self.dvc_transition_enabled = bool(dvc_transition_enabled)
        self.dvc_transition_init_std = float(dvc_transition_init_std)
        self.dvc_layerwise_domain_tokens = bool(dvc_layerwise_domain_tokens)
        self.dvc_layerwise_update_scale = float(dvc_layerwise_update_scale)
        self.dvc_querymix_enabled = bool(dvc_querymix_enabled)
        self.dvc_querymix_mode = str(dvc_querymix_mode or "full").lower()
        self.dvc_querymix_window_chunks = int(dvc_querymix_window_chunks)
        self.dvc_querymix_window_global_last = bool(dvc_querymix_window_global_last)
        self.dvc_support_aware_gate = bool(dvc_support_aware_gate)
        self.dvc_gate_allow_target_fallback = bool(dvc_gate_allow_target_fallback)
        self.dvc_private_enabled = bool(
            dvc_private_enabled
            or self.dvc_private_guided_gate
            or self.dvc_private_aux_weight > 0
        )
        self._last_dvc_cross_aux_logits: Optional[torch.Tensor] = None
        self._last_dvc_private_aux_logits: Optional[torch.Tensor] = None
        self._last_dvc_diversity_loss: Optional[torch.Tensor] = None
        self.dvc_cross_anchor = nn.Embedding(int(self.num_domains), self.token_dim)
        self.dvc_target_domain_embedding = nn.Embedding(int(self.num_domains), self.token_dim)
        if self.dvc_orthogonal_anchor:
            nn.init.orthogonal_(self.dvc_cross_anchor.weight)
            with torch.no_grad():
                self.dvc_cross_anchor.weight.mul_(float(self.token_dim) ** 0.5)
        else:
            nn.init.normal_(self.dvc_cross_anchor.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dvc_target_domain_embedding.weight, mean=0.0, std=0.02)
        self.dvc_cross_query = nn.Sequential(
            nn.LayerNorm(self.token_dim * 3),
            nn.Linear(self.token_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.dvc_cross_view_mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        if self.dvc_private_enabled:
            self.dvc_private_anchor = nn.Embedding(int(self.num_domains), self.token_dim)
            nn.init.normal_(self.dvc_private_anchor.weight, mean=0.0, std=0.02)
            self.dvc_private_query = nn.Sequential(
                nn.LayerNorm(self.token_dim * 3),
                nn.Linear(self.token_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.token_dim),
            )
            self.dvc_private_view_mlp = nn.Sequential(
                nn.LayerNorm(self.token_dim),
                nn.Linear(self.token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.token_dim),
            )
            self.dvc_private_seq_delta = nn.Linear(self.token_dim, self.bottom_output_dim)
            nn.init.normal_(self.dvc_private_seq_delta.weight, mean=0.0, std=float(seq_delta_init_std))
            nn.init.zeros_(self.dvc_private_seq_delta.bias)
        else:
            self.dvc_private_anchor = None
            self.dvc_private_query = None
            self.dvc_private_view_mlp = None
            self.dvc_private_seq_delta = None
        gate_input_dim = self.token_dim * (5 if self.dvc_private_guided_gate else 3)
        self.dvc_gate_mlp = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dvc_view_logit_mlp = (
            nn.Sequential(
                nn.LayerNorm(self.token_dim * 3),
                nn.Linear(self.token_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.dvc_per_view_logits
            else None
        )
        self.dvc_cross_aux_mlp = (
            nn.Sequential(
                nn.LayerNorm(self.token_dim * 3),
                nn.Linear(self.token_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.dvc_cross_aux_weight > 0
            else None
        )
        self.dvc_private_aux_mlp = (
            nn.Sequential(
                nn.LayerNorm(self.token_dim),
                nn.Linear(self.token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.dvc_private_aux_weight > 0
            else None
        )
        self.dvc_cross_seq_delta = nn.Linear(self.token_dim, self.bottom_output_dim)
        nn.init.normal_(self.dvc_cross_seq_delta.weight, mean=0.0, std=float(seq_delta_init_std))
        nn.init.zeros_(self.dvc_cross_seq_delta.bias)
        if self.dvc_transition_enabled:
            transition_vocab_size = int(self.num_domains) * int(self.num_domains) * int(self.num_domains)
            self.dvc_transition_embedding = nn.Embedding(transition_vocab_size, self.token_dim)
            nn.init.normal_(self.dvc_transition_embedding.weight, mean=0.0, std=self.dvc_transition_init_std)
            self.dvc_transition_view_mlp = nn.Sequential(
                nn.LayerNorm(self.token_dim),
                nn.Linear(self.token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.token_dim),
            )
            self.dvc_transition_seq_delta = nn.Linear(self.token_dim, self.bottom_output_dim, bias=False)
            nn.init.normal_(self.dvc_transition_seq_delta.weight, mean=0.0, std=float(seq_delta_init_std))
        else:
            self.dvc_transition_embedding = None
            self.dvc_transition_view_mlp = None
            self.dvc_transition_seq_delta = None
        if self.dvc_layerwise_domain_tokens:
            self.dvc_layerwise_history_norm = nn.LayerNorm(self.token_dim)
            self.dvc_layerwise_token_norm = nn.LayerNorm(self.token_dim)
            self.dvc_layerwise_token_ffn = nn.Sequential(
                nn.LayerNorm(self.token_dim),
                nn.Linear(self.token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.token_dim),
            )
        else:
            self.dvc_layerwise_history_norm = None
            self.dvc_layerwise_token_norm = None
            self.dvc_layerwise_token_ffn = None
        self.dvc_query_mixers = (
            nn.ModuleList(
                [
                    DomainTokenQueryMixer(
                        token_dim=self.token_dim,
                        num_tokens=int(self.num_domains),
                        expand=2,
                        dropout_rate=querymix_dropout_rate,
                    )
                    for _ in range(len(self.encoder.layers))
                ]
            )
            if self.dvc_layerwise_domain_tokens and self.dvc_querymix_enabled
            else None
        )
        self._zero_last_linear(self.dvc_gate_mlp)
        if self.dvc_view_logit_mlp is not None:
            self._small_last_linear(self.dvc_view_logit_mlp)
        if self.dvc_cross_aux_mlp is not None:
            self._small_last_linear(self.dvc_cross_aux_mlp)
        if self.dvc_private_aux_mlp is not None:
            self._small_last_linear(self.dvc_private_aux_mlp)
        if self.dvc_layerwise_token_ffn is not None:
            self._small_last_linear(self.dvc_layerwise_token_ffn)
        self.feature_output_dim = self.bottom_output_dim

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

    def _query_read_history(
        self,
        queries: torch.Tensor,
        history_encoded: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = float(self.token_dim) ** -0.5
        scores = torch.einsum("bkd,bld->bkl", queries, history_encoded) * scale
        scores = scores.masked_fill(~hist_mask.unsqueeze(1), -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * hist_mask.unsqueeze(1).to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        readout = torch.einsum("bkl,bld->bkd", weights, history_encoded)
        return readout, weights

    def _query_read_own_domain_history(
        self,
        queries: torch.Tensor,
        history_encoded: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        domain_known_mask: torch.Tensor,
        *,
        fallback_to_query: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        domain_ids = torch.arange(int(self.num_domains), device=history_encoded.device)
        domain_mask = (
            hist_mask.unsqueeze(1)
            & domain_known_mask.unsqueeze(1)
            & (hist_domain_ids.unsqueeze(1) == domain_ids.view(1, -1, 1))
        )
        scale = float(self.token_dim) ** -0.5
        scores = torch.einsum("bkd,bld->bkl", queries, history_encoded) * scale
        scores = scores.masked_fill(~domain_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * domain_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        readout = torch.einsum("bkl,bld->bkd", weights, history_encoded)

        empty_domain = ~domain_mask.any(dim=-1)
        if fallback_to_query:
            readout = torch.where(empty_domain.unsqueeze(-1), queries, readout)
        else:
            readout = readout * (~empty_domain).unsqueeze(-1).to(dtype=readout.dtype)
        return readout, weights

    def _encode_sequence_with_layerwise_domain_tokens(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        domain_known_mask: torch.Tensor,
        initial_domain_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = tokens
        domain_tokens = initial_domain_tokens
        last_attn = encoded.new_zeros(
            encoded.shape[0],
            int(self.num_domains),
            hist_mask.shape[1],
        )
        for layer_idx, layer in enumerate(self.encoder.layers):
            encoded = layer(
                encoded,
                src_mask=attn_mask,
                src_key_padding_mask=~valid_mask,
            )
            history_for_read = encoded[:, 1 : hist_mask.shape[1] + 1, :]
            history_for_read = history_for_read * hist_mask.unsqueeze(-1).to(dtype=history_for_read.dtype)
            history_for_read = self.dvc_layerwise_history_norm(history_for_read)
            read_queries = self.dvc_layerwise_token_norm(domain_tokens)
            if self.dvc_query_mixers is not None:
                mixed_queries = self.dvc_query_mixers[layer_idx](read_queries)
                read_queries = self.dvc_layerwise_token_norm(mixed_queries)
            domain_readout, domain_attn = self._query_read_own_domain_history(
                read_queries,
                history_for_read,
                hist_mask,
                hist_domain_ids,
                domain_known_mask,
                fallback_to_query=False,
            )
            domain_update = float(self.dvc_layerwise_update_scale) * domain_readout
            domain_tokens = domain_tokens + domain_update
            domain_ffn_update = self.dvc_layerwise_token_ffn(domain_tokens)
            domain_tokens = domain_tokens + domain_ffn_update
            last_attn = domain_attn
        encoder_norm = getattr(self.encoder, "norm", None)
        if encoder_norm is not None:
            encoded = encoder_norm(encoded)
        return encoded, domain_tokens, last_attn

    def _domain_transition_tokens(
        self,
        history_encoded: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
        domain_known_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prev_domain_ids = torch.zeros_like(hist_domain_ids)
        prev_known_mask = torch.zeros_like(domain_known_mask)
        if hist_domain_ids.shape[1] > 1:
            prev_domain_ids[:, 1:] = hist_domain_ids[:, :-1]
            prev_known_mask[:, 1:] = domain_known_mask[:, :-1]

        switch_mask = (
            hist_mask
            & domain_known_mask
            & prev_known_mask
            & (hist_domain_ids != prev_domain_ids)
        )
        safe_target = target_domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        transition_ids = (
            prev_domain_ids.long().clamp(min=0, max=int(self.num_domains) - 1)
            * int(self.num_domains)
            + hist_domain_ids.long().clamp(min=0, max=int(self.num_domains) - 1)
        ) * int(self.num_domains) + safe_target.unsqueeze(1)
        transition_delta = self.dvc_transition_embedding(transition_ids)
        transition_delta = transition_delta * switch_mask.unsqueeze(-1).to(dtype=transition_delta.dtype)
        transition_tokens = history_encoded + transition_delta
        transition_tokens = transition_tokens * switch_mask.unsqueeze(-1).to(dtype=transition_tokens.dtype)
        return transition_tokens, transition_delta, switch_mask

    def _query_read_enter_domain_transitions(
        self,
        queries: torch.Tensor,
        transition_tokens: torch.Tensor,
        switch_mask: torch.Tensor,
        hist_domain_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        domain_ids = torch.arange(int(self.num_domains), device=transition_tokens.device)
        domain_mask = switch_mask.unsqueeze(1) & (
            hist_domain_ids.unsqueeze(1) == domain_ids.view(1, -1, 1)
        )
        scale = float(self.token_dim) ** -0.5
        scores = torch.einsum("bkd,bld->bkl", queries, transition_tokens) * scale
        scores = scores.masked_fill(~domain_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * domain_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        readout = torch.einsum("bkl,bld->bkd", weights, transition_tokens)
        has_token = domain_mask.any(dim=-1)
        readout = readout * has_token.unsqueeze(-1).to(dtype=readout.dtype)
        return readout, weights, has_token

    def encode_parts(self, interaction) -> dict[str, torch.Tensor]:
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        batch_size, max_len, _ = seq_emb.shape
        if max_len > self.max_len:
            raise ValueError(f"Input sequence length {max_len} exceeds model max_len {self.max_len}.")

        device = seq_emb.device
        if seq_len is None:
            seq_len = torch.full((batch_size,), max_len, dtype=torch.long, device=device)
        else:
            seq_len = seq_len.view(-1).long().clamp(min=0, max=max_len)
        hist_mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        safe_domain_ids = domain_ids.long().view(-1).clamp(min=0, max=int(self.num_domains) - 1)
        hist_domain_ids, domain_known_mask = self._history_domain_ids(interaction, hist_mask)
        hist_domain_one_hot = F.one_hot(
            hist_domain_ids.long().clamp(min=0, max=int(self.num_domains) - 1),
            num_classes=int(self.num_domains),
        ).to(dtype=seq_emb.dtype)
        hist_domain_counts = torch.einsum(
            "blk,bl->bk",
            hist_domain_one_hot,
            domain_known_mask.to(dtype=seq_emb.dtype),
        )

        seq_tokens = self.item_token_proj(seq_emb)
        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            seq_tokens = seq_tokens + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        pair_delta = self._domain_pair_delta(hist_domain_ids, domain_known_mask, domain_ids)
        seq_tokens = seq_tokens + pair_delta
        seq_tokens = seq_tokens * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        target_token = self.target_token_proj(target_emb).unsqueeze(1)
        start_token = self.start_token.expand(batch_size, -1, -1)
        tokens = torch.cat([start_token, seq_tokens, target_token], dim=1)

        start_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        target_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        valid_mask = torch.cat([start_mask, hist_mask, target_mask], dim=1)
        position_ids = self._position_ids(hist_mask, seq_len)
        tokens = self.input_dropout(tokens + self.position_embedding(position_ids))

        attn_mask = None
        if self.use_causal_mask:
            total_len = max_len + 2
            attn_mask = self.causal_mask[:total_len, :total_len]

        target_domain_token = self.dvc_target_domain_embedding(safe_domain_ids)
        anchors = self.dvc_cross_anchor.weight.to(device=device, dtype=tokens.dtype)
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
        target_domain_expand = target_domain_token.unsqueeze(1).expand(-1, int(self.num_domains), -1)
        initial_domain_tokens = None
        layerwise_domain_tokens = None
        layerwise_cross_attn = None
        if self.dvc_layerwise_domain_tokens:
            initial_target_expand = target_token.squeeze(1).unsqueeze(1).expand(-1, int(self.num_domains), -1)
            layerwise_query_input = torch.cat([initial_target_expand, target_domain_expand, anchors], dim=-1)
            initial_domain_tokens = self.dvc_cross_query(layerwise_query_input)
            if self.dvc_query_anchor_scale > 0:
                query_anchor_residual = F.layer_norm(anchors, (self.token_dim,)) * float(self.dvc_query_anchor_scale)
            else:
                query_anchor_residual = torch.zeros_like(initial_domain_tokens)
            initial_domain_tokens = initial_domain_tokens + query_anchor_residual
            encoded, layerwise_domain_tokens, layerwise_cross_attn = self._encode_sequence_with_layerwise_domain_tokens(
                tokens,
                valid_mask,
                attn_mask,
                hist_mask,
                hist_domain_ids,
                domain_known_mask,
                initial_domain_tokens,
            )
        else:
            encoded = self.encoder(
                tokens,
                mask=attn_mask,
                src_key_padding_mask=~valid_mask,
            )
        encoded = self.output_norm(encoded)
        history_encoded = encoded[:, 1 : max_len + 1, :]
        history_encoded = history_encoded * hist_mask.unsqueeze(-1).to(dtype=history_encoded.dtype)
        target_encoded = encoded[:, -1, :]

        shared_token = target_encoded
        target_expand = target_encoded.unsqueeze(1).expand(-1, int(self.num_domains), -1)
        if self.dvc_layerwise_domain_tokens:
            cross_queries = initial_domain_tokens
            cross_tokens = layerwise_domain_tokens
            cross_attn = layerwise_cross_attn
        else:
            query_input = torch.cat([target_expand, target_domain_expand, anchors], dim=-1)
            cross_queries = self.dvc_cross_query(query_input)
            if self.dvc_query_anchor_scale > 0:
                query_anchor_residual = F.layer_norm(anchors, (self.token_dim,)) * float(self.dvc_query_anchor_scale)
            else:
                query_anchor_residual = torch.zeros_like(cross_queries)
            cross_queries = cross_queries + query_anchor_residual
        if self.dvc_layerwise_domain_tokens:
            pass
        elif self.dvc_cross_domain_mask:
            cross_tokens, cross_attn = self._query_read_own_domain_history(
                cross_queries,
                history_encoded,
                hist_mask,
                hist_domain_ids,
                domain_known_mask,
            )
        else:
            cross_tokens, cross_attn = self._query_read_history(cross_queries, history_encoded, hist_mask)
        anchor_residual = anchors * float(self.dvc_anchor_residual_scale)
        cross_views = cross_tokens + self.dvc_cross_view_mlp(cross_tokens) + anchor_residual

        if self.dvc_transition_enabled:
            transition_tokens, transition_delta, transition_switch_mask = self._domain_transition_tokens(
                history_encoded,
                hist_mask,
                hist_domain_ids,
                domain_known_mask,
                safe_domain_ids,
            )
            transition_readout, transition_attn, transition_has_token = self._query_read_enter_domain_transitions(
                cross_queries,
                transition_tokens,
                transition_switch_mask,
                hist_domain_ids,
            )
            transition_views = transition_readout + self.dvc_transition_view_mlp(transition_readout)
            transition_views = transition_views * transition_has_token.unsqueeze(-1).to(dtype=transition_views.dtype)
        else:
            transition_delta = None
            transition_tokens = None
            transition_readout = None
            transition_views = None
            transition_attn = None

        if self.dvc_private_enabled:
            private_anchor = self.dvc_private_anchor(safe_domain_ids)
            private_query_input = torch.cat([target_encoded, target_domain_token, private_anchor], dim=-1)
            private_query = self.dvc_private_query(private_query_input)
            private_token, private_attn = self._query_read_history(
                private_query.unsqueeze(1),
                history_encoded,
                hist_mask,
            )
            private_token = private_token.squeeze(1)
            private_attn = private_attn.squeeze(1)
            private_view = private_token + self.dvc_private_view_mlp(private_token)
            private_features = self.dvc_private_seq_delta(private_view)
        else:
            private_query = None
            private_token = None
            private_attn = None
            private_view = None
            private_features = None

        shared_expand = shared_token.unsqueeze(1).expand(-1, int(self.num_domains), -1)
        if self.dvc_private_guided_gate:
            private_gate_context = private_view.detach().unsqueeze(1).expand(-1, int(self.num_domains), -1)
            private_gate_product = cross_views * private_gate_context
            private_gate_diff = cross_views - private_gate_context
            gate_parts = [
                cross_views,
                private_gate_context,
                private_gate_product,
                private_gate_diff,
                shared_expand,
            ]
        else:
            gate_parts = [cross_views, shared_expand, target_domain_expand]
        gate_input = torch.cat(gate_parts, dim=-1)
        gate_logits = self.dvc_gate_mlp(gate_input).squeeze(-1)
        if self.dvc_support_aware_gate:
            gate_support_mask = hist_domain_counts > 0
            if self.dvc_gate_allow_target_fallback:
                target_support = F.one_hot(
                    safe_domain_ids.long().clamp(min=0, max=int(self.num_domains) - 1),
                    num_classes=int(self.num_domains),
                ).to(dtype=torch.bool)
                gate_support_mask = gate_support_mask | target_support
            gate_logits = gate_logits.masked_fill(~gate_support_mask, -1e4)
        gate = torch.softmax(gate_logits, dim=-1)
        gate_entropy = -(gate * torch.log(gate.clamp_min(1e-12))).sum(dim=-1)
        cross_agg = torch.einsum("bk,bkd->bd", gate, cross_views)
        if transition_views is not None:
            transition_agg = torch.einsum("bk,bkd->bd", gate, transition_views)
        else:
            transition_agg = None
        if self.dvc_view_logit_mlp is not None:
            view_logit_input = torch.cat([cross_views, shared_expand, target_domain_expand], dim=-1)
            view_logits = self.dvc_view_logit_mlp(view_logit_input).squeeze(-1)
            view_logit_agg = float(self.dvc_view_logit_scale) * torch.einsum("bk,bk->b", gate, view_logits)
        else:
            view_logits = None
            view_logit_agg = None
        if self.dvc_cross_aux_mlp is not None:
            cross_aux_input = torch.cat([cross_views, shared_expand, target_domain_expand], dim=-1)
            cross_aux_logits = self.dvc_cross_aux_mlp(cross_aux_input).squeeze(-1)
        else:
            cross_aux_logits = None
        self._last_dvc_cross_aux_logits = cross_aux_logits
        if self.dvc_private_aux_mlp is not None and isinstance(private_view, torch.Tensor):
            private_aux_logits = self.dvc_private_aux_mlp(private_view).squeeze(-1)
        else:
            private_aux_logits = None
        self._last_dvc_private_aux_logits = private_aux_logits
        if self.dvc_diversity_weight > 0:
            norm_views = F.normalize(cross_views, dim=-1)
            view_sim = torch.einsum("bkd,bjd->bkj", norm_views, norm_views)
            num_views = int(view_sim.shape[-1])
            if num_views > 1:
                div_loss = (view_sim.sum(dim=(1, 2)) - num_views) / float(num_views * (num_views - 1))
                div_loss = div_loss.mean()
            else:
                div_loss = cross_views.new_zeros(())
        else:
            div_loss = None
        self._last_dvc_diversity_loss = div_loss

        current_x = self._current_features(interaction, batch_size, device)
        current_features = self.current_bottom(current_x)
        base_features = current_features + self.seq_delta(shared_token)
        cross_features = self.dvc_cross_seq_delta(cross_agg)
        if transition_agg is not None:
            transition_features = self.dvc_transition_seq_delta(transition_agg)
        else:
            transition_features = torch.zeros_like(base_features)
        if private_features is None:
            private_features = torch.zeros_like(base_features)
        features = base_features + cross_features + transition_features + private_features
        return {
            "features": features,
            "domain_ids": domain_ids,
            "history_encoded": history_encoded,
            "target_encoded": target_encoded,
            "shared_token": shared_token,
            "cross_queries": cross_queries,
            "cross_tokens": cross_tokens,
            "cross_views": cross_views,
            "cross_gate": gate,
            "cross_attn": cross_attn,
            "cross_agg": cross_agg,
            "transition_delta": transition_delta,
            "transition_tokens": transition_tokens,
            "transition_readout": transition_readout,
            "transition_views": transition_views,
            "transition_attn": transition_attn,
            "transition_agg": transition_agg,
            "transition_features": transition_features,
            "view_logits": view_logits,
            "view_logit_agg": view_logit_agg,
            "cross_aux_logits": cross_aux_logits,
            "private_aux_logits": private_aux_logits,
            "private_query": private_query,
            "private_token": private_token,
            "private_attn": private_attn,
            "private_view": private_view,
            "private_features": private_features,
            "hist_mask": hist_mask,
            "domain_known_mask": domain_known_mask,
            "hist_domain_ids": hist_domain_ids,
        }

    def predict_logits_with_domains(self, interaction) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = self.encode_parts(interaction)
        logits = self.forward(parts["features"], parts["domain_ids"])
        view_logit_agg = parts.get("view_logit_agg")
        if isinstance(view_logit_agg, torch.Tensor):
            logits = logits + view_logit_agg
        return logits, parts["domain_ids"], parts["features"]

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        aux_loss = self.dvc_auxiliary_loss(labels)
        if aux_loss is not None:
            loss = loss + aux_loss
        return loss

    def dvc_auxiliary_loss(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        labels = labels.float().view(-1)
        losses: list[torch.Tensor] = []
        aux_logits = self._last_dvc_cross_aux_logits
        if self.dvc_cross_aux_weight > 0 and isinstance(aux_logits, torch.Tensor):
            cross_labels = labels.view(-1, 1).expand_as(aux_logits)
            aux_loss = F.binary_cross_entropy_with_logits(aux_logits, cross_labels)
            losses.append(float(self.dvc_cross_aux_weight) * aux_loss)
        private_aux_logits = self._last_dvc_private_aux_logits
        if self.dvc_private_aux_weight > 0 and isinstance(private_aux_logits, torch.Tensor):
            private_aux_loss = F.binary_cross_entropy_with_logits(private_aux_logits.view(-1), labels)
            losses.append(float(self.dvc_private_aux_weight) * private_aux_loss)
        div_loss = self._last_dvc_diversity_loss
        if self.dvc_diversity_weight > 0 and isinstance(div_loss, torch.Tensor):
            losses.append(float(self.dvc_diversity_weight) * div_loss)
        if not losses:
            return None
        return torch.stack(losses).sum()
