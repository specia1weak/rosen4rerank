from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from rosen.emb import SchemaManager
from rosen.emb.emblayer import EmbView
from rosen.emb.schema import SequenceSetting
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components.heads import DomainTowerHead
from rosen.models.utils.activation import activation_layer
from rosen.models.utils.common import to_dims
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


def _make_norm(dim: int) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    return nn.LayerNorm(dim)


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    mask_f = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (tokens * mask_f).sum(dim=1) / denom


class Embedding2Tokenization(nn.Module):
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
            nn.Linear(input_dim, hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        tokens = self.mlp(x)
        return tokens.view(x.size(0), self.num_tokens, self.token_dim)


class QueryGeneration(Embedding2Tokenization):
    pass


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        use_sdpa: bool = False,
    ):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.input_dim // self.num_heads
        self.dropout_rate = float(dropout_rate)
        self.use_sdpa = bool(use_sdpa)
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )
        else:
            score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(score, dim=-1)
            if self.dropout_rate > 0:
                attn = F.dropout(attn, p=self.dropout_rate, training=self.training)
            attn = torch.matmul(attn, v)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, query_len, self.input_dim)
        return self.out_proj(attn)


class SequenceRepresentationLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.norm = _make_norm(input_dim)
        self.attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn_norm = _make_norm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            activation_layer(activation),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(input_dim * 2, input_dim),
        )

    def forward(self, s_tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).to(dtype=s_tokens.dtype)
        x = s_tokens + self.attn(self.norm(s_tokens), self.norm(s_tokens))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x + self.ffn(self.ffn_norm(x))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class QueryDecodingLayer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        self.query_norm = _make_norm(input_dim)
        self.sequence_norm = _make_norm(input_dim)
        self.attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout_rate=dropout_rate)

    def forward(
        self,
        global_tokens: torch.Tensor,
        s_tokens: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).to(dtype=s_tokens.dtype)
        decoded = self.attn(self.query_norm(global_tokens), self.sequence_norm(s_tokens))
        return global_tokens + decoded


class QueryBoostingLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_global_token: int,
        num_ns_token: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = int(num_global_token) + int(num_ns_token)
        self.mixer_norm = _make_norm(input_dim)
        self.ffn_norm = _make_norm(input_dim)
        self.token_mixer = nn.Sequential(
            nn.Linear(self.num_tokens, self.num_tokens * 2),
            activation_layer(activation),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(self.num_tokens * 2, self.num_tokens),
        )
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            activation_layer(activation),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(input_dim * 2, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mixer(self.mixer_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + mixed
        return x + self.ffn(self.ffn_norm(x))


class HyFormerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        num_ns_token: int,
        num_global_token: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_global_token = int(num_global_token)
        self.seq_layers = nn.ModuleList(
            [
                SequenceRepresentationLayer(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    activation=activation,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.decode_layers = nn.ModuleList(
            [
                QueryDecodingLayer(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.boost_layers = nn.ModuleList(
            [
                QueryBoostingLayer(
                    input_dim=input_dim,
                    num_global_token=num_global_token,
                    num_ns_token=num_ns_token,
                    activation=activation,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        s_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_count = global_tokens.size(1)
        for layer_idx, (seq_layer, decode_layer, boost_layer) in enumerate(zip(
            self.seq_layers,
            self.decode_layers,
            self.boost_layers,
        )):
            s_tokens = seq_layer(s_tokens, mask)
            decoded_global = decode_layer(global_tokens, s_tokens, mask)
            boosted = boost_layer(torch.cat([decoded_global, ns_tokens], dim=1))
            global_tokens = boosted[:, :global_count, :]
            ns_tokens = boosted[:, global_count:, :]
        return s_tokens, global_tokens, ns_tokens


class HyFormerModel(MSRModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: str = "user_history",
        token_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 2,
        num_ns_token: int = 4,
        num_global_token: int = 4,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128,),
        tower_dropout_rate: float = 0.2,
        tower_batch_norm: bool = False,
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
        target_field_names = {
            setting.field_name
            for setting in self.history_view.target_settings
        }
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
        self.sequence_embedding_dim = self.history_view.embedding_dim
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
            self.relative_time_to_seconds_divisor = _resolve_timestamp_to_seconds_divisor(
                relative_time_unit,
                self.time_field,
            )
            self.relative_time_embedding = RelativeTimeEmbedding(
                embedding_dim=self.token_dim,
                boundaries=tuple(relative_time_boundaries or TIME_BUCKET_BOUNDARIES),
                init_std=float(relative_time_init_std),
            )
        else:
            self.relative_time_to_seconds_divisor = 1
            self.relative_time_embedding = None

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

        dense_context_dim = self.current_embedding_dim + self.token_dim
        self.ns_tokenizer = Embedding2Tokenization(
            input_dim=dense_context_dim,
            token_dim=self.token_dim,
            num_tokens=num_ns_token,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.query_generator = QueryGeneration(
            input_dim=dense_context_dim + self.token_dim,
            token_dim=self.token_dim,
            num_tokens=num_global_token,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.hyformer = HyFormerBlock(
            input_dim=self.token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_ns_token=num_ns_token,
            num_global_token=num_global_token,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.token_dim,
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

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_relative_time:
            seq_emb, target_emb, seq_len, hist_ts = self.history_view.fetch_all(interaction, include_time=True)
        else:
            seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
            hist_ts = None
        batch_size, max_len, _ = seq_emb.shape
        device = seq_emb.device
        seq_len = seq_len.view(-1).long().clamp(min=0, max=max_len)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_len.unsqueeze(1)

        seq_feature_mode = getattr(self, "seq_feature_mode", "all")
        if seq_feature_mode == "time_only":
            s_tokens = seq_emb.new_zeros((batch_size, max_len, self.token_dim))
        else:
            s_tokens = self.item_token_proj(seq_emb)
            if getattr(self, "remove_sequence_projection_base", False):
                s_tokens = s_tokens - self.item_token_proj(torch.zeros_like(seq_emb))
        if self.relative_time_embedding is not None:
            curr_ts = _to_seconds(interaction[self.time_field], self.relative_time_to_seconds_divisor)
            hist_ts = _to_seconds(hist_ts, self.relative_time_to_seconds_divisor)
            s_tokens = s_tokens + self.relative_time_embedding(
                curr_ts,
                hist_ts,
                seq_lens=seq_len,
            )
        s_tokens = s_tokens * mask.unsqueeze(-1).to(dtype=s_tokens.dtype)
        target_token = self.target_token_proj(target_emb)
        current_x = self._current_features(interaction, batch_size, device)

        dense_context = torch.cat([current_x, target_token], dim=-1)
        ns_tokens = self.ns_tokenizer(dense_context)
        seq_pool = _masked_mean(s_tokens, mask)
        global_info = torch.cat([dense_context, seq_pool], dim=-1)
        global_tokens = self.query_generator(global_info)

        _, global_tokens, _ = self.hyformer(s_tokens, global_tokens, ns_tokens, mask)
        features = self.dropout(global_tokens.mean(dim=1))
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
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
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
