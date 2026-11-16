from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn

from rosen.emb import SchemaManager
from rosen.emb.emblayer import EmbView
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.utils.common import build_mlp, to_dims
from rosen.models.utils.general import DNN
from rosen.models.utils.sequence import AttentionSequencePoolingLayer


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


class AdaptiveDomainScalingLayer(nn.Module):
    """ADS PSRG/PCRG block for target-aware sequence pooling."""

    def __init__(
        self,
        sequence_dim: int,
        condition_dim: int,
        max_len: int,
        ads_dim: int = 128,
        hidden_dim: int = 64,
        eta: float = 2.0,
        query_chunks: int = 0,
        use_psrg: bool = True,
        use_pcrg: bool = True,
        att_hidden_units: Iterable[int] = (128, 32),
        att_activation: str = "sigmoid",
    ):
        super().__init__()
        self.sequence_dim = int(sequence_dim)
        self.condition_dim = int(condition_dim)
        self.max_len = int(max_len)
        self.ads_dim = int(ads_dim)
        self.hidden_dim = int(hidden_dim)
        self.eta = float(eta)
        self.use_psrg = bool(use_psrg)
        self.use_pcrg = bool(use_pcrg)
        if self.ads_dim <= 0:
            raise ValueError("ads_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.query_chunks = int(query_chunks)
        if self.query_chunks <= 0:
            self.query_chunks = self.max_len
        self.query_chunks = max(1, min(self.query_chunks, self.max_len))

        self.seq_proj = (
            nn.Linear(self.sequence_dim, self.ads_dim)
            if self.sequence_dim != self.ads_dim
            else nn.Identity()
        )
        self.target_proj = (
            nn.Linear(self.sequence_dim, self.ads_dim)
            if self.sequence_dim != self.ads_dim
            else nn.Identity()
        )

        self.shared_weight = nn.Parameter(torch.empty(self.ads_dim, self.ads_dim))
        self.weight_gen = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.ads_dim * self.ads_dim),
        )
        self.bias_gen = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.ads_dim),
        )
        self.query_gen = nn.Sequential(
            nn.Linear(self.condition_dim + self.ads_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.query_chunks * self.ads_dim),
        )
        self.local_att = DNN(
            inputs_dim=4 * self.ads_dim,
            hidden_units=tuple(int(v) for v in att_hidden_units),
            activation=att_activation,
            dropout_rate=0.0,
            use_bn=False,
        )
        self.att_dense = nn.Linear(int(tuple(att_hidden_units)[-1]), 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.eye_(self.shared_weight)
        for module in (self.weight_gen, self.bias_gen, self.query_gen):
            last = module[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def _personalize_sequence(
        self,
        seq_tokens: torch.Tensor,
        condition: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_psrg:
            return seq_tokens * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

        batch_size = seq_tokens.shape[0]
        private_weight = torch.sigmoid(self.weight_gen(condition)).view(
            batch_size,
            self.ads_dim,
            self.ads_dim,
        )
        generated_weight = self.eta * self.shared_weight.unsqueeze(0) * private_weight
        generated_bias = self.bias_gen(condition).unsqueeze(1)
        seq_tokens = torch.bmm(seq_tokens, generated_weight.transpose(1, 2)) + generated_bias
        return seq_tokens * hist_mask.unsqueeze(-1).to(dtype=seq_tokens.dtype)

    def _personalize_query(
        self,
        target_token: torch.Tensor,
        condition: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        shared_query = target_token.unsqueeze(1).expand(-1, seq_len, -1)
        if not self.use_pcrg:
            return shared_query

        query_private = self.query_gen(torch.cat([condition, target_token], dim=-1)).view(
            target_token.shape[0],
            self.query_chunks,
            self.ads_dim,
        )
        if self.query_chunks == seq_len:
            return shared_query + query_private

        chunk_index = (
            torch.arange(seq_len, device=target_token.device, dtype=torch.long)
            * self.query_chunks
            // max(seq_len, 1)
        ).clamp(max=self.query_chunks - 1)
        return shared_query + query_private.index_select(1, chunk_index)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        keys_len: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = keys.shape
        seq_tokens = self.seq_proj(keys)
        target_token = self.target_proj(query)
        keys_len = keys_len.view(-1).long().clamp(min=0, max=seq_len)
        hist_mask = torch.arange(seq_len, device=keys.device).unsqueeze(0) < keys_len.unsqueeze(1)

        personalized_keys = self._personalize_sequence(seq_tokens, condition, hist_mask)
        personalized_queries = self._personalize_query(target_token, condition, seq_len)
        attention_input = torch.cat(
            [
                personalized_queries,
                personalized_keys,
                personalized_queries - personalized_keys,
                personalized_queries * personalized_keys,
            ],
            dim=-1,
        )
        score = self.att_dense(self.local_att(attention_input)).transpose(1, 2)
        score = torch.where(hist_mask.unsqueeze(1), score, torch.zeros_like(score))
        return torch.matmul(score, personalized_keys).view(batch_size, self.ads_dim)


class DINModel(MSRModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: Optional[str] = None,
        bottom_hidden_dims: Iterable[int] = (512, 256, 128),
        dropout_rate: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = True,
        tower_hidden_dims: Iterable[int] = (128,),
        tower_dropout_rate: float = 0.2,
        att_hidden_units: Iterable[int] = (128, 32),
        exclude_current_fields: Iterable[str] = ("action",),
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = manager.domain_field
        self.LABEL = manager.label_field

        if seq_group_name is None:
            group_names = tuple(self.omni_embedding.seq_groups.keys())
            if len(group_names) != 1:
                raise ValueError(f"seq_group_name is required when groups={group_names}")
            seq_group_name = group_names[0]
        if seq_group_name not in self.omni_embedding.seq_groups:
            raise ValueError(f"Unknown sequence group: {seq_group_name}")

        excluded = set(exclude_current_fields or ())
        base_fields = [
            setting.field_name
            for setting in manager.settings
            if (
                not setting.is_sequence_setting
                and setting.field_name not in excluded
            )
        ]
        self.current_view = EmbView(self.omni_embedding, include_fields=base_fields)
        self.history_view = self.omni_embedding.seq_groups[seq_group_name]

        domain_setting = manager.get_setting(self.DOMAIN)
        self.domain_shift = 1 if getattr(domain_setting, "padding_zero", False) else 0

        hidden_dims = to_dims(bottom_hidden_dims, default=(512, 256, 128))
        self.bottom_input_dim = self.current_view.embedding_dim + self.history_view.embedding_dim
        self.bottom_output_dim = int(hidden_dims[-1])
        self.att_pool = AttentionSequencePoolingLayer(
            embedding_dim=self.history_view.embedding_dim,
            att_hidden_units=tuple(int(v) for v in att_hidden_units),
            att_activation="sigmoid",
        )
        self.bottom = build_mlp(
            self.bottom_input_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.bottom_output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def encode_features(self, interaction):
        current_x = torch.flatten(self.current_view(interaction), start_dim=1)
        seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
        history_x = self.att_pool(target_emb, seq_emb, seq_len.view(-1))
        features = torch.cat([current_x, history_x], dim=-1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        return features, domain_ids

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.bottom(x), domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


class ADSDINModel(MSRModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        seq_group_name: Optional[str] = None,
        bottom_hidden_dims: Iterable[int] = (256, 128),
        dropout_rate: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
        tower_hidden_dims: Iterable[int] = (64,),
        tower_dropout_rate: float = 0.0,
        att_hidden_units: Iterable[int] = (64, 32),
        exclude_current_fields: Iterable[str] = ("action",),
        ads_condition_fields: Iterable[str] = (),
        ads_dim: int = 128,
        ads_hidden_dim: int = 64,
        ads_eta: float = 2.0,
        ads_query_chunks: int = 0,
        ads_use_psrg: bool = True,
        ads_use_pcrg: bool = True,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = manager.domain_field
        self.LABEL = manager.label_field

        if seq_group_name is None:
            group_names = tuple(self.omni_embedding.seq_groups.keys())
            if len(group_names) != 1:
                raise ValueError(f"seq_group_name is required when groups={group_names}")
            seq_group_name = group_names[0]
        if seq_group_name not in self.omni_embedding.seq_groups:
            raise ValueError(f"Unknown sequence group: {seq_group_name}")

        excluded = set(exclude_current_fields or ())
        base_fields = [
            setting.field_name
            for setting in manager.settings
            if (
                not setting.is_sequence_setting
                and setting.field_name not in excluded
            )
        ]
        self.current_view = EmbView(self.omni_embedding, include_fields=base_fields)
        self.history_view = self.omni_embedding.seq_groups[seq_group_name]

        if isinstance(ads_condition_fields, str):
            condition_fields = tuple(field.strip() for field in ads_condition_fields.split(",") if field.strip())
        else:
            condition_fields = tuple(ads_condition_fields or ())
        if not condition_fields:
            condition_fields = (self.DOMAIN,)
        self.condition_view = EmbView(self.omni_embedding, include_fields=condition_fields)

        domain_setting = manager.get_setting(self.DOMAIN)
        self.domain_shift = 1 if getattr(domain_setting, "padding_zero", False) else 0

        self.ads_pool = AdaptiveDomainScalingLayer(
            sequence_dim=self.history_view.embedding_dim,
            condition_dim=self.condition_view.embedding_dim,
            max_len=_infer_group_max_len(self.history_view),
            ads_dim=int(ads_dim),
            hidden_dim=int(ads_hidden_dim),
            eta=float(ads_eta),
            query_chunks=int(ads_query_chunks),
            use_psrg=bool(ads_use_psrg),
            use_pcrg=bool(ads_use_pcrg),
            att_hidden_units=tuple(int(v) for v in att_hidden_units),
            att_activation="sigmoid",
        )

        hidden_dims = to_dims(bottom_hidden_dims, default=(256, 128))
        self.bottom_input_dim = self.current_view.embedding_dim + self.ads_pool.ads_dim
        self.bottom_output_dim = int(hidden_dims[-1])
        self.bottom = build_mlp(
            self.bottom_input_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.bottom_output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def encode_features(self, interaction):
        current_x = torch.flatten(self.current_view(interaction), start_dim=1)
        condition_x = torch.flatten(self.condition_view(interaction), start_dim=1)
        seq_emb, target_emb, seq_len = self.history_view.fetch_all(interaction)
        history_x = self.ads_pool(target_emb, seq_emb, seq_len.view(-1), condition_x)
        features = torch.cat([current_x, history_x], dim=-1)
        domain_ids = interaction[self.DOMAIN].long().view(-1) - self.domain_shift
        return features, domain_ids

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.bottom(x), domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
