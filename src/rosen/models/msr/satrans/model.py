from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from rosen.emb import SchemaManager
from rosen.emb.emblayer import EmbView
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.utils.general import MLP


def _to_str_tuple(value) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return tuple(part for part in parts if part)
    return tuple(str(item) for item in value)


def _to_int_tuple(value, default: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(v) for v in default)
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return tuple(int(part) for part in parts if part)
    return tuple(int(v) for v in value)


class SATransFieldEmbedding(nn.Module):
    """
    Return field-wise embeddings as [batch, field_num, embedding_dim].

    SATrans-style attention operates over fields, while Rosen's normal
    `whole` view returns one flattened vector. This adapter keeps the shared
    OmniEmbLayer tables but stacks each field separately.
    """

    def __init__(
            self,
            manager: SchemaManager,
            omni_layer: nn.Module,
            *,
            include_fields: Iterable[str] | str | None = None,
            exclude_fields: Iterable[str] | str | None = None,
            exclude_domain_from_interactions: bool = False,
            embedding_dim: int | None = None,
    ):
        super().__init__()
        include_fields = _to_str_tuple(include_fields)
        exclude_fields_set = set(_to_str_tuple(exclude_fields) or tuple())
        if exclude_domain_from_interactions:
            exclude_fields_set.update(str(field) for field in manager.domain_fields)
        exclude_fields_tuple = tuple(exclude_fields_set) if exclude_fields_set else None

        self.view = EmbView(
            omni_layer,
            include_fields=include_fields,
            exclude_fields=exclude_fields_tuple,
        )
        self.settings = tuple(setting for setting in self.view.selected_settings if not setting.is_sequence_setting)
        if not self.settings:
            raise ValueError("SATrans requires at least one non-sequence interaction field.")

        base_dims = tuple(int(setting.embedding_dim) for setting in self.settings)
        if embedding_dim is None:
            unique_dims = sorted(set(base_dims))
            if len(unique_dims) != 1:
                raise ValueError(
                    "SATrans requires a target embedding_dim when selected fields have non-uniform "
                    f"embedding dims: {unique_dims}"
                )
            embedding_dim = unique_dims[0]
        self.embedding_dim = int(embedding_dim)
        self.num_fields = len(self.settings)
        self.field_names = tuple(setting.field_name for setting in self.settings)

        projections = []
        for dim in base_dims:
            if int(dim) == self.embedding_dim:
                projections.append(nn.Identity())
            else:
                projections.append(nn.Linear(int(dim), self.embedding_dim, bias=False))
        self.projections = nn.ModuleList(projections)

    @property
    def output_dim(self) -> int:
        return self.num_fields * self.embedding_dim

    def forward(self, interaction) -> torch.Tensor:
        field_embs = self.view(interaction, split_by="name")
        outputs = []
        for setting, projection in zip(self.settings, self.projections):
            field_emb = field_embs[setting.field_name]
            outputs.append(projection(field_emb))
        return torch.stack(outputs, dim=1)


class SATransMetaNet(nn.Module):
    """Domain-conditioned residual MLP used to modulate Q/K/V tensors."""

    def __init__(
            self,
            embedding_dim: int,
            meta_hidden_units: Iterable[int],
            dropout_rate: float = 0.1,
            use_norm: bool = True,
    ):
        super().__init__()
        dims = (int(embedding_dim), *[int(v) for v in meta_hidden_units])
        if dims[-1] != int(embedding_dim):
            dims = (*dims, int(embedding_dim))
        if len(dims) < 2:
            raise ValueError("SATransMetaNet requires at least one generated linear layer.")
        self.dims = dims
        self.param_size = sum(in_dim * out_dim for in_dim, out_dim in zip(dims[:-1], dims[1:]))
        self.dropout = nn.Dropout(float(dropout_rate))
        self.use_norm = bool(use_norm)
        self.layer_norm = nn.LayerNorm(int(embedding_dim), eps=1e-6) if self.use_norm else nn.Identity()

    def forward(self, x: torch.Tensor, meta_params: torch.Tensor) -> torch.Tensor:
        weights = []
        offset = 0
        batch_size = x.size(0)
        for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:]):
            size = in_dim * out_dim
            weight = meta_params[:, offset:offset + size].reshape(batch_size, in_dim, out_dim)
            offset += size
            weights.append(weight)

        residual = x
        out = x
        for layer_idx, weight in enumerate(weights):
            out = torch.bmm(out, weight)
            if layer_idx != len(weights) - 1:
                out = F.relu(out)
        out = self.dropout(out)
        return self.layer_norm(out + residual)


class SATransMetaTransformerLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            meta_hidden_units: Iterable[int],
            *,
            meta_mode: str = "QK",
            head_num: int = 4,
            use_residual: bool = True,
            scaling: bool = True,
            attn_dropout_rate: float = 0.1,
            dropout_rate: float = 0.1,
            use_norm: bool = True,
    ):
        super().__init__()
        embedding_dim = int(embedding_dim)
        head_num = int(head_num)
        if head_num <= 0:
            raise ValueError("head_num must be positive.")
        if embedding_dim % head_num != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible by head_num={head_num}."
            )

        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.head_dim = embedding_dim // head_num
        self.meta_mode = str(meta_mode or "").upper()
        self.use_residual = bool(use_residual)
        self.scaling = bool(scaling)

        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.meta_net = SATransMetaNet(
            embedding_dim=embedding_dim,
            meta_hidden_units=meta_hidden_units,
            dropout_rate=dropout_rate,
            use_norm=use_norm,
        )
        self.attn_dropout = nn.Dropout(float(attn_dropout_rate))
        self.dropout = nn.Dropout(float(dropout_rate))
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6) if use_norm else nn.Identity()

    @property
    def meta_param_size(self) -> int:
        return self.meta_net.param_size

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, field_num, _ = x.shape
        return x.reshape(batch_size, field_num, self.head_num, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, field_num, _ = x.shape
        return x.transpose(1, 2).reshape(batch_size, field_num, self.embedding_dim)

    def forward(self, inputs: torch.Tensor, meta_params: torch.Tensor) -> torch.Tensor:
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        if "Q" in self.meta_mode:
            query = self.meta_net(query, meta_params)
        if "K" in self.meta_mode:
            key = self.meta_net(key, meta_params)
        if "V" in self.meta_mode:
            value = self.meta_net(value, meta_params)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        scores = torch.matmul(query, key.transpose(-2, -1))
        if self.scaling:
            scores = scores / math.sqrt(self.head_dim)
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        result = torch.matmul(attn, value)
        result = self._merge_heads(result)
        result = self.dropout(self.out(result))
        if self.use_residual:
            result = result + inputs
        return self.layer_norm(result)


class SATransBackbone(nn.Module):
    def __init__(
            self,
            field_num: int,
            embedding_dim: int,
            num_domains: int,
            *,
            domain_att_layer_num: int = 3,
            att_head_num: int = 4,
            meta_hidden_units: Iterable[int] = (64, 32),
            meta_mode: str = "QK",
            att_residual: bool = True,
            attn_dropout_rate: float = 0.1,
            dropout_rate: float = 0.1,
            use_norm: bool = True,
            domain_emb_dim: int | None = None,
    ):
        super().__init__()
        self.field_num = int(field_num)
        self.embedding_dim = int(embedding_dim)
        self.output_dim = self.field_num * self.embedding_dim
        self.domain_emb_dim = int(domain_emb_dim or embedding_dim)

        self.domain_embeddings = nn.Embedding(int(num_domains), self.domain_emb_dim)
        meta_hidden_units = _to_int_tuple(meta_hidden_units, (64, 32))
        self.layers = nn.ModuleList([
            SATransMetaTransformerLayer(
                embedding_dim=self.embedding_dim,
                meta_hidden_units=meta_hidden_units,
                meta_mode=meta_mode,
                head_num=att_head_num,
                use_residual=att_residual,
                attn_dropout_rate=attn_dropout_rate,
                dropout_rate=dropout_rate,
                use_norm=use_norm,
            )
            for _ in range(int(domain_att_layer_num))
        ])
        if not self.layers:
            raise ValueError("domain_att_layer_num must be positive.")
        self.domain_map = nn.Linear(self.domain_emb_dim, self.layers[0].meta_param_size)

    def forward(self, field_embeddings: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        domain_ids = domain_ids.long().clamp(min=0, max=self.domain_embeddings.num_embeddings - 1)
        domain_state = F.relu(self.domain_embeddings(domain_ids))
        meta_params = self.domain_map(domain_state)

        att_input = field_embeddings
        for layer in self.layers:
            att_input = layer(att_input, meta_params)
        return att_input.flatten(start_dim=1)


class SATransModel(MSRModel):
    """
    Rosen-native SATrans.

    This is a framework-native implementation of the SATrans core idea:
    scenario/domain embeddings generate meta parameters that modulate
    feature-field self-attention interactions.
    """

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            embedding_dim: int | None = None,
            interaction_fields: Iterable[str] | str | None = None,
            exclude_fields: Iterable[str] | str | None = None,
            exclude_domain_from_interactions: bool = False,
            domain_att_layer_num: int = 3,
            att_head_num: int = 4,
            meta_hidden_units: Iterable[int] | str = (64, 32),
            meta_mode: str = "QK",
            att_residual: bool = True,
            attn_dropout_rate: float = 0.1,
            dropout_rate: float = 0.1,
            use_norm: bool = True,
            domain_emb_dim: int | None = None,
            head_type: str = "shared_linear",
            head_hidden_dims: Iterable[int] | str | None = None,
            tower_hidden_dims: Iterable[int] | str | None = None,
            tower_dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            xavier_init: bool = False,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        if self.DOMAIN is None:
            raise ValueError("SATransModel requires manager.domain_field.")
        if self.LABEL is None:
            raise ValueError("SATransModel requires manager.label_field.")

        self.field_embedding = SATransFieldEmbedding(
            manager,
            self.omni_embedding,
            include_fields=interaction_fields,
            exclude_fields=exclude_fields,
            exclude_domain_from_interactions=exclude_domain_from_interactions,
            embedding_dim=embedding_dim,
        )
        self.backbone = SATransBackbone(
            field_num=self.field_embedding.num_fields,
            embedding_dim=self.field_embedding.embedding_dim,
            num_domains=num_domains,
            domain_att_layer_num=domain_att_layer_num,
            att_head_num=att_head_num,
            meta_hidden_units=meta_hidden_units,
            meta_mode=meta_mode,
            att_residual=att_residual,
            attn_dropout_rate=attn_dropout_rate,
            dropout_rate=dropout_rate,
            use_norm=use_norm,
            domain_emb_dim=domain_emb_dim,
        )

        self.head_type = str(head_type or "shared_linear").lower()
        if self.head_type in {"shared", "shared_linear", "linear"}:
            hidden_dims = _to_int_tuple(head_hidden_dims, ()) if head_hidden_dims is not None else tuple()
            if hidden_dims:
                self.head = MLP(
                    self.backbone.output_dim,
                    *hidden_dims,
                    1,
                    dropout_rate=float(tower_dropout_rate),
                    activation=activation,
                    batch_norm=batch_norm,
                )
            else:
                self.head = nn.Linear(self.backbone.output_dim, 1)
        elif self.head_type in {"domain_tower", "tower"}:
            self.head = DomainTowerHead(
                num_domains=num_domains,
                input_dim=self.backbone.output_dim,
                hidden_dims=_to_int_tuple(tower_hidden_dims, (max(1, self.backbone.output_dim // 2),)),
                dropout_rate=tower_dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
        else:
            raise ValueError(f"Unknown SATrans head_type={head_type!r}")

        if xavier_init:
            self.apply_xavier_initialization()

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.field_embedding.field_names

    def encode_features(self, interaction):
        field_embeddings = self.field_embedding(interaction)
        domain_ids = interaction[self.DOMAIN].long()
        return field_embeddings, domain_ids

    def _forward_dict(self, field_embeddings: torch.Tensor, domain_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(field_embeddings, domain_ids)
        if self.head_type in {"domain_tower", "tower"}:
            logits = self.head(hidden, domain_ids)
        else:
            logits = self.head(hidden).squeeze(-1)
        return {
            "hidden": hidden,
            "logits": logits,
        }

    def forward(self, field_embeddings: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(field_embeddings, domain_ids)["logits"]

    def predict(self, interaction):
        field_embeddings, domain_ids = self.encode_features(interaction)
        return self.forward(field_embeddings, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return F.binary_cross_entropy_with_logits(logits, labels)
