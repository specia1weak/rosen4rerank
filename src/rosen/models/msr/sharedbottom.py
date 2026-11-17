from typing import Iterable, Optional

import torch
from torch import nn

from rosen.emb import SchemaManager
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead, SharedCodebookBottleneck
from rosen.models.utils.activation import activation_layer
from rosen.models.utils.common import build_mlp, to_dims


class SharedBottomBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims=None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            activate_output: bool = False,
    ):
        super().__init__()
        hidden_dims = to_dims(hidden_dims, (input_dim, input_dim))
        self.output_dim = int(hidden_dims[-1])
        self.activate_output = bool(activate_output)
        self.bottom = build_mlp(
            input_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.output_activation = (
            activation_layer(activation, self.output_dim)
            if self.activate_output
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        del domain_ids
        return self.output_activation(self.bottom(x))


class SharedBottomModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Optional[Iterable[int]] = (256, 128, 64),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            activate_backbone_output: bool = False,
            tower_hidden_dims: Optional[Iterable[int]] = (128, 64),
            tower_dropout_rate: float = 0.0,
            include_domain_in_input: bool = True,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.include_domain_in_input = bool(include_domain_in_input)
        self.input_view = (
            self.omni_embedding.whole if self.include_domain_in_input
            else self.omni_embedding.whole_without_domain
        )
        self.input_dim = self.input_view.embedding_dim
        resolved_hidden_dims = (
            self._default_hidden_dims()
            if hidden_dims is None
            else tuple(hidden_dims)
        )

        self.backbone = SharedBottomBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            hidden_dims=resolved_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            activate_output=activate_backbone_output,
        )

        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.backbone.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def _default_hidden_dims(self) -> tuple[int, ...]:
        input_dim = max(1, int(self.input_dim))
        return input_dim, input_dim

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x, domain_ids)
        return self.head(hidden, domain_ids)

    def logits(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.logits(interaction)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if self.lsp.active:
            self.lsp.add("task", loss)
        return loss


class SharedBottomLessModel(SharedBottomModel):
    def _default_hidden_dims(self) -> tuple[int, ...]:
        return (max(1, int(self.input_dim)),)

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Optional[Iterable[int]] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            tower_hidden_dims: Optional[Iterable[int]] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
        )


class SharedBottomPlusModel(SharedBottomModel):
    def _default_hidden_dims(self) -> tuple[int, ...]:
        input_dim = max(1, int(self.input_dim))
        return input_dim, input_dim, max(1, input_dim // 2), input_dim

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Optional[Iterable[int]] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            tower_hidden_dims: Optional[Iterable[int]] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
        )


class SharedBottomCodebookModel(SharedBottomModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Iterable[int] = (256, 128, 64),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            tower_hidden_dims: Iterable[int] = (128, 64),
            tower_dropout_rate: float = 0.0,
            num_tokens: int = 256,
            codebook_dim: int = 8,
            codebook_decay: float = 0.99,
            codebook_input_proj_dim: Optional[int] = None,
            codebook_kmeans_init: bool = False,
            quantize_weight: float = 1e-2,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
        )
        self.quantize_weight = float(quantize_weight)
        self.codebook = SharedCodebookBottleneck(
            input_dim=self.backbone.output_dim,
            projected_dim=codebook_input_proj_dim,
            num_tokens=num_tokens,
            codebook_dim=codebook_dim,
            decay=codebook_decay,
            kmeans_init=codebook_kmeans_init,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.codebook.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor):
        hidden = self.backbone(x, domain_ids)
        quantized, aux = self.codebook(hidden)
        logits = self.head(quantized, domain_ids)
        forward_dict = {
            "hidden": hidden,
            "quantized": quantized,
            "logits": logits,
            **aux,
        }
        return forward_dict

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(x, domain_ids)["logits"]

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_dict(x, domain_ids)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(forward_dict["logits"], labels)
        quantize_loss = forward_dict["quantize_loss"]
        if self.lsp.active:
            self.lsp.add("task", bce_loss)
            self.lsp.add("vq.quantize", quantize_loss, weight=self.quantize_weight)
        return bce_loss + self.quantize_weight * quantize_loss
