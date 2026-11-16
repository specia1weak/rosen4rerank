from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rosen.emb import SchemaManager
from rosen.models.utils.activation import activation_layer
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.msr.pepnet.backbones import EPNetBackbone, PPNetBackbone


class _PEPNetBaseModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Iterable[int] = (128, 64),
            scenario_dim: int = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            activate_backbone_output: bool = False,
            include_domain_in_input: bool = True,
            domain_as_scenario: bool = False,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.include_domain_in_input = bool(include_domain_in_input)
        self.domain_as_scenario = bool(domain_as_scenario)
        if self.domain_as_scenario:
            self.domain_view = self.omni_embedding.domain
            self.agnostic_view = self.omni_embedding.whole_without_domain
            self.input_dim = self.domain_view.embedding_dim + self.agnostic_view.embedding_dim
            if scenario_dim is None:
                scenario_dim = self.domain_view.embedding_dim
        else:
            self.input_view = (
                self.omni_embedding.whole if self.include_domain_in_input
                else self.omni_embedding.whole_without_domain
            )
            self.input_dim = self.input_view.embedding_dim

        self.backbone = self.build_backbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            scenario_dim=scenario_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head_input_activation = (
            activation_layer(activation) if bool(activate_backbone_output) else nn.Identity()
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.backbone.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
    def build_backbone(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims: Iterable[int],
            scenario_dim: Optional[int],
            dropout_rate: float,
            activation: str,
            batch_norm: bool,
    ) -> nn.Module:
        raise NotImplementedError

    def encode_features(self, interaction):
        if self.domain_as_scenario:
            domain_x = torch.flatten(self.domain_view(interaction), start_dim=1)
            agnostic_x = torch.flatten(self.agnostic_view(interaction), start_dim=1)
            x = torch.cat([domain_x, agnostic_x], dim=1)
            return x, interaction[self.DOMAIN].long()
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x, domain_ids)
        hidden = self.head_input_activation(hidden)
        return self.head(hidden, domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return F.binary_cross_entropy_with_logits(logits, labels)


class PPNetModel(_PEPNetBaseModel):
    def build_backbone(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims: Iterable[int],
            scenario_dim: Optional[int],
            dropout_rate: float,
            activation: str,
            batch_norm: bool,
    ) -> nn.Module:
        del activation, batch_norm
        return PPNetBackbone(
            input_dim=input_dim,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            scenario_dim=scenario_dim,
            dropout_rate=dropout_rate,
        )


class EPNetModel(_PEPNetBaseModel):
    def build_backbone(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims: Iterable[int],
            scenario_dim: Optional[int],
            dropout_rate: float,
            activation: str,
            batch_norm: bool,
    ) -> nn.Module:
        return EPNetBackbone(
            input_dim=input_dim,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            scenario_dim=scenario_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )


class PEPNetModel(EPNetModel):
    pass
