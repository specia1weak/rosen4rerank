from typing import Iterable

import torch
from torch import nn

from rosen.emb import SchemaManager
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.msr.ple.layers import PLEFeatureGateLayer, PLELayer, select_domain_output
from rosen.models.utils.activation import activation_layer
from rosen.models.utils.common import to_dims


class PLEEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = int(num_domains)
        self.output_dim = int(expert_dims[-1])
        self.layers = nn.ModuleList()

        layer_input_dim = int(input_dim)
        for _ in range(num_levels):
            layer = PLELayer(
                input_dim=layer_input_dim,
                num_domains=num_domains,
                expert_dims=expert_dims,
                num_specific_experts=num_specific_experts,
                num_shared_experts=num_shared_experts,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            self.layers.append(layer)
            layer_input_dim = layer.output_dim

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        task_inputs = [x for _ in range(self.num_domains)]
        shared_input = x
        for layer in self.layers:
            task_inputs, shared_input = layer(task_inputs, shared_input, domain_ids)
        return select_domain_output(task_outputs=task_inputs, domain_ids=domain_ids)

    def reset_gate_stats(self) -> None:
        for layer in self.layers:
            reset = getattr(layer, "reset_gate_stats", None)
            if callable(reset):
                reset()

    def export_gate_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for level, layer in enumerate(self.layers):
            export = getattr(layer, "export_gate_rows", None)
            if callable(export):
                rows.extend(export(level=int(level)))
        return rows


class PLEVersion1Encoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            detach_gate_input: bool = True,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (64,))
        self.num_domains = int(num_domains)
        self.output_dim = int(expert_dims[-1])
        self.layers = nn.ModuleList()

        layer_input_dim = int(input_dim)
        for _ in range(num_levels):
            layer = PLEFeatureGateLayer(
                input_dim=layer_input_dim,
                num_domains=num_domains,
                expert_dims=expert_dims,
                num_specific_experts=num_specific_experts,
                num_shared_experts=num_shared_experts,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
                detach_gate_input=detach_gate_input,
            )
            self.layers.append(layer)
            layer_input_dim = layer.output_dim

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        task_inputs = [x for _ in range(self.num_domains)]
        shared_input = x
        for layer in self.layers:
            task_inputs, shared_input = layer(task_inputs, shared_input)
        return select_domain_output(task_outputs=task_inputs, domain_ids=domain_ids)


class PLEModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            activate_backbone_output: bool = False,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
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

        self.encoder = PLEEncoder(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_levels=num_levels,
            num_specific_experts=num_specific_experts,
            num_shared_experts=num_shared_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head_input_activation = (
            activation_layer(activation) if bool(activate_backbone_output) else nn.Identity()
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.encoder.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x, domain_ids)
        hidden = self.head_input_activation(hidden)
        return self.head(hidden, domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def reset_train_observations(self) -> None:
        self.encoder.reset_gate_stats()

    def export_train_observation_rows(self) -> list[dict[str, object]]:
        return self.encoder.export_gate_rows()


class PLEVersion1Model(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (64,),
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            detach_gate_input: bool = True,
            activate_backbone_output: bool = False,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
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

        self.encoder = PLEVersion1Encoder(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_levels=num_levels,
            num_specific_experts=num_specific_experts,
            num_shared_experts=num_shared_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            detach_gate_input=detach_gate_input,
        )
        self.head_input_activation = (
            activation_layer(activation) if bool(activate_backbone_output) else nn.Identity()
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.encoder.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x, domain_ids)
        hidden = self.head_input_activation(hidden)
        return self.head(hidden, domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


PLEVesion1Model = PLEVersion1Model
PLEVersion1 = PLEVersion1Model
