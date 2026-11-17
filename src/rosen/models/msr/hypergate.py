from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from rosen.emb import SchemaManager
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.utils.activation import activation_layer
from rosen.models.utils.common import to_dims


def _make_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    *,
    dropout_rate: float = 0.0,
    activation: str = "relu",
    batch_norm: bool = False,
    last_activation: bool = False,
) -> nn.Sequential:
    dims = (int(input_dim), *tuple(int(dim) for dim in hidden_dims))
    if len(dims) < 2:
        raise ValueError("MLP needs at least input and output dimensions.")
    layers: list[nn.Module] = []
    for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = idx == len(dims) - 2
        layers.append(nn.Linear(in_dim, out_dim))
        if (not is_last) or last_activation:
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation_layer(activation, out_dim))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)


class HyperGateSENet(nn.Module):
    """Control-conditioned feature excitation used by HyperGate.

    RecStudio's HyperGate calls ``ctr.SENet(control, embedding)``.  Rosen
    does not carry that module, so this block keeps the same role: a control
    vector produces channel-wise gates for the flattened embedding vector.
    The scale is centered near identity by multiplying sigmoid gates by 2.
    """

    def __init__(
        self,
        control_dim: int,
        feature_dim: int,
        hidden_dim: int | None = None,
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or max(1, min(control_dim, feature_dim) // 2))
        self.gate = nn.Sequential(
            _make_mlp(
                control_dim,
                (hidden_dim, feature_dim),
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=False,
                last_activation=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, control: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * (2.0 * self.gate(control))


class BiasAwareMLP(nn.Module):
    """MLP whose linear outputs can receive per-sample HyperGate biases."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        *,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
    ):
        super().__init__()
        dims = (int(input_dim), *tuple(int(dim) for dim in hidden_dims))
        if len(dims) < 2:
            raise ValueError("BiasAwareMLP needs at least input and output dimensions.")
        self.output_dim = int(dims[-1])
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.linears.append(nn.Linear(in_dim, out_dim))
            self.norms.append(nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity())
            self.activations.append(activation_layer(activation, out_dim))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, bias_list: list[torch.Tensor] | None = None) -> torch.Tensor:
        out = x
        for idx, linear in enumerate(self.linears):
            out = linear(out)
            if bias_list is not None:
                out = out + bias_list[idx]
            out = self.norms[idx](out)
            if idx != len(self.linears) - 1:
                out = self.activations[idx](out)
                out = self.dropout(out)
        return out


class HyperGateBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        *,
        expert_dims: Iterable[int] = (256, 128, 64),
        num_experts: int = 4,
        aug_embed_dim: int = 64,
        aug_hidden_dims: Iterable[int] = (128,),
        gate_hidden_dims: Iterable[int] = (128,),
        enable_embedding_gate: bool = True,
        enable_biasing_gate: bool = True,
        enable_fusion_gate: bool = True,
        gate_input_type: str = "augmented_embedding",
        lambda_d: float = 0.01,
        lambda_t: float = 0.01,
        dropout_rate: float = 0.0,
        gate_dropout_rate: float = 0.0,
        aug_dropout_rate: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_domains = int(num_domains)
        self.num_experts = int(num_experts)
        self.expert_dims = to_dims(expert_dims, (256, 128, 64))
        self.output_dim = int(self.expert_dims[-1])
        self.aug_embed_dim = int(aug_embed_dim)
        self.enable_embedding_gate = bool(enable_embedding_gate)
        self.enable_biasing_gate = bool(enable_biasing_gate)
        self.enable_fusion_gate = bool(enable_fusion_gate)
        self.gate_input_type = str(gate_input_type)
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)

        if self.gate_input_type not in {"augmented_embedding", "id", "embedding"}:
            raise ValueError("gate_input_type must be one of: augmented_embedding, id, embedding")

        self.domain_anchor_embedding: nn.Embedding | None = None
        self.task_anchor_embedding: nn.Embedding | None = None
        self.domain_aug_module: nn.Module | None = None
        self.task_aug_module: nn.Module | None = None

        if self.gate_input_type in {"augmented_embedding", "id"}:
            self.domain_anchor_embedding = nn.Embedding(self.num_domains, self.aug_embed_dim)
            self.task_anchor_embedding = nn.Embedding(1, self.aug_embed_dim)

        if self.gate_input_type == "augmented_embedding":
            aug_dims = (*to_dims(aug_hidden_dims, (128,)), self.aug_embed_dim)
            self.domain_aug_module = _make_mlp(
                self.input_dim,
                aug_dims,
                dropout_rate=aug_dropout_rate,
                activation=activation,
                batch_norm=False,
                last_activation=False,
            )
            self.task_aug_module = _make_mlp(
                self.input_dim,
                aug_dims,
                dropout_rate=aug_dropout_rate,
                activation=activation,
                batch_norm=False,
                last_activation=False,
            )
            domain_control_dim = self.aug_embed_dim * 2
            task_control_dim = self.aug_embed_dim * 2
        elif self.gate_input_type == "id":
            domain_control_dim = self.aug_embed_dim
            task_control_dim = self.aug_embed_dim
        else:
            domain_control_dim = self.input_dim
            task_control_dim = 0

        expert_input_dim = self.input_dim
        if self.enable_embedding_gate:
            self.domain_gate = HyperGateSENet(
                domain_control_dim,
                self.input_dim,
                dropout_rate=gate_dropout_rate,
                activation=activation,
            )
        else:
            self.domain_gate = None

        if self.enable_biasing_gate:
            self.biasing_gates = nn.ModuleList(
                [
                    _make_mlp(
                        domain_control_dim,
                        (*to_dims(gate_hidden_dims, (128,)), hidden_dim),
                        dropout_rate=gate_dropout_rate,
                        activation=activation,
                        batch_norm=False,
                        last_activation=False,
                    )
                    for hidden_dim in self.expert_dims
                ]
            )
            self.experts = nn.ModuleList(
                [
                    BiasAwareMLP(
                        expert_input_dim,
                        self.expert_dims,
                        dropout_rate=dropout_rate,
                        activation=activation,
                        batch_norm=batch_norm,
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            self.biasing_gates = None
            self.experts = nn.ModuleList(
                [
                    _make_mlp(
                        expert_input_dim,
                        self.expert_dims,
                        dropout_rate=dropout_rate,
                        activation=activation,
                        batch_norm=batch_norm,
                        last_activation=False,
                    )
                    for _ in range(self.num_experts)
                ]
            )

        fusion_input_dim = (
            domain_control_dim + task_control_dim
            if self.enable_fusion_gate and self.gate_input_type in {"augmented_embedding", "id"}
            else expert_input_dim
        )
        self.fusion_gate = nn.Sequential(
            _make_mlp(
                fusion_input_dim,
                (*to_dims(gate_hidden_dims, (128,)), self.num_experts),
                dropout_rate=gate_dropout_rate,
                activation=activation,
                batch_norm=False,
                last_activation=False,
            ),
            nn.Softmax(dim=-1),
        )

        self._reset_anchor_embeddings()

    def _reset_anchor_embeddings(self) -> None:
        for emb in (self.domain_anchor_embedding, self.task_anchor_embedding):
            if emb is not None:
                nn.init.normal_(emb.weight, mean=0.0, std=1e-2)

    def _control(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aux_loss = x.new_zeros(())
        task_aug = None

        if self.gate_input_type == "augmented_embedding":
            assert self.domain_aug_module is not None
            assert self.task_aug_module is not None
            assert self.domain_anchor_embedding is not None
            assert self.task_anchor_embedding is not None

            domain_aug = self.domain_aug_module(x.detach())
            domain_anchor = self.domain_anchor_embedding(domain_ids.long()).detach()
            domain_control = torch.cat([domain_aug, domain_anchor], dim=-1)

            if labels is not None and self.lambda_d > 0:
                domain_scores = domain_aug @ self.domain_anchor_embedding.weight.t()
                aux_loss = aux_loss + self.lambda_d * nn.functional.cross_entropy(domain_scores, domain_ids.long())

            task_aug = self.task_aug_module(x.detach())
            task_anchor = self.task_anchor_embedding.weight[0].unsqueeze(0).expand(x.shape[0], -1).detach()
            task_control = torch.cat([task_aug, task_anchor], dim=-1)
            if labels is not None and self.lambda_t > 0:
                task_scores = (task_aug * self.task_anchor_embedding.weight[0].unsqueeze(0)).sum(dim=-1)
                aux_loss = aux_loss + self.lambda_t * nn.functional.binary_cross_entropy_with_logits(
                    task_scores,
                    labels.float(),
                )
        elif self.gate_input_type == "id":
            assert self.domain_anchor_embedding is not None
            assert self.task_anchor_embedding is not None
            domain_control = self.domain_anchor_embedding(domain_ids.long())
            task_control = self.task_anchor_embedding.weight[0].unsqueeze(0).expand(x.shape[0], -1)
        else:
            domain_control = x
            task_control = x.new_empty((x.shape[0], 0))

        if self.enable_fusion_gate and self.gate_input_type in {"augmented_embedding", "id"}:
            fusion_control = torch.cat([task_control, domain_control], dim=-1)
        else:
            fusion_control = x
        return domain_control, fusion_control, aux_loss

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        domain_control, fusion_control, aux_loss = self._control(x, domain_ids, labels)

        expert_input = x
        if self.domain_gate is not None:
            expert_input = self.domain_gate(domain_control, expert_input)

        if self.biasing_gates is not None:
            bias_list = [gate(domain_control) for gate in self.biasing_gates]
            expert_outputs = torch.stack(
                [expert(expert_input, bias_list) for expert in self.experts],
                dim=1,
            )
        else:
            expert_outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=1)

        gate = self.fusion_gate(fusion_control).unsqueeze(-1)
        hidden = torch.sum(gate * expert_outputs, dim=1)
        return hidden, aux_loss


class HyperGateModel(MSRModel):
    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        expert_dims: Iterable[int] = (256, 128, 64),
        num_experts: int = 4,
        aug_embed_dim: int = 64,
        aug_hidden_dims: Iterable[int] = (128,),
        gate_hidden_dims: Iterable[int] = (128,),
        enable_embedding_gate: bool = True,
        enable_biasing_gate: bool = True,
        enable_fusion_gate: bool = True,
        gate_input_type: str = "augmented_embedding",
        lambda_d: float = 0.01,
        lambda_t: float = 0.01,
        dropout_rate: float = 0.0,
        gate_dropout_rate: float = 0.0,
        aug_dropout_rate: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        self.backbone = HyperGateBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_experts=num_experts,
            aug_embed_dim=aug_embed_dim,
            aug_hidden_dims=aug_hidden_dims,
            gate_hidden_dims=gate_hidden_dims,
            enable_embedding_gate=enable_embedding_gate,
            enable_biasing_gate=enable_biasing_gate,
            enable_fusion_gate=enable_fusion_gate,
            gate_input_type=gate_input_type,
            lambda_d=lambda_d,
            lambda_t=lambda_t,
            dropout_rate=dropout_rate,
            gate_dropout_rate=gate_dropout_rate,
            aug_dropout_rate=aug_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.backbone.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden, _ = self.backbone(x, domain_ids, labels)
        return self.head(hidden, domain_ids)

    def logits(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        x, domain_ids = self.encode_features(interaction)
        hidden, aux_loss = self.backbone(x, domain_ids, labels)
        logits = self.head(hidden, domain_ids)
        task_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if self.lsp.active:
            self.lsp.add("task", task_loss)
            self.lsp.add("regularization.control", aux_loss)
        return task_loss + aux_loss
