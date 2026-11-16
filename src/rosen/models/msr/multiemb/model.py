from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from rosen.core.train.context import TrainContext
from rosen.core.train.hooks import CustomTrainStepProtocol
from rosen.emb import SchemaManager
from rosen.models.msr.base import MSRModel
from rosen.models.msr.two_stage import TwoStageRetrainMixin
from rosen.models.utils.common import build_mlp, to_dims


class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.sign(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        del ctx
        return grad_output.clamp_(-1.0, 1.0)


def _hard_gate_from_prob(prob: torch.Tensor, threshold: float) -> torch.Tensor:
    hard = torch.relu(prob - float(threshold))
    return LBSign.apply(hard)


class MultiEmbModel(TwoStageRetrainMixin, MSRModel, CustomTrainStepProtocol):
    """
    Source-like MultiEmb reproduction.

    The reference implementation is not a share-mask router. It is a
    mask-learning model:
    - a domain hypernet generates soft masks
    - each domain gets its own masked embedding stream
    - each domain branch predicts with its own masked input
    - training uses own-domain loss + a shared regularizer over all branches
    - retrain freezes the mask network and keeps the backbone trainable
    """

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            branch_hidden_dims: Iterable[int] = (256, 128),
            mask_hidden_dims: Iterable[int] = (256, 128),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            scaling: float = 1.0,
            lambda1: float = 3.0,
            threshold: float = 0.5,
            search_epochs: int = 1,
            mask_init_temperature: float = 1.0,
            mask_final_temperature: float = 500.0,
            search_steps_hint: int = 2600,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field

        self.shared_embedding = self.omni_embedding.whole
        self.input_dim = self.shared_embedding.embedding_dim

        self.branch_hidden_dims = to_dims(branch_hidden_dims, (256, 128))
        self.mask_hidden_dims = to_dims(mask_hidden_dims, (256, 128))
        self.scaling = float(scaling)
        self.lambda1 = float(lambda1)
        self.threshold = float(threshold)
        self.current_temperature = max(float(mask_init_temperature), 1e-4)
        self.final_temperature = max(float(mask_final_temperature), self.current_temperature)
        self.search_steps_hint = max(1, int(search_steps_hint))
        self._temperature_growth = self.final_temperature ** (1.0 / max(1, self.search_steps_hint - 1))
        self._search_step = 0

        self.domain_hypernet = build_mlp(
            self.input_dim,
            self.mask_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.feature_mask_heads = nn.ModuleList([
            nn.Linear(self.mask_hidden_dims[-1], self.input_dim)
            for _ in range(self.num_domains)
        ])
        self.embed_mask_heads = nn.ModuleList([
            nn.Linear(self.mask_hidden_dims[-1], self.input_dim)
            for _ in range(self.num_domains)
        ])
        self.branches = nn.ModuleList([
            build_mlp(
                self.input_dim,
                (*self.branch_hidden_dims, 1),
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_domains)
        ])

        self._init_two_stage_retrain(
            search_epochs=search_epochs,
            metric_path=("overall", "auc"),
            restore_best_before_retrain=True,
        )
        self._two_stage_model_kwargs = {
            "branch_hidden_dims": self.branch_hidden_dims,
            "mask_hidden_dims": self.mask_hidden_dims,
            "dropout_rate": float(dropout_rate),
            "activation": str(activation),
            "batch_norm": bool(batch_norm),
            "scaling": self.scaling,
            "lambda1": self.lambda1,
            "threshold": self.threshold,
            "search_epochs": self.search_epochs,
            "mask_init_temperature": float(mask_init_temperature),
            "mask_final_temperature": self.final_temperature,
            "search_steps_hint": self.search_steps_hint,
        }
        self.apply_xavier_initialization()

    def encode_features(self, interaction):
        shared_x = torch.flatten(self.shared_embedding(interaction), start_dim=1)
        domain_ids = interaction[self.DOMAIN].long().view(-1)
        return shared_x, shared_x, domain_ids

    def _compute_mask_bundle(self, x: torch.Tensor):
        hidden = self.domain_hypernet(x)
        if self.in_retrain_phase:
            feature_masks = torch.stack(
                [
                    _hard_gate_from_prob(torch.sigmoid(head(hidden)), self.threshold)
                    for head in self.feature_mask_heads
                ],
                dim=1,
            )
            embed_masks = torch.stack(
                [
                    _hard_gate_from_prob(torch.sigmoid(head(hidden)), self.threshold)
                    for head in self.embed_mask_heads
                ],
                dim=1,
            )
            return hidden, feature_masks, embed_masks

        feature_masks = torch.stack(
            [torch.sigmoid(self.current_temperature * head(hidden)) for head in self.feature_mask_heads],
            dim=1,
        )
        embed_masks = torch.stack(
            [torch.sigmoid(self.current_temperature * head(hidden)) for head in self.embed_mask_heads],
            dim=1,
        )
        return hidden, feature_masks, embed_masks

    def _masked_inputs(
            self,
            x: torch.Tensor,
            feature_masks: torch.Tensor,
            embed_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.scaling * x.unsqueeze(1) * feature_masks * embed_masks

    def _forward_dict(
            self,
            shared_x: torch.Tensor,
            mask_x: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden, feature_masks, embed_masks = self._compute_mask_bundle(mask_x)
        masked_inputs = self._masked_inputs(shared_x, feature_masks, embed_masks)
        branch_logits = torch.stack(
            [self.branches[domain_idx](masked_inputs[:, domain_idx]).squeeze(-1) for domain_idx in range(self.num_domains)],
            dim=1,
        )
        selected_logits = branch_logits.gather(dim=1, index=domain_ids.view(-1, 1)).squeeze(-1)
        return {
            "hyper_hidden": hidden,
            "feature_masks": feature_masks,
            "embed_masks": embed_masks,
            "masked_inputs": masked_inputs,
            "branch_logits": branch_logits,
            "selected_logits": selected_logits,
        }

    def forward(self, x: torch.Tensor, mask_x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(x, mask_x, domain_ids)["selected_logits"]

    def predict(self, interaction):
        shared_x, mask_x, domain_ids = self.encode_features(interaction)
        return self.forward(shared_x, mask_x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        shared_x, mask_x, domain_ids = self.encode_features(interaction)
        logits = self.forward(shared_x, mask_x, domain_ids)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def _phase_loss(
            self,
            branch_logits: torch.Tensor,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_losses = []
        for domain_idx in range(self.num_domains):
            branch_losses.append(
                F.binary_cross_entropy_with_logits(
                    branch_logits[:, domain_idx],
                    labels,
                    reduction="none",
                )
            )
        branch_losses_tensor = torch.stack(branch_losses, dim=1)
        own_branch_loss = branch_losses_tensor.gather(dim=1, index=domain_ids.view(-1, 1))
        shared_branch_loss = branch_losses_tensor.mean(dim=1, keepdim=True)
        total_loss = own_branch_loss.mean() + self.lambda1 * shared_branch_loss.mean()
        return total_loss, own_branch_loss.mean(), shared_branch_loss.mean()

    def _build_stage_two_model(self):
        return type(self)(
            self.manager,
            self.num_domains,
            **self._two_stage_model_kwargs,
        )

    def _prepare_retrain_phase(self) -> None:
        for module in [self.domain_hypernet, self.feature_mask_heads, self.embed_mask_heads]:
            for param in module.parameters():
                param.requires_grad = False
        for module in [self.omni_embedding, self.branches]:
            for param in module.parameters():
                param.requires_grad = True

    def _restore_retrain_runtime(self) -> None:
        for module in [self.domain_hypernet, self.feature_mask_heads, self.embed_mask_heads]:
            for param in module.parameters():
                param.requires_grad = False
        for module in [self.omni_embedding, self.branches]:
            for param in module.parameters():
                param.requires_grad = True

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        labels = batch_interaction[self.LABEL].float().view(-1)
        shared_x, mask_x, domain_ids = self.encode_features(batch_interaction)
        forward_dict = self._forward_dict(shared_x, mask_x, domain_ids)
        total_loss, own_loss, shared_loss = self._phase_loss(
            forward_dict["branch_logits"],
            labels,
            domain_ids,
        )
        self.lsp.add("task.own", own_loss)
        self.lsp.add("regularization.shared", shared_loss, weight=self.lambda1)
        loss = self.lsp.settle(total_loss)


        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()

        if not self.in_retrain_phase:
            self._search_step += 1
            self.current_temperature = min(self.final_temperature, self.current_temperature * self._temperature_growth)
        return loss
