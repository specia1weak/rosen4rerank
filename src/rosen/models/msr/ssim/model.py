from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from rosen.core.train.context import TrainContext
from rosen.core.train.hooks import CustomTrainStepProtocol
from rosen.emb import SchemaManager
from rosen.emb.emblayer import OmniEmbLayer
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


class SSIMModel(TwoStageRetrainMixin, MSRModel, CustomTrainStepProtocol):
    """
    Source-like SSIM reproduction.

    The core semantics are kept close to the original implementation:
    - a domain hypernet produces per-domain gate logits
    - expert branches are trained on the shared representation
    - search phase uses soft gates and a share regularizer
    - retrain phase restores the best search checkpoint, freezes mask modules,
      and turns gates hard
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
            lambda1: float = 1.0,
            threshold: float = 0.49,
            search_epochs: int = 1,
            mask_init_temperature: float = 1.0,
            mask_final_temperature: float = 1000.0,
            search_steps_hint: int = 2600,
            retrain_topk: Optional[float] = None,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field

        self.shared_embedding = self.omni_embedding.whole
        self.mask_omni_embedding = OmniEmbLayer(manager=manager)
        self.mask_embedding = self.mask_omni_embedding.whole
        self.input_dim = self.shared_embedding.embedding_dim

        self.branch_hidden_dims = to_dims(branch_hidden_dims, (256, 128))
        self.mask_hidden_dims = to_dims(mask_hidden_dims, (256, 128))
        self.lambda1 = float(lambda1)
        self.threshold = float(threshold)
        self.current_temperature = max(float(mask_init_temperature), 1e-4)
        self.final_temperature = max(float(mask_final_temperature), self.current_temperature)
        self.search_steps_hint = max(1, int(search_steps_hint))
        self.retrain_topk = retrain_topk if retrain_topk is None else float(retrain_topk)
        self._temperature_growth = self.final_temperature ** (1.0 / max(1, self.search_steps_hint - 1))
        self._search_step = 0

        self.domain_hypernet = build_mlp(
            self.input_dim,
            self.mask_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.mask_heads = nn.ModuleList([
            nn.Linear(self.mask_hidden_dims[-1], 1)
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
            "lambda1": self.lambda1,
            "threshold": self.threshold,
            "search_epochs": self.search_epochs,
            "mask_init_temperature": float(mask_init_temperature),
            "mask_final_temperature": self.final_temperature,
            "search_steps_hint": self.search_steps_hint,
            "retrain_topk": self.retrain_topk,
        }
        self.apply_xavier_initialization()
        self._initial_branch_state = {
            k: v.detach().cpu().clone()
            for k, v in self.branches.state_dict().items()
        }

    def encode_features(self, interaction):
        shared_x = torch.flatten(self.shared_embedding(interaction), start_dim=1)
        mask_x = torch.flatten(self.mask_embedding(interaction), start_dim=1)
        domain_ids = interaction[self.DOMAIN].long().view(-1)
        return shared_x, mask_x, domain_ids

    def _compute_branch_logits(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([branch(x).squeeze(-1) for branch in self.branches], dim=1)

    def _compute_mask_logits(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.domain_hypernet(x)
        return torch.stack([head(hidden).squeeze(-1) for head in self.mask_heads], dim=1)

    def _compute_share_mask(
            self,
            mask_logits: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_logits = mask_logits.gather(dim=1, index=domain_ids.view(-1, 1))

        if not self.in_retrain_phase:
            prob = torch.sigmoid(self.current_temperature * selected_logits)
            hard_mask = _hard_gate_from_prob(prob, self.threshold)
            return hard_mask, prob

        if self.retrain_topk is None:
            prob = torch.sigmoid(selected_logits)
            hard_mask = _hard_gate_from_prob(prob, self.threshold)
            return hard_mask, prob

        prob = torch.sigmoid(selected_logits)
        hard_mask = torch.zeros_like(prob)
        topk_ratio = max(0.0, min(1.0, float(self.retrain_topk)))
        for domain_idx in range(self.num_domains):
            domain_indices = torch.nonzero(domain_ids == domain_idx, as_tuple=False).view(-1)
            if domain_indices.numel() == 0:
                continue
            k = max(1, int(round(domain_indices.numel() * topk_ratio)))
            domain_prob = prob[domain_indices, 0]
            topk_indices = torch.topk(domain_prob, k=min(k, domain_indices.numel()), dim=0).indices
            hard_mask[domain_indices[topk_indices], 0] = 1.0
        return hard_mask, prob

    def _forward_dict(
            self,
            shared_x: torch.Tensor,
            mask_x: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        branch_logits = self._compute_branch_logits(shared_x)
        mask_logits = self._compute_mask_logits(mask_x)
        share_mask, share_prob = self._compute_share_mask(mask_logits, domain_ids)
        selected_logits = branch_logits.gather(dim=1, index=domain_ids.view(-1, 1)).squeeze(-1)
        return {
            "branch_logits": branch_logits,
            "mask_logits": mask_logits,
            "share_mask": share_mask,
            "share_prob": share_prob,
            "selected_logits": selected_logits,
        }

    def forward(self, x: torch.Tensor, mask_x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(x, mask_x, domain_ids)["selected_logits"]

    def logits(self, interaction):
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
            share_mask: torch.Tensor,
            share_prob: torch.Tensor,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_branch_loss = []
        for domain_idx in range(self.num_domains):
            per_branch_loss.append(
                F.binary_cross_entropy_with_logits(
                    branch_logits[:, domain_idx],
                    labels,
                    reduction="none",
                )
            )
        per_branch_loss_tensor = torch.stack(per_branch_loss, dim=1)
        own_branch_loss = per_branch_loss_tensor.gather(dim=1, index=domain_ids.view(-1, 1))
        shared_branch_loss = per_branch_loss_tensor.mean(dim=1, keepdim=True)
        main_loss = ((1.0 - share_mask) * own_branch_loss + share_mask * shared_branch_loss).mean()

        if self.in_retrain_phase:
            return main_loss, own_branch_loss.mean(), torch.zeros_like(main_loss)

        share_reg = F.binary_cross_entropy(
            share_prob.clamp(1e-6, 1 - 1e-6),
            torch.ones_like(share_prob),
        )
        total_loss = main_loss + self.lambda1 * share_reg
        return total_loss, own_branch_loss.mean(), share_reg

    def _build_stage_two_model(self):
        return type(self)(
            self.manager,
            self.num_domains,
            **self._two_stage_model_kwargs,
        )

    def _prepare_retrain_phase(self) -> None:
        self.branches.load_state_dict(self._initial_branch_state)
        for module in [self.mask_omni_embedding, self.domain_hypernet, self.mask_heads]:
            for param in module.parameters():
                param.requires_grad = False
        for module in [self.omni_embedding, self.branches]:
            for param in module.parameters():
                param.requires_grad = True

    def _restore_retrain_runtime(self) -> None:
        # Checkpoint loading has already restored the retrain branch weights;
        # only requires_grad flags live outside nn.Module.state_dict().
        for module in [self.mask_omni_embedding, self.domain_hypernet, self.mask_heads]:
            for param in module.parameters():
                param.requires_grad = False
        for module in [self.omni_embedding, self.branches]:
            for param in module.parameters():
                param.requires_grad = True

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        labels = batch_interaction[self.LABEL].float().view(-1)
        shared_x, mask_x, domain_ids = self.encode_features(batch_interaction)
        forward_dict = self._forward_dict(shared_x, mask_x, domain_ids)
        total_loss, own_loss, share_reg = self._phase_loss(
            forward_dict["branch_logits"],
            forward_dict["share_mask"],
            forward_dict["share_prob"],
            labels,
            domain_ids,
        )
        main_loss = total_loss - self.lambda1 * share_reg
        self.lsp.add("task", main_loss)
        self.lsp.add("regularization.share", share_reg, weight=self.lambda1)
        self.lsp.add("diagnostic.own", own_loss, weight=0.0)
        loss = self.lsp.settle(total_loss)


        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()

        if not self.in_retrain_phase:
            self._search_step += 1
            self.current_temperature = min(self.final_temperature, self.current_temperature * self._temperature_growth)
        return loss
