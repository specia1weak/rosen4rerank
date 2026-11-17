from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from rosen.emb import SchemaManager
from rosen.models.msr.components import DomainTowerHead
from rosen.models.msr.sharedbottom import SharedBottomBackbone, SharedBottomModel
from rosen.models.utils.common import build_mlp


def _new_stats() -> Dict[str, float]:
    return {
        "batches": 0.0,
        "sample_count": 0.0,
        "sum": 0.0,
        "sq_sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def _accumulate(stats: Dict[str, float], values: torch.Tensor, *, sample_count: int | None = None) -> None:
    if values.numel() == 0:
        return
    values = values.detach().float().reshape(-1)
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        return
    values = values[finite]
    stats["batches"] += 1.0
    stats["sample_count"] += float(sample_count if sample_count is not None else values.numel())
    stats["sum"] += float(values.sum().item())
    stats["sq_sum"] += float(values.pow(2).sum().item())
    stats["min"] = min(stats["min"], float(values.min().item()))
    stats["max"] = max(stats["max"], float(values.max().item()))


def _mean_std(stats: Dict[str, float]) -> tuple[float, float]:
    count = max(1.0, float(stats["sample_count"]))
    mean = float(stats["sum"]) / count
    var = max(0.0, float(stats["sq_sum"]) / count - mean ** 2)
    return mean, var ** 0.5


def _effective_rank(values: torch.Tensor) -> float:
    if values.shape[0] < 2:
        return 0.0
    centered = values.detach().float() - values.detach().float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    probs = singular / total.clamp_min(1e-12)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


class ContexturePrototypeSharedBottomModel(SharedBottomModel):
    """
    SharedBottom with a small contexture/prototype auxiliary objective.

    The auxiliary target is a deliberately simple context variable A, such as
    domain, label, or (domain, label). It tests whether forcing the shared
    representation to expose a compatible context geometry helps multi-domain
    CTR, while OBS checks for rank collapse and over-dominant domain geometry.
    """

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Iterable[int] = (256, 128, 64),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            activate_backbone_output: bool = False,
            tower_hidden_dims: Iterable[int] = (128, 64),
            tower_dropout_rate: float = 0.0,
            include_domain_in_input: bool = True,
            contexture_target: str = "domain_label",
            contexture_weight: float = 0.02,
            contexture_temperature: float = 0.2,
            contexture_cov_weight: float = 0.0,
            contexture_project_dim: int = 64,
            contexture_project_hidden: int = 0,
            contexture_class_weights: Iterable[float] | None = None,
            contexture_domain_logit_calibration: bool = False,
            contexture_base_prob_field: str = "contexture_base_prob",
            record_step_interval: int = 50,
            monitor_step_interval: int = 0,
    ):
        del monitor_step_interval
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            activate_backbone_output=activate_backbone_output,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            include_domain_in_input=include_domain_in_input,
        )
        target = str(contexture_target).lower()
        if target not in {
            "domain",
            "label",
            "domain_label",
            "domain_label_split",
            "residual4",
            "domain_residual4",
        }:
            raise ValueError(f"Unknown contexture_target={contexture_target!r}.")
        self.contexture_target = target
        self.contexture_base_prob_field = str(contexture_base_prob_field)
        self.contexture_weight = float(contexture_weight)
        self.contexture_temperature = float(contexture_temperature)
        self.contexture_cov_weight = float(contexture_cov_weight)
        self.contexture_domain_logit_calibration = bool(contexture_domain_logit_calibration)
        self.record_step_interval = max(1, int(record_step_interval or 1))
        self._forward_count = 0
        self._record_this_step = False

        if self.contexture_target == "domain":
            self.num_contexts = int(num_domains)
        elif self.contexture_target == "label":
            self.num_contexts = 2
        elif self.contexture_target == "residual4":
            self.num_contexts = 4
        elif self.contexture_target == "domain_residual4":
            self.num_contexts = int(num_domains) * 4
        elif self.contexture_target == "domain_label_split":
            self.num_contexts = int(num_domains) + 2
        else:
            self.num_contexts = int(num_domains) * 2
        if contexture_class_weights is None:
            class_weights = torch.empty(0, dtype=torch.float32)
        else:
            class_weights = torch.as_tensor(list(contexture_class_weights), dtype=torch.float32)
            if tuple(class_weights.shape) != (int(self.num_contexts),):
                raise ValueError(
                    "contexture_class_weights must have shape "
                    f"({int(self.num_contexts)},), got {tuple(class_weights.shape)}"
                )
        self.register_buffer("contexture_class_weights", class_weights, persistent=True)
        self.register_buffer(
            "contexture_residual_thresholds",
            torch.full((int(num_domains), 2), 0.5, dtype=torch.float32),
            persistent=True,
        )

        project_dim = int(contexture_project_dim or self.backbone.output_dim)
        if contexture_project_hidden and int(contexture_project_hidden) > 0:
            self.context_projector = build_mlp(
                self.backbone.output_dim,
                (int(contexture_project_hidden), project_dim),
                dropout_rate=0.0,
                activation=activation,
                batch_norm=False,
            )
        elif project_dim != int(self.backbone.output_dim):
            self.context_projector = nn.Linear(self.backbone.output_dim, project_dim)
        else:
            self.context_projector = nn.Identity()
        self.context_dim = project_dim
        self.context_prototypes = nn.Parameter(torch.empty(self.num_contexts, self.context_dim))
        nn.init.normal_(self.context_prototypes, mean=0.0, std=0.02)
        if self.contexture_domain_logit_calibration:
            self.contexture_logit_scale = nn.Parameter(torch.ones(int(num_domains)))
            self.contexture_logit_bias = nn.Parameter(torch.zeros(int(num_domains)))
        else:
            self.register_parameter("contexture_logit_scale", None)
            self.register_parameter("contexture_logit_bias", None)

        self._metric_stats: Dict[tuple[str, int, int], Dict[str, float]] = defaultdict(_new_stats)
        self._context_metric_stats: Dict[tuple[str, int, int], Dict[str, float]] = defaultdict(_new_stats)
        self._rank_stats: Dict[tuple[str, int], Dict[str, float]] = defaultdict(_new_stats)
        self._pair_stats: Dict[tuple[str, int], Dict[str, float]] = defaultdict(_new_stats)
        self._calibration_stats: Dict[tuple[str, int], Dict[str, float]] = defaultdict(_new_stats)

    @torch.no_grad()
    def set_contexture_residual_thresholds(self, thresholds: torch.Tensor) -> None:
        expected = (int(self.num_domains), 2)
        if tuple(thresholds.shape) != expected:
            raise ValueError(f"residual thresholds must have shape {expected}, got {tuple(thresholds.shape)}")
        self.contexture_residual_thresholds.copy_(thresholds.detach().float().to(self.contexture_residual_thresholds.device))

    def reset_train_observations(self) -> None:
        self._forward_count = 0
        self._record_this_step = False
        self._metric_stats.clear()
        self._context_metric_stats.clear()
        self._rank_stats.clear()
        self._pair_stats.clear()
        self._calibration_stats.clear()

    def _residual4_buckets(
            self,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
            base_probs: torch.Tensor,
    ) -> torch.Tensor:
        label_ids = labels.long().view(-1).clamp(min=0, max=1)
        domain_ids = domain_ids.long().view(-1)
        base_probs = base_probs.float().view(-1).clamp(min=0.0, max=1.0)
        thresholds = self.contexture_residual_thresholds.to(base_probs.device)
        domain_clamped = domain_ids.clamp(min=0, max=max(0, int(self.num_domains) - 1))
        threshold = thresholds[domain_clamped, label_ids]
        easy_negative = (label_ids == 0) & (base_probs <= threshold)
        negative_surprise = (label_ids == 0) & (base_probs > threshold)
        easy_positive = (label_ids == 1) & (base_probs >= threshold)
        positive_surprise = (label_ids == 1) & (base_probs < threshold)
        buckets = torch.zeros_like(label_ids)
        buckets = torch.where(negative_surprise, torch.ones_like(buckets), buckets)
        buckets = torch.where(easy_positive, torch.full_like(buckets, 2), buckets)
        buckets = torch.where(positive_surprise, torch.full_like(buckets, 3), buckets)
        buckets = torch.where(easy_negative, torch.zeros_like(buckets), buckets)
        return buckets

    def _context_targets(
            self,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
            base_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        label_ids = labels.long().view(-1).clamp(min=0, max=1)
        domain_ids = domain_ids.long().view(-1)
        if self.contexture_target == "domain":
            return domain_ids
        if self.contexture_target == "label":
            return label_ids
        if self.contexture_target == "domain_label_split":
            return domain_ids * 2 + label_ids
        if self.contexture_target in {"residual4", "domain_residual4"}:
            if base_probs is None:
                raise ValueError(
                    f"contexture_target={self.contexture_target!r} requires "
                    f"interaction[{self.contexture_base_prob_field!r}]."
                )
            buckets = self._residual4_buckets(labels, domain_ids, base_probs)
            if self.contexture_target == "residual4":
                return buckets
            return domain_ids * 4 + buckets
        return domain_ids * 2 + label_ids

    def _context_logits(self, projected: torch.Tensor) -> torch.Tensor:
        z = F.normalize(projected.float(), p=2, dim=-1)
        prototypes = F.normalize(self.context_prototypes.float(), p=2, dim=-1)
        return torch.matmul(z, prototypes.t()) / max(self.contexture_temperature, 1e-6)

    def _calibrate_logits(self, logits: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if not self.contexture_domain_logit_calibration:
            return logits
        domain_ids = domain_ids.long().view(-1).clamp(min=0, max=max(0, int(self.num_domains) - 1))
        scale = self.contexture_logit_scale[domain_ids]
        bias = self.contexture_logit_bias[domain_ids]
        return logits.view(-1) * scale + bias

    def _covariance_loss(self, projected: torch.Tensor) -> torch.Tensor:
        if self.contexture_cov_weight <= 0.0 or projected.shape[0] < 2:
            return projected.new_zeros(())
        z = projected.float()
        z = (z - z.mean(dim=0, keepdim=True)) / z.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        cov = torch.matmul(z.t(), z) / max(1, z.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        return off_diag.pow(2).mean()

    def _record_geometry(
            self,
            hidden: torch.Tensor,
            projected: torch.Tensor,
            context_logits: torch.Tensor,
            context_targets: torch.Tensor,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step:
            return
        with torch.no_grad():
            hidden_f = hidden.detach().float()
            projected_f = projected.detach().float()
            labels_i = labels.detach().long().view(-1).clamp(min=0, max=1)
            domains_i = domain_ids.detach().long().view(-1)
            targets_i = context_targets.detach().long().view(-1)
            probs = F.softmax(context_logits.detach().float(), dim=-1)
            pred = probs.argmax(dim=-1)
            losses = F.cross_entropy(context_logits.detach().float(), targets_i, reduction="none")
            norms = torch.linalg.vector_norm(projected_f, ord=2, dim=-1)

            _accumulate(self._rank_stats[("overall_effective_rank", -1)], projected_f.new_tensor([_effective_rank(projected_f)]))
            centered = projected_f - projected_f.mean(dim=0, keepdim=True)
            std = centered.std(dim=0, unbiased=False)
            _accumulate(self._rank_stats[("feature_std", -1)], std)
            if projected_f.shape[0] > 1:
                z = centered / std.clamp_min(1e-4)
                cov = torch.matmul(z.t(), z) / max(1, z.shape[0] - 1)
                off_diag = cov - torch.diag(torch.diagonal(cov))
                _accumulate(self._rank_stats[("cov_offdiag_abs", -1)], off_diag.abs())

            for domain_idx in range(int(self.num_domains)):
                domain_mask = domains_i == domain_idx
                domain_count = int(domain_mask.sum().item())
                if domain_count <= 0:
                    continue
                _accumulate(
                    self._rank_stats[("domain_effective_rank", domain_idx)],
                    projected_f.new_tensor([_effective_rank(projected_f[domain_mask])]),
                )
                for label_idx in (0, 1):
                    mask = domain_mask & (labels_i == label_idx)
                    count = int(mask.sum().item())
                    if count <= 0:
                        continue
                    _accumulate(self._metric_stats[("context_loss", domain_idx, label_idx)], losses[mask], sample_count=count)
                    _accumulate(self._metric_stats[("context_acc", domain_idx, label_idx)], (pred[mask] == targets_i[mask]).float(), sample_count=count)
                    _accumulate(self._metric_stats[("repr_norm", domain_idx, label_idx)], norms[mask], sample_count=count)

                context_mod = 4 if self.contexture_target in {"residual4", "domain_residual4"} else 2
                for context_idx in range(int(self.num_contexts)):
                    if self.contexture_target == "domain_residual4" and (context_idx // 4) != domain_idx:
                        continue
                    if self.contexture_target == "domain_label" and (context_idx // 2) != domain_idx:
                        continue
                    if self.contexture_target == "domain" and context_idx != domain_idx:
                        continue
                    context_mask = domain_mask & (targets_i == context_idx)
                    count = int(context_mask.sum().item())
                    if count <= 0:
                        continue
                    bucket_or_label = int(context_idx % context_mod)
                    _accumulate(
                        self._context_metric_stats[("context_loss", domain_idx, bucket_or_label)],
                        losses[context_mask],
                        sample_count=count,
                    )
                    _accumulate(
                        self._context_metric_stats[("context_acc", domain_idx, bucket_or_label)],
                        (pred[context_mask] == targets_i[context_mask]).float(),
                        sample_count=count,
                    )
                    _accumulate(
                        self._context_metric_stats[("repr_norm", domain_idx, bucket_or_label)],
                        norms[context_mask],
                        sample_count=count,
                    )

                pos = domain_mask & (labels_i == 1)
                neg = domain_mask & (labels_i == 0)
                if bool(pos.any().item()) and bool(neg.any().item()):
                    pos_center = projected_f[pos].mean(dim=0)
                    neg_center = projected_f[neg].mean(dim=0)
                    distance = torch.linalg.vector_norm(pos_center - neg_center, ord=2).view(1)
                    cosine = F.cosine_similarity(pos_center.view(1, -1), neg_center.view(1, -1), dim=-1)
                    _accumulate(self._pair_stats[("label_center_l2", domain_idx)], distance)
                    _accumulate(self._pair_stats[("label_center_cosine", domain_idx)], cosine)

    def _record_split_geometry(
            self,
            hidden: torch.Tensor,
            projected: torch.Tensor,
            context_logits: torch.Tensor,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> None:
        if not self.training or not self._record_this_step:
            return
        with torch.no_grad():
            del hidden
            projected_f = projected.detach().float()
            labels_i = labels.detach().long().view(-1).clamp(min=0, max=1)
            domains_i = domain_ids.detach().long().view(-1)
            domain_logits = context_logits.detach().float()[:, :int(self.num_domains)]
            label_logits = context_logits.detach().float()[:, int(self.num_domains):]
            domain_losses = F.cross_entropy(domain_logits, domains_i, reduction="none")
            label_losses = F.cross_entropy(label_logits, labels_i, reduction="none")
            domain_pred = domain_logits.argmax(dim=-1)
            label_pred = label_logits.argmax(dim=-1)
            norms = torch.linalg.vector_norm(projected_f, ord=2, dim=-1)

            _accumulate(self._rank_stats[("overall_effective_rank", -1)], projected_f.new_tensor([_effective_rank(projected_f)]))
            centered = projected_f - projected_f.mean(dim=0, keepdim=True)
            std = centered.std(dim=0, unbiased=False)
            _accumulate(self._rank_stats[("feature_std", -1)], std)
            if projected_f.shape[0] > 1:
                z = centered / std.clamp_min(1e-4)
                cov = torch.matmul(z.t(), z) / max(1, z.shape[0] - 1)
                off_diag = cov - torch.diag(torch.diagonal(cov))
                _accumulate(self._rank_stats[("cov_offdiag_abs", -1)], off_diag.abs())

            for domain_idx in range(int(self.num_domains)):
                domain_mask = domains_i == domain_idx
                domain_count = int(domain_mask.sum().item())
                if domain_count <= 0:
                    continue
                _accumulate(
                    self._rank_stats[("domain_effective_rank", domain_idx)],
                    projected_f.new_tensor([_effective_rank(projected_f[domain_mask])]),
                )
                _accumulate(
                    self._context_metric_stats[("domain_head_loss", domain_idx, -1)],
                    domain_losses[domain_mask],
                    sample_count=domain_count,
                )
                _accumulate(
                    self._context_metric_stats[("domain_head_acc", domain_idx, -1)],
                    (domain_pred[domain_mask] == domains_i[domain_mask]).float(),
                    sample_count=domain_count,
                )
                for label_idx in (0, 1):
                    mask = domain_mask & (labels_i == label_idx)
                    count = int(mask.sum().item())
                    if count <= 0:
                        continue
                    _accumulate(self._metric_stats[("label_head_loss", domain_idx, label_idx)], label_losses[mask], sample_count=count)
                    _accumulate(
                        self._metric_stats[("label_head_acc", domain_idx, label_idx)],
                        (label_pred[mask] == labels_i[mask]).float(),
                        sample_count=count,
                    )
                    _accumulate(self._metric_stats[("repr_norm", domain_idx, label_idx)], norms[mask], sample_count=count)

                pos = domain_mask & (labels_i == 1)
                neg = domain_mask & (labels_i == 0)
                if bool(pos.any().item()) and bool(neg.any().item()):
                    pos_center = projected_f[pos].mean(dim=0)
                    neg_center = projected_f[neg].mean(dim=0)
                    distance = torch.linalg.vector_norm(pos_center - neg_center, ord=2).view(1)
                    cosine = F.cosine_similarity(pos_center.view(1, -1), neg_center.view(1, -1), dim=-1)
                    _accumulate(self._pair_stats[("label_center_l2", domain_idx)], distance)
                    _accumulate(self._pair_stats[("label_center_cosine", domain_idx)], cosine)

    @torch.no_grad()
    def _record_calibration_batch(self, logits: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._record_this_step or not self.contexture_domain_logit_calibration:
            return
        domain_ids = domain_ids.detach().long().view(-1)
        logits_f = logits.detach().float().view(-1)
        for domain_idx in range(int(self.num_domains)):
            mask = domain_ids == domain_idx
            count = int(mask.sum().item())
            if count <= 0:
                continue
            _accumulate(self._calibration_stats[("calibrated_logit", domain_idx)], logits_f[mask], sample_count=count)

    def _forward_dict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        if self.training:
            self._forward_count += 1
            self._record_this_step = (
                self._forward_count == 1
                or (self._forward_count % self.record_step_interval) == 0
            )
        else:
            self._record_this_step = False
        hidden = self.backbone(x, domain_ids)
        logits = self._calibrate_logits(self.head(hidden, domain_ids), domain_ids)
        self._record_calibration_batch(logits, domain_ids)
        projected = self.context_projector(hidden)
        context_logits = self._context_logits(projected)
        return {
            "hidden": hidden,
            "projected": projected,
            "logits": logits,
            "context_logits": context_logits,
            "domain_ids": domain_ids,
        }

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x, domain_ids)
        return self._calibrate_logits(self.head(hidden, domain_ids), domain_ids)

    def logits(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        out = self._forward_dict(interaction)
        logits = out["logits"].view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if self.lsp.active:
            self.lsp.add("task", loss)
        if self.contexture_weight > 0.0 or self.contexture_cov_weight > 0.0:
            base_probs = None
            if self.contexture_target in {"residual4", "domain_residual4"}:
                base_probs = interaction[self.contexture_base_prob_field]
            if self.contexture_weight > 0.0 and self.contexture_target == "domain_label_split":
                label_ids = labels.long().view(-1).clamp(min=0, max=1)
                domain_ids = out["domain_ids"].long().view(-1)
                domain_logits = out["context_logits"][:, :int(self.num_domains)]
                label_logits = out["context_logits"][:, int(self.num_domains):]
                domain_loss = F.cross_entropy(domain_logits, domain_ids)
                label_loss = F.cross_entropy(label_logits, label_ids)
                context_loss = 0.5 * (domain_loss + label_loss)
                loss = loss + self.contexture_weight * context_loss
                if self.lsp.active:
                    self.lsp.add("context.domain_label", context_loss, weight=self.contexture_weight)
                self._record_split_geometry(
                    out["hidden"],
                    out["projected"],
                    out["context_logits"],
                    labels,
                    out["domain_ids"],
                )
            elif self.contexture_target == "domain_label_split":
                self._record_split_geometry(
                    out["hidden"],
                    out["projected"],
                    out["context_logits"],
                    labels,
                    out["domain_ids"],
                )
            else:
                targets = self._context_targets(labels, out["domain_ids"], base_probs)
                if self.contexture_weight > 0.0:
                    class_weights = None
                    if self.contexture_class_weights.numel() == int(self.num_contexts):
                        class_weights = self.contexture_class_weights.to(out["context_logits"].device)
                    context_loss = F.cross_entropy(out["context_logits"], targets, weight=class_weights)
                    loss = loss + self.contexture_weight * context_loss
                    if self.lsp.active:
                        self.lsp.add("context.classification", context_loss, weight=self.contexture_weight)
                self._record_geometry(
                    out["hidden"],
                    out["projected"],
                    out["context_logits"],
                    targets,
                    labels,
                    out["domain_ids"],
                )
            if self.contexture_cov_weight > 0.0:
                covariance_loss = self._covariance_loss(out["projected"])
                loss = loss + self.contexture_cov_weight * covariance_loss
                if self.lsp.active:
                    self.lsp.add("regularization.context_covariance", covariance_loss, weight=self.contexture_cov_weight)
        return loss

    def export_train_observation_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if self.contexture_class_weights.numel() == int(self.num_contexts):
            weights = self.contexture_class_weights.detach().float().cpu()
            for context_idx, weight in enumerate(weights):
                if self.contexture_target == "domain_label":
                    domain = int(context_idx // 2)
                    label = int(context_idx % 2)
                    context = int(context_idx)
                elif self.contexture_target == "domain_residual4":
                    domain = int(context_idx // 4)
                    label = -1
                    context = int(context_idx % 4)
                elif self.contexture_target == "label":
                    domain = -1
                    label = int(context_idx)
                    context = int(context_idx)
                elif self.contexture_target == "domain":
                    domain = int(context_idx)
                    label = -1
                    context = int(context_idx)
                else:
                    domain = -1
                    label = -1
                    context = int(context_idx)
                rows.append({
                    "stat_group": "contexture_class_weight",
                    "metric": "ce_weight",
                    "contexture_target": self.contexture_target,
                    "context": context,
                    "domain": domain,
                    "label": label,
                    "mean": float(weight.item()),
                })
        if self.contexture_target in {"residual4", "domain_residual4"}:
            thresholds = self.contexture_residual_thresholds.detach().float().cpu()
            for domain_idx in range(thresholds.shape[0]):
                for label_idx in range(thresholds.shape[1]):
                    rows.append({
                        "stat_group": "contexture_residual_threshold",
                        "metric": "base_prob_median",
                        "contexture_target": self.contexture_target,
                        "domain": int(domain_idx),
                        "label": int(label_idx),
                        "mean": float(thresholds[domain_idx, label_idx].item()),
                    })
        for (name, domain_idx, label_idx), stats in sorted(self._metric_stats.items()):
            mean, std = _mean_std(stats)
            rows.append({
                "stat_group": "contexture_aux_metric",
                "metric": name,
                "contexture_target": self.contexture_target,
                "domain": int(domain_idx),
                "label": int(label_idx),
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "mean": mean,
                "std": std,
                "min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        for (name, domain_idx, context_idx), stats in sorted(self._context_metric_stats.items()):
            mean, std = _mean_std(stats)
            rows.append({
                "stat_group": "contexture_context_metric",
                "metric": name,
                "contexture_target": self.contexture_target,
                "domain": int(domain_idx),
                "context": int(context_idx),
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "mean": mean,
                "std": std,
                "min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        for (name, domain_idx), stats in sorted(self._rank_stats.items()):
            mean, std = _mean_std(stats)
            rows.append({
                "stat_group": "contexture_repr_geometry",
                "metric": name,
                "contexture_target": self.contexture_target,
                "domain": int(domain_idx),
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "mean": mean,
                "std": std,
                "min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        for (name, domain_idx), stats in sorted(self._pair_stats.items()):
            mean, std = _mean_std(stats)
            rows.append({
                "stat_group": "contexture_domain_label_geometry",
                "metric": name,
                "contexture_target": self.contexture_target,
                "domain": int(domain_idx),
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "mean": mean,
                "std": std,
                "min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        for (name, domain_idx), stats in sorted(self._calibration_stats.items()):
            mean, std = _mean_std(stats)
            rows.append({
                "stat_group": "contexture_logit_calibration",
                "metric": name,
                "contexture_target": self.contexture_target,
                "domain": int(domain_idx),
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "mean": mean,
                "std": std,
                "min": float(stats["min"]) if stats["min"] != float("inf") else 0.0,
                "max": float(stats["max"]) if stats["max"] != float("-inf") else 0.0,
            })
        with torch.no_grad():
            if self.contexture_domain_logit_calibration:
                for domain_idx in range(int(self.num_domains)):
                    rows.append({
                        "stat_group": "contexture_logit_calibration",
                        "metric": "scale_param",
                        "contexture_target": self.contexture_target,
                        "domain": int(domain_idx),
                        "mean": float(self.contexture_logit_scale[domain_idx].detach().float().item()),
                    })
                    rows.append({
                        "stat_group": "contexture_logit_calibration",
                        "metric": "bias_param",
                        "contexture_target": self.contexture_target,
                        "domain": int(domain_idx),
                        "mean": float(self.contexture_logit_bias[domain_idx].detach().float().item()),
                    })
            prototypes = F.normalize(self.context_prototypes.detach().float(), p=2, dim=-1)
            proto_norm = torch.linalg.vector_norm(self.context_prototypes.detach().float(), ord=2, dim=-1)
            for idx, norm in enumerate(proto_norm):
                if self.contexture_target == "domain_label":
                    domain = int(idx // 2)
                    label = int(idx % 2)
                    context = int(idx)
                elif self.contexture_target == "domain_label_split":
                    if idx < int(self.num_domains):
                        domain = int(idx)
                        label = -1
                        context = int(idx)
                    else:
                        domain = -1
                        label = int(idx - int(self.num_domains))
                        context = int(idx)
                elif self.contexture_target == "domain_residual4":
                    domain = int(idx // 4)
                    label = -1
                    context = int(idx % 4)
                elif self.contexture_target == "residual4":
                    domain = -1
                    label = -1
                    context = int(idx)
                elif self.contexture_target == "label":
                    domain = -1
                    label = int(idx)
                    context = int(idx)
                else:
                    domain = int(idx)
                    label = -1
                    context = int(idx)
                rows.append({
                    "stat_group": "contexture_prototype",
                    "metric": "prototype_norm",
                    "contexture_target": self.contexture_target,
                    "context": context,
                    "domain": domain,
                    "label": label,
                    "mean": float(norm.item()),
                })
            if prototypes.shape[0] > 1:
                cosine = torch.matmul(prototypes, prototypes.t())
                for i in range(prototypes.shape[0]):
                    for j in range(i + 1, prototypes.shape[0]):
                        rows.append({
                            "stat_group": "contexture_prototype_pair",
                            "metric": "prototype_cosine",
                            "contexture_target": self.contexture_target,
                            "context_i": int(i),
                            "context_j": int(j),
                            "mean": float(cosine[i, j].item()),
                        })
        return rows
