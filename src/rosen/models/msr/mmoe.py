from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from rosen.emb import SchemaManager
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.utils.activation import activation_layer
from rosen.models.utils.common import build_mlp, to_dims, build_gate


class MMoEBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            gate_dims=None,
            num_experts: Optional[int] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        gate_dims = to_dims(gate_dims, default=(input_dim, input_dim))
        num_experts = int(num_experts or (num_domains + 1))
        self.num_domains = int(num_domains)
        self.output_dim = int(expert_dims[-1])
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            build_mlp(
                input_dim,
                expert_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_experts)
        ])
        self.gates = nn.ModuleList([
            build_gate(
                input_dim,
                gate_dims,
                self.num_experts,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(num_domains)
        ])
        self._gate_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(self._new_gate_stats)
        self._gate_summary: Dict[int, Dict[str, float]] = defaultdict(self._new_gate_stats)

    @staticmethod
    def _new_gate_stats() -> Dict[str, float]:
        return {
            "batches": 0.0,
            "sample_count": 0.0,
            "gate_sum": 0.0,
            "gate_sq_sum": 0.0,
            "gate_min": float("inf"),
            "gate_max": float("-inf"),
        }

    def expert_role(self, expert_idx: int) -> str:
        del expert_idx
        return "expert"

    def reset_gate_stats(self) -> None:
        self._gate_stats.clear()
        self._gate_summary.clear()

    def _record_gate_scores(self, gate: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or gate.numel() == 0:
            return
        with torch.no_grad():
            gate_values = gate.detach().float()
            domain_values = domain_ids.detach().long().view(-1).to(device=gate_values.device)
            for expert_idx in range(self.num_experts):
                values = gate_values[:, expert_idx]
                summary = self._gate_summary[expert_idx]
                summary["batches"] += 1.0
                summary["sample_count"] += float(values.numel())
                summary["gate_sum"] += float(values.sum().item())
                summary["gate_sq_sum"] += float(values.pow(2).sum().item())
                summary["gate_min"] = min(summary["gate_min"], float(values.min().item()))
                summary["gate_max"] = max(summary["gate_max"], float(values.max().item()))
            for domain_idx in range(self.num_domains):
                mask = domain_values == domain_idx
                count = int(mask.sum().item())
                if count <= 0:
                    continue
                domain_gate = gate_values[mask]
                for expert_idx in range(self.num_experts):
                    values = domain_gate[:, expert_idx]
                    stats = self._gate_stats[(domain_idx, expert_idx)]
                    stats["batches"] += 1.0
                    stats["sample_count"] += float(count)
                    stats["gate_sum"] += float(values.sum().item())
                    stats["gate_sq_sum"] += float(values.pow(2).sum().item())
                    stats["gate_min"] = min(stats["gate_min"], float(values.min().item()))
                    stats["gate_max"] = max(stats["gate_max"], float(values.max().item()))

    def export_gate_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for expert_idx, stats in sorted(self._gate_summary.items()):
            count = max(1.0, float(stats["sample_count"]))
            mean = float(stats["gate_sum"]) / count
            var = max(0.0, float(stats["gate_sq_sum"]) / count - mean ** 2)
            rows.append({
                "stat_group": "mmoe_gate_expert",
                "domain": -1,
                "expert_index": int(expert_idx),
                "expert_role": self.expert_role(expert_idx),
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "gate_mean": mean,
                "gate_std": var ** 0.5,
                "gate_min": float(stats["gate_min"]) if stats["gate_min"] != float("inf") else 0.0,
                "gate_max": float(stats["gate_max"]) if stats["gate_max"] != float("-inf") else 0.0,
            })
        for (domain_idx, expert_idx), stats in sorted(self._gate_stats.items()):
            count = max(1.0, float(stats["sample_count"]))
            mean = float(stats["gate_sum"]) / count
            var = max(0.0, float(stats["gate_sq_sum"]) / count - mean ** 2)
            rows.append({
                "stat_group": "mmoe_gate_domain_expert",
                "domain": int(domain_idx),
                "expert_index": int(expert_idx),
                "expert_role": self.expert_role(expert_idx),
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "gate_mean": mean,
                "gate_std": var ** 0.5,
                "gate_min": float(stats["gate_min"]) if stats["gate_min"] != float("inf") else 0.0,
                "gate_max": float(stats["gate_max"]) if stats["gate_max"] != float("-inf") else 0.0,
            })
        return rows

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_scores = torch.stack([gate(x) for gate in self.gates], dim=1)
        batch_indices = torch.arange(x.size(0), device=x.device)
        gate = gate_scores[batch_indices, domain_ids]
        self._record_gate_scores(gate, domain_ids)
        return torch.sum(gate.unsqueeze(-1) * expert_outs, dim=1)


def _domain_l2_balance_gradient(
        grad: torch.Tensor,
        domain_ids: torch.Tensor,
        num_domains: int,
        *,
        eps: float = 1e-12,
        strength: float = 1.0,
        min_domain_samples: int = 1,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    if grad.numel() == 0:
        return grad, []

    domain_ids = domain_ids.to(device=grad.device).long().view(-1)
    if domain_ids.numel() != grad.size(0):
        raise ValueError(
            "domain_ids must have the same batch dimension as the balanced expert output gradient, "
            f"got {domain_ids.numel()} and {grad.size(0)}."
        )

    present: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for domain_idx in range(int(num_domains)):
        mask = domain_ids == domain_idx
        count = int(mask.sum().item())
        if count < int(min_domain_samples):
            continue
        norm = grad[mask].detach().float().norm(p=2)
        present.append((domain_idx, mask, norm))
    if len(present) < 2:
        rows = [
            {
                "domain": float(domain_idx),
                "count": float(int(mask.sum().item())),
                "pre_l2": float(norm.item()),
                "target_l2": float(norm.item()),
                "scale": 1.0,
                "post_l2": float(norm.item()),
                "present_domains": float(len(present)),
            }
            for domain_idx, mask, norm in present
        ]
        return grad, rows

    target = torch.stack([norm for _, _, norm in present]).mean()
    scale_strength = max(0.0, float(strength))
    scaled_grad = grad.clone()
    rows: list[dict[str, float]] = []
    for domain_idx, mask, norm in present:
        if float(norm.item()) <= float(eps):
            scale = norm.new_tensor(1.0)
        else:
            exact_scale = target / norm.clamp_min(float(eps))
            scale = 1.0 + scale_strength * (exact_scale - 1.0)
        scaled_grad[mask] = grad[mask] * scale.to(dtype=grad.dtype)
        post_l2 = norm * scale.detach().float().abs()
        rows.append({
            "domain": float(domain_idx),
            "count": float(int(mask.sum().item())),
            "pre_l2": float(norm.item()),
            "target_l2": float(target.item()),
            "scale": float(scale.detach().float().item()),
            "post_l2": float(post_l2.item()),
            "present_domains": float(len(present)),
        })
    return scaled_grad, rows


class GradientBalancedMMoEBackbone(MMoEBackbone):
    """
    MMoE backbone with a subset of experts protected by domain-wise gradient L2 balancing.

    Forward values and gates are identical to ordinary MMoE. During backward, each
    balanced expert output receives per-domain gradient slices rescaled so every
    present domain has the same total L2 norm, namely the batch mean L2 over
    present domains.
    """

    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims=None,
            gate_dims=None,
            num_normal_experts: int = 2,
            num_balanced_experts: int = 2,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            balance_strength: float = 1.0,
            balance_eps: float = 1e-12,
            min_domain_samples: int = 1,
            balance_gate_gradients: bool = False,
            gate_balance_scope: str = "all",
            gate_balance_strength: Optional[float] = None,
    ):
        self.num_normal_experts = max(0, int(num_normal_experts))
        self.num_balanced_experts = max(0, int(num_balanced_experts))
        if self.num_normal_experts + self.num_balanced_experts <= 0:
            raise ValueError("GradientBalancedMMoEBackbone requires at least one expert.")
        super().__init__(
            input_dim=input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            gate_dims=gate_dims,
            num_experts=self.num_normal_experts + self.num_balanced_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.num_domains = int(num_domains)
        self.balance_strength = float(balance_strength)
        self.balance_eps = float(balance_eps)
        self.min_domain_samples = max(1, int(min_domain_samples))
        self.balance_gate_gradients = bool(balance_gate_gradients)
        self.gate_balance_scope = str(gate_balance_scope or "all").lower()
        if self.gate_balance_scope not in {"all", "balanced"}:
            raise ValueError("gate_balance_scope must be 'all' or 'balanced'.")
        self.gate_balance_strength = (
            float(self.balance_strength)
            if gate_balance_strength is None
            else float(gate_balance_strength)
        )
        self.balanced_expert_start = self.num_normal_experts
        self._grad_balance_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(self._new_stats)
        self._grad_balance_summary: Dict[int, Dict[str, float]] = defaultdict(self._new_summary_stats)
        self._grad_balance_hook_calls = 0
        self._gate_grad_balance_stats: Dict[int, Dict[str, float]] = defaultdict(self._new_stats)
        self._gate_grad_balance_summary: Dict[str, float] = self._new_summary_stats()
        self._gate_grad_balance_hook_calls = 0

    def expert_role(self, expert_idx: int) -> str:
        return "balanced" if int(expert_idx) >= self.balanced_expert_start else "normal"

    @staticmethod
    def _new_stats() -> Dict[str, float]:
        return {
            "batches": 0.0,
            "sample_count": 0.0,
            "pre_l2_sum": 0.0,
            "pre_l2_sq_sum": 0.0,
            "target_l2_sum": 0.0,
            "post_l2_sum": 0.0,
            "scale_sum": 0.0,
            "scale_sq_sum": 0.0,
            "scale_min": float("inf"),
            "scale_max": float("-inf"),
            "present_domains_sum": 0.0,
        }

    @staticmethod
    def _new_summary_stats() -> Dict[str, float]:
        return {
            "batches": 0.0,
            "pre_cv_sum": 0.0,
            "post_cv_sum": 0.0,
            "scale_cv_sum": 0.0,
            "present_domains_sum": 0.0,
        }

    def reset_grad_balance_stats(self) -> None:
        self._grad_balance_stats.clear()
        self._grad_balance_summary.clear()
        self._grad_balance_hook_calls = 0
        self._gate_grad_balance_stats.clear()
        self._gate_grad_balance_summary = self._new_summary_stats()
        self._gate_grad_balance_hook_calls = 0

    def reset_gate_stats(self) -> None:
        super().reset_gate_stats()

    def _record_grad_balance_rows(self, expert_idx: int, rows: list[dict[str, float]]) -> None:
        if not rows:
            return
        self._grad_balance_hook_calls += 1
        pre_values = []
        post_values = []
        scale_values = []
        for row in rows:
            domain_idx = int(row["domain"])
            stats = self._grad_balance_stats[(int(expert_idx), domain_idx)]
            stats["batches"] += 1.0
            stats["sample_count"] += float(row["count"])
            stats["pre_l2_sum"] += float(row["pre_l2"])
            stats["pre_l2_sq_sum"] += float(row["pre_l2"]) ** 2
            stats["target_l2_sum"] += float(row["target_l2"])
            stats["post_l2_sum"] += float(row["post_l2"])
            stats["scale_sum"] += float(row["scale"])
            stats["scale_sq_sum"] += float(row["scale"]) ** 2
            stats["scale_min"] = min(stats["scale_min"], float(row["scale"]))
            stats["scale_max"] = max(stats["scale_max"], float(row["scale"]))
            stats["present_domains_sum"] += float(row["present_domains"])
            pre_values.append(float(row["pre_l2"]))
            post_values.append(float(row["post_l2"]))
            scale_values.append(float(row["scale"]))

        summary = self._grad_balance_summary[int(expert_idx)]
        summary["batches"] += 1.0
        summary["present_domains_sum"] += float(rows[0]["present_domains"])
        summary["pre_cv_sum"] += self._cv(pre_values)
        summary["post_cv_sum"] += self._cv(post_values)
        summary["scale_cv_sum"] += self._cv(scale_values)

    def _record_gate_grad_balance_rows(self, rows: list[dict[str, float]]) -> None:
        if not rows:
            return
        self._gate_grad_balance_hook_calls += 1
        pre_values = []
        post_values = []
        scale_values = []
        for row in rows:
            domain_idx = int(row["domain"])
            stats = self._gate_grad_balance_stats[domain_idx]
            stats["batches"] += 1.0
            stats["sample_count"] += float(row["count"])
            stats["pre_l2_sum"] += float(row["pre_l2"])
            stats["pre_l2_sq_sum"] += float(row["pre_l2"]) ** 2
            stats["target_l2_sum"] += float(row["target_l2"])
            stats["post_l2_sum"] += float(row["post_l2"])
            stats["scale_sum"] += float(row["scale"])
            stats["scale_sq_sum"] += float(row["scale"]) ** 2
            stats["scale_min"] = min(stats["scale_min"], float(row["scale"]))
            stats["scale_max"] = max(stats["scale_max"], float(row["scale"]))
            stats["present_domains_sum"] += float(row["present_domains"])
            pre_values.append(float(row["pre_l2"]))
            post_values.append(float(row["post_l2"]))
            scale_values.append(float(row["scale"]))

        summary = self._gate_grad_balance_summary
        summary["batches"] += 1.0
        summary["present_domains_sum"] += float(rows[0]["present_domains"])
        summary["pre_cv_sum"] += self._cv(pre_values)
        summary["post_cv_sum"] += self._cv(post_values)
        summary["scale_cv_sum"] += self._cv(scale_values)

    @staticmethod
    def _cv(values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        tensor = torch.tensor(values, dtype=torch.float32)
        mean = tensor.mean().abs().clamp_min(1e-12)
        return float((tensor.std(unbiased=False) / mean).item())

    def _tap_balanced_expert(
            self,
            expert_idx: int,
            values: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or not values.requires_grad or self.balance_strength <= 0.0:
            return values
        detached_domains = domain_ids.detach()

        def hook(grad: torch.Tensor) -> torch.Tensor:
            scaled_grad, rows = _domain_l2_balance_gradient(
                grad,
                detached_domains,
                self.num_domains,
                eps=self.balance_eps,
                strength=self.balance_strength,
                min_domain_samples=self.min_domain_samples,
            )
            self._record_grad_balance_rows(expert_idx, rows)
            return scaled_grad

        values.register_hook(hook)
        return values

    def _tap_gate(self, gate: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if (
                not self.training
                or not gate.requires_grad
                or not self.balance_gate_gradients
                or self.gate_balance_strength <= 0.0
        ):
            return gate
        if self.gate_balance_scope == "balanced" and self.num_balanced_experts <= 0:
            return gate
        detached_domains = domain_ids.detach()

        def hook(grad: torch.Tensor) -> torch.Tensor:
            if self.gate_balance_scope == "all":
                balanced_grad, rows = _domain_l2_balance_gradient(
                    grad,
                    detached_domains,
                    self.num_domains,
                    eps=self.balance_eps,
                    strength=self.gate_balance_strength,
                    min_domain_samples=self.min_domain_samples,
                )
                self._record_gate_grad_balance_rows(rows)
                return balanced_grad

            selected_grad = grad[:, self.balanced_expert_start:]
            balanced_selected, rows = _domain_l2_balance_gradient(
                selected_grad,
                detached_domains,
                self.num_domains,
                eps=self.balance_eps,
                strength=self.gate_balance_strength,
                min_domain_samples=self.min_domain_samples,
            )
            self._record_gate_grad_balance_rows(rows)
            balanced_grad = grad.clone()
            balanced_grad[:, self.balanced_expert_start:] = balanced_selected
            return balanced_grad

        gate.register_hook(hook)
        return gate

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        expert_values = []
        for expert_idx, expert in enumerate(self.experts):
            values = expert(x)
            if expert_idx >= self.balanced_expert_start:
                values = self._tap_balanced_expert(expert_idx, values, domain_ids)
            expert_values.append(values)
        expert_outs = torch.stack(expert_values, dim=1)
        gate_scores = torch.stack([gate(x) for gate in self.gates], dim=1)
        batch_indices = torch.arange(x.size(0), device=x.device)
        gate = gate_scores[batch_indices, domain_ids]
        self._record_gate_scores(gate, domain_ids)
        gate = self._tap_gate(gate, domain_ids)
        return torch.sum(gate.unsqueeze(-1) * expert_outs, dim=1)

    def export_grad_balance_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (expert_idx, domain_idx), stats in sorted(self._grad_balance_stats.items()):
            batches = max(1.0, float(stats["batches"]))
            scale_mean = float(stats["scale_sum"]) / batches
            scale_var = max(0.0, float(stats["scale_sq_sum"]) / batches - scale_mean ** 2)
            pre_mean = float(stats["pre_l2_sum"]) / batches
            pre_sq_mean = float(stats["pre_l2_sq_sum"]) / batches
            pre_var = max(0.0, pre_sq_mean - pre_mean ** 2)
            target_mean = float(stats["target_l2_sum"]) / batches
            rows.append({
                "stat_group": "mmoe_grad_balance_domain",
                "domain": int(domain_idx),
                "expert_index": int(expert_idx),
                "expert_role": "balanced",
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "pre_l2_mean": pre_mean,
                "pre_l2_std": pre_var ** 0.5,
                "target_l2_mean": target_mean,
                "post_l2_mean": float(stats["post_l2_sum"]) / batches,
                "scale_mean": scale_mean,
                "scale_std": scale_var ** 0.5,
                "scale_min": float(stats["scale_min"]) if stats["scale_min"] != float("inf") else 1.0,
                "scale_max": float(stats["scale_max"]) if stats["scale_max"] != float("-inf") else 1.0,
                "pre_over_target": pre_mean / max(target_mean, 1e-12),
                "present_domains_mean": float(stats["present_domains_sum"]) / batches,
            })
        for expert_idx, stats in sorted(self._grad_balance_summary.items()):
            batches = max(1.0, float(stats["batches"]))
            rows.append({
                "stat_group": "mmoe_grad_balance_summary",
                "domain": -1,
                "expert_index": int(expert_idx),
                "expert_role": "balanced",
                "batches": int(stats["batches"]),
                "hook_calls": int(self._grad_balance_hook_calls),
                "pre_l2_cv_mean": float(stats["pre_cv_sum"]) / batches,
                "post_l2_cv_mean": float(stats["post_cv_sum"]) / batches,
                "scale_cv_mean": float(stats["scale_cv_sum"]) / batches,
                "present_domains_mean": float(stats["present_domains_sum"]) / batches,
            })
        gate_role = f"gate_{self.gate_balance_scope}"
        for domain_idx, stats in sorted(self._gate_grad_balance_stats.items()):
            batches = max(1.0, float(stats["batches"]))
            scale_mean = float(stats["scale_sum"]) / batches
            scale_var = max(0.0, float(stats["scale_sq_sum"]) / batches - scale_mean ** 2)
            pre_mean = float(stats["pre_l2_sum"]) / batches
            pre_sq_mean = float(stats["pre_l2_sq_sum"]) / batches
            pre_var = max(0.0, pre_sq_mean - pre_mean ** 2)
            target_mean = float(stats["target_l2_sum"]) / batches
            rows.append({
                "stat_group": "mmoe_gate_grad_balance_domain",
                "domain": int(domain_idx),
                "expert_index": -1,
                "expert_role": gate_role,
                "batches": int(stats["batches"]),
                "sample_count": int(stats["sample_count"]),
                "pre_l2_mean": pre_mean,
                "pre_l2_std": pre_var ** 0.5,
                "target_l2_mean": target_mean,
                "post_l2_mean": float(stats["post_l2_sum"]) / batches,
                "scale_mean": scale_mean,
                "scale_std": scale_var ** 0.5,
                "scale_min": float(stats["scale_min"]) if stats["scale_min"] != float("inf") else 1.0,
                "scale_max": float(stats["scale_max"]) if stats["scale_max"] != float("-inf") else 1.0,
                "pre_over_target": pre_mean / max(target_mean, 1e-12),
                "present_domains_mean": float(stats["present_domains_sum"]) / batches,
            })
        if self._gate_grad_balance_hook_calls > 0:
            stats = self._gate_grad_balance_summary
            batches = max(1.0, float(stats["batches"]))
            rows.append({
                "stat_group": "mmoe_gate_grad_balance_summary",
                "domain": -1,
                "expert_index": -1,
                "expert_role": gate_role,
                "batches": int(stats["batches"]),
                "hook_calls": int(self._gate_grad_balance_hook_calls),
                "pre_l2_cv_mean": float(stats["pre_cv_sum"]) / batches,
                "post_l2_cv_mean": float(stats["post_cv_sum"]) / batches,
                "scale_cv_mean": float(stats["scale_cv_sum"]) / batches,
                "present_domains_mean": float(stats["present_domains_sum"]) / batches,
            })
        return rows


class MMoEModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (256, 128, 64),
            num_experts: int = 6,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            activate_backbone_output: bool = False,
            tower_hidden_dims: Iterable[int] = (128, 64),
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

        self.backbone = MMoEBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            gate_dims=tower_hidden_dims,
            num_experts=num_experts,
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
        self.apply_xavier_initialization()

    def encode_features(self, interaction):
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
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def reset_train_observations(self) -> None:
        self.backbone.reset_gate_stats()

    def export_train_observation_rows(self) -> list[dict[str, object]]:
        return self.backbone.export_gate_rows()


class GradientBalancedMMoEModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (256, 128, 64),
            num_normal_experts: int = 2,
            num_balanced_experts: int = 2,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            activate_backbone_output: bool = False,
            tower_hidden_dims: Iterable[int] = (128, 64),
            tower_dropout_rate: float = 0.0,
            include_domain_in_input: bool = True,
            balance_strength: float = 1.0,
            balance_eps: float = 1e-12,
            min_domain_samples: int = 1,
            balance_gate_gradients: bool = False,
            gate_balance_scope: str = "all",
            gate_balance_strength: Optional[float] = None,
            monitor_step_interval: int = 0,
            record_step_interval: int = 0,
    ):
        del record_step_interval
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.include_domain_in_input = bool(include_domain_in_input)
        self.input_view = (
            self.omni_embedding.whole if self.include_domain_in_input
            else self.omni_embedding.whole_without_domain
        )
        self.input_dim = self.input_view.embedding_dim
        self.monitor_step_interval = max(0, int(monitor_step_interval or 0))
        self._loss_calls = 0

        self.backbone = GradientBalancedMMoEBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            gate_dims=tower_hidden_dims,
            num_normal_experts=num_normal_experts,
            num_balanced_experts=num_balanced_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            balance_strength=balance_strength,
            balance_eps=balance_eps,
            min_domain_samples=min_domain_samples,
            balance_gate_gradients=balance_gate_gradients,
            gate_balance_scope=gate_balance_scope,
            gate_balance_strength=gate_balance_strength,
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
        self.apply_xavier_initialization()

    def encode_features(self, interaction):
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
        labels = interaction[self.LABEL].float().view(-1)
        logits = self.predict(interaction).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self._loss_calls += 1
        if self.monitor_step_interval > 0 and self._loss_calls % self.monitor_step_interval == 0:
            rows = self.backbone.export_grad_balance_rows()
            summary = [row for row in rows if row.get("stat_group") == "mmoe_grad_balance_summary"]
            if summary:
                text = " ".join(
                    f"e{row['expert_index']}:pre_cv={row['pre_l2_cv_mean']:.3f}"
                    f"/post_cv={row['post_l2_cv_mean']:.3f}"
                    for row in summary
                )
                print(f"[GradientBalancedMMoE] step={self._loss_calls} {text}", flush=True)
        return loss

    def reset_train_observations(self) -> None:
        self.backbone.reset_gate_stats()
        self.backbone.reset_grad_balance_stats()

    def export_train_observation_rows(self) -> list[dict[str, object]]:
        return self.backbone.export_gate_rows() + self.backbone.export_grad_balance_rows()
