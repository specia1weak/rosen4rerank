from collections import defaultdict
from typing import Dict

import torch
from torch import nn

from rosen.models.utils.common import build_mlp, to_dims
from rosen.models.utils.container import domain_select


class PLELayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = num_domains
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.output_dim = expert_dims[-1]

        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                build_mlp(
                    input_dim,
                    expert_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                for _ in range(num_specific_experts)
            ])
            for _ in range(num_domains)
        ])
        self.shared_experts = nn.ModuleList([
            build_mlp(
                input_dim,
                expert_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(num_shared_experts)
        ])
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_specific_experts + num_shared_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_domains)
        ])
        self.shared_gate = nn.Sequential(
            nn.Linear(input_dim, num_domains * num_specific_experts + num_shared_experts),
            nn.Softmax(dim=-1),
        )
        self.last_task_gates: list[torch.Tensor] = []
        self.last_shared_gate: torch.Tensor | None = None
        self._task_gate_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(self._new_gate_stats)
        self._shared_gate_stats: Dict[tuple[int, int], Dict[str, float]] = defaultdict(self._new_gate_stats)

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

    @staticmethod
    def _accumulate_gate_stats(stats: Dict[str, float], values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().float().reshape(-1)
        stats["batches"] += 1.0
        stats["sample_count"] += float(values.numel())
        stats["gate_sum"] += float(values.sum().item())
        stats["gate_sq_sum"] += float(values.pow(2).sum().item())
        stats["gate_min"] = min(stats["gate_min"], float(values.min().item()))
        stats["gate_max"] = max(stats["gate_max"], float(values.max().item()))

    def reset_gate_stats(self) -> None:
        self._task_gate_stats.clear()
        self._shared_gate_stats.clear()

    def _record_gate_scores(
            self,
            task_gates: list[torch.Tensor],
            shared_gate: torch.Tensor,
            domain_ids: torch.Tensor | None,
    ) -> None:
        if not self.training:
            return
        with torch.no_grad():
            if domain_ids is None:
                domain_ids = torch.full(
                    (shared_gate.shape[0],),
                    -1,
                    dtype=torch.long,
                    device=shared_gate.device,
                )
            domain_ids = domain_ids.detach().long().view(-1).to(device=shared_gate.device)
            for domain_idx, gate_values in enumerate(task_gates):
                gate_values = gate_values.detach().float()
                mask = domain_ids == int(domain_idx)
                if not bool(mask.any().item()):
                    mask = torch.ones(gate_values.shape[0], dtype=torch.bool, device=gate_values.device)
                for expert_idx in range(int(gate_values.shape[1])):
                    self._accumulate_gate_stats(
                        self._task_gate_stats[(int(domain_idx), int(expert_idx))],
                        gate_values[mask, expert_idx],
                    )
            shared_values = shared_gate.detach().float()
            for domain_idx in sorted(int(v) for v in domain_ids.unique().detach().cpu().tolist()):
                mask = domain_ids == int(domain_idx)
                if not bool(mask.any().item()):
                    continue
                for expert_idx in range(int(shared_values.shape[1])):
                    self._accumulate_gate_stats(
                        self._shared_gate_stats[(int(domain_idx), int(expert_idx))],
                        shared_values[mask, expert_idx],
                    )

    def export_gate_rows(self, *, level: int) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for stat_group, stats_map in (
                ("ple_task_gate_domain_expert", self._task_gate_stats),
                ("ple_shared_gate_domain_expert", self._shared_gate_stats),
        ):
            for (domain_idx, expert_idx), stats in sorted(stats_map.items()):
                count = max(1.0, float(stats["sample_count"]))
                mean = float(stats["gate_sum"]) / count
                var = max(0.0, float(stats["gate_sq_sum"]) / count - mean ** 2)
                rows.append({
                    "stat_group": stat_group,
                    "level": int(level),
                    "domain": int(domain_idx),
                    "expert_index": int(expert_idx),
                    "batches": int(stats["batches"]),
                    "sample_count": int(stats["sample_count"]),
                    "gate_mean": mean,
                    "gate_std": var ** 0.5,
                    "gate_min": float(stats["gate_min"]) if stats["gate_min"] != float("inf") else 0.0,
                    "gate_max": float(stats["gate_max"]) if stats["gate_max"] != float("-inf") else 0.0,
                })
        return rows

    def forward(
            self,
            task_inputs: list[torch.Tensor],
            shared_input: torch.Tensor,
            domain_ids: torch.Tensor | None = None,
    ):
        task_outputs = []
        all_specific = []
        shared_outputs = [expert(shared_input) for expert in self.shared_experts]
        task_gates = []

        for domain_idx in range(self.num_domains):
            task_input = task_inputs[domain_idx]
            specific_outputs = [expert(task_input) for expert in self.specific_experts[domain_idx]]
            all_specific.extend(specific_outputs)
            gate = self.task_gates[domain_idx](task_input)
            task_gates.append(gate.detach())
            gate = gate.unsqueeze(-1)
            mix_inputs = torch.stack(specific_outputs + shared_outputs, dim=1)
            task_outputs.append(torch.sum(gate * mix_inputs, dim=1))

        shared_gate = self.shared_gate(shared_input)
        self.last_task_gates = task_gates
        self.last_shared_gate = shared_gate.detach()
        self._record_gate_scores(task_gates, shared_gate, domain_ids)
        shared_gate = shared_gate.unsqueeze(-1)
        shared_mix = torch.stack(all_specific + shared_outputs, dim=1)
        shared_output = torch.sum(shared_gate * shared_mix, dim=1)
        return task_outputs, shared_output


class PLEFeatureGateLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            detach_gate_input: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = num_domains
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.output_dim = expert_dims[-1]
        self.detach_gate_input = bool(detach_gate_input)

        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                build_mlp(
                    input_dim,
                    expert_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                for _ in range(num_specific_experts)
            ])
            for _ in range(num_domains)
        ])
        self.specific_gates = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    build_mlp(
                        input_dim,
                        (*expert_dims[:-1], num_shared_experts * self.output_dim),
                        dropout_rate=dropout_rate,
                        activation=activation,
                        batch_norm=batch_norm,
                    ),
                    nn.Sigmoid(),
                )
                for _ in range(num_specific_experts)
            ])
            for _ in range(num_domains)
        ])
        self.shared_experts = nn.ModuleList([
            build_mlp(
                input_dim,
                expert_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(num_shared_experts)
        ])

    def forward(self, task_inputs: list[torch.Tensor], shared_input: torch.Tensor):
        shared_outputs = torch.stack([expert(shared_input) for expert in self.shared_experts], dim=1)
        shared_output = shared_outputs.sum(dim=1)

        task_outputs = []
        for domain_idx in range(self.num_domains):
            task_input = task_inputs[domain_idx]
            gate_input = task_input.detach() if self.detach_gate_input else task_input
            mixed_specific_outputs = []
            for expert, gate in zip(self.specific_experts[domain_idx], self.specific_gates[domain_idx]):
                specific_output = expert(task_input)
                gate_output = gate(gate_input).reshape(-1, self.num_shared_experts, self.output_dim)
                gated_shared_output = (gate_output * shared_outputs).sum(dim=1)
                mixed_specific_outputs.append(specific_output + gated_shared_output)
            task_outputs.append(torch.stack(mixed_specific_outputs, dim=0).sum(dim=0))
        return task_outputs, shared_output


def select_domain_output(task_outputs: list[torch.Tensor], domain_ids: torch.Tensor) -> torch.Tensor:
    return domain_select(torch.stack(task_outputs, dim=1), domain_ids)
