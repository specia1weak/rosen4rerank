from __future__ import annotations

import csv
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from rosen.utils.observatory.analysis import build_sketch, uniform_sample_rows
from rosen.utils.observatory.metrics.spectral import compute_spectrum
from rosen.utils.observatory.port import CapturedObservation, ObservationPort
from rosen.utils.observatory.probe import ModuleProbeRegistry, ModuleProbeSpec
from rosen.utils.observatory.snapshot import ModelSnapshot, SignalSnapshot
from rosen.utils.observatory.store import SnapshotStore


@dataclass(frozen=True)
class HealthMonitorConfig:
    snapshot_every: int = 100
    summary_capacity: int = 1000
    max_sketch_rows: int = 256
    max_sketch_dims: int = 256
    max_metric_vectors: int = 4096
    max_vector_pairs: int = 2048
    capture_first_step: bool = False
    output_jsonl: str = ""
    print_snapshots: bool = False


class HealthMonitor:
    """Trainer-owned, model-agnostic health snapshot coordinator."""

    def __init__(self, config: HealthMonitorConfig | None = None):
        self.config = config or HealthMonitorConfig()
        self.store = SnapshotStore(self.config.summary_capacity)
        self.probes = ModuleProbeRegistry()
        self._snapshot_sequence = 0
        self._attached = False
        self._jsonl_path = Path(self.config.output_jsonl) if self.config.output_jsonl else None

    def observe_module(
        self,
        module_path: str,
        *,
        prefix: str | None = None,
        include_root: bool = True,
        leaf_policy: str = "representation",
        role: str = "hidden",
        metrics: Sequence[str] = ("basic", "rank"),
        domain_routed: bool = False,
    ) -> None:
        self.probes.add(
            ModuleProbeSpec(
                module_path=str(module_path),
                prefix=str(prefix or module_path or "model"),
                include_root=bool(include_root),
                leaf_policy=str(leaf_policy),
                role=str(role),
                metrics=tuple(str(metric) for metric in metrics),
                domain_routed=bool(domain_routed),
            )
        )

    def ensure_attached(self, model) -> None:
        if self._attached:
            return
        port = getattr(model, "obp", None)
        if not isinstance(port, ObservationPort):
            raise TypeError("HealthMonitor requires model.obp to be an ObservationPort.")
        self.probes.attach(model, port)
        self._attached = True

    def should_capture(self, ctx, *, force: bool = False) -> bool:
        if force:
            return True
        step = int(ctx.global_step) + 1
        if self.config.capture_first_step and step == 1:
            return True
        return step % max(1, int(self.config.snapshot_every)) == 0

    @contextmanager
    def capture_scope(self, model, ctx, *, force: bool = False) -> Iterator[None]:
        self.ensure_attached(model)
        port = model.obp
        with port.collecting(enabled=self.should_capture(ctx, force=force), clear=True):
            yield

    @staticmethod
    def _loss_signals(summary: Mapping[str, Any]) -> dict[str, SignalSnapshot]:
        settlements = summary.get("settlements") or {}
        signals: dict[str, SignalSnapshot] = {}
        for settlement_scope, settlement in settlements.items():
            prefix = (
                "loss"
                if settlement_scope == "main"
                else f"loss.{settlement_scope}"
            )
            total_value = float(settlement["total"])
            total_name = f"{prefix}.total"
            signals[total_name] = SignalSnapshot(
                name=total_name,
                kind="loss",
                role="loss",
                scope="point",
                metrics={
                    "raw": total_value,
                    "weight": 1.0,
                    "weighted": total_value,
                    "finite": math.isfinite(total_value),
                },
                metadata={"settlement_scope": str(settlement_scope)},
            )
            for term_name, term in settlement["terms"].items():
                if term_name == "total":
                    continue
                raw = float(term["raw"])
                weight = float(term["weight"])
                weighted = float(term["weighted"])
                signal_name = f"{prefix}.{term_name}"
                signals[signal_name] = SignalSnapshot(
                    name=signal_name,
                    kind="loss",
                    role="loss",
                    scope="point",
                    metrics={
                        "raw": raw,
                        "weight": weight,
                        "weighted": weighted,
                        "finite": math.isfinite(raw) and math.isfinite(weighted),
                    },
                    metadata={"settlement_scope": str(settlement_scope)},
                )
        return signals

    @staticmethod
    def _vector_matrix(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.reshape(-1, 1)
        vector_dim = int(tensor.shape[-1])
        vector_count = math.prod(int(size) for size in tensor.shape[:-1])
        return tensor.reshape(vector_count, vector_dim)

    @staticmethod
    def _batch_matrix(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.reshape(-1, 1)
        batch_size = int(tensor.shape[0])
        flat_dim = math.prod(int(size) for size in tensor.shape[1:])
        return tensor.reshape(batch_size, flat_dim)

    def _vector_collection_metrics(
        self,
        safe_vectors: torch.Tensor,
        vector_energy: torch.Tensor,
    ) -> dict[str, Any]:
        vector_count, vector_dim = int(safe_vectors.size(0)), int(safe_vectors.size(1))
        empty = {
            "coordinate_participation_ratio": 0.0,
            "pairwise_abs_cosine_mean": 0.0,
            "pairwise_abs_cosine_p95": 0.0,
            "pairwise_sample_count": 0,
            "population_effective_rank": 0.0,
            "population_effective_rank_ratio": 0.0,
        }
        if vector_count == 0 or vector_dim == 0:
            return empty

        squared = safe_vectors.square()
        fourth_energy = squared.square().sum(dim=-1)
        participation = torch.where(
            vector_energy > 1e-12,
            vector_energy.square() / fourth_energy.clamp_min(1e-12) / float(vector_dim),
            torch.zeros_like(vector_energy),
        )

        active = safe_vectors[vector_energy > 1e-12]
        active_count = int(active.size(0))
        pair_mean = 0.0
        pair_p95 = 0.0
        pair_count = 0
        if active_count >= 2 and int(self.config.max_vector_pairs) > 0:
            total_pairs = active_count * (active_count - 1) // 2
            pair_count = min(int(self.config.max_vector_pairs), total_pairs)
            if pair_count == total_pairs:
                pair_indices = torch.triu_indices(active_count, active_count, offset=1)
                left, right = pair_indices[0], pair_indices[1]
            else:
                generator = torch.Generator(device="cpu").manual_seed(0)
                left = torch.randint(active_count, (pair_count,), generator=generator)
                offsets = torch.randint(1, active_count, (pair_count,), generator=generator)
                right = torch.remainder(left + offsets, active_count)
            unit = torch.nn.functional.normalize(active, p=2, dim=-1, eps=1e-12)
            pair_cosine = (unit[left] * unit[right]).sum(dim=-1).abs()
            pair_mean = float(pair_cosine.mean().item())
            pair_p95 = float(torch.quantile(pair_cosine, 0.95).item())

        population_rank = 0.0
        population_rank_ratio = 0.0
        if vector_count >= 2:
            sketch = build_sketch(
                safe_vectors,
                max_rows=max(2, int(self.config.max_sketch_rows)),
                max_dims=max(1, int(self.config.max_sketch_dims)),
            )
            spectrum = compute_spectrum(sketch)
            if float(spectrum.energy.sum().item()) > 1e-12:
                population_rank = spectrum.effective_rank
                observable_rank = max(
                    1,
                    min(int(sketch.size(0)) - 1, int(sketch.size(1))),
                )
                population_rank_ratio = population_rank / observable_rank

        return {
            "coordinate_participation_ratio": float(participation.mean().item()),
            "pairwise_abs_cosine_mean": pair_mean,
            "pairwise_abs_cosine_p95": pair_p95,
            "pairwise_sample_count": pair_count,
            "population_effective_rank": population_rank,
            "population_effective_rank_ratio": population_rank_ratio,
        }

    def _tensor_metrics(self, tensor: torch.Tensor, requested: Sequence[str]) -> dict[str, Any]:
        tensor_float = tensor.detach().float()
        batch_matrix = self._batch_matrix(tensor_float)
        full_vectors = self._vector_matrix(tensor_float)
        vector_count, vector_dim = int(full_vectors.size(0)), int(full_vectors.size(1))
        sampled_vectors = uniform_sample_rows(
            full_vectors,
            max(1, int(self.config.max_metric_vectors)),
        ).cpu()
        metric_vector_count = int(sampled_vectors.size(0))
        if sampled_vectors.numel() == 0:
            finite_vectors = torch.ones_like(sampled_vectors, dtype=torch.bool)
            safe_vectors = sampled_vectors
            vector_energy = sampled_vectors.new_empty((metric_vector_count,))
            l2_metrics = {
                "vector_count": vector_count,
                "vector_dim": vector_dim,
                "metric_vector_count": metric_vector_count,
                "l2_norm_mean": 0.0,
                "l2_norm_std": 0.0,
                "l2_norm_min": 0.0,
                "l2_norm_max": 0.0,
            }
        else:
            finite_vectors = torch.isfinite(sampled_vectors)
            safe_vectors = torch.where(
                finite_vectors,
                sampled_vectors,
                torch.zeros_like(sampled_vectors),
            )
            vector_norms = torch.linalg.vector_norm(safe_vectors, ord=2, dim=-1)
            vector_energy = vector_norms.square()
            l2_metrics = {
                "vector_count": vector_count,
                "vector_dim": vector_dim,
                "metric_vector_count": metric_vector_count,
                "l2_norm_mean": float(vector_norms.mean().item()),
                "l2_norm_std": float(vector_norms.std(unbiased=False).item()),
                "l2_norm_min": float(vector_norms.min().item()),
                "l2_norm_max": float(vector_norms.max().item()),
            }
        requested_set = {str(metric).lower() for metric in requested}
        sample_count, flat_dim = int(batch_matrix.size(0)), int(batch_matrix.size(1))
        if safe_vectors.numel() == 0:
            metrics = {
                "sample_count": sample_count,
                "flat_dim": flat_dim,
                "finite_ratio": 1.0,
                **l2_metrics,
            }
            if "vector_collection" in requested_set:
                metrics.update(self._vector_collection_metrics(safe_vectors, vector_energy))
            return metrics
        finite = finite_vectors.reshape(-1)
        safe = safe_vectors.reshape(-1)
        metrics: dict[str, Any] = {
            "sample_count": sample_count,
            "flat_dim": flat_dim,
            "mean": float(safe.mean().item()),
            "std": float(safe.std(unbiased=False).item()),
            "min": float(safe.min().item()),
            "max": float(safe.max().item()),
            "abs_max": float(safe.abs().max().item()),
            "finite_ratio": float(finite.float().mean().item()),
            "zero_ratio": float((safe.abs() < 1e-8).float().mean().item()),
            **l2_metrics,
        }
        if "vector_collection" in requested_set:
            metrics.update(self._vector_collection_metrics(safe_vectors, vector_energy))
        elif "rank" in requested_set or "spectral" in requested_set:
            sampled_batch = uniform_sample_rows(
                batch_matrix,
                max(2, int(self.config.max_sketch_rows)),
            ).cpu()
            sampled_batch = torch.where(
                torch.isfinite(sampled_batch),
                sampled_batch,
                torch.zeros_like(sampled_batch),
            )
            sketch = build_sketch(
                sampled_batch,
                max_rows=max(2, int(self.config.max_sketch_rows)),
                max_dims=max(1, int(self.config.max_sketch_dims)),
            )
            spectrum = compute_spectrum(sketch)
            denominator = max(1, min(int(sketch.size(0)), int(sketch.size(1))))
            metrics.update({
                "effective_rank": spectrum.effective_rank,
                "rank_ratio": spectrum.effective_rank / denominator,
                "participation_ratio": spectrum.participation_ratio,
                "stable_rank": spectrum.stable_rank,
                "top1_energy_ratio": (
                    float((spectrum.energy[:1].sum() / spectrum.total_energy).item())
                    if spectrum.energy.numel() and float(spectrum.total_energy.item()) > 0.0 else 0.0
                ),
            })
        return metrics

    def _trace_signals(self, port: ObservationPort) -> dict[str, SignalSnapshot]:
        trace = port.summary(clear=True)
        signals: dict[str, SignalSnapshot] = {}
        for name, captured in trace.items():
            if not isinstance(captured, CapturedObservation):
                continue
            value = captured.value
            spec = port.spec(name)
            role = spec.role if spec is not None else "tensor"
            requested = spec.metrics if spec is not None else ("basic", "rank")
            metadata = dict(spec.metadata) if spec is not None else {}
            coordinates = dict(metadata.pop("coordinates", {}))
            if isinstance(value, torch.Tensor):
                metadata["shape"] = list(value.shape)
                signals[captured.full_name] = SignalSnapshot(
                    name=captured.full_name,
                    kind="tensor",
                    role=role,
                    scope="point",
                    metrics=self._tensor_metrics(value, requested),
                    coordinates=coordinates,
                    metadata=metadata,
                )
            else:
                signals[captured.full_name] = SignalSnapshot(
                    name=captured.full_name,
                    kind="state",
                    role=role,
                    scope="point",
                    metrics={"value": value},
                    coordinates=coordinates,
                    metadata=metadata,
                )
        return signals

    def _commit(
        self,
        *,
        ctx,
        stage: str,
        phase: str,
        target: str,
        signals: Mapping[str, SignalSnapshot],
        model_mode: str,
        interval_start: int | None = None,
        interval_end: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelSnapshot:
        self._snapshot_sequence += 1
        step = int(ctx.global_step) + (1 if stage == "train" else 0)
        snapshot = ModelSnapshot(
            snapshot_id=f"{stage}:{phase}:{step}:{self._snapshot_sequence}",
            step=step,
            epoch=int(ctx.epoch),
            batch_idx=int(ctx.batch_idx),
            stage=str(stage),
            phase=str(phase),
            target=str(target),
            model_mode=str(model_mode),
            interval_start=interval_start,
            interval_end=interval_end,
            signals=dict(signals),
            metadata=dict(metadata or {}),
        )
        self.store.commit(snapshot)
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot.to_dict(), ensure_ascii=False, default=str) + "\n")
        if self.config.print_snapshots:
            print(
                f"[HealthSnapshot] step={snapshot.step} stage={snapshot.stage} "
                f"phase={snapshot.phase} signals={len(snapshot.signals)}",
                flush=True,
            )
        return snapshot

    def record_train_step(self, model, ctx) -> ModelSnapshot | None:
        if not self.should_capture(ctx):
            model.obp.clear()
            return None
        step = int(ctx.global_step) + 1
        signals = self._trace_signals(model.obp)
        signals.update(self._loss_signals(model.lsp.summary(clear=True)))
        snapshot = self._commit(
            ctx=ctx,
            stage="train",
            phase=getattr(ctx, "phase", "default"),
            target="student",
            model_mode="train",
            signals=signals,
            interval_start=step,
            interval_end=step,
            metadata={},
        )
        return snapshot

    def record_training_cycle(
        self,
        model,
        ctx,
        *,
        state_names: tuple[str, ...] = (),
        phase: str | None = None,
    ) -> ModelSnapshot | None:
        metadata = {
            "model_updated": bool(state_names),
            "updated_state_names": list(state_names),
        }
        return self._record_probe(
            model,
            ctx,
            phase=phase,
            metadata=metadata,
            allow_empty=bool(state_names),
        )

    def record_probe(
        self,
        model,
        ctx,
        *,
        target: str = "student",
        phase: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelSnapshot | None:
        """Commit one explicitly captured model probe outside periodic train steps."""

        return self._record_probe(
            model,
            ctx,
            target=target,
            phase=phase,
            metadata=metadata,
            allow_empty=False,
        )

    def _record_probe(
        self,
        model,
        ctx,
        *,
        target: str = "student",
        phase: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        allow_empty: bool,
    ) -> ModelSnapshot | None:
        trace = self._trace_signals(model.obp)
        trace.update(self._loss_signals(model.lsp.summary(clear=True)))
        if not trace and not allow_empty:
            return None
        return self._commit(
            ctx=ctx,
            stage=str(ctx.stage),
            phase=str(phase or getattr(ctx, "phase", "default")),
            target=str(target),
            model_mode="train" if model.training else "eval",
            signals=trace,
            metadata=metadata,
        )

    def flush(self, model, ctx, *, stage: str = "train_end") -> ModelSnapshot | None:
        signals = self._loss_signals(model.lsp.summary(clear=True))
        if not signals:
            return None
        step = int(ctx.global_step)
        snapshot = self._commit(
            ctx=ctx,
            stage=stage,
            phase=getattr(ctx, "phase", "default"),
            target="student",
            model_mode="eval" if not model.training else "train",
            signals=signals,
            interval_start=step,
            interval_end=step,
        )
        return snapshot

    def export_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "snapshot_id", "step", "epoch", "batch_idx", "stage", "phase", "target",
            "signal", "kind", "role", "scope", "metric", "value", "metadata",
        ]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for snapshot in self.store.list():
                for signal_name, signal in snapshot.signals.items():
                    for metric, value in signal.metrics.items():
                        writer.writerow({
                            "snapshot_id": snapshot.snapshot_id,
                            "step": snapshot.step,
                            "epoch": snapshot.epoch,
                            "batch_idx": snapshot.batch_idx,
                            "stage": snapshot.stage,
                            "phase": snapshot.phase,
                            "target": snapshot.target,
                            "signal": signal_name,
                            "kind": signal.kind,
                            "role": signal.role,
                            "scope": signal.scope,
                            "metric": metric,
                            "value": value,
                            "metadata": json.dumps(signal.metadata, ensure_ascii=False, default=str),
                        })
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "snapshot_sequence": self._snapshot_sequence,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._snapshot_sequence = int(state.get("snapshot_sequence", self._snapshot_sequence))

    def close(self) -> None:
        self.probes.close()
        self._attached = False


__all__ = ["HealthMonitor", "HealthMonitorConfig"]
