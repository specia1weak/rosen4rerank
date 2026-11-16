from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager
from rosen.models.base import BaseModel
from rosen.utils.observatory.audit.metrics import (
    compare_gradients,
    metric_row,
    safe_auc,
    safe_logloss,
    summarize_gradients,
)
from rosen.utils.observatory.audit.reporting import (
    format_value as _fmt,
    rank_value as _rank_value,
    write_csv,
)


def _as_sequence(value: Optional[Sequence | str]) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


class ObservationAuditWorkflow:
    """Post-training ObservationPort sensitivity and gradient-conflict audit."""

    def __init__(
            self,
            model: BaseModel,
            manager: SchemaManager,
            *,
            result_loader: Iterable[Interaction],
            gradient_loader: Iterable[Interaction],
            device: str | torch.device,
            trained_batches: int = 0,
            seed: int = 2026,
            train_limit: str = "external",
            result_split: str = "test",
            gradient_split: str = "train",
            output_dir: Optional[Path | str] = None,
            run_name: str = "obp_audit",
            model_name: str = "",
            domains: Optional[Sequence[int] | str] = None,
            fields: Optional[Sequence[str] | str] = None,
            gradient_modes: Optional[Sequence[str] | str] = None,
            audit_batches: int = 12,
            gradient_balanced_max_per_label: int = 512,
            topk: int = 10,
            write_outputs: bool = True,
    ):
        self.model = model.to(device)
        self.manager = manager
        self.result_loader = result_loader
        self.gradient_loader = gradient_loader
        self.device = torch.device(device)
        self.trained_batches = int(trained_batches)
        self.seed = int(seed)
        self.train_limit = str(train_limit)
        self.result_split = result_split
        self.gradient_split = gradient_split
        self.output_dir = Path(output_dir) if output_dir else None
        self.run_name = run_name
        self.model_name = model_name or type(model).__name__
        self.domains = [int(domain) for domain in _as_sequence(domains)]
        self.fields = [str(field) for field in _as_sequence(fields)]
        self.gradient_modes = [str(mode) for mode in (_as_sequence(gradient_modes) or ["natural", "label_balanced"])]
        self.audit_batches = int(audit_batches)
        self.gradient_balanced_max_per_label = int(gradient_balanced_max_per_label)
        self.topk = int(topk)
        self.write_outputs = bool(write_outputs)
        self.audit_outputs: dict[str, object] = {}
        self.audit_output_paths: dict[str, Path] = {}

    @torch.no_grad()
    def predict_arrays(
            self,
            loader: Iterable[Interaction],
            *,
            split: str,
            cut_name: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        if loader is None:
            raise ValueError("ObservationAuditWorkflow requires a result loader.")

        labels_chunks: list[np.ndarray] = []
        domain_chunks: list[np.ndarray] = []
        logit_chunks: list[np.ndarray] = []
        prob_chunks: list[np.ndarray] = []
        port = getattr(self.model, "obp", None)
        self.model.eval()
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(self.device)
            labels = batch[self.manager.label_field].float().view(-1)
            domain_ids = batch[self.manager.domain_field].long().view(-1)
            cut_ctx = port.cutting(cut_name) if port is not None and hasattr(port, "cutting") else nullcontext()
            with cut_ctx:
                logits = self.model.predict(batch).float().view(-1)
            probs = torch.sigmoid(logits)
            labels_chunks.append(labels.detach().cpu().numpy().astype(np.int8))
            domain_chunks.append(domain_ids.detach().cpu().numpy().astype(np.int16))
            logit_chunks.append(logits.detach().cpu().numpy().astype(np.float32))
            prob_chunks.append(probs.detach().cpu().numpy().astype(np.float32))
            if batch_idx == 1 or batch_idx % 20 == 0:
                tag = "baseline" if cut_name is None else f"cut={cut_name}"
                print(f"[AuditPredict] split={split} {tag} batches={batch_idx}", flush=True)

        return {
            "labels": np.concatenate(labels_chunks),
            "domains": np.concatenate(domain_chunks),
            "logits": np.concatenate(logit_chunks),
            "probs": np.concatenate(prob_chunks),
        }

    def metric_rows_from_arrays(
            self,
            arrays: Mapping[str, np.ndarray],
            *,
            split: str,
    ) -> list[dict[str, object]]:
        labels = arrays["labels"]
        probs = arrays["probs"]
        domain_values = arrays["domains"]
        rows = [metric_row(
            model_name=self.model_name,
            split=split,
            scope="overall",
            labels=labels,
            preds=probs,
            trained_batches=self.trained_batches,
        )]
        for domain in self.domains or sorted(int(item) for item in np.unique(domain_values)):
            mask = domain_values == int(domain)
            rows.append(metric_row(
                model_name=self.model_name,
                split=split,
                scope=f"d{int(domain)}",
                labels=labels[mask],
                preds=probs[mask],
                trained_batches=self.trained_batches,
            ))
        return rows

    def feature_sensitivity(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        metric_rows: list[dict[str, object]] = []
        sensitivity_rows: list[dict[str, object]] = []
        baseline = self.predict_arrays(self.result_loader, split=self.result_split)
        split_metric_rows = self.metric_rows_from_arrays(baseline, split=self.result_split)
        metric_rows.extend(split_metric_rows)
        base_by_scope = {str(row["scope"]): row for row in split_metric_rows}

        port = getattr(self.model, "obp", None)
        cut_names = tuple(self.fields) if self.fields else tuple(port.cut_names() if port is not None else ())
        for field in cut_names:
            cut_arrays = self.predict_arrays(self.result_loader, split=self.result_split, cut_name=field)
            scopes = [("overall", np.ones(baseline["labels"].shape, dtype=bool))]
            scopes.extend((f"d{int(domain)}", baseline["domains"] == int(domain)) for domain in self.domains)
            for scope, mask in scopes:
                labels = baseline["labels"][mask]
                cut_probs = cut_arrays["probs"][mask]
                base_probs = baseline["probs"][mask]
                delta_logits = baseline["logits"][mask] - cut_arrays["logits"][mask]
                delta_probs = base_probs - cut_probs
                base = base_by_scope[scope]
                cut_auc = safe_auc(labels, cut_probs)
                cut_logloss = safe_logloss(labels, cut_probs)
                base_auc = float(base["auc"])
                base_logloss = float(base["logloss"])
                sensitivity_rows.append({
                    "model": self.model_name,
                    "split": self.result_split,
                    "field": field,
                    "scope": scope,
                    "trained_batches": self.trained_batches,
                    "rows": int(labels.size),
                    "baseline_auc": base_auc,
                    "cut_auc": cut_auc,
                    "auc_drop": base_auc - cut_auc if not math.isnan(cut_auc) else math.nan,
                    "baseline_logloss": base_logloss,
                    "cut_logloss": cut_logloss,
                    "logloss_delta": cut_logloss - base_logloss if not math.isnan(cut_logloss) else math.nan,
                    "mean_abs_delta_logit": float(np.abs(delta_logits).mean()) if delta_logits.size else math.nan,
                    "mean_signed_delta_logit": float(delta_logits.mean()) if delta_logits.size else math.nan,
                    "mean_abs_delta_prob": float(np.abs(delta_probs).mean()) if delta_probs.size else math.nan,
                    "mean_signed_delta_prob": float(delta_probs.mean()) if delta_probs.size else math.nan,
                })
        return metric_rows, sensitivity_rows

    @staticmethod
    def sample_domain_mask(
            *,
            domain_ids: torch.Tensor,
            labels: torch.Tensor,
            domain: int,
            mode: str,
            max_per_label: int,
            generator: torch.Generator,
    ) -> torch.Tensor:
        domain_mask = domain_ids.long().view(-1) == int(domain)
        if mode == "natural":
            return domain_mask
        if mode != "label_balanced":
            raise ValueError(f"Unknown gradient mode: {mode!r}.")
        positives = torch.nonzero(domain_mask & (labels.view(-1) > 0.5), as_tuple=False).view(-1)
        negatives = torch.nonzero(domain_mask & (labels.view(-1) <= 0.5), as_tuple=False).view(-1)
        count = min(int(positives.numel()), int(negatives.numel()))
        if max_per_label > 0:
            count = min(count, int(max_per_label))
        balanced = torch.zeros_like(domain_mask)
        if count <= 0:
            return balanced
        pos_perm = torch.randperm(positives.numel(), device=positives.device, generator=generator)[:count]
        neg_perm = torch.randperm(negatives.numel(), device=negatives.device, generator=generator)[:count]
        balanced[positives[pos_perm]] = True
        balanced[negatives[neg_perm]] = True
        return balanced

    @staticmethod
    def slice_interaction(batch: Interaction, mask: torch.Tensor) -> Interaction:
        indices = torch.nonzero(mask, as_tuple=False).view(-1)
        return batch[indices]

    def zero_grad(self) -> None:
        for param in self.model.parameters():
            param.grad = None

    def gradient_conflict(self) -> list[dict[str, object]]:
        if len(self.domains) != 2:
            raise ValueError("ObservationAuditWorkflow gradient conflict expects exactly two domains.")
        left_domain, right_domain = int(self.domains[0]), int(self.domains[1])
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + self.trained_batches)
        rows: list[dict[str, object]] = []
        port = getattr(self.model, "obp", None)
        if port is None or not hasattr(port, "tapping_gradients"):
            return rows

        self.model.eval()
        for batch_idx, batch in enumerate(self.gradient_loader, start=1):
            if self.audit_batches > 0 and batch_idx > self.audit_batches:
                break
            batch = batch.to(self.device)
            labels = batch[self.manager.label_field].float().view(-1)
            domain_ids = batch[self.manager.domain_field].long().view(-1)
            for mode in self.gradient_modes:
                left_mask = self.sample_domain_mask(
                    domain_ids=domain_ids,
                    labels=labels,
                    domain=left_domain,
                    mode=mode,
                    max_per_label=self.gradient_balanced_max_per_label,
                    generator=generator,
                )
                right_mask = self.sample_domain_mask(
                    domain_ids=domain_ids,
                    labels=labels,
                    domain=right_domain,
                    mode=mode,
                    max_per_label=self.gradient_balanced_max_per_label,
                    generator=generator,
                )
                rows_left = int(left_mask.sum().detach().cpu().item())
                rows_right = int(right_mask.sum().detach().cpu().item())
                if rows_left <= 0 or rows_right <= 0:
                    continue

                self.zero_grad()
                with port.tapping_gradients(enabled=True, clear=True):
                    loss_left = self.model.calculate_loss(self.slice_interaction(batch, left_mask))
                    loss_left.backward()
                    left_grads = port.gradient_snapshot(clear=True)

                self.zero_grad()
                with port.tapping_gradients(enabled=True, clear=True):
                    loss_right = self.model.calculate_loss(self.slice_interaction(batch, right_mask))
                    loss_right.backward()
                    right_grads = port.gradient_snapshot(clear=True)

                for row in compare_gradients(left_grads, right_grads):
                    row.update({
                        "model": self.model_name,
                        "split": self.gradient_split,
                        "mode": mode,
                        "batch_idx": batch_idx,
                        "left_domain": left_domain,
                        "right_domain": right_domain,
                        "rows_left": rows_left,
                        "rows_right": rows_right,
                        "ctr_left": float(labels[left_mask].mean().detach().cpu().item()),
                        "ctr_right": float(labels[right_mask].mean().detach().cpu().item()),
                        "loss_left": float(loss_left.detach().cpu().item()),
                        "loss_right": float(loss_right.detach().cpu().item()),
                    })
                    rows.append(row)
                self.zero_grad()
            if batch_idx == 1 or batch_idx % 5 == 0:
                print(f"[AuditGrad] split={self.gradient_split} batches={batch_idx}", flush=True)
        self.zero_grad()
        return rows

    def _output_paths(self) -> dict[str, Path]:
        if self.output_dir is None:
            return {}
        return {
            "metrics": self.output_dir / f"{self.run_name}.metrics.csv",
            "sensitivity": self.output_dir / f"{self.run_name}.sensitivity.csv",
            "gradient_batches": self.output_dir / f"{self.run_name}.gradient_batches.csv",
            "gradient_summary": self.output_dir / f"{self.run_name}.gradient_summary.csv",
            "summary": self.output_dir / f"{self.run_name}.md",
        }

    def make_report(
            self,
            *,
            output_paths: Mapping[str, Path],
            metrics: Sequence[Mapping[str, object]],
            sensitivity: Sequence[Mapping[str, object]],
            gradient_summary: Sequence[Mapping[str, object]],
    ) -> str:
        lines = [
            "# ObservationPort Audit",
            "",
            f"- model: `{self.model_name}`",
            f"- domains: {self.domains}",
            f"- train_limit: `{self.train_limit}`",
            f"- trained_batches: {self.trained_batches}",
            f"- result_split: `{self.result_split}`",
            f"- gradient_split: `{self.gradient_split}`",
            "",
            "## Output Files",
            "",
        ]
        for label, path in output_paths.items():
            lines.append(f"- {label}: `{path}`")

        lines.extend(["", "## Baseline Metrics", ""])
        for row in metrics:
            if row["scope"] != "overall":
                continue
            lines.append(
                f"- {row['split']}: auc={_fmt(row['auc'])} "
                f"logloss={_fmt(row['logloss'])} ctr={_fmt(row['ctr'])} rows={row['rows']}"
            )

        lines.extend(["", f"## Top Feature Sensitivity ({self.result_split})", ""])
        top_sensitivity = [
            row for row in sensitivity
            if row["scope"] == "overall" and row["split"] == self.result_split
        ]
        for row in sorted(top_sensitivity, key=lambda item: _rank_value(item.get("auc_drop"), -math.inf), reverse=True)[: self.topk]:
            lines.append(
                f"- {row['field']}: auc_drop={_fmt(row['auc_drop'])} "
                f"logloss_delta={_fmt(row['logloss_delta'])} "
                f"abs_logit={_fmt(row['mean_abs_delta_logit'])}"
            )

        summary_mode = self.gradient_modes[-1] if self.gradient_modes else ""
        if any(row.get("mode") == summary_mode for row in gradient_summary):
            top_grad = [
                row for row in gradient_summary
                if row["split"] == self.gradient_split and row["mode"] == summary_mode
            ]
        else:
            top_grad = [
                row for row in gradient_summary
                if row["split"] == self.gradient_split
            ]
        lines.extend(["", f"## Most Conflicting Gradient Taps ({self.gradient_split}/{summary_mode})", ""])
        for row in sorted(top_grad, key=lambda item: _rank_value(item.get("global_cosine"), math.inf))[: self.topk]:
            lines.append(
                f"- {row['tap']}: global_cos={_fmt(row['global_cosine'])} "
                f"mean_cos={_fmt(row['cosine_mean'])} "
                f"neg_batch={_fmt(row['negative_batch_ratio'])} "
                f"type={row['tap_type']}"
            )
        return "\n".join(lines) + "\n"

    def run_observation_audit(self) -> dict[str, object]:
        metrics, sensitivity = self.feature_sensitivity()
        gradient_batches = self.gradient_conflict()
        gradient_summary = summarize_gradients(gradient_batches)

        output_paths = self._output_paths()
        if self.write_outputs and output_paths:
            write_csv(output_paths["metrics"], metrics)
            write_csv(output_paths["sensitivity"], sensitivity)
            write_csv(output_paths["gradient_batches"], gradient_batches)
            write_csv(output_paths["gradient_summary"], gradient_summary)
            report = self.make_report(
                output_paths=output_paths,
                metrics=metrics,
                sensitivity=sensitivity,
                gradient_summary=gradient_summary,
            )
            output_paths["summary"].write_text(report, encoding="utf-8")
            print(report, flush=True)

        self.audit_outputs = {
            "metrics": metrics,
            "sensitivity": sensitivity,
            "gradient_batches": gradient_batches,
            "gradient_summary": gradient_summary,
        }
        self.audit_output_paths = output_paths
        return self.audit_outputs

    def run(self):
        return self.run_observation_audit()
