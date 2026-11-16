from __future__ import annotations

from typing import Optional

import torch


def flatten_metric_dict(metrics: Optional[dict]) -> dict[str, float]:
    if not metrics:
        return {}
    flat = {}
    for evaluator_name, values in metrics.items():
        if not isinstance(values, dict):
            continue
        for metric_name, metric_value in values.items():
            try:
                flat[f"{evaluator_name}.{metric_name}"] = float(metric_value)
            except (TypeError, ValueError):
                continue
    return flat


def metric_change_name(metric_name: str) -> str:
    lowered = metric_name.lower()
    if lowered.endswith(".loss") or lowered.endswith(".logloss") or lowered.endswith(".mse"):
        return f"{metric_name}_delta"
    return f"{metric_name}_drop"


def metric_changes(baseline: Optional[dict], candidate: Optional[dict]) -> dict[str, float]:
    base = flatten_metric_dict(baseline)
    current = flatten_metric_dict(candidate)
    changes = {}
    for name, base_value in base.items():
        if name not in current:
            continue
        current_value = current[name]
        change_name = metric_change_name(name)
        if change_name.endswith("_delta"):
            changes[change_name] = current_value - base_value
        else:
            changes[change_name] = base_value - current_value
    return changes


def accumulate_watch_stats(storage: dict, watch_trace: dict) -> None:
    for name, value in watch_trace.items():
        if not isinstance(value, torch.Tensor):
            continue
        flat = value.detach().float().reshape(value.shape[0], -1).cpu()
        row = storage.setdefault(name, {
            "batches": 0,
            "rows": 0,
            "flat_dim": int(flat.size(1)),
            "mean_sum": 0.0,
            "abs_mean_sum": 0.0,
            "l2_mean_sum": 0.0,
        })
        rows = int(flat.size(0))
        row["batches"] += 1
        row["rows"] += rows
        row["mean_sum"] += float(flat.mean().item()) * rows
        row["abs_mean_sum"] += float(flat.abs().mean().item()) * rows
        row["l2_mean_sum"] += float(flat.norm(dim=1).mean().item()) * rows


def finalize_watch_stats(storage: dict) -> dict[str, dict[str, float]]:
    result = {}
    for name, row in storage.items():
        rows = max(1, int(row["rows"]))
        result[name] = {
            "batches": int(row["batches"]),
            "rows": int(row["rows"]),
            "flat_dim": int(row["flat_dim"]),
            "mean": float(row["mean_sum"]) / rows,
            "abs_mean": float(row["abs_mean_sum"]) / rows,
            "l2_mean": float(row["l2_mean_sum"]) / rows,
        }
    return result


def watch_changes(baseline: dict, candidate: dict) -> dict[str, dict[str, float]]:
    result = {}
    for name, base in baseline.items():
        current = candidate.get(name)
        if current is None:
            continue
        result[name] = {
            "mean_delta": float(current["mean"]) - float(base["mean"]),
            "abs_mean_delta": float(current["abs_mean"]) - float(base["abs_mean"]),
            "l2_mean_delta": float(current["l2_mean"]) - float(base["l2_mean"]),
        }
    return result
