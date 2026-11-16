from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import polars as pl
import torch
from sklearn.metrics import log_loss, roc_auc_score


def safe_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return math.nan
    return float(roc_auc_score(labels, preds))


def safe_logloss(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0:
        return math.nan
    return float(log_loss(labels, np.clip(preds, 1e-7, 1.0 - 1e-7), labels=[0, 1]))


def metric_row(
        *,
        model_name: str,
        split: str,
        scope: str,
        labels: np.ndarray,
        preds: np.ndarray,
        trained_batches: int,
) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split,
        "scope": scope,
        "trained_batches": trained_batches,
        "rows": int(labels.size),
        "positives": int(labels.sum()) if labels.size else 0,
        "ctr": float(labels.mean()) if labels.size else math.nan,
        "pred_mean": float(preds.mean()) if preds.size else math.nan,
        "auc": safe_auc(labels, preds),
        "logloss": safe_logloss(labels, preds),
    }


def gradient_vector(grad: torch.Tensor) -> torch.Tensor:
    grad = grad.detach().float()
    if grad.ndim == 0:
        return grad.reshape(1)
    if grad.ndim == 1:
        return grad.reshape(-1, 1).mean(dim=0)
    return grad.reshape(grad.size(0), -1).mean(dim=0)


def compare_gradients(
        left_grads: Mapping[str, torch.Tensor],
        right_grads: Mapping[str, torch.Tensor],
) -> list[dict[str, object]]:
    rows = []
    hidden_taps = {"shared_z", "head_input", "logits"}
    for name in sorted(set(left_grads) & set(right_grads)):
        left = gradient_vector(left_grads[name])
        right = gradient_vector(right_grads[name])
        if left.numel() != right.numel():
            continue
        dot = float(torch.dot(left, right).detach().cpu().item())
        norm_left = float(torch.linalg.vector_norm(left).detach().cpu().item())
        norm_right = float(torch.linalg.vector_norm(right).detach().cpu().item())
        cosine = dot / (norm_left * norm_right) if norm_left > 0.0 and norm_right > 0.0 else math.nan
        rows.append({
            "tap": name,
            "tap_type": "hidden" if name in hidden_taps else "field",
            "vector_dim": int(left.numel()),
            "dot": dot,
            "norm_left": norm_left,
            "norm_right": norm_right,
            "cosine": cosine,
            "left_shape": "x".join(str(dim) for dim in left_grads[name].shape),
            "right_shape": "x".join(str(dim) for dim in right_grads[name].shape),
        })
    return rows


def summarize_gradients(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pl.DataFrame(rows)
    summary = (
        df.group_by(["model", "split", "mode", "tap", "tap_type"])
        .agg(
            pl.len().alias("batches"),
            pl.max("vector_dim").alias("vector_dim"),
            pl.mean("rows_left").alias("rows_left_mean"),
            pl.mean("rows_right").alias("rows_right_mean"),
            pl.mean("ctr_left").alias("ctr_left_mean"),
            pl.mean("ctr_right").alias("ctr_right_mean"),
            pl.mean("loss_left").alias("loss_left_mean"),
            pl.mean("loss_right").alias("loss_right_mean"),
            pl.sum("dot").alias("dot_sum"),
            (pl.col("norm_left") ** 2).sum().alias("norm_left_sq_sum"),
            (pl.col("norm_right") ** 2).sum().alias("norm_right_sq_sum"),
            pl.mean("cosine").alias("cosine_mean"),
            pl.quantile("cosine", 0.10).alias("cosine_p10"),
            pl.quantile("cosine", 0.50).alias("cosine_p50"),
            (pl.col("cosine") < 0.0).mean().alias("negative_batch_ratio"),
        )
        .with_columns(
            (
                pl.col("dot_sum")
                / (pl.col("norm_left_sq_sum").sqrt() * pl.col("norm_right_sq_sum").sqrt())
            ).alias("global_cosine")
        )
        .sort(["model", "split", "mode", "tap_type", "global_cosine"])
    )
    return summary.to_dicts()
