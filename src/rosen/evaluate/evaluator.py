"""Evaluators consume one resolved EvalBatch and never mutate its tensors."""
from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .batch import EvalBatch
from .metrics import METRIC_REGISTRY


class BaseEvaluator(ABC):
    @abstractmethod
    def collect(self, batch: EvalBatch) -> None:
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


def _metrics(values):
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]
    return list(dict.fromkeys(value.lower() for value in values))


def _pointwise_inputs(batch):
    users, targets, scores, mask = batch.users, batch.targets, batch.scores, batch.mask
    if targets is None or scores is None:
        raise ValueError("Point-wise evaluation requires targets and scores.")
    if targets.ndim not in (1, 2) or scores.shape != targets.shape:
        raise ValueError("Point-wise targets and scores must have matching [B] or [B, K] shapes.")
    if mask is not None and mask.shape != targets.shape:
        raise ValueError("mask must have the same shape as targets.")
    valid = mask > 0 if mask is not None else torch.ones_like(targets, dtype=torch.bool)
    if users is not None:
        if targets.ndim == 2 and users.shape == targets.shape[:1]:
            users = users[:, None].expand_as(targets)
        if users.shape != targets.shape:
            raise ValueError("User IDs must have shape [B] or match the target shape.")
        users = users[valid]
    return users, targets[valid], scores[valid]


class PointWiseEvaluator(BaseEvaluator):
    def __init__(self, *metrics):
        self.metrics = _metrics(metrics)
        if set(self.metrics) - {"auc", "logloss", "gauc"}:
            raise ValueError(f"Unsupported point-wise metrics: {self.metrics}")
        self.clear()

    def collect(self, batch: EvalBatch):
        users, targets, scores = _pointwise_inputs(batch)
        if targets.numel() == 0:
            return
        if "gauc" in self.metrics and users is None:
            raise ValueError("gauc requires user IDs; provide uid_field on EvaluatorManager or register.")
        if users is not None:
            self.all_users.extend(users.detach().cpu().tolist())
        self.all_targets.extend(targets.detach().cpu().tolist())
        self.all_preds.extend(scores.detach().cpu().tolist())

    def summary(self):
        if not self.all_targets:
            return {}
        return {metric: float(METRIC_REGISTRY[metric].calculate(
            y_true=np.array(self.all_targets), y_pred=np.array(self.all_preds), users=self.all_users,
        )) for metric in self.metrics}

    def clear(self):
        self.all_users, self.all_targets, self.all_preds = [], [], []


class TopKEvaluator(BaseEvaluator):
    """Leave-one-out full-catalog metrics; labels are catalog item indices."""
    def __init__(self, *metrics, history_dict=None):
        self.metrics = _metrics(metrics)
        for metric in self.metrics:
            name, separator, cutoff = metric.partition("@")
            if name not in ("hr", "ndcg") or not separator or not cutoff.isdigit() or int(cutoff) <= 0:
                raise ValueError(f"Unsupported full-catalog metric: {metric}")
        self.history_dict = history_dict if history_dict is not None else {}
        self.clear()

    def collect(self, batch: EvalBatch):
        scores, targets, users = batch.pred.full_scores, batch.targets, batch.users
        if scores is None or targets is None or users is None:
            raise ValueError("TopKEvaluator requires full_scores, catalog targets and user IDs.")
        if batch.mask is not None:
            raise ValueError("Full-catalog TopKEvaluator does not accept slot masks; use SlateEvaluator.")
        if targets.shape != scores.shape[:1] or targets.dtype != torch.long or users.shape != targets.shape:
            raise ValueError("Catalog targets must be LongTensor[B], with user IDs [B].")
        if scores.shape[1] == 0 or ((targets < 0) | (targets >= scores.shape[1])).any():
            raise ValueError("Catalog target index is outside full_scores.")
        if not targets.numel():
            return
        local = scores.detach().clone()
        row_ids = torch.arange(scores.shape[0], device=scores.device)
        positives = local[row_ids, targets].clone()
        for row, uid in enumerate(users.tolist()):
            history = self.history_dict.get(uid, [])
            if history:
                local[row, history] = -torch.inf
        local[row_ids, targets] = positives
        max_k = min(max((int(m.split('@')[1]) for m in self.metrics), default=0), scores.shape[1])
        indices = local.topk(max_k, dim=-1).indices
        hits = (indices == targets[:, None]).float()
        for metric in self.metrics:
            name, cutoff = metric.split("@")
            k = min(int(cutoff), scores.shape[1])
            self.sums[metric] += METRIC_REGISTRY[name].calculate(hits_k=hits[:, :k], k=k)
        self.count += scores.shape[0]

    def summary(self):
        return {name: value / self.count for name, value in self.sums.items()} if self.count else {}

    def clear(self):
        self.sums = dict.fromkeys(self.metrics, 0.0)
        self.count = 0


class Evaluator(BaseEvaluator):
    """Compatibility facade: retain mixed point-wise / full-catalog registration."""
    def __init__(self, *metrics, history_dict=None):
        self.metrics = _metrics(metrics)
        point = [metric for metric in self.metrics if "@" not in metric]
        topk = [metric for metric in self.metrics if "@" in metric]
        if topk and history_dict is None:
            raise ValueError("Full-catalog metrics require history_dict (an empty dict is allowed).")
        self.point_evaluator = PointWiseEvaluator(*point) if point else None
        self.topk_evaluator = TopKEvaluator(*topk, history_dict=history_dict) if topk else None

    def collect(self, batch: EvalBatch):
        if self.point_evaluator is not None:
            self.point_evaluator.collect(batch)
        if self.topk_evaluator is not None:
            self.topk_evaluator.collect(batch)

    def summary(self):
        result = {}
        for evaluator in (self.point_evaluator, self.topk_evaluator):
            if evaluator is not None:
                result.update(evaluator.summary())
        return result

    def clear(self):
        for evaluator in (self.point_evaluator, self.topk_evaluator):
            if evaluator is not None:
                evaluator.clear()


class SlateEvaluator(BaseEvaluator):
    """Supervised input-slate score metrics, accumulated across all batches."""
    def __init__(self, *metrics, ndcg_gain_source="label"):
        self.metrics = _metrics(metrics)
        if ndcg_gain_source not in ("label", "position"):
            raise ValueError("ndcg_gain_source must be 'label' or 'position'.")
        self.ndcg_gain_source = ndcg_gain_source
        for metric in self.metrics:
            if metric in ("auc", "logloss", "gauc", "cb"):
                continue
            name, separator, cutoff = metric.partition("@")
            if name != "ndcg" or not separator or not cutoff.isdigit() or int(cutoff) <= 0:
                raise ValueError(f"Unsupported slate metric: {metric}; candidate metrics belong in custom evaluators.")
        self.point_evaluator = PointWiseEvaluator(*(m for m in self.metrics if m in ("auc", "logloss", "gauc")))
        self.clear()

    def collect(self, batch: EvalBatch):
        labels, scores = batch.targets, batch.scores
        if labels is None or scores is None or labels.ndim != 2 or scores.shape != labels.shape:
            raise ValueError("SlateEvaluator requires matching [B, K] targets and scores.")
        if batch.mask is not None and batch.mask.shape != labels.shape:
            raise ValueError("mask must have the same shape as slate targets.")
        if self.point_evaluator.metrics:
            self.point_evaluator.collect(batch)
        labels, scores = labels.detach().cpu().double(), scores.detach().cpu().double()
        valid = batch.mask.detach().cpu() > 0 if batch.mask is not None else torch.ones_like(labels, dtype=torch.bool)
        self.label_sum += labels[valid].sum().item()
        self.pred_sum += scores[valid].sum().item()
        self.slot_count += int(valid.sum())
        relevance = labels if self.ndcg_gain_source == "label" else torch.arange(
            labels.shape[1] - 1, -1, -1, dtype=labels.dtype,
        ).expand_as(labels)
        rows = valid.any(-1)
        for metric in self.sums:
            values = METRIC_REGISTRY["list_ndcg"].calculate(relevance, scores, int(metric.split('@')[1]), mask=valid)
            self.sums[metric] += values[rows].sum().item()
            self.counts[metric] += int(rows.sum())

    def summary(self):
        result = self.point_evaluator.summary() if self.point_evaluator.metrics else {}
        if "cb" in self.metrics and self.slot_count:
            result["cb"] = self.pred_sum / self.label_sum if self.label_sum else 0.0
        result.update({metric: value / self.counts[metric] for metric, value in self.sums.items() if self.counts[metric]})
        return result

    def clear(self):
        self.point_evaluator.clear()
        self.sums = {metric: 0.0 for metric in self.metrics if metric.startswith("ndcg@")}
        self.counts = dict.fromkeys(self.sums, 0)
        self.label_sum = self.pred_sum = 0.0
        self.slot_count = 0


class OrderEvaluator(BaseEvaluator):
    """Evaluate the final permutation, using original-slot labels and masks."""
    def __init__(self, *metrics):
        self.metrics = _metrics(metrics)
        if any(not metric.startswith("ndcg@") for metric in self.metrics):
            raise ValueError("OrderEvaluator supports ndcg@k metrics.")
        self.evaluator = SlateEvaluator(*self.metrics)

    def collect(self, batch: EvalBatch):
        order, labels = batch.order, batch.targets
        if order is None or labels is None or labels.ndim != 2 or order.shape != labels.shape:
            raise ValueError("OrderEvaluator requires matching [B, K] order and targets.")
        expected = torch.arange(order.shape[1], device=order.device).expand_as(order)
        if order.dtype != torch.long or not torch.equal(order.sort(-1).values, expected):
            raise ValueError("order must be a permutation of input slot indices 0..K-1.")
        ranks = torch.arange(order.shape[1], 0, -1, device=order.device, dtype=torch.float32).expand_as(order)
        scores = torch.zeros_like(ranks).scatter(1, order, ranks)
        self.evaluator.collect(replace(batch, scores=scores))

    def summary(self):
        return self.evaluator.summary()

    def clear(self):
        self.evaluator.clear()


class LegacyEvaluatorAdapter(BaseEvaluator):
    """Explicit bridge for an old collect(users, targets, preds, full_scores) object.

    Slot masks need a real migration rather than silently changing old semantics.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def collect(self, batch: EvalBatch):
        if batch.mask is not None:
            raise ValueError("LegacyEvaluatorAdapter does not support slot masks; migrate collect(EvalBatch).")
        self.evaluator.collect(batch.users, batch.targets, batch.scores, batch.pred.full_scores)

    def summary(self):
        return self.evaluator.summary()

    def clear(self):
        self.evaluator.clear()


class EvaluatorDecorator(BaseEvaluator):
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def collect(self, batch: EvalBatch):
        self.evaluator.collect(batch)

    def summary(self):
        return self.evaluator.summary()

    def clear(self):
        self.evaluator.clear()


class LogDecorator(EvaluatorDecorator):
    """Optional standalone metric log; epoch/step logging belongs to the manager."""
    def __init__(self, evaluator, save_path=None, title=None):
        super().__init__(evaluator)
        self.save_path, self.title = Path(save_path) if save_path else None, title
        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def summary(self):
        result = self.evaluator.summary()
        if self.save_path:
            with self.save_path.open("a", encoding="utf-8") as stream:
                stream.write(f"[{datetime.now().isoformat(timespec='seconds')}] {self.title or ''} {result}\n")
        return result
