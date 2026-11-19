"""Metrics consume resolved tensor views; source selection belongs to BatchResolver."""
from abc import ABC, abstractmethod

import numpy as np
import torch

from .batch import EvalBatch
from .metrics import METRIC_REGISTRY


def _metrics(values):
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]
    return list(dict.fromkeys(value.lower() for value in values))


def _cpu(tensor):
    return tensor.detach().cpu() if tensor is not None else None


def _valid(labels, mask):
    if mask is not None and mask.shape != labels.shape:
        raise ValueError("mask must have the same shape as labels")
    return mask > 0 if mask is not None else torch.ones_like(labels, dtype=torch.bool)


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


class PointwiseEvaluator(BaseEvaluator):
    """Global exposure classification and calibration, flattened over valid slots."""
    def __init__(self, *metrics):
        self.metrics = _metrics(metrics)
        if set(self.metrics) - {"auc", "logloss", "gauc", "pcoc"}:
            raise ValueError(f"Unsupported point-wise metrics: {self.metrics}")
        self.clear()

    def collect(self, batch: EvalBatch):
        labels, scores, users = _cpu(batch.labels), _cpu(batch.scores), _cpu(batch.users)
        if scores is None or labels.ndim not in (1, 2) or scores.shape != labels.shape:
            raise ValueError("Point-wise labels and scores require matching [B] or [B,N] shapes")
        valid = _valid(labels, _cpu(batch.mask))
        if "gauc" in self.metrics:
            if users is None:
                raise ValueError("gauc requires user IDs; provide uid_field on EvaluatorManager or register")
            if valid.ndim == 2 and users.shape == valid.shape[:1]:
                users = users[:, None].expand_as(valid)
            if users.shape != valid.shape:
                raise ValueError("User IDs must be [B] or match labels")
        labels, scores = labels[valid].double(), scores[valid].double()
        if not torch.isfinite(labels).all() or not torch.isfinite(scores).all():
            raise ValueError("Valid labels and scores must be finite")
        if set(self.metrics) - {"pcoc"} and not torch.isin(labels, labels.new_tensor([0, 1])).all():
            raise ValueError("Classification metrics require binary labels")
        if "logloss" in self.metrics and ((scores < 0) | (scores > 1)).any():
            raise ValueError("logloss requires probabilities in [0,1]")
        if not labels.numel():
            return
        if "gauc" in self.metrics:
            self._user_batches.append(users[valid].numpy().copy())
        self._label_batches.append(labels.numpy().copy())
        self._score_batches.append(scores.numpy().copy())

    def summary(self):
        if not self._label_batches:
            return {}
        labels = np.concatenate(self._label_batches)
        scores = np.concatenate(self._score_batches)
        users = np.concatenate(self._user_batches) if self._user_batches else None
        return {metric: float(METRIC_REGISTRY[metric].calculate(
            y_true=labels, y_pred=scores, users=users))
            for metric in self.metrics}

    def clear(self):
        self._user_batches, self._label_batches, self._score_batches = [], [], []


class RankingEvaluator(BaseEvaluator):
    """Candidate prediction and request-local ranking; explicit order takes precedence.

    NDCG uses graded labels. Other cutoff metrics use label > 0 as relevance.
    Recall/hit/MRR/MAP exclude requests without positives; NDCG and precision
    include their zero results. Precision divides by valid selected positions.
    Tied scores keep input order.
    """
    _POINTWISE_METRICS = {"auc", "logloss", "gauc", "pcoc"}
    _CUTOFF_METRIC_MAP = {"ndcg": "list_ndcg", "recall": "list_recall", "hit": "list_hit",
                          "hr": "list_hit", "precision": "list_precision", "mrr": "list_mrr", "map": "list_map"}

    def __init__(self, *metrics: str):
        self.metrics = _metrics(metrics)
        point = [m for m in self.metrics if m in self._POINTWISE_METRICS]
        self.cutoffs = {}
        for metric in self.metrics:
            if metric in point or metric in ("list_auc", "in_slate_auc"):
                continue
            name, separator, k = metric.partition("@")
            if name not in self._CUTOFF_METRIC_MAP or not separator or not k.isdigit() or int(k) <= 0:
                raise ValueError(f"Unsupported slot/ranking metric: {metric}")
            self.cutoffs[metric] = (self._CUTOFF_METRIC_MAP[name], int(k))
        self.point_evaluator = PointwiseEvaluator(*point)
        self.clear()

    def collect(self, batch: EvalBatch):
        labels = _cpu(batch.labels).double()
        scores = _cpu(batch.scores).double() if batch.scores is not None else None
        if labels.ndim != 2 or not labels.shape[1]:
            raise ValueError("RankingEvaluator requires matching candidate labels [B,N]")
        valid = _valid(labels, _cpu(batch.mask))
        if not torch.isfinite(labels[valid]).all():
            raise ValueError("Valid ranking labels must be finite")
        needs_scores = bool(self.point_evaluator.metrics) or any(m in self.metrics for m in ("list_auc", "in_slate_auc"))
        if needs_scores and (scores is None or scores.shape != labels.shape):
            raise ValueError("Score-based ranking metrics require aligned scores [B,N]")
        if scores is not None and scores.shape == labels.shape and not torch.isfinite(scores[valid]).all():
            raise ValueError("Valid ranking scores must be finite")
        if self.point_evaluator.metrics:
            self.point_evaluator.collect(batch)
        for metric in ("list_auc", "in_slate_auc"):
            if metric in self.metrics:
                values = METRIC_REGISTRY[metric].calculate(labels, scores, valid)
                legal = torch.isfinite(values)
                self.sums[metric] += values[legal].sum().item()
                self.counts[metric] += int(legal.sum())
        if self.cutoffs:
            if (labels[valid] < 0).any():
                raise ValueError("ranking/ndcg requires nonnegative relevance")
            indices = _cpu(batch.order)
            if indices is None:
                if scores is None or scores.shape != labels.shape:
                    raise ValueError("Ranking requires scores [B,N] or order [B,K]")
                indices = scores.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)
            rows = valid.any(-1)
            for metric, (registry_key, cutoff) in self.cutoffs.items():
                values = METRIC_REGISTRY[registry_key].calculate(labels, indices, valid, k=cutoff)
                legal = rows & torch.isfinite(values)
                self.sums[metric] += values[legal].sum().item()
                self.counts[metric] += int(legal.sum())
        self.row_count += int(valid.any(-1).sum())

    def summary(self):
        if not self.row_count:
            return {}
        result = self.point_evaluator.summary() if self.point_evaluator.metrics else {}
        result.update({m: self.sums[m] / n if n else float("nan") for m, n in self.counts.items()})
        return {m: result[m] for m in self.metrics if m in result}

    def clear(self):
        self.point_evaluator.clear()
        self.row_count = 0
        self.sums = {m: 0. for m in self.metrics if m not in self._POINTWISE_METRICS}
        self.counts = dict.fromkeys(self.sums, 0)


class SlateValueEvaluator(BaseEvaluator):
    """Cross-request value metrics on already aligned whole-slate tensors."""
    def __init__(self, *metrics: str, reduction: str = "sum", ncis_cap: float = 10.,
                 candidate_reduce: str = "none"):
        self.metrics = _metrics(metrics)
        if set(self.metrics) - {"r_auc", "slate_auc", "pcoc", "rmse", "mae", "ncis", "cand_mean", "cand_max"}:
            raise ValueError(f"Unsupported slate metrics: {self.metrics}")
        if reduction not in ("sum", "mean", "dcg"):
            raise ValueError("reduction must be sum, mean or dcg")
        if not np.isfinite(ncis_cap) or ncis_cap <= 0:
            raise ValueError("ncis_cap must be finite and positive")
        if candidate_reduce not in ("none", "mean", "max"):
            raise ValueError("candidate_reduce must be none, mean or max")
        self.candidate_reduce = candidate_reduce
        self.reduction, self.ncis_cap = reduction, ncis_cap
        self.clear()

    def reduce_slate(self, tensor, mask):
        """Reduce slots in [B,K] / [B,M,K]; request scalars [B] pass through."""
        if tensor.ndim == 1:
            return tensor
        if tensor.ndim not in (2, 3) or not tensor.shape[-1]:
            raise ValueError("reduce_slate expects [B], [B,K] or [B,M,K]")
        if mask is not None and tensor.ndim == 3 and mask.shape == (tensor.shape[0], tensor.shape[-1]):
            mask = mask[:, None].expand_as(tensor)
        valid = _valid(tensor, mask)
        values = torch.where(valid, tensor.double(), 0.)
        if self.reduction == "dcg":
            discount = 1. / torch.log2(torch.arange(tensor.shape[-1], device=tensor.device).double() + 2)
            return (values * discount).sum(-1)
        total = values.sum(-1)
        return total / valid.sum(-1).clamp_min(1) if self.reduction == "mean" else total

    def _collect_candidates(self, batch):
        scores, mask = _cpu(batch.scores).double(), _cpu(batch.mask)
        if not scores.shape[1] or not scores.shape[2]:
            raise ValueError("Candidate tensors must be non-empty [B,M,K]")
        if mask is not None and mask.shape == (scores.shape[0], scores.shape[-1]):
            mask = mask[:, None].expand_as(scores)
        valid = _valid(scores, mask)
        candidates = valid.any(-1)
        rows = candidates.any(-1)
        if not torch.isfinite(scores[valid]).all():
            raise ValueError("Valid slate values must be finite")
        predictions = self.reduce_slate(scores, valid)
        value_metrics = [m for m in self.metrics if m not in ("cand_mean", "cand_max")]
        reduced = []
        if value_metrics:
            if self.candidate_reduce == "none":
                raise ValueError("Candidate value metrics require aligned reduction: candidate_reduce='mean' or 'max'")
            selected = predictions.masked_fill(~candidates, -torch.inf).argmax(-1)
            for tensor in [scores, _cpu(batch.labels)] + ([_cpu(batch.ref_scores)] if "ncis" in self.metrics else []):
                tensor = tensor.double()
                if tensor.ndim == 1:
                    values = tensor
                else:
                    if tensor.ndim == 2 and tensor.shape == (scores.shape[0], scores.shape[-1]):
                        tensor = tensor[:, None].expand_as(scores)
                    if tensor.shape != scores.shape:
                        raise ValueError("Candidate labels and references must be [B], [B,K] or aligned [B,M,K]")
                    if not torch.isfinite(tensor[valid]).all():
                        raise ValueError("Valid slate values must be finite")
                    per_candidate = self.reduce_slate(tensor, valid)
                    if self.candidate_reduce == "max":
                        # The predicted best candidate drives all three sources.
                        values = per_candidate.gather(1, selected[:, None])[:, 0]
                    else:
                        values = per_candidate.masked_fill(~candidates, 0).sum(-1) / candidates.sum(-1).clamp_min(1)
                reduced.append(values[rows])
            self._store_values(reduced)
        if "cand_mean" in self.metrics:
            self.cand_total += predictions[candidates].sum().item()
            self.cand_count += int(candidates.sum())
        if "cand_max" in self.metrics:
            self.cand_max += predictions.masked_fill(~candidates, -torch.inf).amax(-1)[rows].sum().item()
        self.cand_rows += int(rows.sum())

    def _store_values(self, values):
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError("Valid slate values must be finite")
        if "ncis" in self.metrics and ((values[0] < 0).any() or (values[2] < 0).any()):
            raise ValueError("ncis requires nonnegative prediction and reference scores")
        if not values[0].numel():
            return
        self._prediction_batches.append(values[0].numpy().copy())
        self._value_batches.append(values[1].numpy().copy())
        if "ncis" in self.metrics:
            self._reference_batches.append(values[2].numpy().copy())

    def collect(self, batch: EvalBatch):
        if batch.scores is None or ("ncis" in self.metrics and batch.ref_scores is None):
            raise ValueError("SlateValueEvaluator requires scores and ncis requires ref_scores")
        if batch.scores.ndim == 3:
            self._collect_candidates(batch)
            return
        if any(m in self.metrics for m in ("cand_mean", "cand_max")):
            raise ValueError("cand_mean/cand_max require [B,M,K] scores")
        mask = _cpu(batch.mask)
        if mask is not None and (mask.ndim != 2 or mask.shape[0] != batch.batch_size):
            raise ValueError("SlateValueEvaluator mask must be [B,K]")
        rows = mask.gt(0).any(-1) if mask is not None else torch.ones(batch.batch_size, dtype=torch.bool)
        tensors = [batch.scores, batch.labels] + ([batch.ref_scores] if "ncis" in self.metrics else [])
        if len({tensor.shape for tensor in tensors if tensor.ndim == 2}) > 1:
            raise ValueError("SlateValueEvaluator slot tensors must align")
        values = []
        for tensor in tensors:
            if tensor.ndim not in (1, 2):
                raise ValueError("SlateValueEvaluator requires [B] or aligned [B,K] tensors")
            values.append(self.reduce_slate(_cpu(tensor).double(), mask)[rows])
        self._store_values(values)

    def summary(self):
        result = {}
        if self._value_batches:
            values = np.concatenate(self._value_batches)
            predictions = np.concatenate(self._prediction_batches)
            references = np.concatenate(self._reference_batches) if self._reference_batches else None
            result = {m: float(METRIC_REGISTRY[m].calculate(y_true=values, y_pred=predictions,
                        y_ref=references, cap=self.ncis_cap)) for m in self.metrics
                      if m not in ("cand_mean", "cand_max")}
        if "cand_mean" in self.metrics and self.cand_count:
            result["cand_mean"] = self.cand_total / self.cand_count
        if "cand_max" in self.metrics and self.cand_rows:
            result["cand_max"] = self.cand_max / self.cand_rows
        return result

    def clear(self):
        self._prediction_batches, self._value_batches, self._reference_batches = [], [], []
        self.cand_total = self.cand_max = 0.
        self.cand_count = self.cand_rows = 0


class RetrievalEvaluator(BaseEvaluator):
    """Leave-one-out full-catalog metrics; labels are catalog item indices."""
    def __init__(self, *metrics, history_dict=None):
        self.metrics = _metrics(metrics)
        for metric in self.metrics:
            name, separator, cutoff = metric.partition("@")
            if name not in ("hr", "ndcg", "recall", "mrr") or not separator or not cutoff.isdigit() or int(cutoff) <= 0:
                raise ValueError(f"Unsupported full-catalog metric: {metric}")
        self.history_dict = history_dict if history_dict is not None else {}
        self.clear()

    def collect(self, batch: EvalBatch):
        scores, labels, users = batch.full_scores, batch.labels, batch.users
        if scores is None or labels is None or users is None:
            raise ValueError("RetrievalEvaluator requires full_scores, catalog labels and user IDs.")
        if batch.mask is not None:
            raise ValueError("Full-catalog RetrievalEvaluator does not accept slot masks; use RankingEvaluator.")
        if scores.ndim != 2:
            raise ValueError("RetrievalEvaluator full_scores must be [B,|I|]")
        if labels.shape != scores.shape[:1] or labels.dtype != torch.long or users.shape != labels.shape:
            raise ValueError("Catalog labels must be LongTensor[B], with user IDs [B].")
        if scores.shape[1] == 0 or ((labels < 0) | (labels >= scores.shape[1])).any():
            raise ValueError("Catalog target index is outside full_scores.")
        if not labels.numel():
            return
        local = scores.detach().clone()
        row_ids = torch.arange(scores.shape[0], device=scores.device)
        positives = local[row_ids, labels].clone()
        for row, uid in enumerate(users.tolist()):
            history = self.history_dict.get(uid, [])
            if history:
                local[row, history] = -torch.inf
        local[row_ids, labels] = positives
        max_k = min(max((int(m.split('@')[1]) for m in self.metrics), default=0), scores.shape[1])
        indices = local.topk(max_k, dim=-1).indices
        hits = (indices == labels[:, None]).float()
        for metric in self.metrics:
            name, cutoff = metric.split("@")
            k = min(int(cutoff), scores.shape[1])
            self.sums[metric] += METRIC_REGISTRY[{"recall": "catalog_recall", "mrr": "catalog_mrr"}.get(name, name)].calculate(hits_k=hits[:, :k], k=k)
        self.count += scores.shape[0]

    def summary(self):
        return {name: value / self.count for name, value in self.sums.items()} if self.count else {}

    def clear(self):
        self.sums = dict.fromkeys(self.metrics, 0.0)
        self.count = 0
