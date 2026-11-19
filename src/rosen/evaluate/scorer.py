"""Extract evaluation scores independently of metric calculation."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Mapping
from math import factorial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import polars as pl
import torch

from rosen.core.prediction import Prediction
from rosen.core.interaction import Interaction

if TYPE_CHECKING:
    from rosen.models.rerank.base import RerankModel


@dataclass(frozen=True)
class EvalContext:
    """Raw, row-filtered source for scorers, used only before EvalBatch exists."""
    pred: Prediction
    inter: Optional[Interaction] = None
    target: Optional[str] = None

    def prediction_slot(self, name):
        if name == "scores":
            return self.pred.get_scores(self.target)
        if name == "order":
            return self.pred.get_order(self.target)
        if name == "chosen_scores":
            return self.pred.resolve_chosen_scores(self.target)
        if name == "candidates.scores":
            return self.pred.candidates.scores if self.pred.candidates is not None else None
        if name == "candidates.orders":
            return self.pred.candidates.orders if self.pred.candidates is not None else None
        value = getattr(self.pred, name)
        return value[self.target] if isinstance(value, Mapping) else value


def normalize_orders(order, batch_size, num_candidates):
    """Normalize to [B,M,K]; [B,K] means one order per row.

    Shared [M,K] is accepted when M != B. Use [1,M,K] for shared
    candidates when M == B, so batch size never silently changes semantics.
    """
    if not isinstance(order, torch.Tensor) or order.dtype != torch.long:
        raise TypeError("order must be a LongTensor")
    single = order.ndim == 2 and order.shape[0] == batch_size
    if order.ndim == 2:
        order = order[:, None] if single else order[None]
    if order.ndim != 3 or order.shape[0] not in (1, batch_size):
        raise ValueError("order must be [B,K], [M,K], [1,M,K] or [B,M,K]")
    k = order.shape[-1]
    if order.shape[1] == 0 or not 1 <= k <= num_candidates:
        raise ValueError("Ordered selection requires non-empty candidates and 1 <= K <= N")
    sorted_order = order.sort(-1).values
    if ((sorted_order[..., 0] < 0).any() or (sorted_order[..., -1] >= num_candidates).any()
            or (sorted_order[..., 1:] <= sorted_order[..., :-1]).any()):
        raise ValueError("Each order must be a partial permutation: unique indices in 0..N-1")
    return order.expand(batch_size, -1, -1), single


def gather_by_order(scores, order):
    if order is None:
        return scores
    if scores.ndim != 2:
        raise ValueError("Reordering fixed scores requires slot scores [B,N]")
    orders, single = normalize_orders(order, scores.shape[0], scores.shape[1])
    orders = orders.to(scores.device)
    result = scores[:, None].expand(-1, orders.shape[1], -1).gather(-1, orders)
    return result[:, 0] if single else result


def _tensor(value, name, batch_size):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must provide a Tensor")
    if value.ndim not in (1, 2, 3) or value.shape[0] != batch_size:
        raise ValueError(f"{name} must provide [B], [B,K] or [B,M,K] scores")
    return value


class BaseScorer(ABC):
    """Read original scores, or score supplied output-position → input-slot orders.

    Original output is [B] or [B,N]. A single order [B,K], 1 <= K <= N, produces
    [B] or [B,K]; candidates [B,M,K] produce [B,M] or [B,M,K].
    Fixed slot scorers only transport existing values; they do not predict
    feedback changes caused by moving an item.
    """
    uses_schema_labels: bool = False

    @abstractmethod
    def score(self, batch: EvalContext, order: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, batch: EvalContext, order: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score(batch, order=order)

    def needs_target(self, pred: Prediction) -> bool:
        """Whether scoring needs an explicit label/task name in EvalContext."""
        return self.uses_schema_labels

    def uses_prediction_target(self, pred: Prediction) -> bool:
        """Whether this source consumes a tensor bound to Prediction.target."""
        return False

    def num_candidates(self, context: EvalContext) -> Optional[int]:
        """Declare this scorer's original width; unknown for arbitrary custom sources."""
        return None


def _apply_order_if_slot_pool(tensor, order, alignment="candidates"):
    if alignment == "selected":
        if order is not None and tensor.ndim != 1:
            if tensor.shape != order.shape:
                raise ValueError("Selected scores must align with one order [B,K] or candidates [B,M,K]")
        return tensor
    # 2D values are candidate pools unless the source explicitly declares selected.
    return gather_by_order(tensor, order) if tensor.ndim == 2 else tensor


def _check_alignment(alignment):
    if alignment not in ("candidates", "selected"):
        raise ValueError("alignment must be candidates or selected")
    return alignment


class PredictionScorer(BaseScorer):
    _SELECTED_ONLY_FIELDS = frozenset({"chosen_scores", "candidates.scores"})

    def __init__(self, field: str = "scores", *, alignment: Optional[str] = None):
        if not isinstance(field, str) or not field.strip():
            raise ValueError("PredictionScorer field must be a non-empty string")
        self.field = field
        if field in self._SELECTED_ONLY_FIELDS:
            if alignment is not None and alignment != "selected":
                raise ValueError(f"PredictionScorer field {field!r} is always selected-order; use alignment='selected'.")
            alignment = "selected"
        elif alignment is None:
            alignment = "candidates"
        self.alignment = _check_alignment(alignment)

    def _raw_tensor(self, context):
        if self.field in ("scores", "chosen_scores", "candidates.scores"):
            return context.prediction_slot(self.field)
        return context.pred.extra.get(self.field.removeprefix("extra."))

    def needs_target(self, pred):
        if self.field == "scores":
            return isinstance(pred.scores, Mapping)
        if self.field == "chosen_scores":
            return (isinstance(pred.chosen_scores, Mapping)
                    or (pred.chosen_scores is None
                        and (isinstance(pred.scores, Mapping) or isinstance(pred.order, Mapping))))
        return False

    def uses_prediction_target(self, pred):
        if self.field == "scores":
            return isinstance(pred.scores, torch.Tensor)
        if self.field == "chosen_scores":
            return (isinstance(pred.chosen_scores, torch.Tensor)
                    or (pred.chosen_scores is None
                        and (isinstance(pred.scores, torch.Tensor) or isinstance(pred.order, torch.Tensor))))
        return self.field == "candidates.scores" and pred.candidates is not None

    def num_candidates(self, context):
        if self.alignment == "selected":
            return None
        tensor = self._raw_tensor(context)
        return tensor.shape[1] if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 else None

    def score(self, batch, order=None):
        if self.field == "scores" and batch.pred.candidates is not None and order is not None:
            raise ValueError("Candidate-pool ordered scoring requires chosen_scores or candidates.scores; original-order scores cannot be gathered.")
        return _apply_order_if_slot_pool(
            _tensor(self._raw_tensor(batch), f"prediction {self.field}", batch.pred.batch_size),
            order, self.alignment)


class LabelScorer(BaseScorer):
    uses_schema_labels = True

    def __init__(self, *, alignment: str = "candidates"):
        self.alignment = _check_alignment(alignment)

    def num_candidates(self, context):
        if self.alignment == "selected":
            return None
        tensor = (context.inter[context.target] if context.inter is not None and context.target is not None
                  and context.target in context.inter else None)
        return tensor.shape[1] if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 else None

    def score(self, batch, order=None):
        return _apply_order_if_slot_pool(_tensor(batch.inter[batch.target] if batch.inter is not None and batch.target is not None else None,
                                "target scores", batch.pred.batch_size), order, self.alignment)


class FieldScorer(BaseScorer):
    def __init__(self, field_name: str, *, alignment: str = "candidates"):
        self.field_name = field_name
        self.alignment = _check_alignment(alignment)

    def score(self, batch, order=None):
        if batch.inter is None:
            raise ValueError("FieldScorer requires batch.inter")
        return _apply_order_if_slot_pool(_tensor(batch.inter[self.field_name], self.field_name, batch.pred.batch_size), order, self.alignment)

    def num_candidates(self, context):
        if self.alignment == "selected":
            return None
        tensor = (context.inter[self.field_name] if context.inter is not None
                  and self.field_name in context.inter else None)
        return tensor.shape[1] if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 else None


class SidecarLookupScorer(BaseScorer):
    """Read an in-memory sidecar by ID; perm_values uses lexicographic K! order.

    Permutation values are already aggregated scalars. Their reduction belongs
    to the artifact producer and cannot be changed during lookup.
    """
    def __init__(self, parquet_path: Path | str, score_col: str = "scores",
                 perm_col: str = "perm_values", id_field: str = "request_id", *,
                 order_col: Optional[str] = None, num_candidates: Optional[int] = None):
        self.id_field = id_field
        if num_candidates is not None and (isinstance(num_candidates, bool)
                or not isinstance(num_candidates, int) or num_candidates < 1):
            raise ValueError("num_candidates must be a positive integer")
        self._explicit_num_candidates = num_candidates
        self._sparse_lookup = None
        self._slots = num_candidates
        frame = pl.read_parquet(parquet_path)
        if order_col is not None:
            self._sparse_lookup, self._base_scores = {}, {}
            for uid, order, score in frame.select(id_field, order_col, score_col).iter_rows():
                if uid is None:
                    raise ValueError("Sidecar IDs must be non-null")
                value = torch.as_tensor(score) if score is not None else None
                if value is None or not torch.isfinite(value).all():
                    raise ValueError("Sidecar scores must be finite numeric values")
                if order is None:
                    if uid in self._base_scores or value.ndim not in (0, 1):
                        raise ValueError("Original sidecar scores must be unique scalars or slot vectors")
                    self._base_scores[uid] = value
                else:
                    if not order or any(isinstance(x, bool) or not isinstance(x, int) for x in order):
                        raise ValueError("Sidecar orders must be non-empty integer selections")
                    n = num_candidates if num_candidates is not None else max(order) + 1
                    normalize_orders(torch.tensor(order, dtype=torch.long)[None], 1, n)
                    key = (uid, tuple(order))
                    if key in self._sparse_lookup:
                        raise ValueError("Duplicate sidecar (ID, order) key")
                    if value.ndim != 0:
                        raise ValueError("Sparse order scores must be scalar values")
                    self._sparse_lookup[key] = value.item()
            return
        ids = frame[id_field]
        if ids.null_count() or ids.n_unique() != len(frame):
            raise ValueError("Sidecar IDs must be unique and non-null")
        self._index = {value: i for i, value in enumerate(ids.to_list())}
        try:
            self._scores = torch.as_tensor(frame[score_col].to_list())
            self._values = torch.as_tensor(frame[perm_col].to_list()) if perm_col in frame.columns else None
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Sidecar scores must be rectangular numeric arrays without nulls") from exc
        _tensor(self._scores, score_col, len(frame))
        if not torch.isfinite(self._scores).all():
            raise ValueError("Sidecar scores must be finite")
        if self._values is not None:
            if self._values.ndim != 2 or not torch.isfinite(self._values).all():
                raise ValueError("perm_values must be finite [N,K!] values")
            self._slots = next((k for k in range(1, 8) if factorial(k) == self._values.shape[1]), None)
            if self._explicit_num_candidates is not None and self._slots != self._explicit_num_candidates:
                raise ValueError("num_candidates disagrees with perm_values width")
            if self._slots is None:
                raise ValueError("perm_values must contain K! lexicographic values for 1 <= K <= 7")
            if self._scores.ndim == 2 and self._scores.shape[1] != self._slots:
                raise ValueError("scores and perm_values disagree on slot count")

    def num_candidates(self, context):
        if self._slots is not None:
            return self._slots
        if self._sparse_lookup is None and self._scores.ndim == 2:
            return self._scores.shape[1]
        return None

    def score(self, batch, order=None):
        if batch.inter is None:
            raise ValueError("SidecarLookupScorer requires batch.inter")
        ids = batch.inter[self.id_field]
        if ids.ndim != 1 or ids.shape[0] != batch.pred.batch_size:
            raise ValueError("Lookup IDs must have shape [B]")
        if self._sparse_lookup is not None:
            id_list = ids.detach().cpu().tolist()
            try:
                if order is None:
                    return torch.stack([self._base_scores[uid] for uid in id_list]).to(ids.device)
                n = self._slots if self._slots is not None else int(order.max().item()) + 1
                orders, single = normalize_orders(order, len(id_list), n)
                values = torch.tensor([[self._sparse_lookup[(uid, tuple(o))] for o in candidates]
                                       for uid, candidates in zip(id_list, orders.cpu().tolist())],
                                      dtype=torch.float32, device=ids.device)
                return values[:, 0] if single else values
            except KeyError as exc:
                raise KeyError(f"Sidecar has no {self.id_field}/order key {exc.args[0]!r}") from exc
        try:
            rows = torch.tensor([self._index[value] for value in ids.detach().cpu().tolist()], dtype=torch.long)
        except KeyError as exc:
            raise KeyError(f"Sidecar has no {self.id_field}={exc.args[0]!r}") from exc
        if order is None:
            return self._scores[rows].to(ids.device)
        if self._values is None:
            raise ValueError("Permutation lookup requires perm_values; original scores cannot stand in for them")
        orders, single = normalize_orders(order, len(rows), self._slots)
        if orders.shape[-1] != self._slots:
            raise ValueError("perm_values only supports full permutations (K=N), not N-choose-K selections")
        orders = orders.cpu()
        # Lehmer rank equals itertools.permutations(range(K)) enumeration.
        rank = torch.zeros(orders.shape[:2], dtype=torch.long)
        for i in range(self._slots - 1):
            rank += (orders[..., i + 1:] < orders[..., i, None]).sum(-1) * factorial(self._slots - i - 1)
        values = self._values[rows[:, None], rank].to(ids.device)
        return values[:, 0] if single else values


class JudgeModelScorer(BaseScorer):
    """Use a frozen reranker; candidate scoring is owned by the model."""
    def __init__(self, model: "RerankModel"):
        self.model = model.eval().requires_grad_(False)

    def num_candidates(self, context):
        return self.model.poi_num

    @torch.no_grad()
    def score(self, batch, order=None):
        if batch.inter is None:
            raise ValueError("JudgeModelScorer requires batch.inter")
        self.model.eval()
        if order is not None:
            orders, single = normalize_orders(order, batch.pred.batch_size, self.model.poi_num)
            result = self.model.score_permutations(batch.inter, orders)
            if not isinstance(result, torch.Tensor) or result.shape != orders.shape:
                raise ValueError("score_permutations must return slot probabilities [B,M,K]")
            return result[:, 0] if single else result
        scores = self.model.predict(batch.inter).scores
        if isinstance(scores, Mapping):
            scores = scores[self.model.LABEL]
        return _tensor(scores, "model prediction", batch.pred.batch_size)
