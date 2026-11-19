"""Semantic model outputs. Every tensor retains a leading batch dimension."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Optional

import torch

TensorOrTaskMap = torch.Tensor | Mapping[str, torch.Tensor]


def _validate_tensor(tensor: torch.Tensor, field_name: str, check_fn: Callable[[torch.Tensor], None]) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Prediction.{field_name} requires Tensor values.")
    if tensor.ndim == 0:
        raise ValueError(f"Prediction.{field_name} tensors require a batch dimension.")
    check_fn(tensor)
    return tensor


def _validate_field(
    value: Optional[TensorOrTaskMap],
    field_name: str,
    check_fn: Callable[[torch.Tensor], None],
    *,
    allow_tensor: bool = True,
    allow_mapping: bool = True,
) -> tuple[Optional[TensorOrTaskMap], list[torch.Tensor]]:
    """Validate a single Tensor or a {task_name: Tensor} mapping and freeze dicts."""
    if value is None:
        return None, []

    if isinstance(value, Mapping):
        if not allow_mapping:
            raise TypeError(f"Prediction.{field_name} must be a Tensor, not a Mapping.")
        frozen = MappingProxyType(dict(value))
        tensors = []
        for task_name, tensor in frozen.items():
            if not isinstance(task_name, str) or not task_name.strip():
                raise TypeError(f"Prediction.{field_name} requires non-empty string keys.")
            tensors.append(_validate_tensor(tensor, field_name, check_fn))
        return frozen, tensors

    if not allow_tensor:
        raise TypeError(f"Prediction.{field_name} must be a Mapping of batch tensors.")
    return value, [_validate_tensor(value, field_name, check_fn)]


def _check_scores(t: torch.Tensor) -> None:
    if t.ndim not in (1, 2) or not t.is_floating_point():
        raise ValueError("scores must be floating [B], [B, N], or [B, K] tensors.")


def _check_order(t: torch.Tensor) -> None:
    if t.ndim != 2 or t.dtype != torch.long:
        raise ValueError("order must be LongTensor[B, K], mapping output positions to input slots.")


def _check_chosen_scores(t: torch.Tensor) -> None:
    if t.ndim != 2 or not t.is_floating_point():
        raise ValueError("chosen_scores must be floating [B, K] tensors aligned to Prediction.order.")


def _check_full_scores(t: torch.Tensor) -> None:
    if t.ndim != 2 or not t.is_floating_point():
        raise ValueError("full_scores must be a floating [B, Num_Items] tensor.")


def _validate_chosen_scores_alignment(order, chosen_scores) -> None:
    if chosen_scores is None:
        return
    if order is None:
        raise ValueError("Prediction.chosen_scores requires Prediction.order.")
    if isinstance(chosen_scores, Mapping):
        for task_name, tensor in chosen_scores.items():
            if isinstance(order, Mapping):
                if task_name not in order:
                    raise ValueError(f"Prediction.chosen_scores task {task_name!r} has no matching Prediction.order.")
                selected_order = order[task_name]
            else:
                selected_order = order
            if tensor.shape != selected_order.shape:
                raise ValueError(f"Prediction.chosen_scores[{task_name!r}] must match Prediction.order shape.")
        return
    if isinstance(order, Mapping):
        raise ValueError("Single-tensor Prediction.chosen_scores cannot align with a multi-task Prediction.order.")
    if chosen_scores.shape != order.shape:
        raise ValueError("Prediction.chosen_scores must match Prediction.order shape.")


def _lookup_task_tensor(value, target, field_name):
    if not isinstance(value, Mapping):
        return value
    if target is None:
        raise ValueError(f"Prediction.{field_name} is a multi-task mapping {sorted(value)}; specify target.")
    if target not in value:
        raise KeyError(f"Task {target!r} not found in Prediction.{field_name}; available: {sorted(value)}.")
    return value[target]


@dataclass(frozen=True)
class CandidatePool:
    """M slates with slot-aligned scores and output-position → input-slot orders.

    Both tensors are [B, M, K]; scores are floating and orders are long.
    """
    scores: torch.Tensor
    orders: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.scores, torch.Tensor) or self.scores.ndim != 3 or not self.scores.is_floating_point():
            raise ValueError("CandidatePool.scores must be a floating Tensor[B, M, K].")
        if not isinstance(self.orders, torch.Tensor) or self.orders.ndim != 3 or self.orders.dtype != torch.long:
            raise ValueError("CandidatePool.orders must be a LongTensor[B, M, K].")
        if self.scores.shape != self.orders.shape:
            raise ValueError("CandidatePool.scores and CandidatePool.orders must share shape [B, M, K].")
        if self.orders.shape[1] == 0 or self.orders.shape[2] == 0:
            raise ValueError("CandidatePool requires non-empty candidates (M > 0, K > 0).")

    @property
    def batch_size(self) -> int:
        return self.orders.shape[0]

    @property
    def num_candidates(self) -> int:
        return self.orders.shape[1]

    @property
    def slate_size(self) -> int:
        return self.orders.shape[2]

    def select(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select one candidate per row, returning (order [B,K], scores [B,K])."""
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long or indices.shape != (self.batch_size,):
            raise ValueError("CandidatePool.select requires a LongTensor[B] of candidate indices.")
        if (indices < 0).any() or (indices >= self.num_candidates).any():
            raise ValueError("CandidatePool.select indices are out of range [0, M).")
        order_rows = torch.arange(self.batch_size, device=self.orders.device)
        score_rows = torch.arange(self.batch_size, device=self.scores.device)
        return (self.orders[order_rows, indices.to(self.orders.device)],
                self.scores[score_rows, indices.to(self.scores.device)])

    def _map_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "CandidatePool":
        return CandidatePool(scores=fn(self.scores), orders=fn(self.orders))

    def __getitem__(self, mask: torch.Tensor) -> "CandidatePool":
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (self.batch_size,):
            raise ValueError("CandidatePool row selection requires a boolean Tensor[B].")
        return self._map_tensors(lambda tensor: tensor[mask.to(tensor.device)])

    def to(self, device) -> "CandidatePool":
        return self._map_tensors(lambda tensor: tensor.to(device))


@dataclass(frozen=True)
class Prediction:
    """Standardized model output container.

    - scores:      Floating [B] or input-slot scores [B, N].
    - order:       LongTensor [B, K] selecting/arranging K output positions from N input slots.
    - chosen_scores: Floating [B, K] aligned with order output positions.
    - candidates:  Typed primary pool of M scored slates [B, M, K].
    - full_scores: Floating [B, |I|] full-catalog retrieval scores.
    - target:      Optional label name for single-tensor fields and the candidate pool.
    - extra:       Read-only dict of auxiliary/diagnostic batch tensors.
    """
    scores: Optional[TensorOrTaskMap] = None
    order: Optional[TensorOrTaskMap] = None
    chosen_scores: Optional[TensorOrTaskMap] = None
    candidates: Optional[CandidatePool] = None
    full_scores: Optional[torch.Tensor] = None
    target: Optional[str] = None
    extra: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.target is not None and (not isinstance(self.target, str) or not self.target.strip()):
            raise ValueError("Prediction.target must be a non-empty label name.")
        if self.candidates is not None and not isinstance(self.candidates, CandidatePool):
            raise TypeError("Prediction.candidates must be a CandidatePool or None.")

        scores, t_scores = _validate_field(self.scores, "scores", _check_scores)
        order, t_order = _validate_field(self.order, "order", _check_order)
        chosen_scores, t_chosen = _validate_field(self.chosen_scores, "chosen_scores", _check_chosen_scores)
        full_scores, t_full = _validate_field(
            self.full_scores, "full_scores", _check_full_scores, allow_mapping=False
        )
        extra, t_extra = _validate_field(
            self.extra, "extra", lambda _: None, allow_tensor=False
        )
        _validate_chosen_scores_alignment(order, chosen_scores)

        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "chosen_scores", chosen_scores)
        object.__setattr__(self, "full_scores", full_scores)
        object.__setattr__(self, "extra", extra)

        t_candidates = [self.candidates.scores, self.candidates.orders] if self.candidates is not None else []
        all_tensors = t_scores + t_order + t_chosen + t_candidates + t_full + t_extra
        if not all_tensors:
            raise ValueError("Prediction requires at least one batch tensor.")
        if len({t.shape[0] for t in all_tensors}) != 1:
            raise ValueError("All Prediction tensors must share the batch dimension.")

    def get_scores(self, target: Optional[str] = None) -> Optional[torch.Tensor]:
        """Return the score tensor, indexing by `target` if `scores` is a multi-task mapping."""
        return _lookup_task_tensor(self.scores, target if target is not None else self.target, "scores")

    def get_order(self, target: Optional[str] = None) -> Optional[torch.Tensor]:
        """Return the order tensor, indexing by `target` if `order` is a multi-task mapping."""
        return _lookup_task_tensor(self.order, target if target is not None else self.target, "order")

    def get_chosen_scores(self, target: Optional[str] = None) -> Optional[torch.Tensor]:
        """Return explicit order-aligned scores, selecting a task when necessary."""
        return _lookup_task_tensor(self.chosen_scores, target if target is not None else self.target, "chosen_scores")

    def resolve_chosen_scores(self, target: Optional[str] = None) -> torch.Tensor:
        """Use explicit chosen scores, or gather direct-model input scores by order.

        Candidate-pool models must supply explicit chosen_scores: their original
        slate scores do not predict the selected slate's position-dependent scores.
        """
        chosen = self.get_chosen_scores(target)
        if chosen is not None:
            return chosen
        if self.candidates is not None:
            raise ValueError("Candidate-pool Prediction must provide explicit chosen_scores; cannot gather original-order scores.")
        scores, order = self.get_scores(target), self.get_order(target)
        if scores is None or order is None:
            raise ValueError("Resolving chosen_scores requires explicit chosen_scores or both scores and order.")
        if scores.ndim != 2:
            raise ValueError("Gathering chosen_scores requires slot scores [B, N].")
        return scores.gather(-1, order.to(scores.device))

    def with_extra(self, updates: Optional[Mapping[str, torch.Tensor]] = None, /,
                   **kwargs: torch.Tensor) -> "Prediction":
        """Add diagnostic tensors while preserving every typed prediction field."""
        merged = dict(self.extra)
        if updates is not None:
            merged.update(updates)
        merged.update(kwargs)
        return Prediction(scores=self.scores, order=self.order, chosen_scores=self.chosen_scores,
                          candidates=self.candidates, full_scores=self.full_scores, target=self.target, extra=merged)

    @property
    def batch_size(self) -> int:
        for value in (self.scores, self.order, self.chosen_scores, self.full_scores, self.extra):
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, Mapping) and value:
                return next(iter(value.values())).shape[0]
        if self.candidates is not None:
            return self.candidates.batch_size
        raise RuntimeError("Prediction contains no tensors.")

    def _map_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Prediction":
        def apply(value):
            if value is None:
                return None
            if isinstance(value, Mapping):
                return {name: fn(tensor) for name, tensor in value.items()}
            return fn(value)

        return Prediction(
            scores=apply(self.scores),
            order=apply(self.order),
            chosen_scores=apply(self.chosen_scores),
            candidates=self.candidates._map_tensors(fn) if self.candidates is not None else None,
            full_scores=apply(self.full_scores),
            target=self.target,
            extra=apply(self.extra),
        )

    def __getitem__(self, mask: torch.Tensor) -> "Prediction":
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (self.batch_size,):
            raise ValueError("Prediction row selection requires a boolean Tensor[B].")
        return self._map_tensors(lambda tensor: tensor[mask.to(tensor.device)])

    def to(self, device) -> "Prediction":
        """Move all batch tensors while preserving index dtypes."""
        return self._map_tensors(lambda tensor: tensor.to(device))


def as_prediction(value, *, target: Optional[str] = None) -> Prediction:
    """Wrap raw Tensor outputs into a Prediction; reject legacy raw dicts."""
    if isinstance(value, Prediction):
        return value
    if isinstance(value, torch.Tensor):
        return Prediction(scores=value, target=target)
    raise TypeError("predict() must return Prediction or Tensor; migrate dictionary outputs to Prediction fields.")
