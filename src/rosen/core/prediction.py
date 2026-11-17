"""Semantic model outputs. Every tensor retains a leading batch dimension."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Optional

import torch


@dataclass(frozen=True)
class Prediction:
    """Scores use input-slot coordinates; order maps output positions to input slots.

    Mapping keys in scores/order are schema label names. Extra contains only
    batch tensors, never global constants. Freezing protects attributes and keys,
    not tensor contents: consumers must not mutate the shared tensors.
    """
    scores: Optional[torch.Tensor | Mapping[str, torch.Tensor]] = None
    order: Optional[torch.Tensor | Mapping[str, torch.Tensor]] = None
    full_scores: Optional[torch.Tensor] = None
    target: Optional[str] = None
    extra: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.target is not None and (not isinstance(self.target, str) or not self.target.strip()):
            raise ValueError("Prediction.target must be a non-empty label name.")
        tensors = []
        for slot in ("scores", "order", "full_scores", "extra"):
            value = getattr(self, slot)
            if value is None:
                continue
            if isinstance(value, Mapping):
                if slot == "full_scores":
                    raise TypeError("full_scores must be a Tensor.")
                value = MappingProxyType(dict(value))
                object.__setattr__(self, slot, value)
                entries = value.items()
            else:
                if slot == "extra":
                    raise TypeError("Prediction.extra must be a Mapping of batch tensors.")
                entries = ((slot, value),)
            for name, tensor in entries:
                if not isinstance(name, str) or not name.strip() or not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Prediction.{slot} requires string keys and Tensor values.")
                if tensor.ndim == 0:
                    raise ValueError(f"Prediction.{slot} tensors require a batch dimension.")
                if slot == "scores" and (tensor.ndim not in (1, 2) or not tensor.is_floating_point()):
                    raise ValueError("scores must be floating [B] or [B, K] tensors.")
                if slot == "order" and (tensor.ndim != 2 or tensor.dtype != torch.long):
                    raise ValueError("order must be LongTensor[B, K], mapping output positions to input slots.")
                if slot == "full_scores" and (tensor.ndim != 2 or not tensor.is_floating_point()):
                    raise ValueError("full_scores must be a floating [B, Num_Items] tensor.")
                tensors.append(tensor)
        if not tensors:
            raise ValueError("Prediction requires at least one batch tensor.")
        if len({tensor.shape[0] for tensor in tensors}) != 1:
            raise ValueError("All Prediction tensors must share the batch dimension.")

    @property
    def batch_size(self):
        for value in (self.scores, self.order, self.full_scores, self.extra):
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, Mapping) and value:
                return next(iter(value.values())).shape[0]
        raise RuntimeError("Prediction contains no tensors.")

    def _map_tensors(self, fn):
        def apply(value):
            if value is None:
                return None
            if isinstance(value, Mapping):
                return {name: fn(tensor) for name, tensor in value.items()}
            return fn(value)
        return Prediction(scores=apply(self.scores), order=apply(self.order),
                          full_scores=apply(self.full_scores), target=self.target, extra=apply(self.extra))

    def __getitem__(self, mask):
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (self.batch_size,):
            raise ValueError("Prediction row selection requires a boolean Tensor[B].")
        return self._map_tensors(lambda tensor: tensor[mask.to(tensor.device)])

    def to(self, device):
        """Move all batch tensors while preserving index dtypes."""
        return self._map_tensors(lambda tensor: tensor.to(device))


def as_prediction(value, *, target=None):
    """Only legacy scalar/slate Tensor outputs receive an automatic target."""
    if isinstance(value, Prediction):
        return value
    if isinstance(value, torch.Tensor):
        return Prediction(scores=value, target=target)
    raise TypeError("predict() must return Prediction or Tensor; migrate dictionary outputs to semantic slots.")
