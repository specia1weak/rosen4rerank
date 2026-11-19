"""Resolved tensor inputs passed from BatchResolver to evaluators."""
from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass(frozen=True)
class EvalBatch:
    labels: torch.Tensor
    scores: Optional[torch.Tensor] = None
    ref_scores: Optional[torch.Tensor] = None
    order: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    users: Optional[torch.Tensor] = None
    full_scores: Optional[torch.Tensor] = None

    def __post_init__(self):
        labels = self.labels
        if not isinstance(labels, torch.Tensor) or labels.ndim == 0:
            raise ValueError("EvalBatch.labels requires a Tensor with a batch dimension")
        for field in fields(self):
            tensor = getattr(self, field.name)
            if tensor is not None and (not isinstance(tensor, torch.Tensor) or tensor.ndim == 0
                                      or tensor.shape[0] != labels.shape[0]):
                raise ValueError(f"EvalBatch.{field.name} must be a Tensor with the same batch dimension")

    @property
    def batch_size(self):
        return self.labels.shape[0]
