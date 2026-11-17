"""Task-independent tensor operations for fixed-width slate models."""

import inspect
import itertools
import warnings
from typing import Literal

import torch
from torch.nn import functional as F

from rosen.models.base import BaseModel


def fold_history_pages(flat_seq: torch.Tensor, page_num: int, poi_num: int) -> torch.Tensor:
    if page_num <= 0 or poi_num <= 0 or flat_seq.ndim < 2:
        raise ValueError("History requires positive page_num/poi_num and shape [B, P*K, ...].")
    if flat_seq.shape[1] != page_num * poi_num:
        raise ValueError(f"Expected history length {page_num * poi_num}, got {flat_seq.shape[1]}.")
    return flat_seq.reshape(flat_seq.shape[0], page_num, poi_num, *flat_seq.shape[2:])


def build_full_permutations(poi_num: int) -> torch.Tensor:
    if isinstance(poi_num, bool) or not isinstance(poi_num, int) or not 1 <= poi_num <= 7:
        raise ValueError("Full permutations require an integer poi_num between 1 and 7 (factorial growth).")
    return torch.tensor(list(itertools.permutations(range(poi_num))), dtype=torch.long)


def apply_permutation(slate_tensor: torch.Tensor, perm_indices: torch.Tensor) -> torch.Tensor:
    if slate_tensor.ndim < 2 or perm_indices.ndim not in (2, 3):
        raise ValueError("Expected slate [B,K,...] and permutations [M,K] or [B,M,K].")
    if perm_indices.dtype != torch.long or perm_indices.device != slate_tensor.device:
        raise ValueError("Permutation indices must be long tensors on the slate device.")
    if perm_indices.ndim == 2:
        perm_indices = perm_indices.unsqueeze(0).expand(slate_tensor.shape[0], -1, -1)
    b, m, k = perm_indices.shape
    if b != slate_tensor.shape[0] or k != slate_tensor.shape[1]:
        raise ValueError("Permutation batch/slot dimensions must match the slate.")
    rest = slate_tensor.shape[2:]
    indices = perm_indices.reshape(b, m * k, *((1,) * len(rest))).expand(b, m * k, *rest)
    return slate_tensor.gather(1, indices).reshape(b, m, k, *rest)


def masked_bce_with_logits(logits, labels, mask=None, reduction: Literal["mean", "valid_mean"] = "mean"):
    """Mean over all slots, or over mask weight. No implicit TF equivalence."""
    if logits.shape != labels.shape or (mask is not None and mask.shape != logits.shape):
        raise ValueError("Logits, labels and optional mask must have identical shapes.")
    if reduction not in ("mean", "valid_mean"):
        raise ValueError("reduction must be 'mean' or 'valid_mean'.")
    weight = torch.ones_like(logits) if mask is None else mask.to(logits.dtype)
    # Invalid labels may be arbitrary sentinels; exclude them before evaluating BCE.
    safe_labels = torch.where(weight != 0, labels, 0).to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none") * weight
    return loss.mean() if reduction == "mean" else loss.sum() / weight.sum().clamp_min(1.0)


class RerankModel(BaseModel):
    fold_history_pages = staticmethod(fold_history_pages)
    build_full_permutations = staticmethod(build_full_permutations)
    apply_permutation = staticmethod(apply_permutation)
    masked_bce_with_logits = staticmethod(masked_bce_with_logits)

    @classmethod
    def from_manager(cls, manager, **kwargs):
        parameters = inspect.signature(cls.__init__).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return cls(manager, **kwargs)
        ignored = sorted(set(kwargs) - (set(parameters) - {"self", "manager"}))
        if ignored:
            warnings.warn(f"{cls.__name__}.from_manager ignored unsupported kwargs: {ignored}", stacklevel=2)
        return cls(manager, **{k: v for k, v in kwargs.items() if k not in ignored})
