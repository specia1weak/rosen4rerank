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


def select_by_order(slate_tensor: torch.Tensor, perm_indices: torch.Tensor) -> torch.Tensor:
    """Gather [B,N,...] using shared [M,K] or row-wise [B,M,K] indices.

    This tensor primitive also transports repeated training decoder choices;
    external scored orders are checked for uniqueness at score_permutations.
    """
    if slate_tensor.ndim < 2 or perm_indices.ndim not in (2, 3):
        raise ValueError("Expected slate [B,K,...] and permutations [M,K] or [B,M,K].")
    if perm_indices.dtype != torch.long or perm_indices.device != slate_tensor.device:
        raise ValueError("Permutation indices must be long tensors on the slate device.")
    if perm_indices.ndim == 2:
        perm_indices = perm_indices.unsqueeze(0).expand(slate_tensor.shape[0], -1, -1)
    b, m, k = perm_indices.shape
    if b != slate_tensor.shape[0] or m == 0 or not 1 <= k <= slate_tensor.shape[1]:
        raise ValueError("Order batch/slot dimensions require matching B, M > 0 and 1 <= K <= N.")
    if (perm_indices < 0).any() or (perm_indices >= slate_tensor.shape[1]).any():
        raise ValueError("Selection indices must be in [0,N).")
    rest = slate_tensor.shape[2:]
    indices = perm_indices.reshape(b, m * k, *((1,) * len(rest))).expand(b, m * k, *rest)
    return slate_tensor.gather(1, indices).reshape(b, m, k, *rest)


def masked_bce_with_logits(logits, labels, mask=None,
                           reduction: Literal["mean", "valid_mean", "nonzero_mean"] = "mean"):
    """Divide weighted BCE by slot count, weight sum, or nonzero weight count."""
    if logits.shape != labels.shape or (mask is not None and mask.shape != logits.shape):
        raise ValueError("Logits, labels and optional mask must have identical shapes.")
    if reduction not in ("mean", "valid_mean", "nonzero_mean"):
        raise ValueError("reduction must be 'mean', 'valid_mean' or 'nonzero_mean'.")
    weight = torch.ones_like(logits) if mask is None else mask.to(logits.dtype)
    # Invalid labels may be arbitrary sentinels; exclude them before evaluating BCE.
    safe_labels = torch.where(weight != 0, labels, 0).to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none") * weight
    if reduction == "mean":
        return loss.mean()
    denominator = weight.count_nonzero() if reduction == "nonzero_mean" else weight.sum()
    return loss.sum() / denominator.clamp_min(1)


class RerankModel(BaseModel):
    requires_full_slate = False
    outputs_candidate_pool = False
    select_by_order = staticmethod(select_by_order)
    fold_history_pages = staticmethod(fold_history_pages)
    build_full_permutations = staticmethod(build_full_permutations)
    masked_bce_with_logits = staticmethod(masked_bce_with_logits)

    def score_permutations(self, interaction, orders):
        """Return slot probabilities [B,M,K] for explicit selections, 1 <= K <= N."""
        raise NotImplementedError

    @classmethod
    def from_manager(cls, manager, **kwargs):
        accepted = set()
        for ancestor in cls.__mro__:
            parameters = inspect.signature(ancestor.__init__).parameters
            accepted.update(name for name, p in parameters.items()
                            if name not in ("self", "manager") and p.kind in (
                                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                break
        ignored = sorted(set(kwargs) - accepted)
        if ignored:
            warnings.warn(f"{cls.__name__}.from_manager ignored unsupported kwargs: {ignored}", stacklevel=2)
        return cls(manager, **{k: v for k, v in kwargs.items() if k not in ignored})
