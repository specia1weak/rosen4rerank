"""One chunked tensor path for encoded candidate slates and fixed context."""
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..base import select_by_order


@dataclass(frozen=True)
class EncodedSlate:
    """Input-slot features [B,N,...], priors [B,N,1], mask [B,N].

    context is the model's typed encoding of fixed history. Its original batch
    axis is retained and shared across candidate chunks.
    """
    items: torch.Tensor
    dense: torch.Tensor
    mask: torch.Tensor
    context: Any = None


@dataclass(frozen=True)
class OCPMContext:
    history_pages: torch.Tensor
    page_mask: torch.Tensor


def _pad_slots(tensor, width):
    padding = width - tensor.shape[2]
    if not padding:
        return tensor
    zeros = tensor.new_zeros((*tensor.shape[:2], padding, *tensor.shape[3:]))
    return torch.cat((tensor, zeros), dim=2)


def score_candidate_logits(encoded: EncodedSlate, orders: torch.Tensor, *, scorer: Callable,
                           chunk_size: int = 32, candidate_dense: torch.Tensor | None = None):
    """Gather/right-pad K items and score [B,chunk,N,...] with an explicit mask."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    b, n = encoded.items.shape[:2]
    if encoded.dense.shape != (b, n, 1) or encoded.mask.shape != (b, n):
        raise ValueError("Encoded priors and mask must align to item slots [B,N].")
    if orders.ndim == 2:
        orders = orders[None].expand(b, -1, -1)
    if orders.ndim != 3 or orders.shape[0] != b or not orders.shape[1] or not 1 <= orders.shape[2] <= n:
        raise ValueError("Candidate orders require [B,M,K], M > 0 and 1 <= K <= N.")
    if candidate_dense is not None and candidate_dense.shape != (*orders.shape, 1):
        raise ValueError("Candidate dense scores must be [B,M,K,1].")
    outputs = []
    for start in range(0, orders.shape[1], chunk_size):
        chunk = orders[:, start:start + chunk_size]
        items = _pad_slots(select_by_order(encoded.items, chunk), n)
        dense = (select_by_order(encoded.dense, chunk) if candidate_dense is None else
                 candidate_dense[:, start:start + chunk.shape[1]])
        dense = _pad_slots(dense, n)
        mask = _pad_slots(select_by_order(encoded.mask, chunk), n)
        logits = scorer(items, dense, mask, encoded.context)
        if logits.shape != (b, chunk.shape[1], n):
            raise ValueError("Encoded slate scorer must return logits [B,M,N].")
        outputs.append(logits[..., :chunk.shape[-1]])
    return torch.cat(outputs, dim=1)


def score_candidate_probs(encoded: EncodedSlate, orders: torch.Tensor, *, scorer: Callable,
                          chunk_size: int = 32, candidate_dense: torch.Tensor | None = None):
    """All judges return probabilities with invalid selected slots zeroed."""
    logits = score_candidate_logits(encoded, orders, scorer=scorer,
                                    chunk_size=chunk_size, candidate_dense=candidate_dense)
    return logits.sigmoid() * select_by_order(encoded.mask, orders)
