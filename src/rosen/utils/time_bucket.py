from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn


# Matches the official kddsample baseline: 64 bucket boundaries in seconds,
# bucket id 0 reserved for padding, 1..64 for valid relative-time buckets.
# [-, 1), [1, 5)
TIME_BUCKET_BOUNDARIES = (
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
)


# Long-sequence ablation: preserve every baseline boundary through 24 hours,
# split 24h--72h into 6-hour intervals, and merge the sparsely used tail.  It
# deliberately has the same length as TIME_BUCKET_BOUNDARIES so model and
# checkpoint tensor shapes remain unchanged.
LONG_SEQUENCE_TIME_BUCKET_BOUNDARIES = (
    *TIME_BUCKET_BOUNDARIES[:47],
    108000, 129600, 151200, 172800,
    194400, 216000, 237600, 259200,
    345600, 432000, 604800, 1209600,
    2592000, 5184000, 15552000, 31536000,
)


def get_num_time_buckets(boundaries: Optional[Iterable[int]] = None) -> int:
    effective_boundaries = tuple(boundaries) if boundaries is not None else TIME_BUCKET_BOUNDARIES
    if not effective_boundaries:
        raise ValueError("Time bucket boundaries must be a non-empty sequence.")
    # 0 is reserved for padding, 1..N for valid buckets.
    return len(effective_boundaries) + 1


def build_padding_mask(seq_lens: torch.Tensor, max_len: int) -> torch.Tensor:
    if seq_lens.ndim != 1:
        raise ValueError(f"Expected seq_lens to have shape [B], got {tuple(seq_lens.shape)}")
    idx = torch.arange(max_len, device=seq_lens.device).unsqueeze(0)
    return idx >= seq_lens.unsqueeze(1)


def bucketize_relative_time(
    curr_ts: torch.Tensor,
    hist_ts: torch.Tensor,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    boundaries: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """
    Convert absolute timestamps into official-style relative-time bucket ids.

    Args:
        curr_ts: shape [B] or [B, 1], current interaction timestamps.
        hist_ts: shape [B, L], historical timestamps aligned to a sequence.
        seq_lens: optional shape [B], used to mask padded sequence positions.
        padding_mask: optional shape [B, L], True means padding.
        boundaries: optional iterable of ascending second-level boundaries.

    Returns:
        LongTensor with shape [B, L], where 0 is padding and 1..N are buckets.
    """
    if hist_ts.ndim != 2:
        raise ValueError(f"Expected hist_ts to have shape [B, L], got {tuple(hist_ts.shape)}")

    if curr_ts.ndim == 2 and curr_ts.shape[1] == 1:
        curr_ts = curr_ts.squeeze(1)
    if curr_ts.ndim != 1:
        raise ValueError(f"Expected curr_ts to have shape [B], got {tuple(curr_ts.shape)}")
    if curr_ts.shape[0] != hist_ts.shape[0]:
        raise ValueError(
            f"Batch size mismatch between curr_ts {tuple(curr_ts.shape)} and hist_ts {tuple(hist_ts.shape)}"
        )

    if padding_mask is None and seq_lens is not None:
        padding_mask = build_padding_mask(seq_lens, hist_ts.shape[1])
    elif padding_mask is not None and padding_mask.shape != hist_ts.shape:
        raise ValueError(
            f"Expected padding_mask to have shape {tuple(hist_ts.shape)}, got {tuple(padding_mask.shape)}"
        )

    if boundaries is None:
        boundaries = TIME_BUCKET_BOUNDARIES
    boundary_tensor = torch.as_tensor(
        tuple(boundaries),
        device=hist_ts.device,
        dtype=hist_ts.dtype,
    )
    if boundary_tensor.ndim != 1 or boundary_tensor.numel() == 0:
        raise ValueError("Time bucket boundaries must be a non-empty 1D sequence.")

    curr_ts = curr_ts.to(hist_ts.device, dtype=hist_ts.dtype)
    time_diff = torch.clamp(curr_ts.unsqueeze(1) - hist_ts, min=0)

    bucket_ids = torch.bucketize(time_diff, boundary_tensor, right=False)
    max_bucket_idx = boundary_tensor.numel() - 1
    bucket_ids = torch.clamp(bucket_ids, min=0, max=max_bucket_idx) + 1
    bucket_ids = bucket_ids.to(torch.long)

    effective_padding = hist_ts.eq(0)
    if padding_mask is not None:
        effective_padding = effective_padding | padding_mask.to(device=hist_ts.device)
    bucket_ids[effective_padding] = 0
    return bucket_ids


def bucketize_adjacent_time(
    hist_ts: torch.Tensor,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    boundaries: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Bucketize the gap from each valid history token to its predecessor.

    Bucket 0 is used for padding and for the first visible token because that
    token has no predecessor in the supplied sequence window. Valid positions
    from index 1 onward use buckets 1..N, including zero-second gaps.
    """
    if hist_ts.ndim != 2:
        raise ValueError(
            f"Expected hist_ts to have shape [B, L], got {tuple(hist_ts.shape)}"
        )
    if padding_mask is None and seq_lens is not None:
        padding_mask = build_padding_mask(seq_lens, hist_ts.shape[1])
    elif padding_mask is not None and padding_mask.shape != hist_ts.shape:
        raise ValueError(
            f"Expected padding_mask to have shape {tuple(hist_ts.shape)}, "
            f"got {tuple(padding_mask.shape)}"
        )

    effective_boundaries = tuple(boundaries or TIME_BUCKET_BOUNDARIES)
    boundary_tensor = torch.as_tensor(
        effective_boundaries,
        device=hist_ts.device,
        dtype=hist_ts.dtype,
    )
    if boundary_tensor.ndim != 1 or boundary_tensor.numel() == 0:
        raise ValueError("Time bucket boundaries must be a non-empty 1D sequence.")

    bucket_ids = torch.zeros_like(hist_ts, dtype=torch.long)
    if hist_ts.shape[1] > 1:
        gaps = torch.clamp(hist_ts[:, 1:] - hist_ts[:, :-1], min=0)
        gap_buckets = torch.bucketize(gaps, boundary_tensor, right=False)
        gap_buckets = torch.clamp(
            gap_buckets,
            min=0,
            max=boundary_tensor.numel() - 1,
        ).to(torch.long) + 1
        bucket_ids[:, 1:] = gap_buckets

    effective_padding = hist_ts.eq(0)
    if padding_mask is not None:
        effective_padding = effective_padding | padding_mask.to(device=hist_ts.device)
    bucket_ids[effective_padding] = 0
    bucket_ids[:, 0] = 0
    return bucket_ids


class RelativeTimeEmbedding(nn.Module):
    """
    Dynamic relative-time encoder built on top of bucketized time deltas.
    use seconds

    Usage:
        time_emb = RelativeTimeEmbedding(embedding_dim=64)
        rel_time = time_emb(curr_ts, hist_ts, seq_lens=seq_len)   # [B, L, 64]
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        boundaries: Optional[Iterable[int]] = None,
        padding_idx: int = 0,
        init_std: float = 1e-4,
    ):
        super().__init__()
        self.boundaries = tuple(boundaries) if boundaries is not None else TIME_BUCKET_BOUNDARIES
        self.num_buckets = get_num_time_buckets(self.boundaries)
        self.padding_idx = padding_idx
        self.init_std = init_std
        self.embedding = nn.Embedding(
            num_embeddings=self.num_buckets,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.reset_parameters()

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight.data, mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def bucketize(
        self,
        curr_ts: torch.Tensor,
        hist_ts: torch.Tensor,
        *,
        seq_lens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return bucketize_relative_time(
            curr_ts=curr_ts,
            hist_ts=hist_ts,
            seq_lens=seq_lens,
            padding_mask=padding_mask,
            boundaries=self.boundaries,
        )

    def forward(
        self,
        curr_ts: torch.Tensor,
        hist_ts: torch.Tensor,
        *,
        seq_lens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_bucket_ids: bool = False,
    ):
        bucket_ids = self.bucketize(
            curr_ts=curr_ts,
            hist_ts=hist_ts,
            seq_lens=seq_lens,
            padding_mask=padding_mask,
        )
        time_emb = self.embedding(bucket_ids)
        if return_bucket_ids:
            return time_emb, bucket_ids
        return time_emb


class AdjacentTimeEmbedding(nn.Module):
    """Embedding of within-window gaps between adjacent history tokens."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        boundaries: Optional[Iterable[int]] = None,
        padding_idx: int = 0,
        init_std: float = 1e-4,
    ):
        super().__init__()
        self.boundaries = tuple(boundaries or TIME_BUCKET_BOUNDARIES)
        self.num_buckets = get_num_time_buckets(self.boundaries)
        self.padding_idx = padding_idx
        self.init_std = init_std
        self.embedding = nn.Embedding(
            num_embeddings=self.num_buckets,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight.data, mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(
        self,
        hist_ts: torch.Tensor,
        *,
        seq_lens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_bucket_ids: bool = False,
    ):
        bucket_ids = bucketize_adjacent_time(
            hist_ts,
            seq_lens=seq_lens,
            padding_mask=padding_mask,
            boundaries=self.boundaries,
        )
        time_emb = self.embedding(bucket_ids)
        if return_bucket_ids:
            return time_emb, bucket_ids
        return time_emb
