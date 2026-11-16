from __future__ import annotations

import torch

from rosen.utils.time_bucket import bucketize_relative_time


def test_relative_time_bucket_keeps_valid_zero_delta_and_masks_padding() -> None:
    current = torch.tensor([100, 200])
    history = torch.tensor(
        [
            [100, 95, 90, 0],
            [210, 150, 0, 0],
        ]
    )
    seq_len = torch.tensor([3, 2])

    buckets = bucketize_relative_time(
        current,
        history,
        seq_lens=seq_len,
        boundaries=(5, 10),
    )

    assert buckets.tolist() == [
        [1, 1, 2, 0],
        [1, 2, 0, 0],
    ]
