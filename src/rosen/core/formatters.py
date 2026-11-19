"""NumPy/Torch column conversion primitives, independent of schema and datasets."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

PaddingSide = Literal["left", "right"]
TruncateMode = Literal["tail", "head"]


def _is_missing_sequence(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, np.ndarray) and value.ndim == 0:
        item = value.item()
        return item is None or (isinstance(item, (float, np.floating)) and bool(np.isnan(item)))
    return False


def _as_sequence(value: Any) -> Sequence[Any] | np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return value
    return list(value)


def _validate_pad_args(padding_side: PaddingSide, truncate_mode: TruncateMode) -> None:
    if padding_side not in ("left", "right"):
        raise ValueError(f"Unknown padding_side: {padding_side}")
    if truncate_mode not in ("tail", "head"):
        raise ValueError(f"Unknown truncate_mode: {truncate_mode}")


def _pad_list(
    sequences: Any,
    max_len: int,
    padding_side: PaddingSide = "right",
    pad_val: int | float = 0,
    dtype: Any = np.int64,
    truncate_mode: TruncateMode = "tail",
) -> np.ndarray:
    _validate_pad_args(padding_side, truncate_mode)
    padded = np.full((len(sequences), max_len), pad_val, dtype=dtype)
    for i, seq in enumerate(sequences):
        if _is_missing_sequence(seq):
            continue
        seq_vals = _as_sequence(seq)
        valid_len = min(len(seq_vals), max_len)
        if valid_len == 0:
            continue
        trunc_seq = seq_vals[-valid_len:] if truncate_mode == "tail" else seq_vals[:valid_len]
        if padding_side == "right":
            padded[i, :valid_len] = trunc_seq
        else:
            padded[i, max_len - valid_len:] = trunc_seq
    return padded


def _pad_nested_list(
    sequences: Any,
    max_seq_len: int,
    max_tag_len: int,
    padding_side: PaddingSide = "right",
    pad_val: int = 0,
    truncate_mode: TruncateMode = "tail",
    tag_truncate_mode: TruncateMode = "head",
) -> np.ndarray:
    _validate_pad_args(padding_side, truncate_mode)
    if tag_truncate_mode not in ("tail", "head"):
        raise ValueError(f"Unknown tag_truncate_mode: {tag_truncate_mode}")
    padded = np.full((len(sequences), max_seq_len, max_tag_len), pad_val, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if _is_missing_sequence(seq):
            continue
        seq_vals = _as_sequence(seq)
        valid_seq_len = min(len(seq_vals), max_seq_len)
        if valid_seq_len == 0 or max_tag_len == 0:
            continue
        trunc_seq = seq_vals[-valid_seq_len:] if truncate_mode == "tail" else seq_vals[:valid_seq_len]
        start_idx = 0 if padding_side == "right" else max_seq_len - valid_seq_len
        for j in range(valid_seq_len):
            tags = trunc_seq[j]
            if _is_missing_sequence(tags):
                continue
            tags_vals = _as_sequence(tags)
            valid_tag_len = min(len(tags_vals), max_tag_len)
            if valid_tag_len == 0:
                continue
            trunc_tags = (
                tags_vals[:valid_tag_len] if tag_truncate_mode == "head" else tags_vals[-valid_tag_len:]
            )
            padded[i, start_idx + j, :valid_tag_len] = trunc_tags
    return padded


class ColumnFormatter(ABC):
    @abstractmethod
    def format(self, data: Any) -> torch.Tensor:
        pass


@dataclass(frozen=True)
class DenseFormatter(ColumnFormatter):
    def format(self, data: Any) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)


@dataclass(frozen=True)
class VectorDenseFormatter(ColumnFormatter):
    dim: int
    zero_fill: bool = True

    def format(self, data: Any) -> torch.Tensor:
        arr = np.zeros((len(data), self.dim), dtype=np.float32)
        for i, row in enumerate(data):
            if _is_missing_sequence(row):
                row_arr = np.empty(0, dtype=np.float32)
            else:
                row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
            if row_arr.size == 0:
                if not self.zero_fill:
                    raise ValueError(
                        f"Expected non-empty vector dim={self.dim}, but got an empty value at row {i}."
                    )
                continue
            use_len = min(row_arr.shape[0], self.dim)
            arr[i, :use_len] = row_arr[:use_len]
        return torch.from_numpy(arr)


@dataclass(frozen=True)
class IntFormatter(ColumnFormatter):
    def format(self, data: Any) -> torch.Tensor:
        if getattr(data, "dtype", None) == np.uint32:
            data = data.astype(np.int64)
        return torch.tensor(data, dtype=torch.long)


@dataclass(frozen=True)
class FallbackFormatter(ColumnFormatter):
    def format(self, data: Any) -> torch.Tensor:
        dtype = getattr(data, "dtype", None)
        if dtype in (np.float32, np.float64):
            return torch.tensor(data, dtype=torch.float32)
        if dtype == np.uint32:
            return torch.tensor(data.astype(np.int64), dtype=torch.long)
        return torch.tensor(data)


@dataclass(frozen=True)
class PaddedIntSequenceFormatter(ColumnFormatter):
    max_len: int
    padding_side: PaddingSide = "right"
    pad_val: int = 0
    truncate_mode: TruncateMode = "tail"

    def format(self, data: Any) -> torch.Tensor:
        return torch.from_numpy(_pad_list(
            data, max_len=self.max_len, padding_side=self.padding_side,
            pad_val=self.pad_val, dtype=np.int64, truncate_mode=self.truncate_mode,
        ))


@dataclass(frozen=True)
class PaddedFloatSequenceFormatter(ColumnFormatter):
    max_len: int
    padding_side: PaddingSide = "right"
    pad_val: float = 0.0
    truncate_mode: TruncateMode = "tail"

    def format(self, data: Any) -> torch.Tensor:
        return torch.from_numpy(_pad_list(
            data, max_len=self.max_len, padding_side=self.padding_side,
            pad_val=self.pad_val, dtype=np.float32, truncate_mode=self.truncate_mode,
        ))


@dataclass(frozen=True)
class PaddedNestedSequenceFormatter(ColumnFormatter):
    max_seq_len: int
    max_tag_len: int
    padding_side: PaddingSide = "right"
    pad_val: int = 0
    truncate_mode: TruncateMode = "tail"
    tag_truncate_mode: TruncateMode = "head"

    def format(self, data: Any) -> torch.Tensor:
        return torch.from_numpy(_pad_nested_list(
            data, max_seq_len=self.max_seq_len, max_tag_len=self.max_tag_len,
            padding_side=self.padding_side, pad_val=self.pad_val,
            truncate_mode=self.truncate_mode, tag_truncate_mode=self.tag_truncate_mode,
        ))


DEFAULT_FALLBACK_FORMATTER = FallbackFormatter()
