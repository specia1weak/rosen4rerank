from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Literal, Optional

import polars as pl
import torch
from torch import nn

from rosen.core.enum_type import FeatureSource

from rosen.core.formatters import (
    ColumnFormatter,
    DenseFormatter,
    IntFormatter,
    PaddedFloatSequenceFormatter,
    PaddedIntSequenceFormatter,
    PaddingSide,
    TruncateMode,
)

_AUX_SCALAR_FORMATTERS: Dict[str, type[ColumnFormatter]] = {
    "float32": DenseFormatter,
    "int64": IntFormatter,
}
_AUX_SEQ_FORMATTERS: Dict[str, type[ColumnFormatter]] = {
    "float32": PaddedFloatSequenceFormatter,
    "int64": PaddedIntSequenceFormatter,
}


class EmbType(Enum):
    UNKNOWN = "none"
    SPARSE = "sparse"
    MULTI_SPARSE = "multi_sparse"
    QUANTILE = "quantile"
    SPARSE_SEQ = "sparse_seq"
    SPARSE_SET = "sparse_set"
    DENSE = "dense"
    LOG1P_DENSE = "log1p_dense"
    VECTOR_DENSE = "vector_dense"
    DENSE_SEQ = "dense_seq"
    SEQUENCE = "sequence"


@dataclass(frozen=True)
class AuxField:
    """Typed model input carried by the schema, without fitting or embedding."""
    field_name: str
    max_len: Optional[int] = None
    dtype: Literal["float32", "int64"] = "float32"
    padding_side: Literal["left", "right"] = "right"
    pad_val: float | int = 0.0

    def __post_init__(self):
        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise ValueError("AuxField.field_name must be a non-empty string")
        if self.max_len is not None and (isinstance(self.max_len, bool)
                or not isinstance(self.max_len, int) or self.max_len <= 0):
            raise ValueError("AuxField.max_len must be None or a positive integer")
        if self.dtype not in ("float32", "int64"):
            raise ValueError("AuxField.dtype must be float32 or int64")
        if self.padding_side not in ("left", "right"):
            raise ValueError("AuxField.padding_side must be left or right")
        if not isinstance(self.pad_val, (int, float)) or not math.isfinite(self.pad_val):
            raise ValueError("AuxField.pad_val must be finite")
        if self.dtype == "int64" and (self.pad_val != int(self.pad_val)
                or not -(2 ** 63) <= self.pad_val < 2 ** 63):
            raise ValueError("Integer AuxField.pad_val must be representable as int64")

    def get_formatters(self) -> Dict[str, ColumnFormatter]:
        if self.max_len is None:
            formatter_cls = _AUX_SCALAR_FORMATTERS.get(self.dtype)
            if formatter_cls is None:
                raise ValueError(f"Unsupported AuxField dtype: {self.dtype}")
            return {self.field_name: formatter_cls()}
        formatter_cls = _AUX_SEQ_FORMATTERS.get(self.dtype)
        if formatter_cls is None:
            raise ValueError(f"Unsupported AuxField dtype: {self.dtype}")
        pad_val = int(self.pad_val) if self.dtype == "int64" else float(self.pad_val)
        return {self.field_name: formatter_cls(
            max_len=self.max_len, padding_side=self.padding_side, pad_val=pad_val,
        )}


@dataclass(frozen=True)
class SlateLabel:
    """Fixed-width list supervision, independent from embedding features."""

    field_name: str
    max_len: int
    mask_field: Optional[str] = None
    padding_side: Literal["left", "right"] = "right"
    pad_val: float = 0.0

    def __post_init__(self):
        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise ValueError("SlateLabel.field_name must be a non-empty string.")
        if isinstance(self.max_len, bool) or not isinstance(self.max_len, int) or self.max_len <= 0:
            raise ValueError("SlateLabel.max_len must be a positive integer.")
        if self.padding_side not in ("left", "right"):
            raise ValueError("SlateLabel.padding_side must be 'left' or 'right'.")
        if self.mask_field is not None and (
            not isinstance(self.mask_field, str) or not self.mask_field.strip()
        ):
            raise ValueError("SlateLabel.mask_field must be None or a non-empty string.")

    def get_formatters(self) -> Dict[str, ColumnFormatter]:
        formatters: Dict[str, ColumnFormatter] = {
            self.field_name: PaddedFloatSequenceFormatter(
                max_len=self.max_len, padding_side=self.padding_side, pad_val=self.pad_val,
            )
        }
        if self.mask_field:
            formatters[self.mask_field] = PaddedFloatSequenceFormatter(
                max_len=self.max_len, padding_side=self.padding_side, pad_val=0.0,
            )
        return formatters


@dataclass
class SeqGroupConfig:
    group_name: str
    seq_len_field_name: str
    max_len: int
    padding_side: Literal["left", "right"] = "right"
    time_field_name: Optional[str] = None


@dataclass
class ProfileSeqGroupConfig(SeqGroupConfig):
    owner_field_name: str = "user_id"
    id_field_name: Optional[str] = None
    anchor_field_name: Optional[str] = None
    profile_name: Optional[str] = None
    history_time_filter: Literal[
        "cursor_order",
        "strict_time",
        "drop_same_ts",
        "same_ts_only",
    ] = "cursor_order"
    backend: Literal["profile"] = "profile"


class EmbSetting(ABC):
    emb_type = EmbType.UNKNOWN

    def __init__(
        self,
        field_name: str,
        embedding_dim: int,
        source: FeatureSource = FeatureSource.UNKNOWN,
        padding_zero: bool = True,
        use_oov: bool = True,
    ):
        self.field_name = field_name
        self.embedding_dim = embedding_dim
        self.source = source
        self.is_fitted = False

        self.padding_zero = padding_zero
        self.use_oov = use_oov
        self.vocab: Dict[str, int] = {}
        self.oov_idx: int = -1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return self.vocab_size + int(self.padding_zero) + int(self.use_oov)

    @property
    def serialized_type_name(self) -> str:
        return self.emb_type.name

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name}

    @property
    def requires_embedding_module(self) -> bool:
        return self.num_embeddings > 0

    @property
    def embedding_field_name(self) -> str:
        return self.field_name

    @property
    def is_sequence_setting(self) -> bool:
        return False

    def get_output_field_names(self) -> List[str]:
        return [self.field_name]

    def get_raw_field_names(self) -> List[str]:
        return [self.field_name]

    @abstractmethod
    def get_fit_exprs(self) -> List[pl.Expr]:
        pass

    @abstractmethod
    def parse_fit_result(self, result_df: pl.DataFrame):
        pass

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        return self.parse_fit_result(result_df)

    @abstractmethod
    def get_transform_expr(self):
        pass

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element fitting.")

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element transform.")

    @abstractmethod
    def get_formatters(self) -> Dict[str, ColumnFormatter]:
        pass

    def get_sequence_formatters(
        self,
        max_len: int,
        padding_side: PaddingSide,
        truncate_mode: TruncateMode = "tail",
    ) -> Dict[str, ColumnFormatter]:
        base_formatters = self.get_formatters()
        fmt = base_formatters[self.field_name]
        if isinstance(fmt, DenseFormatter):
            return {self.field_name: PaddedFloatSequenceFormatter(
                max_len=max_len, padding_side=padding_side, truncate_mode=truncate_mode,
            )}
        if isinstance(fmt, IntFormatter):
            return {self.field_name: PaddedIntSequenceFormatter(
                max_len=max_len, padding_side=padding_side, truncate_mode=truncate_mode,
            )}
        raise NotImplementedError(
            f"Formatter {fmt.__class__.__name__} cannot be promoted to a sequence automatically."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.serialized_type_name,
            "field_name": self.field_name,
            "embedding_size": self.embedding_dim,
            "num_embeddings": self.num_embeddings,
            "vocab_size": self.vocab_size,
            "feature_source": self.source.name,
            "is_fitted": self.is_fitted,
            "padding_zero": self.padding_zero,
            "use_oov": self.use_oov,
            "vocab": self.vocab,
            "oov_idx": self.oov_idx,
        }

    def load_state(self, state_dict: Dict[str, Any]):
        self.is_fitted = state_dict.get("is_fitted", True)
        self.vocab = state_dict.get("vocab", {})
        self.use_oov = state_dict.get("use_oov", self.use_oov)
        self.padding_zero = state_dict.get("padding_zero", self.padding_zero)
        self.oov_idx = state_dict.get("oov_idx", self.oov_idx)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbSetting":
        pass

    @abstractmethod
    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        pass
