from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from rosen.core.formatters import (
    DEFAULT_FALLBACK_FORMATTER,
    ColumnFormatter,
    DenseFormatter,
    FallbackFormatter,
    IntFormatter,
    PaddedFloatSequenceFormatter,
    PaddedIntSequenceFormatter,
    PaddedNestedSequenceFormatter,
    PaddingSide,
    TruncateMode,
    VectorDenseFormatter,
)
from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager

__all__ = [
    "PaddingSide", "TruncateMode", "ColumnFormatter", "DenseFormatter",
    "VectorDenseFormatter", "IntFormatter", "FallbackFormatter",
    "PaddedIntSequenceFormatter", "PaddedFloatSequenceFormatter",
    "PaddedNestedSequenceFormatter", "DEFAULT_FALLBACK_FORMATTER",
    "FeatureContext", "TensorFormatter",
]


def _dedupe_names(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(name for name in names if name))


@dataclass(frozen=True)
class FeatureContext:
    manager: SchemaManager
    read_col_names: tuple[str, ...]
    output_col_names: tuple[str, ...]
    extra_col_formatters: Mapping[str, ColumnFormatter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "read_col_names", tuple(self.read_col_names))
        object.__setattr__(self, "output_col_names", tuple(self.output_col_names))
        object.__setattr__(
            self, "extra_col_formatters", MappingProxyType(dict(self.extra_col_formatters)),
        )

    def __reduce__(self):
        # MappingProxyType itself is not picklable; DataLoader spawn needs the context.
        return (type(self), (
            self.manager, self.read_col_names, self.output_col_names,
            dict(self.extra_col_formatters),
        ))

    @classmethod
    def _build(
        cls,
        manager: SchemaManager,
        base_read_cols: Iterable[str],
        extra_col_names: Iterable[str] | None,
        extra_col_formatters: Mapping[str, ColumnFormatter] | None,
        *,
        include_labels: bool,
    ) -> FeatureContext:
        formatter_map = dict(extra_col_formatters or {})
        extra_names = list(extra_col_names or [])
        output_cols = _dedupe_names([*manager.fields(), *extra_names, *formatter_map.keys()])
        read_cols = _dedupe_names([*base_read_cols, *extra_names, *formatter_map.keys()])
        if not include_labels:
            label_set = set(manager.label_fields)
            output_cols = [name for name in output_cols if name not in label_set]
            read_cols = [name for name in read_cols if name not in label_set]
        return cls(
            manager=manager,
            read_col_names=tuple(read_cols),
            output_col_names=tuple(output_cols),
            extra_col_formatters=formatter_map,
        )

    @classmethod
    def from_manager(
        cls,
        manager: SchemaManager,
        extra_col_names: Iterable[str] | None = None,
        extra_col_formatters: Mapping[str, ColumnFormatter] | None = None,
        *,
        include_labels: bool = True,
    ) -> FeatureContext:
        """Build an encoded-data context; inference may omit labels while retaining masks."""
        return cls._build(
            manager, manager.fields(), extra_col_names, extra_col_formatters,
            include_labels=include_labels,
        )

    @classmethod
    def from_raw_manager(
        cls,
        manager: SchemaManager,
        extra_col_names: Iterable[str] | None = None,
        extra_col_formatters: Mapping[str, ColumnFormatter] | None = None,
        *,
        include_labels: bool = True,
    ) -> FeatureContext:
        """Build a raw-data context using pre-transform source columns."""
        return cls._build(
            manager, manager.raw_fields(), extra_col_names, extra_col_formatters,
            include_labels=include_labels,
        )


@dataclass
class TensorFormatter:
    context: FeatureContext
    fallback_formatter: ColumnFormatter = DEFAULT_FALLBACK_FORMATTER
    col_formatters: dict[str, ColumnFormatter] = field(init=False)

    def __post_init__(self) -> None:
        self.col_formatters = self._compile_formatters()

    def _compile_formatters(self) -> dict[str, ColumnFormatter]:
        formatters = self.context.manager.get_formatters()
        formatters.update(self.context.extra_col_formatters)
        return formatters

    def format_tensors(self, batch_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        tensor_dict: dict[str, torch.Tensor] = {}
        for col, data in batch_dict.items():
            formatter = self.col_formatters.get(col, self.fallback_formatter)
            try:
                tensor_dict[col] = formatter.format(data)
            except Exception as exc:
                raise RuntimeError(
                    f"列 [{col}] 在转换为 Tensor 时发生错误，匹配到的 formatter 为: {formatter}。错误信息: {exc}"
                ) from exc
        return tensor_dict

    def format(self, batch_dict: Mapping[str, Any]) -> Interaction:
        return Interaction(self.format_tensors(batch_dict))
