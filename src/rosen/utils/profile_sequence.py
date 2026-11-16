from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import polars as pl
import pyarrow.parquet as pq


@dataclass
class ProfileSequenceOutputs:
    main_paths: list[Path]
    profile_paths: list[Path]
    manifest_path: Path


def _as_paths(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(path) for path in paths]


def _part_files(part: str | Path) -> list[Path]:
    part = Path(part)
    if part.is_dir():
        return sorted(part.glob("*.parquet"))
    return [part] if part.exists() else []


def _ordered_unique(fields: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(fields))


def _parquet_columns(files: Sequence[Path]) -> set[str]:
    if not files:
        return set()
    return set(pq.ParquetFile(files[0]).schema_arrow.names)


def _existing_input_paths(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(path) for path in paths if Path(path).exists()]


def _filter_existing_fields(paths: Sequence[Path], fields: Optional[Iterable[str]]) -> Optional[list[str]]:
    if fields is None:
        return None
    cols = _parquet_columns(paths)
    return [field for field in fields if field in cols]


def _main_select_fields(
    files: Sequence[Path],
    select_fields: Optional[Iterable[str]],
    seq_len_col: str,
) -> list[str]:
    source_cols = _parquet_columns(files)
    if select_fields is None:
        fields = [name for name in source_cols if not name.startswith("__bb_")]
    else:
        fields = [name for name in select_fields if name in source_cols or name == seq_len_col]
    if seq_len_col not in fields:
        fields.append(seq_len_col)
    return _ordered_unique(fields)


def build_profile_sequence_partitions(
    flat_part_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    group_name: str,
    user_col: str,
    time_col: str,
    feature_mapping: Mapping[str, str],
    seq_len_col: str,
    label_col: Optional[str] = None,
    positive_label: Any = 1,
    select_fields: Optional[Iterable[str]] = None,
    redo: bool = True,
    compression: str = "zstd",
) -> ProfileSequenceOutputs:
    """Build lightweight main parts plus row-per-user sequence profiles.

    Each input partition must contain a disjoint user set. The main output keeps
    current-sample fields and a cursor (`seq_len_col`). The profile output stores
    full positive histories for each user with aligned list columns.
    """

    output_dir = Path(output_dir)
    main_dir = output_dir / "main_parts"
    profile_dir = output_dir / "profiles" / group_name
    manifest_path = output_dir / "profile_manifest.json"

    if output_dir.exists() and redo:
        shutil.rmtree(output_dir)
    main_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    main_paths: list[Path] = []
    profile_paths: list[Path] = []
    part_dirs = _as_paths(flat_part_dirs)

    for idx, part in enumerate(part_dirs):
        files = _part_files(part)
        if not files:
            continue

        main_path = main_dir / f"{part.name}.parquet"
        profile_path = profile_dir / f"{part.name}.parquet"
        main_paths.append(main_path)
        profile_paths.append(profile_path)

        if main_path.exists() and profile_path.exists() and not redo:
            continue

        cols = _parquet_columns(files)
        missing = [col for col in [user_col, time_col, *(feature_mapping.keys())] if col not in cols]
        if label_col is not None and label_col not in cols:
            missing.append(label_col)
        if missing:
            raise ValueError(f"Missing columns in {part}: {sorted(set(missing))}")

        is_positive = (
            (pl.col(label_col) == positive_label).cast(pl.UInt32)
            if label_col is not None
            else pl.lit(1, dtype=pl.UInt32)
        )

        ordered_lf = (
            pl.scan_parquet([str(path) for path in files])
            .with_row_index("__bb_row_nr")
            .sort([user_col, time_col, "__bb_row_nr"])
            .with_columns(is_positive.alias("__bb_is_positive"))
            .with_columns(
                (
                    pl.col("__bb_is_positive").cum_sum().over(user_col)
                    - pl.col("__bb_is_positive")
                )
                .cast(pl.UInt32)
                .alias(seq_len_col)
            )
        )

        profile_exprs = [pl.col(src_col) for src_col in feature_mapping.keys()]
        (
            ordered_lf
            .filter(pl.col("__bb_is_positive") == 1)
            .group_by(user_col, maintain_order=True)
            .agg(profile_exprs)
            .rename(dict(feature_mapping))
            .sink_parquet(profile_path, compression=compression)
        )

        main_fields = _main_select_fields(files, select_fields, seq_len_col)
        (
            ordered_lf
            .select(main_fields)
            .sink_parquet(main_path, compression=compression)
        )
        print(f"    -> profile seq part {idx + 1}/{len(part_dirs)}: {main_path}")

    manifest = {
        "group_name": group_name,
        "user_col": user_col,
        "time_col": time_col,
        "seq_len_col": seq_len_col,
        "label_col": label_col,
        "positive_label": positive_label,
        "feature_mapping": dict(feature_mapping),
        "main_paths": [str(path) for path in main_paths],
        "profile_paths": [str(path) for path in profile_paths],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return ProfileSequenceOutputs(
        main_paths=main_paths,
        profile_paths=profile_paths,
        manifest_path=manifest_path,
    )


def write_filtered_parquet(
    input_paths: Iterable[str | Path],
    out_path: str | Path,
    *,
    fields: Iterable[str],
    filter_expr: Optional[pl.Expr] = None,
    compression: str = "zstd",
    redo: bool = True,
) -> Path:
    out_path = Path(out_path)
    if out_path.exists() and not redo:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = _existing_input_paths(input_paths)
    selected_fields = _filter_existing_fields(paths, fields) or []
    lf = pl.scan_parquet([str(path) for path in paths])
    if filter_expr is not None:
        lf = lf.filter(filter_expr)
    lf.select(selected_fields).sink_parquet(out_path, compression=compression)
    return out_path
