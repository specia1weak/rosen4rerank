from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class SplitBucketedOutputs:
    train_dir: Path
    valid_path: Path
    test_path: Path
    train_paths: list[Path]
    row_counts: dict[str, int]
    manifest_path: Path


def _resolve_parquet_paths(paths: Iterable[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.parquet")))
        elif path.exists():
            resolved.append(path)
    if not resolved:
        raise FileNotFoundError("No input parquet files found.")
    return resolved


def _parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _select_existing_fields(paths: list[Path], fields: Optional[Iterable[str]]) -> Optional[list[str]]:
    if fields is None:
        return None
    existing = set(_parquet_columns(paths[0]))
    return [field for field in fields if field in existing]


def _count_parquet_rows(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        total += int(pq.ParquetFile(path).metadata.num_rows)
    return total


class _ParquetWriterGroup:
    def __init__(self, compression: str):
        self.compression = compression
        self._writers: dict[str, pq.ParquetWriter] = {}

    def write(self, key: str, path: Path, df: pl.DataFrame) -> int:
        if df.height == 0:
            return 0
        table = df.to_arrow()
        if key not in self._writers:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writers[key] = pq.ParquetWriter(
                str(path),
                table.schema,
                compression=self.compression,
            )
        self._writers[key].write_table(table)
        return int(df.height)

    def close(self) -> None:
        first_error = None
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                first_error = first_error or exc
        self._writers.clear()
        if first_error is not None:
            raise first_error


def _empty_table_from_schema(path: Path, fields: Optional[list[str]]) -> pa.Table:
    schema = pq.ParquetFile(path).schema_arrow
    if fields is not None:
        schema = pa.schema([field for field in schema if field.name in fields])
    arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_empty_if_missing(path: Path, source_path: Path, fields: Optional[list[str]], compression: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_empty_table_from_schema(source_path, fields), path, compression=compression)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _hash_range_target_expr(hash_col: str, target_count: int) -> pl.Expr:
    if target_count == 1:
        return pl.lit(0, dtype=pl.UInt32)
    if _is_power_of_two(target_count):
        shift = 64 - (target_count.bit_length() - 1)
        step = 1 << shift
        return (pl.col(hash_col) // step).cast(pl.UInt32)
    return (
        ((pl.col(hash_col).cast(pl.Float64) / float(1 << 64)) * target_count)
        .floor()
        .clip(0, target_count - 1)
        .cast(pl.UInt32)
    )


def write_split_bucketed_main(
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    fields: Optional[Iterable[str]] = None,
    train_filter: pl.Expr,
    valid_filter: pl.Expr,
    test_filter: pl.Expr,
    train_bucket_count: int = 64,
    seed: int = 2026,
    batch_size: int = 262_144,
    compression: str = "zstd",
    redo: bool = True,
    train_shuffle_mode: Literal["target_sort", "append_bucket"] = "target_sort",
    shuffle_train_bucket_order: bool = False,
) -> SplitBucketedOutputs:
    """Split main parquet parts and write train rows into bucketed parquet files.

    The input is scanned once. Train rows are distributed by row-level hash into
    `train.shuffle/part-*.parquet`; valid/test are appended to single parquet
    files. target_sort range-partitions train rows by hash high bits and sorts
    each target by the full hash, which removes source-part append-order leakage.
    ``shuffle_train_bucket_order`` additionally randomizes the order in which
    the sorted targets are assigned to the output part files.
    """

    paths = _resolve_parquet_paths(input_paths)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train.shuffle"
    valid_path = output_dir / "valid.parquet"
    test_path = output_dir / "test.parquet"
    manifest_path = output_dir / "split_manifest.json"

    if train_bucket_count <= 0:
        raise ValueError("train_bucket_count must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if train_shuffle_mode not in {"target_sort", "append_bucket"}:
        raise ValueError("train_shuffle_mode must be either 'target_sort' or 'append_bucket'.")
    if shuffle_train_bucket_order and train_shuffle_mode != "target_sort":
        raise ValueError("shuffle_train_bucket_order requires train_shuffle_mode='target_sort'.")

    if not redo and train_dir.exists() and valid_path.exists() and test_path.exists():
        train_paths = sorted(train_dir.glob("*.parquet"))
        return SplitBucketedOutputs(
            train_dir=train_dir,
            valid_path=valid_path,
            test_path=test_path,
            train_paths=train_paths,
            row_counts={
                "train": _count_parquet_rows(train_paths),
                "valid": _count_parquet_rows([valid_path]),
                "test": _count_parquet_rows([test_path]),
            },
            manifest_path=manifest_path,
        )

    if output_dir.exists() and redo:
        if train_dir.exists():
            shutil.rmtree(train_dir)
        for path in (valid_path, test_path, manifest_path, output_dir / "_train_shuffle_tmp"):
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    selected_fields = _select_existing_fields(paths, fields)
    if selected_fields is None:
        read_columns = None
    else:
        # The split filters are expected to use columns already in selected_fields
        # for the current preprocessing scripts.
        read_columns = selected_fields

    bucket_paths = [
        train_dir / f"part-{bucket_id:03d}.parquet"
        for bucket_id in range(train_bucket_count)
    ]
    for path in bucket_paths:
        if path.exists():
            path.unlink()

    writers = _ParquetWriterGroup(compression=compression)
    row_counts = {"train": 0, "valid": 0, "test": 0}
    bucket_row_counts = {bucket_id: 0 for bucket_id in range(train_bucket_count)}
    tmp_cols = ["__bb_file_idx", "__bb_row_idx", "__bb_hash", "__bb_bucket"]
    staging_root = output_dir / "_train_shuffle_tmp"
    staging_paths = [
        staging_root / f"part-{bucket_id:05d}.parquet"
        for bucket_id in range(train_bucket_count)
    ]
    bucket_expr = (
        _hash_range_target_expr("__bb_hash", train_bucket_count)
        if train_shuffle_mode == "target_sort"
        else (pl.col("__bb_hash") % train_bucket_count).cast(pl.UInt32)
    )
    if train_shuffle_mode == "target_sort":
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

    try:
        for file_idx, path in enumerate(paths):
            row_offset = 0
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
                df = pl.from_arrow(batch)
                if selected_fields is not None:
                    df = df.select(selected_fields)

                train_df = df.filter(train_filter)
                valid_df = df.filter(valid_filter)
                test_df = df.filter(test_filter)

                if train_df.height > 0:
                    train_df = (
                        train_df
                        .with_row_index("__bb_row_idx", offset=row_offset)
                        .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
                        .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias("__bb_hash"))
                        .with_columns(bucket_expr.alias("__bb_bucket"))
                    )

                    for key, bucket_df in train_df.partition_by(
                        "__bb_bucket",
                        maintain_order=False,
                        include_key=True,
                        as_dict=True,
                    ).items():
                        bucket_id = int(key[0] if isinstance(key, tuple) else key)
                        if train_shuffle_mode == "target_sort":
                            written = writers.write(
                                f"stage-{bucket_id}",
                                staging_paths[bucket_id],
                                bucket_df,
                            )
                        else:
                            clean_df = bucket_df.drop(tmp_cols)
                            written = writers.write(
                                f"train-{bucket_id}",
                                bucket_paths[bucket_id],
                                clean_df,
                            )
                        row_counts["train"] += written
                        bucket_row_counts[bucket_id] += written

                row_counts["valid"] += writers.write("valid", valid_path, valid_df)
                row_counts["test"] += writers.write("test", test_path, test_df)
                row_offset += batch.num_rows

            print(f"    -> split bucketize {file_idx + 1}/{len(paths)}: {path}")
    finally:
        writers.close()

    output_target_order = list(range(train_bucket_count))
    if shuffle_train_bucket_order:
        random.Random(seed).shuffle(output_target_order)

    if train_shuffle_mode == "target_sort":
        for output_id, target_id in enumerate(output_target_order):
            staging_path = staging_paths[target_id]
            bucket_path = bucket_paths[output_id]
            if not staging_path.exists():
                continue
            (
                pl.scan_parquet(staging_path)
                .sort("__bb_hash")
                .drop(tmp_cols)
                .sink_parquet(bucket_path, compression=compression)
            )
        shutil.rmtree(staging_root, ignore_errors=True)

    _write_empty_if_missing(valid_path, paths[0], selected_fields, compression)
    _write_empty_if_missing(test_path, paths[0], selected_fields, compression)

    train_paths = [path for path in bucket_paths if path.exists()]
    manifest = {
        "train_dir": str(train_dir),
        "valid_path": str(valid_path),
        "test_path": str(test_path),
        "train_paths": [str(path) for path in train_paths],
        "row_counts": row_counts,
        "bucket_row_counts": {
            str(output_id): int(bucket_row_counts[target_id])
            for output_id, target_id in enumerate(output_target_order)
            if bucket_row_counts[target_id] > 0
        },
        "target_bucket_row_counts": {
            str(bucket_id): int(count)
            for bucket_id, count in bucket_row_counts.items()
            if count > 0
        },
        "train_bucket_count": int(train_bucket_count),
        "train_shuffle_mode": train_shuffle_mode,
        "train_shuffle_partition": "hash_range_high_bits" if train_shuffle_mode == "target_sort" else "append_bucket",
        "shuffle_train_bucket_order": bool(shuffle_train_bucket_order),
        "output_target_bucket_order": output_target_order,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "input_paths": [str(path) for path in paths],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return SplitBucketedOutputs(
        train_dir=train_dir,
        valid_path=valid_path,
        test_path=test_path,
        train_paths=train_paths,
        row_counts=row_counts,
        manifest_path=manifest_path,
    )
