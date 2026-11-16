from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, Literal, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def _resolve_input_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidates = [Path(paths)]
    else:
        candidates = [Path(path) for path in paths]

    resolved: list[Path] = []
    for path in candidates:
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.parquet")))
        elif path.exists():
            resolved.append(path)

    if not resolved:
        raise FileNotFoundError("No input parquet files found for shuffle checkpoint.")
    return resolved


def _parquet_columns(paths: list[Path]) -> list[str]:
    return pq.ParquetFile(paths[0]).schema_arrow.names


def _filter_existing_fields(paths: list[Path], fields: Optional[Iterable[str]]) -> Optional[list[str]]:
    if fields is None:
        return None
    existing = set(_parquet_columns(paths))
    return [field for field in fields if field in existing]


def _empty_table(paths: list[Path], fields: Optional[list[str]]) -> pa.Table:
    schema = pq.ParquetFile(paths[0]).schema_arrow
    if fields is not None:
        schema = pa.schema([field for field in schema if field.name in fields])
    arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _append_parquet_batches(
    input_paths: list[Path],
    out_path: Path,
    *,
    batch_size: int,
    compression: str,
    rand_col: str,
    shuffle_rows: bool,
    seed: int,
) -> None:
    writer = None
    wrote_rows = False
    try:
        chunk_idx = 0
        for path in input_paths:
            parquet_file = pq.ParquetFile(path)
            for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
                df = pl.from_arrow(batch)
                if df.height == 0:
                    continue
                if shuffle_rows:
                    row_seed = (int(seed) + chunk_idx * 1000003 + batch_idx * 9176) & 0xFFFFFFFF
                    df = df.sample(fraction=1.0, shuffle=True, seed=row_seed)
                if rand_col in df.columns:
                    df = df.drop(rand_col)
                table = df.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
                writer.write_table(table)
                wrote_rows = True
            chunk_idx += 1
    finally:
        if writer is not None:
            writer.close()

    if not wrote_rows:
        table = _empty_table(input_paths, None)
        pq.write_table(table, out_path, compression=compression)


class _ParquetWriterGroup:
    def __init__(self, compression: str):
        self.compression = compression
        self._writers: dict[int, pq.ParquetWriter] = {}

    def write(self, key: int, path: Path, df: pl.DataFrame) -> int:
        if df.height == 0:
            return 0
        table = df.to_arrow()
        if key not in self._writers:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writers[key] = pq.ParquetWriter(str(path), table.schema, compression=self.compression)
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


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _hash_range_target_expr(rand_col: str, target_count: int) -> pl.Expr:
    if target_count == 1:
        return pl.lit(0, dtype=pl.UInt32)
    if _is_power_of_two(target_count):
        shift = 64 - (target_count.bit_length() - 1)
        step = 1 << shift
        return (pl.col(rand_col) // step).cast(pl.UInt32)
    return (
        ((pl.col(rand_col).cast(pl.Float64) / float(1 << 64)) * target_count)
        .floor()
        .clip(0, target_count - 1)
        .cast(pl.UInt32)
    )


def _write_target_sort_shuffled_parquet(
    input_paths: list[Path],
    out_path: Path,
    *,
    selected_fields: Optional[list[str]],
    filter_expr: Optional[pl.Expr],
    seed: int,
    target_count: int,
    batch_size: int,
    compression: str,
    temp_dir: str | Path | None,
) -> None:
    """External random-key shuffle.

    Rows are first range-partitioned by the high bits of a stable row hash, then
    each target is sorted by the full hash. Concatenating targets in target id
    order is equivalent to sorting the full dataset by the random key, without
    materializing the full table in memory.
    """

    rand_col = "__bb_shuffle_id"
    target_col = "__bb_shuffle_target"
    tmp_cols = ["__bb_file_idx", "__bb_row_idx", rand_col, target_col]
    existing_cols = set(_parquet_columns(input_paths))
    conflict_cols = sorted(existing_cols.intersection(tmp_cols))
    if conflict_cols:
        raise ValueError(f"Input parquet contains reserved shuffle columns: {conflict_cols}")

    tmp_dir = Path(temp_dir) if temp_dir is not None else out_path.parent / f"_{out_path.stem}_shuffle_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    staging_root = tmp_dir / "stage"
    final_root = tmp_dir / "final"
    staging_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    staging_paths = [
        staging_root / f"part-{target_id:05d}.parquet"
        for target_id in range(target_count)
    ]
    read_columns = None if filter_expr is not None else selected_fields
    writers = _ParquetWriterGroup(compression=compression)

    try:
        for file_idx, path in enumerate(input_paths):
            row_offset = 0
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
                df = pl.from_arrow(batch)
                if filter_expr is not None:
                    df = df.filter(filter_expr)
                    if df.height == 0:
                        row_offset += batch.num_rows
                        continue
                    if selected_fields is not None:
                        df = df.select(selected_fields)
                elif selected_fields is not None:
                    df = df.select(selected_fields)

                if df.height > 0:
                    df = (
                        df
                        .with_row_index("__bb_row_idx", offset=row_offset)
                        .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
                        .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias(rand_col))
                        .with_columns(_hash_range_target_expr(rand_col, target_count).alias(target_col))
                    )

                    for key, target_df in df.partition_by(
                        target_col,
                        maintain_order=False,
                        include_key=True,
                        as_dict=True,
                    ).items():
                        target_id = int(key[0] if isinstance(key, tuple) else key)
                        writers.write(target_id, staging_paths[target_id], target_df)

                row_offset += batch.num_rows
            print(f"    -> shuffle targetize {file_idx + 1}/{len(input_paths)}")
    finally:
        writers.close()

    try:
        target_paths: list[Path] = []
        for target_id, staging_path in enumerate(staging_paths):
            if not staging_path.exists():
                continue
            final_path = final_root / f"part-{target_id:05d}.parquet"
            (
                pl.scan_parquet(staging_path)
                .sort(rand_col)
                .drop(tmp_cols)
                .sink_parquet(final_path, compression=compression)
            )
            target_paths.append(final_path)

        if out_path.exists():
            out_path.unlink()
        if target_paths:
            _append_parquet_batches(
                target_paths,
                out_path,
                batch_size=batch_size,
                compression=compression,
                rand_col=rand_col,
                shuffle_rows=False,
                seed=seed,
            )
        else:
            pq.write_table(_empty_table(input_paths, selected_fields), out_path, compression=compression)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def write_shuffled_parquet(
    input_paths: str | Path | Iterable[str | Path],
    out_path: str | Path,
    *,
    fields: Optional[Iterable[str]] = None,
    filter_expr: Optional[pl.Expr] = None,
    seed: int = 2026,
    bucket_count: int = 64,
    batch_size: int = 65536,
    compression: str = "zstd",
    redo: bool = True,
    exact: bool = False,
    mode: Literal["target_sort", "bucket_chunk"] = "target_sort",
    temp_dir: str | Path | None = None,
) -> Path:
    """Write a physically shuffled parquet checkpoint.

    The default target_sort mode performs an external random-key sort by
    range-partitioning rows into power-of-two targets, sorting each target by
    the stable row hash, and concatenating targets in key-range order.
    Set mode="bucket_chunk" to use the older low-memory chunk shuffle.
    Set exact=True only when a single in-memory/logical full sort is required.
    """

    paths = _resolve_input_paths(input_paths)
    out_path = Path(out_path)
    if out_path.exists() and not redo:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected_fields = _filter_existing_fields(paths, fields)
    rand_col = "__bb_rand"

    if exact:
        lf = pl.scan_parquet([str(path) for path in paths])
        if filter_expr is not None:
            lf = lf.filter(filter_expr)
        if selected_fields is not None:
            lf = lf.select(selected_fields)
        (
            lf
            .with_row_index("__bb_row_idx")
            .with_columns(pl.struct(["__bb_row_idx"]).hash(seed=seed).alias(rand_col))
            .sort(rand_col)
            .drop(["__bb_row_idx", rand_col])
            .sink_parquet(out_path, compression=compression)
        )
        return out_path

    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if mode not in {"target_sort", "bucket_chunk"}:
        raise ValueError("mode must be either 'target_sort' or 'bucket_chunk'.")

    if mode == "target_sort":
        _write_target_sort_shuffled_parquet(
            paths,
            out_path,
            selected_fields=selected_fields,
            filter_expr=filter_expr,
            seed=seed,
            target_count=bucket_count,
            batch_size=batch_size,
            compression=compression,
            temp_dir=temp_dir,
        )
        return out_path

    # TODO: This bucket/chunk path is not equivalent to a true global shuffle.
    # If upstream files are sorted by user/time and downstream training reads
    # output shards sequentially, large local order blocks can remain. Add a
    # round-robin/random shard reader or make exact hash-sort the default for
    # no-sequence datasets where the memory cost is acceptable.
    tmp_dir = Path(temp_dir) if temp_dir is not None else out_path.parent / f"_{out_path.stem}_shuffle_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    bucket_root = tmp_dir / "buckets"
    bucket_root.mkdir(parents=True, exist_ok=True)
    for bucket_id in range(bucket_count):
        (bucket_root / f"b{bucket_id:03d}").mkdir(parents=True, exist_ok=True)

    file_counts = {bucket_id: 0 for bucket_id in range(bucket_count)}
    read_columns = None if filter_expr is not None else selected_fields

    try:
        for file_idx, path in enumerate(paths):
            row_offset = 0
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
                df = pl.from_arrow(batch)
                if filter_expr is not None:
                    df = df.filter(filter_expr)
                    if df.height == 0:
                        row_offset += batch.num_rows
                        continue
                    if selected_fields is not None:
                        df = df.select(selected_fields)

                df = (
                    df
                    .with_row_index("__bb_row_idx", offset=row_offset)
                    .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
                    .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias(rand_col))
                    .with_columns((pl.col(rand_col) % bucket_count).cast(pl.UInt32).alias("__bb_bucket"))
                )

                for key, bucket_df in df.partition_by(
                    "__bb_bucket",
                    maintain_order=False,
                    include_key=True,
                    as_dict=True,
                ).items():
                    bucket_id = int(key[0] if isinstance(key, tuple) else key)
                    chunk_path = (
                        bucket_root
                        / f"b{bucket_id:03d}"
                        / f"c{file_counts[bucket_id]:06d}.parquet"
                    )
                    (
                        bucket_df
                        .drop(["__bb_bucket", "__bb_file_idx", "__bb_row_idx"])
                        .write_parquet(chunk_path, compression=compression)
                    )
                    file_counts[bucket_id] += 1

                row_offset += batch.num_rows
            print(f"    -> shuffle bucketize {file_idx + 1}/{len(paths)}")

        rng = random.Random(seed)
        bucket_order = list(range(bucket_count))
        rng.shuffle(bucket_order)

        shuffled_chunks: list[Path] = []
        for bucket_id in bucket_order:
            bucket_files = sorted((bucket_root / f"b{bucket_id:03d}").glob("*.parquet"))
            rng.shuffle(bucket_files)
            shuffled_chunks.extend(bucket_files)

        if out_path.exists():
            out_path.unlink()
        if shuffled_chunks:
            _append_parquet_batches(
                shuffled_chunks,
                out_path,
                batch_size=batch_size,
                compression=compression,
                rand_col=rand_col,
                shuffle_rows=True,
                seed=seed,
            )
        else:
            pq.write_table(_empty_table(paths, selected_fields), out_path, compression=compression)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_path


def shuffle_parquet_files(*args, **kwargs) -> Path:
    return write_shuffled_parquet(*args, **kwargs)
