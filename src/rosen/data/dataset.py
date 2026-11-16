from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from rosen.emb import SchemaManager
from .padding import ColumnFormatter, FeatureContext, TensorFormatter


ScannerSource = Union[str, Path, Sequence[Union[str, Path]], pl.LazyFrame]


def _resolve_parquet_paths(source: Union[str, Path, Sequence[Union[str, Path]]]) -> list[str]:
    if isinstance(source, (list, tuple)):
        paths = [str(Path(path)) for path in source]
    else:
        path_obj = Path(source)
        if path_obj.is_dir():
            paths = [str(path) for path in sorted(path_obj.glob("*.parquet"))]
        else:
            paths = [str(path_obj)]

    if not paths:
        raise FileNotFoundError(f"No parquet files found from input: {source}")
    return paths


def _parquet_top_level_columns(path: Union[str, Path]) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _fill_null_scalar(col_type: pa.DataType):
    if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
        return pa.scalar("", type=col_type)
    if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
        return pa.scalar([], type=col_type)
    if pa.types.is_boolean(col_type):
        return pa.scalar(False, type=col_type)
    return pa.scalar(0, type=col_type)


def arrow_table_to_batch_dict(
    table: pa.Table,
    selected_col_names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    selected_set = set(selected_col_names) if selected_col_names is not None else None
    batch_dict: Dict[str, np.ndarray] = {}
    for col_name, col_data in zip(table.column_names, table.columns):
        if selected_set is not None and col_name not in selected_set:
            continue

        clean_col = pc.fill_null(col_data, _fill_null_scalar(col_data.type))
        batch_dict[col_name] = clean_col.to_numpy(zero_copy_only=False)
    return batch_dict


def _concat_tables(tables: Sequence[pa.Table]) -> Optional[pa.Table]:
    non_empty_tables = [table for table in tables if table is not None and table.num_rows > 0]
    if not non_empty_tables:
        return None
    if len(non_empty_tables) == 1:
        return non_empty_tables[0]
    return pa.concat_tables(non_empty_tables)


class DataScanner:
    def __init__(
        self,
        source: ScannerSource,
        read_cols: Optional[Iterable[str]] = None,
        filter_expr: Optional[pl.Expr] = None,
    ):
        self.source = source
        self.read_cols = list(dict.fromkeys(read_cols or [])) or None
        self.filter_expr = filter_expr
        self.parquet_paths = None if isinstance(source, pl.LazyFrame) else _resolve_parquet_paths(source)
        self.lazy_frame = self._build_lazy_frame(source)

    def _build_lazy_frame(self, source: ScannerSource) -> pl.LazyFrame:
        if isinstance(source, pl.LazyFrame):
            lf = source
        else:
            lf = pl.scan_parquet(self.parquet_paths)

        if self.filter_expr is not None:
            lf = lf.filter(self.filter_expr)
        if self.read_cols is not None:
            lf = lf.select(self.read_cols)
        return lf

    def _apply_worker_shard(self, lf: pl.LazyFrame, worker_id: int, num_workers: int) -> pl.LazyFrame:
        if num_workers <= 1:
            return lf
        worker_col = "__scanner_worker_row_id"
        return (
            lf.with_row_index(worker_col)
            .filter((pl.col(worker_col) % num_workers) == worker_id)
            .drop(worker_col)
        )

    def _iter_lazy_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
        lf = self._apply_worker_shard(self.lazy_frame, worker_id=worker_id, num_workers=num_workers)
        for batch_df in lf.collect_batches(
            chunk_size=batch_size,
            maintain_order=True,
            engine="streaming",
        ):
            if batch_df.height == 0:
                continue
            yield batch_df.to_arrow()

    def _iter_parquet_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
        row_group_offset = 0
        for parquet_file in self.parquet_paths or []:
            parquet_file_reader = pq.ParquetFile(parquet_file)
            row_group_count = parquet_file_reader.metadata.num_row_groups
            if num_workers > 1:
                row_groups = [
                    row_group_idx
                    for row_group_idx in range(row_group_count)
                    if (row_group_offset + row_group_idx) % num_workers == worker_id
                ]
                if not row_groups:
                    row_group_offset += row_group_count
                    continue
            else:
                row_groups = None

            for record_batch in parquet_file_reader.iter_batches(
                batch_size=batch_size,
                row_groups=row_groups,
                columns=self.read_cols,
            ):
                yield pa.Table.from_batches([record_batch])
            row_group_offset += row_group_count

    def iter_batches(
        self,
        batch_size: int,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> Iterator[pa.Table]:
        if self.parquet_paths is not None and self.filter_expr is None:
            yield from self._iter_parquet_batches(batch_size, worker_id, num_workers)
            return

        yield from self._iter_lazy_batches(batch_size, worker_id, num_workers)


class DataTransformer:
    def __init__(
        self,
        manager: SchemaManager,
        preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
        filter_expr: Optional[pl.Expr] = None,
        output_col_names: Optional[Iterable[str]] = None,
    ):
        self.manager = manager
        self.preprocess_exprs = list(preprocess_exprs or [])
        self.filter_expr = filter_expr
        self.output_col_names = list(dict.fromkeys(output_col_names or manager.fields()))

    def __call__(self, raw_table: pa.Table) -> Optional[pa.Table]:
        if raw_table.num_rows == 0:
            return None

        raw_df = pl.from_arrow(raw_table)
        if self.preprocess_exprs:
            raw_df = raw_df.with_columns(self.preprocess_exprs)
        if self.filter_expr is not None:
            raw_df = raw_df.filter(self.filter_expr)
            if raw_df.height == 0:
                return None

        transformed_df = (
            self.manager.transform(raw_df.lazy())
            .select(self.output_col_names)
            .collect(engine="streaming")
        )
        if transformed_df.height == 0:
            return None
        return transformed_df.to_arrow()


class ShuffleBuffer:
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = True,
        selected_col_names: Optional[Sequence[str]] = None,
    ):
        self.capacity = max(batch_size, int(capacity))
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.selected_col_names = list(selected_col_names) if selected_col_names is not None else None
        self.reset()

    def reset(self):
        self.buffer: list[pa.Table] = []
        self.current_rows = 0

    def push(self, table: Optional[pa.Table]):
        if table is None or table.num_rows == 0:
            return
        self.buffer.append(table)
        self.current_rows += table.num_rows

    def _drain(self, force: bool) -> Iterator[Dict[str, np.ndarray]]:
        table = _concat_tables(self.buffer)
        if table is None or table.num_rows == 0:
            self.reset()
            return

        total_rows = table.num_rows
        batch_dict = arrow_table_to_batch_dict(table, selected_col_names=self.selected_col_names)
        indices = np.random.permutation(total_rows) if self.shuffle else np.arange(total_rows)
        num_complete_batches = total_rows // self.batch_size

        for batch_idx in range(num_complete_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]
            yield {col: arr[batch_indices] for col, arr in batch_dict.items()}

        residual_start = num_complete_batches * self.batch_size
        residual_indices = indices[residual_start:]
        residual_count = len(residual_indices)

        self.reset()
        if residual_count == 0:
            return

        if force:
            if not self.drop_last:
                yield {col: arr[residual_indices] for col, arr in batch_dict.items()}
            return

        residual_table = table.take(pa.array(residual_indices, type=pa.int64()))
        self.buffer = [residual_table]
        self.current_rows = residual_table.num_rows

    def yield_batches(self) -> Iterator[Dict[str, np.ndarray]]:
        if self.current_rows < self.capacity:
            return
        yield from self._drain(force=False)

    def flush(self) -> Iterator[Dict[str, np.ndarray]]:
        yield from self._drain(force=True)


class PipelineStreamDataset(IterableDataset):
    def __init__(
        self,
        scanner: DataScanner,
        formatter: TensorFormatter,
        transformer: Optional[DataTransformer] = None,
        buffer: Optional[ShuffleBuffer] = None,
        scan_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.scanner = scanner
        self.transformer = transformer
        self.buffer = buffer
        self.formatter = formatter
        self.scan_batch_size = scan_batch_size or (buffer.batch_size if buffer is not None else 4096)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if self.buffer is not None:
            self.buffer.reset()

        for raw_table in self.scanner.iter_batches(
            batch_size=self.scan_batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            table = self.transformer(raw_table) if self.transformer else raw_table
            if table is None or table.num_rows == 0:
                continue

            if self.buffer is None:
                yield self.formatter.format(
                    arrow_table_to_batch_dict(table, selected_col_names=self.formatter.context.output_col_names)
                )
                continue

            self.buffer.push(table)
            for batch_dict in self.buffer.yield_batches():
                yield self.formatter.format(batch_dict)

        if self.buffer is not None:
            for batch_dict in self.buffer.flush():
                yield self.formatter.format(batch_dict)


class ParquetStreamDataset(PipelineStreamDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ):
        context = FeatureContext.from_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        scanner = DataScanner(parquet_path, read_cols=context.read_col_names)
        formatter = TensorFormatter(context)
        buffer = ShuffleBuffer(
            capacity=shuffle_buffer_size if shuffle else batch_size,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=context.output_col_names,
        )
        super().__init__(
            scanner=scanner,
            formatter=formatter,
            transformer=None,
            buffer=buffer,
            scan_batch_size=batch_size * 4 if shuffle else batch_size,
        )

        self.parquet_path = parquet_path
        self.parquet_paths = list(scanner.parquet_paths or [])
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trigger_size = buffer.capacity
        self.context = context
        self.output_col_names = list(context.output_col_names)
        self.valid_col_names = list(context.output_col_names)
        self.read_col_names = list(context.read_col_names)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.col_formatters = formatter.col_formatters


def _empty_object_sequences(batch_size: int) -> np.ndarray:
    values = np.empty(batch_size, dtype=object)
    for idx in range(batch_size):
        values[idx] = []
    return values


def _default_missing_column(name: str, formatter: ColumnFormatter, batch_size: int) -> np.ndarray:
    formatter_name = formatter.__class__.__name__
    if formatter_name.startswith("Padded"):
        return _empty_object_sequences(batch_size)
    if formatter_name in {"DenseFormatter"}:
        return np.zeros(batch_size, dtype=np.float32)
    if formatter_name in {"VectorDenseFormatter"}:
        dim = int(getattr(formatter, "dim", 0))
        return np.zeros((batch_size, dim), dtype=np.float32)
    return np.zeros(batch_size, dtype=np.int64)


class ProfileSequenceStore:
    HISTORY_TIME_FILTERS = {
        "cursor_order",
        "strict_time",
        "drop_same_ts",
        "same_ts_only",
    }

    def __init__(
        self,
        manager: SchemaManager,
        profile_root: Union[str, Path],
        group_dirs: Optional[Mapping[str, Union[str, Path]]] = None,
    ):
        self.manager = manager
        self.profile_root = Path(profile_root)
        self.group_dirs = {str(k): Path(v) for k, v in dict(group_dirs or {}).items()}
        self.groups = self._compile_groups()
        self.tables = {
            group_name: self._load_group(group_name, spec)
            for group_name, spec in self.groups.items()
        }

    def _compile_groups(self) -> Dict[str, dict]:
        groups: Dict[str, dict] = {}
        for setting in self.manager.settings:
            if not setting.is_sequence_setting:
                continue
            group_name = setting.group_name
            group = setting.group
            spec = groups.setdefault(
                group_name,
                {
                    "owner_col": getattr(group, "owner_field_name", None) or self.manager.uid_field,
                    "seq_len_col": setting.seq_len_field_name,
                    "max_len": int(setting.max_len),
                    "fields": [],
                    "time_field": setting.time_field_name,
                    "anchor_col": getattr(group, "anchor_field_name", None),
                    "history_time_filter": getattr(
                        group,
                        "history_time_filter",
                        "cursor_order",
                    ),
                    "truncate_mode": setting.truncate_mode,
                },
            )
            shared_values = {
                "owner_col": getattr(group, "owner_field_name", None) or self.manager.uid_field,
                "seq_len_col": setting.seq_len_field_name,
                "max_len": int(setting.max_len),
                "time_field": setting.time_field_name,
                "anchor_col": getattr(group, "anchor_field_name", None),
                "history_time_filter": getattr(group, "history_time_filter", "cursor_order"),
                "truncate_mode": setting.truncate_mode,
            }
            for key, value in shared_values.items():
                if spec[key] != value:
                    raise ValueError(
                        f"Profile sequence group '{group_name}' has inconsistent "
                        f"{key}: {spec[key]!r} != {value!r}."
                    )
            if setting.field_name not in spec["fields"]:
                spec["fields"].append(setting.field_name)
            if setting.time_field_name and setting.time_field_name not in spec["fields"]:
                spec["fields"].append(setting.time_field_name)
            if spec["history_time_filter"] not in self.HISTORY_TIME_FILTERS:
                raise ValueError(
                    f"Unknown history_time_filter {spec['history_time_filter']!r} "
                    f"for profile sequence group '{group_name}'."
                )
        return groups

    def _resolve_group_dir(self, group_name: str) -> Path:
        if group_name in self.group_dirs:
            return self.group_dirs[group_name]
        return self.profile_root / group_name

    @staticmethod
    def _profile_value_to_array(value) -> np.ndarray:
        if value is None:
            return np.asarray([], dtype=np.int64)
        try:
            return np.asarray(value, dtype=np.int64)
        except (TypeError, ValueError):
            return np.asarray(value, dtype=object)

    def _load_group(self, group_name: str, spec: dict) -> Dict[int, Dict[str, np.ndarray]]:
        group_dir = self._resolve_group_dir(group_name)
        if group_dir.is_dir():
            files = sorted(group_dir.glob("*.parquet"))
        else:
            files = [group_dir] if group_dir.exists() else []
        if not files:
            raise FileNotFoundError(f"No profile parquet files found for group '{group_name}' under {group_dir}")

        read_cols = [spec["owner_col"], *spec["fields"]]
        df = pl.read_parquet([str(path) for path in files], columns=read_cols)
        table: Dict[int, Dict[str, np.ndarray]] = {}
        for row in df.iter_rows(named=True):
            uid = int(row[spec["owner_col"]])
            table[uid] = {}
            for field in spec["fields"]:
                value = row[field]
                table[uid][field] = self._profile_value_to_array(value)
        return table

    def inject(self, batch_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.groups:
            return batch_dict
        batch_size = len(next(iter(batch_dict.values())))

        for group_name, spec in self.groups.items():
            owner_col = spec["owner_col"]
            seq_len_col = spec["seq_len_col"]
            if owner_col not in batch_dict or seq_len_col not in batch_dict:
                raise KeyError(
                    f"Profile group '{group_name}' requires columns "
                    f"'{owner_col}' and '{seq_len_col}' in main batch."
                )

            users = np.asarray(batch_dict[owner_col]).reshape(-1)
            cursors = np.asarray(batch_dict[seq_len_col]).reshape(-1)
            table = self.tables[group_name]
            max_len = int(spec["max_len"])
            history_time_filter = spec["history_time_filter"]
            time_field = spec["time_field"]
            anchor_col = spec["anchor_col"]
            anchors = None
            if history_time_filter != "cursor_order":
                if not time_field or not anchor_col:
                    raise ValueError(
                        f"Profile group '{group_name}' requires time_field_name and "
                        f"anchor_field_name for history_time_filter={history_time_filter!r}."
                    )
                if anchor_col not in batch_dict:
                    raise KeyError(
                        f"Profile group '{group_name}' requires anchor column "
                        f"'{anchor_col}' in main batch."
                    )
                anchors = np.asarray(batch_dict[anchor_col]).reshape(-1)

            sliced_fields = {
                field: np.empty(batch_size, dtype=object)
                for field in spec["fields"]
            }
            actual_lens = np.zeros(batch_size, dtype=np.int64)

            for row_idx, (uid, cursor) in enumerate(zip(users, cursors)):
                profile = table.get(int(uid))
                cursor_int = max(0, int(cursor))
                base_field = spec["fields"][0]
                base_seq = profile.get(base_field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
                end = min(cursor_int, len(base_seq))
                selected_indices = np.arange(end, dtype=np.int64)

                if history_time_filter != "cursor_order":
                    history_time = profile.get(time_field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
                    if len(history_time) < end:
                        raise ValueError(
                            f"Profile group '{group_name}' has {len(history_time)} "
                            f"timestamps for cursor {end} (user={int(uid)})."
                        )
                    candidate_time = history_time[:end]
                    anchor = anchors[row_idx]
                    if history_time_filter in {"strict_time", "drop_same_ts"}:
                        selected_indices = selected_indices[candidate_time < anchor]
                    else:
                        selected_indices = selected_indices[candidate_time == anchor]

                if spec["truncate_mode"] == "head":
                    selected_indices = selected_indices[:max_len]
                elif spec["truncate_mode"] == "tail":
                    selected_indices = selected_indices[-max_len:]
                else:
                    raise ValueError(
                        f"Unknown truncate_mode {spec['truncate_mode']!r} "
                        f"for profile sequence group '{group_name}'."
                    )
                actual_lens[row_idx] = len(selected_indices)

                for field in spec["fields"]:
                    full_seq = profile.get(field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
                    if len(full_seq) < end:
                        raise ValueError(
                            f"Profile group '{group_name}' field '{field}' has "
                            f"length {len(full_seq)} for cursor {end} (user={int(uid)})."
                        )
                    sliced_fields[field][row_idx] = full_seq[selected_indices]

            batch_dict.update(sliced_fields)
            batch_dict[seq_len_col] = actual_lens

        return batch_dict


class ProfileParquetStreamDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        profile_root: Union[str, Path],
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
        group_dirs: Optional[Mapping[str, Union[str, Path]]] = None,
        profile_store: Optional[ProfileSequenceStore] = None,
    ):
        super().__init__()
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.context = FeatureContext.from_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        self.formatter = TensorFormatter(self.context)
        self.profile_store = (
            profile_store
            if profile_store is not None
            else ProfileSequenceStore(
                manager=manager,
                profile_root=profile_root,
                group_dirs=group_dirs,
            )
        )

        parquet_paths = _resolve_parquet_paths(parquet_path)
        available_cols = _parquet_top_level_columns(parquet_paths[0])
        read_cols = [col for col in self.context.output_col_names if col in available_cols]
        for spec in self.profile_store.groups.values():
            required_cols = [spec["owner_col"], spec["seq_len_col"]]
            if spec["history_time_filter"] != "cursor_order":
                required_cols.append(spec["anchor_col"])
            for required_col in required_cols:
                if required_col in available_cols and required_col not in read_cols:
                    read_cols.append(required_col)

        self.scanner = DataScanner(parquet_path, read_cols=read_cols)
        self.buffer = ShuffleBuffer(
            capacity=shuffle_buffer_size if shuffle else batch_size,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=None,
        )
        self.scan_batch_size = batch_size * 4 if shuffle else batch_size
        self.parquet_path = parquet_path
        self.parquet_paths = parquet_paths
        self.profile_root = Path(profile_root)
        self.output_col_names = list(self.context.output_col_names)
        self.valid_col_names = list(self.context.output_col_names)
        self.read_col_names = list(read_cols)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.trigger_size = self.buffer.capacity
        self.col_formatters = self.formatter.col_formatters

    def _complete_batch(self, batch_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not batch_dict:
            return batch_dict
        batch_size = len(next(iter(batch_dict.values())))
        self.profile_store.inject(batch_dict)
        for col in self.context.output_col_names:
            if col in batch_dict:
                continue
            formatter = self.formatter.col_formatters.get(col, self.formatter.fallback_formatter)
            batch_dict[col] = _default_missing_column(col, formatter, batch_size)
        return {col: batch_dict[col] for col in self.context.output_col_names if col in batch_dict}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        self.buffer.reset()

        for raw_table in self.scanner.iter_batches(
            batch_size=self.scan_batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            self.buffer.push(raw_table)
            for batch_dict in self.buffer.yield_batches():
                yield self.formatter.format(self._complete_batch(batch_dict))

        for batch_dict in self.buffer.flush():
            yield self.formatter.format(self._complete_batch(batch_dict))


class RawParquetStreamDataset(PipelineStreamDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
        transform_buffer_size: Optional[int] = None,
        raw_filter_expr: Optional[pl.Expr] = None,
        raw_preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
    ):
        preprocess_exprs = list(raw_preprocess_exprs or [])
        scanner_filter_expr = raw_filter_expr if not preprocess_exprs else None
        transformer_filter_expr = None if scanner_filter_expr is not None else raw_filter_expr
        context = FeatureContext.from_raw_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        scanner = DataScanner(
            parquet_path,
            read_cols=context.read_col_names,
            filter_expr=scanner_filter_expr,
        )
        formatter = TensorFormatter(context)
        transformer = DataTransformer(
            manager=manager,
            preprocess_exprs=preprocess_exprs,
            filter_expr=transformer_filter_expr,
            output_col_names=context.output_col_names,
        )
        capacity = transform_buffer_size
        if capacity is None:
            capped_rows = batch_size * (8 if shuffle else 1)
            capacity = max(batch_size, min(shuffle_buffer_size, capped_rows))
        buffer = ShuffleBuffer(
            capacity=capacity,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=context.output_col_names,
        )
        super().__init__(
            scanner=scanner,
            formatter=formatter,
            transformer=transformer,
            buffer=buffer,
            scan_batch_size=batch_size * 2 if shuffle else batch_size,
        )

        self.parquet_path = parquet_path
        self.parquet_paths = list(scanner.parquet_paths or [])
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trigger_size = buffer.capacity
        self.context = context
        self.output_col_names = list(context.output_col_names)
        self.valid_col_names = list(context.output_col_names)
        self.read_col_names = list(context.read_col_names)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.raw_filter_expr = raw_filter_expr
        self.raw_preprocess_exprs = preprocess_exprs
        self.transform_buffer_size = buffer.capacity
        self.col_formatters = formatter.col_formatters
