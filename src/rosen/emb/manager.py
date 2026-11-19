# ==========================================
# 2. 调度层 (Manager Layer) - 计算图统筹
# ==========================================
import gc
import json
import shutil
import ctypes
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union
import polars as pl

from rosen.emb.split import SplitConfig, SplitContext, create_split_strategy
from rosen.emb.schema import AuxField, EmbSetting, EmbType, MultiSparseSetting, SlateLabel, SparseEmbSetting
from rosen.core.enum_type import FeatureSource
from rosen.core.formatters import ColumnFormatter

from rosen.utils.sort import sort_parquet_inplace
from rosen.utils.shuffle import write_shuffled_parquet


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def collect_lazy(lf: pl.LazyFrame, streaming: bool = False) -> pl.DataFrame:
    if not streaming:
        return lf.collect()
    try:
        return lf.collect(streaming=True)
    except TypeError:
        return lf.collect(engine="streaming")
    except ValueError:
        return lf.collect()


def trim_process_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


VOCAB_DIR_NAME = "vocab"
VOCAB_VALUE_COL = "value"
VOCAB_ID_COL = "id"


class SchemaManager:
    ITEM_PROFILE_NAME = "item_profile.parquet"
    USER_PROFILE_NAME = "user_profile.parquet"
    SCHEMA_META_NAME = "feature_meta.json"
    WHOLE_DATA_NAME = "whole_dataframe.parquet"
    def __init__(self, settings_list: List[EmbSetting], schema_dir: str,
                 time_field=None,
                 label_fields: Union[str, SlateLabel, Iterable[Union[str, SlateLabel]]]=None,
                 domain_fields: Union[str, Iterable[str]]=None,
                 aux_fields: Union[AuxField, Iterable[AuxField]]=None
                 ):
        self.settings = settings_list

        ## path & dir
        self.schema_dir = Path(schema_dir)
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self.meta_filepath = self.schema_dir / self.SCHEMA_META_NAME

        ## field_name
        self.uid_field = next(
            (s.field_name for s in self.settings if s.source == FeatureSource.USER_ID),
            None
        )
        self.iid_field =next(
            (s.field_name for s in self.settings if s.source == FeatureSource.ITEM_ID),
            None
        )
        self.time_field = time_field
        self._label_specs: dict[str, Optional[SlateLabel]] = {}
        for label in ensure_list(label_fields):
            if isinstance(label, SlateLabel):
                name, spec = label.field_name, label
            elif isinstance(label, str) and label.strip():
                name, spec = label, None
            else:
                raise ValueError("label_fields must contain non-empty strings or SlateLabel instances.")
            if name in self._label_specs and self._label_specs[name] != spec:
                raise ValueError(f"Conflicting label declarations for '{name}'.")
            self._label_specs[name] = spec
        self._label_fields = list(self._label_specs)
        mask_layouts = {}
        for spec in self._label_specs.values():
            if spec is None or spec.mask_field is None:
                continue
            layout = (spec.max_len, spec.padding_side)
            if spec.mask_field in mask_layouts and mask_layouts[spec.mask_field] != layout:
                raise ValueError(f"Shared mask '{spec.mask_field}' requires matching max_len and padding_side.")
            mask_layouts[spec.mask_field] = layout
        self._domain_fields = ensure_list(domain_fields)
        self._aux_specs: dict[str, AuxField] = {}
        occupied = {name for setting in self.settings for name in setting.get_output_field_names()}
        occupied.update(name for name in (self.time_field, *self.label_fields, *self.mask_fields, *self.domain_fields) if name)
        for aux in ensure_list(aux_fields):
            if not isinstance(aux, AuxField):
                raise TypeError("aux_fields must contain AuxField instances")
            if aux.field_name in occupied:
                raise ValueError(f"AuxField '{aux.field_name}' conflicts with an existing schema field")
            if aux.field_name in self._aux_specs and self._aux_specs[aux.field_name] != aux:
                raise ValueError(f"Conflicting AuxField declarations for '{aux.field_name}'")
            self._aux_specs[aux.field_name] = aux

    @property
    def aux_fields(self) -> tuple[str, ...]:
        return tuple(self._aux_specs)

    def get_aux_spec(self, field_name: str) -> Optional[AuxField]:
        return self._aux_specs.get(field_name)

    @property
    def label_field(self) -> Optional[str]:
        return next(iter(self._label_fields), None)
    @property
    def label_fields(self) -> tuple[str, ...]:
        return tuple(self._label_fields)
    def get_label_spec(self, field_name: str) -> Optional[SlateLabel]:
        """Return list supervision metadata, or None for scalar/unknown fields."""
        return self._label_specs.get(field_name)
    @property
    def mask_fields(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            spec.mask_field for spec in self._label_specs.values()
            if spec is not None and spec.mask_field is not None
        ))
    @property
    def domain_field(self):
        return next(iter(self._domain_fields), None)
    @property
    def domain_fields(self) -> tuple:
        return tuple(self._domain_fields)

    def _sync_sequence_fit_state(self):
        for setting in self.settings:
            if hasattr(setting, "element_setting"):
                setting.is_fitted = setting.element_setting.is_fitted

    def _unfitted_fit_settings(self):
        """Fit shared targets once, using their first registered source only."""
        self._sync_sequence_fit_state()
        seen = set()
        settings = []
        for setting in self.settings:
            target = getattr(setting, "element_setting", setting)
            if target.is_fitted or id(target) in seen:
                continue
            seen.add(id(target))
            settings.append(setting)
        return settings


    def _parse_output_dir(self, output_dir=None):
        if output_dir is None:
            output_dir = self.schema_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        return output_dir


    def prepare_data(self, lazy_df: pl.LazyFrame, output_dir: Union[Path, str]=None, redo=False):
        """
        不建议使用，全自动化流程：自动判断输入文件类型，自动构建计算图
        """
        output_parquet = self._parse_output_dir(output_dir) / self.WHOLE_DATA_NAME
        if self.meta_filepath.exists() and output_parquet.exists() and not redo:
            print("[Info] 检测到已存在元数据和目标文件，跳过处理直接加载 Schema。")
            self.load_schema()
            return pl.scan_parquet(output_parquet)

        print("[*] 正在执行计算图截断与中间态落盘，请稍候...")
        lazy_df = self._make_checkpoint(lazy_df)

        print("[*] 阶段一：构建 Fit 计算图...")
        unfitted_settings = self._unfitted_fit_settings()
        fit_exprs = [expr for setting in unfitted_settings for expr in setting.get_fit_exprs()]

        if fit_exprs:
            print(f"[*] 启动单次全表扫描获取统计量，包含 {len(fit_exprs)} 个计算节点...")
            fit_result = lazy_df.select(fit_exprs).collect()
            for setting in unfitted_settings:
                setting.parse_fit_result(fit_result)
            print("[+] 统计量计算与映射表构建完成。")
        else:
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
        print("[*] 阶段二：构建 Transform 计算图...")
        transform_exprs = self._build_transform_exprs()
        print(f"[*] 启动流式转换计算，引擎正在将数据写入 {output_parquet} ...")
        lazy_df.with_columns(transform_exprs) \
            .sink_parquet(output_parquet)
        self.save_schema()
        print("[+] 预处理全流程结束！")
        return pl.scan_parquet(output_parquet)


    def fit(self, train_raw_lf: pl.LazyFrame, low_memory: bool = False):
        """
        阶段一：仅使用训练集拟合统计量和词表。

        :param train_raw_lf: Polars LazyFrame 格式的训练数据
        :param low_memory: 是否开启低内存模式。
                           - False (默认，推荐): 单次全表扫描，所有特征并行计算。速度极快，但内存开销较大。
                           - True (内存受限时): 逐个特征扫描。内存占用极低，但 I/O 放大，耗时成倍增加。
        """
        if self.meta_filepath.exists():
            self.load_schema()
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
            return
        unfitted_settings = self._unfitted_fit_settings()
        if not unfitted_settings:
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
            self.save_schema()
            return

        if low_memory:
            print(f"[*] 启动极低内存 Fit 模式，将对 {len(unfitted_settings)} 个特征进行逐列扫描 (耗时较长)...")
            for i, setting in enumerate(unfitted_settings):
                fit_exprs = setting.get_fit_exprs()
                if not fit_exprs:
                    continue
                print(f"    -> 正在拟合 [{i + 1}/{len(unfitted_settings)}]: {setting.field_name}")
                col_fit_result = collect_lazy(train_raw_lf.select(fit_exprs), streaming=True)
                setting.parse_fit_result(col_fit_result)
                del col_fit_result # 显式释放哈希表内存
                trim_process_memory()
            print("[+] 逐列统计量与词表构建完成。")
        else:
            print(f"[*] 启动高性能并行 Fit 模式，单次扫描全表计算 {len(unfitted_settings)} 个统计量...")
            fit_exprs = [expr for s in unfitted_settings for expr in s.get_fit_exprs()]
            if fit_exprs:
                # 即使是并行，也建议开启 streaming=True，让 Polars 底层自己尽力优化和 Spilling
                fit_result = collect_lazy(train_raw_lf.select(fit_exprs), streaming=True)
                for setting in unfitted_settings:
                    setting.parse_fit_result(fit_result)
        self.save_schema()

    def _checkpoint_path(self, file_name: Union[str, Path]) -> Path:
        path = Path(file_name)
        if not path.is_absolute():
            path = self.schema_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def make_checkpoint(self, lf, file_name="custom_checkpoint.parquet", redo=True):
        path = self._checkpoint_path(file_name)
        if not path.exists() or redo:
            self._make_checkpoint(lf, file_name)
            return pl.scan_parquet(path)
        else:
            return pl.scan_parquet(path)

    def make_sorted_checkpoint(
        self,
        lf,
        file_name="custom_sorted_checkpoint.parquet",
        sort_by=None,
        descending=False,
        redo=True,
        memory_limit: str = "84GB",
    ):
        if sort_by is None:
            raise ValueError("sort_by is required for make_sorted_checkpoint.")
        path = self._checkpoint_path(file_name)
        if not path.exists() or redo:
            self._make_checkpoint(lf, file_name)
            sort_parquet_inplace(
                path,
                sort_by,
                descending=descending,
                memory_limit=memory_limit,
                temp_dir=path.parent / f"_{path.stem}_sort_tmp",
            )
        return pl.scan_parquet(path)

    def make_shuffled_checkpoint(
        self,
        lf,
        file_name="custom_shuffled_checkpoint.parquet",
        seed: int = 2026,
        bucket_count: int = 128,
        batch_size: int = 65536,
        redo: bool = True,
        exact: bool = False,
        mode: Literal["target_sort", "bucket_chunk"] = "target_sort",
        compression: str = "zstd",
    ):
        path = self._checkpoint_path(file_name)
        if path.exists() and not redo:
            return pl.scan_parquet(path)

        source_path = path.parent / f"_{path.stem}_unshuffled.parquet"
        if source_path.exists():
            source_path.unlink()
        try:
            lf.sink_parquet(source_path, compression=compression)
            write_shuffled_parquet(
                [source_path],
                path,
                seed=seed,
                bucket_count=bucket_count,
                batch_size=batch_size,
                compression=compression,
                redo=True,
                exact=exact,
                mode=mode,
                temp_dir=path.parent / f"_{path.stem}_shuffle_tmp",
            )
        finally:
            if source_path.exists():
                source_path.unlink()
        return pl.scan_parquet(path)

    def transform(self, raw_lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        阶段二：根据已固化的 Schema 进行流式转换
        """
        transform_exprs = self._build_transform_exprs()
        return raw_lf.with_columns(transform_exprs)

    def transform_csv_partitioned(
        self,
        input_paths: Union[str, Path, Iterable[Union[str, Path]]],
        output_dir: Union[str, Path],
        uid_field: Optional[str] = None,
        n_parts: int = 100,
        batch_size: int = 50000,
        batches_per_read: int = 1,
        buffer_flush_rows: int = 500000,
        read_csv_kwargs: Optional[Mapping[str, Any]] = None,
        redo: bool = True,
    ) -> List[Path]:
        """
        分批读取 CSV，边 transform 边按原始 uid hash 分区落盘。

        输出结构:
            output_dir/p000/c0000.parquet
            output_dir/p000/c0001.parquet
            output_dir/p001/c0000.parquet
            ...
        """
        uid_field = uid_field or self.uid_field
        if uid_field is None:
            raise ValueError("uid_field is required when manager.uid_field is not set.")
        if n_parts <= 0:
            raise ValueError("n_parts must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batches_per_read <= 0:
            raise ValueError("batches_per_read must be positive.")
        if buffer_flush_rows <= 0:
            raise ValueError("buffer_flush_rows must be positive.")

        output_dir = Path(output_dir)
        if output_dir.exists() and redo:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        part_dirs = [output_dir / f"p{pid:03d}" for pid in range(n_parts)]
        for part_dir in part_dirs:
            part_dir.mkdir(parents=True, exist_ok=True)

        existing = [path for part_dir in part_dirs for path in part_dir.glob("*.parquet")]
        if existing and not redo:
            return part_dirs

        transform_exprs = self._build_transform_exprs()
        part_col = "__bb_part"
        buffers = {pid: [] for pid in range(n_parts)}
        buffer_rows = {pid: 0 for pid in range(n_parts)}
        file_counts = {pid: 0 for pid in range(n_parts)}

        def flush(pid: int):
            if not buffers[pid]:
                return
            out = pl.concat(buffers[pid], how="vertical")
            path = part_dirs[pid] / f"c{file_counts[pid]:04d}.parquet"
            out.write_parquet(path)
            file_counts[pid] += 1
            buffers[pid] = []
            buffer_rows[pid] = 0

        csv_kwargs = dict(read_csv_kwargs or {})
        csv_kwargs["batch_size"] = batch_size
        total_rows = 0

        for input_path in ensure_list(input_paths):
            input_path = Path(input_path)
            print(f"[*] transform csv partitioned: {input_path}")
            reader = pl.read_csv_batched(input_path, **csv_kwargs)

            while True:
                batches = reader.next_batches(batches_per_read)
                if batches is None:
                    break

                for raw_df in batches:
                    if raw_df.height == 0:
                        continue
                    if uid_field not in raw_df.columns:
                        raise ValueError(f"uid_field '{uid_field}' not found in {input_path}")
                    if part_col in raw_df.columns:
                        raise ValueError(f"reserved column '{part_col}' already exists in {input_path}")

                    df = (
                        raw_df.lazy()
                        .with_columns(
                            (pl.col(uid_field).hash(seed=0) % n_parts)
                            .cast(pl.UInt32)
                            .alias(part_col)
                        )
                        .with_columns(transform_exprs)
                        .pipe(collect_lazy, streaming=True)
                    )

                    for key, part_df in df.partition_by(
                        part_col,
                        maintain_order=False,
                        include_key=True,
                        as_dict=True,
                    ).items():
                        pid = int(key[0])
                        part_df = part_df.drop(part_col)
                        buffers[pid].append(part_df)
                        buffer_rows[pid] += part_df.height
                        if buffer_rows[pid] >= buffer_flush_rows:
                            flush(pid)

                    total_rows += df.height
                    del raw_df, df

            print(f"    -> done, cumulative rows={total_rows:,}")

        for pid in range(n_parts):
            flush(pid)

        print(f"[+] partitioned transform done: rows={total_rows:,}, output={output_dir}")
        return part_dirs

    def extract_sequence_partitions(
        self,
        flat_part_dirs: Iterable[Union[str, Path]],
        output_dir: Union[str, Path],
        max_seq_len: int,
        user_col: str,
        time_col: str,
        feature_mapping: Mapping[str, str],
        seq_len_col: str = "seq_len",
        label_col: Optional[str] = None,
        positive_label: Any = 1,
        select_fields: Optional[Iterable[str]] = None,
        redo: bool = True,
    ) -> List[Path]:
        from rosen.utils.sequential import extract_history_sequences

        output_dir = Path(output_dir)
        if output_dir.exists() and redo:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: List[Path] = []
        fields = list(select_fields) if select_fields is not None else None

        for idx, part in enumerate(flat_part_dirs):
            part = Path(part)
            out_path = output_dir / f"{part.name}.parquet"
            outputs.append(out_path)
            if out_path.exists() and not redo:
                continue

            if part.is_dir():
                files = sorted(part.glob("*.parquet"))
            else:
                files = [part] if part.exists() else []

            if not files:
                continue

            seq_lf = extract_history_sequences(
                pl.scan_parquet([str(path) for path in files]),
                max_seq_len=max_seq_len,
                user_col=user_col,
                time_col=time_col,
                feature_mapping=dict(feature_mapping),
                seq_len_col=seq_len_col,
                label_col=label_col,
                positive_label=positive_label,
            )
            if fields is not None:
                seq_lf = seq_lf.select(fields)
            seq_lf.sink_parquet(out_path)
            print(f"    -> seq part {idx + 1}: {out_path}")

        return outputs

    def _build_transform_exprs(self) -> List[pl.Expr]:
        transform_exprs: List[pl.Expr] = []
        seen_output_names = set()
        for setting in self.settings:
            if setting.emb_type == EmbType.UNKNOWN:
                continue
            expr = setting.get_transform_expr()
            expr_list = expr if isinstance(expr, list) else [expr]
            for item in expr_list:
                try:
                    output_name = item.meta.output_name()
                except Exception:
                    output_name = None
                if output_name is not None and output_name in seen_output_names:
                    continue
                if output_name is not None:
                    seen_output_names.add(output_name)
                transform_exprs.append(item)
        return transform_exprs

    def _make_checkpoint(self, lazy_df, file_name="temp_checkpoint.parquet"):
        temp_checkpoint_path = self._checkpoint_path(file_name)
        lazy_df.sink_parquet(temp_checkpoint_path)
        lazy_df = pl.scan_parquet(temp_checkpoint_path)
        return lazy_df

    # NOTE 不推荐使用
    def generate_profiles(self, lazy_df: pl.LazyFrame, output_dir: str=None, redo=False):
        """
        利用 FeatureSource 统筹提取并保存静态画像表 (Profile)
        """
        item_profile_path = self._parse_output_dir(output_dir) / self.ITEM_PROFILE_NAME
        user_profile_path = self._parse_output_dir(output_dir) / self.USER_PROFILE_NAME

        print("[*] 阶段三：开始提取静态特征查找表 (Profiles)...")
        # 1. Item Profile
        item_features = [setting.field_name for setting in self.settings
                         if setting.source == FeatureSource.ITEM]
        if self.iid_field and (not item_profile_path.exists() or redo):
            # 确保主键在提取列表中
            if self.iid_field not in item_features:
                item_features.insert(0, self.iid_field)

            print(f"[*] 正在提取 Item 画像，包含特征: {item_features}")
            (
                lazy_df
                .select(item_features)
                .drop_nulls(subset=[self.iid_field])  # 剔除由于 left join 产生的空物品
                .unique(subset=[self.iid_field])  # 按物品 ID 去重
                .sort(self.iid_field)  # 【极其关键】按 ID 从小到大排序！
                .sink_parquet(item_profile_path)
            )
        # 2. 提取 User Profile (同理)
        user_features = [
            setting.field_name
            for setting in self.settings
            if setting.source == FeatureSource.USER
        ]

        if self.uid_field and (not user_profile_path.exists() or redo):
            if self.uid_field not in user_features:
                user_features.insert(0, self.uid_field)

            print(f"[*] 正在提取 User 画像，包含特征: {user_features}")
            (
                lazy_df
                .select(user_features)
                .drop_nulls(subset=[self.uid_field])
                .unique(subset=[self.uid_field])
                .sort(self.uid_field)
                .sink_parquet(user_profile_path)
            )
        print("[+] 静态查找表提取完成！")

    def split_dataset(
        self,
        lf: pl.LazyFrame,
        strategy: Literal['loo', 'time', 'sequential_ratio', 'random_ratio'] = "random_ratio",
        output_dir: str = None,
        redo: bool = False,
        config: Optional[SplitConfig] = None,
        **kwargs,
    ):
        output_dir = self._parse_output_dir(output_dir)
        split_ctx = SplitContext(
            uid_field=self.uid_field,
            iid_field=self.iid_field,
            time_field=self.time_field,
            checkpoint_fn=self._make_checkpoint  # 把方法当做变量传递进去 (Callback)
        )
        splitter = create_split_strategy(strategy=strategy, context=split_ctx, config=config, **kwargs)
        return splitter.split(lf, output_dir, redo)

    def save_as_dataset(self, train_lf, valid_lf=None, test_lf=None, output_dir: str = None, redo: bool = False):
        """
        如果预设的split_dataset不和你的想法，那么把三个lf传过来，这边直接保存。利好Kuairand等预设test的数据集
        """
        output_dir = self._parse_output_dir(output_dir)
        train_path = output_dir / "train.parquet"
        valid_path = output_dir / "valid.parquet"
        test_path = output_dir / "test.parquet"
        split_paths = str(train_path), str(valid_path), str(test_path)
        if not redo and (train_path.exists() or valid_path.exists() or test_path.exists()):
            return split_paths
        gc.collect()
        for lf, save_path, name in zip([train_lf, valid_lf, test_lf], split_paths, ["Train", "Valid", "Test"]):
            if lf is not None:
                print(f"    -> 正在写入 {name} 集合到 {save_path}")
                lf.sink_parquet(str(save_path),)
        print("[+] 所有数据落盘完成！")
        return split_paths

    def save_schema(self):
        self._sync_sequence_fit_state()
        vocab_dir = self.meta_filepath.parent / VOCAB_DIR_NAME
        vocab_dir.mkdir(parents=True, exist_ok=True)
        meta_data = []
        for setting in self.settings:
            data = setting.to_dict()
            for state in (data, data.get("element_setting_state")):
                if not isinstance(state, dict) or not isinstance(state.get("vocab"), dict) or not state["vocab"]:
                    continue
                ref = f"{VOCAB_DIR_NAME}/{state['field_name']}.parquet"
                vocab = state["vocab"]
                rows = sorted(vocab.items(), key=lambda item: int(item[1]))
                pl.DataFrame(
                    {
                        VOCAB_VALUE_COL: [str(key) for key, _ in rows],
                        VOCAB_ID_COL: [int(value) for _, value in rows],
                    }
                ).write_parquet(self.meta_filepath.parent / ref)
                state["vocab"] = ref
            meta_data.append(data)
        with open(self.meta_filepath, 'w') as f:
            json.dump(meta_data, f, indent=4)

    def load_schema(self):
        if not self.meta_filepath.exists():
            print(f"[Warning] 找不到元数据文件 {self.meta_filepath}，无法加载 schema。")
            return

        with open(self.meta_filepath, 'r', encoding='utf-8') as f:
            meta_data_list = json.load(f)

        for data in meta_data_list:
            for state in (data, data.get("element_setting_state")):
                if not isinstance(state, dict) or not isinstance(state.get("vocab"), str):
                    continue
                vocab_df = pl.read_parquet(
                    self.meta_filepath.parent / state["vocab"],
                    columns=[VOCAB_VALUE_COL, VOCAB_ID_COL],
                )
                state["vocab"] = {
                    str(row[VOCAB_VALUE_COL]): int(row[VOCAB_ID_COL])
                    for row in vocab_df.iter_rows(named=True)
                }

        meta_dict = {item["field_name"]: item for item in meta_data_list}
        for setting in self.settings:
            if setting.field_name in meta_dict:
                saved_state = meta_dict[setting.field_name]
                saved_type = saved_state.get("type")
                if saved_type not in setting.compatible_type_names:
                    raise ValueError(
                        f"特征 {setting.field_name} 的类型不匹配: "
                        f"期望 {sorted(setting.compatible_type_names)}, JSON 中为 {saved_type}"
                    )

                setting.load_state(saved_state)
                print(f"[-] 成功恢复特征状态: {setting.field_name}")
            else:
                print(f"[Warning] 特征 {setting.field_name} 在 JSON 元数据中未找到，将保持默认/未拟合状态。")
        self._sync_sequence_fit_state()

    def source2emb_dim(self, *sources: FeatureSource):
        emb_dim = 0
        for setting in self.settings:
            if setting.source in sources:
                emb_dim += setting.embedding_dim
        return emb_dim

    def get_setting(self, field_name):
        for setting in self.settings:
            if setting.field_name == field_name:
                return setting
        return None

    def fields(self, *, include_labels: bool = True) -> list[str]:
        """Encoded output columns, deduplicated in schema declaration order."""
        cols: list[str] = []
        for setting in self.settings:
            cols.extend(setting.get_output_field_names())
        cols.extend(name for name in (
            self.time_field, *self.label_fields, *self.mask_fields,
            *self.domain_fields, *self.aux_fields,
        ) if name)
        deduped = list(dict.fromkeys(cols))
        if not include_labels:
            label_set = set(self.label_fields)
            deduped = [name for name in deduped if name not in label_set]
        return deduped

    def raw_fields(self, *, include_labels: bool = True) -> list[str]:
        """Pre-transform source columns; generated columns are omitted."""
        cols: list[str] = []
        for setting in self.settings:
            cols.extend(setting.get_raw_field_names())
        cols.extend(name for name in (
            *self.aux_fields, *self.label_fields, *self.mask_fields,
            *self.domain_fields, self.time_field,
        ) if name)
        deduped = list(dict.fromkeys(cols))
        if not include_labels:
            label_set = set(self.label_fields)
            deduped = [name for name in deduped if name not in label_set]
        return deduped

    def get_formatters(self) -> dict[str, ColumnFormatter]:
        """Compile feature, supervision and auxiliary conversion rules."""
        formatters: dict[str, ColumnFormatter] = {}
        for setting in self.settings:
            formatters.update(setting.get_formatters())
        for spec in self._label_specs.values():
            if spec is None:
                continue
            for col, fmt in spec.get_formatters().items():
                if col == spec.field_name:
                    formatters[col] = fmt
                else:
                    formatters.setdefault(col, fmt)
        for spec in self._aux_specs.values():
            formatters.update(spec.get_formatters())
        return formatters


# ==========================================
# 3. 使用示例 (假设使用 KuaiRand)
# ==========================================
if __name__ == "__main__":
    # 假设特征注册
    user_id_setting = SparseEmbSetting("user_id", FeatureSource.USER)

    # 高级序列特征映射：无论原数据是 "1,5,6" 还是 ["1", "5", "6"]，瞬间处理完毕
    tags = MultiSparseSetting(
        field_name="tag",
        source=FeatureSource.ITEM,
        is_string_format=True,
        separator=","
    )

    manager = SchemaManager([user_id_setting, tags])

    from rosen.datasets.kuairand import KuaiRand
    item_lf = pl.scan_csv(KuaiRand.VIDEO_FEATURES)
    inter_lf = pl.scan_csv(KuaiRand.STD_LOG_FORMER_DATA)
    user_lf = pl.scan_csv(KuaiRand.USER_FEATURES)

    whole_lf = inter_lf.join(item_lf, on="video_id", how="left")
    whole_lf = whole_lf.join(user_lf, on="user_id", how="left")
    # 只需要喂入路径，不用 pd.read_csv。即使 kuairand_log.csv 有 50GB 也能在普通笔记本上跑完
    # manager.prepare_data("kuairand_log.csv", "kuairand_processed.parquet")
    manager.prepare_data(whole_lf, "kuairand_processed.parquet", redo=True)
    print("架构编译成功，可供调用。")

    import pandas as pd
    df = pd.read_parquet("kuairand_processed.parquet")
    print(df.head(3).to_string())
