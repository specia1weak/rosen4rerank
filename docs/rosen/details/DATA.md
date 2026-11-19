# data/

`data/` 是把 `LazyFrame / Parquet` 变成 `Interaction` 的流水线层。

## 1. 总流程

```text
Parquet / LazyFrame
  -> DataScanner
  -> (optional) DataTransformer
  -> ShuffleBuffer
  -> TensorFormatter
  -> Interaction
```

## 2. `DataScanner`

源码在 [`src/rosen/data/dataset.py`](../../../src/rosen/data/dataset.py)。

```python
scanner = DataScanner(
    source="train.parquet",          # str / Path / list[str] / pl.LazyFrame
    read_cols=["user_id", "item_id"],
    filter_expr=pl.col("label") > 0,
)
```

### 行为

- 如果 `source` 是 parquet 路径且没有 `filter_expr`，会优先走 `pyarrow.parquet` 的批读取。
- 如果 `source` 是 `pl.LazyFrame` 或存在过滤条件，会走 Polars lazy pipeline。
- 多 worker 时会自动分片：
  - parquet 路径模式按文件索引切分
  - lazy 模式按行号取模切分

### `iter_batches(batch_size, worker_id=0, num_workers=1)`

返回的是 `pyarrow.Table` 迭代器，不是 tensor。

## 3. `DataTransformer`

```python
transformer = DataTransformer(
    manager=manager,
    preprocess_exprs=[pl.col("date").cast(pl.Utf8).str.to_date()],
    filter_expr=pl.col("label").is_not_null(),
    output_col_names=manager.fields(),
)
```

它做三件事：

1. 把 `pyarrow.Table` 转成 `pl.DataFrame`
2. 可选执行预处理表达式和过滤
3. 调 `manager.transform(...)` 再转回 `pyarrow.Table`

## 4. `ShuffleBuffer`

```python
buffer = ShuffleBuffer(
    capacity=2_000_000,
    batch_size=4096,
    shuffle=True,
    drop_last=True,
)
```

### 行为

- 先积累到 `capacity`，再统一产出 batch。
- batch 维度上的 shuffle 是在 buffer 内做的。
- `yield_batches()` 只有在当前缓存行数达到 `capacity` 时才真正输出。
- `flush()` 用来把最后一段剩余数据吐出来。

## 5. `PipelineStreamDataset`

这是所有流式数据集的基类。

```python
dataset = PipelineStreamDataset(
    scanner=scanner,
    formatter=formatter,
    transformer=transformer,
    buffer=buffer,
)
```

`__iter__()` 会按 worker 自动读取、转换、切 batch、格式化，最终输出 `Interaction`。

## 6. `ParquetStreamDataset`

这是最常用的版本，前提是你已经有编码后的 parquet。

```python
ds = ParquetStreamDataset(
    parquet_path="train.parquet",
    manager=manager,
    batch_size=4096,
    shuffle=True,
    drop_last=True,
)
```

### 关键点

- 默认读取列由 `FeatureContext.from_manager(manager)` 推导。
- `extra_col_names` 和 `extra_col_formatters` 可以补充额外列。
- `batch_size` 只控制 dataset 内部 batch 的大小，不是外层 `DataLoader` 的 batch。

## 7. `RawParquetStreamDataset`

用于“原始 parquet，读取时再 transform”。

```python
ds = RawParquetStreamDataset(
    parquet_path="train_raw.parquet",
    manager=manager,
    batch_size=4096,
    shuffle=True,
    raw_preprocess_exprs=[pl.col("date").cast(pl.Utf8)],
    raw_filter_expr=pl.col("split") == "train",
)
```

### 行为细节

- 如果 `raw_preprocess_exprs` 为空，`raw_filter_expr` 会先在 scanner 阶段生效。
- 如果 `raw_preprocess_exprs` 不为空，过滤会延后到 transformer 阶段。

## 8. `FeatureContext` 和 `TensorFormatter`

源码在 [`src/rosen/data/padding.py`](../../../src/rosen/data/padding.py)。

### `FeatureContext`

```python
FeatureContext.from_manager(manager)
FeatureContext.from_raw_manager(manager)
# 独立推理：读取同一套输入和 mask，不要求标签列存在。
FeatureContext.from_manager(manager, include_labels=False)
```

- `from_manager()` 用于已经编码好的数据。
- `from_raw_manager()` 用于原始数据，需要同时推导读入列和输出列。
  它由 RawParquetStreamDataset 实际使用：读入 `manager.raw_fields()`，输出 `manager.fields()`。
  例如 history_len 由 transform 生成，不应要求原始 parquet 包含此列。
- 两个工厂都支持关键字 `include_labels=False`：从读入列和输出列排除全部
  `manager.label_fields`，保留关联的 `mask_fields`、AuxField 和其他上下文列。
  不补造标签，也不根据 parquet 是否有标签自动切换训练/推理语义。
- 正式辅助输入在 `SchemaManager(aux_fields=[AuxField(...)])` 中声明名称、类型和长度，
  dataset 与独立推理共用其读取及格式化规则，无需重复 `extra_col_names` 或 formatter。
- 列名保存为 tuple；额外 formatter 映射复制后设为只读，防止调用方修改原 dict 影响上下文。
  上下文支持 pickle 和 DataLoader spawn；它仍引用原 manager，不会深度冻结 manager 或自定义 formatter。

### `TensorFormatter`

```python
formatter = TensorFormatter(context)
interaction = formatter.format(batch_dict)
```

它调用 `manager.get_formatters()` 获取统一的特征、SlateLabel 和 AuxField 规则，
再合并 `extra_col_formatters`（优先级最高），把 numpy / list 转成 tensor 并包成 `Interaction`。
正式规则由各声明的 `get_formatters()` 产生；不探测残缺 manager 是否提供可选方法。
推理上下文保留同一套掩码与辅助字段格式化规则。

## 9. Formatter 类型

规范定义在 [`core/formatters.py`](../../../src/rosen/core/formatters.py)，只依赖 NumPy/Torch。
按本轮明确要求，原公共类和常量仍从 `data.padding` re-export，指向同一对象。

- `DenseFormatter`
- `VectorDenseFormatter`
- `IntFormatter`
- `FallbackFormatter`
- `PaddedIntSequenceFormatter`
- `PaddedFloatSequenceFormatter`
- `PaddedNestedSequenceFormatter`

### 约束

- `VectorDenseFormatter` 会固定到 `dim`，超长截断，短的补 0。
- `PaddedNestedSequenceFormatter` 用于嵌套 list，例如“标签序列的序列”。
- 序列 formatter 的 `truncate_mode="head"/"tail"` 分别保留头部/尾部，默认保留尾部，
  与左右 padding 无关。`SequenceSetting` 把自身方向同时传给元素与时间列 formatter。
- `MultiSparseSetting` 固定保留标签头部，与 Polars `list.head(max_tag_len)` 一致；
  嵌套 formatter 的外层按序列方向截断，内层由 `tag_truncate_mode` 独立控制（默认 head）。
- `FallbackFormatter` 只是兜底，不要拿它当正式 schema 规则。

