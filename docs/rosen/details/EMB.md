# emb/

这是 rosen 最核心的一层。它决定了“原始字段如何变成可训练张量”。

## 1. 三层结构

```text
Schema 层   -> 每个字段的规则定义
Manager 层  -> 规则编排、拟合、转换、落盘
Embedding 层 -> 在 forward 时按规则取 tensor / 查表
```

## 2. Schema 层

源码在 [`src/rosen/emb/schema`](../../../src/rosen/emb/schema)。

### `AuxField`：不进入 Embedding 的模型输入

```python
from rosen.emb import AuxField, SchemaManager

manager = SchemaManager(settings, "schema", label_fields=labels, aux_fields=[
    AuxField("prior_long_view", max_len=6),
    AuxField("page_mask", max_len=2),
    AuxField("request_id", dtype="int64"),
    AuxField("user_id_raw", dtype="int64"),
])
```

AuxField 支持 float32/int64、标量或 max_len 列表、左右 padding 和 pad_val。
这些列自动进入 manager.fields()、原始/已转换 parquet 读取和 TensorFormatter；
不加入 settings，不参与 fit，也不创建 embedding。与既有模型字段或监督字段重名会报错。
aux_fields 和 get_aux_spec(name) 用于读取声明。与 SlateLabel 一样，调用方加载 Schema 时
需重新提供声明；feature_meta.json 仍仅存已拟合的特征状态，兼容已有数据。
额外的 ID/调试列仍可用 extra_col_names；原有 extra_col_formatters 保留显式覆盖行为。
KR 与 ML1M 重排示范的正式输入及原始 ID 均在 manager 工厂声明 AuxField，loader 不再按任务
维护额外列清单。独立推理使用 `FeatureContext.from_manager(manager, include_labels=False)`，
自动保留 AuxField 和 SlateLabel 关联 mask，跳过标签。KR 的 `prior_ctr` 继续保存在数据中，
当前重排与评估不消费它，因此不读入模型 batch。

### `EmbType`

```text
UNKNOWN
SPARSE
MULTI_SPARSE
QUANTILE
SPARSE_SEQ
SPARSE_SET
DENSE
LOG1P_DENSE
VECTOR_DENSE
DENSE_SEQ
SEQUENCE
```

### `SeqGroupConfig`

```python
SeqGroupConfig(
    group_name: str,
    seq_len_field_name: str,
    max_len: int,
    padding_side: "left" | "right" = "right",
    time_field_name: Optional[str] = None,
)
```

### `EmbSetting`

所有 setting 的抽象基类，定义了这条生命周期：

```text
get_fit_exprs()
parse_fit_result()
get_transform_expr()
get_formatters()
compute_tensor()
```

并提供这些通用属性：

- `field_name`
- `embedding_dim`
- `source`
- `padding_zero`
- `use_oov`
- `vocab`
- `oov_idx`
- `num_embeddings`

### `SparseEmbSetting`

```python
SparseEmbSetting(
    field_name,
    source,
    embedding_dim=16,
    padding_zero=True,
    use_null=True,
    fill_null="null",
    min_freq=1,
    use_oov=True,
)
```

要点：

- 会统计词频。
- 低于 `min_freq` 的值不会进入词表。
- 支持 `fill_null="null" | "oov" | "zero" | None`。
- `compute_tensor()` 通过 `nn.Embedding` 查表。

### `MultiSparseSetting`

用于 list / 分隔字符串表示的多值特征。

```python
MultiSparseSetting(
    field_name,
    source,
    embedding_dim=16,
    max_tag_len=5,
    is_string_format=False,
    separator=",",
    padding_zero=True,
    min_freq=1,
    use_oov=True,
    agg="sum",
)
```

要点：

- 支持字符串和 list 两种输入格式。
- 会先清洗、截断，再做词表映射。
- `agg="sum"` 时直接求和；`agg="mean"` 时会做 mean pooling。

### `QuantileEmbSetting`

把连续值分位数离散化后再查表。

### `MinMaxDenseSetting`

对数值做 min-max 归一化，**不建 embedding table**。

### `Log1PDenseSetting`

```python
from rosen.emb.schema import Log1PDenseSetting

duration = Log1PDenseSetting("duration", FeatureSource.ITEM)
```

计算自然对数 `ln(1 + x)`，不需要训练统计量，构造后即可 transform，不建 embedding table。
在 Float64 上调用 Polars `log1p()`，最后输出 float32，以保留小数精度并支持较大的原始值。
null 或无法转为数值的值补 0；`-1 < x < 0` 输出负数，`x == -1` 输出 `-inf`，
`x < -1` 输出 NaN，已有 NaN/inf 保留。调用方应按数据语义准备有效输入。
可直接作为 `SequenceSetting` 的 element_setting，复用序列截断、padding 和 DenseFormatter。
schema 类型为 `LOG1P_DENSE`，支持 `to_dict/from_dict` 与 manager 保存/加载。

### `VectorDenseSetting`

固定长度向量特征，原始值会保留为 list，真正的 padding / truncation 在 formatter 阶段处理。

### `SequenceSetting`

历史序列 setting。它复用 `element_setting` 的词表，不自己再建一套。

关键约束：

- `from_dict()` 没有实现，说明它不能完全靠 JSON 重建。
- 输出字段通常包括：
  - 序列本体
  - `seq_len_field_name`
  - 可选 `time_field_name`

### `SlateLabel`

标签和曝光掩码属于监督字段，不放进 `SequenceSetting` 或 `SeqGroupConfig`。

```python
from rosen.emb import SchemaManager, SlateLabel

manager = SchemaManager(
    settings_list=settings,
    schema_dir="workspace/example/schema",
    label_fields=[SlateLabel("click", max_len=5, mask_field="mask"), "retention"],
)
assert manager.label_fields == ("click", "retention")
assert manager.mask_fields == ("mask",)
```

`SlateLabel` 是不可变声明，要求非空字段名、正整数 `max_len` 和合法 padding 方向。
`get_label_spec("click")` 返回声明，标量/未知字段返回 `None`。相同声明去重，冲突声明报错；
多个标签共用 mask 时长度和 padding 方向必须一致，manager 在构造时校验。
prepared/raw dataset 均自动读取 mask，生成 float32 `[B, K]` 标签与掩码。缺失整行或空列表
使用 padding，标签默认补 0，mask 总是补 0；左/右 padding 均支持，超长列表取尾部。
显式 `extra_col_formatters` 可以覆盖默认规则。

调用方重建 manager 时继续声明监督规格；`save_schema` / `load_schema` 仍只保存/恢复特征状态。
`BaseModel.get_label_mask` 可读取声明的 mask，或回退到与标签同形状的全 1；有效位如何进入
loss 由模型决定，formatter 不从点击值推导有效曝光。

## 3. Manager 层

源码在 [`src/rosen/emb/manager.py`](../../../src/rosen/emb/manager.py)。

### 初始化

```python
manager = SchemaManager(
    settings_list=settings,
    work_dir="./workspace/demo",
    time_field="timestamp",
    label_fields="label",
    domain_fields="domain_id",
)
```

### 真实会发生什么

- `uid_field` 会自动从 `FeatureSource.USER_ID` 的 setting 找。
- `iid_field` 会自动从 `FeatureSource.ITEM_ID` 的 setting 找。
- `fit()` 会把 schema 存成 `feature_meta.json`。
- `transform()` 只负责套用已经固化的表达式。

共享同一个 `element_setting` 实例的多个序列只拟合一次。两种内存模式均按 settings
顺序取第一个来源；即使原子本身也在顶层，也遵循该顺序。后续序列独有值不会加入词表，
后续来源的数值也不会扩展 min/max。需要覆盖其他来源时，先显式拟合原子再共享它。
保存与加载时同步各包装器的 fitted 状态，同源稀疏序列复用一张 embedding 表。

### 常用方法

- `fit(train_raw_lf, low_memory=False)`
- `transform(raw_lf)`
- `prepare_data(lazy_df, output_dir=None, redo=False)`，不推荐作为主路径
- `split_dataset(lf, strategy="random_ratio", **kwargs)`
- `save_as_dataset(train_lf, valid_lf, test_lf, output_dir=None, redo=False)`
- `save_schema()`
- `load_schema()`
- `generate_profiles(lazy_df, output_dir=None, redo=False)`
- `make_checkpoint(lf, file_name="custom_checkpoint.parquet", redo=True, sort_by=None, descending=False)`
- `fields()`
- `get_setting(field_name)`
- `source2emb_dim(*sources)`

### 切分策略

`split_dataset()` 支持：

- `loo`
- `time`
- `sequential_ratio`
- `random_ratio`

对应配置类分别是：

- `LooConfig(k_core=3)`
- `SequentialRatioConfig(train_ratio=0.8, valid_ratio=0.1)`
- `TimeSplitConfig(valid_start, test_start)`
- `RandomRatioConfig(train_ratio=0.8, valid_ratio=0.1, group_by=None)`

### 一个重要坑

`fit()` 发现 `feature_meta.json` 已存在时，会直接加载旧 schema 并跳过重新拟合。
如果你修改了 setting 定义，但还在用同一个 `work_dir`，很容易以为代码生效了，实际上拿到的是旧结果。

## 4. Embedding 层

源码在 [`src/rosen/emb/emblayer.py`](../../../src/rosen/emb/emblayer.py)。

### `RecEmbedding`

对 `nn.Embedding` 的轻量包装，支持初始化方式和 `reinitialize()`。

### `BoleEmbLayer`

按 `FeatureSource` 分组，把 `EmbSetting.compute_tensor()` 逐个拼起来。

### `SideEmb`

旧式“按 source 取特征”封装，现在更多时候直接用 `OmniEmbLayer`。

### `EmbView`

只是一个轻量视图对象，不持有参数，负责把调用转发给 `OmniEmbLayer`。

### `SeqGroupView`

用来把某个序列组里的：

- 序列本体
- 目标特征
- 序列长度
- 可选时间序列

一次性取出来。

### `OmniEmbLayer`

这是当前最重要的类。

```python
omni = OmniEmbLayer(manager=manager)
# 或
omni = OmniEmbLayer(emb_settings=settings)
```

#### 预置视图

- `whole`
- `whole_without_domain`
- `user_all`
- `item_all`
- `inter`
- `inter_without_domain`
- `user_id`
- `item_id`
- `domain`
- `domain_id`

#### 调用方式

```python
emb = omni.whole(interaction)
emb_dict = omni.whole(interaction, split_by="source")
emb_by_name = omni.whole(interaction, split_by="name")
fields = omni.whole(interaction, split_by="stack")  # [B, F, D]
slate = omni.seq_groups["slate"](interaction, split_by="stack")  # [B, K, F, D]
```

`stack` 保留字段轴，沿 `dim=-2` 堆叠；已有前导维不变，因此也适用于更高阶的字段张量。
字段顺序与默认拼接一致。所有字段的实际 shape 必须一致，异宽字段或不同长度的序列应通过
`include_fields` / `exclude_fields` 分开选择。stack 不新增参数、不改变 embedding 初始化或共享关系。

`view.num_fields` 表示字段数；`view.field_dim` 表示统一的字段宽度，字段异宽或视图为空时
抛出 `ValueError`。`view.embedding_dim` 继续表示拼接后的总宽度。空视图前向继续返回 `None`。

#### 过滤规则

- `target_sources` 先筛 source。
- `include_fields` / `exclude_fields` 再按字段过滤。
- 序列 setting 默认不会混进 `split_by="none"` 或 `split_by="stack"` 的结果，除非你显式 include 它。
  `seq_groups[...]` 视图已经显式选择本组字段。

#### 输出

- `split_by="none"` -> `Tensor[B, D]`
- `split_by="source"` -> `Dict[FeatureSource, Tensor[B, D_i]]`
- `split_by="name"` -> `Dict[str, Tensor[B, D_i]]`
- `split_by="stack"` -> `Tensor[B, ..., F, D]`，其中 `F` 为字段数，`D` 为共有字段宽度

#### 额外功能

`reinitialize_large_vocab_embeddings(vocab_size_threshold, ...)` 可以把词表过大的 embedding 重新初始化，但这是一个显式 opt-in 的工具，不在默认训练流里自动执行。


## 旧接口隔离

emblayer.py 只定义 RecEmbedding、EmbView、SeqGroupView、OmniEmbLayer。
BoleEmbLayer、各 SideEmb、ProfileEncoder 及 SeqEmbedder 的十个类移至 legacy.py，类体保持原样。
原 rosen.emb.emblayer 导入和旧 pickle 查找通过延迟属性转发到同一个 legacy 类；
现代模型不加载旧画像接口。SeqGroupView.fetch_all 不会把 ValueError 对象当 Tensor 返回。
