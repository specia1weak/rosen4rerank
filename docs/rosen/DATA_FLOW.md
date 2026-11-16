# 数据流

这篇文档只讲一件事：`rosen` 里的数据是怎样从原始表流到训练 batch 的。

如果你认同“一个实验文件里应该能看清完整数据流向”，这篇就是那条主链路的解释版。

## 1. 总链路

```text
raw dataframe / lazyframe
    -> split
    -> fit(train_raw)
    -> transform(train/valid/test)
    -> save_as_dataset(...)
    -> ParquetStreamDataset / RawParquetStreamDataset
    -> Interaction
```

这里最重要的原则是：

- 切分是实验逻辑
- schema 拟合和编码是基础设施
- dataset 只是把编码后的数据流式组织成 batch

## 2. 原始数据阶段

你通常从一个宽表开始：

- `pl.scan_parquet(...)`
- `pl.scan_csv(...)`
- 多张表 join 之后的 `LazyFrame`

这一步仍然属于实验入口脚本自己负责的范围。

框架不会替你决定：

- 哪些表要 join
- 哪些列要保留
- 哪些样本要过滤

## 3. 切分阶段

切分可以有两种常见写法：

### 显式手工切分

例如按时间、日期或业务规则：

```python
train_raw = whole_lf.filter(pl.col("date") <= 20220506)
valid_raw = whole_lf.filter(pl.col("date") == 20220507)
test_raw = whole_lf.filter(pl.col("date") == 20220508)
```

### 用 `SchemaManager.split_dataset(...)`

适合比较标准的切分策略：

- `random_ratio`
- `sequential_ratio`
- `time`
- `loo`

项目当前更推荐“看得见规则”的显式切分，尤其是在真实实验脚本里。

## 4. `fit(train_raw)` 阶段

`SchemaManager.fit()` 只使用训练集拟合：

- 稀疏词表
- 数值归一化统计量
- 其他特征相关元数据

这一步的核心目的是：把“训练集学到的 schema”固定下来。

当前还有一个很重要的行为：

- `work_dir/feature_meta.json` 已存在时，`fit()` 会直接加载旧 schema
- 这会影响重复实验的结果复用

所以当你更换了特征定义或词表规则时，要有意识地更换 `work_dir` 或清理旧元数据。

## 5. `transform(...)` 阶段

`transform()` 的职责是：

- 按照已经固化的 schema 对原始列做编码或变换
- 生成模型真正会消费的输出列

这里要注意：

- 不是所有保留列都必须定义成 `EmbSetting`
- `label_fields` 和 `domain_fields` 即使不做 embedding，也会被保留

这正是为什么 domain 列通常可以不注册 embedding，但后续模型仍然能从 batch 里拿到它。

## 6. `save_as_dataset(...)` 阶段

这一步把编码后的 `LazyFrame` 落盘成更稳定的 parquet 数据集形式，后面可以反复被 dataset 消费。

主收益是：

- 数据处理与训练解耦
- 编码结果可复用
- 大数据量时不必每次训练都重新编码

## 7. `ParquetStreamDataset` 阶段

`ParquetStreamDataset` 的职责不是“再做一次 schema”，而是：

- 流式扫描 parquet
- 读取字段
- 组织 batch
- 产出 `Interaction`

有一个很容易误解的点：

- `ParquetStreamDataset` 本身已经按 `batch_size` 产出 batch
- 它输出的是 batch 级对象，不是单条样本

因此如果你再额外包一层 `DataLoader`，必须使用：

```python
DataLoader(dataset, batch_size=None)
```

## 8. `Interaction` 阶段

训练和评估时，模型真正收到的是 `Interaction`。

你可以把它理解成：

- 一个以字段名为 key 的 batch tensor 容器
- 里面包含模型前向所需列，以及 label、domain、uid 等上下文字段

这也是为什么上游字段保留规则非常重要。后面的模型和 trainer 都默认这套 batch 结构已经对齐。

## 9. 一张输入输出表

| 阶段 | 输入 | 输出 | 谁负责 |
|---|---|---|---|
| 原始表准备 | 原始 csv/parquet/join 后 lazyframe | `LazyFrame` | 实验脚本 |
| 切分 | `LazyFrame` | `train_raw/valid_raw/test_raw` | 实验脚本或 `split_dataset()` |
| `fit` | `train_raw` | 固化 schema 元数据 | `SchemaManager` |
| `transform` | raw split | encoded split | `SchemaManager` |
| `save_as_dataset` | encoded split | parquet dataset path | `SchemaManager` |
| dataset | parquet path + manager | `Interaction` batch | `ParquetStreamDataset` |

## 10. 最常见的三个坑

### 坑一：改了特征规则，但还是复用了旧 schema

表现：

- 新增列不生效
- vocab 和预期不一致
- 训练结果异常稳定但不对

先检查 `work_dir/feature_meta.json` 是否来自旧实验。

### 坑二：把 domain 列当成“只有做 embedding 才会出现在 batch”

不是。只要它被声明为 `domain_fields`，就可以被保留到输出结构里。

### 坑三：再包一层 `DataLoader` 导致 batch 维度错

因为 `ParquetStreamDataset` 已经产出 batch 了。

## 11. 继续阅读

- 想看数据相关模块细节：[`details/DATA.md`](details/DATA.md) 和 [`details/EMB.md`](details/EMB.md)
- 想看这条链路如何进入训练：[`TRAINING_FLOW.md`](TRAINING_FLOW.md)
