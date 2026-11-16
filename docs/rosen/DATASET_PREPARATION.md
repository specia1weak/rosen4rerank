# 新数据集处理

这篇文档对应用户希望处理一个历史未接入的新数据集，把它转成 rosen 可训练 parquet 的场景。

目标产物通常是：

```text
workspace/<dataset>/
  feature_meta.json
  train.parquet
  valid.parquet
  test.parquet
```

对于 profile/sequence 数据集，目标产物也必须位于 workspace，例如：

```text
workspace/datasets/<dataset>/<version>/
  main/train.shuffle/*.parquet
  main/valid.parquet
  main/test.parquet
  profiles/...
```

## 1. 先问清楚任务定义

处理数据前先确认：

- label 是什么
- domain 字段是什么
- user/item id 字段是什么
- train/valid/test 如何切分
- 是否需要用户 profile 或行为序列
- 是否是 full train/full eval，还是只做 smoke/profile

如果用户没有明确指定，优先采用已有数据集脚本里的业务切分规则，而不是随意 random split。

## 2. datasource 是信息完备的数据源边界

数据源根由环境变量 `ROSEN_DATASOURCE_ROOT` 指定；未设置时，当前机器回退到
`/home/lx/datasource/`。datasource 可以包含两类资产：

- 供应方或公开数据集的原始文件；
- 信息无损、实验无关的规范化形式，例如保留 ALICCP 全部行列与顺序语义的 typed Parquet。

规范化加工能否进入 datasource，使用两个判断标准：

1. 后续合理实验能否从它取得足够且完整的信息；不能因过滤、采样、截断、字段裁剪、train-fit 编码
   或不可逆映射而丢失实验自由度。
2. 后续二次加工是否足够简单；规范化结果应消除昂贵的原始解析，但不能绑定某个模型、domain 子集、
   序列长度、split 或采样口径。

通过两个测试的 canonical parquet 属于数据接入结果，可以写入 datasource 并在生成 manifest 后冻结。
普通训练、评估和实验预处理只能读取 datasource，不能随意写入。

允许的读取和写出位置：

- 原始或无损规范化输入：`$ROSEN_DATASOURCE_ROOT/<dataset>/`
- 可复用派生数据：`workspace/datasets/<dataset>/<version>/`
- 实验输出：`workspace/runs/<dataset>/<run>/` 或已有的 `workspace/<dataset>/...`
- 临时 smoke 输出：`workspace/smoke/<dataset>/...` 或 `/tmp/...`

禁止把实验相关的 split、schema、top-k domain 过滤、train-fit encoded parquet、正样本 profile、
截断 sequence store、shuffle shard、cache、日志或 checkpoint 写进 datasource。无损的格式转换、
类型规范化和便于扫描的物理分区可以接受，但必须保留全部信息、由正式脚本生成，并记录 manifest。

历史机器上已有一些加工资产，应逐项按“信息完备性 + 低成本派生性”判断。ALICCP 的无损 Parquet
可视为 canonical source；top5、positive-only profile、domain 重映射或截断 sequence 等是 legacy
实验派生路径，不能作为新脚本模板。在完成引用清单和迁移验证前，也不要直接移动或删除历史资产。

## 3. 基础非序列数据流程

推荐主链路：

```text
raw csv/parquet
    -> Polars LazyFrame
    -> join user/item/context side features
    -> select keep columns
    -> split train/valid/test
    -> SchemaManager.fit(train_raw)
    -> SchemaManager.transform(...)
    -> save_as_dataset(...)
```

关键点：

- `fit()` 只能用 train。
- `label_fields` 和 `domain_fields` 即使不注册 embedding，也会进入输出字段。
- 改了特征定义时，要换 `work_dir` 或清理旧 `feature_meta.json`。
- `save_as_dataset()` 产出的 parquet 是模型输入，不等同 raw data。

## 4. profile/sequence 数据流程

如果用户需要长序列、用户历史、profile 复用，优先看已有工具：

- [`src/rosen/utils/profile_sequence.py`](../../src/rosen/utils/profile_sequence.py)
- [`src/rosen/utils/split_shuffle.py`](../../src/rosen/utils/split_shuffle.py)

推荐流程：

```text
raw logs
    -> 按 uid hash 分区
    -> 每个 user partition 内按 user/time/row_nr 稳定排序
    -> 构造 profile-backed sequence main_parts + profiles
    -> train/valid/test split
    -> train 物理 shuffle
```

注意：

- profile 不一定需要 train/valid/test 划分，它是按用户历史构造的可复用侧信息。
- main 样本必须按实验口径切分。
- valid/test 不做物理 shuffle。
- train 默认使用 [`SHUFFLE.md`](SHUFFLE.md) 中的 `target_sort`。

## 5. shuffle 默认策略

正式训练数据默认做物理 shuffle。

推荐：

```text
train_shuffle_mode = "target_sort"
bucket_count       = power-of-two
```

如果是 user shard 数据，常见配置：

```text
target_count = user_partitions * 4
例如 256 user shards -> 1024 train targets
```

更多细节见 [`SHUFFLE.md`](SHUFFLE.md)。

## 6. 数据处理脚本放哪里

- 新数据集的正式处理入口放 `@examples/<dataset>/`。
- 可以复用的通用逻辑放 `src/rosen/utils/`。
- 数据集路径适配器如果会长期使用，再放 `src/rosen/datasets/`。
- 一次性临时分析脚本不要提升到核心框架。

命名建议：

```text
@examples/<dataset>/<dataset>_prepare.py
@examples/<dataset>/<dataset>_profile_seq.py
@examples/<dataset>/train_<model_or_task>.py
```

## 7. 必做检查

处理完成后至少检查：

- train/valid/test 行数
- user/item/domain/label 的空值和取值范围
- train 是否物理 shuffle
- schema 是否来自当前特征定义
- parquet 文件是否可被 `ParquetStreamDataset` 读出一个 batch
- 如果有 profile/sequence，检查 `seq_len`、最大长度、profile 行数和用户覆盖

可以用 Polars/pyarrow 做轻量检查，不要一上来启动完整训练。

## 8. OOM 风险评估

数据接入前要估算：

- raw join 后宽表大小
- 稀疏词表规模
- 是否有大字符串/列表列
- profile 列是否会膨胀
- shuffle 临时文件峰值空间
- `fit(low_memory=True)` 是否必要

对于大数据：

- 优先 LazyFrame 和 streaming collect/sink。
- 避免把全表 collect 到内存。
- 先做 100 step 或小分区 smoke。
- 记录 CPU/RAM/GPU/磁盘空间。

## 9. 最终汇报格式

用户通常最关心“能不能训练”。汇报时给：

```text
数据路径:
train/valid/test 行数:
schema 路径:
shuffle 策略:
smoke 或 full train 结果:
风险/未完成:
```

如果只是预处理完成，还没训练，要明确说“数据已 ready，但还没产生 AUC”。
