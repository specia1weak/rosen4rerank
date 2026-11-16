# 框架维护与架构优化

这篇文档对应用户说“rosen 这里有坏味道”“这个逻辑该不该上游”“代码结构是不是不对”的场景。

目标不是把代码改得更抽象，而是让训练链路更清楚、更可复用、更少一次性补丁。

## 1. 先判断坏味道属于哪一层

按训练链路定位：

```text
raw data / preprocessing
    -> SchemaManager / EmbSetting
    -> parquet dataset / scanner / shuffle
    -> Interaction
    -> model / custom_models
    -> Trainer / Evaluator
    -> experiment script / logs / docs
```

常见归类：

| 表现 | 优先检查 |
|---|---|
| 同一数据处理逻辑在多个脚本复制 | `utils/`、`SchemaManager`、dataset preparation docs |
| schema 复用导致结果怪 | `feature_meta.json`、`SchemaManager.fit()`、workdir |
| dataset 读取慢或顺序偏置 | `ParquetStreamDataset`、物理 shuffle、worker 设置 |
| 新模型参数到处硬编码 | `cfg.extras`、`MODEL_KWARGS`、`custom_models` 注册 |
| trainer 逻辑变复杂 | `custom_train_step`、hook、是否该新增 Trainer |
| 指标口径说不清 | `EvaluatorManager`、domain filter、test/valid 路径 |

## 2. 维护原则

- 优先让一个实验入口能读清楚完整链路，而不是过早隐藏到框架黑盒。
- 能放 `src/custom_models/` 的研究模型，不要急着放进核心 `src/rosen/models/msr/`。
- 能作为通用数据工具沉到 `src/rosen/utils/` 的，不要散落在多个 `@examples/`。
- 核心接口变更要尽量向后兼容；不能兼容时要保留 fallback 或明确迁移路径。
- 不要因为一次实验需要而污染公共抽象。
- 文档和最小验证是框架能力的一部分，不是事后装饰。

## 3. 推荐处理步骤

1. 复述用户指出的坏味道，并确认它影响的是可读性、复用性、性能、正确性还是实验可比性。
2. 查现有文档：
   - 架构：[`ARCHITECTURE.md`](ARCHITECTURE.md)
   - 设计理念：[`DESIGN_PHILOSOPHY.md`](DESIGN_PHILOSOPHY.md)
   - 数据流：[`DATA_FLOW.md`](DATA_FLOW.md)
   - 训练流：[`TRAINING_FLOW.md`](TRAINING_FLOW.md)
   - 模块细节：[`details/`](details)
3. 用源码确认真实行为，不只看文档。
4. 选择最小修改点：
   - 公共能力：`src/rosen/...`
   - 实验模型：`src/custom_models/...`
   - 数据处理入口：`@examples/<dataset>/...`
   - 人类结论：`research/...`
5. 保留兼容开关或 fallback，尤其是数据处理、shuffle、dataset 读取这类高风险路径。
6. 跑最小验证，再跑用户关心的数据集或实验。
7. 在最终汇报里说明：
   - 改了哪个层次
   - 为什么不是只改脚本
   - 有什么兼容路径
   - 验证结果是什么

## 4. 不应该做什么

- 不要把 `@examples/` 的一次性实验参数提升成全局配置。
- 不要为了“优雅”新增多层抽象，导致实验入口看不出真实数据流。
- 不要在没有验证的情况下替换历史数据处理逻辑。
- 不要默默删除 workspace/logs 产物。
- 不要把历史实验结论和当前代码行为混在一起说。

## 5. 需要重点保护的契约

### Schema 契约

`SchemaManager.fit(train_raw)` 只能用训练集拟合 schema。`feature_meta.json` 一旦存在，后续 fit 可能直接加载旧状态。

### Dataset 契约

`ParquetStreamDataset` 是 batch-level iterable dataset。外层 `DataLoader` 必须使用 `batch_size=None`。

### Model 契约

MSR 模型默认从 `Interaction` 里拿 domain 和 label。`predict()` 的输出会直接进入 evaluator。

### Experiment 契约

同一个结论至少要能追到：

- 数据集路径
- schema/workdir
- 训练脚本
- 参数
- log
- 主指标

## 6. 什么时候可以“上线替代”

满足这些条件时，一个新工具或新逻辑可以替代旧逻辑：

- 语义比旧逻辑更清楚，或者明确修复旧逻辑的正确性问题。
- 已保留旧逻辑作为 fallback，至少一个版本周期内不硬删。
- 在小表上有 correctness/smoke 验证。
- 在代表性数据集上跑过一次端到端实验。
- 文档写清楚默认策略、fallback 和风险。

最近的例子是物理 shuffle：`target_sort` 修复了 append-order 泄漏，并保留 `bucket_chunk` / `append_bucket` 作为兼容模式。细节见 [`SHUFFLE.md`](SHUFFLE.md)。
