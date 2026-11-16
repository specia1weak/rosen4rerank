# 新数据集接入 / Parquet 化

## 适用场景

用户因为想要实验，从而希望处理一个历史未处理过的新数据集，或者把一个已有 raw 数据集转成 rosen 可训练 parquet。

## 先读什么

- `docs/rosen/DATA_FLOW.md`
- `docs/rosen/details/DATASETS.md`
- schema 用法：`docs/rosen/details/EMB.md`
- 现有相似数据集脚本：`@examples/<dataset>/`
- 大数据和顺序风险相关时，查 `src/rosen/utils/split_shuffle.py`、`src/rosen/utils/profile_sequence.py`

## 边界约束

- datasource 可以保存原始文件，或信息无损、实验无关、易于二次派生的 canonical 数据；例如保留
  ALICCP 全部信息的 typed Parquet。普通实验只读 datasource。
- top-k 过滤、字段裁剪、采样、split、train-fit schema/编码、profile、截断序列和 shuffle 等
  实验派生结果写入 workspace，不能因为“会复用”就写回 datasource。
- 判断 canonical 加工是否合法时，同时验证信息完备性和后续低成本派生性，并保存转换脚本与 manifest。
- 数据处理服务于实验，不要脱离 label、domain、切分和评估口径单独“清洗数据”。
- `SchemaManager.fit()` 应只基于 train split。
- train/valid/test、profile、sequence 的边界要说清楚，不能让训练样本和评估样本混在一起。
- 大数据处理前要关注磁盘、内存、临时文件和 shuffle 成本。

## 需要做的事情

### 理解实验需求和 raw 数据

先确认这个数据集要支持什么实验，而不是只看 raw 文件能不能读。需要理解 label、domain、用户/物品主键、时间字段、切分规则，以及是否需要 profile 或 sequence。

同时要快速检查 raw 数据结构和规模，判断数据是否需要 join、过滤、类型修正、缺失处理，是否存在会带来内存压力的大列或列表列。

### 设计 rosen 数据形态

把 raw 数据转成 rosen 支持的形态：

- 非序列任务通常产出 transformed `train.parquet`、`valid.parquet`、`test.parquet`。
- 序列/profile 任务需要额外考虑 `main` 样本和 `profile` 侧信息的关系。
- train 用于正式训练时应考虑物理 shuffle；valid/test 保持稳定顺序。

设计时要让后续实验脚本能清楚看到：数据来自哪里，怎么切分，schema 怎么拟合，最终模型消费哪些字段。

先区分当前任务是在建立 canonical source，还是在构造实验数据：前者可以受控写入 datasource 并在
完成后冻结；后者必须写入 `workspace/datasets/<dataset>/<version>/`。

### 编写处理逻辑并验证

优先复用已有 dataset、schema、profile、shuffle 工具。只有当现有工具无法表达当前数据形态时，才新增处理逻辑。

处理完成后不急着 full train，先做基础验证：

- parquet 能正常扫描。
- schema 能 fit/transform。
- dataset 能读出一个 batch。
- train/valid/test 行数和关键分布合理。
- sequence/profile 的长度和覆盖范围合理。

### 评估资源和稳定性风险

如果数据较大，需要提前估计 OOM 和 I/O 风险。重点关注 raw join、schema fit、profile 构造、shuffle、parquet 落盘的峰值内存和磁盘空间。

长时间处理任务要记录日志，让用户能观察进度。

### 报告情况并写回文档

汇报时要说明：

- 产出数据路径。
- train/valid/test 或 main/profile 的规模。
- schema 和切分口径。
- 是否已经做过 smoke。
- 是否已经能训练，还是只是数据 ready。
- 资源风险和后续建议。

如果这是长期会复用的数据集接入，需要写回文档或交接记录。

## 文档写回要求

如果是新数据集的长期接入，需要补：

- 数据来源和 raw 结构。
- 切分规则和实验口径。
- schema/profile/sequence 设计。
- 输出目录约定。
- 处理命令和 smoke 结果。
- OOM 或性能注意事项。

可以写在 `docs/rosen/` 的数据集文档，也可以先写到对应 `research/handoffs/`。

## 完成标准

- 产出可训练 parquet 或 profile-seq 数据集。
- 能读出一个 batch。
- 用户知道数据路径、行数、切分、shuffle 策略。
- 如果没有训练 AUC，要明确说明“数据 ready，但还未训练”。
