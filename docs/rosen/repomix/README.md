# rosen 核心数据流源码包

导出文件：[`rosen-core.md`](rosen-core.md)。只收录 rosen 核心链路及其相关实现；入选文件完整保留，不删除注释、空行或函数体，不做代码压缩。源码中的历史注释、示例和现有行为也原样保留。这个文件是阅读包，不是可单独运行的源码发行包。

## 数据怎么走

```text
实验脚本读取、join、过滤原始表（Polars LazyFrame）
  -> 显式切分 train / valid / test，或 SchemaManager.split_dataset()
  -> SchemaManager.fit(train_raw)：拟合词表、数值统计量，保存 schema
  -> SchemaManager.transform(split)：按 schema 编码各 split
  -> SchemaManager.save_as_dataset()：写入 parquet
  -> ParquetStreamDataset：扫描、worker 分片、可选 shuffle、组 batch
  -> TensorFormatter：按字段 formatter 转 tensor、截断和 padding
  -> Interaction：字段名 -> batch tensor
  -> OmniEmbLayer：按字段/source/序列组取 embedding
  -> 模型 calculate_loss() / predict()
  -> Trainer：更新参数、评估、早停、checkpoint
```

原始数据 join、过滤、切分规则和特征选择由实验脚本明确决定。`EmbSetting` 子类定义每个字段的 fit、transform、tensor formatter 和 embedding 取值方式；`SchemaManager` 负责组织这些规则。

`fit()` 发现 `schema_dir/feature_meta.json` 已存在时直接加载。schema 保存各 setting 状态，词表另存到相对路径 `vocab/*.parquet`。`fields()` 汇总 setting 输出列，以及时间、label、domain 字段；dataset 还支持通过 `extra_col_names` / `extra_col_formatters` 保留额外列。

已编码 parquet 走 `ParquetStreamDataset`；原始 parquet 需要边读边编码时走 `RawParquetStreamDataset`。两者已经产出 batch，外层 `DataLoader` 使用 `batch_size=None`。验证、测试通常设置 `shuffle=False, drop_last=False`，避免丢失尾批。

profile 路径由 `ProfileDatasetBundle` 读取 manifest、schema 和 split。`ProfileParquetStreamDataset` 通过 `ProfileSequenceStore` 按 owner 和 cursor 取历史前缀，再按时间规则过滤、按 max_len 截断，最后格式化为 `Interaction`。这条路径复用侧表中的历史，避免每行重复存储整段历史。

## batch 之后怎么走

`BaseModel` 持有 manager、`OmniEmbLayer`、`LossPort` 和 `ObservationPort`。`OmniEmbLayer.whole` 汇合普通用户/物品/交互特征；序列通过 `seq_groups` 的视图单独读取。`SharedBottomModel` 是包内附带的具体例子：embedding -> backbone -> 按 domain 选择 tower -> logits，训练使用 BCEWithLogits。

`BaseTrainer` 的默认 step 是 `zero_grad -> calculate_loss -> lsp.settle -> backward -> clip_grad_norm -> optimizer.step`。模型实现 `CustomTrainStepProtocol.custom_train_step(batch, ctx)` 时接管该 step 的优化过程，并负责结算 loss、返回标量 Tensor。epoch/eval 周期方法由 `BaseTrainerWithLife` 解释；EMA 和两阶段运行也有独立实现。

默认评估从 batch 取 manager 的 uid 和首个 label，调用 `model.predict()`，以 `batch_preds_1d` 交给 `EvaluatorManager`；它按全量/domain 等过滤器分发到 evaluator，最后汇总、清空。`predict()` 的分数语义由模型决定。已有 Top-K evaluator 接收二维物品得分和目标物品索引，但默认 Trainer 不会自动调用 `full_sort_predict()` 或切换为 Top-K 收集。

checkpoint 保存模型、主 optimizer、epoch、global_step，以及模型通过 `TrainerStateProtocol` 提供的额外训练状态。`run()` 根据验证指标执行早停和 best 保存；不会自动重载 best 后测试，需要入口显式安排。

## 源码阅读顺序

下列路径均为仓库相对路径，可在导出文件中搜索对应的 `## File:`：

1. `src/rosen/emb/manager.py`、`emb/schema/*.py`、`emb/split.py`：字段规则、编码、切分、持久化。
2. `src/rosen/data/dataset.py`、`data/padding.py`、`data/profile.py`、`core/interaction.py`：parquet 到 batch，以及 profile 历史装配。
3. `src/rosen/emb/emblayer.py`、`models/base.py`、`models/msr/sharedbottom.py`：batch 到 embedding 和模型输出。
4. `src/rosen/core/trainers/base.py`、`core/train/{context,hooks,loss}.py`：优化步骤和扩展契约。
5. `src/rosen/evaluate/{manager,evaluator,metrics}.py`：分组汇总与指标口径。
6. `src/rosen/experiment/{config,param}.py`：工作目录、配置和 CLI 参数。

模型只附 SharedBottom 实现及其直接组件、MSR 基类和注册入口，没有打包整个模型库；观测只附 BaseModel 使用的 port。其他模型、观测历史存储、实验脚本和数据不在本包内。`src/rosen/models/rerank/` 在生成时为空，因此没有该目录的源码条目。

## 重新生成

在仓库根目录运行（本次使用 Repomix 1.18.1）：

```bash
repomix --config docs/rosen/repomix/repomix.config.json
```

选取范围以 [`repomix.config.json`](repomix.config.json) 为准。Repomix 使用 Markdown 代码围栏包装完整文件，可能规范化文件首尾空白；文件内部的注释、空行和代码保持原样。
