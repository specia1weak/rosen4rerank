# API 参考

这份索引只列当前源码里确实存在、并且在训练主链路里有意义的入口。更细的行为解释请跳转到 `details/` 文档或直接看源码。

## 先记住这些入口

| 主题 | 入口 |
|---|---|
| 特征编排 | `rosen.emb.SchemaManager` |
| 特征定义 | `rosen.emb.schema.*Setting` |
| 流式数据集 | `rosen.data.dataset.ParquetStreamDataset` |
| 统一 embedding | `rosen.emb.emblayer.OmniEmbLayer` |
| 多场景模型构建 | `rosen.models.msr.build_model` |
| 训练基础组件 | `rosen.core.train` |
| Trainer 实现 | `rosen.core.trainers.BaseTrainer` |
| 训练健康快照 | `rosen.utils.observatory.HealthMonitor` |
| 评估 | `rosen.evaluate.evaluator.PointwiseEvaluator` |
| 多评估器编排 | `rosen.evaluate.manager.EvaluatorManager` |
| 参数解析 | `rosen.experiment.param.ParamManager` |

当前评估主名称为 PointwiseEvaluator / RankingEvaluator / SlateValueEvaluator / RetrievalEvaluator。
每个接口只提供规范名，旧类名、参数别名和视图别名已删除；完整指标与候选聚合见 [评估契约](details/EVALUATE.md)。
辅助模型输入使用 rosen.emb.AuxField，由 SchemaManager(aux_fields=...) 自动读取并格式化，
不创建 embedding，详见 [Embedding 与辅助输入](details/EMB.md)。

## Core

### 枚举与交互

- [`core/enum_type.py`](../../src/rosen/core/enum_type.py)
  - `ModelType`
  - `KGDataLoaderState`
  - `EvaluatorType`
  - `InputType`
  - `FeatureType`
  - `FeatureSource`
- [`core/interaction.py`](../../src/rosen/core/interaction.py)
  - `Interaction`
  - `cat_interactions(interactions)`

### 预测契约

- [`core/prediction.py`](../../src/rosen/core/prediction.py)
  - `CandidatePool(scores, orders)`：浮点分数与 long 索引均为 `[B,M,K]`；`select(indices[B])` 返回每行的 `(order, scores)`。
  - `Prediction(scores=None, order=None, chosen_scores=None, candidates=None, full_scores=None, target=None, extra=...)`。
  - `get_chosen_scores(target=None)` 读取显式选中分数；`resolve_chosen_scores(target=None)` 优先取显式分数，直接打分模型按 order gather，候选池模型缺显式分数时报错。
  - `with_extra(updates=None, **kwargs)` 合并诊断字段，保留所有类型化字段；行过滤和 `to(device)` 同步处理候选池、选中分数和多任务 Mapping。

CandidatePool 同时从 `rosen.core` 和 `rosen.evaluate` 导出。

### 训练

- [`core/train/context.py`](../../src/rosen/core/train/context.py)
  - `EvaluationFeedback`
  - `TrainContext`
  - `TrainerDataLoaders`
  - `TrainerComponents`
- [`core/train/hooks.py`](../../src/rosen/core/train/hooks.py)
  - `CustomTrainStepProtocol`
  - `ObservationSourceProtocol`
  - `TrainingCycleProtocol`
  - `TrainingPhaseProtocol`
  - `TrainerStateProtocol`
- [`core/train/early_stopper.py`](../../src/rosen/core/train/early_stopper.py)
  - `EarlyStopper`
- [`core/train/loss.py`](../../src/rosen/core/train/loss.py)
  - `LossTerm`
  - `LossPort`
- [`core/train/ema.py`](../../src/rosen/core/train/ema.py)
  - `EMAConfig`
  - `EMAController`
- [`core/train/two_stage.py`](../../src/rosen/core/train/two_stage.py)
  - `StageReport`
  - `StageSnapshot`
  - `TwoStagePhase`
  - `TwoStageTrainingProtocol`
- [`core/trainers/base.py`](../../src/rosen/core/trainers/base.py)
  - `BaseTrainer`
  - `BaseTrainer.evaluate_test()` 使用 `TrainerDataLoaders.test`；注入 `TrainerComponents.timer` 时自动记录 train/valid/test epoch。
- [`core/trainers/with_life.py`](../../src/rosen/core/trainers/with_life.py)
  - `BaseTrainerWithLife`
- [`core/trainers/ema.py`](../../src/rosen/core/trainers/ema.py)
  - `EMATrainer`
- [`core/trainers/two_stage.py`](../../src/rosen/core/trainers/two_stage.py)
  - `TwoStageRunResult`
  - `TwoStageRunner`

### Observatory 健康快照

- [`utils/observatory/port.py`](../../src/rosen/utils/observatory/port.py)
  - `ObservationPort`
  - `ObservationSpec`
- [`utils/observatory/monitor.py`](../../src/rosen/utils/observatory/monitor.py)
  - `HealthMonitorConfig`
  - `HealthMonitor`
- [`utils/observatory/probe.py`](../../src/rosen/utils/observatory/probe.py)
  - `ModuleProbeSpec`
  - `ModuleProbeRegistry`
- [`utils/observatory/snapshot.py`](../../src/rosen/utils/observatory/snapshot.py)
  - `ModelSnapshot`
  - `SignalSnapshot`
- [`utils/observatory/store.py`](../../src/rosen/utils/observatory/store.py)
  - `SnapshotStore`

最小接入方式是把 `HealthMonitor` 注入 `TrainerComponents(health_monitor=...)`，并使用
`BaseTrainerWithLife` 或其子类；普通 `BaseTrainer` 会明确拒绝该组件。`rosen.models` 中的核心模型
只暴露 Port，不读取或写入 OBP。通用观测使用 `monitor.observe_module(...)` 注册无侵入模块 probe；必须声明
业务语义或局部中间量时，在 `custom_models` 的实验模型中调用 `self.obp.observe(...)`。

每个 `BaseModel` 固定拥有两个互不合并的当前 step 端口：

- `model.obp: ObservationPort`：一次被采样前向中的中间变量；
- `model.lsp: LossPort`：一次逻辑训练 step 中已经结算的命名 Loss。

两个端口都提供 `summary(clear=False)` 与 `clear_summary()`，但都不保存历史。`LossPort.settle()` 返回
仍带计算图的标量 `Tensor`，供训练代码 backward；详细的 detached Loss 字典通过 `lsp.summary()` 读取。
历史由 `HealthMonitor` 在采样 step 合并两个 summary 后写入 `SnapshotStore`。

`BaseTrainer.train_step()` 自动提供默认 `main` Loss scope，普通 default/custom 模型只需 `add()` 和
`settle()`；只有 GAN 等在同一 custom step 内执行多个独立 backward objective 的模型才显式使用
`with model.lsp.scope("generator"):` 一类命名 scope。

## Data

- [`data/dataset.py`](../../src/rosen/data/dataset.py)
  - `DataScanner`
  - `DataTransformer`
  - `ShuffleBuffer`
  - `PipelineStreamDataset`
  - `ParquetStreamDataset`
  - `RawParquetStreamDataset`
  - `ProfileSequenceStore`
  - `ProfileParquetStreamDataset`
- [`data/profile.py`](../../src/rosen/data/profile.py)
  - `ProfileDatasetLayout`
  - `ProfileDatasetBundle`
- [`data/padding.py`](../../src/rosen/data/padding.py)
  - `FeatureContext`
  - `TensorFormatter`
- [`core/formatters.py`](../../src/rosen/core/formatters.py)（原语规范定义；下列公共符号也从 data.padding 重导出）
  - `PaddingSide` / `TruncateMode`
  - `ColumnFormatter`
  - `DenseFormatter`
  - `VectorDenseFormatter`
  - `IntFormatter`
  - `FallbackFormatter`
  - `PaddedIntSequenceFormatter`
  - `PaddedFloatSequenceFormatter`
  - `PaddedNestedSequenceFormatter`

`FeatureContext.from_manager(manager, *, include_labels=False)` 用于已编码的无标签推理；
`from_raw_manager` 也支持该选项。只排除 schema 标签列，保留关联 mask 和所有 AuxField，
由 TensorFormatter 复用训练时的类型、padding 和长度规则。默认 `include_labels=True`。

## Embedding 与特征

### 特征 schema

- [`emb/schema/base.py`](../../src/rosen/emb/schema/base.py)
  - `EmbType`
  - `SeqGroupConfig`
  - `ProfileSeqGroupConfig`
  - `SlateLabel`（也可从 `rosen.emb` 导入）
  - `EmbSetting`

`SchemaManager(..., label_fields=...)` 接受字符串、`SlateLabel` 或两者的混合可迭代对象。
`SlateLabel(field_name, max_len, mask_field=None, padding_side="right", pad_val=0.0)`
声明列表监督；它不属于特征 setting 或序列组。`label_field` 仍返回第一个标签名称或 `None`，
`label_fields` 始终是去重保序的字符串元组。重复同名且规格不一致的声明会报错。
`get_label_spec(name)` 对列表标签返回规格，对标量及未知字段返回 `None`；`mask_fields`
返回去重保序的关联掩码字段。共享 mask 必须具有一致的 `max_len` 和 `padding_side`。

两条 Parquet dataset 链路自动读取这些 mask；标签和 mask 使用 float32 列表 formatter。
标签 padding 使用 `pad_val`，mask padding 为 0，左右 padding 按声明执行，超长列表沿用
现有 formatter 的取尾部规则。整行 null/空列表填充为 padding；不从标签值推导 mask。
`extra_col_formatters` 仍拥有最终覆盖权。监督声明由调用方构造 manager 时提供，
本次不改变 `feature_meta.json` 的格式或将监督规格写入该特征状态文件。

共享同一原子 setting 实例的多个序列（包括该原子也出现在顶层的情况），`fit()` 每个目标只
收集、解析一次统计量，两种内存模式均按 settings 顺序采用第一个来源。后续来源独有值
不参与词表或归一化统计，不进行自动合并；需要更完整词表时应预先拟合原子或调整来源顺序。
fit/save/load 会同步各序列包装器的 `is_fitted`，共享对象和 embedding 表关系保持不变。

运行时如只使用部分序列字段，调用方直接向 `SchemaManager` 传入对应的 `SequenceSetting`
子集；同组 setting 共享一个 `SeqGroupConfig`。`load_schema()` 只恢复当前 manager 注册的字段，
不要求运行时 settings 与数据处理阶段的完整 schema 集合一致。未注册字段不会进入 parquet
读取、formatter 或模型序列视图。`ProfileSeqGroupConfig.history_time_filter` 声明 profile 历史的
时间口径；`ProfileSequenceStore` 先在 cursor 前缀上应用该规则，再按 group 的 `max_len` /
`SequenceSetting.truncate_mode` 截断所有对齐字段，最后由 `TensorFormatter` 生成 `Interaction`。
- [`emb/schema/categorical.py`](../../src/rosen/emb/schema/categorical.py)
  - `SparseEmbSetting`
  - `MultiSparseSetting`
  - `QuantileEmbSetting`
- [`emb/schema/numerical.py`](../../src/rosen/emb/schema/numerical.py)
  - `MinMaxDenseSetting`
  - `Log1PDenseSetting(field_name, source)`：无需拟合的 `ln(1+x)`，输出 float32，null 补 0，支持序列元素。
  - `VectorDenseSetting`
- [`emb/schema/sequence.py`](../../src/rosen/emb/schema/sequence.py)
  - `SequenceSetting`

### 编排与 embedding

- [`emb/manager.py`](../../src/rosen/emb/manager.py)
  - `SchemaManager`
  - `fields(*, include_labels=True)`：编码后的输出列，去重保序。
  - `raw_fields(*, include_labels=True)`：转换前读入列，不含生成的序列长度列。
  - `get_formatters()`：汇总 setting、SlateLabel、AuxField 的转换规则。

原始流式输入由 `FeatureContext.from_raw_manager()` 使用 raw_fields 读取，转换后按 fields 输出。
序列 formatter 支持 `truncate_mode="head"/"tail"`；SequenceSetting 同步传给元素和时间列。
MultiSparseSetting 的标签固定保留头部，嵌套序列的内外截断方向独立。

- [`emb/emblayer.py`](../../src/rosen/emb/emblayer.py)
  - `RecEmbedding`
  - `BoleEmbLayer`
  - `SideEmb`
  - `EmbView`
  - `SeqGroupView`
  - `OmniEmbLayer`

`OmniEmbLayer` / `EmbView` / `SeqGroupView` 支持 `split_by="stack"`：把每个字段的
`[B, ..., D]` 表征沿倒数第二维堆叠为 `[B, ..., F, D]`。字段顺序沿用现有 source / setting
遍历顺序，与同一视图默认拼接输出的顺序一致；`include_fields` 用于筛选，不重新排序。
所有字段的实际 tensor shape 必须一致，否则抛出包含字段名及 shape 的 `ValueError`。
未显式 include 的序列字段和默认拼接路径一样被排除；序列组视图已显式选择本组字段。
空视图仍返回 `None`。

`EmbView.num_fields` 返回当前选择的字段数，`EmbView.field_dim` 返回字段共有的 embedding
维度；空视图或字段宽度不一致时访问 `field_dim` 会报错。原有 `embedding_dim` 仍表示
拼接后的总宽度，`field_segments` / `field_slices` 仍描述拼接布局。元数据不保证序列长度等
前导维一致，这些在 stack 前向时检查。旧式 `BoleEmbLayer` / `SideEmb` 不新增 stack 支持。

- [`emb/split.py`](../../src/rosen/emb/split.py)
  - `LooConfig`
  - `SequentialRatioConfig`
  - `TimeSplitConfig`
  - `RandomRatioConfig`
  - `build_split_config()`
  - `create_split_strategy()`

## Evaluate

- [`evaluate/metrics.py`](../../src/rosen/evaluate/metrics.py)
  - `AUC`
  - `LogLoss`
  - `GAUC`
  - `HR`
  - `NDCG`
  - `ListNDCG`（列表 relevance，注册键 `list_ndcg`；不改变原全库 `ndcg`）
  - `InSlateAUC`、`PCOC`、`RegressionAUC`、`SlateAUC`、`RMSE`、`MAE`、`NCIS`
  - `METRIC_REGISTRY`
- [`evaluate/evaluator.py`](../../src/rosen/evaluate/evaluator.py)
  - `BaseEvaluator`（抽象），以及 `PointwiseEvaluator`、`RetrievalEvaluator`、`RankingEvaluator`、`SlateValueEvaluator`
- [`evaluate/scorer.py`](../../src/rosen/evaluate/scorer.py)
  - `EvalContext(pred, inter=None, target=None)`：前置解析使用的原始来源
  - `BaseScorer`、`PredictionScorer`、`LabelScorer`、`FieldScorer`、`SidecarLookupScorer`、`JudgeModelScorer`
- [`evaluate/resolver.py`](../../src/rosen/evaluate/resolver.py)
  - `BatchResolver(pred_scorer=None, label_scorer=None, order_field="order", orders=None, ref_scorer=None, view="ranking", mask_field=None)`
  - `BatchResolver.for_candidates(label_scorer=None, ref_scorer=None, mask_field=None)`：使用 `candidates.scores` / `candidates.orders` 构造主候选整刷视图
  - `resolve(context, *, mask=None, users=None, to_cpu=...)`：通过显式搬运函数复用同一 collect 内的 CPU 副本
- [`evaluate/manager.py`](../../src/rosen/evaluate/manager.py)
  - `EvaluatorFilter`、`AllowAllFilter`、`DomainFilter`、`EvaluatorManager`

模型输出 `Prediction`；`EvaluatorManager.register(name, evaluator, filter_fn=None, *, target=None,
uid_field=None, resolver=None)` 显式组装来源与评价。行过滤后，Resolver 执行 Scorer 并输出纯 Tensor
EvalBatch(labels, scores, ref_scores, order, mask, users, full_scores)，不保留 pred/inter。
Manager 默认给 SlateValueEvaluator 配置 slate，PointwiseEvaluator / RetrievalEvaluator 配置 score_only，其他评价器配置 ranking；显式 Resolver 优先。
ranking 保留 N 候选和可选 order；score_only 忽略 order；slate 有 order 时将三类分数与 mask/users
投射至选中 K 位置，无 order 时保留原列表。原 N 在打分前取自原始来源或 Scorer.num_candidates。自定义 label_scorer 无需猜测 schema 的多个标签。
所有 Evaluator 只接收 EvalBatch；日志统一由 Manager 写入。旧 Evaluator/Order/Legacy/Decorator
类和 cb 指标已删除，调用方显式选类，校准指标统一使用 pcoc（0 分母为 NaN）。
详见 [预测、解析与评估](details/EVALUATE.md)。

BaseScorer 通过 `uses_schema_labels: bool` 声明 schema 标签依赖，`needs_target(pred)` 声明任务名依赖，
`uses_prediction_target(pred)` 声明是否消费绑定 Prediction.target 的 Tensor。BatchResolver 汇总三类来源的声明，
Manager 只使用 Resolver 接口。相同过滤器实例和原 Tensor 在一次 collect 内复用筛选及 CPU 搬运；
PointwiseEvaluator / SlateValueEvaluator 按 batch 保存独立 NumPy 数组，在 summary 时 concatenate，clear 释放缓存。

## Models

### 基础模型层

- [`models/base.py`](../../src/rosen/models/base.py)
  - `BaseModel`

`BaseModel` 提供 `LABEL`、`LABELS`、`DOMAIN`（来自 manager），以及
`get_label_mask(interaction, label_field=None, dtype=torch.float32)`。默认针对 `LABEL`；
若声明了 mask 且 batch 中存在该列则仅转换 dtype 并返回，否则生成与标签同形状、同设备的
全 1 张量。未配置任何标签且没有显式指定时抛出 `ValueError`。此方法不计算 loss、不推导
padding 有效位；未声明 mask 的列表标签需由模型明确决定哪些槽位参与监督。

- [`models/msr/base.py`](../../src/rosen/models/msr/base.py)
  - `MSRModel`：`logits(interaction)` 返回训练原始分；基类 `predict(interaction)` 返回 sigmoid 概率 Tensor，Trainer 统一包装 Prediction。
- [`models/msr/components/heads.py`](../../src/rosen/models/msr/components/heads.py)
  - `DomainTowerHead`

### 构建入口

列表重排使用独立入口 [`models/rerank`](../../src/rosen/models/rerank)：

可选 `dnn`、`dcn`、`prm`、`extr`、`kuaishou`、`pier`、`prs`、`nar`、`seq2slate`、
`pier_submodular`、`mge3`、`mge_cosine`、`lc_mge`、`mge_metric`、`mge_cosine_metric`，
以及保留旧默认配置的 `mge`。各自类、输入要求与参考源码对应见
[独立模型](details/RERANK_BASELINES.md)。

```python
from rosen.models.rerank import RerankModel, MultiGERerankModel, build_rerank_model
model = build_rerank_model(manager, "mge", **model_kwargs)
# LC 预设：离散 LC + 生成器候选分数 + 非零权重数归一化 BCE
lc_model = build_rerank_model(manager, "lc_mge", **model_kwargs)
```

`MultiGERerankModel` 通过 `submodularity_mode="none"/"cosine"/"lc"/"lc_soft"` 切换三生成器、
Cosine、离散 LC 与可微扩展；`candidate_dense_mode="prior"/"generator"` 选择生成候选进入
OCPM 的原始先验或生成器分数。私有参数由 `from_manager` 内省过滤，未知参数告警。
`RerankModel` 提供 `fold_history_pages`、`build_full_permutations`（K<=7）、
`select_by_order`、`masked_bce_with_logits`（mean/valid_mean/nonzero_mean），同名函数也可直接导入。
`SlateRerankModel.score_permutations(interaction, orders[B,M,K])` 是所有内置模型共用的裁判入口，
按 `candidate_chunk_size`（默认32）分块，用显式 mask 评分，输出无效槽为0。
`encode_slate()` 返回 `EncodedSlate(items,dense,mask,context)`；新模型实现
`_encode_items` / `_encode_context` / `_score_encoded`，logits 与 score_permutations 不允许子类覆写。
`OCPMRerankModel` 为 MultiGE/PIER 共享 OAM 历史页预编码，KuaiShou 预编码历史及 K/V。
`loss_components` 可审计各子损失，`calculate_loss` 接入 LossPort；`predict` 返回 Prediction，scores 为原列表概率、order 为最终排列，主候选分数与排列位于 candidates，显式选中分数位于 chosen_scores。extra 只保留诊断池和统计；outputs_candidate_pool 为主候选输出能力标记。具体输入、常量、复杂度及非等价边界见 [RERANK](details/RERANK.md)。

多场景入口继续为：

- [`models/msr/__init__.py`](../../src/rosen/models/msr/__init__.py)
  - `MODEL_REGISTRY`
  - `build_model(schema_manager, num_domains, model_cls, **model_kwargs)`

当前注册表包含这些名字：

```text
sharedbottom
sharedbottom_less
sharedbottom_plus
mmoe
ple
ple_v1
star
satrans
m3oe
m3oe_v1
m3oe_v2
m2m
ppnet
epnet
pepnet
crocodile
pareto
hierrec
automtl
riple
hamur
sdsp_ple
sdsp_mmoe
ssim
multiemb
```

### backbone 与模型工具

- 当前仓库已经没有单独的 `models/backbone/` 目录。常用共享底座与骨干实现主要位于 [`models/msr/`](../../src/rosen/models/msr) 及其子目录。
- [`models/utils/container.py`](../../src/rosen/models/utils/container.py)
  - `domain_select`
  - `MultiScenarioCloneBase`
  - `MultiScenarioContainer`
  - `MultiTaskContainer`
- [`models/utils/general.py`](../../src/rosen/models/utils/general.py)
  - `DNN`
  - `MLP`
  - `FeatureBifurcator`
  - `BifurcatedLinear`
  - `BifurcatedMLP`
  - `ModuleFactory`

## Experiment

- [`experiment/__init__.py`](../../src/rosen/experiment/__init__.py)
  - `ROOT_DIR`
  - `WORKSPACE`
  - `ignore_future_warning()`
  - `change_root_workdir()`
  - `set_all()`
- [`experiment/param.py`](../../src/rosen/experiment/param.py)
  - `ConfigBase`
  - `seed_everything()`
  - `ParamManager`
- [`experiment/engine.py`](../../src/rosen/experiment/engine.py)
  - `GridSearchEngine`
- [`experiment/tracker.py`](../../src/rosen/experiment/tracker.py)
  - `TrainingTracker`

## Datasets

- [`datasets/base.py`](../../src/rosen/datasets/base.py)
  - `DatasetBase`
- [`datasets/movielens.py`](../../src/rosen/datasets/movielens.py)
  - `MovieLensDataset`
- [`datasets/kuairand.py`](../../src/rosen/datasets/kuairand.py)
  - `KuaiRandDataset`
- [`datasets/aliccp.py`](../../src/rosen/datasets/aliccp.py)
  - `AliCCPDataset`
- [`datasets/douban.py`](../../src/rosen/datasets/douban.py)
  - `DoubanDataset`
- [`datasets/taac2026.py`](../../src/rosen/datasets/taac2026.py)
  - `TAAC2026Dataset`
- [`datasets/overview.py`](../../src/rosen/datasets/overview.py)
  - `get_general_info()`
  - `get_group_stats()`
  - `get_head_info()`

## Utils

注意：`rosen.utils` 顶层包只重导出了时间桶相关工具。其他工具需要从子模块导入。

- [`utils/optimize.py`](../../src/rosen/utils/optimize.py)
  - `split_params_by_decay()`
  - `create_complex_optimizer_groups()`
- [`utils/sequential.py`](../../src/rosen/utils/sequential.py)
  - `extract_history_sequences()`
  - `extract_seq_len()`
  - `extract_history_items()`
- [`utils/sample.py`](../../src/rosen/utils/sample.py)
  - `AbstractSampler`
  - `PolarsUISampler`
  - `extract_history_dict()`
- [`utils/time.py`](../../src/rosen/utils/time.py)
  - `timer`
  - `NamedTimer`
  - `CudaNamedTimer`
- [`utils/time_bucket.py`](../../src/rosen/utils/time_bucket.py)
  - `TIME_BUCKET_BOUNDARIES`
  - `get_num_time_buckets()`
  - `build_padding_mask()`
  - `bucketize_relative_time()`
  - `RelativeTimeEmbedding`
- [`utils/recorder.py`](../../src/rosen/utils/recorder.py)
  - `IndividualReLURecorder`
  - `ExplicitFeatureRecorder`
- [`utils/visualize.py`](../../src/rosen/utils/visualize.py)
  - `plot_bias_distributions()`
  - `plot_sparsity_distributions()`
  - `plot_sparsity_ecdf()`
  - `plot_power2_sparsity()`
- [`utils/process.py`](../../src/rosen/utils/process.py)
  - `set_priority()`
  - `get_cpu_load_rank()`
  - `set_affinity()`
  - `get_affinity()`
  - `get_idle_cpus()`
- [`utils/task_chain.py`](../../src/rosen/utils/task_chain.py)
  - `auto_queue()`

## 当前不建议照抄的历史名字

如果你在旧笔记、旧实验脚本里看到以下符号，请优先回到 `src/rosen` 确认：

- `SparseSetEmbSetting`
- `SparseSeqEmbSetting`
- `SharedVocabSeqSetting`
- `SeqGroupEmbSetting`
- `preset_workdir`

这些名字至少不是当前文档所依据的公共主路径。

### 可组合打分器与重排指标

Scorer 接收 EvalContext，构造参数仅出现在 BatchResolver；Evaluator 不持有或调用 Scorer。
RankingEvaluator 支持 auc/logloss/gauc/pcoc/in_slate_auc/ndcg@k，自动优先 batch.order，否则按 scores
稳定排序。ListNDCG 接收 relevance + topk_indices，IDCG 保留全部 N 候选；不再接收 scores/order 开关。
SlateValueEvaluator 支持 r_auc/slate_auc/pcoc/rmse/mae/ncis，以及 sum/mean/dcg；ncis 读取 ref_scores。
当前类型和调用示例见 [评估契约](details/EVALUATE.md)。
