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
| 评估 | `rosen.evaluate.evaluator.Evaluator` |
| 多评估器编排 | `rosen.evaluate.manager.EvaluatorManager` |
| 参数解析 | `rosen.experiment.param.ParamManager` |

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
  - `ColumnFormatter`
  - `DenseFormatter`
  - `VectorDenseFormatter`
  - `IntFormatter`
  - `FallbackFormatter`
  - `PaddedIntSequenceFormatter`
  - `PaddedFloatSequenceFormatter`
  - `PaddedNestedSequenceFormatter`
  - `FeatureContext`
  - `TensorFormatter`

## Embedding 与特征

### 特征 schema

- [`emb/schema/base.py`](../../src/rosen/emb/schema/base.py)
  - `EmbType`
  - `SeqGroupConfig`
  - `ProfileSeqGroupConfig`
  - `EmbSetting`

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
  - `VectorDenseSetting`
- [`emb/schema/sequence.py`](../../src/rosen/emb/schema/sequence.py)
  - `SequenceSetting`

### 编排与 embedding

- [`emb/manager.py`](../../src/rosen/emb/manager.py)
  - `SchemaManager`
- [`emb/emblayer.py`](../../src/rosen/emb/emblayer.py)
  - `RecEmbedding`
  - `BoleEmbLayer`
  - `SideEmb`
  - `EmbView`
  - `SeqGroupView`
  - `OmniEmbLayer`
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
  - `METRIC_REGISTRY`
- [`evaluate/evaluator.py`](../../src/rosen/evaluate/evaluator.py)
  - `PointWiseEvaluator`
  - `TopKEvaluator`
  - `Evaluator`
  - `EvaluatorDecorator`
  - `LogDecorator`
- [`evaluate/manager.py`](../../src/rosen/evaluate/manager.py)
  - `EvaluatorFilter`
  - `AllowAllFilter`
  - `DomainFilter`
  - `EvaluatorManager`

## Models

### 基础模型层

- [`models/base.py`](../../src/rosen/models/base.py)
  - `BaseModel`
- [`models/msr/base.py`](../../src/rosen/models/msr/base.py)
  - `MSRModel`
- [`models/msr/components/heads.py`](../../src/rosen/models/msr/components/heads.py)
  - `DomainTowerHead`

### 构建入口

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
