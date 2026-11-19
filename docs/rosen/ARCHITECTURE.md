# 架构设计

当前仓库可以按 6 个层次理解，但更准确的说法不是“严格框架分层”，而是“按训练链路分组的模块集”。

## 分层视图

```text
L6  experiment/     参数解析、脚本调度、实验辅助
L5  core/train/ + core/trainers/  训练基础组件与 Trainer 实现
L4  models/         基础模型、多场景模型、head、模型工具
L3  evaluate/       指标计算、多评估器编排
L2  emb/ + data/    特征规则、编码、embedding、流式数据集
L1  core/ + utils/ + datasets/  基础枚举、Interaction、工具函数、数据路径封装
```

## 一条真实的执行链

```text
raw parquet / lazyframe
    -> SchemaManager.split_dataset(...)
    -> SchemaManager.fit(train_raw)
    -> SchemaManager.transform(...)
    -> save_as_dataset(...)
    -> ParquetStreamDataset / RawParquetStreamDataset
    -> Interaction
    -> OmniEmbLayer
    -> MSR model / custom BaseModel
    -> EvaluatorManager
    -> BaseTrainer
```

## 每层负责什么

### L1: 基础抽象

- [`core/formatters.py`](../../src/rosen/core/formatters.py)
  - 只依赖 NumPy/Torch 的列转换与 padding 原语，不导入 `emb` 或 `data`。
- [`core/enum_type.py`](../../src/rosen/core/enum_type.py)
  - `FeatureSource`、`InputType`、`EvaluatorType` 等基础枚举。
- [`core/interaction.py`](../../src/rosen/core/interaction.py)
  - batch 级别的 tensor 容器，训练和评估都围绕它传递。
- [`core/prediction.py`](../../src/rosen/core/prediction.py)
  - Prediction 区分原序 scores、最终 order、选中 chosen_scores 和类型化 CandidatePool；extra 仅承载诊断。
  - 切片、设备迁移和 with_extra 都保留类型化字段，resolve_chosen_scores 统一选中得分解析。
- [`utils/`](../../src/rosen/utils)
  - 负采样、时序序列构造、优化器分组、时间桶、可视化等杂项工具。
- [`datasets/`](../../src/rosen/datasets)
  - 若干本地数据集路径适配器，不负责自动下载。

### L2: 特征与数据流

- [`emb/schema/`](../../src/rosen/emb/schema)
  - 每个字段如何拟合、如何转换、如何转 tensor、如何前向取值，都由 `EmbSetting` 子类定义。
- [`emb/manager.py`](../../src/rosen/emb/manager.py)
  - 负责 `fit / transform / split / save / load` 的总调度。
- [`emb/emblayer.py`](../../src/rosen/emb/emblayer.py)
  - `OmniEmbLayer` 把当前 batch 的 `Interaction` 转成模型可用的 embedding。
  - 输出布局由 `split_by` 选择；`stack` 保留字段轴 `[B, ..., F, D]`，不解释重排、历史页面等任务语义。
    页面折叠、排列及监督仍由后续模型与实验代码决定。
- [`data/dataset.py`](../../src/rosen/data/dataset.py)
  - 流式扫描 parquet，按 worker 分片、shuffle、format，最终产出 `Interaction`。
- [`data/profile.py`](../../src/rosen/data/profile.py)
  - 把 prepared profile dataset 的逻辑 split 映射到 manifest、schema 和 profile store。
  - 训练入口只提供数据集根目录与 split 名称；特征 schema、序列裁剪和模型语义仍由实验脚本显式定义。

### L3: 评估

- [`evaluate/evaluator.py`](../../src/rosen/evaluate/evaluator.py)
  - 一个抽象基类与 Pointwise/Ranking/SlateValue/Retrieval 四个核心评价器，均消费纯张量，只提供规范名。
  - Pointwise 负责曝光指标；Ranking 负责请求内排序；SlateValue 负责整刷/候选价值与 OPE；Retrieval 负责全库召回。
- [`evaluate/manager.py`](../../src/rosen/evaluate/manager.py)
  - 多 evaluator 注册表，通过 target 同时选择预测与标签、由 schema 推导 mask，并按 domain 行过滤后分别汇总。
  - 行过滤后由 BatchResolver/Scorer 解析与投射，EvalBatch 只有 labels/scores/ref_scores/order/mask/users/full_scores 张量。
  - Scorer 通过 uses_schema_labels、needs_target(pred)、uses_prediction_target(pred) 声明依赖；Manager/Resolver 不探测具体 Scorer 类型或字段。
  - 同一次 collect 复用相同过滤器实例的筛选结果及相同 Tensor 的 CPU 副本；跨 batch 缓存为独立 NumPy 数组，summary 时拼接。

### L4: 模型

- [`models/base.py`](../../src/rosen/models/base.py)
  - `BaseModel` 自动挂上 `OmniEmbLayer`、单 step `LossPort` 与轻量 `ObservationPort`。
- [`models/msr/`](../../src/rosen/models/msr)
  - 所有多场景模型的实现和注册表。
- [`models/msr/`](../../src/rosen/models/msr)
  - 共享底座、MMoE、PLE、STAR 等主要多场景实现都在这里或其子目录里。
- [`models/msr/components/heads.py`](../../src/rosen/models/msr/components/heads.py)
  - `DomainTowerHead` 把共享表征映射到 domain-specific logit。
- [`models/rerank/`](../../src/rosen/models/rerank)
  - 独立的 `RerankModel`、排列算子与 MultiGE 模型族，共享 attention 和可配置生成器目标。
  - 模型 `predict()` 返回语义 Prediction；outputs_candidate_pool 声明主候选池能力，候选池和选中得分是类型化字段，不扩展 Trainer 协议。
    数据形状、梯度边界及参考实现差异见 [RERANK](details/RERANK.md)。
  - SlateRerankModel 统一 encode_slate → 分块候选 → _score_encoded，显式传入 mask；
    OCPMRerankModel 共享预编码历史页，KuaiShou 共享历史表示及 K/V，不逐候选构造 Interaction。

### L5: 训练

- [`core/train/`](../../src/rosen/core/train)
  - 定义 context、Loss、protocol、早停和 EMA 控制器，不导入 Trainer 实现。
- [`core/trainers/base.py`](../../src/rosen/core/trainers/base.py)
  - `BaseTrainer` 只定义默认训练/验证循环和 `custom_train_step` 支持，不解释训练周期或健康监控协议。
- [`core/trainers/with_life.py`](../../src/rosen/core/trainers/with_life.py)
  - `BaseTrainerWithLife` 解释固定的 epoch/eval 训练周期协议，并可接入 HealthMonitor。
- [`core/trainers/ema.py`](../../src/rosen/core/trainers/ema.py)
  - `EMATrainer(BaseTrainerWithLife)` 在正常 step 和周期额外更新后维护 EMA，并负责 EMA 评估/checkpoint。
- [`core/train/hooks.py`](../../src/rosen/core/train/hooks.py)
  - `CustomTrainStepProtocol` 由 Base 解释；`TrainingCycleProtocol` 只由 WithLife 解释。
- [`core/train/two_stage.py`](../../src/rosen/core/train/two_stage.py)
  - 两阶段模型的显式终止与新模型初始化协议；普通模型不实现。
- [`core/trainers/two_stage.py`](../../src/rosen/core/trainers/two_stage.py)
  - 用同一个 Trainer factory 依次创建两套独立训练运行，不复制正在运行的 Trainer。
- [`core/train/early_stopper.py`](../../src/rosen/core/train/early_stopper.py)
  - 自动挑指标、维护最佳值和 patience。
- [`utils/observatory/`](../../src/rosen/utils/observatory)
  - `HealthMonitor` 被 `BaseTrainerWithLife` 组合持有；按完整 `ModelSnapshot` 保存 Tensor、Loss 和 state。
  - `ObservationPort` 只生产一次 forward 的 trace，不负责采样周期、历史和输出。

训练相关依赖方向固定为：

```text
models -> core.train contracts
core.trainers -> core.train contracts
core.trainers -> optional HealthMonitor
core.train -/-> core.trainers
```

### L6: 参数和实验

- [`experiment/param.py`](../../src/rosen/experiment/param.py)
  - `ConfigBase`、`ParamManager`、`seed_everything`。
- [`experiment/engine.py`](../../src/rosen/experiment/engine.py)
  - `GridSearchEngine` 用 subprocess 启多个实验脚本。
- [`experiment/tracker.py`](../../src/rosen/experiment/tracker.py)
  - 一个独立的 checkpoint / vector 跟踪小工具，但当前 `BaseTrainer` 默认不会自动接入它。

## 三个关键契约

### 1. `SchemaManager` 产出的字段集合决定后续 batch 结构

`manager.fields()` 会把：

- 所有 `setting.get_output_field_names()`
- `time_field`
- `label_fields`
- `mask_fields`（来自 `SlateLabel` 的关联监督掩码）
- `domain_fields`
- `aux_fields`（无需拟合/嵌入的类型化模型输入及原始 ID）

统一加入输出列集合。后面的 dataset formatter 和 trainer 都默认按这个结构工作。
列表监督由 `SlateLabel` 声明，和特征序列组分开管理；它只决定标签及掩码如何进入 batch，
不规定 loss 或评估口径。共享原子 setting 的拟合以第一个注册来源为准，不自动并入其他序列。
训练 dataset 与独立推理均从 manager 构建 FeatureContext；推理显式使用
`include_labels=False` 排除标签并保留关联 mask，辅助输入不再另行声明名称或 formatter。

`manager.raw_fields()` 查询转换前所需的原始列，`manager.get_formatters()` 汇总各 setting、
SlateLabel 和 AuxField 的转换规则；`fields()` / `raw_fields()` 均支持 `include_labels=False`。
`FeatureContext.from_raw_manager()` 服务于 RawParquetStreamDataset 的在线 transform：
例如原始文件只有 history，history_len 在转换后才生成，不能按输出列集合提前读取。

格式化依赖方向固定为 `data/padding -> emb -> core/formatters`，data 也可直接使用原语。
各特征声明负责选择 formatter，TensorFormatter 只合并 manager 规则与显式覆盖，再执行转换。
特征 formatter 不再反向导入 data，也不需要函数内导入规避循环。
`SequenceSetting.truncate_mode` 同时控制序列元素与时间列；MultiSparse 内层标签始终保留头部，
与 Polars `list.head(max_tag_len)` 一致。详见 [数据契约](details/DATA.md)。

### 2. 模型前向依赖 `Interaction`

大多数 MSR 模型的最小路径是：

```python
x = self.omni_embedding.whole(interaction)
domain_ids = interaction[self.DOMAIN].long()
logits = self.backbone(x, domain_ids)
```

这也是为什么 domain 列即使不做 embedding，也必须保留在 batch 里。

### 3. 评估默认走 `model.predict()`

`BaseTrainer` 不区分“训练 logits”和“评估概率”，它只会调用 `predict()`。因此你需要自己保证：

- 评估指标和 `predict()` 输出语义匹配。
- 特殊来源由 BatchResolver 子类读取 Prediction.extra/Interaction；BaseEvaluator 只接收最终 Tensor，预测仍由模型负责。

内置 MSR/unified 的 `logits(interaction)` 用于训练，基类 `predict()` 返回 sigmoid 概率 Tensor。
其他模型的 `predict()` 返回 Tensor 或 Prediction。Trainer 在实际模型预测后统一包装裸 Tensor，默认
目标取该模型的 LABEL，覆盖 EMA teacher。Manager 自动绑定 schema，注册 target 与
Prediction.target 确定同名标签及预测、自动查询监督 mask；多个未指定目标的标签会报错。
所有 evaluator 统一收集 EvalBatch，特殊评估继承 BaseEvaluator 实现，不新增模型 eval step。

## 哪些东西不要误以为是“框架约定”

- `TrainingTracker` 不是 trainer 默认依赖。
- `prepare_data()` 虽然存在，但源码注释已经明确“不建议使用”；推荐显式走 `split -> fit -> transform`。
- `@examples/ml-1m` 中的很多实验文件是历史草稿，不代表当前公共 API。
