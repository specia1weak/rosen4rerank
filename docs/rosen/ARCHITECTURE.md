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

- [`core/enum_type.py`](../../src/rosen/core/enum_type.py)
  - `FeatureSource`、`InputType`、`EvaluatorType` 等基础枚举。
- [`core/interaction.py`](../../src/rosen/core/interaction.py)
  - batch 级别的 tensor 容器，训练和评估都围绕它传递。
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
- [`data/dataset.py`](../../src/rosen/data/dataset.py)
  - 流式扫描 parquet，按 worker 分片、shuffle、format，最终产出 `Interaction`。
- [`data/profile.py`](../../src/rosen/data/profile.py)
  - 把 prepared profile dataset 的逻辑 split 映射到 manifest、schema 和 profile store。
  - 训练入口只提供数据集根目录与 split 名称；特征 schema、序列裁剪和模型语义仍由实验脚本显式定义。

### L3: 评估

- [`evaluate/evaluator.py`](../../src/rosen/evaluate/evaluator.py)
  - 单个 evaluator，内部组合 point-wise / top-k evaluator。
- [`evaluate/manager.py`](../../src/rosen/evaluate/manager.py)
  - 多 evaluator 注册表，支持按 domain 过滤后分别汇总。

### L4: 模型

- [`models/base.py`](../../src/rosen/models/base.py)
  - `BaseModel` 自动挂上 `OmniEmbLayer`、单 step `LossPort` 与轻量 `ObservationPort`。
- [`models/msr/`](../../src/rosen/models/msr)
  - 所有多场景模型的实现和注册表。
- [`models/msr/`](../../src/rosen/models/msr)
  - 共享底座、MMoE、PLE、STAR 等主要多场景实现都在这里或其子目录里。
- [`models/msr/components/heads.py`](../../src/rosen/models/msr/components/heads.py)
  - `DomainTowerHead` 把共享表征映射到 domain-specific logit。

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
- `domain_fields`

统一加入输出列集合。后面的 dataset formatter 和 trainer 都默认按这个结构工作。

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
- 如果模型需要特殊评估逻辑，可以覆写 `predict_step()` 或整个 `evaluate_epoch()`。

## 哪些东西不要误以为是“框架约定”

- `TrainingTracker` 不是 trainer 默认依赖。
- `prepare_data()` 虽然存在，但源码注释已经明确“不建议使用”；推荐显式走 `split -> fit -> transform`。
- `@examples/ml-1m` 中的很多实验文件是历史草稿，不代表当前公共 API。
