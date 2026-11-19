# core/

`core/` 提供的是最底层但最常用的两个东西：

1. 推荐训练里通用的数据容器 `Interaction`
2. 训练循环相关的 `BaseTrainer` / `TrainContext` / `EarlyStopper`

## 1. 枚举

源码在 [`src/rosen/core/enum_type.py`](../../../src/rosen/core/enum_type.py)。

### `ModelType`

- `GENERAL`
- `SEQUENTIAL`
- `CONTEXT`
- `KNOWLEDGE`
- `TRADITIONAL`
- `DECISIONTREE`

### `KGDataLoaderState`

- `RSKG`
- `RS`
- `KG`

### `EvaluatorType`

- `RANKING`
- `VALUE`

### `InputType`

- `POINTWISE`
- `PAIRWISE`
- `LISTWISE`

### `FeatureType`

- `TOKEN`
- `FLOAT`
- `TOKEN_SEQ`
- `FLOAT_SEQ`

### `FeatureSource`

- `UNKNOWN`
- `INTERACTION`
- `USER`
- `ITEM`
- `USER_ID`
- `ITEM_ID`
- `SEQ`
- `SEQ_GROUP`
- `KG`
- `NET`

## 2. `Interaction`

源码在 [`src/rosen/core/interaction.py`](../../../src/rosen/core/interaction.py)。

`Interaction` 是一个 batch 级别的 tensor 容器，本质上可以把它理解成“能搬到 GPU 上的字典”。

```python
from rosen.core.interaction import Interaction

batch = Interaction({
    "user_id": torch.tensor([1, 2]),
    "item_id": torch.tensor([10, 11]),
    "label": torch.tensor([1, 0]),
})
```

### 常用方法

- `to(device, selected_field=None)`
- `cpu()`
- `numpy()`
- `repeat(sizes)`
- `repeat_interleave(repeats, dim=0)`
- `update(new_inter)`
- `drop(column)`
- `shuffle()`
- `sort(by, ascending=True)`
- `add_prefix(prefix)`

### 行为特点

- `Interaction["field"]` 返回单列 tensor。
- `Interaction[index]` 返回裁切后的新 `Interaction`。
- `Interaction.to(...)` 不会原地修改，而是返回新对象。
- `shuffle()` / `sort()` 会原地改内部顺序。

### `cat_interactions(interactions)`

把多个 `Interaction` 按 batch 维拼起来。前提是字段集合完全一致。

## 3. 训练上下文

源码在 [`src/rosen/core/train/context.py`](../../../src/rosen/core/train/context.py)。

### `TrainContext`

```python
@dataclass
class TrainContext:
    epoch: int
    global_step: int
    batch_idx: int
    optimizer: torch.optim.Optimizer
    manager: Any
    cfg: Any
    timer: Any
    recorder: Any = None
    stage: str = "train"
    phase: str = "default"
    evaluation_feedback: Optional[EvaluationFeedback] = None
```

它主要用于把训练状态和实验上下文传给自定义 step / 训练周期方法。不提供模型私有
`kwargs` 通道；新的通用上下文应当增加明确字段，模型私有需求由 Protocol 或 custom step 表达。

### `TrainerDataLoaders`

```python
@dataclass
class TrainerDataLoaders:
    train: Iterable[Interaction]
    valid: Optional[Iterable[Interaction]]
    test: Optional[Iterable[Interaction]] = None
```

### `TrainerComponents`

```python
@dataclass
class TrainerComponents:
    evaluator_manager: Any = None
    recorder: Any = None
    timer: Any = None
    early_stopper: Optional[EarlyStopper] = None
    health_monitor: Any = None
```

组件全部是显式注入；`early_stopper` 只接受正式 `EarlyStopper` 或 `None`，不会通过 `getattr` 静默兼容
不完整对象。`core.train` 不导入具体 Observatory、Evaluator 或 Trainer 实现。独立的第二阶段通过 factory
得到新的 `EarlyStopper`，不在原对象上调用 reset。

`health_monitor` 只由 `BaseTrainerWithLife` 及其子类使用。传给普通 `BaseTrainer` 会直接报错，避免用户误以为
监控已经启用。

## 4. 训练协议

源码在 [`src/rosen/core/train/hooks.py`](../../../src/rosen/core/train/hooks.py)。

### `CustomTrainStepProtocol`

实现这个协议后，`BaseTrainer.train_epoch()` 会优先调用你的 `custom_train_step(batch, ctx)`。

### `TrainingCycleProtocol`

定义三个固定周期方法：

- `on_train_epoch_start(ctx)`
- `on_train_epoch_end(ctx)`
- `on_eval_epoch_end(metrics, ctx)`

普通模型不继承该协议。只有确实需要 epoch/eval 边界行为的模型才实现它；周期方法通常返回 `None`，发生额外
模型更新时直接返回准确的 `tuple[str, ...]` state names。协议只由 `BaseTrainerWithLife` 解释；普通
`BaseTrainer` 只支持 `CustomTrainStepProtocol`。模型侧训练状态通过独立的 `TrainerStateProtocol` 保存。

## 5. Loss 与训练返回

### `LossPort`

`BaseModel.lsp` 是一个单 step、多 objective scope 的账本。模型可从任意成员方法调用
`add(name, tensor, weight=...)`。`settle(total_loss)` 返回带计算图的标量 Tensor，供拥有本次 optimizer
transaction 的代码 backward；同时保存 detached 的命名项摘要，并清空当前 scope 中带图的 term。

```python
raw_loss = model.calculate_loss(batch)
loss = model.lsp.settle(raw_loss)  # 默认 main scope 由 BaseTrainer 提供
loss.backward()
```

`lsp.summary(clear=False)` 返回当前 logical step 中全部 settlement 的详细字典；`clear_summary()` 只清摘要。
`start_step(...)` 同时设置 step/epoch/batch/stage 上下文并丢弃上一步摘要。LossPort 不负责历史、区间统计或
采样周期。一个 scope 正常退出时不能留下未结算 term，否则立即报错。

default step 由 Trainer settle；custom step 和训练周期中的额外 optimizer 由模型 settle。普通 custom step
直接使用默认 `main`，只有 GAN 等独立 backward objective 才显式 `with lsp.scope(name)`。context manager 是
唯一 scope 切换入口，不提供容易遗漏恢复的 `switch_scope()/set_scope()`。custom step 必须返回标量 Tensor，
Base 不接受 Python float，也不会二次结算模型返回值。

训练周期不定义 Result 包装对象。M3oE 使用 custom step 与 epoch-end 周期方法；后者直接返回被 architecture
optimizer 修改的 state names。Loss/调试信号经由模型自己的 LossPort/ObservationPort 进入 HealthMonitor，
不经过 Trainer 控制返回值。MultiEmb/SSIM 使用独立的两阶段 Protocol。

## 6. 两阶段训练 Protocol

源码分别位于：

- [`src/rosen/core/train/two_stage.py`](../../../src/rosen/core/train/two_stage.py)
- [`src/rosen/core/trainers/two_stage.py`](../../../src/rosen/core/trainers/two_stage.py)

`TwoStageTrainingProtocol` 只要求第一阶段模型实现两个方法：

```python
def should_finish_stage_one(report: StageReport) -> bool: ...
def initialize_stage_two(snapshot: StageSnapshot) -> BaseModel: ...
```

`StageReport` 提供 epoch、global step 与类型化 evaluation targets；`StageSnapshot` 是第一阶段终止时刻的
detached CPU model state。`initialize_stage_two()` 必须返回不同的模型对象，模型自行决定加载终止状态还是
内部保存的 best state，以及哪些参数 reset/freeze。

`TwoStageRunner` 用 `TwoStagePhase.STAGE_ONE/STAGE_TWO` 调用同一个 `trainer_factory(model, stage)` 两次。
阶段枚举只供实验隔离输出路径，模型不处理事件字符串。Runner 不 `deepcopy(Trainer)`，并拒绝复用 Trainer、
optimizer、EarlyStopper、HealthMonitor、Evaluator、timer、recorder 或 EMA controller，也会检查第二阶段
optimizer 没有引用第一阶段模型参数。manager、loader 和只读 cfg 值可以复用；如果启用 checkpoint，两个
Trainer 必须使用不同目录，避免重启后的 epoch/step 文件名碰撞。Runner 也会拒绝两个阶段复用同一个
generator/iterator；可重复迭代的 dataset/loader 才能共享。

## 7. `EarlyStopper`

源码在 [`src/rosen/core/train/early_stopper.py`](../../../src/rosen/core/train/early_stopper.py)。

```python
EarlyStopper(patience=5, min_delta=0.0)
```

### 选择指标的规则

`step(summary_dict, epoch=None)` 会从 `summary_dict` 里挑一个指标做 early-stop 依据：

1. 优先看 `overall` / `all` 这类评估器名。
2. 再按指标名优先级搜索：
   - `auc`
   - `gauc`
   - `ndcg@10`
   - `ndcg@20`
   - `ndcg`
   - `hr@10`
   - `hr@20`
   - `hr`
   - `recall@10`
   - `recall@20`
   - `recall`
   - `logloss`
   - `loss`
3. 找不到时退化为该评估器里的第一个指标。

注意：这里的 `recall@k` 只是 `EarlyStopper` 的候选指标名之一，并不代表当前 `evaluate.metrics` 已经实现了 `recall`。

### 模式

- `loss` / `logloss` / `mae` / `mse` / `rmse` 使用 `min` 模式。
- 其他指标使用 `max` 模式。

### 返回值

`step(...) -> (is_best, should_stop)`

`reset()` 用于 phase 语义断点后开启新的 patience 窗口。

## 8. `BaseTrainer`

源码在 [`src/rosen/core/trainers/base.py`](../../../src/rosen/core/trainers/base.py)。

`core/train` 存放 Trainer 依赖的基础组件，`core/trainers` 存放 `BaseTrainer`
及其通用派生类。基础包不会反向导入或重新导出 Trainer 实现。

### 初始化

```python
BaseTrainer(
    model,
    optimizer,
    manager,
    loaders,
    components,
    cfg,
)
```

### 默认流程

```text
run()
  -> train_epoch()
  -> evaluate_epoch()
  -> EarlyStopper.step(...)
  -> save_checkpoint(tag="best", metrics=...)
```

### 默认训练步

`train_step()` 是唯一包装点。如果模型没有实现 `custom_train_step`，就会走：

```python
with model.lsp.scope():  # main
    loss = model.lsp.settle(model.calculate_loss(batch))
    loss.backward()
clip_grad_norm_(...)
optimizer.step()
```

`_default_train_step()` 是内部执行路径，不再作为可绕过 `train_step()` 的第二个公共训练入口。

### 默认评估步

`evaluate_epoch()` 会：

1. 把 batch 移到 `cfg.device`
2. 取 `uids = batch[self.manager.uid_field]`
3. 取 `labels = batch[self.manager.label_field]`
4. 调 `scores = self.predict_step(batch)`，默认再委托给 `self.model.predict(batch)`
5. 调 `self.evaluator.collect(uids, labels, batch, batch_preds_1d=scores)`

这也是为什么 `predict()` 的语义必须和 evaluator 匹配。

### checkpoint

`save_checkpoint()` 只有在 `cfg.ckpt_dir` 非空时才会写文件。

保存字段包括：

- `epoch`
- `global_step`
- `model_state_dict`
- `optimizer_state_dict`
- `model_trainer_state_dict`
- `metrics`
- `experiment_name`
- `dataset_name`

`load_checkpoint()` 会恢复 model、main optimizer、trainer counter 和可选模型侧训练状态。
`BaseTrainerWithLife` 另外保存 HealthMonitor reducer state；`EMATrainer` 再保存全部 EMA teacher。

## 9. `HealthMonitor`

`HealthMonitor` 位于 `rosen.utils.observatory`，由 `BaseTrainerWithLife` 从 `TrainerComponents` 接入。它负责：

- 让采样 scope 包住 default/custom `train_step()`；
- 在同一采样 step 合并 `ObservationPort.summary()` 与 `LossPort.summary()`；
- 计算默认 basic/rank 摘要；
- 按完整 `ModelSnapshot` 保留最近 N 个历史点；
- 记录训练周期函数产生的 ObservationPort trace 和更新 state names；
- 输出 JSONL 和 CSV。

它不执行 optimizer step、freeze、phase transition 或 EMA 更新。

默认 `snapshot_every=100`、`capture_first_step=False`。Loss 与 Tensor 都是采样 step 的 point 信号；当前
实现不把最近 100 步 Loss 聚合后和第 100 步 Tensor 混在同一个快照。两个 Port 均不持有历史，历史只属于
`SnapshotStore`。

## 10. 实战注意事项

- `BaseTrainer` 不是 `TrainingTracker` 的封装层，两个系统彼此独立。
- 内置 MSR/unified 的 `logits()` 提供原始分，`predict()` 提供概率；LogLoss 使用概率，外部调用 predict 后不要重复 sigmoid。
- 默认 `train_epoch()` 会对 batch 做 `batch.to(cfg.device)`，所以 `Interaction` 必须是可搬运 tensor 容器。
