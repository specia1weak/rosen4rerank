# 训练流

这篇文档解释 `rosen` 里“batch 进入模型以后”会发生什么。

重点是默认训练路径，而不是所有可扩展点的穷举。

## 1. 总链路

```text
Interaction batch
    -> BaseTrainer.train_step()
       -> _default_train_step() or model.custom_train_step(...)
       -> scalar Tensor for backward/logging
       -> model.lsp summary (detached named terms)
    -> model.predict(...) during eval
    -> evaluator.collect(...)
    -> evaluator.summary(...)
    -> EarlyStopper.step(...)
    -> optional checkpoint save
```

## 2. `BaseTrainer` 的默认训练循环

`trainer.run()` 当前大致会做：

1. `train_epoch()`
2. `evaluate_epoch()`
3. 如果配置了 `EarlyStopper`，根据验证指标判断：
   - 是否刷新 best
   - 是否保存 checkpoint
   - 是否提前停止

这条路径在 [`src/rosen/core/trainers/base.py`](../../src/rosen/core/trainers/base.py) 里是显式可见的。

`rosen.core.train` 只提供 context、protocol、早停和 EMA 控制器等基础组件；
具体 Trainer 从 `rosen.core.trainers` 导入，两者不互相包装导出。

`BaseTrainer` 不解释 `TrainingCycleProtocol`，也不接受 HealthMonitor。需要 epoch/eval 周期函数或健康快照时，
显式使用 `BaseTrainerWithLife`；需要 EMA 时使用其子类 `EMATrainer`。

## 3. 训练阶段默认走 `calculate_loss()`

默认训练步是：

```python
with model.lsp.scope():  # BaseTrainer 自动提供默认 main scope
    raw_loss = model.calculate_loss(batch)
    loss = model.lsp.settle(raw_loss)
loss.backward()
clip_grad_norm_(...)
optimizer.step()
return loss
```

这意味着：

- 如果你的模型没有特殊训练逻辑，最小实现就是写好 `calculate_loss()`
- 如果你的模型在同一 batch 内需要不同 backward/optimizer 流程，可以实现 `custom_train_step`
- 如果模型需要 epoch/eval 边界行为，显式实现 `TrainingCycleProtocol`
- 如果完全不需要特殊训练，不继承任何训练 Protocol

### Loss 分项

`BaseModel` 自带 `self.lsp`。任意嵌套方法都可以在当前 step scope 内登记 Loss：

```python
task = F.binary_cross_entropy_with_logits(logits, labels)
aux = self.encoder.aux_loss()
if self.lsp.active:
    self.lsp.add("task", task)
    self.lsp.add("regularization.encoder", aux, weight=0.1)
return task + 0.1 * aux
```

`BaseTrainer.train_step()` 对 default/custom 两条路径都自动打开默认 `main` scope。default step 由 Trainer 调
`model.lsp.settle(total_loss)`；custom step 或训练周期内的额外 optimizer 由模型自己结算。`settle()` 返回
仍可 backward 的标量 Tensor，同时把当前 scope 的命名项保存成 detached summary 并清空带图 term。
`LossPort` 不做跨 step 累计，也不保存历史；`start_step()` 会丢弃上一步 summary。

读取当前 step 的详细组成：

```python
summary = model.lsp.summary(clear=False)
# summary["settlements"][scope]["terms"][name]
# -> {"raw": ..., "weight": ..., "weighted": ...}
```

`calculate_loss()` 只负责登记分项并返回 raw total，不负责 settle。深层模块只调用 `add()`；真正拥有一次
backward/optimizer transaction 的代码块同时拥有 settle 职责。

## 4. 评估阶段默认走 `predict()`

默认评估步会调用：

```python
scores = self.predict_step(batch_interaction)
```

而 `predict_step()` 默认又只是：

```python
return self.model.predict(batch_interaction)
```

所以评估语义最终由 `model.predict()` 决定。

## 5. 这为什么重要

因为 `BaseTrainer` 不会替你区分：

- 训练时用 logits
- 评估时用概率

它只会把 `predict()` 的输出直接交给 evaluator。

因此你要自己保证：

- 只算 AUC 时，raw logits 通常没问题
- 需要 `logloss` 等概率语义时，`predict()` 最好返回概率
- 如果需要更特殊的评估行为，可以覆写 `predict_step()` 或整个 `evaluate_epoch()`

## 6. `EarlyStopper` 实际怎么工作

当前 `EarlyStopper` 会从 evaluator summary 里自动挑一个指标：

- 优先挑 `overall` / `all`
- 再按偏好的 metric 名顺序找，例如 `auc`、`gauc`、`ndcg@10`、`logloss`

它不是“你必须手动指定一个唯一 early stop 指标”的设计。

因此多人协作时最好约定：

- `overall` evaluator 要始终存在
- 主指标名字尽量稳定

这样 early stop 行为才可预测。

## 7. checkpoint 什么时候会保存

`BaseTrainer` 只有在两个条件同时满足时才会保存 checkpoint：

1. `cfg.ckpt_dir` 非空
2. `EarlyStopper` 认为当前是 best

如果 `ckpt_dir == ""`，训练会跑，但不会写 checkpoint。

## 8. Protocol、custom step 和可扩展点

当前常见扩展点有两类：

### 训练步扩展

- `CustomTrainStepProtocol`

适合：

- 多优化器
- 额外损失项调度

custom step 必须自行结算 LossPort，并直接返回标量 Tensor。单目标 custom step 直接使用 Trainer 已打开的
`main` scope，不需要再写 `with self.lsp.scope("main")`。只有同一个 custom step 内存在多个独立 backward
objective 时才使用额外命名 scope：

```python
main_loss = self.lsp.settle(main_total)
main_loss.backward()
main_optimizer.step()

with self.lsp.scope("adversarial"):
    self.lsp.add("task", adversarial_raw)
    adversarial_loss = self.lsp.settle(adversarial_raw)
adversarial_loss.backward()
adversarial_optimizer.step()
```

多个 LossTerm、多个参数组、梯度累计或多个 optimizer 共用一次 backward 都仍是一个 `main` settlement；
scope 只区分独立 backward objective。Trainer 只在整个 custom step 完成后按最终模型状态通知 `EMATrainer`
更新一次 EMA，不解释 optimizer 的名字或数量。

### 训练周期协议

- `TrainingCycleProtocol`

适合：

- 训练 epoch 开始/结束时做统计
- 评估结束后记录额外状态
- epoch-end 辅助 optimizer

该协议只由 `BaseTrainerWithLife` 解释，普通 `BaseTrainer` 不会调用。周期函数通常返回 `None`；如果额外
修改了 `model.state_dict()` 中的状态，直接返回准确的 `tuple[str, ...]`。M3oE 的 epoch-end architecture
更新因此直接返回架构参数名，EMA 子类只同步这些 state。Loss 通过 LossPort 记录；模型健康信息由
HealthMonitor 模块 probe 或 `custom_models` 中的 ObservationPort 信号提供，不再包装成通用 Result。需要保存第二 optimizer 或 phase 状态时，模型额外实现
`TrainerStateProtocol`。

### 两阶段训练

- `TwoStageTrainingProtocol`
- `TwoStageRunner`

只有确实需要“先搜索、再用新模型重训”的第一阶段模型实现 Protocol：

```python
class SearchModel(BaseModel):
    def should_finish_stage_one(self, report: StageReport) -> bool:
        return report.epoch >= self.search_epochs

    def initialize_stage_two(self, snapshot: StageSnapshot) -> BaseModel:
        model = RetrainModel(...)
        model.load_state_dict(snapshot.model_state)
        model.prepare_retrain()
        return model
```

实验提供只接收模型的 factory：

```python
def trainer_factory(model, stage: TwoStagePhase):
    return EMATrainer(
        model,
        build_optimizer(model),
        manager,
        build_loaders(stage),
        build_fresh_components(),
        build_fresh_run_config(stage),
        ema_config=ema_config,
    )

result = TwoStageRunner(trainer_factory).run(search_model)
```

factory 会分别收到 `TwoStagePhase.STAGE_ONE` 和 `STAGE_TWO`；这是供实验隔离 checkpoint/log 路径的类型化
标识，模型不需要记事件字符串。第二阶段必须是新的模型、Trainer、optimizer、EarlyStopper、HealthMonitor
和 EMA controller；dataset、manager 与只读配置可以复用。`BaseTrainer` 只知道通用 `stop_condition`，不知道
search/retrain 规则。两阶段模型不会通过 eval 周期方法就地改变正在训练的模型。若启用 checkpoint，两个 cfg
必须指向不同目录，否则重新开始的 epoch/step 计数会产生同名文件，Runner 会直接拒绝运行第二阶段。可重复
迭代的 dataset/loader 可以复用；generator 等一次性 iterator 必须在 factory 中重新创建。

## 9. 健康快照如何接入

```python
monitor = HealthMonitor(HealthMonitorConfig(snapshot_every=100, summary_capacity=1000))
monitor.observe_module("head", prefix="domain_tower", domain_routed=True)

components = TrainerComponents(
    evaluator_manager=evaluator,
    health_monitor=monitor,
)

trainer = BaseTrainerWithLife(..., components=components, ...)
```

采样 step 中，Trainer 会让一个 capture scope 包住完整 `train_step()`，然后同时读取 `model.obp.summary()`
与 `model.lsp.summary()`。模块 probe 输出、`custom_models` 显式声明的 OBP 信号和同一步 Loss 分项组成一个 point
`ModelSnapshot`。默认 `snapshot_every=100` 且不额外捕获第一步，因此 step 1～99 不形成历史，step 100 才
保存一次对齐快照。非采样 step 不做隐式 Loss 区间聚合。`SnapshotStore` 按完整快照淘汰，默认容量 1000
表示最近 1000 个采样点，而不是连续 batch 数。

## 10. 一张职责表

| 组件 | 默认职责 | 最常见扩展理由 |
|---|---|---|
| `BaseTrainer` | 最小 train/eval、custom step、early stop | 普通训练与调试 |
| `BaseTrainerWithLife` | 解释训练周期协议、接入健康快照 | epoch/eval 行为、观测 |
| `HealthMonitor` | 被动采样、快照、历史、导出 | 调试模型健康，不改变训练 |
| `CustomTrainStepProtocol` | 模型私有的特殊 batch 更新 | 多 backward、多 optimizer |
| `TrainingCycleProtocol` | 模型私有的 epoch/eval 边界行为 | 异步 optimizer、状态汇报 |
| `TwoStageTrainingProtocol` | 第一阶段终止与第二阶段新模型构造 | search 后独立 retrain |
| `TwoStageRunner` | 编排两个彼此独立的 Trainer run | 两阶段实验入口 |
| `LossPort` | 当前 step 的命名 Loss 账本 | 追踪辅助 Loss 与动态 weight |
| `ObservationPort` | 当前采样 step 的中间变量 trace | custom model 声明语义健康信号 |
| `SnapshotStore` | 保留完整、对齐的历史快照 | latest/history 查询与导出 |
| `model.calculate_loss()` | 定义训练目标 | 普通模型训练 |
| `model.predict()` | 定义评估输出 | 评估语义控制 |
| `EvaluatorManager` | 汇总一个或多个 evaluator | 多 domain、多口径评估 |
| `EarlyStopper` | 自动挑指标并维护 patience | 稳定提前停止 |

## 11. 最常见的五个坑

### 坑一：`predict()` 返回值和 evaluator 预期不一致

这是训练能跑、指标却怪的最常见原因之一。

### 坑二：以为 `BaseTrainer` 会自动保存 checkpoint

不会。`ckpt_dir` 为空时直接跳过。

### 坑三：以为 early stop 指标是固定写死的

不是。它会按 evaluator 和 metric 偏好自动挑。

### 坑四：训练逻辑太特殊，但还硬塞进 `calculate_loss()`

如果同一 batch 需要多次反向或多个 optimizer，使用 `custom_train_step`；只有确实需要 epoch/eval 边界行为的
模型才实现 `TrainingCycleProtocol`。

### 坑五：在模型里保存跨 step 的 Loss Tensor 或中间 Tensor

这会无意保留计算图。LossPort 只活一个 step；长期历史只保存 detached 的轻量快照指标。

## 12. 继续阅读

- 训练模块细节：[`details/CORE.md`](details/CORE.md)
- 评估模块细节：[`details/EVALUATE.md`](details/EVALUATE.md)
- 整体链路：[`ARCHITECTURE.md`](ARCHITECTURE.md)
- 健康快照完整设计：[`OBSERVATORY_HEALTH_SNAPSHOT_DESIGN.md`](OBSERVATORY_HEALTH_SNAPSHOT_DESIGN.md)
