# Observatory 模型健康快照设计

> 状态：第一阶段已实现并完成 KuaiRand top5 EMA 实验验证；高级 profile/reducer/anomaly 仍待实现  
> 日期：2026-07-18  
> 适用范围：训练期模型观测、Loss 组成追踪、阶段性健康快照、异常诊断  
> 兼容策略：允许破坏当前 `TensorObservatory` 的使用方式，以新设计的职责清晰度为优先

当前代码已经落地：彼此独立的 `LossPort`/`ObservationPort`、训练 Protocol、
`ObservationPort.observe()`、`HealthMonitor`、`ModelSnapshot`、`SignalSnapshot`、`SnapshotStore`、
模块级 forward probe、domain-routed probe、JSONL/CSV sink、同一步 point Loss/Tensor 快照、EMA 模型更新接入、
MultiEmb/SSIM 两个独立训练阶段，以及 M3oE epoch-end 异步架构更新。项目已明确放弃独立生命周期对象；
本文后面的“建议”“阶段”仍保留为完整目标；
凡是超出上述清单的内容不能因为写在本文里就视为已经实现。

模型包边界已经进一步收紧：`src/rosen/models` 只保留纯模型计算、LossPort 与训练 Protocol，不读取或
写入 OBP。通用模型健康状态由 HealthMonitor module probe 获取；需要语义命名、梯度 tap、cut 或无法通过
模块边界取得的局部变量时，在 `src/custom_models` 中实现观测版模型。

当前模型端口固定为：

```text
BaseModel
├── obp: ObservationPort
└── lsp: LossPort
```

两个 Port 都只表达当前 step，都提供 `summary(clear=False)` 和 `clear_summary()`，但不合并、不内置 Collector、
不保存历史。默认 HealthMonitor 每 100 step 同时读取两者；历史只由 `SnapshotStore` 保存。

## 1. 背景

Rosen 目前已经具备三类相关能力：

- `ObservationPort`
  - custom model 可以显式注册和捕获中间 Tensor；核心 `rosen.models` 只持有端口。
  - 支持普通 capture、cut、watch 和 gradient tap。
- `TensorObservatory`
  - 把 Tensor 拉平后计算基础统计、谱统计和关系统计。
  - 按名称保存最近若干次 `record()` 的结果。
- 各模型自己的调试逻辑
  - 通过 `_latest_debug`、`monitor_step_interval`、`record_step_interval` 和手写 `print()` 输出 Loss、rank、gate、路由等状态。

这些能力可以回答局部问题，但还不能稳定表达下面这个更高层目标：

> 在一个确定的训练时刻，查看同一次模型计算中的所有关键中间变量、全部优化 Loss，以及它们相对于过去阶段的变化，从而判断模型是否健康、是否塌缩、是否出现数值异常，以及各项 Loss 权重是否合理。

当前系统的主要缺口不是指标数量不足，而是缺少按同一训练时刻对齐的“完整模型快照”数据概念。

## 2. 当前实现的事实与问题

### 2.1 当前保存的是每个 Tensor 的独立窗口

`TensorObservatory` 当前使用：

```python
features: dict[str, deque[dict]]
```

默认 `window_size=50`，表示每个 Tensor 名称各自保留最近 50 次 `record()`。

它不是：

```text
ModelSnapshot(step=200)
├── hidden
├── logits
├── task_loss
└── aux_loss
```

而是：

```text
hidden
├── step 200
├── step 400
└── step 600

logits
├── step 100
├── step 200
└── step 300
```

不同名称可以拥有不同的记录频率、不同的时间范围和不同的缺失情况，因此无法可靠还原某个 step 的完整模型状态。

### 2.2 “最近 50 次”不等于最近 50 个 batch

当前记录频率由模型或 example 自己决定：

- 一些旧模型每个训练 batch 都调用 `record()`。
- 新的 ObservationPort 模型通常每 200 step 才收集一次。
- 一些评估脚本每个 epoch 临时创建一个新的 Observatory，只聚合当前若干 probe batch。

因此 `window_size=50` 可能表示：

- 最近 50 个 batch；
- 最近 10000 个训练 step 内的 50 个采样点；
- 当前一次 probe 中的若干 batch；
- 或其他由调用方决定的口径。

当前系统没有统一区分：

- 采样间隔；
- 历史保留容量；
- 默认展示窗口；
- 输出间隔。

### 2.3 当前窗口聚合会隐藏最后一次异常

`get_window_stats()` 默认对当前保留窗口进行平均，而不是展示最后一次状态。

如果前 49 次 effective rank 约为 30，最后一次突然降为 1，窗口均值仍接近 29.4，模型的最新塌缩会被平均值掩盖。

因此新设计不应默认用历史均值替代最新状态。默认报告应优先展示：

- 当前值；
- 上一次值；
- 变化量；
- 短期趋势；
- 异常状态。

历史聚合应成为显式查询，而不是默认语义。

### 2.4 当前容量不能直接扩大到 1000

当前每次 `record()` 都会构造并保存 sketch，即使配置只有 `metrics=("basic",)`。

默认 sketch 最大为：

```text
256 rows × 256 dims × float32 ≈ 256 KB
```

仅计算 sketch，不计其他向量和 Python 对象开销：

```text
单 Tensor × 50 条记录   ≈ 12.5 MB
单 Tensor × 1000 条记录 ≈ 250 MB
10 个 Tensor × 1000 条  ≈ 2.5 GB
```

所以更长的历史必须保存轻量指标结果，而不能长期保存每次计算所使用的 sketch。

### 2.5 当前指标配置没有真正按需执行

代码中已经存在：

```text
metrics/basic.py
metrics/spectral.py
metrics/correlation.py
metrics/routing.py
metrics/vq.py
metrics/registry.py
```

但主 `TensorObservatory.record()` 没有通过这些模块形成真正的指标调度器。

当前只要 metrics 中出现 `spectral`、`correlation` 或 `cosine` 任意一个，就会调用旧的 `spectral_stats()`，一次性计算 rank、energy、dead dim、correlation 和 cosine 等整套指标。

结果是：

- `metrics=("spectral",)` 仍会计算 correlation 和 cosine；
- basic-only 仍会保存 sketch；
- 指标配置看似可选，实际成本和输出仍然成组绑定；
- `metrics/registry.py` 还只是名称集合，不是可执行的 registry。

### 2.6 ObservationSpec 的语义没有进入 Recorder

`ObservationSpec` 已经包含：

```text
role
axes
metrics
metadata
```

但当前 Connector 只把 `metrics` 转成 `TensorMonitorOptions`，没有把 role、axes 和 metadata 传给 Recorder。

因此 Recorder 无法根据语义自动选择合适指标：

- hidden 适合 rank；
- logits 适合数值范围和有限性检查；
- routing probability 适合 entropy 和 usage；
- Loss 适合 raw、weight 和 contribution；
- label/domain id 不应被当作普通连续表示计算 rank。

### 2.7 当前默认输出容易信息爆炸

当前默认会：

- 计算较多全局 scalar；
- 对每个 Tensor 显示 per-dim 或 top-dim 表；
- 按 Tensor 分段打印完整 Polars 表；
- 可选打印 Tensor pair 关系表。

当模型注册十几个中间变量时，用户需要在大量表格中手工寻找真正异常的信号。

新设计的默认输出必须是一张紧凑的模型健康总览。per-dim、relation、correlation、raw Tensor 等内容只在显式请求或异常诊断时启用。

## 3. 设计目标

新设计需要满足：

1. 一次快照同时包含模型中间 Tensor 和所有优化 Loss。
2. 同一快照中的所有信号共享 step、epoch、stage、target 等上下文。
3. Loss 和 Tensor 在存储、查询、历史和报告层是同等的一等信号。
4. 不要求 Loss 和 Tensor 使用完全相同的采样成本与聚合口径。
5. 默认提供有限性、均值方差、数值范围和适用时的 rank 健康信息。
6. 默认不保存大量 sketch、per-dim 向量或 raw Tensor。
7. 支持最近约 1000 个轻量快照的历史。
8. 支持初始、epoch end、train end 的固定 probe 快照。
9. 支持 NaN、Inf、异常突变触发的详细快照和短期 burst 调试。
10. 模型只声明“观测什么”和“有哪些 Loss”，不负责采样周期、历史、打印和文件输出。
11. 允许破坏旧 API，但必须覆盖旧 Observatory 的 capture、历史、关系分析、cut、watch、gradient tap、
    激活 hook、干预审计和模型自定义聚合能力。
12. 支持 search/retrain、warmup/search/finetune、交替优化等多阶段训练，快照必须带有明确 phase。
13. 模型私有训练行为由 opt-in Protocol 管理，模型无关能力由 Trainer 派生类管理；HealthMonitor 只观察。
14. 对需要连续统计但不需要 raw history 的信号，支持逐步在线归约并在 interval/epoch/phase/run 边界提交。

## 4. 明确不做的事情

新设计默认不做：

- 保存最近 50 个连续 batch 的完整 Tensor 快照；
- 保存所有 Python 局部变量；
- 长期保存每个快照的完整 raw Tensor；
- 对所有 Tensor 默认计算 correlation、cosine、CKA 和关系分析；
- 让模型自己判断第多少 step 应该输出报告；
- 让 LossPort 决定日志频率或历史保留策略；
- 通过数值范围猜测一个 Tensor 是 logits 还是 probability；
- 用一个隐含窗口平均值代表模型的最新健康状态。

如果需要连续 batch 诊断，应显式进入 burst 模式，而不是由正常训练持续承担该成本。

## 5. 核心概念

### 5.1 ObservationPort

custom model 中间 Tensor 的生产端；核心模型默认只由外部 module probe 观测。

custom model 通过统一入口声明一个信号：

```python
shared_z = self.observe(
    "shared_z",
    shared_z,
    role="hidden",
)
```

目标语义：

- 未进入 capture scope 时近似 no-op；
- 进入 capture scope 时保存 detached 的当前信号；
- 同时维护稳定的 SignalSpec；
- 可扩展 cut、watch 和 gradient tap；
- 不负责指标计算、历史和输出。

长期应避免当前 `register()`、`observe()`、`capture_many()` 三套语义交叠。新 `observe()` 应真正表示“声明并在需要时捕获一个可观测信号”。

### 5.2 LossPort

挂在 `BaseModel` 上的单次 Loss 计算账本。

Trainer 自动提供默认 `main` scope，模型任意成员方法都可以：

```python
self.lsp.add("task", task_loss)
self.lsp.add("vq.commitment", commitment_loss, weight=0.1)
self.lsp.add("router.balance", balance_loss, weight=self.balance_weight)
```

LossPort 负责：

- 维护当前 loss scope；
- 防止同名 Loss 静默覆盖；
- 保留 raw value、weight 和 weighted contribution；
- 生成总 Loss；
- 生成 detached 的 Loss 摘要；
- 不跨 micro-step 保存计算图；
- 不决定 snapshot 周期和输出位置。

`settle()` 产生两个结果：直接返回仍带计算图的标量 Tensor；同时在 Port 内留下当前 step 的 detached summary。

```python
self.lsp.add("task", task_loss)
self.lsp.add("vq.commitment", commitment_loss, weight=0.1)
loss = self.lsp.settle()

loss.backward()
details = self.lsp.summary()
```

default step 的 settlement 由 Trainer 负责；custom step 或周期额外 optimizer 的 settlement 由拥有该更新的
模型代码负责。普通模型不感知 scope；只有一个 custom step 内存在多个独立 backward objective 时才显式进入
额外命名 scope。`calculate_loss()` 只登记 term 并返回 raw total，不自行结算。

### 5.3 ModelSnapshot

某个确定训练时刻的完整模型健康状态。

```python
@dataclass(frozen=True)
class ModelSnapshot:
    snapshot_id: str
    step: int
    epoch: int
    batch_idx: int
    stage: str
    phase: str
    target: str
    model_mode: str
    interval_start: Optional[int]
    interval_end: Optional[int]
    signals: Mapping[str, SignalSnapshot]
    anomalies: tuple[HealthAnomaly, ...]
```

关键字段：

- `stage`
  - `train`、`eval`、`probe`、`train_end`、`anomaly`。
- `phase`
  - `default`、`search`、`retrain`、`warmup`、`finetune`、`epoch_end_arch_update` 等训练语义阶段。
- `target`
  - `student`、`ema0p995` 等。
- `interval_start/end`
  - 表明 Loss 等区间聚合数据覆盖哪些 step。
- `model_mode`
  - 明确快照来自 train mode 还是 eval mode。

### 5.4 SignalSnapshot

Loss 和 Tensor 共用的快照信号结构：

```python
@dataclass(frozen=True)
class SignalSnapshot:
    name: str
    kind: str
    role: str
    scope: str
    coordinates: Mapping[str, object]
    metrics: Mapping[str, object]
    metadata: Mapping[str, object]
```

其中：

- `kind`
  - `tensor`、`loss`、`gradient`、`parameter`、`state`。
- `scope`
  - `point`：当前 step/batch 的点采样。
  - `interval`：最近一段 step 的聚合。
  - `probe`：固定 probe batch 或 probe dataset 的聚合。
- `coordinates`
  - 结构化维度，例如 `domain=1`、`expert=3`、`level=0`、`stage=post_mix`，避免把所有维度编码进字符串名。

“Loss 与 Tensor 同等”指的是它们统一进入 SignalSnapshot、历史和报告，不代表对标量 Loss 计算 rank。

### 5.5 SnapshotStore

按完整快照保留历史，而不是按 Tensor 名称保留独立 deque。

```python
class SnapshotStore:
    snapshots: deque[ModelSnapshot]
```

核心查询：

```python
store.latest()
store.list()
store.history("shared_z", "effective_rank")
store.history("loss.task", "mean")
store.trend("shared_z", "effective_rank", last=10)
store.aggregate(last=50)
store.compare(step_a=5000, step_b=10000)
```

默认查询不隐式执行最近 50 次平均。

### 5.6 HealthMonitor

Trainer 持有的总协调器：

```text
HealthMonitor
├── SnapshotPolicy
├── SnapshotBuilder
├── MetricEngine
├── OnlineReducerRegistry
├── SnapshotStore
├── AnomalyDetector
├── RunManifest / ParameterInventory
└── SnapshotSink / Reporter
```

模型不持有 HealthMonitor，也不直接打印报告。

## 6. 默认采样与保留策略

推荐初始默认值：

```python
HealthMonitorConfig(
    snapshot_every=100,
    summary_capacity=1000,
    detail_capacity=10,
    raw_capacity=1,
    default_trend_points=10,
    capture_initial_probe=True,
    capture_epoch_end_probe=True,
    capture_train_end_probe=True,
    capture_on_nonfinite=True,
)
```

### 6.1 默认不保留连续 50 个完整 batch 快照

正常训练只在每 100 个训练 micro-step 生成一次完整在线快照。

连续 batch 诊断使用显式 burst：

```python
monitor.capture_next(
    batches=20,
    profile="detailed",
)
```

异常触发时可以自动进入 burst：

```python
AnomalyPolicy(
    capture_on_nonfinite=True,
    burst_batches=10,
)
```

### 6.2 分层保留

#### Summary history

最近 1000 个完整 ModelSnapshot，只保存轻量 scalar 和少量固定长度元数据。

默认不保存：

- sketch；
- per-dim 向量；
- raw Tensor；
- Tensor pair matrix。

#### Detail history

最近 10 个详细快照，可以包含：

- per-dim top-k；
- spectrum；
- sketch；
- routing/VQ 详细指标；
-关系分析所需的受限数据。

典型来源：

- epoch-end probe；
- train-end probe；
- anomaly；
- 手动 capture。

#### Raw history

默认最多 1 个 sampled raw 快照，只在异常或显式手动请求时生成。

raw capture 必须限制：

- 最大行数；
- 最大元素数；
- 是否转 CPU；
- 是否允许落盘。

## 7. 不同信号的采样口径

### 7.1 Loss

当前默认口径与中间 Tensor 完全对齐：只读取 Snapshot step 的 LossPort summary。例如 step 100：

```python
{
    "scope": "point",
    "raw": 0.417,
    "weight": 1.0,
    "weighted": 0.417,
}
```

默认 `snapshot_every=100` 且 `capture_first_step=False`：step 1～99 不生成历史，step 100 同时读取 LSP 与 OBP。
LossPort 不做区间均值或在线累计。若以后需要 steps 1～100 的 mean/std/min/max，应由 HealthMonitor 的显式
interval reducer 作为另一种采样策略实现，并明确标为 interval；该能力不进入 LSP，也不能冒充与 point Tensor
完全同口径。

### 7.2 中间 Tensor

普通中间 Tensor 默认只在 Snapshot step 捕获一次：

```python
{
    "scope": "point",
    "mean": 0.017,
    "std": 0.482,
    "abs_max": 3.91,
    "finite_ratio": 1.0,
    "effective_rank": 27.3,
}
```

Snapshot 中仍保留 scope，以便未来显式 reducer 能与当前默认 point 信号区分。

对 MMoE gate、AllExpertMixer stream norm、contexture geometry 这类历史上需要覆盖整段训练的信号，
允许在 SignalSpec 中显式设置 `capture_policy=every_step_summary`。HealthMonitor 每步只短暂取得 detached
输入，用 Welford、count/sum/sum-square、min/max、受限 histogram 或模型提供的 reducer 更新在线状态，不保存
raw Tensor。在线状态可以在 interval、epoch、phase 或 run 边界 flush 成 SignalSnapshot。

因此需要严格区分：

```text
snapshot_only       每 N step 捕获一次点快照，默认策略
every_step_summary  每步更新在线归约，不保留逐步 raw
boundary_only       只在固定训练周期边界捕获
probe_only          只在固定 probe 上捕获
audit_only          只在显式审计工作流中启用
```

旧实现中的 `record_step_interval` 迁移为 HealthMonitor policy override，而不是继续成为模型构造参数。

### 7.3 Probe

为了跨训练阶段可靠比较，默认在以下阶段对固定、不 shuffle 的 probe batch 或 probe dataset 生成快照：

```text
before_train
epoch_end
train_end
```

Probe Snapshot 必须记录：

- probe 名称；
- split；
-固定样本标识或 batch fingerprint；
- model mode；
- target；
-包含的 batch 数和样本数。

EMA student/teacher 对比必须使用同一个 probe。

## 8. 聚合语义

### 8.1 同一快照内的多 batch 聚合

mean、variance、min、max、finite count 等指标应通过 count/sum/sum-square 或 Welford 算法按样本数合并。

不能简单平均多个 batch 的 mean 和 variance，因为最后一个小 batch 不应与完整 batch 权重相同。

### 8.2 Rank 聚合

多个 batch 的 effective rank 平均值不等于合并样本后的整体 effective rank。

如果一个 Probe Snapshot 需要整体 rank，应：

1. 对多个 batch 进行受限 reservoir/sketch 合并；
2. 在合并后的 sketch 上计算一次 rank；
3. 保存最终 scalar；
4. summary 模式下丢弃 sketch。

如果只平均每批 rank，指标必须明确命名为：

```text
mean_batch_effective_rank
```

### 8.3 多个历史快照的时间聚合

时间窗口聚合由查询层显式执行：

```python
store.aggregate(last=50)
```

不能把滚动平均重复写入每个 ModelSnapshot，否则会造成数据冗余，并且未来无法用不同窗口重新计算。

## 9. 默认指标 Profile

MetricEngine 必须根据 role 和显式 profile 选择指标，不能再把 spectral、correlation 和 cosine 捆绑执行。

### 9.1 Hidden / representation

默认保存：

```text
shape
sample_count
mean
std
abs_max
finite_ratio
zero_ratio
effective_rank
rank_ratio
top1_energy_ratio
```

其中：

```text
rank_ratio = effective_rank / min(num_samples, flat_dim)
```

### 9.2 Logits

默认保存：

```text
mean
std
min
max
abs_max
finite_ratio
```

不默认计算 rank。binary logit saturation 等任务相关指标按需开启。

### 9.3 Loss

单步原始结构：

```text
raw
weight
weighted
contribution_ratio
finite
```

区间结构：

```text
count
mean
std
min
max
last
weighted_mean
contribution_ratio
finite_ratio
```

### 9.4 Routing

当 role/profile 明确为 routing 时按需计算：

```text
entropy
expert_usage
expert_dead_ratio
```

### 9.5 VQ

当 role/profile 明确为 VQ 时按需计算：

```text
used_code_ratio
code_entropy
quantized_cos
residual_norm
```

### 9.6 默认关闭的指标

```text
per_dim
correlation
cosine
CKA
subspace overlap
完整 spectrum
raw Tensor
```

这些指标通过显式 profile 或 override 开启。

## 10. MetricEngine

现有 metrics 模块应接成真正的 dispatcher：

```python
METRIC_REGISTRY = {
    "basic": compute_basic_health,
    "rank": compute_rank_health,
    "per_dim": compute_per_dim_health,
    "correlation": compute_correlation,
    "cosine": compute_cosine,
    "routing": compute_routing_health,
    "vq": compute_vq_health,
}
```

执行时：

```python
for metric_name in resolved_profile.metrics:
    result.update(METRIC_REGISTRY[metric_name](metric_input))
```

要求：

- basic-only 不构造或保存 sketch；
- rank 可以临时构造 sketch，但 summary history 只保存 scalar；
- spectral-only 不顺带计算 correlation/cosine；
- role/axes/metadata 必须进入 MetricInput；
- 不同 profile 的成本必须可解释和可测试。

## 11. 默认报告

默认报告是一张紧凑总览，不逐 Tensor 打印大段 per-dim 表。

示例：

```text
[Model Health]
step=10000 epoch=3 stage=train target=student interval=9901..10000

Loss
signal               mean      std      last   weight  contribution
task                0.4212   0.0261   0.4170    1.00       94.2%
vq.commitment       0.2580   0.0210   0.2490    0.10        5.8%

Tensor
signal          mean      std   abs_max  finite    rank  rank_ratio   delta_rank
shared_z       0.017    0.482      3.91   1.000    27.3       0.426         -0.8
head_input     0.012    0.507      4.12   1.000    25.8       0.403         -1.2
logits        -2.130    1.410      8.72   1.000       -           -            -

Warnings
none
```

默认重点：

- 当前值；
- 相对上次快照的变化；
- 非有限值；
- rank collapse；
- Loss contribution 异常变化。

详细查询：

```python
monitor.detail("shared_z", snapshot="latest")
monitor.history("shared_z", "effective_rank")
monitor.compare(step_a=5000, step_b=10000)
```

## 12. 异常与 Burst 模式

不通过连续保存完整 batch 来等待异常，而是每步运行廉价哨兵：

```text
total loss finite
Loss abs value
可选 gradient norm
可选 critical logits finite/abs_max
```

异常触发后：

- 立即生成 anomaly snapshot；
- 提升到 detail profile；
- 可保存 sampled raw Tensor；
- 可自动捕获接下来 N 个连续 batch；
- 可根据策略停止训练。

```python
AnomalyPolicy(
    capture_on_nonfinite=True,
    max_abs_loss=1e4,
    burst_batches=10,
    raw_sample_rows=32,
)
```

为了查看异常前的上下文，可以永久保留最近若干步的轻量 scalar ring buffer：

```text
total_loss
各 Loss term
gradient norm
```

但不默认保留异常前每个 batch 的完整 Tensor 快照。

## 13. Trainer 与 Protocol 接入

逐模型审查后的最终边界是：模型私有的特殊训练使用 Protocol，模型无关的通用训练能力使用 Trainer
派生类。独立生命周期对象会复制 Trainer 的边界并形成事件控制总线，因此已经删除。

```text
CustomTrainStepProtocol
  模型私有的 batch 更新：多 backward、多 optimizer、特殊梯度处理

TrainingCycleProtocol
  模型私有的 epoch/eval 边界行为：异步 optimizer、状态调度与评估后更新

BaseTrainerWithLife / EMATrainer / audit Trainer
  模型无关的通用训练能力

HealthMonitor
  被动保存 Loss、Tensor、gradient、parameter、state、history 和 report
```

普通模型只实现 `calculate_loss()`，不继承任何训练 Protocol。`BaseTrainer` 只支持普通/custom step；
`BaseTrainerWithLife` 才解释 TrainingCycleProtocol 并从 TrainerComponents 接入 HealthMonitor。HealthMonitor
不得切 phase、清 optimizer、冻结参数或更新 selector。

### 13.1 唯一 train_step 包装点

模型不再拥有 record/print 周期。

Trainer 负责：

```python
with health_monitor.capture_scope(model, ctx):
    result = trainer.train_step(batch, ctx)
```

逻辑上：

```text
HealthMonitor 判断本 step 是否需要完整快照
    ↓
启用 ObservationPort capture scope
    ↓
BaseTrainer.train_step
    └── 默认 main LossPort scope
        ├── _default_train_step
        └── custom_train_step
            └── 可选的额外 objective scope
    ↓
scalar loss Tensor + detached LossPort summary
    ↓
SnapshotBuilder 合并 Observation trace 与 Loss signals
    ↓
MetricEngine 计算本 profile 指标
    ↓
SnapshotStore.commit(ModelSnapshot)
```

这要求 Trainer 先具备唯一的 `train_step()` 扩展入口，确保 default 和 custom train step 都经过同一个 observation wrapper。

`custom_train_step` 是正式能力，不是迁移期 fallback。MultiEmb、SSIM、Pareto、AutoMTL、SDSP 和 M3oE
都可以使用它。一个 custom step 内无论执行多少 optimizer，默认都只构成一次逻辑模型更新；EMATrainer
在整个 custom step 返回以后按最终模型状态更新一次，不解释 optimizer 名字或数量。

M3oE 还需要每 epoch 一次 architecture optimizer，因此继续使用已经存在的 `on_train_epoch_end` 周期方法。
该方法返回局部 `updated_state_names`，只为避免 EMA 对普通参数多衰减一次；它不是事件总线。

### 13.2 最小周期函数返回

周期函数通常返回 `None`。只有确实发生额外模型更新时，直接返回修改的 state names：

```python
return ("backbone.arch_alpha", "backbone.arch_beta")
```

返回值不携带 Loss、audit name、metadata 或 Trainer 命令。Loss 由 LossPort 生产；状态观测由 module probe
或 custom model 的 ObservationPort 信号生产，HealthMonitor 被动消费。EMA 子类只用 state names 做局部同步。模型若拥有第二 optimizer、search best 或温度等
非 model state，实现独立的 `TrainerStateProtocol` 进入 checkpoint。

### 13.3 eval、EMA 与两个独立训练阶段

两阶段训练不在同一个 Trainer 内切换 phase。第一阶段模型显式实现 `TwoStageTrainingProtocol`，实验用
`TwoStageRunner` 编排：

```text
factory(stage_one_model, STAGE_ONE) -> fresh Trainer A
    -> 正常训练、评估、EMA、EarlyStopper、HealthMonitor
    -> 每次评估后形成 StageReport
    -> should_finish_stage_one(report) 是额外终止条件
    -> 捕获 detached CPU StageSnapshot
    -> initialize_stage_two(snapshot) 返回全新模型
factory(stage_two_model, STAGE_TWO) -> fresh Trainer B
    -> 重新创建 optimizer、EMA、EarlyStopper、HealthMonitor 与计数器
    -> 按普通 Trainer 完整训练
```

`StageReport` 保留 `EvaluationFeedback` 的 target 映射，因此 mixin 可以明确选择 student、primary 或某个 EMA
target 来追踪 search best。模型在 `initialize_stage_two()` 内决定 restore best/terminal、reset 和 freeze；
Trainer 不解释这些规则。

EMA 不需要 archive/reseed 回调：第二个 `EMATrainer` 构造时直接从第二阶段新模型创建 teacher。第二阶段
EarlyStopper 也是新对象，不可能继承 search patience/best。两个 HealthMonitor 分别拥有独立 SnapshotStore，
阶段历史不会静默混合；需要联合查看时由 `TwoStageRunResult` 在实验报告层组合。

### 13.4 checkpoint/resume 是训练状态契约

完整 checkpoint 至少包含：

```text
model state
trainer counters: epoch / global_step
main optimizer state
scheduler states
optional model trainer state（第二 optimizer、phase、search best 等）
EMA states and phase-local update counters
EarlyStopper phase state
HealthMonitor policy / online reducer / sink cursor state
Python / NumPy / Torch RNG state（需要精确恢复随机 selector 时）
phase audit journal
```

这用于修复当前不可完整恢复的状态：TwoStageRetrainMixin 的 best state 和 phase、AutoMTL 的 arch optimizer
与导出状态、M3oE 的 arch optimizer、SDSP selector/value function/sim_domain/prototype、以及在线观测
归约器。快照历史本身可以采用 append-only 外部 sink，不要求全部塞进 checkpoint，但 reducer 未 flush
状态和 sink cursor 必须可恢复。

### 13.5 MSR 注册模型逐项复现审查（2026-07-18）

审查不是只看类名，而是给 33 个 `MODEL_REGISTRY` key 各跑一个真实 backward/optimizer step，并让
`HealthMonitor(snapshot_every=1)` 接收结果。DIN 两项使用合法 sequence group schema；HierRec 使用它要求的
四段 MLP 维度；SDSP 使用必填的 per-domain `scaled_batch`。结论如下。

| registry key | 训练语义 | 当前 Obs/Loss 复现结论 |
|---|---|---|
| `sharedbottom` | default step | 通过；核心无显式 OBS，backbone/head probe 可看共享表征、logits 与塔内层 |
| `sharedbottom_codebook` | default + VQ Loss | 通过；`task`、`vq.quantize` 分项进入同一快照 |
| `sharedbottom_less` | default step | 通过；核心无显式 OBS，使用同一组 module probe |
| `sharedbottom_plus` | default step | 通过；核心无显式 OBS，使用同一组 module probe |
| `din` | sequence architecture | 合法 sequence schema 下通过；训练无特殊 phase，模块 probe 可复现层输出 |
| `ads_din` | sequence architecture | 合法 sequence schema 下通过；ADS 内部可用模块 probe，不需要专用 Trainer |
| `all_expert_mixer` | default + 多个可选辅助 Loss + 大量在线聚合 | 训练与复合 Loss 分项通过；历史聚合仍由模型内部 reducer/export 保留，尚未迁到新 reducer registry |
| `contexture_prototype_sharedbottom` | default + context/covariance Loss + 在线几何聚合 | 训练与复合 Loss 分项通过；旧几何聚合仍可用，尚未统一进入 SnapshotStore |
| `mmoe` | default + gate 在线聚合 | 通过；旧 gate rows 仍可导出，通用层快照可用 module probe |
| `mmoe_grad_balance` | default + backward gradient balancing/聚合 | 通过；原梯度平衡行为保留，旧 aggregate 尚未迁到 HealthMonitor reducer |
| `ple` | default + gate 在线聚合 | 通过；旧 gate rows 保留，层快照可用 module probe |
| `ple_v1` | default step | 通过；无 Trainer 特例 |
| `star` | default step | 通过；无 Trainer 特例 |
| `star_add` | default step | 通过；无 Trainer 特例 |
| `m3oe` | custom main step + epoch-end architecture step | 通过；custom step、epoch 周期方法、独立 optimizer/Loss scope/checkpoint，EMA 能看到局部异步更新 |
| `m3oe_v1` | 同上，domain factor | 通过；与 `m3oe` 共用 Protocol 实现 |
| `m3oe_v2` | 同上，beta-domain factor | 通过；与 `m3oe` 共用 Protocol 实现 |
| `m2m` | default step | 通过；无 Trainer 特例 |
| `ppnet` | default step | 通过；无 Trainer 特例 |
| `epnet` | default step | 通过；无 Trainer 特例 |
| `pepnet` | default step | 通过；无 Trainer 特例 |
| `crocodile` | default + disentanglement Loss | 通过；task/disentanglement 已分项 |
| `pareto` | custom step，按 domain 动态加权 | 通过；每个出现的 domain Loss 与动态 weight 被保留，EMA buffer 随模型 checkpoint 保存 |
| `hierrec` | architecture-only | 合法四段维度下通过；无 Trainer 特例 |
| `automtl` | custom step + warmup/search/finetune + arch optimizer/export | 连续训练可复现；同一 custom step 的两个 optimizer 只触发一次 EMA；arch optimizer、离散化进度的完整 resume 仍待 TrainerStateProtocol 迁移 |
| `riple` | default + disentanglement Loss | 通过；task/regularization 已分项；另有 `aux_loss_weight` 未参与原公式的旧模型问题 |
| `satrans` | architecture-only | 通过；无 Trainer 特例 |
| `hamur` | architecture-only | 通过；无 Trainer 特例 |
| `hypergate` | default + control auxiliary Loss | 通过；task/control 总辅助项已分项；domain/task 两个 control 子项尚未拆开 |
| `sdsp_ple` | custom step + prototype Loss + eval-driven selector | 连续训练可复现，task/prototype 分项通过；selector/value/prototype 的完整 resume 尚待 TrainerStateProtocol 迁移 |
| `sdsp_mmoe` | custom step + prototype Loss + eval-driven selector | 与 `sdsp_ple` 相同 |
| `ssim` | search→新模型 retrain，restore/reset/freeze | 通过；Protocol、CPU snapshot、新模型/Trainer/optimizer/EMA/EarlyStopper 隔离均有测试 |
| `multiemb` | search→新模型 retrain，restore/freeze | 通过；Protocol、命名 Loss、best-state、新模型与独立健康快照均有测试 |

由此可见，“模型结构不同”不等于需要新 Trainer：24 个普通 key 全部使用 default step；三种 M3oE 使用
custom step 加 epoch 周期方法；MultiEmb/SSIM 保留 custom step 并由 `TwoStageRunner` 启动独立 retrain；Pareto、AutoMTL、
SDSP 的 custom step 仍有正当性。当前不能宣称 33 个 key 的精确 resume 全部完成，缺口明确集中在 AutoMTL
和 SDSP，而不是通用 forward observation。

## 14. 依赖与目录

基础 Loss 契约：

```text
rosen/core/train/loss.py
├── LossTerm
└── LossPort
```

Observatory：

```text
rosen/utils/observatory/
├── port.py
├── snapshot.py
├── store.py
├── policy.py
├── profiles.py
├── collector.py
├── formatting.py
└── metrics/
```

长期可以把完整 Observatory 子系统从 `rosen.utils` 提升为：

```text
rosen.observatory
```

依赖方向：

```text
BaseModel
  -> core.train.loss
  -> ObservationPort

Trainer observation integration
  -> HealthMonitor
  -> core.train contracts

core.train
  不依赖 core.trainers
  不依赖完整 Observatory
```

## 15. 允许的破坏性迁移

新设计允许删除或替换：

```text
TensorObservatory.window_size
每个 Tensor 一个 deque 的历史结构
get_window_stats() 作为主 API
ExplicitFeatureRecorder
ctx.recorder
record_observation_port()
模型内 monitor_step_interval
模型内 record_step_interval
模型内 verbose_window_stats
模型内部直接 print Observatory 表
依赖 _latest_debug 搬运大批 Tensor 的通用模式
```

替代为：

```text
ObservationPort
LossPort
HealthMonitor
SnapshotPolicy
ModelSnapshot
SignalSnapshot
SnapshotStore
MetricProfile
SnapshotReporter
```

迁移时不为了保留旧 API 而牺牲新的 Protocol 边界和存储语义。

### 15.1 历史 Obs 基础能力去向

“允许不兼容”不等于可以静默删除能力。第一阶段按下面方式保留或复现：

| 历史能力 | 当前去向 | 状态 |
|---|---|---|
| forward capture | custom model 的 `ObservationPort.observe/capture` + 通用 module probe | 已进入新快照主链；核心模型不显式 capture |
| 最近历史 | `SnapshotStore(capacity)`，按完整快照淘汰 | 已替代 per-name deque 主语义 |
| latest | `SnapshotStore.latest()` | 已实现，不用窗口均值冒充最新值 |
| scalar/tensor history | `SnapshotStore.history(signal, metric, filters...)` | 已实现轻量指标历史 |
| basic/rank/spectral | HealthMonitor basic/rank 摘要 | 已实现第一版；profile registry 未完成 |
| relation/correlation/cosine | 旧 `TensorObservatory` 与 metrics 模块 | 能力保留，尚未接入新 Snapshot profile |
| cut | `ObservationPort.cut/cutting` + audit Trainer | 保留并有回归测试 |
| watch | `ObservationPort.watch/watching` + cut audit | 保留并有回归测试 |
| gradient tap | `ObservationPort.tap_gradient/tapping_gradients` + observation audit | 保留并有回归测试 |
| activation hook | 通用 `observe_module` 覆盖 ReLU 等边界；`IndividualReLURecorder` 暂保留 | 可复现，旧类尚未删除 |
| field-zeroing intervention | `field_zeroing_intervention` | 保留并有回归测试 |
| domain gradient-conflict | `ObservationAuditWorkflow` | 保留并有回归测试 |
| 模型自定义 every-step aggregate | MMoE/PLE/Contexture/AllExpertMixer 原 reducer/export | 功能保留，尚未迁到 provider/reducer registry |
| JSONL/CSV 输出 | HealthMonitor JSONL sink 与 `export_csv()` | 已实现 |
| 绘图 | 旧 plots API 从标量历史消费 | 能力保留，尚未建设新默认 reporter |

因此当前没有为了新 API 删除可用的审计能力，但也不能把“旧工具仍可调用”表述成“所有能力都已统一”。
下一阶段的工作是迁移关系指标和模型专用 reducer，而不是重新发明这些能力。

## 16. 分阶段实现顺序

截至 2026-07-18：阶段 1、Snapshot 基础数据模型、ObservationPort 主入口和 Trainer HealthMonitor 的最小闭环
已经实现；阶段 2 完成了 M3oE 与 MultiEmb/SSIM 的独立两阶段编排，AutoMTL/SDSP 尚未迁完；MetricEngine registry、
every-step reducer registry、EMA target representation probe、异常/burst、紧凑 reporter 仍未实现。旧的
cut/watch/gradient tap、激活 hook、field-zeroing、gradient-conflict 和模型专用 aggregate 没有删除，因此
历史实验仍可运行，但它们还不是统一 Snapshot API 的一部分。

### 阶段 1：Loss 契约

- 增加唯一的 `BaseTrainer.train_step()`。
- 定义 `LossTerm` 与 `LossPort`。
- default/custom train step 直接返回标量 Tensor。
- `LossPort.settle()` 返回 Tensor、保存 detached summary，并清空当前 scope 的带图 term。
- LossPort 不保留跨 micro-step 计算图。

### 阶段 2：Protocol、阶段切换与 checkpoint

- 保留 `CustomTrainStepProtocol` 与 `TrainingCycleProtocol`，不增加独立 lifecycle 对象。
- 定义 `TrainingCycleProtocol`、`TrainerStateProtocol` 和类型化 `EvaluationFeedback`。
- 定义 `TwoStageTrainingProtocol`、`StageReport`、`StageSnapshot` 与 `TwoStageRunner`。
- 两阶段用类型化 stage 枚举调用同一个 factory，创建两个独立 Trainer；不复制 Trainer，也不在 eval hook 内切换模型。
- 明确 feedback target；第二阶段重新创建 optimizer、EMA、EarlyStopper、HealthMonitor 和计数器。
- checkpoint/resume 覆盖模型侧训练状态、全部 optimizer、EMA、RNG 和在线 reducer。
- MultiEmb/SSIM 两阶段 smoke 验证新模型构造、best restore、freeze/reset 与运行状态隔离。

### 阶段 3：Snapshot 数据模型和存储

- 定义 `ModelSnapshot`、`SignalSnapshot`。
- 实现 `SnapshotStore` 和分层 retention。
- 实现 latest/history/trend/compare 查询。
- 不实现旧 per-name deque 兼容层，或只保留短期迁移适配器。

### 阶段 4：MetricEngine 与在线归约

- 把现有 metrics 模块接成真正 registry。
- 删除旧硬编码 `spectral_stats()` 总开关路径。
- 实现 role/profile 驱动的指标选择。
- 确保 basic-only 不构造长期 sketch。
- 实现 every-step summary 的 reducer registry 和 interval/epoch/phase/run flush。
- 模型特有派生指标使用 provider/reducer，不污染通用 MetricEngine。

### 阶段 5：ObservationPort 接入

- 统一 `observe()` 的注册和 capture 语义。
- 暴露完整 specs。
- SnapshotBuilder 接收 trace + specs。
- 保留 cut/watch/gradient 能力，但不让它们污染默认健康报告。
- 增加 coordinates、capture_policy、reduction_scope 和 state 信号。

### 阶段 6：Trainer HealthMonitor

- 实现 `SnapshotPolicy`。
- observation wrapper 覆盖完整 `train_step()`。
- 每 N step 同时读取 LSP/OBP 当前 summary，提交 point 快照。
- 如确有需求，再在 Monitor 中增加显式 interval reducer，不改变两个 Port 的职责。
- 实现 initial/epoch-end/train-end probe。
- 接收周期函数的 Observation trace 和更新 state names；两个训练阶段各自维护独立 SnapshotStore。

### 阶段 7：报告、异常和 Burst

- 实现紧凑 overview。
- 实现 anomaly detector。
- 实现 detail/raw retention。
- 实现手动与自动 burst。

### 阶段 8：迁移模型并删除旧系统

- 把 `_latest_debug + recorder.record()` 迁到 `observe()`。
- 把手工 Loss debug 字段迁到 `LossPort`。
- 删除模型内部记录周期和打印逻辑。
- 删除不再使用的 compatibility recorder。
- 按本文模型矩阵逐项迁移；每个注册键通过对应的 default/custom/hook smoke 后才删除旧入口。
- `reset_train_observations/export_*_rows` 分别迁移为 scope reset、store query/export 和 RunManifest。

## 17. 验证要求

实现时至少覆盖：

1. 同一 step 的 Tensor 和 Loss 进入同一个 ModelSnapshot。
2. custom/default train step 都能生成完整快照。
3. Snapshot 中所有值均 detach，不保留旧计算图。
4. summary capacity 按完整 snapshot 淘汰，而不是按名称独立淘汰。
5. basic-only 不保存 sketch。
6. rank-only 不计算 correlation/cosine。
7. 1000 个 summary snapshot 的内存规模可控。
8. 不同 batch size 的 mean/variance 按样本数正确合并。
9. Probe rank 基于合并 sketch，而不是错误平均 batch rank。
10. 默认报告不输出 per-dim 大表。
11. latest 明确展示最后一次异常，不被历史平均隐藏。
12. anomaly 能触发 detail snapshot 和 burst。
13. student 与 EMA target 使用同一 probe 时可比较。
14. 无 observation 的普通模型仍能记录 Loss 快照。
15. 无 HealthMonitor 时训练主路径保持低开销。
16. every-step summary 只保留 reducer 状态，不保留逐 batch raw，并能按 epoch/phase/run 正确 flush。
17. coordinates 能表达 domain/expert/level/stage，查询结果与旧聚合行数值一致。
18. cut/watch、gradient tap、激活 hook、field-zeroing 和 domain gradient-conflict 审计均有替代入口。
19. 两阶段使用不同模型、Trainer 和 SnapshotStore，search/retrain history 不会静默混合。
20. MultiEmb 第二阶段新模型恢复 best search state，并冻结 mask 网络；新 optimizer 不含 search state。
21. SSIM 第二阶段新模型恢复 best search state、重置 branches，并冻结 search-only 参数。
22. 第二阶段 EMA teacher 从第二阶段新模型直接初始化，不含 search EMA 平滑残留。
23. `StageSnapshot` 是 detached CPU clone；第二阶段 checkpoint 可独立恢复 model/optimizer/EMA 状态。
24. AutoMTL 的 warmup/search/finetune、arch optimizer、逐 epoch离散化和 architecture export 可恢复。
25. M3oE 的 epoch-end auxiliary backward 使用独立 Loss scope 和 optimizer state。
26. Pareto 动态 domain Loss、EMA 权重和 warmup 边界在 resume 后一致。
27. SDSP 的 selector、routing state、prototype 和 eval-driven 更新在 resume 后一致。
28. codebook eval/probe 不会因观测动作额外改变模型 buffer。
29. AllExpertMixer、MMoE、PLE、Contexture 的历史 train aggregate 可由 reducer/export 重建。
30. 33 个注册模型键均有一次针对其训练类型的 smoke，并验证声明的 Loss term 和必要 state signal。

## 18. 第二阶段仍待决定的问题

以下问题留到具体实现阶段决定：

- summary history 是否只保存在内存，还是同步 append 到 JSONL/Parquet。
- `snapshot_every=100` 是否作为全局默认，或由数据规模/实验配置决定。
- 固定 probe batch 的持有方式和 batch fingerprint 格式。
- DDP 下 Loss/Tensor summary 的跨 rank 合并策略。
- gradient/parameter health 是否进入第一版，还是后续扩展。
- anomaly snapshot 是否允许自动重新 forward 当前 batch。
- Observatory 是否在本次重构中直接提升为 `rosen.observatory`。

这些问题不影响本文已经确认的核心决策：

> 不再默认保存连续 50 个完整 batch 记录；以对齐的 ModelSnapshot 为历史单位；默认在同一个采样 step 同时读取 LSP 与 OBP，并只保存 point 健康摘要和适用时的 rank；详细数据、区间 reducer、异常或阶段 probe 均按需启用。
