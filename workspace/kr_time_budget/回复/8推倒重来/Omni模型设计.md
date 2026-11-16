# 用OmniEmbedding搭建时间预算模型

Argon 9继续沿用本页的Omni/EmbView与特征路由，新增尺度参数化、shape下界与内容幅度对照。当前入口见[双分支模型](../../pipeline/time_budget_v8.py)、[双分支实验报告](../9继续双分支/双分支实验报告.md)；下文保留v7联合训练基准的实现说明。

模型已实现：[time_budget_v7.py](../../pipeline/time_budget_v7.py)，类名`TimeBudgetModel`。不改Rosen核心，也不继续继承v3→v4→v5的实验补丁链；复用已有稳定的单威布尔标量函数。默认采用乘积Loss的端到端联合训练，原冻结预算的两阶段方式仅作为显式对照。完整train/valid/test联合训练已完成，详见[联合训练报告](联合训练报告.md)。

## 默认训练语义：一个乘积Loss同时更新两塔

```python
model = TimeBudgetModel(manager, mode='ours', clock='dual')  # training_strategy='joint'
gamma, _, _ = model.budget_components(batch)
stay = torch.sigmoid(model.content_net(model.content_view(batch)).squeeze(-1))
prediction = gamma * stay
loss = BCE(prediction, y_continue)
loss.backward()  # 预算Embedding/MLP与内容Embedding/MLP同时接收梯度
optimizer.step()
```

默认不预热、不冻结、不detach、不分别给两塔拟合退出标签；普通Trainer负责一次forward/backward/update。内容末层保持随机非零权重，3.5仅作为bias初始化，不要求联合训练的输出等于固定先验。

训练语义类似ESMM的乘积概率参与联合优化，但不等同于完整ESMM：后者还有CTR标签的辅助损失；当前只有继续/退出监督，没有自动增加预算或内容真值标签。[ESMM原文](https://arxiv.org/html/1804.07931v2)

## 用到的原生能力

| 接口 | 本模型中的用途 |
| --- | --- |
| `OmniEmbLayer.get_settings(include_fields=...)` | 从原Schema选择预算参数网的34字段设置，不重写编码 |
| `EmbView(..., include_fields=...)` | 直接生成34字段预算输入、48字段内容输入，不再用全量向量置零掩码 |
| `omni_embedding.whole` | Fusion读取全部54字段 |
| `view.embedding_dim` | 自动计算MLP输入维度，不手工计数 |
| `view.selected_settings` | 检查实际生效的设置与顺序 |
| `view.field_segments` / `view.field_slices` | 获取每字段拼接位置，供断点和路由测试使用 |
| `view(batch, split_by='name')` | 查看逐字段编码结果 |
| `view(batch, split_by='source')` | 查看按Rosen来源分组的表示；业务路由仍按字段清单决定 |

视图不持有参数，只引用Omni。预算塔独立创建34字段的`budget_embedding`；从主Omni复制同名表的初始值，保证初始编码相同、存储独立。内容塔通过`content_view`访问主Omni；共享背景在两塔各自学习，冻结预算不会冻结内容侧用户画像和上下文。

```python
budget_settings = self.omni_embedding.get_settings(include_fields=BUDGET_PARAMETER_FIELDS)
self.budget_embedding = OmniEmbLayer(emb_settings=budget_settings)
self.budget_view = EmbView(self.budget_embedding, include_fields=BUDGET_PARAMETER_FIELDS)
self.content_view = EmbView(self.omni_embedding, include_fields=CONTENT_TOWER_FIELDS)
self.fusion_view = self.omni_embedding.whole
```

顺序由Omni的实际settings决定，`include_fields`负责筛选，不承诺按传入列表重排。模型与调试统一使用视图提供的字段切片。

## 网络与公式

- 预算：34字段→252维→Linear(252,64)→SiLU→Linear(64,4)。前两维为分钟尺度/形状，后两维为屏数尺度/形状。
- 内容：48字段→316维→64→32→1，末尾sigmoid得到stay。
- Fusion：54字段→322维→64→32→1，末尾sigmoid。Content-only基线采用48字段，不再沿用历史18字段基线。
- `mode='ours', training_strategy='joint'`（默认）：直接优化BCE(gamma×stay,y)，同时更新两塔。
- `training_strategy='two_stage'`：显式对照，先训练gamma×sigmoid(3.5)，再冻结预算、训练gamma.detach()×stay。
- `mode='budget'/'content'/'fusion'`：分别使用对应的单阶段预测目标。

`clock='time'`仅使用分钟分支；`clock='dual'`计算gamma_time×gamma_step、logS_time+logS_step。两种模式构造完全相同的模块和4维预算末层，未用的屏数输出在单时钟中不影响预测，避免网络深度或共有初始化成为额外变量。四种mode也保持相同的模块构造顺序。

分钟和屏数分别复用稳定的`single_weibull_survival`：scale=softplus(raw)×30+0.5（各自以分钟、屏为单位），shape=clamp(1.05+softplus(raw),max=5)。当前只复用原参数化，不声称数值30对两种单位均最优。

预算公式只读`elapsed_min_raw`和`req_idx_raw`，不读取可能供网络缩放的同名特征。步长分别为1分钟和1屏，屏数起点为req_idx_raw−1。gamma/S不提前截断，只在BCE入口保护数值。固定背景且两个时钟都不回退时检验单调性；已有数据的分钟回退不能忽略，也不改为cummax。

仅two_stage对照在切换时使用零末层/bias=3.5以保证预测逐值一致；首步上游数据梯度为0，下一步恢复。该对照由TwoStageRunner分别构造新的Trainer和optimizer，预算Embedding及预算MLP全冻结。联合训练没有这个切换，也不允许误调用TwoStageRunner启动预算预热。

## 数据与调试

当前继续复用data/v3的编码、log1p与sum聚合，不在搭模型时顺便修改预处理。该版本的elapsed_min仍是原始分钟，调试入口在LazyFrame中显式添加`elapsed_min_raw`别名；已有req_idx_raw直接保留。没有写回数据，不重拟合Schema。

以后若对网络数值做缩放，应新建数据版本并在缩放前保存raw列；模型缺少raw列会直接报错，不隐式取已缩放值兜底。新视图改变输入维度，历史v5内容MLP checkpoint不能直接加载，不宣称与v5初始化配对。

直接Debug：[debug_time_budget.py](../../debug_time_budget.py)。

```bash
uv run python workspace/kr_time_budget/debug_time_budget.py --stop-at views --output workspace/kr_time_budget/runs/my_model_debug
uv run python workspace/kr_time_budget/debug_time_budget.py --stop-at forward --output workspace/kr_time_budget/runs/my_forward_debug
uv run python workspace/kr_time_budget/debug_time_budget.py --clock time --output workspace/kr_time_budget/runs/my_time_debug
```

`views`断点：`model.content_view`、`view_fields`、`view_shapes`、`view_slices`、`field_embeddings`、`source_embeddings`。

`forward`断点：`clock_parts`、原始时钟、`gamma`、`survival`、`log_survival`、`initial_budget_grad`；可进入模型和原生Omni。

`trained`断点：`product`、`prediction_table`，检查最终乘积。two_stage模式还可查看`result`检查切换和冻结。为保留日志与checkpoint，重复调试使用新`--output`。旧`stage-two`断点名仍兼容。

运行旧两阶段对照：

```bash
uv run python workspace/kr_time_budget/debug_time_budget.py --training-strategy two_stage --output workspace/kr_time_budget/runs/my_frozen_debug
```

## 验证结果与范围

12项模型测试覆盖实际字段路由/切片、两塔参数独立、共享背景进入两塔、raw与网络时钟分离、单/双时钟公式、单调网格、两个预算头梯度、四mode目标、共有初始化、旧两阶段零跳变与冻结。新增联合训练检查明确要求同一个乘积BCE在首batch给两塔MLP及各自用户Embedding产生非零梯度，并由同一个optimizer实际改变两侧参数。

真实KR v3 smoke：每阶段8192条train（2个batch，每批4096条），valid/test各4096条；train≤20220506、valid=20220507、test=20220508，均为各split前部的少量数据。seed2026，CPU4线程，AdamW lr=1e-3，普通权重衰减1e-6，Embedding/bias/一维参数零衰减。分钟与双时钟均完成原生TwoStageRunner训练，阶段切换误差0，预算全state逐值冻结。小样本loss和AUC不作为效果结论。

验证文件：[模型测试](../../runs/exit_v7_smoke/logs/model-tests.log)、[分钟链路日志](../../runs/exit_v7_smoke/logs/time.log)、[双时钟链路日志](../../runs/exit_v7_smoke/logs/dual.log)、[分钟摘要](../../runs/exit_v7_smoke/time/summary.json)、[双时钟摘要](../../runs/exit_v7_smoke/dual/summary.json)。

默认联合训练另有小样本验证：[测试](../../runs/exit_v7_smoke/logs/joint-tests.log)、[真实batch联合训练日志](../../runs/exit_v7_smoke/logs/joint-dual.log)、[联合训练摘要](../../runs/exit_v7_smoke/joint_dual/summary.json)。

正式实验入口：[run_exit_v7_joint.py](../../pipeline/run_exit_v7_joint.py)。Ours和Fusion各完整训练1epoch、421次更新，共用KR v3数据、seed2026与batch4096，按完整valid LogLoss选择checkpoint，再评估48,898条test。Ours AUC=0.684934、User-GAUC=0.532021，Fusion AUC=0.696801、User-GAUC=0.489217。两塔首个真实更新均有非零梯度且参数改变，完整test预测通过乘积公式与checkpoint独立重载核对。没有内容辅助Loss；详细口径、分量诊断与证据见[报告](联合训练报告.md)。
