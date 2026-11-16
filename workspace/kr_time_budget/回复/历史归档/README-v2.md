# Argon 反馈 v2

> 历史资料：保留原阶段口径；本页“当前”指写作时。现阶段结论请以[定稿报告](../14定稿/定稿实验报告.md)为准。

任务依据：[3反馈.md](../../../../Argon-plans/3反馈.md)。实施选择见 [v2 计划](20261116_kr-time-budget-v2.md)。v1 的脚本、数据、结果保持原样。

五折结果与诊断：[v2 执行报告](20261116_kr-time-budget-v2-report.md)。

## 入口

- `features_v2.py`：用户 train/valid/test 切分、自身过去会话先验、折内物品先验和 schema。
- `workspace/kr_time_budget/pipeline/time_budget_v2.py`：tag-only 编码器、固定背景混合威布尔预算、相乘模型。
- `run_exit_v2.py`：全 valid 选择 checkpoint、阶段切换诊断、五折 test/OOF。
- `report_v2.py`：汇总审计和人类可读结果。

在项目根目录执行：

```bash
mkdir -p workspace/kr_time_budget/runs/exit_v2
uv run python -u -m workspace.kr_time_budget.pipeline.run_exit_v2 > workspace/kr_time_budget/runs/exit_v2/run.log 2>&1
uv run python -m workspace.kr_time_budget.pipeline.report_v2
uv run python -m pytest -q workspace/kr_time_budget/tests tests/rosen/utils/test_optimize.py
```

默认复用 `workspace/kr_time_budget/data/v1/` 下 `requests.parquet`、`sessions.parquet`、`prior_events.parquet` 的重建内容；不重读 raw，不改变切场/倒切/label。v2 的折内派生数据和 schema 写 `workspace/kr_time_budget/data/v2/`。输入/输出可分别用 `--source`、`--data`、`--output` 指定。

`--folds 0 --epochs 2 --smoke --output workspace/kr_time_budget/runs/exit_v2_smoke` 做单batch链路验证。Ours至少2epoch，覆盖一个预热epoch和一个乘积epoch。已有metrics的模型跳过；发现未完成模型目录则报错，先移走未完成目录再重跑，避免混入旧训练记录。

## 本轮协议

1. 外层沿用5折用户切分，每折外层训练用户中以 seed=2026+fold 固定抽取10%（向上取整）作为valid，余者为inner train。schema和物品质量先验只用inner train，不使用valid/test用户拟合。
2. 每个用户，包括valid/test，都用自己的已完成过去会话统计。时间条件为历史会话结束日期严格早于当前场开始日期；以场开始日期查询，使跨午夜场也固定。训练用户的更早会话总体先验提供m=5场平滑。若总体历史也尚不存在，数值回退为0。这个用户历史属于推断时已知行为，不是将测试目标交给优化器。
3. `elapsed_minutes = elapsed_end_s_DIAGNOSTIC/60`，即本屏最后一条timestamp减场首timestamp，再加本屏末条clipped播放量。遵循Leader的屏末观察口径；本轮不将任务叙述成屏前候选打分。`delta_minutes=1.0`，视频片长仅留给内容/Fusion特征。
4. 完全不建立slate_items/history_items的Embedding，tag保留。标签/历史构造仍可使用原始video_id做past-day物品质量先验。低基数tag使用原始字符串签名，v1已知的多tag字符串处理保持不变。
5. 两塔接收同一个字段集合；预算参数路径显式掩码时钟/序号、动态B3、B4和序列，只激活B1与场开始情境（hour/weekend/tab/之前当日消费）。这是反馈所要求的路由约束，不声称两塔激活相同特征。预算参数在固定checkpoint的同一场内不变。
6. 预算编码器与内容编码器独立持有、初始tensor完全相同。预算context和内容编码器的用户类别表不会共享可训练参数。Ours预热只更新budget_net及预算编码器；阶段2冻结这两部分，gamma.detach。内容头bias=+2.6，记录切换时概率比和valid loss跳变，以及预算hash和gamma输出在stage2前后是否精确一致。内容编码器从初始状态进入stage2，没有静默复制预热状态。
7. 四模型每次都构造同样模块；每折初始全state hash一致，数据物理顺序和batch完全一致。三epoch，batch4096，AdamW lr=1e-3，矩阵weight_decay=1e-6；Embedding/bias/一维参数零衰减。没有item-ID，所以不启用反馈中“若保留ID”的频次50、Embedding decay和Dropout分支。
8. 每epoch训练step的25%/50%/100%做全valid评估，最小valid LogLoss选择checkpoint，并列按User-GAUC。Ours先从第一完整epoch的三个检查点选warmup，再从该状态开始stage2；阶段2起点也评估候选，后续最多两个完整epoch。全量三epoch训练均执行，最终只测试选中的checkpoint一次。因为warmup可回选部分epoch，`optimizer_steps_executed`和选中checkpoint的有效更新谱系不同，见JSON。
9. 核心指标User-GAUC、Pooled-AUC、LogLoss；Intra-Step GAUC按精确req_idx聚合，附加按(user_active_degree,req_idx_bin)聚合。bin=1,2,3,4,5,6–10,11–20,21–50,>50；均样本数加权，单类组排除并报告覆盖。

## 数值和解释

固定budget参数时人工递增e可保证S非增。原始屏末公式在少量相邻日志上会回退；保留Leader原公式，不用cummax悄悄修改。诊断分开统计原时钟回退，以及时钟未回退但S上升的实现违规；后者必须为0。时钟回退不被用来拒绝本轮实验口径。

固定1分钟的生存比是预测y_continue的结构特征；`下一屏继续`标签和`再生存一分钟`不是定义上完全相同的事件，不将概率直接解释为已标定的分钟生存概率。

v2同时修复数据先验、去除item-ID、调整训练/选checkpoint、改变时钟和参数路由。与v1比较是修正版整体筛查，不把增益归因到其中一项。最终效果按实测汇报，不预先保证优于Fusion；后续消融需要单独排定。
