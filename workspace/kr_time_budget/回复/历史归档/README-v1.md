# KR 时间预算预实验

> 历史资料：保留原阶段口径；本页“当前”指写作时。现阶段结论请以[定稿报告](../14定稿/定稿实验报告.md)为准。

当前修正版：[Argon 反馈 v2](README-v2.md)。以下记录 v1 历史协议，供复现旧结果。

本目录执行 `Argon-plans/` 中当前阶段的 6.1：request 构建和四模型退出预测。框架核心不变；实验模型在 `workspace/kr_time_budget/pipeline/time_budget.py`。

阶段报告：[KR 时间预算实验与审计](20261116_kr-time-budget-argon-report.md)。本实验后续记录统一写入 `workspace/kr_time_budget/`。

## 运行

在项目根目录执行：

```bash
mkdir -p workspace/kr_time_budget/data/v1/logs
uv run python -u -m workspace.kr_time_budget.pipeline.build_requests > workspace/kr_time_budget/data/v1/logs/build.log 2>&1
mkdir -p workspace/kr_time_budget/runs/exit_v1
uv run python -u -m workspace.kr_time_budget.pipeline.run_exit > workspace/kr_time_budget/runs/exit_v1/run.log 2>&1
uv run python -u -m workspace.kr_time_budget.pipeline.audit > workspace/kr_time_budget/runs/exit_v1/audit.log 2>&1
uv run python -m pytest -q workspace/kr_time_budget/tests
```

数据目录和训练结果目录用 `--output` 指定；训练的数据根用 `--root` 指定。原始数据默认通过 rosen 的 `KuaiRandDataset.DATA_KUAI_1K` 定位，遵循 `ROSEN_DATASOURCE_ROOT`；`build_requests --source` 可直接指定 CSV 目录。重建数据须使用全新输出目录，并同步更换训练输出；不要混用旧 `feature_meta.json`。

只跑一个 batch 的链路检查：

```bash
uv run python -m workspace.kr_time_budget.pipeline.run_exit --folds 0 --epochs 1 --smoke --output workspace/kr_time_budget/runs/exit_smoke
```

`run_exit` 跳过已有 `metrics.json` 的完成模型。每个 epoch 保存 `last.pt`，但中断的模型默认从头重跑；它不声称从 optimizer 状态恢复。需重跑未完成模型时先移走其目录，防止 `epochs.jsonl` 混入两次训练。日志重定向不影响 checkpoint。

## 数据与边界

- 两份标准日志和一份随机探索日志合并。按被使用的原始列去重，稳定排序 `(user_id,time_ms,source_order,source_row)`；不是原始文件的逐字节去重。
- 相邻日志满足 `next_time_ms - (time_ms + clip(play_time_ms,0,300000)) > 30min` 时切场。
- 每用户首尾观测场排除，缓解观测窗口左右截断；这不能证明其他场均完整。
- 总交互数 `<6` 的场不产 request。其余从末尾倒切 M=6，头部余数进入首个 request 之前的历史。
- 模型时钟：当前列表第一条记录时间减本场第一条记录时间。文档末尾提出的“当前屏末条时间+播放时长”保留在 `elapsed_end_s_DIAGNOSTIC`，不作为输入。
- `w/R/G` 使用非负原始播放时长，单位秒；切场和完播诊断中的播放量上限为 300 秒。片长截为 3–600 秒；完播率截为 0–1。
- 当天前序会话统计只含已结束的前场。物品先验只用训练用户在请求日期之前的日志，按总体先验加 20 个伪计数平滑。
- 用户场均预算只用训练用户、此前已完成会话；v1 还保守滞后一个完整日（`prior_available_date=结束日期+1日`，asof 仅取更早日）。测试用户使用这些训练场的总体先验。
- 不读全月 `video_features_statistic_1k.csv`，避免全月反馈聚合进入当前输入。使用的用户画像和视频静态表没有历史快照时间，无法证明事前可用。
- 历史 ID/tag 取前两个 chunk（最多 12 条），不含当前 chunk；低频 item `<5` 映射 OOV。tag 的原始字符串作为类别签名，多样性也是签名数，未拆成多标签集合。
- `GroupKFold(5)` 按用户分组，所有会话跟随用户，无训练/测试用户交集。每折 schema 仅 train fit，train 固定物理 shuffle，test 保持顺序。没有另设 valid 或按测试选 epoch。
- 使用 `SchemaManager`、`SequenceSetting`、`ParquetStreamDataset` 和 `DataLoader(batch_size=None)`。训练数据经 rosen 转成 batch 后缓存到 CPU，四模型复用同一顺序。

## 模型与训练

- Content：Block 3+4；用户/时钟数值和用户/场景类别全置零。
- Budget：Block 1+2；列表总片长还作为生存公式的 `delta`，不包含内容评分。
- Fusion：X_all → 三层 MLP → sigmoid。
- Ours：同一个 X_all 同时给混合威布尔参数塔和内容塔，输出严格 `gamma * content_stay`。前 3 epoch 只训 gamma；后 5 epoch 训乘积，budget_net 学习率 1e-4，其他可训练参数 1e-3。共享编码器也继续学习，因此不能声称整条预算路径已冻结。
- 固定 8 epoch、batch=4096、AdamW；矩阵权重 decay=1e-6，Embedding/bias/一维参数零衰减；无 scheduler、无 EMA、无 early stop。所有 mode 构造同样模块并从同一 seed 初始化，逐模型初始 state hash 完全一致；未使用的 head 不参与梯度。
- 序列用 attention pooling；DNN 无 dropout。本轮是初始无正则筛查，不是 CAVE 的逐细节复现。
- 威布尔生存比在 log 域先相减，避免分别截断 S(e) 和 S(e+delta) 产生伪比值；scale 初始化乘 30，使时间尺度与分钟单位匹配。

GAUC 都按 eligible 组的 request 数加权；单类组排除，排除规模见 JSON。Session-GAUC 使用真实重建 session_id，不是 user-day。Pooled-AUC 使用全体 OOF 样本。

该标签只有每场末屏为0，因而无需训练的 `-req_idx` 就能在所有 eligible 会话上取得 Session-GAUC=1。这个指标不能单独支撑机制结论；审计同时给出正/负时间和序号的无训练排序对照。

## 结论边界

倒切用到了未来会话长度，六条逐条交互不是真实同时曝光的 slate。官方仅定义 time_ms 为 interaction timestamp，没有保证曝光开始语义。若按开始时间解释，大量相邻列表存在播放时间重叠，历史反馈是否已经可用无法确认。

所以当前产物为 **离线预测筛查**；线上/机制因果结论标记 `INVALID_FOR_CAUSAL_COMPARISON`。不能宣称“已证实零泄漏”。

预算参数随当前 X_all 变化，`S(e_t|X_t)` 不构成固定会话预算的统一累计生存曲线。`audit.py` 仅导出明确标为 heuristic 的截断逆生存分数和有效样本量；它不是已识别的 IPCW，也不是行为策略 propensity。随机探索标记不是完整 slate 的日志概率。

按 Argon 文档末尾“现阶段6.1”的范围，本轮不训练 6.2 Generator/6.3 LTV。w/R/G 标签已经保留。后续必须先明确时间戳/反馈可用性、真实 request 边界、目标策略及动作概率，并给出 CAVE uAUC/bAUC 的精确定义；不可预先保证 LTV 指标“全面胜出”。
