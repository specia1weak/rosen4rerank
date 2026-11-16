# KR 时间预算 v2：Argon 反馈执行结果

> 历史资料：保留原阶段口径；本页“当前”指写作时。现阶段结论请以[定稿报告](../14定稿/定稿实验报告.md)为准。

依据：[3反馈.md](../../../../Argon-plans/3反馈.md)。本轮按用户/Leader口径执行屏末挂钟与固定1分钟步长，纠正v1的测试用户先验失配、多epoch高基数ID过拟合风险、两阶段冲击和预算编码器漂移。**结论：相乘模型的User-GAUC仍低于Fusion，本轮没有证实冻结预算×内容残差的优势。**

## 五折全量结果

1000用户，118,041场，1,856,774条request；外层五折用户分组，每条request仅在其测试折预测一次。每折从外层训练用户中留出10%作为valid，schema和全局/物品先验只用inner train。三个完整epoch均执行，按valid LogLoss选择checkpoint，仅测试选中模型。没有用test调参。

| 模型 | User-GAUC ↑ | Pooled-AUC ↑ | LogLoss ↓ | Intra-Step GAUC ↑ |
|---|---:|---:|---:|---:|
| Content / tag-only | 0.550325 | 0.614331 | 0.231358 | 0.573480 |
| Budget / fixed context | 0.531130 | 0.582748 | 0.234773 | 0.495910 |
| Fusion / tag-only | 0.580714 | 0.636553 | 0.231407 | 0.556605 |
| Survival × Content | 0.544476 | 0.602151 | 0.233915 | 0.515079 |

仅使用各折inner train继续比例的常数预测，合并OOF LogLoss=0.236691。

Intra-Step GAUC按精确req_idx分组、样本数加权，排除单类组。Ours覆盖 1,846,855/1,856,774 request、355/878 组。额外的(user_active_degree,req_idx_bin) GAUC=0.520477，详见JSON。它们是固定进度的预测诊断，不能单靠分桶宣称机制识别。

Ours−Fusion User-GAUC差值 -0.036238；以用户为单位、2000次配对bootstrap的95%区间 [-0.040060, -0.032419]。这是单seed预测差异，不是机制因果归因。

## 相较v1

| 模型 | v1 User-GAUC | v2 User-GAUC | v1 LogLoss | v2 LogLoss |
|---|---:|---:|---:|---:|
| Content / tag-only | 0.512893 | 0.550325 | 1.054437 | 0.231358 |
| Budget / fixed context | 0.553237 | 0.531130 | 0.247206 | 0.234773 |
| Fusion / tag-only | 0.513803 | 0.580714 | 1.035535 | 0.231407 |
| Survival × Content | 0.502992 | 0.544476 | 0.759975 | 0.233915 |

v2同时修改多项已指定协议，并新增inner valid；不能把总体变化全部归因于去除item-ID，或把v1失败解释为时间预算思路失败。v1仅用训练用户历史兜底测试用户，产生了不必要的先验失配；本轮所有用户都使用自身严格过去的已完成会话，再以训练总体先验m=5平滑。测试用户的行为历史是当前时刻已知信息，与将未来监督目标放入优化器不同。

## 六条修改的落实

1. 不建立slate_items/history_items的Embedding，仅tag、片长、过去日物品质量先验；本轮选择反馈允许的默认去ID方案，没有运行可选保留ID/频次50/dropout分支。
2. 自身会话历史对train/valid/test一致，查询时刻固定到当前场开始日期。历史会话结束日期严格早于当前场开始日期；不再额外滞后一天。train总体过去日先验以5场伪计数平滑。
3. 预算参数只看场开始固定背景。B2时钟/序号、动态B3、B4及序列均掩码；保留开场hour/weekend/tab/前序当日消费。仅掩码B2/B4而允许B3动态变化仍不能保证固定分布，本轮选择严格静态版本。
4. elapsed使用本屏末条timestamp减场首timestamp，加本屏末条clipped播放量；delta=1分钟。片长留在内容/Fusion输入。当前实验为屏末观察协议，不再套用v1的屏前时钟解释。
5. 内容头bias=+2.6。Stage1只训练gamma和独立预算编码器，固定一个完整epoch；从该epoch的valid检查点选warmup，然后Stage2冻结预算编码器和budget_net，并gamma.detach，训练两完整epoch内容残差。
6. 每epoch的25%/50%/100%做full valid。核心指标User-GAUC/Pooled-AUC/LogLoss，增加精确进度Intra-Step和活跃度×进度分桶GAUC，不再把退化的Session-GAUC当主要证据。

## 实现验证与机制诊断

- 每折四模型初始全state hash完全一致，同一折train物理顺序、batch4096、最大3epoch一致。AdamW lr=1e-3，矩阵decay=1e-6，Embedding/bias/一维参数零衰减；本轮无item-ID，不适用保留ID分支的Embedding衰减例外。
- 两塔字段接口一致，预算参数路径按反馈约束显式路由，激活字段不同。预算与内容编码器初始tensor一致、可训练参数独立。Stage2后预算hash和probe gamma逐元素精确不变；所有五折通过。
- 切换时乘积/gamma均值比范围 0.930762–0.932984，不再接近0.5；预算输出无训练漂移；概率乘积误差为0。
- OOF gamma均值 0.933043，content_stay均值 0.990737，stay>0.99比例 81.04%，gamma−乘积均值 0.008580。这显示内容乘子趋近1，冻结后对预算预测的修正较小。预热gamma已经直接拟合完整继续标签，而乘子只能下调gamma；这是需要下一步检查的约束，不预先断定因果。
- 固定参数时，人工递增e的S单调测试通过；全量OOF中“原时钟未回退但S上升”的次数为0。屏末原公式在少量相邻chunk上会回退；按要求原样保留并单独统计，不用cummax悄悄改口径。
- 原始KR约束和v1观测保留为背景，本轮不以未经对齐的时间戳推断拒绝Leader的实验定义。固定1分钟生存比作为继续预测的归纳偏置，不将下一屏标签定义等同于已校准的1分钟生存事件。

## Checkpoint与更新谱系

| 折 | 模型 | 选中epoch /训练步 | valid LogLoss | test User-GAUC | test LogLoss |
|---|---|---:|---:|---:|---:|
| 0 | content | 3 / 972 | 0.236601 | 0.550452 | 0.229227 |
| 0 | budget | 1 / 81 | 0.239738 | 0.536355 | 0.232309 |
| 0 | fusion | 2 / 486 | 0.240254 | 0.581251 | 0.232227 |
| 0 | ours | 2 / 486 | 0.239747 | 0.536937 | 0.232344 |
| 1 | content | 3 / 987 | 0.239328 | 0.548997 | 0.233811 |
| 1 | budget | 1 / 83 | 0.242719 | 0.531543 | 0.236520 |
| 1 | fusion | 3 / 987 | 0.239297 | 0.579488 | 0.230724 |
| 1 | ours | 3 / 823 | 0.241800 | 0.551151 | 0.235379 |
| 2 | content | 2 / 413 | 0.250425 | 0.543497 | 0.230225 |
| 2 | budget | 1 / 165 | 0.249775 | 0.530070 | 0.232790 |
| 2 | fusion | 3 / 990 | 0.246296 | 0.572907 | 0.229692 |
| 2 | ours | 3 / 825 | 0.248218 | 0.547002 | 0.232558 |
| 3 | content | 3 / 993 | 0.234457 | 0.558591 | 0.228896 |
| 3 | budget | 2 / 414 | 0.236079 | 0.529616 | 0.235975 |
| 3 | fusion | 3 / 993 | 0.232021 | 0.588330 | 0.229257 |
| 3 | ours | 3 / 993 | 0.236368 | 0.531350 | 0.233020 |
| 4 | content | 3 / 741 | 0.229196 | 0.550086 | 0.234634 |
| 4 | budget | 1 / 83 | 0.233812 | 0.528067 | 0.236270 |
| 4 | fusion | 3 / 987 | 0.229576 | 0.581595 | 0.235134 |
| 4 | ours | 3 / 987 | 0.231880 | 0.555940 | 0.236272 |

选中epoch内的检查点可能为0.25/0.5epoch，不能一概称训练满3epoch的模型。Ours Stage1虽执行完整1epoch，但回选最佳warmup检查点再开始Stage2；日志的step为实际执行步编号，实际保留谱系为warmup选中步数+已选Stage2更新步数。Stage2起点也参与候选，避免将内容未适应的初始乘积强制作为改进。所有最终测试都加载本run的best.pt，checkpoint含stage和valid指标。

## 下一步建议

先与用户讨论两个有区分力的对照：预算预热是否需要给内容留出可学习空间；预算参数只看场初背景与允许历史状态更新各回答什么问题。后一方案不应继续声称跨步S属于固定分布。若要确认ID过拟合的归因，再单独做同协议tag-only与保留频次≥50 ID+指定正则的配对消融；不从本轮多修复结果直接下因果结论。

## 产物

- 方案：`workspace/kr_time_budget/history/20261116_kr-time-budget-v2.md`
- 说明：`workspace/kr_time_budget/history/README-v2.md`
- 数据/折内schema：`workspace/kr_time_budget/data/v2/fold_*/`
- 模型：`workspace/kr_time_budget/pipeline/time_budget_v2.py`
- 训练入口：`workspace/kr_time_budget/pipeline/run_exit_v2.py`
- 全日志：`workspace/kr_time_budget/runs/exit_v2/run.log`
- 指标/审计：`workspace/kr_time_budget/runs/exit_v2/summary.json`、`audit.json`
- OOF/checkpoint/valid记录：`workspace/kr_time_budget/runs/exit_v2/fold_*/{content,budget,fusion,ours}/`
- 测试日志：`workspace/kr_time_budget/data/v2/logs/tests.log`

本实验后续代码、报告与产物统一放workspace/kr_time_budget。
