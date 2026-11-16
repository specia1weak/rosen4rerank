# KR v3：Argon 4 特征集执行结果

> 历史资料：保留原阶段口径；本页“当前”指写作时。现阶段结论请以[定稿报告](../14定稿/定稿实验报告.md)为准。

已按 `Argon-plans/4.特征集敲定.md` 构建54字段日期切分数据，并完成四模型 full train/full test。
**本轮相乘模型未超过 Fusion；所有模型 User-GAUC 还低于简单负时钟基线，不能仅凭整体 AUC 宣称方案有效。**

## 数据与训练口径

| 切分 | Request | 用户 | 日期 |
| --- | ---: | ---: | --- |
| train | 1,721,745 | 1000 | 20220408–20220506 |
| valid | 57,872 | 894 | 20220507 |
| test | 48,898 | 855 | 20220508 |

完整日志先切场、倒切M=6，再过滤训练范围Top5 Session场景；主tab取首条。完整保留在单个切分内的Session，跨边界268场单独排除。每用户首尾场排除；不足一屏头部余数参与历史。
训练先验只用严格过去日；验证/测试固定使用截至5月6日快照。用户场均先验m=5平滑，物品CTR/long_view先验α=20。正反馈为每视频任一like/comment/follow/forward。tag保留重复、按时间拼接、原生MultiSparse sum及head截断。
基线各1epoch；Ours预热1epoch、冻结整条预算路径后再1epoch。batch4096、AdamW lr=1e-3，普通矩阵decay=1e-6，Embedding/bias/一维参数零衰减。固定seed2026，无EMA、无scheduler。每epoch 25%/50%/100% full valid，以LogLoss选checkpoint，最后完整test一次。
使用原生BaseTrainer、EvaluatorManager、EarlyStopper、TwoStageRunner；四模型初始全state hash与train物理顺序一致。Ours多一个预热epoch且路由/目标不同，本轮属于结构筛查，不是训练量相同的单变量因果比较。

## 测试结果

| 模型 | Pooled AUC | User-GAUC | LogLoss | Intra-Step GAUC |
| --- | ---: | ---: | ---: | ---: |
| content | 0.628318 | 0.523503 | 0.239604 | 0.575942 |
| budget | 0.677942 | 0.376764 | 0.233359 | 0.616609 |
| fusion | 0.696984 | 0.484769 | 0.230444 | 0.635806 |
| ours | 0.691269 | 0.422550 | 0.231489 | 0.626844 |

训练继续率常数预测的测试LogLoss：0.246175。User-GAUC排除单类用户，覆盖823/855用户、48,855/48,898条Request，按有效样本数加权。Intra-Step按精确req_idx分组，也排除单类组，覆盖见JSON。

| 无训练排序诊断 | Pooled AUC | User-GAUC |
| --- | ---: | ---: |
| -req_idx | 0.354700 | 0.592362 |
| -elapsed_min | 0.433671 | 0.605780 |

负时钟在整体AUC和用户内排序上的方向不同，说明只看跨用户的Pooled AUC会掩盖用户内排序不足。该诊断无需训练，也没有用它重新选择模型。新日期切分与v2五折用户切分不同，不能把两版数字直接当成特征增益。

## 实现核对

- 完整54字段，无裸video/author/music ID Embedding；标签通过原生MultiSparse计算，不再将整串标签当单个类别。
- 测试预测与原始request_id/user_id/label逐条核对，48,898条完整且唯一；Session无跨切分，输入无空值。
- 三个单阶段基线实际各执行421步；Ours执行421+421步。所有本轮选中checkpoint都在对应阶段第421步，详见selection.json。
- 每条有效Request恰好6个事件，事件无重复归属；验证/测试用户都有自身过去场历史。测试slate事件中约61.48%没有过去日物品记录，质量先验回退为总体率。
- Ours第二阶段预算全部state与第一阶段snapshot逐tensor一致，probe gamma最大变化0；预测乘积误差0。
- 原始测试屏末时钟回退1,682次；时钟未回退却S上升的次数为0。未静默改时钟。
- 18项测试通过，包括过去日先验、前屏历史、标签拆分、参数独立所有权、同初始化和冻结验证。

Ours测试gamma均值0.934063，content_stay均值0.989912，stay>0.99占76.32%。内容乘子仍接近1；本次未证明冻结预算×内容残差有效。

## 产物与复现

- 数据：[manifest](../../data/v3/manifest.json)、[数据审计](../../data/v3/data_audit.json)。
- 代码：[预处理](../../pipeline/build_requests_v3.py)、[Schema](../../pipeline/kr_request_schema.py)。
- 代码：[模型](../../pipeline/time_budget_v3.py)、[训练](../../pipeline/run_exit_v3.py)。
- 训练：[全日志](../../runs/exit_v3/run.log)、[指标](../../runs/exit_v3/summary.json)、[审计](../../runs/exit_v3/audit.json)。
- 各模型：`workspace/kr_time_budget/runs/exit_v3/{content,budget,fusion,ours}/`；含checkpoint、预测、选择记录、分域结果。
- 数据日志：`workspace/kr_time_budget/data/v3/logs/build.log`；测试：同目录tests-all.log。
- 配置与命令：[复现说明](../6/复现说明.md)。
