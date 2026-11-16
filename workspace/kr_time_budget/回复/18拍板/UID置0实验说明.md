# UID置0配对消融：5轮逐epoch

用户要求移除uid或置0，选择改动最快的方法重新实验。本轮选择在内存batch中将编码后的`user_id`统一置0，不修改原始parquet或重新拟合Schema。

## 唯一输入变化

`user_id`的Setting启用`padding_zero`，两套Omni中对应Embedding均为`padding_idx=0`且第0行为零向量。因此置0消除UID表示，不是让所有人共用一个可学习的用户向量。预算塔、内容塔及Fusion全部使用同一屏蔽后的输入。

`user_id_raw`保留用于User-AUC；其余53个模型字段、审计字段、标签、顺序完全保留。画像、历史时长与屏数先验仍可表达用户差异，本轮不声称移除了所有用户信息。

模型维度和UID表形状保留，以便完整初始模型hash与原对照逐字节一致；UID表在零衰减组，置0后不产生UID梯度，逐轮checkpoint核对其未更新。

屏蔽发生在batch预处理。重载本轮checkpoint后仍需调用`zero_uid_inputs`或等价置0处理，不能直接拿保留的原UID表对真实UID推理，并将其当成本轮模型输出。

## 沿用协议

- 对照：[原UID逐epoch5轮实验](逐epoch5轮实验报告.md)，复用已完成结果，不重跑。
- KR v3，train截至2022-05-06，valid=05-07，test=05-08；请求数1,721,745 / 57,872 / 48,898。train-fit Schema及训练物理顺序不变。
- single_step、single_step_pace、mixture_step、mixture_step_pace、fusion、fusion_pace；seed2026/2027/2028。
- 从头连续5epoch，每轮421步，共2105步，batch4096，无梯度累积。轮间不回滚模型、不重置Adam。
- AdamW预算LR .005，内容/Fusion LR .001；普通权重decay1e-6，Embedding/bias/一维参数零decay；clip10，CPU4线程，无scheduler/EMA。
- Ours仍为一个乘积BCE同时更新两塔，无辅助Loss、预热、冻结。
- 每轮末直接评估完整valid和test，第5轮固定停止，test只观测；不与预算内valid最优checkpoint混比。
- 输出Pooled AUC、LogLoss、request-weighted User-AUC、Intra-Step GAUC、分域结果，并计算同seed/配置/epoch的置0减原UID差值。

## 验证和证据

6个配置的测试均验证两塔/Fusion视图UID向量为0、原batch不被改写、raw UID仍可分组、改变raw UID不影响预测、UID梯度为0而活跃head可训练。

运行时逐张量核对所有非UID输入不变；逐配置核对原对照初始模型hash及optimizer groups。每轮评估前后模型、Adam状态和Torch RNG相同；checkpoint逐轮核对UID两套表未更新。完成后只读扫描预测，重新计算指标并与旧实验逐epoch配对。

[代码](../../pipeline/run_uid_zero.py) · [报告生成](../../pipeline/report_uid_zero.py) · [运行目录](../../runs/epochwise5_uid_zero) · [训练日志](../../runs/epochwise5_uid_zero/logs/train.log) · [输入审计](../../runs/epochwise5_uid_zero/input_audit.json)。

报告归入本目录，作为第18份后的补充特征消融，不创建尚不存在的第19份计划编号。
