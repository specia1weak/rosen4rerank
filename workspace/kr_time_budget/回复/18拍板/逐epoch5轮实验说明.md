# 逐epoch训练与test观测：5轮固定上限

用户要求：逐epoch训练，每个epoch直接测test，第5轮结束停止。本轮沿用最终实验矩阵，不改变模型和数据定义。

## 固定协议

| 项目 | 本轮设置 |
| --- | --- |
| 数据 | 已有KR v3与train-fit Schema；不重新预处理 |
| 切分 | train截至2022-05-06，valid=05-07，test=05-08 |
| 请求数 | train 1,721,745 / valid 57,872 / test 48,898 |
| 配置 | single_step、single_step_pace、mixture_step、mixture_step_pace、fusion、fusion_pace |
| seed | 2026、2027、2028 |
| 训练 | 每配置/seed从头连续5epoch，421步/轮，共2105步 |
| 模型/特征 | 原KRExitModel及54字段路由；Ours屏数预算，pace仍为原屏末定义 |
| Loss | Ours一个乘积BCE同时更新两塔；Fusion普通BCE |
| 优化器 | 一个AdamW连续使用5轮；budget LR .005，content/Fusion LR .001 |
| Decay | 普通权重1e-6，Embedding、bias和一维参数0 |
| Batch | 4096，无梯度累积；原物理shuffle顺序每轮重复 |
| 其他 | CPU4线程，clip=10，无scheduler/EMA/预热/冻结/辅助Loss |
| 评估时刻 | 每epoch结束，直接使用当前轮末模型预测完整valid/test |
| 停止 | 第5轮固定停止，不以test指标或patience提前结束 |
| 选点 | test不选点；另记录轮末valid最低LogLoss所在轮，但不回滚权重 |
| 指标 | Pooled AUC、LogLoss、request-weighted User-AUC、Intra-Step GAUC及分域结果 |

这是一轮新的观测协议，区别于此前“预算内valid最优checkpoint”。不能用第13份的best checkpoint加上第3轮末Adam状态拼接续训；其模型和优化器可能不是同一时点。本轮从头跑，逐轮保留对齐的模型、Adam moments及RNG状态。

## 运行与产物

[训练入口](../../pipeline/run_epochwise_exit.py)复用原生BaseTrainer的train_step；只改变epoch末评估调度。两项回归测试比较有无逐轮观测，以及反转test标签后的训练轨迹，均要求5轮末模型及Adam状态完全一致。

独立输出：[runs/epochwise5](../../runs/epochwise5)。[训练日志](../../runs/epochwise5/logs/train.log)、[配置](../../runs/epochwise5/config.json)、[覆盖](../../runs/epochwise5/coverage.json)。

每条轨迹在`<seed>/<config>/epoch_01`到`epoch_05`存储checkpoint、valid/test预测与指标；`epochs.json`实时更新已完成各轮结果，`status.json`记录进度。总summary在每条完整轨迹结束时更新。

新增检查：每次评估前后模型hash、Adam状态hash、Torch RNG不变；第1/3轮末hash与历史plan13对应terminal hash逐字节比对。只读报告入口[report_epochwise_exit.py](../../pipeline/report_epochwise_exit.py)重新核对完整预测覆盖和全部指标，不重新运行模型。

本轮不把反复观察test当成新的独立证据，也不改变屏末pace与离线分屏的解释边界。下游W/M不是本轮训练内容。
