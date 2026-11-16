# KR 时间预算 v2：按 Argon 第三份反馈执行

> 历史资料：保留原阶段口径；本页“当前”指写作时。现阶段结论请以[定稿报告](../14定稿/定稿实验报告.md)为准。

依据：`Argon-plans/3反馈.md`。v1 数据和实验结果保持原样。v2 复用 v1 的原始 request/session/prior-event 重建产物，重新做折内先验、schema、valid 和模型训练；输出分别为 `workspace/kr_time_budget/data/v2/`、`workspace/kr_time_budget/runs/exit_v2/`。

## 本轮固定选择

- 屏末挂钟：`elapsed_end_s_DIAGNOSTIC / 60`，遵循用户/Leader 本轮实验口径；`delta=1.0` 分钟。
- 默认不建立/读取 item-ID Embedding，只保留 tag 序列、片长和训练用户过去日物品质量先验。本轮不运行可选的保留 ID 分支。
- 保留外层用户五折。从每个外层训练折确定性抽出10%用户作为 valid。模型/schema/物品先验/总体平滑先验只 fit 内层 train 用户。
- train/valid/test 用户均使用自身严格过去已完成会话。用户先验在场开始时固定，以会话结束日期 < 当前会话开始日期为界，训练总体先验加 m=5 场平滑；不再让测试用户丢掉自身历史。
- Budget 参数只由场开始固定背景决定：B1、开场时刻/场景、开场前当日消费。B2 中时钟/序号和全部动态 B3/B4、列表序列不进入参数网络。动态 B3 若进入参数也不保证单调，所以本轮使用更严格的场开始固定版本。
- Ours Stage1固定1完整epoch训练 gamma；Stage2最多2epoch训练乘积。内容头最后层 bias=+2.6。预算编码器独立持有（初始与内容编码器精确相同），Stage2预算编码器+budget_net均冻结且 gamma.detach，确保内容更新不漂移预算函数。仅 x.detach 不能阻止共享编码器参数变化，本轮不采用那个不充分的实现。
- 四模型最大3epoch、batch4096、lr1e-3、AdamW矩阵decay1e-6，Embedding/bias/一维参数零衰减；本轮无item-ID，因此不涉及反馈中可选ID分支的decay例外。
- 每epoch在25%/50%/100%训练步时做全valid评估，按valid LogLoss选checkpoint（并列按User-GAUC）。Ours阶段1只选择预热checkpoint，完整相乘模型只在Stage2选择；Stage2起点也评估，记录切换前后预测与loss变化。test仅对最终选定checkpoint评估一次。
- 核心指标User-GAUC、Pooled-AUC、LogLoss。补充按精确req_idx分组的Intra-Step GAUC，以及按(user_active_degree,req_idx_bin)分组指标。分组按样本数加权、单类组排除，报告覆盖率；它们是固定进度的预测诊断，不称因果识别。

## 验证和汇报

先测自身历史隔离、训练/valid/test用户交集、固定参数路由、生存率随人工时钟的单调性、Stage2预算输出不变、阶段切换drop、共有初始化和乘积。训练期间先汇报第一折valid结果与真实阶段切换指标，再完成五折。不能在结果前保证提升；v1-v2同时更改多项协议，比较属于修正版screening而非单变量归因。
