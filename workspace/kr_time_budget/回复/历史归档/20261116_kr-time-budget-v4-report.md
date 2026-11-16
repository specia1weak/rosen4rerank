# KR v4：Argon 5 反馈执行报告

> 历史资料：保留原阶段口径；本页“当前”指写作时。现阶段结论请以[定稿报告](../14定稿/定稿实验报告.md)为准。

**两处修改均已落实：相乘模型的AUC/LogLoss略优于Fusion，但User-GAUC仍低于Fusion。内容饱和显著缓解；反馈所预测的User-GAUC达到0.61+没有实现。**

依据：[5反馈 5.md](<../../../../Argon-plans/5反馈 5.md>)。本次复用原v3数据，不重新预处理、不改特征/切分；旧报告见 [v3执行报告](20261116_kr-time-budget-v3-report.md)。

## 本次改动

1. 预算改为单威布尔：scale=softplus(raw_scale)×30+0.5，shape=clamp(1.05+softplus(raw_shape),max=5)。原始输出从6维改2维。累计风险差用float64与expm1/log1p稳定计算，避免大数相减造成gamma回弹。
2. Budget模式及Ours预热均预测gamma×s0，s0=sigmoid(2.6)=0.9308615923。进入Ours第二阶段时，将未参与预热的内容头最后一层置零权重、bias=2.6，冻结预算分支。
3. Content/Fusion初始化和训练完全保留v3；零头只在Ours阶段转换时设置。四模型同名共有参数与v3逐tensor核对（排除预算末层），Content/Fusion完整测试预测与v3逐值相同，最大误差0。
4. 沿用rosen BaseTrainer、EarlyStopper、TwoStageRunner；新增全valid零跳变检查、gamma单调断言和参数分布。所有改动位于本workspace。

## 数据与训练

KR 1K；train=20220408–20220506，1,721,745条/1000用户；valid=20220507，57,872条/894用户；test=20220508，48,898条/855用户。沿用54字段、Session整场不交叉、训练过去日先验、评估0506快照。数据：[manifest](../../data/v3/manifest.json)。
基线各1epoch、Ours预热1epoch+内容1epoch；batch4096，seed2026，AdamW lr=1e-3，普通矩阵weight_decay=1e-6，Embedding/bias/一维参数零衰减；无scheduler/EMA/累积。每epoch 25%/50%/100%完整valid，以LogLoss选checkpoint；Ours还比较stage2起点。最终只评选中checkpoint的完整test。基线执行421步，Ours421+421步，本轮选中均为阶段第421步。
本轮同数据、同顺序、共有初始化相同，但同时修改IFR、预热目标和阶段衔接，是修正版结构对照；不能将变化单独归因给其中一项。Ours比单阶段模型多一轮训练，不宣称等训练量因果比较。

## 完整测试结果

| 模型 | v3 AUC | v4 AUC | v3 User-GAUC | v4 User-GAUC | v3 LogLoss | v4 LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| content | 0.628318 | 0.628318 | 0.523503 | 0.523503 | 0.239604 | 0.239604 |
| budget | 0.677942 | 0.630923 | 0.376764 | 0.591388 | 0.233359 | 0.245839 |
| fusion | 0.696984 | 0.696984 | 0.484769 | 0.484769 | 0.230444 | 0.230444 |
| ours | 0.691269 | 0.697785 | 0.422550 | 0.473160 | 0.231489 | 0.229861 |

v4 Ours−Fusion：AUC +0.000801，LogLoss -0.000582；User-GAUC -0.011609，用户配对bootstrap 95%区间 [-0.018061, -0.006309]。单seed不作机制因果结论。

| 模型 | v4 Intra-Step GAUC |
| --- | ---: |
| content | 0.575942 |
| budget | 0.614969 |
| fusion | 0.635806 |
| ours | 0.635881 |

User-GAUC为用户内AUC、有效样本数加权，排除单类用户；覆盖823/855用户、48,855/48,898条。Intra-Step按精确req_idx分组，覆盖131组、46,797条。分域指标在summary.json。
无训练排序诊断沿用同一个test：-elapsed_min User-GAUC=0.605780，-req_idx=0.592362。Budget达到0.591388，仍未超过负时钟；Ours为0.473160，仍未解决最终预测的用户内排序。

## 机制核对

- 原v3并非只是假设shape<1：Ours两个分量shape<1的样本分量占比实测为98.98%。其测试中43,935对时间不回退的相邻Request里，gamma上升40,880次。该证据支持检查递减风险率，但不证明所有排序差异都由它造成。
- v4在同一43,935对相邻Request上，S上升0次、gamma上升0次；原时钟回退仍为1,682次，未改公式。另以256个固定背景、0/1/5/10/20/60/120分钟网格核对gamma，无违规。
- 完整57,872条valid在预热最优状态→stage2第0步的预测逐样本精确一致；最大误差0、LogLoss差0。第二阶段结束预算全部state与snapshot逐tensor相同，probe gamma漂移0。乘积误差0。
- Ours测试stay均值：v3 0.989912 → v4 0.940802；stay>0.99：76.32% → 3.06%；stay<0.85：5.27%。已释放上下调节空间。
- v4测试gamma均值0.995457；Budget预测均值0.926632，Ours预测均值0.936568，真实测试继续率0.932901。
- 22项相关测试通过；包括数值公式、极端时间、IFR双单调、零跳变、冻结、同初始化和原数据协议。

## 与反馈预测不同的地方

s0=0.930862低于训练继续率0.936478，因此预热输出gamma×s0≤s0，存在显式上限。实际gamma升到约0.99546，而不是反馈估计的0.97～0.98。我们没有静默更换s0或提前调参。该约束使预热倾向提高gamma；不能把这个gamma直接视为已识别的纯时间预算概率。
IFR保证固定Session背景下gamma随时间非增，不保证同一用户跨不同Session的排序；跨场参数不同，右塔也可改变最终排序。v4 Ours时间不回退的相邻Request中，最终乘积仍上升24,723次，因为内容特征随屏改变。这符合当前模型定义，不是gamma单调实现失败。
Pooled与User-GAUC方向差异、IFR约束以及单侧乘积限制都值得诊断，但不能据此推导改IFR后User-GAUC必为1−旧值。当前数据不支持“已彻底消除过拟合”或“保证打穿全部指标”的结论。

## 入口

- [v4模型](../../pipeline/time_budget_v4.py)、[v4训练](../../pipeline/run_exit_v4.py)、[v4报告生成](../../pipeline/report_v4.py)、[v4测试](../../tests/test_v4.py)。
- [全日志](../../runs/exit_v4/run.log)、[汇总指标](../../runs/exit_v4/summary.json)、[审计](../../runs/exit_v4/audit.json)、[机制核对](../../runs/exit_v4/mechanism_audit.json)、[零跳变与冻结](../../runs/exit_v4/ours/stage_switch.json)。
- [复现说明](../6/复现说明.md)、[总入口](整理前总入口.md)。
