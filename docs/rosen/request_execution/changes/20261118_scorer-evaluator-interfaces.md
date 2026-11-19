# 打分器与评估器抽象接口

用户将重排评价拆成槽位打分、排列探索和整刷价值判断，本阶段明确只实现一个
BaseScorer、五个打分器，以及 SlotEvaluator / SlateEvaluator 两个抽象接口。

## 设计与范围

- scorer.py 新增 Pred / Target / Field / Lookup / Model 五个单职责子类。
- SlotEvaluator / SlateEvaluator 继承 BaseEvaluator；保留 collect/summary/clear 抽象契约，
  整刷接口额外声明 reduce_slate。暂不实现新指标、探索评估或交付表导出。
- BaseTrainer、Dataset、EvaluatorManager、EvalBatch 无修改。
- ModelScorer 所需 score_permutations 放在模型侧，评估模块不推断特征列名。
  当前字段/先验/mask 重排，历史固定；MultiGE 用 OCPM 和固定上游先验给候选打分。
- 原具体 SlateEvaluator 更名 LegacySlateEvaluator，旧实验与相关测试显式改用旧类，
  OrderEvaluator 内部继续使用旧实现。本阶段没有切换历史实验指标。
- 外部调用方若实例化原 SlateEvaluator，需迁移到 LegacySlateEvaluator；新名称是抽象接口。

## 约定

[B,K] 是单排列；[B,M,K] 是候选轴。共享候选可用 [1,M,K]，避免 B=M 的二维歧义。
Lookup 按 ID 查表，候选标量严格使用字典序 K! 编号；不从原序值推导新排列价值。
ModelScorer 冻结传入模型并在 no_grad/eval 下运行，不参与训练。

## 验证

专项测试覆盖 720 全排列与子集查表、大整数 request_id、过滤后的 ID 对齐、非法排列、
缺失数据、真实模型候选重新打分、冻结推理，以及抽象接口子类的 Manager 注册。
相关评估/重排/KR 接口测试 126 passed；日志 logs/evaluate_scorer_interfaces.log。
核心回归 tests/rosen 为 380 passed；日志 logs/evaluate_scorer_full_regression.log。

后续按用户指示增加具体指标和探索评估，不在本阶段自动执行完整方案或恢复实验。
