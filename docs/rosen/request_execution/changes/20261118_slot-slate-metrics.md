# SlotEvaluator / SlateEvaluator 指标实现

继打分器与抽象接口之后，用户要求为这两个评估器添加具体指标。本次直接实现已有方案
中的两组指标，不扩展到 ExplorationEvaluator，不切换 KR 入口，不运行实验训练。

## 设计与行为

- SlotEvaluator：auc/logloss/gauc、pcoc、ndcg@k、in_slate_auc。
  分类指标复用 PointWiseEvaluator；行内 AUC 使用排序与并列平均秩，仅对有效双类行平均。
- SlateEvaluator：r_auc/slate_auc、pcoc、rmse/mae、ncis。
  原序 scorer 可返回 [B] 或 [B,K]；sum/mean/dcg 将有效槽位聚合为整刷标量。
- R-AUC 由 Kendall 排序/Fenwick 计数与 tie 桶数还原，O(N log N) 时间、O(N) 空间；
  预测并列计半分，真实价值相等的 pair 不入分母，不构造 [N,N]。
- 整刷标量全量累计后计算，不平均 batch AUC；PCOC 为全量预测和除真实和。
- NCIS 强制显式 ref_scorer，执行所给 capped ratio 公式；普通预测值比不冒充真实策略概率。
- 有效区域数据要求有限；全 mask 行跳过，padding NaN/哨兵不参与。
  0 分母 PCOC、无合法行 in_slate_auc、无可比较 pair R-AUC、单类 slate_auc、
  零总权重 NCIS 返回 NaN，空评估返回 {}。
- LogLoss 显式声明二分类 labels=[0,1]，修复全量标签仅一类时无法计算的问题。
- BaseEvaluator 仍是抽象基类，两个新评估器已可直接实例化，也可继承扩展。
  Scorer、EvalBatch、EvaluatorManager、Dataset、Trainer 与模型代码本轮无修改。

## 验证

独立手算/NumPy/sklearn 对照：槽位和整刷公式、mask 与无效行、并列分、
全量和拆 batch 一致、自定义 Scorer 覆盖默认视图、clear/idempotent summary、
R-AUC 随机 ties pair 穷举及 10 万行排序计算、NCIS cap/零权重/零 reference。

- 评估专项：67 passed，logs/evaluate_metric_implementation.log。
- 核心与 KR 接口回归：400 passed，logs/evaluate_metrics_regression.log。
- git diff --check 通过。

文档：docs/rosen/details/EVALUATE.md 与 API_REFERENCE.md 更新为当前可实例化接口。
