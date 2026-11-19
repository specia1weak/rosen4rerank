# 统一 N 选 K 的重排评估

## 背景与设计

用户要求以输入 N 候选、输出 K 位置（1≤K≤N）的业务契约，统一评估层重复实现。
本次沿用显式 Scorer 组装，不修改训练器、Dataset、EvaluatorManager、EvalBatch 或模型。

- scorer._orders 校验合法范围、长度和无重复，不再强制 K=N；固定槽位分数、标签和 mask
  均可按 order gather 成 [B,K] / [B,M,K]。
- ListNDCG 的 IDCG 来自整个 N 候选池，DCG 来自显式选择位置；分数 argsort 路径复用同一公式。
- SlotEvaluator 合并分类、校准、列表排序逻辑，支持 cb 兼容名、位置 relevance 和 use_order。
  use_order 仅支持 NDCG，不要求模型提供原位 scores。
- OrderEvaluator / LegacySlateEvaluator 变为 SlotEvaluator 薄封装，没有独立累加器和计算逻辑。
- SlateEvaluator(use_order=True) 将同一 order 传给 pred/value/ref scorer，并同步投射 mask。
- Lookup 全排列价值表明确拒绝 K<N；ModelScorer 委托模型评分，模型仍需自身支持相应宽度。
- 删除 KR 的 PriorEvaluator，直接注册 SlotEvaluator + FieldScorer；六个既有注册名及指标结构保留。
- 公共 __all__ 名单未改变。Prediction 仅补充 N/K 坐标说明，无输出字段/API 更改。

## 行为修正

旧 OrderEvaluator 先 scatter 成假分再排序，会将无效物品移到末尾，改变其他输出位置。
现在显式选择的无效槽位收益为 0 且占据原输出位置，符合用户给出的 DCG 公式。
对应一项旧测试的期望修正；全有效 N=N 场景与旧公式逐值一致。分数排序仍将无效候选放末尾。

## 验证

- 新增 10 选 4：分数/标签/mask 投射，整个候选池 IDCG，漏选好物品惩罚，
  order-only Prediction、Manager 路由、非法越界/重复/空选择，裁判委托、Lookup 拒绝部分选择。
- sum/mean/dcg 与 NCIS 的 selected mask 对齐；cb=0 与 pcoc=NaN 的零分母约定。
- KR 注册名与各指标键完整性检查；公开导出符号集合不变。
- 评估与 KR 专项：96 passed，logs/evaluate_n_choose_k.log。
- 完整回归：581 passed，logs/evaluate_n_choose_k_full_regression.log。
- 只读复算已保存 LC_MGE / DNN / DCN 各 48,898 屏预测，AUC/LogLoss/GAUC/CB 差值 0，
  NDCG 最大绝对差约 2.22e-16（批次累计浮点误差）。未训练、未重新推理。
  日志 logs/evaluate_n_choose_k_saved_predictions.log。

## 影响范围

框架契约文档与 API 更新，KR 旧 benchmark/early stopper 保持兼容。既有固定宽度模型
本身没有改成动态 N 选 K 生成器；本次实现的是评估输入、投射和指标的 N 选 K 支持。
