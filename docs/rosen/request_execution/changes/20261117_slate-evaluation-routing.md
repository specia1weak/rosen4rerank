# 20261117 多任务与列表评估路由（Step 3）

## 背景与问题定位

原 BaseTrainer 强制读取默认 UID 和标签，阻塞无样本级 UID、多标签和字典预测模型。
EvaluatorManager 只能按 domain 过滤，无法显式选择预测头和监督字段；原 TopKEvaluator 是
全库单正例召回评估，不能直接解释 `[B,K]` 列表标签。

## 设计决策

- Trainer 只对默认字段做安全读取，不存在时传 None；预测 Tensor 或字典原样透传，完整
  Interaction 保留。student 保留 predict_step 扩展点，EMA 仍调用实际 teacher.predict。
- register 保留原三个位置参数，新增关键字 pred_key、label_field、mask_field、uid_field。
  label/UID 未指定时沿用 collect 默认参数，mask 必须显式指定，不从 SlateLabel 自动推导。
  显式字段或预测键缺失会报错，防止静默用错监督。
- Manager 只解析路由和行过滤，对所有预测 Tensor 及全库打分分别切片，后续 evaluator
  不会复用前一个 evaluator 的局部切片。没有 pred_key 时保留完整字典。
- 内置 evaluator 增加 collect_batch，完整接收过滤后的 Interaction、预测及解析后的上下文。
  自定义 evaluator 也可以实现该接口，不要求继承或使用预设指标。旧四参数 collect 继续兼容；
  自定义旧接口使用 mask_field 时明确提示实现 collect_batch，避免丢弃 mask。
- PointWiseEvaluator 自己选择展平语义：同形 `[B]` / `[B,K]` 预测和标签按 mask>0 过滤，
  无 mask 则全有效；行 UID 广播到槽位。有效位为空时跳过。AUC/LogLoss 不要求 UID，
  GAUC 缺 UID 时明确报错。没有 pred_key 的预测字典报错。
- SlateEvaluator 对列表单独计算 NDCG、CB，可组合原有 point-wise 指标和生成器候选集指标。
  LogDecorator 转发 batch 收集，日志和原 EarlyStopper 的消费方式保持不变。
- 不新增 CustomEvalStepProtocol，不修改模型的 predict / calculate_loss，不实现重排模型。

## 指标口径

- 概率指标接收概率。AUC/LogLoss/GAUC 复用原 METRIC_REGISTRY，不改其单一类别等边界行为。
- NDCG 增益 `2**rel-1`、折损 `1/log2(rank+1)`，masked 槽位排序在最后且增益归零。
  IDCG=0 的行记 0；全部无效的行不计入均值；分数相同时保持原顺序。按有效行数累计，
  不平均各 batch 的均值。新增 list_ndcg 注册项，不改变原全库 ndcg 实现。
- relevance 默认来自标签，可显式 ndcg_gain_source="position" 使用 `[K-1,...,0]`。
  这只兼容参考脚本的 relevance 定义，不宣称复刻原脚本最后一批等其他聚合行为。
- CB 是全评估集有效槽位预测总和/标签总和；标签和为零时记 0。
- ctr_topk / ctr_beamk / ctr_randk / ctr_all 是 `[B,M,K]` 的候选概率。沿 K 求均值，普通
  指标对全部候选平均；best 指标默认每个样本取最佳候选后再平均。可用 pred_key_map 换键。
- best_mode="batch_max" 显式提供每次 collect 全 batch 取最大值再按 collect 次数平均的
  兼容模式。该模式会受 batch 边界和 domain 过滤影响；标准模式按样本累计，不受尾批大小影响。
- 原 slate 的 mask 不用于候选排列，因为位置可能变化；候选分数由模型对齐处理，仍按全部 K
  求均值。这些是模型估计的候选收益，不是实测线上收益。

## 影响范围与兼容性

修改 BaseTrainer 的默认字段读取和 evaluate 包；不修改 schema、数据、MSR、EMA controller
或 EarlyStopper。既有默认 1D Tensor 预测、四参数 collect、domain 过滤、全库 top-k 和
日志装饰器继续工作。保留无 mask 时旧 Evaluator.collect 重载，不强塞新增关键字参数。

使用新多头或 mask 功能的脚本应显式注册路由；无 mask 的二维 point-wise 标签按所有槽位
有效处理。既有数据/日志不需迁移，历史模型和实验无需重训。

## 验证

新增 `tests/rosen/evaluate/test_slate_evaluator_and_routing.py`，共 27 个参数化用例：

- 多头多标签、错误键/缺 mask、缺 UID 的 GAUC、UID 两种布局、DomainFilter 与槽位 mask
  组合、空过滤、不同注册项对全库打分独立切片。
- NDCG/CB/AUC/LogLoss/GAUC 的精确数值；位置 relevance、分数 tie、零 relevance。
- 四类生成器的普通/best 指标、自定义键与 batch_max；不同大小 batch 的标准聚合一致。
- 完整 batch 的自定义 evaluator、旧接口对象、旧 collect 重载、LogDecorator、全库 HR/NDCG。
- BaseTrainer 与 EMATrainer 的字典预测训练/评估/早停；默认标签缺列或未配置及无 UID。
  改变 teacher 权重后验证其指标与 student 区分；直接验证 Slate NDCG 可供早停使用。

全量 `uv run pytest -q`：**270 passed，381 warnings，5.40 秒**，含此前 243 个测试。
compileall、`git diff --check` 通过。全部使用合成张量，不产生数据集或模型效果结论。
本地日志：[专项及 Trainer 回归](../../../../logs/slate_evaluation/targeted-tests.log)、
[全量回归](../../../../logs/slate_evaluation/pytest.log)。

## 后续可修改点

多次前向、额外采样、外部环境的评估执行仍可在 Trainer 既有扩展点实现；普通自定义指标优先
使用 collect_batch，不扩展 Manager 的预设指标分支。候选自身 mask、其他收益估计和对照口径
可独立扩展。Step 4 的 RerankModel / LC_MGE 尚未开始，不能将本步视作具体模型复现完成。

公共用法与完整输入/聚合契约见 [EVALUATE](../../details/EVALUATE.md)。
