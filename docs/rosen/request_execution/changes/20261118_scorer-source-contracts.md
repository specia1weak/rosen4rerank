# 打分来源元数据与 GAUC 输入契约修正

## 修正

- BaseScorer.num_candidates 默认返回 None，不跨来源猜测；PredScorer、TargetScorer、
  FieldScorer 仅声明各自原始张量的宽度，一维来源不回退其他字段。Target 使用成员查询，
  不依赖 Interaction.columns，支持无 target 和映射型来源。
- Resolver 仅为实际存在的预测构造默认 PredScorer，宽度查询和实际打分共用同一组来源。
- requires_target 补上正在消费的、声明了 target 的单 Tensor 预测/order/full_scores，
  使自定义标签来源也不能静默重命名模型任务。完全自定义且未使用的预测不触发无关冲突。
- PointWiseEvaluator 在槽位过滤/空 batch 返回之前检查 GAUC 用户字段与形状。
  SlotEvaluator 委托同一逻辑，无重复检查。合法全 mask batch 不累计数据。

## 验证

评估专项 127 passed，logs/evaluate_source_contracts.log。
新增复现：50 候选与未使用 6 维预测共存、单一来源宽度、无 columns 的来源映射、
无预测时不构造默认打分器、默认/显式 PredScorer 的任务冲突、完全自定义来源保持独立，
以及 PointWise/Slot 下缺失/错误/正确形状 users 的全 mask 与后续正常 batch。

完整回归记录在 logs/evaluate_source_contracts_regression.log。
契约文档及单文件源码导出更新；不运行训练、不修改模型。
