# 评估解析前置与兼容类清理

## 背景

用户要求彻底消除 Manager 与 Evaluator 的双重解析，不保留旧包装类，并使 EvalBatch
成为只含最终张量的 Resolved View。上轮 Scorer 构造参数与 use_order 均从评价器移出。

## 最终设计

Interaction + Prediction → Manager 行过滤 → ScoreContext → BatchResolver → EvalBatch → Evaluator。

- EvalBatch 的 targets 必需，其他字段为 scores/ref_scores/order/mask/users/full_scores；无 pred/inter。
- ScoreContext 仅位于解析层；五个 Scorer 均接受该来源，构造参数交给 BatchResolver。
- BatchResolver 显式选择 ranking/candidates/slate 视图。ranking 保留 N 候选及 order；
  candidates 评价输入分数并忽略 order；slate 调用候选 Scorer 并同步投射 mask/二维 users。
- 每个注册项在行过滤后执行一次 Resolver；自定义目标不强制通过无关 schema 多标签推断。
- evaluator.py 仅包含 BaseEvaluator 与 PointWise/TopK/Slot/Slate；从重构前 383 行到 252 行，
  减少约 34.2%。无 Scorer、use_order、replace、order 投射和原始来源访问。
- 删除 Evaluator、LegacySlateEvaluator、OrderEvaluator、EvaluatorDecorator、LogDecorator，
  同步修改召回/CTR/KR 调用及测试，不保留 Deprecated 别名。
- pcoc 归 PointWiseEvaluator，Slot 直接委托原 EvalBatch；删除 cb，零分母统一 NaN。
- ListNDCG 只接收 relevance/topk_indices/mask 与可选 cutoff，保留全候选池 IDCG；
  显式索引和 scores 的稳定排序统一在 SlotEvaluator，不在 Metric 分支。
- TopKEvaluator 从 full_scores 张量取全库值，历史过滤不原地修改共享分数。
- Manager 是唯一日志落盘位置。Trainer、模型训练/预测、Dataset 不因本轮重构改动。
- 实验候选诊断由 MGECandidateResolver 先转换候选 scores/targets/mask，再交给自定义评价器。

## 迁移影响

Scorer.score 的输入从旧 EvalBatch 改为 ScoreContext；自定义来源与 order 选择移到 register 的
resolver=BatchResolver(...)。自定义 Evaluator 不能再读取 batch.pred/inter。原 cb 键变为 pcoc。
旧 wrapper 的调用方需显式注册核心评价器。历史 run/checkpoint/报告与源码快照不修改。

## 验证

- 全量回归 586 passed，logs/evaluate_resolved_regression.log。
- 覆盖多标签自定义价值不被误拦、过滤先于 Scorer 且单次调用、原候选/选序/整页视图隔离、
  scores/targets/ref/mask/users 同步投射、N 选 K 的全池 IDCG、全库历史过滤、
  裁判 Lookup/ModelScorer 经 Resolver 的链路、NCIS 与单点/整页 PCOC 的一致边界。
- 已保存 LC_MGE / DNN / DCN 各 48,898 屏只读复算，cb→pcoc 迁移后指标核对。
  日志 logs/evaluate_resolved_saved_predictions.log；不训练、不重新推理这些实验模型。
- 编译检查、git diff --check、单文件源码导出逐段校验。

## 文档

EVALUATE.md 重写为最终职责与注册示例，API_REFERENCE/ARCHITECTURE/TRAINING_FLOW 与
RERANK/KR 入口说明同步。当前重排单文件源码包一并刷新，仅保留路径与源码。
