# 重排评估命名、候选价值与辅助输入

## 背景与范围

按用户提供的重构规格收敛 Pointwise / Ranking / SlateValue / Retrieval 四种评估语义，
保留已有 workspace 调用。模型依旧输出 Prediction，只有 Resolver 产生 EvalBatch。
本次未迁移或删除旧 Embedding 类，也未重训历史实验。

## 设计与实现

- EvalBatch 的真实真值字段为 labels；targets 构造参数和只读属性保留。
- 四个 Evaluator 和五个 Scorer 使用新名称；旧类名、ScoreContext、_orders、_permute、
  target_scorer 和 candidates 视图保留兼容入口。
- Ranking 增加 recall/hit/hr/precision/MRR/MAP 与 list_auc；IDCG 始终使用完整候选池。
  GAUC 使用全局排序与并列平均秩，向量化统计用户 AUC，不逐用户调用 sklearn。
- SlateValue 支持 [B,M,K]，cand_mean 按有效候选加权，cand_max 对请求等权。
  跨请求指标必须显式指定候选归约；max 使用预测最优候选，标签和参考分数同步选择。
  修复用户示例中三维 NCIS 未收集 reference 及各来源独立取最大值的错误。
- BatchResolver 根据 Scorer.needs_target 检查标签、预测、参考来源。
  Pointwise / Retrieval 默认 score_only，SlateValue 默认 slate，Ranking 默认 ranking。
  PredictionScorer("extra.topk") 与 order_field="extra.topk_order" 可直接评多候选。
- 二维静态来源始终作为候选池 gather，不采纳“下标越界就认为已切片”的启发式。
  标量来源 [B] 保持不变，三维来源由调用方明确按候选对齐。
- SidecarLookupScorer 支持稠密全排列与稀疏 (request_id, order) 查表，验证重复键、非有限值和非法选择。
  稀疏表原序分数由 order=null 行提供，缺失则报错；推荐显式声明 num_candidates。
- select_by_order 支持 K<=N，并保持 apply_permutation 别名。底层 gather 仍允许训练期重复解码记录；
  评估排列入口严格校验唯一性。共用的模型适配器选择当前特征，右补零，传送 mask/长度并保持历史不变。
  MultiGE 的全宽裁判路径保留原实现，部分列表使用同一右补零语义。
- AuxField 自动进入原始/转换 parquet 读取与 TensorFormatter，不参与 fit 或 embedding，
  禁止与监督/特征字段冲突。声明由调用方在加载 schema 时提供，不改变 feature_meta.json 格式。
- BaseTrainer 记录 epoch/step，默认反馈 target 使用 model.LABEL 或 default；修复 fetch_all 返回异常对象而不抛出的错误。

## 兼容边界

- dataclasses.fields(EvalBatch) 中 targets 已改名 labels；dataclasses.replace 应使用 labels=。
  仓库唯一的 replace(..., targets=...) 调用和相关断言已同步更新。
- candidates 仍可传入，但 resolver.view 规范化为 score_only。
- 静态 [B] 分数在给定 order 时保持原值；它不表示排列相关的反事实价值。
- 固定宽度模型的 K<N 评分是右补空槽位的语义，不宣称成为动态长度网络；
  不改变 NAR/AR 生成器本身的输出长度。EXTR 因广告/自然位分区要求完整宽度，明确拒绝部分列表。
- 三维候选的 labels/ref_scores 可为 [B]、共享 [B,K] 或对齐的 [B,M,K]；
  二维 [B,M] 预聚合候选值没有独立轴标识，不能自动与槽位区分，未新增猜测逻辑。
- 指标空请求/无正例口径和 precision 分母见 [评估文档](../../details/EVALUATE.md)。

## 验证

```sh
uv run pytest tests/rosen workspace/ml1m-rerank/tests \
  workspace/kr_time_budget/tests/test_architecture_examples.py \
  workspace/kr_time_budget/tests/test_rerank_benchmark_report.py -q --tb=short
```

结果：532 passed。包含核心所有 tests/rosen、KR 现有多任务/重排链路和 ML1M Pointwise/AR 链路；
新增手算排序指标、并列 GAUC、候选坐标对齐、三维 NCIS、稀疏 sidecar、AuxField、
各模型部分列表评分与未选物品/标签不影响裁判输出的检查。
日志：[20261119_rerank_refactor_tests.log](../../../../logs/20261119_rerank_refactor_tests.log)。

## 后续

按需迁移 workspace 旧导入和手写多候选评估器；本轮未改变历史指标键、训练数据或 checkpoint。
更新后的原始源码交接件位于 workspace/kr_time_budget/回复/新架构示例/重排链路代码审阅.md。
