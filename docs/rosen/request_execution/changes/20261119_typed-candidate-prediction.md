# 类型化主候选池与选中得分

## 问题

重排主候选及选中分数原先放在 Prediction.extra，模型子类重建 Prediction 时容易丢字段。
推理侧探测 extra 键、评估侧列举模型名，使原序 OCPM scores 和选中列表得分的区别无法由契约表达。

## 决策

- 新增 frozen CandidatePool(scores, orders)，两列均为 [B,M,K]；select(indices[B]) 返回每请求选中的 (order, scores)。
- Prediction 新增 chosen_scores 与 candidates，验证 selected scores 与 order 的任务及形状对齐。
- get_chosen_scores 读取显式字段；resolve_chosen_scores 优先使用显式字段，直接打分模型从原序 scores 按 order gather。
  候选池模型缺显式选中得分时直接报错，不允许从原序 OCPM 分数推导所选列表的预测。
- with_extra 合并诊断并保留全部类型化字段；行切片、空 batch、多任务 Mapping 和设备迁移同步处理所有字段。
- Scorer 使用 chosen_scores / candidates.scores，Resolver 使用 candidates.orders；主候选工厂为 BatchResolver.for_candidates()。
  不增加含义不明的 candidates 简写，沿用一套规范命名。
- 主候选池及选中得分不再写入 extra；lastk/randk/beamk/all 和多样性统计仍作为诊断字段。

## 评估语义

沿用已有 BaseScorer 依赖声明、Manager 的行筛选/CPU 搬运复用、评价器的 batch NumPy 缓存。
Manager 仍通过 Resolver 的依赖接口检查目标，新增 chosen_scores Mapping 的歧义判断，
没有重新引入具体 Scorer 类型或 field 探测。

默认 slate 视图消费主 order，且有 chosen_scores/candidates 时，通过 PredictionScorer("chosen_scores")
调用 resolve_chosen_scores。这样默认整刷评价也不会误用原序 OCPM 分数。
候选池模型显式以 PredictionScorer("scores") 对给定 order 评分会报错；score_only 仍评价原序分数。
chosen_scores 与 candidates.scores 固定为 selected 坐标，不允许声明为 candidates 后重复 gather。
选中字段的 num_candidates 元数据查询不执行得分解析，不把输出宽度 K 当作原始候选数 N。

## 模型与管线

RerankModel.outputs_candidate_pool 默认 False，MultiGE/PIER 为 True，相关子类继承该能力。
MultiGE/PIER 使用 CandidatePool.select 保存实际选择，metric/多样性子类使用 with_extra。
PIER.predict 复用已有输入计算原序分数，移除了末尾 logits 造成的第二次 embedding 读取。
参数、buffer、初始化和训练损失未调整。

KR 根据注册模型的能力声明添加主候选评价，不再通过模型名元组识别输出。
独立推理统一调用 resolve_chosen_scores，并用 model.poi_num 校验排列宽度。
训练后 test 导出也增加 chosen_scores 列，所有推理导出使用同一解析契约。

## 迁移范围与影响

涉及 core/prediction、core/evaluate 导出、Scorer/Resolver/Manager、MultiGE/PIER 子类、
KR pipeline、相关测试、当前文档及源码阅读包。只提供新契约，未添加旧键兼容或告警层。

| 已移除的主输出访问 | 当前访问 |
| --- | --- |
| pred.extra["topk"] | pred.candidates.scores |
| pred.extra["topk_order"] | pred.candidates.orders |
| pred.extra["chosen_scores"] | pred.chosen_scores / pred.resolve_chosen_scores() |
| 手动重建 Prediction 扩展诊断 | pred.with_extra(...) |
| 主候选 extra scorer/order 字符串组合 | BatchResolver.for_candidates() |

extra.* 仍可用于独立诊断池和自定义张量。历史 runs/source_snapshot、日志和指标未改写。
早前 Prediction迁移.md 的完整 test 核验是历史证据，本次没有重训或完整数据集复测。
主候选模型输出的数值没有调整，但默认 slate 价值评价现在使用正确的选中得分，
不能把此前从原序分数 gather 得到的价值指标视为相同口径。

## 验证

```bash
uv run pytest tests workspace/kr_time_budget/tests workspace/ml1m-rerank/tests
```

848 passed；[完整日志](../../../../logs/20261119_typed_prediction_full_tests.log)。
包含 35 项新增核心/评估契约用例、PIER 单次输入读取、未知注册名的能力路由，
以及现有重排训练、梯度、checkpoint、无标签推理、KR/ML1M 数据链路测试。
覆盖池的类型/形状/空轴/选择下标、选中任务对齐、with_extra、切片、空 batch、
设备与梯度、拒绝错误原序 gather、部分列表、目标冲突、候选标签/mask/用户投射和诊断池。

另从本次改动前的已校验阅读包加载旧 predict，使用相同权重和合成 batch，CPU float32、eval/no_grad，
对 11 类候选模型逐张量验证 rtol=atol=0：原序 scores、最终 order、选中分数、主候选分数/排列和诊断池全部一致。
包括 PIER、PRS、NAR、Seq2Slate、PIERSubmodular、两个 metric 版本、mge3、mge_cosine、mge、lc_mge。
[逐值核对日志](../../../../logs/20261119_typed_prediction_parity.log)。这是合成数据实现一致性核对，不是数据集效果结论。

交接 MD 及四份 Repomix 包从当前源码重新导出并校验，不直接编辑内嵌代码块。
后续参考 [评估契约](../../details/EVALUATE.md)、[重排契约](../../details/RERANK.md) 和 [API](../../API_REFERENCE.md)。
