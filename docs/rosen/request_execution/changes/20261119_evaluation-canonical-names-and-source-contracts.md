# 评估唯一命名、来源依赖声明与批量缓存

## 背景与约束

评估和重排链路曾为同一接口提供多套名称，仓库内部与新示例也混用这些名称。
Manager 通过具体 Scorer 类型判断标签来源，Resolver 通过具体类型及 field 判断目标依赖；
每个注册项重复筛选和 CPU 搬运，Pointwise/SlateValue 又将所有样本拆为 Python scalar 缓存。

用户明确要求不兼容旧代码。现在直接移除旧入口，不提供 DeprecationWarning、兼容别名或 fallback。
此约束已写入根目录 AGENTS.md 和架构执行规范。

## 唯一规范名

| 已删除入口 | 当前入口 |
| --- | --- |
| PointWiseEvaluator | PointwiseEvaluator |
| SlotEvaluator | RankingEvaluator |
| SlateEvaluator | SlateValueEvaluator |
| TopKEvaluator | RetrievalEvaluator |
| ScoreContext | EvalContext |
| PredScorer | PredictionScorer |
| TargetScorer | LabelScorer |
| LookupScorer | SidecarLookupScorer |
| ModelScorer | JudgeModelScorer |
| scorer._orders / scorer._permute | normalize_orders / gather_by_order |
| resolver.PredScorer / resolver.TargetScorer | 从 rosen.evaluate 导入规范 Scorer 名称 |
| EvalBatch(targets=...) / batch.targets | EvalBatch(labels=...) / batch.labels |
| BatchResolver(target_scorer=...) / resolver.target_scorer | label_scorer |
| resolver.requires_target(pred) | resolver.needs_target(pred) |
| view="candidates" | view="score_only" |
| apply_permutation（模块及 RerankModel 上） | select_by_order |

EvalBatch 恢复标准 frozen dataclass 构造器，labels 必需，__post_init__ 校验张量 batch 维度。
Resolver 直接使用公开的 normalize_orders / gather_by_order，不再导入私有旧名。
Scorer 的 alignment="candidates" 表示输入候选坐标，仍是该参数的规范取值，与已删除的 view 别名无关。

## 来源依赖与职责

BaseScorer 提供三个独立声明：

- uses_schema_labels：是否使用 schema 声明的标签，默认 False，LabelScorer 为 True。
- needs_target(pred)：是否需要 EvalContext.target 标签/任务名，默认取 uses_schema_labels。
- uses_prediction_target(pred)：是否消费绑定 Prediction.target 的 Tensor，默认 False。

PredictionScorer 在读取主 scores Mapping 时声明需要任务名，读取单 Tensor 主 scores 时声明消费其目标。
自定义 Scorer 可独立声明依赖，不需要继承某个具体实现。
Resolver 汇总 label/pred/ref 三类来源的声明，自己处理 order/full_scores 的依赖。
Manager 仅使用 Resolver 的接口完成标签合法性和目标冲突检查，不探测其 Scorer 类型或字段。

## 筛选、搬运和缓存

同一次 collect 为相同过滤器实例复用筛选结果；默认全选过滤器由 Manager 共享。
过滤器必须是纯函数，两个不同实例各自执行，不能假设任意 callable 语义相同。
CPU 副本按原 Tensor 身份复用，通过 resolve(..., to_cpu=...) 显式注入。
缓存以弱引用确认源 Tensor 身份，防止临时 Tensor 释放后的 Python ID 复用，且不滞留临时裁判 GPU Tensor；缓存只存活于当前 collect。
候选投射及冻结裁判仍使用原设备，下一 batch 不复用旧筛选或旧 Tensor 副本。

PointwiseEvaluator 按 batch 保存 labels/scores/users NumPy 数组，SlateValueEvaluator 按 batch
保存 predictions/values/references 数组。缓存拥有独立内存，原 Tensor 后续改写不会污染已有样本。
summary 对各列 concatenate 后算全局指标，重复 summary 不消费缓存；clear 释放全部数组。
候选均值/最大值仍用累加量。精确全局 AUC/r_auc 等仍需 O(N) 数值缓存，拼接时还会分配连续数组。

## 迁移范围

- src/rosen/evaluate/、src/rosen/models/rerank/ 的定义、导出和内部调用。
- tests/ 下的评估、重排、Trainer、MSR/unified 预测契约测试。
- KR pipeline、测试和报告生成脚本，包括历史入口的当前可执行源码。
- ML1M 重排 pipeline、测试及报告生成脚本。
- 当前 API、架构、评估和重排文档，以及源码阅读包。

已有 runs/ 内的日志、指标和 source_snapshot，以及 before_sources/ 保持为历史证据，
不改写其中的旧源码。历史 changes/ 记录描述当时行为，当前接口以本记录和 EVALUATE.md 为准。
外部调用方需直接迁移旧 import、构造参数和视图取值；自定义 Resolver 覆写 resolve 时需接收并使用 to_cpu。
不需要为这次接口及缓存调整重训或重做数据准备；本次未运行新的数据集实验。

## 验证

完整测试命令：

```bash
uv run pytest tests workspace/kr_time_budget/tests workspace/ml1m-rerank/tests
```

结果：810 passed。日志：[全量测试](../../../../logs/20261119_canonical_rerank_names_tests.log)。
覆盖删除入口报错、自定义 Scorer 三种来源声明、目标冲突、共享及独立过滤器、CPU 搬运次数、
跨 batch 指标、重复 summary、clear、输入改写、临时源 Tensor 释放、冻结裁判、KR 和 ML1M 数据/模型闭环。

额外只读内存检查：CPU、4 batch × 50,000 个合成样本，Pointwise 缓存 labels/scores/users。
12 个 NumPy 数组共 4,800,000 字节；tracemalloc 累计阶段当前内存 4,804,176 字节。
同样三列的原 Python scalar-list 布局参考为 20,472,686 字节，约减少 76.5%。
这只比较缓存阶段的 Python/NumPy 可追踪分配，不代表总进程 RSS、Torch 原生内存或 summary 拼接峰值。
日志：[缓存内存检查](../../../../logs/20261119_evaluation_batch_storage_memory.log)。

交接文件和四份 Repomix 阅读包从当前源码重新生成，逐文件核对内嵌源码，不直接编辑代码块。
继续维护时看 [评估契约](../../details/EVALUATE.md) 和 [API 参考](../../API_REFERENCE.md)。
