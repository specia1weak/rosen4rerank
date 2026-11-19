# 预测、解析与评估

```text
Interaction + model.predict() 的 Prediction
  → EvaluatorManager：按注册项过滤行、确定目标/mask/用户
  → EvalContext：只供 Scorer/Resolver 使用的原始来源
  → BatchResolver：执行 Scorer，一次性完成候选/输出坐标投射
  → EvalBatch：只有最终 Tensor
  → Evaluator.collect / summary / clear
```

## 输出与解析张量

Prediction 是模型的语义输出；scores 为 [B] 或输入槽位 [B,N]，order 为输出位置到输入槽位的
LongTensor[B,K]，1≤K≤N，行内无重复；full_scores 为全库 [B,|I|]。
chosen_scores 是与 order 对齐的浮点 [B,K]；多任务 scores/order/chosen_scores Mapping 使用 schema 标签名。
candidates 为 CandidatePool(scores, orders)，两列均为 [B,M,K]，分数与各自候选的输出位置对齐。
extra 只放其他诊断池、统计和自定义张量，不承担主候选池或选中分数协议。

CandidatePool.select(indices[B]) 逐行选择候选并返回 (order, chosen_scores)；构造器校验类型、形状及非空 M/K。
原始候选数 N 由评估来源声明，排列唯一性和越界由 Resolver 在投射前校验。
Prediction.get_chosen_scores 只读显式选中字段；resolve_chosen_scores 优先返回该字段，
直接打分模型没有候选池时按 order 从原序 scores gather。候选池模型缺显式 chosen_scores 直接报错。
with_extra 合并诊断并保留所有类型化字段；行切片与 to(device) 同步处理完整输出。

EvalBatch 只包含 labels（必需）、scores、ref_scores、order、mask、users、full_scores。
字段只能是同 batch 大小的 Tensor 或 None；不包含 Prediction 或 Interaction。
Resolver 在进入评估器前 detach 并转 CPU，模型裁判推理仍使用原 Interaction 的设备。

## register 显式组装来源与视图

```python
from rosen.evaluate import (
    EvaluatorManager, BatchResolver, FieldScorer,
    PointwiseEvaluator, RetrievalEvaluator, RankingEvaluator, SlateValueEvaluator,
)

result = EvaluatorManager(schema, uid_field="user_id_raw", log_path="logs/eval.log")
result.register("ctr", PointwiseEvaluator("auc", "gauc", "logloss", "pcoc"), target="is_click")
result.register("rescore", RankingEvaluator("auc", "in_slate_auc", "ndcg@6", "pcoc"),
                target="long_view", resolver=BatchResolver(view="score_only"))
result.register("prior", RankingEvaluator("ndcg@6"), target="long_view",
                resolver=BatchResolver(pred_scorer=FieldScorer("prior_long_view"), view="score_only"))
result.register("selected", RankingEvaluator("ndcg@6"), target="long_view")
result.register("slate_value", SlateValueEvaluator("r_auc", "slate_auc", "rmse", "mae", "pcoc", "ncis"),
                target="long_view", resolver=BatchResolver(view="slate", ref_scorer=FieldScorer("propensity")))
result.register("candidates", SlateValueEvaluator("cand_mean", "cand_max", reduction="mean"),
                target="long_view", resolver=BatchResolver.for_candidates())
```

register 未显式传 resolver 时，SlateValueEvaluator 默认采用 slate，PointwiseEvaluator / RetrievalEvaluator
默认 score_only，RankingEvaluator 和其他自定义评估器默认 ranking。
显式传入的 Resolver 始终优先；可以用 score_only 明确评价原位分数或原整页。

BatchResolver 的三种 view：

| view | 构造的 EvalBatch |
| --- | --- |
| ranking（默认） | 保留完整 N 候选的 labels/scores/mask，保留模型显式 order；没有 order 时按 scores 排序 |
| score_only | 保留完整 N 候选的 labels/scores/mask，不传 order，用于原序模型分数或上游先验排序 |
| slate | 有 order 时将 pred/target/ref scorer 和 mask、二维 users 一起投射到 K 个位置；无 order 时评价原列表；order 不传入下游 |

Scorer 接收 EvalContext(pred, inter, target)，不接收 EvalBatch。五种来源：PredictionScorer 读取
当前目标预测，LabelScorer 读取主标签，FieldScorer 读取自定义字段，SidecarLookupScorer 按 request_id
查伴生 parquet，JudgeModelScorer 使用冻结模型推理。三种 scorer 在 BatchResolver 中显式组装，
默认 pred/target 使用 PredictionScorer/LabelScorer；reference 仅在配置 ref_scorer 时提供。
默认 slate 视图消费主 order 时，如有 chosen_scores 或 candidates，则通过 PredictionScorer("chosen_scores")
调用 resolve_chosen_scores；它不会将原序 OCPM 分数当作重排后的预测。显式 score_only 仍读取原序 scores。
候选池模型在有 order 的评分中显式使用 PredictionScorer("scores") 会报错。

Scorer 自己声明依赖，Manager 和 Resolver 不检查具体 Scorer 类型或读取 field 等实现字段：

- `uses_schema_labels: bool`：是否使用 schema 声明的标签；默认 False，LabelScorer 为 True。
- `needs_target(pred) -> bool`：是否需要 EvalContext.target 标签/任务名；默认取 uses_schema_labels，多任务 PredictionScorer 按其来源声明。
- `uses_prediction_target(pred) -> bool`：是否消费绑定 Prediction.target 的来源；默认 False，PredictionScorer 在读取单 Tensor 主分数、选中分数或主候选池时声明该依赖。

BatchResolver 对 label/pred/ref 三类来源汇总这些依赖，并计入自己使用的 order/full_scores。
Manager 通过 Resolver 的同名接口执行 schema 校验及目标冲突检查。自定义 Scorer 可独立声明这些能力，
无需继承 LabelScorer 或 PredictionScorer。

自定义 Scorer 继承 BaseScorer，实现 score(context, order=None)。例如多目标加权价值应从
context.inter 计算；如果支持 slate 投射，还需在给定 order 时返回所选位置的值。
order 在打分前按原始 N 校验，不使用已投射的 [B,K] 或聚合后的 [B] 反推 N。
原候选宽度可来自 mask、多任务选定的预测槽位、原标签，或 Scorer.num_candidates(context)。
BaseScorer 默认不猜宽度（返回 None）。PredictionScorer 只读取当前预测槽位，LabelScorer 只读取
当前主标签，FieldScorer 只读取自己指定的字段；一维来源返回 None，不借其他字段的宽度。
JudgeModelScorer 声明模型宽度，SidecarLookupScorer 声明排列表或自身槽位宽度；
获取宽度不额外执行打分。缺少原始宽度的自定义标量 Scorer 可以覆盖 num_candidates。

这类自定义 label_scorer 不需要 Manager 猜测某个 schema label；可显式设置
BatchResolver(mask_field="valid_candidates")。当按 order 评分且返回整刷标量时，由原始来源或 Scorer 声明确定 N；
候选 mask 如有提供，则用于过滤没有有效输出位置的整行。

SidecarLookupScorer 初始化读取内存表并建立 ID 索引，ID 唯一且非空；缺 ID 报错。
原序读 scores；候选读字典序 K! 编号的 perm_values。预聚合标量的 reduction 由产出方决定；
稠密全排列表明确拒绝 K<N。指定 order_col 后可读稀疏 (request_id, order) → score 表，
支持任意 N≥K；建议显式传 num_candidates=N。重复键、非整数/重复/越界选择和非有限分数报错。
order=None 需要对应 ID 的 order 为 null 的原序行；没有该行则报错，不猜测原排列价值。
JudgeModelScorer 将裁判设为 eval、冻结参数并 no_grad，候选打分委托 model.score_permutations。
所有内置裁判均使用 SlateRerankModel 的编码/分块入口，物品查表及固定历史编码不随候选逐个重复。
候选掩码显式传入 _score_encoded，输出统一将无效槽位置零；没有 Interaction 控制键。
通用 Slate 模型以右补零和同步 mask 定义 K<N 的裁判输入；EXTR、PIER/PRS/PIERSubmodular 和 MultiGE 族明确拒绝部分列表。
此接口不改变生成器自身的输出长度。

## 行过滤和目标解析

EvaluatorManager 保留 bind(schema) 的同对象约束，filter_fn 返回 bool 或布尔 Tensor[B]。
按过滤器实例复用 inter 和 Prediction 的行切片，包括候选池与选中得分；空行跳过，然后执行各项 Resolver。

注册表条目使用内部冻结数据类 _EvalTask，保存 evaluator/filter_fn/target/uid_field/resolver；
内部属性通过 task.resolver 等访问，不再使用配置字典索引。collect 按顺序调用
_filter_rows → _resolve_target → _resolve_mask → _resolve_users → resolver.resolve → evaluator.collect。
目标完成默认推断后，若 Resolver 不需要目标则立即返回。行过滤器接受普通 Callable；
空 batch 即使过滤器返回 True 也会跳过解析。显式 Resolver 始终保留，不依据其布尔值替换。

行过滤器必须是纯函数。同一次 collect 中，使用相同过滤器实例的注册项复用筛选结果；
默认全选过滤器由 Manager 共享，不同过滤器实例独立执行。CPU 副本按原 Tensor 身份复用，
通过 `BatchResolver.resolve(..., to_cpu=...)` 显式注入搬运函数；独立调用 resolve 默认 detach/cpu。
这些缓存仅存活于当前 collect，下一 batch 重新建立。Scorer 及冻结裁判仍在原设备执行，
Evaluator 接收已解析的 CPU Tensor；自定义 Resolver 若覆写 resolve，需接收和使用 to_cpu。

默认来源按 register(target=...) > Prediction.target > schema 唯一标签选择任务。
不猜 Mapping 的第一个 key，多标签歧义报错；默认标签来源检查 schema 声明和冲突。
自定义 label_scorer 且不依赖默认预测 Mapping 时，不强迫选择一个 schema label。
但只要实际消费了声明 target 的单 Tensor 预测或 order，注册 target 与 Prediction.target
冲突仍须报错；完全由自定义字段取分且忽略模型 order 时，不用无关预测的 target 限制自定义价值。
mask 来自 resolver.mask_field 或 SlateLabel.mask_field；users 优先注册 uid_field，其次 Manager、schema。

全库 scores 通过 EvalBatch.full_scores 传递，不借原始 Prediction 的后门。
日志只由 Manager 的 summary(epoch, step, stage=...) 写入文件，BaseTrainer 显式区分 valid/test，不在 Evaluator 包装日志类。

## 四个核心评估器

| 类 | 输入 | 指标 |
| --- | --- | --- |
| PointwiseEvaluator | [B] 或匹配 [B,N]，有效槽位展平 | auc / gauc / logloss / pcoc |
| RetrievalEvaluator | full_scores [B,|I|]，labels 为物品索引 [B]，users [B] | hr@k / recall@k / ndcg@k / mrr@k，支持历史过滤 |
| RankingEvaluator | labels [B,N]，scores [B,N] 和/或 order [B,K] | 单点指标 + list_auc（别名 in_slate_auc）/ ndcg@k / recall@k / hit@k（hr@k）/ precision@k / mrr@k / map@k |
| SlateValueEvaluator | 已对齐 [B,K]、[B] 或候选 [B,M,K] | r_auc / slate_auc / pcoc / rmse / mae / ncis / cand_mean / cand_max |

所有评估器继承 BaseEvaluator，实现 collect(EvalBatch)、summary()、clear()。
Evaluator、LegacySlateValueEvaluator、OrderEvaluator、EvaluatorDecorator、LogDecorator 已删除，
不提供 Deprecated 别名。旧 cb 指标统一迁移为 pcoc，真实和为 0 时统一 NaN。
RankingEvaluator 自己只统计列表指标，单点指标直接委托 PointwiseEvaluator.collect(batch)。

### 排序与 NDCG

RankingEvaluator 在 batch.order 存在时评价它，否则稳定降序排序 scores，无效候选排末尾。
显式选择要求 1≤K≤N、下标在 [0,N)、行内不重复；对所有列表 NDCG 指标共用同一组索引。
底层 ListNDCG.calculate(relevance, topk_indices, mask=None, *, k=None) 不接收 scores 或 order 开关。
relevance/mask 保留整个 N 候选池，IDCG 始终来自完整候选池的前 k 项，DCG 从选中索引 gather。
k 默认取所给索引宽度；显式 k 大于输出长度时，缺失位置收益为 0，仍用完整 k 的理想收益。
增益为 2**relevance-1，位置折损为 1/log2(position+2)。无正例行记 0，全 mask 行跳过。
显式 order 内的无效位置收益为 0，但保留它占据的位置，不自动前移其他物品。

in_slate_auc 需要候选 scores 和二值标签，按行排序及并列平均秩计算 AUC，
只对有正有负的合法行等权平均。底层 InSlateAUC 自带二值校验，单独注册该指标也会校验；
单类行不参与，全部不合法时为 NaN。Slot 的预测分与标签均转换为 CPU float64，再进入列表算子。
分类指标同样需要二值标签；只有 pcoc 时价值可以是连续数值。
GAUC 按用户和分数排序后向量化计算并列平均秩，按合法用户的有效样本数加权。
GAUC 在过滤有效槽位前检查 users 是否存在、形状是否为 [B] 或匹配 [B,N]，全 mask batch
也不能绕过契约检查；合法的空 batch 不累计结果。

### 整刷价值

SlateValueEvaluator 不持有 Scorer，不读取 order，不推断 N，不重排 mask。
sum/mean/dcg 聚合已对齐的槽位；mean 除以有效位置数，dcg 为原价值的线性折损和，
不做指数增益。[B] 已聚合标量保持不变。collect 和 reduce_slate 支持 [B,M,K]；
mask 可以是 [B,M,K] 或对所有候选相同的 [B,K]。cand_mean 对所有有效候选的聚合价值求均值，
cand_max 对每个请求的最大候选价值求请求均值。无效候选和全空请求不参与。
三维输入计算跨请求价值指标时必须显式指定 candidate_reduce="mean" 或 "max"；
mean 对有效候选求均值，max 按预测价值选一个候选，并同步选择同一候选的标签和参考分数。
标签和参考分数可为 [B]、所有候选共享的 [B,K] 或已对齐的 [B,M,K]。
多候选预聚合 [B,M] 不会被自动猜成候选轴；当前二维输入始终按槽位解释。

r_auc 是不同真实价值的 pair 上预测同向比例，预测并列计半分；排序/Fenwick 计数为
O(N log N) 时间、O(N) 空间，不构造 [N,N]。slate_auc 以真实价值>0 为二分类标签。
PCOC 是全量预测和/真实和；RMSE/MAE 直接比较整刷值。
NCIS 使用 ref_scores：w=min(pred/max(ref,1e-6), cap)，返回 sum(w*value)/sum(w)。
pred/ref 需非负。输入为真实策略概率时才具有对应的重要性采样含义。

全 mask 行不参与，mask 区域外允许哨兵/NaN；有效值需有限。
全局指标在全量收集后计算，不平均 batch AUC。空结果为 {}，summary 不清空状态。
无可比较 pair 的 r_auc、单类 slate_auc、零分母 pcoc、零总权重 ncis 均为 NaN。

PointwiseEvaluator 按 batch 保存展平后的 labels/scores/users NumPy 数组；SlateValueEvaluator
按 batch 保存整刷 predictions/values/references 数组，候选统计仍使用累加量。数组拥有独立内存，
后续原 Tensor 改写不会污染已收集样本。summary 只拼接一次各列数组后计算全局指标，
不将每个样本转成 Python scalar；重复 summary 不消费缓存，clear 释放所有 batch 数组。
精确 AUC、r_auc 等仍需保留 O(N) 数值缓存，summary 期间 concatenate 会额外分配连续数组。

## 自定义评价

主候选池使用 BatchResolver.for_candidates()，等价于 PredictionScorer("candidates.scores") 配合
order_field="candidates.orders" 和 view="slate"。单个选中结果使用 PredictionScorer("chosen_scores")；
这两个类型化分数字段固定为 alignment="selected"，不能声明为 candidates，避免二次 gather。
字段只使用 candidates.scores / candidates.orders 两个明确路径，不增加含义不明的 candidates 简写。
诊断池可使用 PredictionScorer("extra.diag_scores", alignment="selected") 配合
order_field="extra.diag_orders"，同步投射标签、mask 和二维用户字段。
PredictionScorer、FieldScorer、LabelScorer 可显式指定 alignment="selected"，用于已经按输出位置
对齐的 [B,K] 或 [B,M,K]，即使 K=N 也不再次重排。普通原序来源默认 alignment="candidates"；类型化选中字段固定为 selected。
二维来源始终作为原始候选池，所有声明的候选宽度必须一致，不靠长度或下标猜测已选坐标。
自定义 BaseScorer 负责自身的 score(context, order) 坐标处理，Resolver 不按具体 Scorer 类型开后门。
标量 [B] 与排列无关，直接保留；单个 [B,K] 不广播到无关的候选集合。

共享常量候选通过 BatchResolver(view="slate", orders=shared_orders) 显式配置，接受 [M,K] 或 [1,M,K]，
包括 M=B 时也固定表示共享，不被 Manager 的行过滤切片。Prediction.extra 仍只存逐请求的 batch 张量；
自定义 Resolver 提供的共享候选也走 normalize_orders，不再提前按首维拒绝。
ranking 视图只接受每请求一个 [B,K]，且 scores/labels/ref_scores 必须匹配原候选维度；
只注册 NDCG 也不能绕过 scores 对齐校验。

KR 根据 RerankModel.outputs_candidate_pool 注册核心 SlateValueEvaluator("cand_mean", "cand_max", reduction="mean")，
移除了 MGECandidateResolver/MGECandidateEvaluator，输出键改为 cand_mean/cand_max。
候选预测均分不代表真实收益。

目前没有 ExplorationEvaluator、排列空间 HR 或 reward/rg/win_rate。来源解析变化不会
自动补充搜索或裁判的有效性证据。

## 唯一规范名与空结果口径

类名只提供 PointwiseEvaluator / RankingEvaluator / SlateValueEvaluator / RetrievalEvaluator、
PredictionScorer / LabelScorer / FieldScorer / SidecarLookupScorer / JudgeModelScorer / EvalContext。
标签只用 EvalBatch(labels=...)、batch.labels 和 BatchResolver(label_scorer=...)；
三种视图只接受 ranking / score_only / slate。使用 dataclasses.replace 时写 replace(batch, labels=...)。
有序选择算子只用 select_by_order，Scorer 共用 normalize_orders / gather_by_order。
旧名称、旧构造参数、旧属性和旧视图取值直接报错，不提供别名、告警兼容层或 fallback。

列表 recall/hit/MRR/MAP 对全池没有正例的请求返回 NaN 并从均值排除；NDCG 将此类请求计为 0。
precision 以选出的有效位置数为分母，无正例且有有效选择时计 0；没有有效选择时排除。
输出 K 小于指标 cutoff 时，不补选物品；NDCG 的 IDCG 仍从全池取 cutoff 个。

Prediction.get_scores/get_order/get_chosen_scores 在多任务且没有 target 时抛出 ValueError，在目标不存在时
抛出列有可用任务名的 KeyError；显式参数优先，否则使用 Prediction.target。
EvalContext 的 scores/order 解析共用这两个入口。
