# 列表重排：RerankModel 与 MultiGE

DNN、DCN、PRM、EXTR、KuaiShou、PIER、PRS、NAR、Seq2Slate 及 metric/多样性变体已提供
独立模型类。完整参考目录映射、用法与差异见 [独立模型说明](RERANK_BASELINES.md)，
源码导航见 [rerank/README](../../../src/rosen/models/rerank/README.md)。以下主要说明 MultiGE。

`rosen.models.rerank` 独立于 MSR 注册表。模型只实现 `calculate_loss()` / `predict()`，
沿用 `BaseTrainer`、`EMATrainer`、`LossPort` 及显式评估路由；没有模型私有 eval step。

## 有序选择与裁判接口

select_by_order([B,N,...], orders) 支持共享 [M,K] 和逐样本 [B,M,K]，返回 [B,M,K,...]，
1≤K≤N。该算子只提供 select_by_order 一个名称；底层 gather 允许训练期解码记录中的重复选择，
对外 score_permutations / JudgeModelScorer 在入口严格校验唯一合法下标。

所有内置模型共用 SlateRerankModel.score_permutations：先校验合法排列，再 encode_slate 一次，
使用 components/scoring.py 按 candidate_chunk_size 分块，得到 [B,chunk,N,...] 输入并返回 [B,M,K]。
K<N 时选择预编码物品，右补零至固定宽度，将显式 mask 同步选择/补齐并传给评分函数，
输出截取 K 个位置并将无效槽置零。不再构造候选 Interaction，也不注入任何私有控制键。
这个语义只用于支持掩码/空槽位的通用模型；
没有把固定宽度 OAM 改成动态长度网络，也不改变 NAR/AR 生成器的输出长度。
EXTR 的广告/自然位分区，以及 PIER/PRS/PIERSubmodular 和 MultiGE 族的 OAM 维持固定宽度，requires_full_slate=True，
会明确拒绝 K<N。MultiGE / NAR / Seq2Slate / metric 变体不声明动态长度生成能力。

MultiGERerankModel 与 PIERRerankModel 继承 OCPMRerankModel，再继承 SlateRerankModel，
共用同一个 score_permutations，不再覆写裁判执行循环。OCPMContext 仅保存原 batch 的历史页表示和页掩码，
OAM 历史编码在候选分块前做一次，通过 PageTargetAttention 的候选轴广播复用。
KuaiShou 预编码历史物品及 K/V，候选当前列表的注意力仍按每个排列重算。
不跨调用缓存：每次使用当前参数和当前 Interaction，训练梯度经过共享历史编码正常汇总。

模型扩展实现 _encode_items(cate,dense,mask)、_encode_context(interaction,cate,dense) 和
_score_encoded(items,dense,mask,context)，最后一个显式接收 [B,M,N] mask 并返回 [B,M,N] logits。
logits 与 score_permutations 是固定入口；子类覆写这两个方法会在类定义时报错，必须迁移到上述扩展点。
_mask 只负责从原请求读取掩码，之后候选有效性由独立张量传递；不会再读取原 Interaction 的隐藏开关。
EncodedSlate(items,dense,mask,context) 保存输入槽位表示与模型自有的固定上下文。

所有子模块注册名、注册顺序及初始化 RNG 保持不变；_score_candidates 统一委托
components/scoring.py 的 score_candidate_probs，后者共用 score_candidate_logits。
score_candidate_logits(encoded,orders,scorer=...,chunk_size=...,candidate_dense=...) 只接受新显式接口，
旧 OCPM 专属多参数调用已移除；ListwiseGenerator.forward 的 history_pages 参数接收预编码 [B,P,D]。
默认 eval_diagnostic_pools=True 保留诊断池输出；设置为 False 时只评主候选池，跳过
lastk/randk/beamk/all 候选，scores、最终 order、chosen_scores 和主候选池分数不变。

## 数据与构建

当前列表与历史列表的每个属性分别使用 `SequenceSetting`，共享同一个原子 setting。
历史数据始终存为长度 `P*K` 的一维序列，进入模型才折叠。两个分类视图需具有相同字段数、
embedding 维度与对齐的 `embedding_field_name`。按 group 或显式字段集合选择，每侧二选一。

```python
from rosen.models.rerank import MultiGERerankModel, build_rerank_model
from rosen.evaluate import BatchResolver, RankingEvaluator
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import EvaluatorManager

# manager 中已经配置 slate_cate / hist_cate 两个序列组和
# SlateLabel("ctr_label", max_len=5, mask_field="mask")。
model = MultiGERerankModel(
    manager,
    slate_cate_group="slate_cate",
    hist_cate_group="hist_cate",
    slate_dense_field="cur_hist_ctr",
    page_mask_field="page_mask",
    poi_num=5, page_num=5, top_k=20,
    submodularity_mode="lc",
    eval_all_perms=False,
)
# 同义工厂：build_rerank_model(manager, "mge", **以上关键字参数)
# LC 参考口径：build_rerank_model(manager, "lc_mge", **字段与维度参数)

evaluators = EvaluatorManager(manager)
evaluators.register("point", PointwiseEvaluator("auc", "logloss"))
evaluators.register("slate", RankingEvaluator("ndcg@5", "pcoc"), resolver=BatchResolver(view="score_only"))
evaluators.register("order", RankingEvaluator("ndcg@5"))
```

`slate_dense_field` 从 Interaction 读取先验分 `[B,K]` 或 `[B,K,1]`，不要求与分类字段
共组。无需拟合/嵌入的先验和历史页 mask `[B,P]` 通过 SchemaManager 的 AuxField 声明：
`AuxField("cur_hist_ctr", max_len=K)`、`AuxField("page_mask", max_len=P)`。
原始 ID 同样声明为 `AuxField("request_id", dtype="int64")`。
Dataset 自动选择并格式化这些列，独立推理使用
`FeatureContext.from_manager(manager, include_labels=False)` 共用同一声明，无需额外列清单或 formatter。
模型不会替 Scanner 添加列。
未设置 `page_mask_field` 时历史页全有效；设置后缺列则报错。`mask` 通过 SlateLabel 自动读取，
评估 Manager 根据 Prediction.target 自动查询关联 mask。外层 DataLoader 使用 `batch_size=None`。

`build_rerank_model(manager, "lc_mge", ...)` 显式预设 `submodularity_mode="lc"`、
`candidate_dense_mode="generator"`、`loss_reduction="nonzero_mean"`；调用方仍可覆盖。
`mge` 和直接构造 `MultiGERerankModel` 保留原默认值：`lc`、`prior`、`mean`。
两入口共享同一模型实现与参数结构，不新增一份 LC 网络。

## 组件与状态归属

- `fold_history_pages`：`[B,P*K,...] -> [B,P,K,...]`，支持非连续张量。
- `build_full_permutations`：字典序 `[K!,K]`，仅接受整数 `1<=K<=7`。
- `select_by_order`：输入 `[B,K,...]`，索引 `[M,K]` / `[B,M,K]`，输出 `[B,M,K,...]`。
  为保留 teacher forcing 的 argmax 记录，允许重复索引。
- `OAMBlock`：intra-attribute 沿 K 注意力，残差后池化；inter-attribute 沿 F 注意力，
  残差后池化为页表征。另返回未加残差的 intra-attribute 输出供 IntraPOI 使用。
  全部属性/POI 向量化；默认 attention logits 除以 8，与维度平方根缩放是不同约定。
- `PageTargetAttention`：有效页满足 `page_mask==1`；全无效时兴趣为零。
  `[B,M,D]` 广播或 `[B*M,D]` 的 repeat_interleave 保持样本与历史对齐。
- `PPMLSimHashRetriever`：位置加权平均 K/F，按参考代码逐元素乘 hash 后取 sign，
  距离为绝对差求和（异号可贡献 2），再按历史页加权。它不是标准随机超平面投影。
- `BeamSearchRetriever`：对全排列先验分乘曝光率后求和排序，是穷举粗筛启发式。

所有排列、位置编码、hash、时间距离权重、曝光率、随机候选 ID 均为 buffer；随模型迁移
设备及保存 checkpoint，不进入数据集。位置/hash 默认使用独立 seed 的正均匀随机常量，
距离权重为 `1/(floor(p/2)+2)`；曝光率为 `max(1-0.1*k,0)`。可以显式传入所有常量。
默认随机候选固定、无放回，避免评估/早停因每次重新抽样而变化。距离并列使用稳定排序。

## 共享权重与梯度

模型仅持有一份 OAM、IntraPOI、TargetAttention，调用时显式传入各模块。OCPM 和
Listwise 各有独立 MLP；两个自回归生成器各有独立 encoder、投影、GRU，默认共享 PointerHead，
可用 `share_pointer_head=False` 分离。共享模块仅注册一次，EMA deepcopy 保持内部引用关系。

四个生成器入口统一 detach 分类 embedding、历史 embedding、先验分与页 mask。
因此生成器辅助损失可以更新共享 attention 与自身网络，不能更新 embedding 表。
OCPM 监督分支使用有梯度的 embedding，metric 变体的当前 slate 同样保持梯度；
metric 路径只构建一次输入，不再计算随后被覆盖的原序 OCPM。Teacher forcing 与合法 greedy
选序仍需两次生成，后者在 no_grad 下执行，不保留无用生成图。可选 contrastive 项也属于 evaluator 训练目标，
通过历史兴趣路径更新历史 embedding，候选 embedding 与先验分保持 detach。

线性权重按二维 Xavier 初始化；新增 submodularity 模块在所有共有模块之后初始化，
同 seed 的共有参数逐元素一致。OmniEmbLayer 沿用框架默认 embedding 初始化（std=1e-4），
不宣称与 TF 的零初始化/预训练恢复流程、正态 attention 初始化等价。
先验分尺度会影响拼接后的贡献，应在真实实验里按数据口径检查，而不是依赖 shape/finite loss。

## 四种模式与目标函数

`submodularity_mode`：

| 模式 | 生成器数 | 第四生成器目标 |
| --- | ---: | --- |
| `none` | 3 | 无第四生成器 |
| `cosine` | 4 | teacher 交叉熵的 reward 从 y 改为 y-coverage；coverage 是同位置特征余弦相似度，对前三生成器求均值并 detach |
| `lc` | 4 | teacher 交叉熵；离散 LC 只记入 loss_diagnostics，不进入优化项 |
| `lc_soft` | 4 | teacher 交叉熵减可微 expected-position 多样性项，属于实验扩展 |

`lc` 精确保留参考代码的定义：直接比较排列记录中的数值 `perm[...,i] < perm[...,j]`，
对所有生成器对（四生成器时六对）累计关系矩阵的不同元素数，再对 B 求均值。
它不是先求每个 item 的位置再比较的 Kendall 距离。训练 argmax 有重复时也保留此计算。
离散 LC 本身不训练任何参数，已从 LossPort 优化项移除；通过 loss_diagnostics["lc_exact"]
单独读取，KR metrics.json 也记录该诊断，避免用负常数偏移误读训练收敛。

`lc_soft` 对第四生成器的 `[B,step,item]` 概率计算 `sum(step * p)`，形成期望位置；
再用 `sigmoid((pos_j-pos_i)/temperature)` 形成软先后关系，计算与前三个生成器的固定关系
的平方差；前三者之间的固定两两距离不进入优化目标。对角线不计。参考顺序通过 one-hot 的位置和
转换，训练重复索引仍保留，不做 permutation 修补。该 surrogate 不宣称数值趋近原 LC。

自回归使用 PyTorch `GRUCell`，默认 hidden=64，PointerHead 为 `2H ->64->16->1`。
训练损失固定第 t 步选择原始位置 t，并对下一步输入与初始状态 detach；记录 argmax 可以重复。
`predict()` 始终 greedy，从可选集合移除刚选 item，保证合法排列，不依赖调用方先执行 eval。
`calculate_loss()` 固定使用 teacher forcing。标签 mask 同时用于辅助 BCE 与逐步交叉熵。

`masked_bce_with_logits` 的 `mean` 为加权 loss 除 B*K；`valid_mean` 除 mask 权重和；
`nonzero_mean` 除非零权重个数（全零时返回零）。
**mean/valid_mean 不应笼统称为 TF 默认口径**：[TF1 sigmoid_cross_entropy 文档](https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/sigmoid_cross_entropy)
默认 `SUM_BY_NONZERO_WEIGHTS`，即除非零权重数；二值 mask 时与 `valid_mean` 一致。
`nonzero_mean` 同时支持非二值权重的这一分母约定。自回归 loss 按步骤求和、每步对 B 求均值，
不受 BCE reduction 开关影响。

`loss_components(interaction)` 提供图上的独立标量，便于审计梯度；`calculate_loss` 汇总并在
LossPort 活跃时登记 `gen.pointwise`、`gen.listwise`、`gen.generative`、可选
`gen.submodularity`、可微模式独立的 `gen.coverage`、`evaluator` 及可选 `contrastive`。对比项默认关闭，开启后为
`-contrastive_weight*(SimHash最佳候选均分-最差候选均分)`。

## 输出与计算成本

`predict()` 返回带类型化候选池与选中得分的 `Prediction`：

- scores 是原始槽位的 `[B,K]` sigmoid 概率。
- order 是 G 个生成器中屏均分最高者的输入槽位索引 `[B,K]`，不是全排列最优解。
- candidates 为 CandidatePool，scores/orders 均为 `[B,G,K]`，分数按候选输出位置对齐。
- chosen_scores 是模型选择候选时直接保存的 `[B,K]` 预测，与最终 order 对齐。
- extra 中其他候选族 lastk/randk/beamk 是 `[B,min(top_k,K!),K]`；
  all 仅在 eval_all_perms=True 时提供。每族同时附带同形的 `<族名>_order` 排列索引。
- 导出器调用 resolve_chosen_scores()：候选池模型使用显式选中分数，直接打分模型按 order gather 原槽位分数。
- outputs_candidate_pool 在 MultiGE/PIER 上为 True，子类继承；KR 按此能力注册 BatchResolver.for_candidates()。

候选分数乘以随排列移动的曝光 mask，模型选择的均分分母仍为 K，不随本次接口迁移改变。
候选指标直接使用核心 SlateValueEvaluator；Resolver 依 candidates.orders 同步投射标签和 mask。
主候选及选中得分不再放入 extra，不提供旧键兼容；扩展诊断通过 with_extra() 保留类型化字段。
默认 slate 评估使用选中预测，候选池模型缺 chosen_scores 时拒绝从原序 OCPM 分数 gather。
原 out/ctr_out/q_out/ctr_* /out_index 字典接口已替换，不再保留多个主分数别名。

候选和原始列表使用同一已训练 OCPM head；不保留参考代码不同名字导致的独立 top-k 输出层。
Pointwise 直接对原序 dense 打分，logits、监督标签和排序下标均为原始槽位坐标；
不再将 SimHash top-one 的相对位置索引直接套到原始物品上。

`candidate_dense_mode` 明确选择生成列表进入 OCPM 的 dense 特征：

- `prior`：将原始先验分随候选排列 gather，保留旧版 rosen 行为。
- `generator`：传入 Pointwise/Listwise 的 sigmoid 输出，以及两个自回归生成器每一步的最大
  softmax 概率。Pointwise/Listwise 分数按各自提出的候选索引 gather，AR 分数按解码步骤排列，
  所有 dense 分数明确与候选物品位置对齐并 detach。lastk/randk/beamk/all 仍使用
  各自排列后的原始先验分，原列表监督也仍使用原始先验分。

该选择可能改变最终 `order`，不能从旧 checkpoint 的参数 shape 推断。保存模型构造配置，
重载时同时恢复 `candidate_dense_mode`；历史 `prior` 结果不能直接改名为 LC 参考口径。
参考代码单独命名候选输出 head、候选历史错位等问题仍按上文使用已训练 head 和正确对齐修正。

`candidate_chunk_size` 沿候选轴分块注意力打分；全排列粗筛仍有阶乘复杂度。
`eval_all_perms=False` 只关闭全部排列的 OCPM 输出，不关闭穷举粗筛。K<=7 是硬上限而非
内存保证；默认 K=5。候选收益是模型估计值，未测量真实重排点击率。

## 验证边界

测试使用合成固定种子张量，逐属性/逐 POI/逐排列参考式核对算子，检查梯度隔离、
共有参数初始化、候选分块、历史对齐、Teacher Forcing 与 Greedy、Trainer/EMA/账本/早停和
checkpoint 恢复。没有 TensorFlow 权重转换及跨框架逐层数值对齐，PyTorch GRU 的门控约定
也不能直接等同 TF1 GRUCell。这里交付可训练的结构移植，不给出 Avito/论文效果复现结论。

## 可执行训练与独立推理

KR 示例将原始事件、冻结先验、列表数据、训练、valid 选择 checkpoint、完整 test 和排列导出
串成显式流程。运行目录新增 `rerank/model_config.json`，记录 schema 路径和所有模型构造参数。

```bash
uv run python -m workspace.kr_time_budget.run_architecture_examples \
  --mode lc --candidate-dense-mode generator \
  --data workspace/kr_time_budget/data/lc_mge_new \
  --output workspace/kr_time_budget/runs/lc_mge_new

uv run python -m workspace.kr_time_budget.predict_lc_mge \
  --run workspace/kr_time_budget/runs/lc_mge_new/rerank \
  --input workspace/kr_time_budget/data/lc_mge_new/rerank_lc_mge_new/test.ordered.parquet \
  --output workspace/kr_time_budget/runs/lc_mge_new/predictions.parquet
```

独立推理从配置重建模型，严格加载 checkpoint；输入必须由该 schema 编码。它只读取特征、
`mask`、历史页 mask、长播先验和请求 ID，不读取 `is_click`/`long_view` 标签。
输出 `scores` 为原列表概率、`out_index` 为最终排列、`chosen_scores` 为选中候选的 OCPM 概率；
后两者按输出位置对齐。保留 schema 及词表，不能只分发 `.pt` 文件。
示例的 K=6/P=2、long_view 标签与 KR 数据准备属于调用方实验约定。

实现见 [训练入口](../../../workspace/kr_time_budget/pipeline/run_architecture_examples.py)、
[独立推理](../../../workspace/kr_time_budget/pipeline/predict_lc_mge.py)；
实测流程与限制见 [LC_MGE 补全记录](../../../workspace/kr_time_budget/回复/新架构示例/LC_MGE补全.md)。

## 2026-11-19 审计修复后的实验边界

命名预设统一在 presets.py，resolve_rerank_config 合并默认与显式参数，调用方显式参数优先。
KR 不再维护独立 variants 覆盖表；protocol.json 保存每个模型的 effective_model_configs。
NAR/Seq2Slate/metric 的 AR 损失统一每步 batch mean，不再乘当前 batch_size；尾批保持同量纲。
重排 checkpoint 依据 valid chosen_order.ndcg@6，pointwise 仍依据 valid long_view.auc。

这些是训练/预测语义修复，不是与旧版数值等价的整理。旧 benchmark 及第二部分等价快照
只描述对应的历史源码；不能把其指标当作修复版结果。详细范围见
[完整审计修复记录](../request_execution/changes/20261119_rerank-review-fixes.md)。
