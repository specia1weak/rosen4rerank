This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where content has been formatted for parsing in markdown style.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
- Pay special attention to the Repository Description. These contain important context and guidelines specific to this project.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: docs/rosen/repomix/examples/kr1k-rerank.md, docs/rosen/repomix/repomix.rerank.config.json, docs/rosen/repomix/README.md, workspace/kr_time_budget/run_architecture_examples.py, workspace/kr_time_budget/pipeline/architecture_examples.py, workspace/kr_time_budget/pipeline/prepare_architecture_examples.py, workspace/kr_time_budget/pipeline/run_architecture_examples.py, workspace/kr_time_budget/tests/test_architecture_examples.py, workspace/kr_time_budget/回复/新架构示例/用法.md, workspace/kr_time_budget/回复/新架构示例/实验报告.md, src/rosen/datasets/kuairand.py, src/rosen/models/rerank/**/*.py, docs/rosen/details/RERANK.md, tests/rosen/models/rerank/*.py, workspace/kr_time_budget/回复/新架构示例/Prediction迁移.md, workspace/kr_time_budget/回复/新架构示例/verify_prediction_contract.py
- Files matching these patterns are excluded: **/__pycache__/**, **/*.pyc, **/.DS_Store, docs/rosen/repomix/rosen-core.md, docs/rosen/repomix/kr1k-*.md, docs/rosen/repomix/manifest.json
- Files matching default ignore patterns are excluded
- Content has been formatted for parsing in markdown style

# User Provided Header
新示例二：KR1K 六视频一屏 MultiGE 长播重排。完整实验与 rerank 模型组件；第一阶段冻结先验代码一并保留。 先读 docs/rosen/repomix/examples/kr1k-rerank.md。附件为阅读包，运行请使用仓库源文件路径。

# Directory Structure
````
docs/
  rosen/
    details/
      RERANK.md
    repomix/
      examples/
        kr1k-rerank.md
      README.md
      repomix.rerank.config.json
src/
  rosen/
    datasets/
      kuairand.py
    models/
      rerank/
        components/
          __init__.py
          attention.py
          diversity.py
          generators.py
          retrieval.py
          scoring.py
        __init__.py
        base.py
        dcn.py
        dnn.py
        extr.py
        kuaishou.py
        mge_metric.py
        mge.py
        nar.py
        pier_submodular.py
        pier.py
        presets.py
        prm.py
        prs.py
        seq2slate.py
        slate.py
tests/
  rosen/
    models/
      rerank/
        test_audit_regressions.py
        test_encoded_judging.py
        test_mge_rerank.py
        test_partial_judging.py
        test_reference_models.py
        test_shared_scoring.py
workspace/
  kr_time_budget/
    pipeline/
      architecture_examples.py
      prepare_architecture_examples.py
      run_architecture_examples.py
    tests/
      test_architecture_examples.py
    回复/
      新架构示例/
        Prediction迁移.md
        verify_prediction_contract.py
        实验报告.md
        用法.md
    run_architecture_examples.py
````

# Files

## File: docs/rosen/details/RERANK.md
````markdown
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
````

## File: docs/rosen/repomix/examples/kr1k-rerank.md
````markdown
# 示例二：KR1K 六视频重排

这是旧六视频预算实验数据上的新架构示例：保留分屏与切分，监督使用原始 `long_view`，不宣称与旧预算模型的 `y_continue` 目标等价。附件包含完整实验源码和框架 rerank 子包。

阅读顺序（实验路径位于 `workspace/kr_time_budget/`）：

1. `pipeline/architecture_examples.py`：`rerank_manager` 用 SlateLabel 声明两种列表标签与 mask；当前屏和历史屏共享四个原子 setting；`make_reranker` 显式选择长播目标。
2. `pipeline/prepare_architecture_examples.py`：`group_slates` 按 request/slot 组 6 视频列表，只读取同 session 的前两屏历史；`prepare_rerank` 按 event_id 对齐冻结 Pointwise 先验。
3. `src/rosen/models/rerank/base.py`、`components/*.py`、`mge.py`：列表算子、OAM、PPML/SimHash、四个生成器与共享 OCPM；详细契约见同包 `docs/rosen/details/RERANK.md`。
4. `pipeline/architecture_examples.py`：`Prediction.target` 声明长播，Manager 自动查 mask；RankingEvaluator 评价最终排列，按 outputs_candidate_pool 能力注册 BatchResolver.for_candidates()，读取类型化候选分数并对齐标签/mask 后交给 SlateValueEvaluator。
5. `pipeline/run_architecture_examples.py`：`main` 先冻结 Pointwise 先验，再准备列表、调用 `train_example("rerank", ...)`，最终导出全部 test 排列。

输入形状：当前 `[B,6,4,8]`，拍扁历史 `[B,12,4,8]` 在模型内折为 `[B,2,6,4,8]`；长播标签/mask 为 `[B,6]`；pred.order 为 `[B,6]`。

仓库根目录运行共同入口：

```bash
uv run python -m workspace.kr_time_budget.run_architecture_examples
```

默认两个任务各完整训练 1 轮并测完整 valid/test。共享模块保留第一阶段代码，因为冻结先验是第二阶段的实际依赖；train 先验为 in-sample，未做 OOF，valid/test 由冻结模型生成。输入依赖已准备的 `data/v3/` 与 KR1K 侧表，数据根由 `ROSEN_DATASOURCE_ROOT` 指定。

本次 train 1,721,745 屏、test 48,898 屏；会话时间切分与 Pointwise 相同。长播 AUC 0.755150；最终排列 NDCG@6=0.470777，冻结先验=0.486740，未取得排序收益。日志：`workspace/kr_time_budget/runs/architecture_full_20261117/logs/train.log`。完整口径与限制见包内实验报告。

本例 `lc` 为离散无梯度多样性统计项；scores 承载原序长播概率，chosen_scores 与最终 order 对齐，candidates 同时承载主候选分数与排列。不要将 OCPM 分数排序、最终生成排列或候选估计均分混作同一个指标。

当前输出为 Prediction(scores/order/chosen_scores/candidates/target/extra)，已移除主候选 extra 键；同包 `Prediction迁移.md` 是早前字典迁移的完整 test 核验记录。本次类型化验证使用同权重合成 batch，不重新声称完整 test 复测。
````

## File: docs/rosen/repomix/README.md
````markdown
# rosen 源码与 KR1K 示例阅读包

四份导出文件可分别发送，也可一起阅读：

| 文件 | 内容 |
| --- | --- |
| [rosen-core.md](rosen-core.md) | 当前框架核心链路，含 stack、SlateLabel、共享拟合、Prediction/EvalBatch、完整 MSR、unified 与 rerank 子包 |
| [kr1k-multitask.md](kr1k-multitask.md) | 新示例一：逐曝光 CTR + 长播双任务，含模型、数据准备、训练、评估及专项测试 |
| [kr1k-rerank.md](kr1k-rerank.md) | 新示例二：六视频一屏、两屏历史的 MultiGE 长播重排，含完整模型组件与实验链路 |
| [kr1k-original.md](kr1k-original.md) | 原始 KR1K 示例 `docs/rosen/details/kuairand1k.py`，原文保留，附数据路径适配器 |

入选文件完整保留，不删除注释、空行或函数体，不做代码压缩。这些是阅读包，不是可单独运行的源码发行包；运行时使用仓库中的原始路径。两个新示例共用准备和训练模块，两个附件都保留这些完整文件，任务阅读入口分别见包内导读。

## 数据怎么走

```text
实验脚本读取、join、过滤原始表（Polars LazyFrame）
  -> 显式切分 train / valid / test，或 SchemaManager.split_dataset()
  -> SchemaManager.fit(train_raw)：拟合词表、数值统计量，保存 schema
  -> SchemaManager.transform(split)：按 schema 编码各 split
  -> SchemaManager.save_as_dataset()：写入 parquet
  -> ParquetStreamDataset：扫描、worker 分片、可选 shuffle、组 batch
  -> TensorFormatter：按字段 formatter 转 tensor、截断和 padding
  -> Interaction：字段名 -> batch tensor
  -> OmniEmbLayer：按字段/source/序列组取 embedding
  -> 模型 calculate_loss() / predict()
  -> Trainer：更新参数、评估、早停、checkpoint
```

原始数据 join、过滤、切分规则和特征选择由实验脚本明确决定。`EmbSetting` 子类定义每个字段的 fit、transform、tensor formatter 和 embedding 取值方式；`SchemaManager` 负责组织这些规则。

`fit()` 发现 `schema_dir/feature_meta.json` 已存在时直接加载。schema 保存各 setting 状态，词表另存到相对路径 `vocab/*.parquet`。共享同一个原子 setting 的序列只拟合一次，以 settings 顺序中第一来源为准。`fields()` 汇总 setting 输出列，以及时间、label、mask、domain 字段；dataset 还支持通过 `extra_col_names` / `extra_col_formatters` 保留额外列。

标签通过字符串或 `SlateLabel` 声明，`label_field` / `label_fields` 始终返回字符串。列表标签与 mask 由 `TensorFormatter` 按声明截断、padding；它们不进入特征序列组。

已编码 parquet 走 `ParquetStreamDataset`；原始 parquet 需要边读边编码时走 `RawParquetStreamDataset`。两者已经产出 batch，外层 `DataLoader` 使用 `batch_size=None`。验证、测试通常设置 `shuffle=False, drop_last=False`，避免丢失尾批。

profile 路径由 `ProfileDatasetBundle` 读取 manifest、schema 和 split。`ProfileParquetStreamDataset` 通过 `ProfileSequenceStore` 按 owner 和 cursor 取历史前缀，再按时间规则过滤、按 max_len 截断，最后格式化为 `Interaction`。这条路径复用侧表中的历史，避免每行重复存储整段历史。

## batch 之后怎么走

`BaseModel` 持有 manager、标签属性、`OmniEmbLayer`、`LossPort` 和 `ObservationPort`。`OmniEmbLayer.whole` 汇合普通用户/物品/交互特征；序列通过 `seq_groups` 的视图单独读取，`split_by="stack"` 保留字段轴 `[B, ..., F, D]`。`SharedBottomModel` 是按 domain 选 tower 的多场景例子；所有内置 MSR/unified 子类统一提供 logits(inter)，基类 predict 返回 sigmoid 概率 Tensor，无需每个模型构造 Prediction；KR1K 多任务例子则共享 backbone，独立预测 CTR 与长播，两者语义不同。

`BaseTrainer` 的默认 step 是 `zero_grad -> calculate_loss -> lsp.settle -> backward -> clip_grad_norm -> optimizer.step`。模型实现 `CustomTrainStepProtocol.custom_train_step(batch, ctx)` 时接管该 step 的优化过程，并负责结算 loss、返回标量 Tensor。epoch/eval 周期方法由 `BaseTrainerWithLife` 解释；EMA 和两阶段运行也有独立实现。

默认评估调用实际模型的 `predict()`，将裸 Tensor 包装成 `Prediction(scores=..., target=model.LABEL)`，
显式 Prediction 则原样保留，覆盖 EMA teacher。Manager 自动绑定 schema，`register(target=...)`
或 Prediction.target 同时确定标签与同名预测，并查询 SlateLabel mask。多标签未指定目标时明确报错。
PointwiseEvaluator 展平有效曝光；RankingEvaluator 按候选分数或最终排列评价排序，SlateValueEvaluator 评价已对齐的整刷价值。

所有 evaluator 统一继承 BaseEvaluator，实现 `collect(EvalBatch)`、`summary()`、`clear()`。
EvalContext 保存已过滤的 Prediction/Interaction，BatchResolver 通过 Scorer 解析成只有 Tensor 的 EvalBatch。
KR1K 根据 outputs_candidate_pool 能力使用 BatchResolver.for_candidates() 读取 CandidatePool，
对齐标签/mask 后交给 SlateValueEvaluator。chosen_scores 为一等字段，导出统一使用 resolve_chosen_scores()；extra 仅作诊断。
RetrievalEvaluator 消费 full_scores，并在副本上屏蔽历史；默认 Trainer 不自动调用 full_sort_predict。
旧字典输出与 collect_batch 路由需迁移；自定义评价器统一继承 BaseEvaluator，实现 collect(EvalBatch)。
接口只提供规范名；旧类名、targets/target_scorer 参数及 candidates 视图已删除。
Scorer 自己声明 schema/target 依赖，同一 collect 复用过滤和 CPU 搬运；评价器按 batch 缓存数组。
裁判统一走 SlateRerankModel 的预编码/分块路径，候选 mask 显式传入；OCPM 历史页和 KuaiShou 历史 K/V 在分块前预编码，Interaction 不再注入控制字段。

checkpoint 保存模型、主 optimizer、epoch、global_step，以及模型通过 `TrainerStateProtocol` 提供的额外训练状态。`run()` 根据验证指标执行早停和 best 保存；不会自动重载 best 后测试，需要入口显式安排。

## 源码阅读顺序

下列路径均为仓库相对路径，可在导出文件中搜索对应的 `## File:`：

1. `src/rosen/emb/manager.py`、`emb/schema/*.py`、`emb/split.py`：字段规则、编码、切分、持久化。
2. `src/rosen/data/dataset.py`、`data/padding.py`、`data/profile.py`、`core/interaction.py`：parquet 到 batch，以及 profile 历史装配。
3. `src/rosen/emb/emblayer.py`、`models/base.py`、`models/msr/sharedbottom.py`：batch 到 embedding 和模型输出。
4. `src/rosen/core/trainers/base.py`、`core/train/{context,hooks,loss}.py`：优化步骤和扩展契约。
5. `src/rosen/core/prediction.py`、`src/rosen/evaluate/{batch,manager,evaluator,metrics}.py`：语义预测、统一评估上下文与指标。
6. `src/rosen/experiment/{config,param}.py`：工作目录、配置和 CLI 参数。
7. `src/rosen/models/rerank/{base,mge}.py` 与 `components/*.py`：列表算子、共享注意力、粗筛、生成器、OCPM 与梯度边界。

核心包附完整 MSR、unified 与 rerank 子包、公共 logits/predict 契约及相关回归测试。观测附 BaseModel 使用的 port 及直接消费 logits 的审计路径。KR1K 实验代码放在独立附件，数据、日志、checkpoint 不打包。

## 三个示例怎么读

- [多任务导读](examples/kr1k-multitask.md)：先看 `pointwise_manager`、`KRPointwiseMultiTaskModel`，再看 `evaluators("pointwise")` 与 `train_example`。
- [重排导读](examples/kr1k-rerank.md)：先看 `rerank_manager`、`make_reranker`、`group_slates`，再看 MultiGE、RankingEvaluator 与 SlateValueEvaluator。
- [原始示例导读](examples/kr1k-original.md)：原始单标签、多场景脚本保持原样，历史切分用法按实际代码说明。

历史两个新示例已各跑完整训练集 1 轮、完整 valid/test；早前 Prediction 迁移以原 checkpoint 复测完整 test，详情见附件报告。最终排列 NDCG@6 低于冻结先验，未取得排序收益。本次候选池类型化通过完整测试和 11 类候选模型的同权重合成 batch 逐值核对，未重跑完整数据集实验；候选预测均分不代表真实反馈收益。

## 重新生成

在仓库根目录运行（本次使用 Repomix 1.18.1）：

```bash
uv run python docs/rosen/repomix/rebuild.py
```

需要本机 `repomix` CLI（本次 1.18.1）。脚本按四份配置重新生成，逐文件核对代码块与源文件内容，并写入 [manifest.json](manifest.json) 记录文件清单、字节数和 SHA-256。只生成核心包仍可运行 `repomix --config docs/rosen/repomix/repomix.config.json`。

选取范围以本目录 `repomix*.config.json` 为准。Repomix 使用 Markdown 代码围栏包装完整文件，可能规范化文件首尾空白；文件内部的注释、空行和代码保持原样。修改源码后重新生成，不直接编辑生成的四份阅读包。
````

## File: docs/rosen/repomix/repomix.rerank.config.json
````json
{
  "output": {
    "filePath": "docs/rosen/repomix/kr1k-rerank.md",
    "style": "markdown",
    "parsableStyle": true,
    "headerText": "新示例二：KR1K 六视频一屏 MultiGE 长播重排。完整实验与 rerank 模型组件；第一阶段冻结先验代码一并保留。 先读 docs/rosen/repomix/examples/kr1k-rerank.md。附件为阅读包，运行请使用仓库源文件路径。",
    "fileSummary": true,
    "directoryStructure": true,
    "removeComments": false,
    "removeEmptyLines": false,
    "compress": false,
    "git": {
      "sortByChanges": false
    }
  },
  "include": [
    "docs/rosen/repomix/examples/kr1k-rerank.md",
    "docs/rosen/repomix/repomix.rerank.config.json",
    "docs/rosen/repomix/README.md",
    "workspace/kr_time_budget/run_architecture_examples.py",
    "workspace/kr_time_budget/pipeline/architecture_examples.py",
    "workspace/kr_time_budget/pipeline/prepare_architecture_examples.py",
    "workspace/kr_time_budget/pipeline/run_architecture_examples.py",
    "workspace/kr_time_budget/tests/test_architecture_examples.py",
    "workspace/kr_time_budget/回复/新架构示例/用法.md",
    "workspace/kr_time_budget/回复/新架构示例/实验报告.md",
    "src/rosen/datasets/kuairand.py",
    "src/rosen/models/rerank/**/*.py",
    "docs/rosen/details/RERANK.md",
    "tests/rosen/models/rerank/*.py",
    "workspace/kr_time_budget/回复/新架构示例/Prediction迁移.md",
    "workspace/kr_time_budget/回复/新架构示例/verify_prediction_contract.py"
  ],
  "ignore": {
    "useGitignore": false,
    "useDotIgnore": false,
    "useDefaultPatterns": true,
    "customPatterns": [
      "**/__pycache__/**",
      "**/*.pyc",
      "**/.DS_Store",
      "docs/rosen/repomix/rosen-core.md",
      "docs/rosen/repomix/kr1k-*.md",
      "docs/rosen/repomix/manifest.json"
    ]
  },
  "security": {
    "enableSecurityCheck": true
  }
}
````

## File: src/rosen/datasets/kuairand.py
````python
import polars as pl
from rosen.datasets.base import DatasetBase
import pprint

class KuaiRandDataset(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "KuaiRand"
    DATA_KUAI_1K = BASE_DIR / "KuaiRand-1K" / "data"
    STD_LOG_FORMER_DATA = DATA_KUAI_1K / "log_standard_4_08_to_4_21_1k.csv"
    STD_LOG_FORMER_DATA_P1 = STD_LOG_FORMER_DATA
    STD_LOG_FORMER_DATA_P2 = DATA_KUAI_1K / "log_standard_4_22_to_5_08_1k.csv"
    RAND_LOG_FORMER_DATA = DATA_KUAI_1K / "log_random_4_22_to_5_08_1k.csv"
    USER_FEATURES = DATA_KUAI_1K / "user_features_1k.csv"
    VIDEO_FEATURES = DATA_KUAI_1K / "video_features_basic_1k.csv"

# ==========================================
# 执行与测试示例
# ==========================================
if __name__ == "__main__":
    # 为了防止路径不存在报错，这里用 try-except 包裹文件读取逻辑
    # 请确保你的实际路径中存在这些 CSV 文件
    from rosen.datasets.overview import get_general_info, get_head_info, get_group_stats
    try:
        # 读取三个文件 (这里假设存在且格式标准)
        df_log = pl.read_csv(KuaiRandDataset.STD_LOG_FORMER_DATA_P2)
        df_user = pl.read_csv(KuaiRandDataset.USER_FEATURES)
        df_video = pl.read_csv(KuaiRandDataset.VIDEO_FEATURES)

        datasets = {
            "Log Standard": df_log,
            "User Features": df_user,
            "Video Features": df_video
        }

        # 1. 测试通用信息打印
        print("=== 一、通用信息统计 ===")
        for name, df in datasets.items():
            info = get_general_info(df)
            print(f"\n[{name}] 数据集统计:")
            # 使用 pprint 打印字典，结构清晰且不会省略数据
            pprint.pprint(info, depth=4, compact=False)

        # 2. 测试分组统计函数 (以 df_log 中的某个假设列为例，例如 'is_click')
        print("\n=== 二、分组统计 ===")
        # 这里假设 df_log 有一个 'is_click' 列，你可以替换为你实际想看的列名
        test_col = "tab"
        group_stats = get_group_stats(df_log, test_col)
        print(f"对 '{test_col}' 的分组统计结果:")
        pprint.pprint(group_stats)

        # 3. 测试 Head(3) 视图
        print("\n=== 三、Head(3) 完整数据及类型视图 ===")
        head_view = get_head_info(df_user)  # 以 User Features 为例
        print("[User Features] 前三行完整数据:")
        pprint.pprint(head_view)

    except FileNotFoundError as e:
        print(f"文件未找到，请检查路径: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
````

## File: src/rosen/models/rerank/components/__init__.py
````python
from .attention import (IntraAttrSelfAttention, InterAttrSelfAttention, OAMBlock,
                        IntraPOISelfAttention, PageTargetAttention)
from .retrieval import PPMLSimHashRetriever, BeamSearchRetriever
from .scoring import EncodedSlate, OCPMContext, score_candidate_logits, score_candidate_probs
from .generators import (PointwiseGenerator, ListwiseGenerator, AutoregressiveGenerator,
                         GenerativeGenerator, SubmodularityGenerator, PointerHead,
                         lc_exact_score, lc_soft_score)

__all__ = ["IntraAttrSelfAttention", "InterAttrSelfAttention", "OAMBlock", "IntraPOISelfAttention",
           "PageTargetAttention", "PPMLSimHashRetriever", "BeamSearchRetriever", "PointwiseGenerator",
           "ListwiseGenerator", "AutoregressiveGenerator", "GenerativeGenerator", "SubmodularityGenerator",
           "PointerHead", "lc_exact_score", "lc_soft_score", "EncodedSlate", "OCPMContext",
           "score_candidate_logits", "score_candidate_probs"]
````

## File: src/rosen/models/rerank/components/attention.py
````python
"""Vectorized attention with the reference MGE residual structure and scale."""

import torch
from torch import nn


def masked_softmax(logits, valid):
    """Zero weights for empty rows as well as masked keys."""
    weights = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(-1) * valid
    return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)


class _Attention(nn.Module):
    def __init__(self, num_tokens, emb_dim, scale=8.0):
        super().__init__()
        if num_tokens <= 0 or emb_dim <= 0 or scale <= 0:
            raise ValueError("Attention dimensions and scale must be positive.")
        self.scale = scale
        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.pool = nn.Linear(num_tokens * emb_dim, emb_dim, bias=False)
        # All parameters are independent 2D linear weights: no hidden fan-in axis.
        for layer in (self.q, self.k, self.v, self.pool):
            nn.init.xavier_uniform_(layer.weight)

    def attend(self, x):
        weights = (self.q(x) @ self.k(x).transpose(-1, -2) / self.scale).softmax(-1)
        return weights @ self.v(x)


class IntraAttrSelfAttention(_Attention):
    def __init__(self, num_pois, num_attrs, emb_dim, scale=8.0):
        super().__init__(num_pois, emb_dim, scale)
        self.num_pois, self.num_attrs, self.emb_dim = num_pois, num_attrs, emb_dim

    def forward(self, x):
        if x.shape[-3:] != (self.num_pois, self.num_attrs, self.emb_dim):
            raise ValueError("Intra-attribute attention expects [..., K, F, D] matching its configuration.")
        per_attr = x.transpose(-3, -2)  # [..., F, K, D]
        delta = self.attend(per_attr)
        pooled = self.pool((per_attr + delta).flatten(-2))
        return pooled, delta.transpose(-3, -2)


class InterAttrSelfAttention(_Attention):
    def __init__(self, num_attrs, emb_dim, scale=8.0):
        super().__init__(num_attrs, emb_dim, scale)

    def forward(self, x):
        return self.pool((x + self.attend(x)).flatten(-2))


class OAMBlock(nn.Module):
    def __init__(self, num_pois, num_attrs, emb_dim, scale=8.0):
        super().__init__()
        self.intra = IntraAttrSelfAttention(num_pois, num_attrs, emb_dim, scale)
        self.inter = InterAttrSelfAttention(num_attrs, emb_dim, scale)

    def forward(self, x):
        pooled, per_poi = self.intra(x)
        return self.inter(pooled), per_poi


class IntraPOISelfAttention(InterAttrSelfAttention):
    """Uses the POI axis as a batch axis, attending only across its attributes."""


class PageTargetAttention(nn.Module):
    def forward(self, target, behavior, page_mask):
        if behavior.ndim != 3 or page_mask.shape != behavior.shape[:2]:
            raise ValueError("Expected history [B,P,D] and page_mask [B,P].")
        if target.shape[-1] != behavior.shape[-1]:
            raise ValueError("Target and history embedding widths must agree.")
        if target.ndim == 3:  # [B,M,D], preserve sample-major layout by broadcasting
            if target.shape[0] != behavior.shape[0]:
                raise ValueError("Structured targets and history must have the same batch size.")
            behavior = behavior[:, None]
            mask = page_mask[:, None]
        elif target.ndim == 2:
            if target.shape[0] % behavior.shape[0]:
                raise ValueError("Target batch must be a multiple of the history batch.")
            count = target.shape[0] // behavior.shape[0]
            behavior = behavior.repeat_interleave(count, dim=0)
            mask = page_mask.repeat_interleave(count, dim=0)
        else:
            raise ValueError("Expected targets [B,D], [B*M,D] or [B,M,D].")
        valid = mask == 1
        logits = (target.unsqueeze(-2) * behavior).sum(-1)
        weights = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(-1)
        weights = weights * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        return (weights.unsqueeze(-1) * behavior).sum(-2)
````

## File: src/rosen/models/rerank/components/diversity.py
````python
"""Per-sample candidate statistics with explicit definitions (no fixed batch size)."""
import torch
from torch.nn import functional as F

from .generators import lc_exact_score


def candidate_statistics(embeddings, orders):
    flat = embeddings.flatten(2)
    pairs = torch.triu_indices(flat.shape[1], flat.shape[1], 1, device=flat.device)
    result = {"candidate_lc_index": lc_exact_score(orders)}
    if pairs.shape[1]:
        a, b = flat[:, pairs[0]], flat[:, pairs[1]]
        l2 = (a - b).norm(dim=-1)
        cosine = 1 - F.cosine_similarity(a, b, dim=-1)
        result.update(candidate_mean_l2=l2.mean(-1), candidate_max_l2=l2.amax(-1),
                      candidate_mean_cosine_distance=cosine.mean(-1), candidate_max_cosine_distance=cosine.amax(-1))
    else:
        result.update({name: flat.new_zeros(flat.shape[0]) for name in (
            "candidate_mean_l2", "candidate_max_l2", "candidate_mean_cosine_distance", "candidate_max_cosine_distance")})
    return result
````

## File: src/rosen/models/rerank/components/generators.py
````python
"""Independent encoders/decoders sharing caller-owned attention and pointer heads."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, widths, activation):
    layers = []
    for width in widths:
        if width <= 0:
            raise ValueError("MLP widths must be positive.")
        layer = nn.Linear(input_dim, width)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.extend((layer, activation()))
        input_dim = width
    return nn.Sequential(*layers), input_dim


class ContextEncoder(nn.Module):
    def __init__(self, num_attrs, emb_dim, mlp_a=(60, 32, 20)):
        super().__init__()
        self.mlp_a, width = mlp(num_attrs * emb_dim, mlp_a, nn.SiLU)
        self.output_dim = 3 * emb_dim + 1 + width

    def forward(self, cate, history, dense, page_mask, oam, intra_poi, target_attention):
        hist_pages, _ = oam(history)
        return self.encode_slate(cate, dense, hist_pages, page_mask, oam, intra_poi, target_attention)

    def encode_slate(self, cate, dense, hist_pages, page_mask, oam, intra_poi, target_attention):
        page, per_poi = oam(cate)
        interest = target_attention(page, hist_pages, page_mask)
        k = cate.shape[-3]
        return torch.cat((page.unsqueeze(-2).expand(*page.shape[:-1], k, -1),
                          interest.unsqueeze(-2).expand(*interest.shape[:-1], k, -1), intra_poi(per_poi),
                          dense, self.mlp_a(cate.flatten(-2))), dim=-1)


class PointwiseGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, selected_prior):
        logits = self.linear(selected_prior).squeeze(-1)
        return logits, logits.argsort(dim=-1, descending=True, stable=True)[:, None]


class ListwiseGenerator(nn.Module):
    """Also serves as the supervised OCPM, with an independently owned encoder."""

    def __init__(self, num_attrs, emb_dim, mlp_a=(60, 32, 20), mlp_b=(128, 32)):
        super().__init__()
        self.encoder = ContextEncoder(num_attrs, emb_dim, mlp_a)
        self.mlp_b, width = mlp(self.encoder.output_dim, mlp_b, nn.ReLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, cate, history_pages, dense, page_mask, oam, intra_poi, target_attention):
        features = self.encoder.encode_slate(cate, dense, history_pages, page_mask, oam, intra_poi, target_attention)
        logits = self.output(self.mlp_b(features)).squeeze(-1)
        return logits, logits.argsort(dim=-1, descending=True, stable=True).unsqueeze(-2)


class PointerHead(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.layers, width = mlp(hidden_dim * 2, (64, 16), nn.ReLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, candidates, state):
        features = torch.cat((candidates, state[:, None].expand_as(candidates)), -1)
        return self.output(self.layers(features)).squeeze(-1)


@dataclass
class AutoregressiveOutput:
    indices: torch.Tensor       # [B,1,K]; training argmax records need not be permutations
    probabilities: torch.Tensor  # [B,K,K], step then item
    max_scores: torch.Tensor
    loss: torch.Tensor


class AutoregressiveGenerator(nn.Module):
    def __init__(self, num_attrs, emb_dim, mlp_a=(60, 32, 20), hidden_dim=64):
        super().__init__()
        self.encoder = ContextEncoder(num_attrs, emb_dim, mlp_a)
        self.project = nn.Linear(self.encoder.output_dim, hidden_dim)
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, cate, history, dense, page_mask, oam, intra_poi, target_attention,
                pointer_head, *, labels=None, label_mask=None, coverage=None, teacher_forcing=None):
        features = self.project(self.encoder(cate, history, dense, page_mask, oam, intra_poi, target_attention))
        b, k, _ = features.shape
        teacher_forcing = self.training if teacher_forcing is None else teacher_forcing
        if teacher_forcing and labels is None:
            raise ValueError("Teacher forcing requires labels.")
        state = features.mean(1).detach()
        next_input = torch.zeros_like(state)
        available = torch.ones(b, k, dtype=torch.bool, device=features.device)
        probabilities, indices, maxima = [], [], []
        loss = features.new_zeros(())
        rows = torch.arange(b, device=features.device)
        for step in range(k):  # Sequential decoding is inherently autoregressive.
            state = self.gru(next_input, state)
            logits = pointer_head(features, state).masked_fill(~available, -torch.inf)
            probs = logits.softmax(-1)
            chosen = probs.argmax(-1)
            indices.append(chosen)
            probabilities.append(probs)
            maxima.append(probs.amax(-1))
            selected = torch.full_like(chosen, step) if teacher_forcing else chosen
            if teacher_forcing:
                reward = labels[:, step]
                if coverage is not None:
                    reward = reward - coverage[:, step]
                if label_mask is not None:
                    reward = torch.where(label_mask[:, step] > 0, reward, 0) * label_mask[:, step]
                epsilon = 1e-6 if coverage is not None else 1e-9
                loss = loss - (reward * probs[:, step].clamp(epsilon, 1).log()).mean()
            available = available.scatter(1, selected[:, None], False)
            next_input = features[rows, selected].detach()
        return AutoregressiveOutput(torch.stack(indices, 1)[:, None], torch.stack(probabilities, 1),
                                    torch.stack(maxima, 1), loss)


# The two decoders have separate parameters; sharing is restricted to caller-owned heads/attention.
GenerativeGenerator = AutoregressiveGenerator
SubmodularityGenerator = AutoregressiveGenerator


def cosine_coverage(cate, other_slates):
    dot = (cate[:, None] * other_slates).sum((-1, -2))
    norm = cate[:, None].square().sum((-1, -2)).sqrt() * other_slates.square().sum((-1, -2)).sqrt()
    return (dot / norm.clamp_min(1e-4)).mean(1).detach()


def lc_exact_score(permutations):
    """Reference index-comparison statistic, not an inverse-permutation distance.

    Shape [B,G,K]. This deliberately preserves the original TF definition even
    for repeated training argmax indices. Discrete indices have no gradient.
    """
    relations = permutations[..., :, None] < permutations[..., None, :]
    pairs = torch.triu_indices(permutations.shape[1], permutations.shape[1], 1, device=permutations.device)
    return (relations[:, pairs[0]] != relations[:, pairs[1]]).float().sum((-1, -2, -3))


def lc_soft_score(probabilities, reference_permutations, tau=1.0):
    """Sum squared precedence differences over every pair, using one soft order.

    This is an experimental objective, not a relaxation numerically equivalent
    to lc_exact_score's comparisons of raw permutation indices.
    """
    if tau <= 0:
        raise ValueError("lc_soft temperature must be positive.")
    k = probabilities.shape[-1]
    steps = torch.arange(k, dtype=probabilities.dtype, device=probabilities.device)
    expected = (probabilities * steps[None, :, None]).sum(1)
    soft = torch.sigmoid((expected[:, None, :] - expected[:, :, None]) / tau)
    # Convert each reference order to positions per item. Training orders may
    # repeat; one-hot membership retains the explicitly recorded behavior.
    ref_positions = (F.one_hot(reference_permutations, k).to(probabilities.dtype) * steps[None, None, :, None]).sum(-2)
    reference = (ref_positions[..., None, :] > ref_positions[..., :, None]).to(probabilities.dtype).detach()
    diagonal = ~torch.eye(k, dtype=torch.bool, device=probabilities.device)
    soft_pairs = ((soft[:, None] - reference).square() * diagonal).sum((-1, -2, -3))
    # Reference-reference pairs are constant w.r.t. the learned order.
    # Only the soft-to-reference term belongs in the optimization objective.
    return soft_pairs
````

## File: src/rosen/models/rerank/components/retrieval.py
````python
"""Fixed model constants and exhaustive-permutation coarse ranking."""

import torch
from torch import nn

from ..base import select_by_order, build_full_permutations


def _constant(value, shape, default):
    tensor = default if value is None else torch.as_tensor(value, dtype=torch.float32).clone()
    if tensor.shape != shape or not torch.isfinite(tensor).all():
        raise ValueError(f"Expected finite constant with shape {tuple(shape)}, got {tuple(tensor.shape)}.")
    return tensor.detach()


class PPMLSimHashRetriever(nn.Module):
    def __init__(self, poi_num, page_num, emb_dim, *, position_encoding=None,
                 hash_vector=None, distance_weight=None, seed=2026):
        super().__init__()
        permutations = build_full_permutations(poi_num)
        if page_num <= 0 or emb_dim <= 0:
            raise ValueError("page_num and emb_dim must be positive.")
        # Local RNG: buffer creation never shifts shared trainable initialization.
        rng = torch.Generator().manual_seed(seed)
        self.register_buffer("full_permutations", permutations)
        self.register_buffer("position_encoding", _constant(
            position_encoding, (poi_num, emb_dim), torch.rand(poi_num, emb_dim, generator=rng)))
        self.register_buffer("hash_vector", _constant(
            hash_vector, (page_num, emb_dim), torch.rand(page_num, emb_dim, generator=rng)))
        self.register_buffer("distance_weight", _constant(
            distance_weight, (1, page_num), (1 / (torch.arange(page_num) // 2 + 2).float())[None]))

    def pool(self, pages):
        """PPML averages position-weighted embeddings over POIs and attributes."""
        return (pages * self.position_encoding[:, None, :]).mean(dim=(-3, -2))

    def forward(self, slate, history, page_mask=None):
        if history.shape[1] != self.hash_vector.shape[0]:
            raise ValueError("History page count does not match SimHash configuration.")
        # [B,K!,K,D] is enough: average attributes before gathering permutations.
        per_poi = slate.mean(-2)
        candidates = select_by_order(per_poi, self.full_permutations)
        perm_signature = (candidates * self.position_encoding).mean(-2)
        hist_signature = self.pool(history)
        page_hash = torch.sign(hist_signature * self.hash_vector)
        perm_hash = torch.sign(perm_signature[:, :, None] * self.hash_vector)
        distance = (perm_hash - page_hash[:, None]).abs().sum(-1)
        weights = self.distance_weight
        if page_mask is not None:
            weights = weights * (page_mask == 1)
        return (distance * weights[:, None]).sum(-1)

    @staticmethod
    def top_k(distances, k):
        if not 1 <= k <= distances.shape[-1]:
            raise ValueError("Retrieval k must lie in [1, number of permutations].")
        return distances.argsort(dim=-1, stable=True)[:, :k]

    @staticmethod
    def last_k(distances, k):
        if not 1 <= k <= distances.shape[-1]:
            raise ValueError("Retrieval k must lie in [1, number of permutations].")
        return distances.argsort(dim=-1, descending=True, stable=True)[:, :k]


class BeamSearchRetriever(nn.Module):
    """Reference heuristic: rank exhaustive priors, not an iterative beam decoder."""

    def __init__(self, poi_num, expose_rate=None):
        super().__init__()
        default = (1.0 - 0.1 * torch.arange(poi_num)).clamp_min(0)[None]
        self.register_buffer("expose_rate", _constant(expose_rate, (1, poi_num), default))

    def forward(self, permutation_priors, top_k):
        if permutation_priors.ndim != 3 or permutation_priors.shape[-1] != self.expose_rate.shape[-1]:
            raise ValueError("Beam prior scores must be [B,K!,K].")
        if not 1 <= top_k <= permutation_priors.shape[1]:
            raise ValueError("top_k exceeds the available permutations.")
        return (permutation_priors * self.expose_rate).sum(-1).argsort(dim=-1, descending=True, stable=True)[:, :top_k]
````

## File: src/rosen/models/rerank/components/scoring.py
````python
"""One chunked tensor path for encoded candidate slates and fixed context."""
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..base import select_by_order


@dataclass(frozen=True)
class EncodedSlate:
    """Input-slot features [B,N,...], priors [B,N,1], mask [B,N].

    context is the model's typed encoding of fixed history. Its original batch
    axis is retained and shared across candidate chunks.
    """
    items: torch.Tensor
    dense: torch.Tensor
    mask: torch.Tensor
    context: Any = None


@dataclass(frozen=True)
class OCPMContext:
    history_pages: torch.Tensor
    page_mask: torch.Tensor


def _pad_slots(tensor, width):
    padding = width - tensor.shape[2]
    if not padding:
        return tensor
    zeros = tensor.new_zeros((*tensor.shape[:2], padding, *tensor.shape[3:]))
    return torch.cat((tensor, zeros), dim=2)


def score_candidate_logits(encoded: EncodedSlate, orders: torch.Tensor, *, scorer: Callable,
                           chunk_size: int = 32, candidate_dense: torch.Tensor | None = None):
    """Gather/right-pad K items and score [B,chunk,N,...] with an explicit mask."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    b, n = encoded.items.shape[:2]
    if encoded.dense.shape != (b, n, 1) or encoded.mask.shape != (b, n):
        raise ValueError("Encoded priors and mask must align to item slots [B,N].")
    if orders.ndim == 2:
        orders = orders[None].expand(b, -1, -1)
    if orders.ndim != 3 or orders.shape[0] != b or not orders.shape[1] or not 1 <= orders.shape[2] <= n:
        raise ValueError("Candidate orders require [B,M,K], M > 0 and 1 <= K <= N.")
    if candidate_dense is not None and candidate_dense.shape != (*orders.shape, 1):
        raise ValueError("Candidate dense scores must be [B,M,K,1].")
    outputs = []
    for start in range(0, orders.shape[1], chunk_size):
        chunk = orders[:, start:start + chunk_size]
        items = _pad_slots(select_by_order(encoded.items, chunk), n)
        dense = (select_by_order(encoded.dense, chunk) if candidate_dense is None else
                 candidate_dense[:, start:start + chunk.shape[1]])
        dense = _pad_slots(dense, n)
        mask = _pad_slots(select_by_order(encoded.mask, chunk), n)
        logits = scorer(items, dense, mask, encoded.context)
        if logits.shape != (b, chunk.shape[1], n):
            raise ValueError("Encoded slate scorer must return logits [B,M,N].")
        outputs.append(logits[..., :chunk.shape[-1]])
    return torch.cat(outputs, dim=1)


def score_candidate_probs(encoded: EncodedSlate, orders: torch.Tensor, *, scorer: Callable,
                          chunk_size: int = 32, candidate_dense: torch.Tensor | None = None):
    """All judges return probabilities with invalid selected slots zeroed."""
    logits = score_candidate_logits(encoded, orders, scorer=scorer,
                                    chunk_size=chunk_size, candidate_dense=candidate_dense)
    return logits.sigmoid() * select_by_order(encoded.mask, orders)
````

## File: src/rosen/models/rerank/__init__.py
````python
"""List-wise reranking models; independent from the multi-scenario registry."""

from .base import (RerankModel, fold_history_pages, build_full_permutations,
                   select_by_order, masked_bce_with_logits)
from .mge import MultiGERerankModel
from .slate import OCPMRerankModel, SlateRerankModel
from .dnn import DNNRerankModel
from .dcn import DCNRerankModel
from .prm import PRMRerankModel
from .extr import EXTRRerankModel
from .kuaishou import KuaiShouRerankModel
from .pier import PIERRerankModel
from .prs import PRSRerankModel
from .nar import NARRerankModel
from .seq2slate import Seq2SlateRerankModel
from .pier_submodular import PIERSubmodularRerankModel
from .mge_metric import MultiGEMetricRerankModel, SubmodularMultiGEMetricRerankModel


RERANK_MODEL_REGISTRY = {
    "dnn": DNNRerankModel, "dcn": DCNRerankModel, "prm": PRMRerankModel,
    "extr": EXTRRerankModel, "kuaishou": KuaiShouRerankModel,
    "pier": PIERRerankModel, "prs": PRSRerankModel,
    "nar": NARRerankModel, "seq2slate": Seq2SlateRerankModel,
    "pier_submodular": PIERSubmodularRerankModel,
    "mge_metric": MultiGEMetricRerankModel, "mge_cosine_metric": SubmodularMultiGEMetricRerankModel,
    "mge3": MultiGERerankModel, "mge_cosine": MultiGERerankModel,
    "mge": MultiGERerankModel, "lc_mge": MultiGERerankModel,
}

from .presets import resolve_rerank_config


def build_rerank_model(schema_manager, model_cls="mge", **model_kwargs):
    if isinstance(model_cls, str):
        name = model_cls.lower()
        if name not in RERANK_MODEL_REGISTRY:
            raise ValueError(f"Unknown rerank model '{model_cls}'. Available: {sorted(RERANK_MODEL_REGISTRY)}")
        model_kwargs = resolve_rerank_config(name, **model_kwargs)
        model_cls = RERANK_MODEL_REGISTRY[name]
    if not isinstance(model_cls, type) or not issubclass(model_cls, RerankModel):
        raise TypeError("model_cls must be a RerankModel subclass or registered name.")
    return model_cls.from_manager(schema_manager, **model_kwargs)


__all__ = ["RerankModel", "SlateRerankModel", "OCPMRerankModel", "MultiGERerankModel", "DNNRerankModel", "DCNRerankModel",
           "PRMRerankModel", "EXTRRerankModel", "KuaiShouRerankModel", "PIERRerankModel", "PRSRerankModel",
           "NARRerankModel", "Seq2SlateRerankModel", "RERANK_MODEL_REGISTRY", "build_rerank_model", "resolve_rerank_config",
           "PIERSubmodularRerankModel", "MultiGEMetricRerankModel", "SubmodularMultiGEMetricRerankModel",
           "select_by_order", "fold_history_pages", "build_full_permutations", "masked_bce_with_logits"]
````

## File: src/rosen/models/rerank/base.py
````python
"""Task-independent tensor operations for fixed-width slate models."""

import inspect
import itertools
import warnings
from typing import Literal

import torch
from torch.nn import functional as F

from rosen.models.base import BaseModel


def fold_history_pages(flat_seq: torch.Tensor, page_num: int, poi_num: int) -> torch.Tensor:
    if page_num <= 0 or poi_num <= 0 or flat_seq.ndim < 2:
        raise ValueError("History requires positive page_num/poi_num and shape [B, P*K, ...].")
    if flat_seq.shape[1] != page_num * poi_num:
        raise ValueError(f"Expected history length {page_num * poi_num}, got {flat_seq.shape[1]}.")
    return flat_seq.reshape(flat_seq.shape[0], page_num, poi_num, *flat_seq.shape[2:])


def build_full_permutations(poi_num: int) -> torch.Tensor:
    if isinstance(poi_num, bool) or not isinstance(poi_num, int) or not 1 <= poi_num <= 7:
        raise ValueError("Full permutations require an integer poi_num between 1 and 7 (factorial growth).")
    return torch.tensor(list(itertools.permutations(range(poi_num))), dtype=torch.long)


def select_by_order(slate_tensor: torch.Tensor, perm_indices: torch.Tensor) -> torch.Tensor:
    """Gather [B,N,...] using shared [M,K] or row-wise [B,M,K] indices.

    This tensor primitive also transports repeated training decoder choices;
    external scored orders are checked for uniqueness at score_permutations.
    """
    if slate_tensor.ndim < 2 or perm_indices.ndim not in (2, 3):
        raise ValueError("Expected slate [B,K,...] and permutations [M,K] or [B,M,K].")
    if perm_indices.dtype != torch.long or perm_indices.device != slate_tensor.device:
        raise ValueError("Permutation indices must be long tensors on the slate device.")
    if perm_indices.ndim == 2:
        perm_indices = perm_indices.unsqueeze(0).expand(slate_tensor.shape[0], -1, -1)
    b, m, k = perm_indices.shape
    if b != slate_tensor.shape[0] or m == 0 or not 1 <= k <= slate_tensor.shape[1]:
        raise ValueError("Order batch/slot dimensions require matching B, M > 0 and 1 <= K <= N.")
    if (perm_indices < 0).any() or (perm_indices >= slate_tensor.shape[1]).any():
        raise ValueError("Selection indices must be in [0,N).")
    rest = slate_tensor.shape[2:]
    indices = perm_indices.reshape(b, m * k, *((1,) * len(rest))).expand(b, m * k, *rest)
    return slate_tensor.gather(1, indices).reshape(b, m, k, *rest)


def masked_bce_with_logits(logits, labels, mask=None,
                           reduction: Literal["mean", "valid_mean", "nonzero_mean"] = "mean"):
    """Divide weighted BCE by slot count, weight sum, or nonzero weight count."""
    if logits.shape != labels.shape or (mask is not None and mask.shape != logits.shape):
        raise ValueError("Logits, labels and optional mask must have identical shapes.")
    if reduction not in ("mean", "valid_mean", "nonzero_mean"):
        raise ValueError("reduction must be 'mean', 'valid_mean' or 'nonzero_mean'.")
    weight = torch.ones_like(logits) if mask is None else mask.to(logits.dtype)
    # Invalid labels may be arbitrary sentinels; exclude them before evaluating BCE.
    safe_labels = torch.where(weight != 0, labels, 0).to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none") * weight
    if reduction == "mean":
        return loss.mean()
    denominator = weight.count_nonzero() if reduction == "nonzero_mean" else weight.sum()
    return loss.sum() / denominator.clamp_min(1)


class RerankModel(BaseModel):
    requires_full_slate = False
    outputs_candidate_pool = False
    select_by_order = staticmethod(select_by_order)
    fold_history_pages = staticmethod(fold_history_pages)
    build_full_permutations = staticmethod(build_full_permutations)
    masked_bce_with_logits = staticmethod(masked_bce_with_logits)

    def score_permutations(self, interaction, orders):
        """Return slot probabilities [B,M,K] for explicit selections, 1 <= K <= N."""
        raise NotImplementedError

    @classmethod
    def from_manager(cls, manager, **kwargs):
        accepted = set()
        for ancestor in cls.__mro__:
            parameters = inspect.signature(ancestor.__init__).parameters
            accepted.update(name for name, p in parameters.items()
                            if name not in ("self", "manager") and p.kind in (
                                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                break
        ignored = sorted(set(kwargs) - accepted)
        if ignored:
            warnings.warn(f"{cls.__name__}.from_manager ignored unsupported kwargs: {ignored}", stacklevel=2)
        return cls(manager, **{k: v for k, v in kwargs.items() if k not in ignored})
````

## File: src/rosen/models/rerank/dcn.py
````python
"""DCN with vector cross recurrence and the reference's concatenated cross states."""
import torch
from torch import nn

from .slate import SlateRerankModel
from .components.generators import mlp


class CrossNetwork(nn.Module):
    def __init__(self, width, num_layers):
        super().__init__()
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("cross_layers must be a positive integer.")
        self.weights = nn.ModuleList(nn.Linear(width, 1, bias=False) for _ in range(num_layers))
        self.biases = nn.ParameterList(nn.Parameter(torch.zeros(width)) for _ in range(num_layers))
        for layer in self.weights:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        state, states = x, [x]
        for weight, bias in zip(self.weights, self.biases):
            state = x * weight(state) + bias + state
            states.append(state)
        return torch.cat(states, -1)


class DCNRerankModel(SlateRerankModel):
    def __init__(self, manager, *, cross_layers=3, mlp_a=(60, 32, 20), mlp_b=(64, 32), **kwargs):
        super().__init__(manager, **kwargs)
        self.cross = CrossNetwork(self.item_dim, cross_layers)
        self.encoder, width = mlp(self.item_dim * (cross_layers + 2), mlp_a, nn.SiLU)
        self.head, width = mlp(width + 1, mlp_b, nn.SiLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_items(self, cate, dense, mask):
        x = cate.flatten(-2)
        # The source concatenates input_embedding and y_cross, which includes x0.
        return self.encoder(torch.cat((x, self.cross(x)), -1))

    def _score_encoded(self, items, dense, mask, context):
        return self.output(self.head(torch.cat((items, dense), -1))).squeeze(-1)
````

## File: src/rosen/models/rerank/dnn.py
````python
"""DNN baseline: independent per-item SiLU towers with a scalar prior."""
import torch
from torch import nn

from .slate import SlateRerankModel
from .components.generators import mlp


class DNNRerankModel(SlateRerankModel):
    def __init__(self, manager, *, mlp_a=(60, 32, 20), mlp_b=(128, 32), **kwargs):
        super().__init__(manager, **kwargs)
        self.encoder, width = mlp(self.item_dim, mlp_a, nn.SiLU)
        self.head, width = mlp(width + 1, mlp_b, nn.SiLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_items(self, cate, dense, mask):
        return self.encoder(cate.flatten(-2))

    def _score_encoded(self, items, dense, mask, context):
        return self.output(self.head(torch.cat((items, dense), -1))).squeeze(-1)
````

## File: src/rosen/models/rerank/extr.py
````python
"""EXTR: organic self-attention and causal ad-to-organic target attention."""
import math

import torch
from torch import nn

from .slate import SlateRerankModel
from .components.attention import masked_softmax
from .components.generators import mlp


class EXTRAttentionBlock(nn.Module):
    """Reference Q/K/V -> residual LayerNorm -> residual FFN (no output projection)."""
    def __init__(self, width, heads=4, attention_dropout=.1, ffn_dropout=.2):
        super().__init__()
        if heads <= 0 or width % heads:
            raise ValueError("EXTR item embedding width must be divisible by num_heads.")
        self.heads, self.head_width = heads, width // heads
        self.q, self.k, self.v = (nn.Linear(width, width) for _ in range(3))
        self.norm = nn.LayerNorm(width, eps=1e-8)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.ffn = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(ffn_dropout),
                                 nn.Linear(width, width), nn.Dropout(ffn_dropout))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, query, context, query_valid, context_valid, *, causal=False):
        b, t, width = query.shape
        s = context.shape[1]
        q = self.q(query).reshape(b, t, self.heads, self.head_width).transpose(1, 2)
        k = self.k(context).reshape(b, s, self.heads, self.head_width).transpose(1, 2)
        v = self.v(context).reshape(b, s, self.heads, self.head_width).transpose(1, 2)
        valid = context_valid[:, None, None, :]
        if causal:
            allowed = torch.arange(s, device=query.device)[None, :] <= torch.arange(t, device=query.device)[:, None]
            valid = valid & allowed[None, None]
        weights = masked_softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_width), valid)
        delta = (self.attention_dropout(weights) @ v).transpose(1, 2).reshape(b, t, width)
        x = self.norm(query + delta)
        return (x + self.ffn(x)) * query_valid[..., None]


class EXTRRerankModel(SlateRerankModel):
    requires_full_slate = True

    def __init__(self, manager, *, natural_indices=None, ad_indices=None, num_heads=4,
                 attention_dropout=.1, ffn_dropout=.2, mlp_a=(60, 32, 20), mlp_b=(128, 32), **kwargs):
        super().__init__(manager, **kwargs)
        natural = list(range(0, self.poi_num, 2)) if natural_indices is None else list(natural_indices)
        ads = list(range(1, self.poi_num, 2)) if ad_indices is None else list(ad_indices)
        if not natural or not ads or sorted(natural + ads) != list(range(self.poi_num)):
            raise ValueError("Non-empty natural_indices/ad_indices must partition every slate slot exactly once.")
        self.register_buffer("natural_indices", torch.tensor(natural, dtype=torch.long))
        self.register_buffer("ad_indices", torch.tensor(ads, dtype=torch.long))
        self.organic_attention = EXTRAttentionBlock(self.item_dim, num_heads, attention_dropout, ffn_dropout)
        self.ad_attention = EXTRAttentionBlock(self.item_dim, num_heads, attention_dropout, ffn_dropout)
        self.encoder, width = mlp(self.item_dim, mlp_a, nn.SiLU)
        self.head, width = mlp(len(ads) * self.item_dim + self.poi_num * (width + 1), mlp_b, nn.SiLU)
        self.output = nn.Linear(width, self.poi_num)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_items(self, cate, dense, mask):
        return cate.flatten(-2)

    def _score_encoded(self, items, dense, mask, context):
        b, m, n, width = items.shape
        valid = mask.flatten(0, 1) > 0
        dense = dense.flatten(0, 1)
        x = items.reshape(b * m, n, width) * valid[..., None]
        organic, ads = x[:, self.natural_indices], x[:, self.ad_indices]
        organic_valid, ad_valid = valid[:, self.natural_indices], valid[:, self.ad_indices]
        organic = self.organic_attention(organic, organic, organic_valid, organic_valid)
        target = self.ad_attention(ads, organic, ad_valid, organic_valid, causal=True)
        encoded = self.encoder(x) * valid[..., None]
        features = torch.cat((target.flatten(1), (dense * valid[..., None]).flatten(1), encoded.flatten(1)), -1)
        return self.output(self.head(features)).reshape(b, m, n)
````

## File: src/rosen/models/rerank/kuaishou.py
````python
"""KuaiShou reranker: target-conditioned residual pooling of current and prior items."""
from dataclasses import dataclass

import torch
from torch import nn

from .slate import SlateRerankModel
from .components.attention import masked_softmax
from .components.generators import mlp


@dataclass(frozen=True)
class TargetContextEncoding:
    items: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor


class TargetContextPool(nn.Module):
    def __init__(self, slots, width, scale=8.):
        super().__init__()
        if scale <= 0:
            raise ValueError("attention_scale must be positive.")
        self.scale = scale
        self.q, self.k, self.v = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.pool = nn.Linear(slots * width, width, bias=False)
        for layer in (self.q, self.k, self.v, self.pool):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, target, context, valid):
        return self.score_encoded(target, self.encode_context(context, valid))

    def encode_context(self, context, valid):
        return TargetContextEncoding(context, self.k(context), self.v(context), valid)

    def score_encoded(self, target, context: TargetContextEncoding):
        items, keys, values, valid = context.items, context.keys, context.values, context.valid
        while keys.ndim < target.ndim:
            items, keys, values, valid = (value.unsqueeze(1) for value in (items, keys, values, valid))
        weights = masked_softmax(self.q(target) @ keys.transpose(-1, -2) / self.scale, valid.unsqueeze(-2))
        delta = weights @ values
        residual = (items.unsqueeze(-3) + delta.unsqueeze(-2)) * valid.unsqueeze(-2).unsqueeze(-1)
        return self.pool(residual.flatten(-2))


class KuaiShouRerankModel(SlateRerankModel):
    def __init__(self, manager, *, hist_dense_field=None, mlp_a=(60, 32, 10),
                 mlp_b=(50, 20), attention_scale=8., **kwargs):
        super().__init__(manager, **kwargs)
        self.require_history()
        self.hist_dense_field = hist_dense_field
        self.encoder, width = mlp(self.item_dim, mlp_a, nn.SiLU)
        width += 1
        self.order_context = TargetContextPool(self.poi_num, width, attention_scale)
        self.history_context = TargetContextPool(self.page_num * self.poi_num, width, attention_scale)
        self.head, width = mlp(3 * width, mlp_b, nn.SiLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_items(self, cate, dense, mask):
        return self.encoder(cate.flatten(-2))

    def _encode_context(self, interaction, cate, dense):
        history, pages = self._history_inputs(interaction, cate)
        history = history.flatten(1, 2)
        if self.hist_dense_field is None:
            hist_prior = dense.new_zeros(history.shape[:2] + (1,))
        else:
            hist_prior = interaction[self.hist_dense_field]
            if hist_prior.ndim == 2:
                hist_prior = hist_prior.unsqueeze(-1)
            if hist_prior.shape != history.shape[:2] + (1,):
                raise ValueError("Historical prior scores must be [B,P*K] or [B,P*K,1].")
            hist_prior = hist_prior.to(dense.dtype)
        previous = torch.cat((self.encoder(history.flatten(-2)), hist_prior), -1)
        hist_valid = (pages == 1).repeat_interleave(self.poi_num, -1)
        return self.history_context.encode_context(previous, hist_valid)

    def _score_encoded(self, items, dense, mask, context: TargetContextEncoding):
        current = torch.cat((items, dense), -1)
        valid = mask > 0
        order_context = self.order_context(current, current, valid)
        hist_context = self.history_context.score_encoded(current, context)
        features = torch.cat((current, order_context, hist_context), -1)
        return self.output(self.head(features)).squeeze(-1)
````

## File: src/rosen/models/rerank/mge_metric.py
````python
"""MultiGE metric variants: supervise the best generated candidate and report diversity."""
import torch

from .mge import MultiGERerankModel
from .presets import resolve_rerank_config
from .base import select_by_order, masked_bce_with_logits
from .components.diversity import candidate_statistics


class MultiGEMetricRerankModel(MultiGERerankModel):
    preset_name = "mge_metric"

    def __init__(self, manager, **kwargs):
        super().__init__(manager, **resolve_rerank_config(self.preset_name, **kwargs))

    def loss_components(self, interaction):
        if self.LABEL is None:
            raise ValueError("Training a reranker requires a label field.")
        inputs = self._inputs(interaction)
        cate, history, dense, pages = inputs
        labels, slot_mask = interaction[self.LABEL], self._mask(interaction, cate)
        _, train_aux = self._generate(*inputs, labels=labels, label_mask=slot_mask, teacher_forcing=True)
        terms = self._generator_loss_components(labels, slot_mask, train_aux)
        # Teacher-forced loss and greedy legal selection are distinct passes.
        # Selection builds no unused generator graph; only the chosen judge loss
        # updates current/history embeddings and OCPM.
        with torch.no_grad():
            orders, aux = self._generate(*inputs, teacher_forcing=False)
        logits = self.candidate_logits(
            cate, history, dense, pages, orders,
            candidate_dense=aux["candidate_dense"] if self.candidate_dense_mode == "generator" else None)
        mask = select_by_order(self._mask(interaction, cate), orders)
        best = (logits.sigmoid() * mask).mean(-1).argmax(-1)
        rows = torch.arange(cate.shape[0], device=cate.device)
        labels = select_by_order(interaction[self.LABEL], orders)
        terms["evaluator"] = masked_bce_with_logits(logits[rows, best], labels[rows, best],
                                                    mask[rows, best], self.loss_reduction)
        if self.use_contrastive_loss:
            terms["contrastive"] = self._contrastive_loss(inputs, slot_mask, train_aux)
        return terms

    def predict(self, interaction):
        pred = super().predict(interaction)
        cate, _, _, _ = self._inputs(interaction)
        orders = pred.candidates.orders
        stats = candidate_statistics(select_by_order(cate, orders), orders)
        return pred.with_extra(stats)


class SubmodularMultiGEMetricRerankModel(MultiGEMetricRerankModel):
    preset_name = "mge_cosine_metric"
````

## File: src/rosen/models/rerank/mge.py
````python
"""Multi-generator reranking with a shared supervised OCPM evaluator."""

from typing import Literal, Optional, Sequence

import torch

from rosen.core.prediction import CandidatePool, Prediction
from .base import select_by_order, masked_bce_with_logits
from .slate import OCPMRerankModel
from .components.attention import OAMBlock, IntraPOISelfAttention, PageTargetAttention
from .components.retrieval import PPMLSimHashRetriever, BeamSearchRetriever
from .components.scoring import EncodedSlate, score_candidate_probs
from .components.generators import (PointwiseGenerator, ListwiseGenerator, AutoregressiveGenerator,
                                    PointerHead, cosine_coverage, lc_exact_score, lc_soft_score)


class MultiGERerankModel(OCPMRerankModel):
    def __init__(
        self, manager, *, slate_dense_field: str,
        slate_cate_fields: Optional[Sequence[str]] = None, slate_cate_group: Optional[str] = None,
        hist_cate_fields: Optional[Sequence[str]] = None, hist_cate_group: Optional[str] = None,
        page_mask_field: Optional[str] = None, label_field: Optional[str] = None,
        poi_num: int = 5, page_num: int = 5, top_k: int = 20,
        submodularity_mode: Literal["none", "cosine", "lc", "lc_soft"] = "lc",
        mlp_a: Sequence[int] = (60, 32, 20), mlp_b: Sequence[int] = (128, 32),
        hidden_dim: int = 64, attention_scale: float = 8.0,
        share_pointer_head: bool = True, lc_weight: float = 1.0, lc_temperature: float = 1.0,
        loss_reduction: Literal["mean", "valid_mean", "nonzero_mean"] = "mean",
        use_contrastive_loss: bool = False, contrastive_weight: float = 1.0,
        eval_all_perms: bool = True, eval_diagnostic_pools: bool = True, candidate_chunk_size: int = 32,
        candidate_dense_mode: Literal["prior", "generator"] = "prior",
        position_encoding=None, hash_vector=None, distance_weight=None, expose_rate=None,
        retrieval_seed: int = 2026,
    ):
        super().__init__(manager, slate_dense_field=slate_dense_field,
                         slate_cate_fields=slate_cate_fields, slate_cate_group=slate_cate_group,
                         hist_cate_fields=hist_cate_fields, hist_cate_group=hist_cate_group,
                         page_mask_field=page_mask_field, label_field=label_field,
                         poi_num=poi_num, page_num=page_num, loss_reduction=loss_reduction,
                         candidate_chunk_size=candidate_chunk_size)
        self.require_history()
        if submodularity_mode not in ("none", "cosine", "lc", "lc_soft"):
            raise ValueError("submodularity_mode must be none, cosine, lc or lc_soft.")
        if not isinstance(page_num, int) or page_num <= 0 or hidden_dim <= 0 or candidate_chunk_size <= 0:
            raise ValueError("Page count, hidden width and candidate chunk size must be positive.")
        if top_k <= 0 or loss_reduction not in ("mean", "valid_mean", "nonzero_mean") or lc_temperature <= 0:
            raise ValueError("Invalid top_k, loss_reduction or lc_temperature.")
        if candidate_dense_mode not in ("prior", "generator"):
            raise ValueError("candidate_dense_mode must be prior or generator.")
        self.submodularity_mode = submodularity_mode
        self.lc_weight, self.lc_temperature = lc_weight, lc_temperature
        self.use_contrastive_loss, self.contrastive_weight = use_contrastive_loss, contrastive_weight
        self.eval_all_perms, self.candidate_chunk_size = eval_all_perms, candidate_chunk_size
        self.eval_diagnostic_pools = eval_diagnostic_pools
        self.candidate_dense_mode = candidate_dense_mode

        f, d = self.num_attrs, self.emb_dim
        self.retriever = PPMLSimHashRetriever(poi_num, page_num, d, position_encoding=position_encoding,
                                              hash_vector=hash_vector, distance_weight=distance_weight,
                                              seed=retrieval_seed)
        self.beam_retriever = BeamSearchRetriever(poi_num, expose_rate)
        self.top_k = min(top_k, self.retriever.full_permutations.shape[0])
        rng = torch.Generator().manual_seed(retrieval_seed + 1)
        self.register_buffer("random_candidate_ids", torch.randperm(
            self.retriever.full_permutations.shape[0], generator=rng)[:self.top_k])

        # Shared modules are registered exactly once. Every caller receives them
        # explicitly; deepcopy/EMA/state_dict preserve one unambiguous owner.
        self.oam = OAMBlock(poi_num, f, d, attention_scale)
        self.intra_poi = IntraPOISelfAttention(f, d, attention_scale)
        self.target_attention = PageTargetAttention()
        self.evaluator = ListwiseGenerator(f, d, mlp_a, mlp_b)
        self.pointwise_generator = PointwiseGenerator()
        self.listwise_generator = ListwiseGenerator(f, d, mlp_a, mlp_b)
        self.generative_generator = AutoregressiveGenerator(f, d, mlp_a, hidden_dim)
        self.pointer_head = PointerHead(hidden_dim)
        # Optional modules come after all common initialization (paired variants).
        self.submodularity_generator = (AutoregressiveGenerator(f, d, mlp_a, hidden_dim)
                                        if submodularity_mode != "none" else None)
        self.submodularity_pointer_head = (PointerHead(hidden_dim)
                                           if self.submodularity_generator is not None and not share_pointer_head else None)

    def _generate(self, cate, history, dense, page_mask, *, labels=None, label_mask=None, teacher_forcing=False):
        cate, history, dense, page_mask = (value.detach() for value in (cate, history, dense, page_mask))
        distance = self.retriever(cate, history, page_mask)
        # Pointwise logits, labels and proposed item indices share input-slot coordinates.
        point_logits, point_indices = self.pointwise_generator(dense)
        common = (self.oam, self.intra_poi, self.target_attention)
        list_logits, list_indices = self.listwise_generator(cate, self.oam(history)[0], dense, page_mask, *common)
        generative = self.generative_generator(cate, history, dense, page_mask, *common, self.pointer_head,
                                               labels=labels, label_mask=label_mask, teacher_forcing=teacher_forcing)
        indices = torch.cat((point_indices, list_indices, generative.indices), 1)
        # Score-based heads output input-slot scores; OCPM candidate dense features
        # must follow each proposed output order. AR maxima already follow decode steps.
        candidate_dense = [select_by_order(point_logits.sigmoid(), point_indices)[:, 0],
                           select_by_order(list_logits.sigmoid(), list_indices)[:, 0], generative.max_scores]
        aux = {"pointwise_logits": point_logits, "listwise_logits": list_logits,
               "generative": generative, "distance": distance}
        if self.submodularity_generator is not None:
            coverage = (cosine_coverage(cate, select_by_order(cate, indices))
                        if self.submodularity_mode == "cosine" else None)
            head = self.submodularity_pointer_head if self.submodularity_pointer_head is not None else self.pointer_head
            sub = self.submodularity_generator(cate, history, dense, page_mask, *common, head,
                                               labels=labels, label_mask=label_mask, coverage=coverage,
                                               teacher_forcing=teacher_forcing)
            if self.submodularity_mode == "lc_soft":
                aux["coverage"] = lc_soft_score(sub.probabilities, indices, self.lc_temperature).mean()
            indices = torch.cat((indices, sub.indices), 1)
            candidate_dense.append(sub.max_scores)
            if self.submodularity_mode == "lc":
                aux["coverage"] = lc_exact_score(indices).mean()
            aux["submodularity"] = sub
        aux["candidate_dense"] = torch.stack(candidate_dense, 1).unsqueeze(-1).detach()
        return indices, aux

    def _score_candidates(self, cate, history, dense, page_mask, mask, permutations, *, candidate_dense=None):
        encoded = EncodedSlate(cate, dense, mask, self._history_context(history, page_mask))
        return score_candidate_probs(
            encoded, permutations, scorer=self._score_encoded,
            chunk_size=self.candidate_chunk_size, candidate_dense=candidate_dense)

    def _generator_loss_components(self, labels, mask, aux):
        self.loss_diagnostics = {}
        terms = {
            "gen.pointwise": masked_bce_with_logits(aux["pointwise_logits"], labels, mask, self.loss_reduction),
            "gen.listwise": masked_bce_with_logits(aux["listwise_logits"], labels, mask, self.loss_reduction),
            "gen.generative": aux["generative"].loss,
        }
        if self.submodularity_generator is not None:
            terms["gen.submodularity"] = aux["submodularity"].loss
            if self.submodularity_mode == "lc_soft":
                terms["gen.coverage"] = -self.lc_weight * aux["coverage"]
            elif self.submodularity_mode == "lc":
                # This discrete statistic is diagnostic, not an optimization term.
                self.loss_diagnostics["lc_exact"] = aux["coverage"].detach()
        return terms

    def _contrastive_loss(self, inputs, mask, aux):
        cate, history, dense, page_mask = inputs
        perms = self.retriever.full_permutations
        best = perms[self.retriever.top_k(aux["distance"], self.top_k)]
        worst = perms[self.retriever.last_k(aux["distance"], self.top_k)]
        best_scores = self._score_candidates(cate.detach(), history, dense.detach(), page_mask, mask, best)
        worst_scores = self._score_candidates(cate.detach(), history, dense.detach(), page_mask, mask, worst)
        return -self.contrastive_weight * (best_scores.mean() - worst_scores.mean())

    def loss_components(self, interaction):
        if self.LABEL is None:
            raise ValueError("Training a reranker requires a label field.")
        inputs = self._inputs(interaction)
        cate, history, dense, page_mask = inputs
        labels, mask = interaction[self.LABEL], self._mask(interaction, cate)
        _, aux = self._generate(*inputs, labels=labels, label_mask=mask, teacher_forcing=True)
        terms = self._generator_loss_components(labels, mask, aux)
        evaluator_logits = self._evaluator_logits(*inputs)
        terms["evaluator"] = masked_bce_with_logits(evaluator_logits, labels, mask, self.loss_reduction)
        if self.use_contrastive_loss:
            terms["contrastive"] = self._contrastive_loss(inputs, mask, aux)
        return terms

    def predict(self, interaction):
        cate, history, dense, page_mask = self._inputs(interaction)
        mask = self._mask(interaction, cate)
        generated, aux = self._generate(cate, history, dense, page_mask, teacher_forcing=False)
        original = self._evaluator_logits(cate, history, dense, page_mask)
        candidates = {"ctr_topk": generated}
        if self.eval_diagnostic_pools:
            permutations = self.retriever.full_permutations
            all_priors = select_by_order(dense, permutations).squeeze(-1)
            candidates.update({
                "ctr_lastk": permutations[self.retriever.last_k(aux["distance"], self.top_k)],
                "ctr_randk": permutations[self.random_candidate_ids],
                "ctr_beamk": permutations[self.beam_retriever(all_priors, self.top_k)],
            })
            if self.eval_all_perms:
                candidates["ctr_all"] = permutations
        scores = {name: self._score_candidates(
                      cate, history, dense, page_mask, mask, indices,
                      candidate_dense=(aux["candidate_dense"]
                                       if name == "ctr_topk" and self.candidate_dense_mode == "generator" else None))
                  for name, indices in candidates.items()}
        pool = CandidatePool(scores=scores["ctr_topk"], orders=generated)
        chosen_order, chosen_scores = pool.select(pool.scores.mean(-1).argmax(-1))
        extra = {name.removeprefix("ctr_"): value for name, value in scores.items() if name != "ctr_topk"}
        extra.update({f"{name.removeprefix('ctr_')}_order": indices.expand(cate.shape[0], -1, -1)
                      if indices.ndim == 2 else indices for name, indices in candidates.items() if name != "ctr_topk"})
        return Prediction(scores=original.sigmoid(), order=chosen_order, chosen_scores=chosen_scores,
                          candidates=pool, target=self.LABEL, extra=extra)
````

## File: src/rosen/models/rerank/nar.py
````python
"""NAR reference variant: three auxiliary generators, listwise-only candidate."""
from .mge import MultiGERerankModel
from .presets import resolve_rerank_config


class NARRerankModel(MultiGERerankModel):
    candidate_index = 1

    def __init__(self, manager, **kwargs):
        if kwargs.pop("submodularity_mode", "none") != "none":
            raise ValueError("NAR/Seq2Slate use three generators; submodularity_mode must be none.")
        super().__init__(manager, **resolve_rerank_config("nar", **kwargs))

    def _generate(self, *args, **kwargs):
        orders, aux = super()._generate(*args, **kwargs)
        selection = slice(self.candidate_index, self.candidate_index + 1)
        aux["candidate_dense"] = aux["candidate_dense"][:, selection]
        return orders[:, selection], aux
````

## File: src/rosen/models/rerank/pier_submodular.py
````python
"""PIER diversity variant: append the farthest non-retrieved slate before OCPM."""
import torch

from .pier import PIERRerankModel
from .base import select_by_order
from .components.diversity import candidate_statistics


class PIERSubmodularRerankModel(PIERRerankModel):
    def _top_orders(self, cate, distance):
        ids = self.retriever.top_k(distance, self.top_k)
        permutations = self.retriever.full_permutations
        top = permutations[ids]
        if self.top_k == len(permutations):
            return top
        cate = cate.detach()
        center = select_by_order(cate, top).mean(1)
        distances = []
        for chunk in permutations.split(self.candidate_chunk_size):
            values = select_by_order(cate, chunk)
            distances.append((values - center[:, None]).square().flatten(2).sum(-1))
        distances = torch.cat(distances, 1).scatter(1, ids, -torch.inf)
        additional = permutations[distances.argmax(-1)][:, None]
        return torch.cat((top, additional), 1)

    def predict(self, interaction):
        pred = super().predict(interaction)
        cate, _ = self._slate_inputs(interaction)
        orders = pred.candidates.orders
        stats = candidate_statistics(select_by_order(cate, orders), orders)
        return pred.with_extra(stats)
````

## File: src/rosen/models/rerank/pier.py
````python
"""PIER: PPML/SimHash retrieval followed by supervised OCPM candidate selection."""
import torch

from rosen.core.prediction import CandidatePool, Prediction
from .slate import OCPMRerankModel
from .base import select_by_order, masked_bce_with_logits
from .components.attention import OAMBlock, IntraPOISelfAttention, PageTargetAttention
from .components.generators import ListwiseGenerator
from .components.retrieval import PPMLSimHashRetriever, BeamSearchRetriever


class PIERRerankModel(OCPMRerankModel):
    def __init__(self, manager, *, top_k=20, mlp_a=(60, 32, 20), mlp_b=(128, 32),
                 attention_scale=8., use_contrastive_loss=True, contrastive_weight=.01,
                 eval_all_perms=True, candidate_chunk_size=32, position_encoding=None,
                 hash_vector=None, distance_weight=None, expose_rate=None, retrieval_seed=2026, **kwargs):
        super().__init__(manager, candidate_chunk_size=candidate_chunk_size, **kwargs)
        self.require_history()
        if not isinstance(top_k, int) or top_k <= 0 or candidate_chunk_size <= 0:
            raise ValueError("top_k and candidate_chunk_size must be positive.")
        self.retriever = PPMLSimHashRetriever(
            self.poi_num, self.page_num, self.emb_dim, position_encoding=position_encoding,
            hash_vector=hash_vector, distance_weight=distance_weight, seed=retrieval_seed)
        self.beam_retriever = BeamSearchRetriever(self.poi_num, expose_rate)
        self.top_k = min(top_k, len(self.retriever.full_permutations))
        rng = torch.Generator().manual_seed(retrieval_seed + 1)
        self.register_buffer("random_candidate_ids", torch.randperm(
            len(self.retriever.full_permutations), generator=rng)[:self.top_k])
        self.candidate_chunk_size, self.eval_all_perms = candidate_chunk_size, eval_all_perms
        self.use_contrastive_loss, self.contrastive_weight = use_contrastive_loss, contrastive_weight
        self.oam = OAMBlock(self.poi_num, self.num_attrs, self.emb_dim, attention_scale)
        self.intra_poi = IntraPOISelfAttention(self.num_attrs, self.emb_dim, attention_scale)
        self.target_attention = PageTargetAttention()
        self.evaluator = ListwiseGenerator(self.num_attrs, self.emb_dim, mlp_a, mlp_b)

    def _top_orders(self, cate, distance):
        return self.retriever.full_permutations[self.retriever.top_k(distance, self.top_k)]

    def candidate_orders(self, cate, history, dense, pages):
        permutations = self.retriever.full_permutations
        distance = self.retriever(cate.detach(), history.detach(), pages)
        result = {
            "topk": self._top_orders(cate, distance),
            "lastk": permutations[self.retriever.last_k(distance, self.top_k)],
            "randk": permutations[self.random_candidate_ids],
            "beamk": permutations[self.beam_retriever(select_by_order(dense, permutations).squeeze(-1), self.top_k)],
        }
        if self.eval_all_perms:
            result["all"] = permutations
        return result

    def _supervised_loss(self, interaction, inputs):
        logits = self._evaluator_logits(*inputs)
        return masked_bce_with_logits(logits, interaction[self.LABEL], self._mask(interaction, logits), self.loss_reduction)

    def loss_components(self, interaction):
        if self.LABEL is None:
            raise ValueError("Supervised training requires a label field.")
        inputs = self._inputs(interaction)
        terms = {"supervised": self._supervised_loss(interaction, inputs)}
        if self.use_contrastive_loss:
            cate, history, dense, pages = inputs
            mask = self._mask(interaction, dense)
            permutations = self.retriever.full_permutations
            distance = self.retriever(cate.detach(), history.detach(), pages)
            scores = []
            for orders in (self._top_orders(cate, distance),
                           permutations[self.retriever.last_k(distance, self.top_k)]):
                logits = self.candidate_logits(cate.detach(), history, dense.detach(), pages, orders)
                scores.append((logits.sigmoid() * select_by_order(mask, orders)).mean())
            terms["contrastive"] = -self.contrastive_weight * (scores[0] - scores[1])
        return terms

    def predict(self, interaction):
        cate, history, dense, pages = self._inputs(interaction)
        mask = self._mask(interaction, dense)
        original = self._evaluator_logits(cate, history, dense, pages)
        orders = self.candidate_orders(cate, history, dense, pages)
        scores = {name: self.candidate_logits(cate, history, dense, pages, value).sigmoid()
                  * select_by_order(mask, value) for name, value in orders.items()}
        pool = CandidatePool(scores=scores["topk"], orders=orders["topk"])
        chosen_order, chosen_scores = pool.select(pool.scores.mean(-1).argmax(-1))
        extra = {name: value for name, value in scores.items() if name != "topk"}
        extra.update({f"{name}_order": value[None].expand(cate.shape[0], -1, -1)
                      if value.ndim == 2 else value for name, value in orders.items() if name != "topk"})
        return Prediction(scores=original.sigmoid(), order=chosen_order, chosen_scores=chosen_scores,
                          candidates=pool, target=self.LABEL, extra=extra)
````

## File: src/rosen/models/rerank/presets.py
````python
"""One source of variant defaults; explicit caller arguments always win."""

_GENERATOR_DEFAULTS = {"candidate_dense_mode": "generator", "loss_reduction": "nonzero_mean"}
RERANK_PRESETS = {
    "mge3": {**_GENERATOR_DEFAULTS, "submodularity_mode": "none"},
    "mge_cosine": {**_GENERATOR_DEFAULTS, "submodularity_mode": "cosine"},
    "lc_mge": {**_GENERATOR_DEFAULTS, "submodularity_mode": "lc"},
    "nar": {**_GENERATOR_DEFAULTS, "submodularity_mode": "none"},
    "seq2slate": {**_GENERATOR_DEFAULTS, "submodularity_mode": "none"},
    "mge_metric": {**_GENERATOR_DEFAULTS, "submodularity_mode": "none"},
    "mge_cosine_metric": {**_GENERATOR_DEFAULTS, "submodularity_mode": "cosine"},
}


def resolve_rerank_config(name, **overrides):
    """Apply a named preset without rewriting any explicit override."""
    return {**RERANK_PRESETS.get(name.lower(), {}), **overrides}
````

## File: src/rosen/models/rerank/prm.py
````python
"""PRM baseline: item encoding, one inter-item attention, then per-item scoring."""
import torch
from torch import nn

from .slate import SlateRerankModel
from .components.generators import mlp
from .components.attention import masked_softmax


class PRMRerankModel(SlateRerankModel):
    def __init__(self, manager, *, mlp_a=(60, 32, 10), mlp_b=(50, 20), attention_scale=8., **kwargs):
        super().__init__(manager, **kwargs)
        if attention_scale <= 0:
            raise ValueError("attention_scale must be positive.")
        self.attention_scale = attention_scale
        self.encoder, width = mlp(self.item_dim, mlp_a, nn.SiLU)
        width += 1
        self.q, self.k, self.v = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.head, width = mlp(width, mlp_b, nn.SiLU)
        self.output = nn.Linear(width, 1)
        for layer in (self.q, self.k, self.v, self.output):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _encode_items(self, cate, dense, mask):
        return self.encoder(cate.flatten(-2))

    def _score_encoded(self, items, dense, mask, context):
        x = torch.cat((items, dense), -1)
        valid = mask > 0
        weights = masked_softmax(self.q(x) @ self.k(x).transpose(-1, -2) / self.attention_scale, valid.unsqueeze(-2))
        context = (weights @ self.v(x)) * valid[..., None]
        return self.output(self.head(context)).squeeze(-1)
````

## File: src/rosen/models/rerank/prs.py
````python
"""PRS: supervise the first exposure-weighted beam candidate with aligned labels."""
from .pier import PIERRerankModel
from .base import select_by_order, masked_bce_with_logits


class PRSRerankModel(PIERRerankModel):
    def _supervised_loss(self, interaction, inputs):
        cate, history, dense, pages = inputs
        permutations = self.retriever.full_permutations
        ids = self.beam_retriever(select_by_order(dense, permutations).squeeze(-1), 1)
        order = permutations[ids]
        logits = self.candidate_logits(cate, history, dense, pages, order)[:, 0]
        # The TF source feeds sigmoid probabilities as logits and leaves labels
        # in the input order. Use raw logits and transport labels/masks together.
        labels = select_by_order(interaction[self.LABEL], order)[:, 0]
        mask = select_by_order(self._mask(interaction, dense), order)[:, 0]
        return masked_bce_with_logits(logits, labels, mask, self.loss_reduction)
````

## File: src/rosen/models/rerank/seq2slate.py
````python
"""Seq2Slate reference variant: greedy GRU-pointer candidate with auxiliary heads."""
from .nar import NARRerankModel


class Seq2SlateRerankModel(NARRerankModel):
    candidate_index = 2
````

## File: src/rosen/models/rerank/slate.py
````python
"""Shared schema/input and supervised output contract for independent rerankers."""
from typing import Optional, Sequence

import torch

from rosen.core.prediction import Prediction
from rosen.emb.emblayer import EmbView
from .base import RerankModel, fold_history_pages, masked_bce_with_logits
from .components.scoring import EncodedSlate, OCPMContext, score_candidate_logits, score_candidate_probs


class SlateRerankModel(RerankModel):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if {'logits', 'score_permutations'} & cls.__dict__.keys():
            raise TypeError("Slate models implement _encode_items/_encode_context/_score_encoded; logits and score_permutations are fixed entry points.")

    def __init__(self, manager, *, slate_dense_field: str,
                 slate_cate_fields: Optional[Sequence[str]] = None, slate_cate_group: Optional[str] = None,
                 hist_cate_fields: Optional[Sequence[str]] = None, hist_cate_group: Optional[str] = None,
                 page_mask_field: Optional[str] = None, label_field: Optional[str] = None,
                 poi_num: int = 5, page_num: int = 5, loss_reduction: str = "nonzero_mean",
                 candidate_chunk_size: int = 32):
        super().__init__(manager)
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (poi_num, page_num)):
            raise ValueError("poi_num and page_num must be positive integers.")
        if loss_reduction not in ("mean", "valid_mean", "nonzero_mean"):
            raise ValueError("Invalid loss_reduction.")
        if isinstance(candidate_chunk_size, bool) or not isinstance(candidate_chunk_size, int) or candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be a positive integer.")
        self.candidate_chunk_size = candidate_chunk_size
        self.poi_num, self.page_num = poi_num, page_num
        self.slate_dense_field, self.page_mask_field = slate_dense_field, page_mask_field
        self.LABEL = label_field or self.LABEL
        self.loss_reduction = loss_reduction
        self.slate_view = self._view(slate_cate_fields, slate_cate_group, poi_num)
        self.history_view = (self._view(hist_cate_fields, hist_cate_group, page_num * poi_num)
                             if hist_cate_fields is not None or hist_cate_group is not None else None)
        self.num_attrs, self.emb_dim = self.slate_view.num_fields, self.slate_view.field_dim
        self.item_dim = self.num_attrs * self.emb_dim
        if self.history_view is not None:
            current = tuple(s.embedding_field_name for s in self.slate_view.selected_settings)
            history = tuple(s.embedding_field_name for s in self.history_view.selected_settings)
            if current != history or self.history_view.field_dim != self.emb_dim:
                raise ValueError("Current/history attributes must align and share atomic embeddings.")

    def _view(self, fields, group, length):
        if (fields is None) == (group is None):
            raise ValueError("Specify exactly one of cate_fields or cate_group per view.")
        view = self.omni_embedding.seq_groups[group] if group is not None else EmbView(
            self.omni_embedding, include_fields=fields)
        selected = view.selected_settings
        if not selected or (fields is not None and set(fields) != {s.field_name for s in selected}):
            raise ValueError("Select existing non-empty sequence fields.")
        if any(not s.is_sequence_setting or s.max_len != length for s in selected):
            raise ValueError(f"Categorical fields must be sequences of max_len={length}.")
        return view

    def require_history(self):
        if self.history_view is None:
            raise ValueError(f"{type(self).__name__} requires hist_cate_fields or hist_cate_group.")

    def _slate_inputs(self, interaction):
        cate = self.slate_view(interaction, split_by="stack")
        dense = interaction[self.slate_dense_field]
        if dense.ndim == 2:
            dense = dense.unsqueeze(-1)
        if dense.shape != cate.shape[:2] + (1,):
            raise ValueError("Prior scores must be [B,K] or [B,K,1].")
        return cate, dense.to(cate.dtype)

    def _inputs(self, interaction):
        cate, dense = self._slate_inputs(interaction)
        history, pages = self._history_inputs(interaction, cate)
        return cate, history, dense, pages

    def _history_inputs(self, interaction, reference):
        self.require_history()
        history = fold_history_pages(self.history_view(interaction, split_by="stack"), self.page_num, self.poi_num)
        pages = (interaction[self.page_mask_field] if self.page_mask_field else
                 reference.new_ones(reference.shape[0], self.page_num))
        if pages.shape != (reference.shape[0], self.page_num):
            raise ValueError("page_mask must be [B,P].")
        return history, pages

    def _mask(self, interaction, reference):
        spec = self.manager.get_label_spec(self.LABEL) if self.LABEL else None
        mask = (interaction[spec.mask_field] if spec and spec.mask_field and spec.mask_field in interaction
                else reference.new_ones(reference.shape[:2]))
        if mask.shape != reference.shape[:2]:
            raise ValueError("Slot mask must be [B,K].")
        return mask.to(reference.dtype)

    def _encode_items(self, cate, dense, mask):
        return cate

    def _encode_context(self, interaction, cate, dense):
        return None

    def encode_slate(self, interaction) -> EncodedSlate:
        """Lookup items once and encode history once, before any candidate expansion."""
        cate, dense = self._slate_inputs(interaction)
        mask = self._mask(interaction, dense)
        return EncodedSlate(self._encode_items(cate, dense, mask), dense, mask,
                            self._encode_context(interaction, cate, dense))

    def _score_encoded(self, items, dense, mask, context):
        """Return logits [B,M,N] using explicit candidate slots and fixed context."""
        raise NotImplementedError

    def logits(self, interaction):
        encoded = self.encode_slate(interaction)
        return self._score_encoded(encoded.items[:, None], encoded.dense[:, None],
                                   encoded.mask[:, None], encoded.context)[:, 0]

    def _validate_scored_orders(self, interaction, orders):
        if (not isinstance(orders, torch.Tensor) or orders.dtype != torch.long or orders.ndim != 3
                or orders.shape[0] != len(interaction) or not orders.shape[1]
                or not 1 <= orders.shape[2] <= self.poi_num):
            raise ValueError("score_permutations requires LongTensor[B,M,K], M > 0 and 1 <= K <= N.")
        sorted_orders = orders.sort(-1).values
        if ((sorted_orders[..., 0] < 0).any() or (sorted_orders[..., -1] >= self.poi_num).any()
                or (sorted_orders[..., 1:] <= sorted_orders[..., :-1]).any()):
            raise ValueError("Each scored order must select unique indices in [0,N).")
        if self.requires_full_slate and orders.shape[2] != self.poi_num:
            raise ValueError(f"{type(self).__name__} requires full slate width K == {self.poi_num}")
        if orders.device != interaction[self.slate_dense_field].device:
            raise ValueError("Scored orders must be on the same device as slate features.")

    def loss_components(self, interaction):
        if self.LABEL is None:
            raise ValueError("Supervised training requires a label field.")
        logits = self.logits(interaction)
        return {"supervised": masked_bce_with_logits(
            logits, interaction[self.LABEL], self._mask(interaction, logits), self.loss_reduction)}

    def score_permutations(self, interaction, orders):
        """One encoded, chunked path for all supported full and partial slates."""
        self._validate_scored_orders(interaction, orders)
        return score_candidate_probs(self.encode_slate(interaction), orders,
                                     scorer=self._score_encoded, chunk_size=self.candidate_chunk_size)

    def calculate_loss(self, interaction):
        terms = self.loss_components(interaction)
        if self.lsp.active:
            for name, value in terms.items():
                self.lsp.add(name, value)
        return sum(terms.values())

    def predict(self, interaction):
        logits = self.logits(interaction)
        valid = self._mask(interaction, logits) > 0
        order = logits.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        return Prediction(scores=logits.sigmoid(), order=order, target=self.LABEL)


class OCPMRerankModel(SlateRerankModel):
    """Shared fixed-width judge with OAM history encoded before candidate chunks."""
    requires_full_slate = True
    outputs_candidate_pool = True

    def _history_context(self, history, pages):
        return OCPMContext(self.oam(history)[0], pages)

    def _encode_context(self, interaction, cate, dense):
        return self._history_context(*self._history_inputs(interaction, cate))

    def _score_encoded(self, items, dense, mask, context: OCPMContext):
        return self.evaluator(items, context.history_pages, dense, context.page_mask,
                              self.oam, self.intra_poi, self.target_attention)[0]

    def _evaluator_logits(self, cate, history, dense, pages):
        return self.evaluator(cate, self.oam(history)[0], dense, pages,
                              self.oam, self.intra_poi, self.target_attention)[0]

    def candidate_logits(self, cate, history, dense, pages, orders, *, candidate_dense=None):
        encoded = EncodedSlate(cate, dense, dense.new_ones(cate.shape[:2]), self._history_context(history, pages))
        return score_candidate_logits(encoded, orders, scorer=self._score_encoded,
                                       chunk_size=self.candidate_chunk_size, candidate_dense=candidate_dense)
````

## File: tests/rosen/models/rerank/test_audit_regressions.py
````python
"""Semantic checks for the review findings, including cold-start gradient ownership."""
import pytest
import torch
from torch.nn import functional as F

from rosen.core.interaction import Interaction
from rosen.models.rerank import select_by_order
from tests.rosen.models.rerank.test_reference_models import case


@pytest.mark.parametrize('name', ['mge_metric', 'mge_cosine_metric'])
def test_metric_main_loss_trains_current_embedding_without_history(tmp_path, name, monkeypatch):
    model, batch = case(tmp_path, name, candidate_chunk_size=10)
    batch['pages'].zero_()
    captured, calls = {}, []
    inputs = model._inputs
    generate = model._generate

    def tracked_inputs(inter):
        value = inputs(inter)
        value[0].retain_grad()
        captured['cate'] = value[0]
        calls.append('inputs')
        return value

    def tracked_generate(*args, **kwargs):
        calls.append((kwargs['teacher_forcing'], torch.is_grad_enabled()))
        return generate(*args, **kwargs)

    judge_calls = []
    handle = model.evaluator.register_forward_hook(lambda *args: judge_calls.append(True))
    monkeypatch.setattr(model, '_inputs', tracked_inputs)
    monkeypatch.setattr(model, '_generate', tracked_generate)
    terms = model.loss_components(batch)
    handle.remove()
    terms['evaluator'].backward()
    assert calls == ['inputs', (True, True), (False, False)]
    assert len(judge_calls) == 1  # No discarded original-slate OCPM pass.
    assert captured['cate'].grad.abs().sum() > 0
    for table in model.omni_embedding.emb_modules.values():
        assert table.embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.generative_generator.parameters())


@pytest.mark.parametrize('name', ['mge3', 'mge_cosine', 'nar', 'seq2slate', 'mge_metric', 'mge_cosine_metric'])
def test_generator_objectives_are_batch_means_including_tail_batches(tmp_path, name):
    model, batch = case(tmp_path, name)
    single = batch[:1]
    duplicate = Interaction({key: single[key].repeat(3, *([1] * (single[key].ndim-1))) for key in single})
    small = model.loss_components(single)
    large = model.loss_components(duplicate)
    for key in small:
        torch.testing.assert_close(small[key], large[key], rtol=1e-5, atol=1e-6)
    gradient_small = torch.autograd.grad(small['gen.generative'], model.pointer_head.output.weight, retain_graph=True)[0]
    gradient_large = torch.autograd.grad(large['gen.generative'], model.pointer_head.output.weight)[0]
    torch.testing.assert_close(gradient_small, gradient_large, rtol=1e-4, atol=1e-6)


def test_pointwise_logits_labels_and_orders_share_original_slot_coordinates(tmp_path, monkeypatch):
    model, batch = case(tmp_path, 'mge3')
    with torch.no_grad():
        model.pointwise_generator.linear.weight.fill_(2.)
        model.pointwise_generator.linear.bias.fill_(-.5)
    # Deliberately make retrieved source order nonidentity; it must not relabel pointwise items.
    monkeypatch.setattr(model.retriever, 'top_k', lambda distance, k: torch.full((2, k), 5, dtype=torch.long))
    inputs = model._inputs(batch)
    orders, aux = model._generate(*inputs, teacher_forcing=False)
    expected_logits = batch['prior'] * 2 - .5
    expected_order = expected_logits.argsort(dim=-1, descending=True, stable=True)
    torch.testing.assert_close(aux['pointwise_logits'], expected_logits)
    torch.testing.assert_close(orders[:, 0], expected_order)
    torch.testing.assert_close(aux['candidate_dense'][:, 0, :, 0], expected_logits.sigmoid().gather(1, expected_order))
    expected_loss = F.binary_cross_entropy_with_logits(expected_logits, batch['label'],
                                                       weight=batch['mask'], reduction='sum') / batch['mask'].count_nonzero()
    torch.testing.assert_close(model.loss_components(batch)['gen.pointwise'], expected_loss)


def test_exact_lc_is_diagnostic_and_not_a_constant_optimization_offset(tmp_path):
    model, batch = case(tmp_path, 'lc_mge', submodularity_mode='lc')
    terms = model.loss_components(batch)
    _, aux = model._generate(*model._inputs(batch), labels=batch['label'], label_mask=batch['mask'], teacher_forcing=True)
    assert 'lc_exact' in model.loss_diagnostics
    assert not model.loss_diagnostics['lc_exact'].requires_grad
    torch.testing.assert_close(terms['gen.submodularity'], aux['submodularity'].loss)
    assert 'gen.coverage' not in terms
    assert terms['gen.submodularity'] >= 0


def test_soft_coverage_objective_contains_only_differentiable_pairs(tmp_path):
    from rosen.models.rerank.components.generators import lc_soft_score
    probs = torch.rand(2, 3, 3, requires_grad=True)
    orders = torch.tensor([[[0, 1, 2], [2, 1, 0]]] * 2)
    value = lc_soft_score(probs, orders)
    # Explicit soft/reference-only comparison; no reference/reference constant.
    steps = torch.arange(3.)
    expected_positions = (probs * steps[None, :, None]).sum(1)
    soft = torch.sigmoid(expected_positions[:, None, :] - expected_positions[:, :, None])
    inverse = orders.argsort(-1)
    ref = (inverse[..., None, :] > inverse[..., :, None]).float()
    expected = ((soft[:, None] - ref).square() * (~torch.eye(3, dtype=torch.bool))).sum((-1, -2, -3))
    torch.testing.assert_close(value, expected)
    value.sum().backward()
    assert probs.grad.abs().sum() > 0


@pytest.mark.parametrize('name', ['pier', 'prs', 'pier_submodular', 'mge3', 'mge_metric'])
def test_judge_and_prediction_agree_on_masked_candidate_scores(tmp_path, name):
    model, batch = case(tmp_path, name, candidate_dense_mode='prior') if name.startswith('mge') else case(tmp_path, name)
    model.eval()
    with torch.no_grad():
        pred = model.predict(batch)
        judged = model.score_permutations(batch, pred.candidates.orders)
    torch.testing.assert_close(judged, pred.candidates.scores)
    selected = (pred.candidates.orders == pred.order[:, None]).all(-1).long().argmax(-1)
    torch.testing.assert_close(pred.chosen_scores, pred.candidates.scores[torch.arange(2), selected])
````

## File: tests/rosen/models/rerank/test_encoded_judging.py
````python
"""One explicit-mask, encoded-context judge path across every rerank model."""
import math

import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.models.rerank import RERANK_MODEL_REGISTRY, SlateRerankModel
from rosen.models.rerank.components.scoring import EncodedSlate, score_candidate_logits
from tests.rosen.models.rerank.test_reference_models import case


ORDERS = torch.tensor([[[2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 1, 0], [0, 2, 1]],
                       [[1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [0, 1, 2]]])


def explicit_candidate(batch, order, *, n=3):
    """Independent schema-field construction; no private keys or model adapters."""
    data = dict(batch.interaction)
    for field in ('cur_item', 'cur_category', 'prior', 'mask'):
        values = batch[field].gather(1, order)
        data[field] = torch.cat((values, values.new_zeros((len(batch), n - order.shape[1]))), dim=1)
    data['slate_len'] = (data['mask'] > 0).sum(-1)
    data.pop('label', None)
    return Interaction(data)


@pytest.mark.parametrize('name', tuple(RERANK_MODEL_REGISTRY))
def test_every_judge_uses_the_same_entry_point_and_matches_explicit_slate_logits(tmp_path, name):
    model, batch = case(tmp_path, name)
    model.eval()
    assert type(model).score_permutations is SlateRerankModel.score_permutations
    assert type(model).logits is SlateRerankModel.logits
    batch['mask'] = torch.tensor([[1., 0., .5], [0., 1., 1.]])
    supported_widths = (3,) if model.requires_full_slate else (1, 2, 3)
    with torch.no_grad():
        for k in supported_widths:
            orders = ORDERS[..., :k]
            expected = []
            for candidate in orders.unbind(dim=1):
                explicit = explicit_candidate(batch, candidate)
                expected.append(model.logits(explicit).sigmoid()[:, :k] * batch['mask'].gather(1, candidate))
            expected = torch.stack(expected, dim=1)
            for chunk_size in (1, 2, 99):
                model.candidate_chunk_size = chunk_size
                torch.testing.assert_close(model.score_permutations(batch, orders), expected, rtol=1e-5, atol=2e-7)
    assert not hasattr(model, '_ordered_slate_interactions')


@pytest.mark.parametrize('name', tuple(RERANK_MODEL_REGISTRY))
def test_items_and_history_are_encoded_once_per_judge_before_candidate_chunks(tmp_path, name, monkeypatch):
    model, batch = case(tmp_path, name)
    model.eval()
    model.candidate_chunk_size = 2
    seen, calls = [], {'slate': 0, 'history': 0, 'history_context': 0}
    forward = model.omni_embedding.forward
    scorer = model._score_encoded

    def lookup(interaction, *args, **kwargs):
        calls['slate'] += 1
        assert interaction is batch
        return forward(interaction, *args, **kwargs)

    # EmbView resolves fields through OmniEmbLayer.forward. Count both views,
    # and separately inspect the fixed history encoder or K/V projections.
    monkeypatch.setattr(model.omni_embedding, 'forward', lookup)
    handles = []
    if model.outputs_candidate_pool:
        history_inputs, history_tensors = model._history_inputs, []

        def history_source(*args):
            history, pages = history_inputs(*args)
            history_tensors.append(history)
            return history, pages

        monkeypatch.setattr(model, '_history_inputs', history_source)

        def track_history(module, args):
            if any(args[0] is history for history in history_tensors):
                calls['history'] += 1
        handles.append(model.oam.register_forward_pre_hook(track_history))
    elif name == 'kuaishou':
        handles.append(model.history_context.k.register_forward_pre_hook(lambda *args: calls.__setitem__('history_context', calls['history_context'] + 1)))

    def score(items, dense, mask, context):
        seen.append((items.shape, mask.clone(), context))
        return scorer(items, dense, mask, context)

    monkeypatch.setattr(model, '_score_encoded', score)
    with torch.no_grad():
        result = model.score_permutations(batch, ORDERS)
    for handle in handles:
        handle.remove()
    assert result.shape == (2, 5, 3) and len(seen) == math.ceil(5 / model.candidate_chunk_size)
    assert [shape[1] for shape, _, _ in seen] == [2, 2, 1]
    assert all(context is seen[0][2] for _, _, context in seen)
    # All candidates share one current view plus one history view only when used.
    assert calls['slate'] == (2 if model.outputs_candidate_pool or name == 'kuaishou' else 1)
    if model.outputs_candidate_pool:
        assert calls['history'] == 1
    if name == 'kuaishou':
        assert calls['history_context'] == 1


@pytest.mark.parametrize('name', ['prm', 'kuaishou'])
def test_partial_scorer_receives_explicit_mask_without_schema_mask_or_hidden_fields(tmp_path, name, monkeypatch):
    model, batch = case(tmp_path, name)
    model.LABEL = None
    batch.drop('mask')
    batch.drop('label')
    model.eval()
    original_fields = set(batch.columns)
    scorer, observed = model._score_encoded, []

    def masked(items, dense, mask, context):
        assert mask.shape == (2, 2, 3)
        torch.testing.assert_close(mask, torch.tensor([[[1., 0., 0.]] * 2] * 2))
        assert items[:, :, 1:].count_nonzero() == dense[:, :, 1:].count_nonzero() == 0
        observed.append(mask)
        return scorer(items, dense, mask, context)

    monkeypatch.setattr(model, '_score_encoded', masked)
    model.candidate_chunk_size = 2
    orders = ORDERS[:, :2, :1]
    with torch.no_grad():
        result = model.score_permutations(batch, orders)
    assert len(observed) == 1 and result.shape == (2, 2, 1)
    assert set(batch.columns) == original_fields


@pytest.mark.parametrize('name', ['prm', 'kuaishou', 'pier', 'mge_metric'])
def test_chunked_judge_gradients_equal_explicit_full_candidate_evaluation(tmp_path, name):
    model, batch = case(tmp_path, name)
    model.eval()
    orders = ORDERS[:, :3]
    model.candidate_chunk_size = 2
    model.score_permutations(batch, orders).sum().backward()
    actual = {key: None if value.grad is None else value.grad.clone() for key, value in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    for candidate in orders.unbind(1):
        explicit = explicit_candidate(batch, candidate)
        mask = batch['mask'].gather(1, candidate)
        (model.logits(explicit).sigmoid() * mask).sum().backward()
    for key, value in model.named_parameters():
        assert (value.grad is None) == (actual[key] is None), key
        if value.grad is not None:
            torch.testing.assert_close(value.grad, actual[key], rtol=1e-4, atol=1e-6, msg=key)


@pytest.mark.parametrize('name', ['prm', 'pier', 'mge3'])
@pytest.mark.parametrize('order', [torch.zeros(2, 1, 3), torch.tensor([[[0, 0, 1]], [[0, 1, 2]]]),
                                   torch.tensor([[[0, 1, 3]], [[0, 1, 2]]]),
                                   torch.tensor([[[-1, 1, 2]], [[0, 1, 2]]]),
                                   torch.zeros(2, 0, 3, dtype=torch.long)])
def test_invalid_orders_fail_before_lookup_or_context_encoding(tmp_path, name, order, monkeypatch):
    model, batch = case(tmp_path, name)

    def forbidden(*args):
        raise AssertionError('Invalid order reached feature/context encoding')

    monkeypatch.setattr(model, 'encode_slate', forbidden)
    with pytest.raises(ValueError):
        model.score_permutations(batch, order)


def test_new_models_must_use_explicit_encoded_scoring_extension_points():
    with pytest.raises(TypeError, match='_score_encoded'):
        class ImplicitJudge(SlateRerankModel):
            def logits(self, interaction):
                return interaction['untracked_control']

    with pytest.raises(TypeError, match='fixed entry points'):
        class ParallelJudge(SlateRerankModel):
            def score_permutations(self, interaction, orders):
                return torch.ones_like(orders)


@pytest.mark.parametrize('name', ['dnn', 'prm', 'kuaishou'])
def test_custom_source_mask_is_selected_once_then_passed_explicitly(tmp_path, name, monkeypatch):
    model, batch = case(tmp_path, name)
    model.eval()
    batch['alternate_validity'] = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    calls = []

    def source_mask(interaction, reference):
        calls.append(interaction)
        return interaction['alternate_validity'].to(reference.dtype)

    monkeypatch.setattr(model, '_mask', source_mask)
    orders = ORDERS[..., :2]
    with torch.no_grad():
        actual = model.score_permutations(batch, orders)
        assert calls == [batch]
        expected = []
        for candidate in orders.unbind(1):
            explicit = explicit_candidate(batch, candidate)
            selected = batch['alternate_validity'].gather(1, candidate)
            explicit['alternate_validity'] = torch.cat((selected, selected.new_zeros((2, 1))), 1)
            expected.append(model.logits(explicit).sigmoid()[:, :2] * selected)
        torch.testing.assert_close(actual, torch.stack(expected, 1), rtol=1e-5, atol=2e-7)


def test_bad_encoded_scorer_shape_is_rejected_instead_of_broadcasting():
    encoded = EncodedSlate(torch.ones(2, 3, 4), torch.ones(2, 3, 1), torch.ones(2, 3))
    with pytest.raises(ValueError, match='scorer must return logits'):
        score_candidate_logits(encoded, ORDERS, scorer=lambda items, dense, mask, context: mask[..., 0])
````

## File: tests/rosen/models/rerank/test_mge_rerank.py
````python
import itertools
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rosen.core.enum_type import FeatureSource
from rosen.core.interaction import Interaction
from rosen.core.train import EMAConfig, EarlyStopper, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer, EMATrainer
from rosen.emb import SchemaManager, SlateLabel
from rosen.emb.schema import SeqGroupConfig, SequenceSetting, SparseEmbSetting
from rosen.evaluate import BatchResolver, RankingEvaluator
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.models.rerank import (
    MultiGERerankModel, select_by_order, build_full_permutations,
    build_rerank_model, fold_history_pages, masked_bce_with_logits,
)
from rosen.models.rerank.components import (
    BeamSearchRetriever, IntraPOISelfAttention, OAMBlock, PageTargetAttention, PPMLSimHashRetriever,
)
from rosen.models.rerank.components.generators import cosine_coverage, lc_exact_score, lc_soft_score


def example(tmp_path, mode="lc", k=3, p=2, **kwargs):
    atoms = [SparseEmbSetting(name, FeatureSource.ITEM, embedding_dim=4) for name in ("item", "cate")]
    for atom in atoms:
        atom._build_vocab_indices([str(i) for i in range(10)])
    slate = SeqGroupConfig("slate", "slate_len", max_len=k)
    history = SeqGroupConfig("history", "hist_len", max_len=p * k)
    settings = [SequenceSetting(f"cur_{s.field_name}", s, slate) for s in atoms]
    settings += [SequenceSetting(f"pre_{s.field_name}", s, history) for s in atoms]
    schema = SchemaManager(settings, tmp_path / "schema", label_fields=SlateLabel("label", k, "mask"))
    model = MultiGERerankModel(
        schema, slate_dense_field="prior", slate_cate_group="slate", hist_cate_group="history",
        page_mask_field="pages", poi_num=k, page_num=p, top_k=2, mlp_a=(8,), mlp_b=(8,),
        hidden_dim=8, submodularity_mode=mode, **kwargs,
    )
    ids = torch.arange(2 * k).reshape(2, k) % 8 + 1
    hist = torch.arange(2 * p * k).reshape(2, p * k) % 8 + 1
    batch = Interaction({
        "cur_item": ids, "cur_cate": ids.flip(-1), "pre_item": hist, "pre_cate": hist.flip(-1),
        "slate_len": torch.full((2,), k), "hist_len": torch.full((2,), p * k),
        "prior": torch.linspace(.1, .9, 2 * k).reshape(2, k),
        "label": (torch.arange(2 * k).reshape(2, k) % 2).float(),
        "mask": torch.ones(2, k), "pages": torch.ones(2, p),
    })
    batch["mask"][0, -1] = 0
    batch["pages"][0, -1] = 0
    return model, batch


@pytest.mark.parametrize("tail", [(), (1,), (2, 4)])
@pytest.mark.parametrize("batched", [False, True])
def test_permutation_gather_values_and_gradients(tail, batched):
    slate = torch.arange(2 * 3 * math.prod(tail), dtype=torch.float64).reshape(2, 3, *tail).requires_grad_()
    permutations = torch.tensor([[2, 0, 1], [1, 2, 0]])
    if batched:
        permutations = torch.stack((permutations, permutations.flip(0)))
    actual = select_by_order(slate, permutations)
    expected = torch.stack([slate[b, permutations[b] if batched else permutations] for b in range(2)])
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(slate.grad, torch.full_like(slate, 2))


def test_fold_full_permutations_and_validation():
    x = torch.arange(48).reshape(2, 4, 6).transpose(1, 2)
    torch.testing.assert_close(fold_history_pages(x, 2, 3).flatten(1, 2), x)
    assert build_full_permutations(3).tolist() == list(map(list, itertools.permutations(range(3))))
    for k in (0, 8, True, 2.5):
        with pytest.raises(ValueError):
            build_full_permutations(k)
    with pytest.raises(ValueError, match="length"):
        fold_history_pages(x, 3, 3)
    with pytest.raises(ValueError, match="batch/slot"):
        select_by_order(torch.ones(2, 3), torch.zeros(3, 2, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="long"):
        select_by_order(torch.ones(2, 3), torch.zeros(2, 3))


def test_masked_bce_reductions_sentinels_and_zero_mask():
    logits = torch.tensor([[0., math.log(3)], [0., -math.log(3)]], requires_grad=True)
    labels = torch.tensor([[1., 1.], [float("nan"), 0.]])
    mask = torch.tensor([[1., 1.], [0., 1.]])
    expected = math.log(2) + 2 * math.log(4 / 3)
    assert masked_bce_with_logits(logits, labels, mask).item() == pytest.approx(expected / 4)
    assert masked_bce_with_logits(logits, labels, mask, "valid_mean").item() == pytest.approx(expected / 3)
    loss = masked_bce_with_logits(logits, labels, torch.zeros_like(mask), "valid_mean")
    loss.backward()
    assert loss == 0
    assert logits.grad.count_nonzero() == 0
    with pytest.raises(ValueError, match="identical"):
        masked_bce_with_logits(logits, labels[0])


def test_masked_bce_nonzero_mean_matches_weighted_tf_reduction():
    logits = torch.zeros(1, 4, requires_grad=True)
    labels = torch.tensor([[1., 0., float("nan"), 1.]])
    mask = torch.tensor([[.5, 2., 0., 1.]])
    loss = masked_bce_with_logits(logits, labels, mask, "nonzero_mean")
    assert loss.item() == pytest.approx(3.5 * math.log(2) / 3)
    assert masked_bce_with_logits(logits, labels, mask, "valid_mean").item() == pytest.approx(math.log(2))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-.25, 1., 0., -.5]]) / 3)
    empty = masked_bce_with_logits(logits, labels, torch.zeros_like(mask), "nonzero_mean")
    assert torch.isfinite(empty) and empty == 0


def test_oam_matches_explicit_attribute_and_poi_loops():
    torch.manual_seed(17)
    oam = OAMBlock(3, 2, 4)
    x = torch.randn(2, 3, 2, 4, requires_grad=True)
    pools, deltas = [], []
    for f in range(2):
        values = x[:, :, f]
        delta = (oam.intra.q(values) @ oam.intra.k(values).transpose(-1, -2) / 8).softmax(-1) @ oam.intra.v(values)
        deltas.append(delta)
        pools.append(oam.intra.pool((values + delta).flatten(1)))
    pooled = torch.stack(pools, 1)
    inter = (oam.inter.q(pooled) @ oam.inter.k(pooled).transpose(-1, -2) / 8).softmax(-1) @ oam.inter.v(pooled)
    expected = oam.inter.pool((pooled + inter).flatten(1))
    page, per_poi = oam(x)
    torch.testing.assert_close(per_poi, torch.stack(deltas, 2))
    torch.testing.assert_close(page, expected)
    intra_poi = IntraPOISelfAttention(2, 4)
    expected_poi = []
    for slot in range(3):
        values = per_poi[:, slot]
        delta = (intra_poi.q(values) @ intra_poi.k(values).transpose(-1, -2) / 8).softmax(-1) @ intra_poi.v(values)
        expected_poi.append(intra_poi.pool((values + delta).flatten(1)))
    torch.testing.assert_close(intra_poi(per_poi), torch.stack(expected_poi, 1))
    # A new leading history-page axis must act as a batch axis.
    pages, _ = oam(x[:, None].expand(-1, 2, -1, -1, -1))
    torch.testing.assert_close(pages, page[:, None].expand(-1, 2, -1))
    page.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert all(param.ndim == 2 for param in oam.parameters())
    # Check branch scale without claiming a universal architecture threshold.
    ratio = per_poi.square().mean().sqrt() / x.square().mean().sqrt()
    assert .05 < ratio < 5


def test_target_attention_preserves_batch_alignment_and_zero_history():
    attention = PageTargetAttention()
    history = torch.tensor([[[1., 0.], [2., 0.]], [[0., 5.], [0., 7.]]])
    mask = torch.tensor([[1., 0.], [0., 1.]])
    target = torch.randn(2, 3, 2)
    expected = torch.tensor([[[1., 0.]] * 3, [[0., 7.]] * 3])
    torch.testing.assert_close(attention(target, history, mask), expected)
    torch.testing.assert_close(attention(target.flatten(0, 1), history, mask), expected.flatten(0, 1))
    assert attention(target, history, torch.zeros_like(mask)).count_nonzero() == 0


def test_ppml_simhash_and_exposure_ranking_match_loop_reference():
    torch.manual_seed(3)
    rng_state = torch.random.get_rng_state().clone()
    retriever = PPMLSimHashRetriever(3, 2, 2, position_encoding=[[1., .5], [.2, 2.], [3., .1]],
                                    hash_vector=[[1., -1.], [-2., 1.]], distance_weight=[[.5, .25]])
    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert not list(retriever.parameters())
    slate = torch.randn(2, 3, 2, 2)
    hist = torch.randn(2, 2, 3, 2, 2)
    actual = retriever(slate, hist)
    expected = torch.zeros_like(actual)
    for b in range(2):
        for m, order in enumerate(retriever.full_permutations):
            candidate = (slate[b, order] * retriever.position_encoding[:, None]).mean((0, 1))
            for p in range(2):
                page = (hist[b, p] * retriever.position_encoding[:, None]).mean((0, 1))
                distance = (torch.sign(candidate * retriever.hash_vector[p]) - torch.sign(page * retriever.hash_vector[p])).abs().sum()
                expected[b, m] += distance * retriever.distance_weight[0, p]
    torch.testing.assert_close(actual, expected)
    assert retriever.top_k(actual, 2).shape == (2, 2)
    assert retriever.last_k(actual, 2).shape == (2, 2)
    assert retriever(slate, hist, torch.zeros(2, 2)).count_nonzero() == 0
    beam = BeamSearchRetriever(3)
    prior = torch.tensor([[.2, .9, .5], [.6, .1, .3]])
    candidate_priors = select_by_order(prior, retriever.full_permutations)
    scores = (candidate_priors * torch.tensor([1., .9, .8])).sum(-1)
    torch.testing.assert_close(beam(candidate_priors, 2), scores.argsort(dim=-1, descending=True, stable=True)[:, :2])
    with pytest.raises(ValueError, match="finite constant"):
        PPMLSimHashRetriever(3, 2, 2, hash_vector=torch.ones(3, 2))


def test_cosine_and_lc_statistics_match_definitions_and_soft_reward_has_gradient():
    orders = torch.tensor([[[0, 2, 1], [2, 1, 0], [0, 1, 2], [1, 0, 2]],
                           [[2, 2, 2], [1, 1, 2], [0, 1, 2], [2, 0, 1]]])
    expected = []
    for batch in orders.tolist():
        total = 0
        for a, b in itertools.combinations(batch, 2):
            total += sum((a[i] < a[j]) != (b[i] < b[j]) for i in range(3) for j in range(3))
        expected.append(total)
    torch.testing.assert_close(lc_exact_score(orders), torch.tensor(expected, dtype=torch.float32))
    assert not lc_exact_score(orders).requires_grad
    cate = torch.randn(2, 3, 2, 4, requires_grad=True)
    others = select_by_order(cate, orders[:, :3])
    expected_cosine = (cate[:, None] * others).sum((-1, -2)) / (
        cate[:, None].norm(dim=(-1, -2)) * others.norm(dim=(-1, -2))).clamp_min(1e-4)
    torch.testing.assert_close(cosine_coverage(cate, others), expected_cosine.mean(1))
    assert not cosine_coverage(cate, others).requires_grad
    logits = torch.randn(2, 3, 3, requires_grad=True)
    lc_soft_score(logits.softmax(-1), orders[:, :3]).mean().backward()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["none", "cosine", "lc", "lc_soft"])
def test_modes_predictions_loss_and_legal_inference(tmp_path, mode):
    torch.manual_seed(7)
    model, batch = example(tmp_path, mode)
    terms = model.loss_components(batch)
    assert set(terms) == ({"gen.pointwise", "gen.listwise", "gen.generative", "evaluator"} |
                          ({"gen.submodularity"} if mode != "none" else set()) |
                          ({"gen.coverage"} if mode == "lc_soft" else set()))
    assert all(torch.isfinite(term) for term in terms.values())
    # predict uses greedy decoding even if the caller forgot eval().
    model.train()
    with torch.no_grad():
        outputs = model.predict(batch)
        generated, _ = model._generate(*model._inputs(batch))
    m = 3 if mode == "none" else 4
    assert generated.shape == (2, m, 3)
    torch.testing.assert_close(generated.sort(-1).values, torch.arange(3).expand(2, m, 3))
    assert outputs.scores.shape == (2, 3) and outputs.target == model.LABEL
    assert outputs.candidates.scores.shape == (2, m, 3)
    assert outputs.extra["all"].shape == (2, 6, 3)
    assert outputs.order.shape == (2, 3)
    for name, values in outputs.extra.items():
        if not name.endswith("_order"):
            assert torch.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    best = outputs.candidates.scores.mean(-1).argmax(-1)
    torch.testing.assert_close(outputs.order, generated[torch.arange(2), best])
    torch.testing.assert_close(outputs.candidates.orders, generated)



@pytest.mark.parametrize("mode", ["none", "cosine", "lc", "lc_soft"])
def test_generators_cannot_update_embeddings_but_share_attention(tmp_path, mode):
    torch.manual_seed(11)
    model, batch = example(tmp_path, mode)
    batch["prior"].requires_grad_()
    terms = model.loss_components(batch)
    sum(v for k, v in terms.items() if k.startswith("gen.")).backward()
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in model.omni_embedding.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.oam.parameters())
    assert all(p.grad is None for p in model.evaluator.parameters())
    assert batch["prior"].grad is None
    if mode == "lc_soft":
        model.zero_grad(set_to_none=True)
        _, aux = model._generate(*model._inputs(batch), labels=batch["label"], label_mask=batch["mask"], teacher_forcing=True)
        aux["coverage"].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.submodularity_generator.parameters())
    model.zero_grad(set_to_none=True)
    model.loss_components(batch)["evaluator"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.omni_embedding.parameters())
    assert all(p.grad is None for p in model.generative_generator.parameters())
    assert batch["prior"].grad is not None and batch["prior"].grad.abs().sum() > 0


def test_teacher_forcing_records_may_repeat_but_greedy_is_a_permutation(tmp_path):
    model, batch = example(tmp_path)

    class FixedHead(nn.Module):
        def forward(self, candidates, state):
            return torch.arange(candidates.shape[1], device=candidates.device).float().expand(candidates.shape[:2])

    model.pointer_head = FixedHead()
    _, train = model._generate(*model._inputs(batch), labels=batch["label"], label_mask=batch["mask"], teacher_forcing=True)
    _, evaluate = model._generate(*model._inputs(batch), teacher_forcing=False)
    assert train["generative"].indices.tolist() == [[[2, 2, 2]], [[2, 2, 2]]]
    assert evaluate["generative"].indices.tolist() == [[[2, 1, 0]], [[2, 1, 0]]]
    probs = train["generative"].probabilities
    expected_loss = sum(-(batch["label"][:, t] * batch["mask"][:, t] * probs[:, t, t].clamp_min(1e-9).log()).mean() for t in range(3))
    torch.testing.assert_close(train["generative"].loss, expected_loss)


def test_chunked_ocpm_matches_individual_history_aligned_evaluation(tmp_path):
    model, batch = example(tmp_path, candidate_chunk_size=2)
    cate, history, dense, page_mask = model._inputs(batch)
    mask = model._mask(batch, cate)
    orders = model.retriever.full_permutations
    actual = model._score_candidates(cate, history, dense, page_mask, mask, orders)
    expected = []
    for b in range(2):
        scores = []
        for order in orders:
            logits, _ = model.evaluator(cate[b:b + 1, order], model.oam(history[b:b + 1])[0], dense[b:b + 1, order],
                                        page_mask[b:b + 1], model.oam, model.intra_poi, model.target_attention)
            scores.append(logits.sigmoid()[0] * mask[b, order])
        expected.append(torch.stack(scores))
    torch.testing.assert_close(actual, torch.stack(expected))
    model.candidate_chunk_size = 99
    torch.testing.assert_close(actual, model._score_candidates(cate, history, dense, page_mask, mask, orders))


@pytest.mark.parametrize("mode", ["none", "cosine", "lc", "lc_soft"])
def test_generator_dense_scores_reach_ocpm_in_candidate_coordinates(tmp_path, mode):
    model, batch = example(tmp_path, mode, candidate_dense_mode="generator", candidate_chunk_size=2)

    class DenseEvaluator(nn.Module):
        def forward(self, cate, history, dense, page_mask, *attention):
            logits = dense.squeeze(-1)
            return logits, logits.argsort(-1, descending=True)[:, None]

    model.evaluator = DenseEvaluator()
    inputs = model._inputs(batch)
    generated, aux = model._generate(*inputs, teacher_forcing=False)
    expected_dense = [select_by_order(aux["pointwise_logits"].sigmoid(), generated[:, 0:1])[:, 0],
                      select_by_order(aux["listwise_logits"].sigmoid(), generated[:, 1:2])[:, 0],
                      aux["generative"].max_scores]
    if mode != "none":
        expected_dense.append(aux["submodularity"].max_scores)
    torch.testing.assert_close(aux["candidate_dense"].squeeze(-1), torch.stack(expected_dense, 1))
    assert not aux["candidate_dense"].requires_grad
    pred = model.predict(batch)
    expected = aux["candidate_dense"].squeeze(-1).sigmoid() * select_by_order(batch["mask"], generated)
    torch.testing.assert_close(pred.candidates.scores, expected)
    best = expected.mean(-1).argmax(-1)
    torch.testing.assert_close(pred.order, generated[torch.arange(2), best])
    for family in ("lastk", "randk", "beamk", "all"):
        priors = select_by_order(batch["prior"], pred.extra[f"{family}_order"])
        torch.testing.assert_close(pred.extra[family], priors.sigmoid() * select_by_order(
            batch["mask"], pred.extra[f"{family}_order"]))


def test_custom_dense_candidate_chunking_matches_single_candidate_ocpm(tmp_path):
    model, batch = example(tmp_path, candidate_chunk_size=2)
    cate, history, dense, pages = model._inputs(batch)
    mask = model._mask(batch, cate)
    orders = model.retriever.full_permutations
    values = torch.linspace(-2, 2, 2 * 6 * 3).reshape(2, 6, 3, 1)
    actual = model._score_candidates(cate, history, dense, pages, mask, orders, candidate_dense=values)
    expected = []
    for b in range(2):
        scores = []
        for m, order in enumerate(orders):
            logits, _ = model.evaluator(cate[b:b + 1, order], model.oam(history[b:b + 1])[0], values[b:b + 1, m],
                                        pages[b:b + 1], model.oam, model.intra_poi, model.target_attention)
            scores.append(logits.sigmoid()[0] * mask[b, order])
        expected.append(torch.stack(scores))
    torch.testing.assert_close(actual, torch.stack(expected))
    model.candidate_chunk_size = 99
    torch.testing.assert_close(actual, model._score_candidates(
        cate, history, dense, pages, mask, orders, candidate_dense=values))
    with pytest.raises(ValueError, match="Candidate dense"):
        model._score_candidates(cate, history, dense, pages, mask, orders, candidate_dense=values[:, :-1])


def test_lc_mge_factory_presets_and_explicit_overrides(tmp_path):
    baseline, batch = example(tmp_path)
    options = dict(slate_dense_field="prior", slate_cate_group="slate", hist_cate_group="history",
                   poi_num=3, page_num=2, eval_all_perms=False)
    model = build_rerank_model(baseline.manager, "LC_MGE", **options)
    assert model.submodularity_mode == "lc"
    assert model.candidate_dense_mode == "generator"
    assert model.loss_reduction == "nonzero_mean"
    assert torch.isfinite(model.calculate_loss(batch))
    with torch.no_grad():
        assert model.predict(batch).order.shape == (2, 3)
    old = build_rerank_model(baseline.manager, "mge", **options)
    assert old.candidate_dense_mode == "prior" and old.loss_reduction == "mean"
    override = build_rerank_model(baseline.manager, "lc_mge", candidate_dense_mode="prior", **options)
    assert override.candidate_dense_mode == "prior"
    with pytest.raises(ValueError, match="candidate_dense_mode"):
        build_rerank_model(baseline.manager, "lc_mge", candidate_dense_mode="unknown", **options)


def test_common_initialization_constants_factory_and_label_free_prediction(tmp_path):
    torch.manual_seed(19)
    baseline, batch = example(tmp_path, "none")
    torch.manual_seed(19)
    enhanced, _ = example(tmp_path, "lc", eval_all_perms=False, share_pointer_head=False)
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, enhanced.state_dict()[name], rtol=0, atol=0)
    assert "retriever.full_permutations" in dict(enhanced.named_buffers())
    assert not any("oam" in name for name, _ in enhanced.generative_generator.named_parameters())
    params = list(enhanced.parameters())
    assert len(params) == len({id(p) for p in params})
    unlabeled = Interaction({k: batch[k] for k in batch.columns if k != "label"})
    with torch.no_grad():
        assert "all" not in enhanced.predict(unlabeled).extra
    with pytest.warns(UserWarning, match="unsupported"):
        rebuilt = build_rerank_model(baseline.manager, "MGE", unsupported=True,
                                    slate_dense_field="prior", slate_cate_group="slate", hist_cate_group="history",
                                    poi_num=3, page_num=2)
    assert isinstance(rebuilt, MultiGERerankModel)
    with pytest.raises(ValueError, match="Unknown"):
        build_rerank_model(baseline.manager, "missing")


@pytest.mark.parametrize("trainer_class", [BaseTrainer, EMATrainer])
def test_training_loss_ledger_evaluation_early_stop_and_checkpoint(tmp_path, trainer_class):
    torch.manual_seed(23)
    model, batch = example(tmp_path, "lc_soft", k=5, p=2, eval_all_perms=False,
                           use_contrastive_loss=True, contrastive_weight=.1)
    eval_manager = EvaluatorManager(log_path=tmp_path / "metrics.log")
    eval_manager.register("point", PointwiseEvaluator("auc", "logloss"))
    eval_manager.register("slate", RankingEvaluator("ndcg@5", "pcoc"), resolver=BatchResolver(view="score_only"))
    eval_manager.register("order", RankingEvaluator("ndcg@5"))
    loader = DataLoader([batch], batch_size=None)
    options = {"ema_config": EMAConfig(decays=(.5,), eval_student=True)} if trainer_class is EMATrainer else {}
    trainer = trainer_class(
        model, torch.optim.Adam(model.parameters(), lr=0.), model.manager,
        TrainerDataLoaders(loader, loader), TrainerComponents(evaluator_manager=eval_manager, early_stopper=EarlyStopper(patience=1)),
        SimpleNamespace(device="cpu", max_epochs=5, train_log_interval=100, ckpt_dir=str(tmp_path / "ckpt"),
                        experiment_name="mge", dataset_name="synthetic"), **options,
    )
    result = trainer.run()
    assert trainer.epoch == 2
    assert set(result["point"]) == {"auc", "logloss"}
    assert set(result["slate"]) == {"ndcg@5", "pcoc"}
    ledger = model.lsp.summary()["settlements"]["main"]
    assert set(ledger["terms"]) == {"gen.pointwise", "gen.listwise", "gen.generative", "gen.submodularity", "gen.coverage", "evaluator", "contrastive"}
    assert ledger["total"] == pytest.approx(sum(term["weighted"] for term in ledger["terms"].values()), abs=1e-5)
    # Zero LR above makes early stopping deterministic. Exercise a real update
    # separately so the integration also checks optimizer and EMA transactions.
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for group in trainer.optimizer.param_groups:
        group["lr"] = 1e-3
    trainer.train_epoch()
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    with torch.no_grad():
        expected = model.predict(batch)
    path = trainer.save_checkpoint("roundtrip", metrics=result)
    with torch.no_grad():
        next(model.parameters()).add_(1)
        model.retriever.position_encoding.zero_()
        if trainer_class is EMATrainer:
            next(trainer.ema.teachers[.5].parameters()).add_(2)
    trainer.load_checkpoint(path)
    with torch.no_grad():
        restored = model.predict(batch)
    torch.testing.assert_close(restored.scores, expected.scores, rtol=0, atol=0)
    torch.testing.assert_close(restored.order, expected.order, rtol=0, atol=0)
    for key in expected.extra:
        torch.testing.assert_close(restored.extra[key], expected.extra[key], rtol=0, atol=0)
    if trainer_class is EMATrainer:
        assert set(trainer.last_eval_metrics_by_checkpoint) == {"student", "ema0p5"}
        teacher = trainer.ema.teachers[.5]
        assert teacher.slate_view.omni is teacher.omni_embedding
````

## File: tests/rosen/models/rerank/test_partial_judging.py
````python
import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.evaluate import EvalContext, JudgeModelScorer, Prediction
from rosen.models.rerank import MultiGERerankModel, RERANK_MODEL_REGISTRY, select_by_order
from tests.rosen.models.rerank.test_reference_models import NAMES, case


def test_select_by_order_preserves_feature_axes_and_shared_orders():
    values = torch.arange(2 * 10 * 3).reshape(2, 10, 3)
    shared = torch.tensor([[9, 2, 0, 5], [1, 7, 3, 8]])
    selected = select_by_order(values, shared)
    assert selected.shape == (2, 2, 4, 3)
    torch.testing.assert_close(selected[:, 0], values[:, shared[0]])


@pytest.mark.parametrize('name', [n for n in NAMES if n not in ('extr', 'pier', 'prs', 'pier_submodular')
                                 and not issubclass(RERANK_MODEL_REGISTRY[n], MultiGERerankModel)])
def test_partial_judge_matches_explicit_padded_features_and_ignores_unselected_items(tmp_path, name):
    model, batch = case(tmp_path, name)
    model.eval()
    batch['mask'] = torch.ones_like(batch['mask'])
    order = torch.tensor([[[1, 0]], [[1, 0]]])
    before = {key: value.clone() for key, value in batch.interaction.items()}
    with torch.no_grad():
        result = model.score_permutations(batch, order)
        explicit = Interaction({key: value.clone() for key, value in batch.interaction.items()})
        for key in ('cur_item', 'cur_category', 'prior', 'mask'):
            explicit[key] = torch.cat((batch[key][:, [1, 0]], torch.zeros_like(batch[key][:, :1])), 1)
        explicit['slate_len'] = torch.full_like(batch['slate_len'], 2)
        expected = model.logits(explicit).sigmoid()[:, :2]
        torch.testing.assert_close(result[:, 0], expected)
        changed = Interaction({key: value.clone() for key, value in batch.interaction.items()})
        for key in ('cur_item', 'cur_category'):
            changed[key][:, 2] = 7
        changed['prior'][:, 2] = -100
        changed['label'].fill_(-123)
        torch.testing.assert_close(model.score_permutations(changed, order), result)
    assert result.shape == (2, 1, 2) and torch.isfinite(result).all()
    for key, value in before.items():
        torch.testing.assert_close(batch[key], value)
    judge = JudgeModelScorer(model)
    torch.testing.assert_close(judge(EvalContext(Prediction(order=order[:, 0]), batch), order[:, 0]), result[:, 0])


def test_partial_mask_is_applied_even_when_schema_has_no_mask(tmp_path):
    model, batch = case(tmp_path, 'prm')
    # Do not mutate the shared manager: create an unlabeled inference configuration.
    model.LABEL = None
    batch.drop('mask')
    result = model.eval().score_permutations(batch, torch.tensor([[[1]], [[0]]]))
    assert result.shape == (2, 1, 1) and torch.isfinite(result).all()


@pytest.mark.parametrize('name', ['prm', 'mge3'])
def test_judges_reject_nonpermutations_at_the_boundary(tmp_path, name):
    model, batch = case(tmp_path, name)
    for order in (torch.tensor([[[1, 1]], [[0, 1]]]), torch.tensor([[[1, 3]], [[0, 1]]])):
        with pytest.raises(ValueError):
            model.score_permutations(batch, order)


@pytest.mark.parametrize('name', ['extr', 'pier', 'prs', 'pier_submodular', 'mge3', 'mge_cosine', 'nar', 'seq2slate', 'mge_metric'])
def test_fixed_slot_models_explicitly_reject_partial_lists(tmp_path, name):
    model, batch = case(tmp_path, name)
    with pytest.raises(ValueError):
        model.score_permutations(batch, torch.tensor([[[1, 0]], [[1, 0]]]))
````

## File: tests/rosen/models/rerank/test_reference_models.py
````python
"""Behavior, equations and Trainer integration for the independent rerankers."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rosen.core.enum_type import FeatureSource
from rosen.core.interaction import Interaction
from rosen.core.train import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer
from rosen.emb import SchemaManager, SlateLabel
from rosen.emb.schema import SparseEmbSetting, SequenceSetting, SeqGroupConfig
from rosen.evaluate import RankingEvaluator
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.models.rerank import RERANK_MODEL_REGISTRY, select_by_order, build_rerank_model, masked_bce_with_logits
from rosen.models.rerank.dcn import CrossNetwork
from rosen.models.rerank.extr import EXTRAttentionBlock
from rosen.models.rerank.kuaishou import TargetContextPool
from rosen.models.rerank.mge import MultiGERerankModel


NAMES = ("dnn", "dcn", "prm", "extr", "kuaishou", "pier", "prs", "nar", "seq2slate",
         "pier_submodular", "mge_metric", "mge_cosine_metric")


def case(tmp_path, name, **overrides):
    torch.manual_seed(71)
    atoms = [SparseEmbSetting(n, FeatureSource.ITEM, embedding_dim=4) for n in ("item", "category")]
    for atom in atoms:
        atom._build_vocab_indices([str(i) for i in range(12)])
    current, history = SeqGroupConfig("slate", "slate_len", 3), SeqGroupConfig("history", "history_len", 6)
    settings = [SequenceSetting(f"cur_{s.field_name}", s, current) for s in atoms]
    settings += [SequenceSetting(f"hist_{s.field_name}", s, history) for s in atoms]
    manager = SchemaManager(settings, tmp_path / name / "schema", label_fields=SlateLabel("label", 3, "mask"))
    kwargs = dict(slate_cate_group="slate", hist_cate_group="history", slate_dense_field="prior",
                  page_mask_field="pages", poi_num=3, page_num=2, mlp_a=(8,), mlp_b=(8,))
    if name in ("pier", "prs", "pier_submodular", "nar", "seq2slate", "mge_metric", "mge_cosine_metric", "mge3", "mge_cosine"):
        kwargs.update(top_k=2, eval_all_perms=False, candidate_chunk_size=2)
    if name in ("nar", "seq2slate", "mge_metric", "mge_cosine_metric", "mge3", "mge_cosine"):
        kwargs["hidden_dim"] = 8
    kwargs.update(overrides)
    model = build_rerank_model(manager, name, **kwargs)
    batch = Interaction({
        "cur_item": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "cur_category": torch.tensor([[3, 1, 2], [6, 4, 5]]),
        "hist_item": torch.tensor([[1, 3, 2, 3, 2, 1], [6, 4, 5, 6, 5, 4]]),
        "hist_category": torch.tensor([[3, 1, 2, 1, 2, 3], [4, 5, 6, 5, 4, 6]]),
        "slate_len": torch.tensor([3, 3]), "history_len": torch.tensor([6, 6]),
        "prior": torch.tensor([[.1, .5, .9], [.7, .3, .2]]),
        "pages": torch.tensor([[1., 0.], [1., 1.]]),
        "label": torch.tensor([[1., 0., 0.], [0., 1., 1.]]),
        "mask": torch.tensor([[1., 1., 0.], [1., 1., 1.]]),
    })
    return model, batch


def test_registry_exposes_distinct_implementations():
    assert all(name in RERANK_MODEL_REGISTRY for name in NAMES)
    assert len({RERANK_MODEL_REGISTRY[name] for name in NAMES}) == len(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_trainable_predictions_loss_and_checkpoint_restore(tmp_path, name):
    model, batch = case(tmp_path, name)
    loader = DataLoader([batch], batch_size=None)
    evaluators = EvaluatorManager(model.manager)
    evaluators.register("overall", PointwiseEvaluator("auc", "logloss"))
    evaluators.register("order", RankingEvaluator("ndcg@3"))
    cfg = SimpleNamespace(device="cpu", max_epochs=1, train_log_interval=100,
                          ckpt_dir=str(tmp_path / name / "checkpoints"), experiment_name=name, dataset_name="synthetic")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    trainer = BaseTrainer(model, optimizer, model.manager, TrainerDataLoaders(loader, loader),
                          TrainerComponents(evaluator_manager=evaluators), cfg)
    before = {key: value.clone() for key, value in model.named_parameters()}
    trainer.run()
    assert any(not torch.equal(before[key], value) for key, value in model.named_parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        prediction = model.predict(batch)
    assert prediction.scores.shape == prediction.order.shape == (2, 3)
    assert model.outputs_candidate_pool == (prediction.candidates is not None)
    assert not {'topk', 'topk_order', 'chosen_scores'} & prediction.extra.keys()
    if model.outputs_candidate_pool:
        assert prediction.chosen_scores.shape == prediction.order.shape
        assert prediction.resolve_chosen_scores() is prediction.chosen_scores
    else:
        torch.testing.assert_close(prediction.resolve_chosen_scores(), prediction.scores.gather(-1, prediction.order))
    assert torch.isfinite(prediction.scores).all()
    assert ((prediction.scores >= 0) & (prediction.scores <= 1)).all()
    torch.testing.assert_close(prediction.order.sort(-1).values, torch.arange(3).expand(2, 3))
    terms = model.lsp.summary()["settlements"]["main"]
    assert terms["total"] == pytest.approx(sum(v["weighted"] for v in terms["terms"].values()), abs=1e-5)
    checkpoint = trainer.save_checkpoint("roundtrip")
    with torch.no_grad():
        next(model.parameters()).add_(10)
    trainer.load_checkpoint(checkpoint)
    unlabeled = Interaction({key: batch[key] for key in batch.columns if key != "label"})
    with torch.no_grad():
        restored = model.predict(unlabeled)
    torch.testing.assert_close(restored.scores, prediction.scores, rtol=0, atol=0)
    torch.testing.assert_close(restored.order, prediction.order, rtol=0, atol=0)
    teacher = copy.deepcopy(model)
    assert teacher.slate_view.omni is teacher.omni_embedding
    assert all(p.ndim <= 2 for p in model.parameters())


@pytest.mark.parametrize("name", ["dnn", "dcn", "prm"])
def test_item_scorers_are_permutation_equivariant(tmp_path, name):
    model, batch = case(tmp_path, name)
    model.eval()
    order = torch.tensor([2, 0, 1])
    changed = Interaction({key: batch[key].clone() for key in batch.columns})
    for key in ("cur_item", "cur_category", "prior", "label", "mask"):
        changed[key] = changed[key][:, order]
    torch.testing.assert_close(model.logits(changed), model.logits(batch)[:, order])


def test_dnn_is_item_local_and_prm_uses_other_items(tmp_path):
    for name in ("dnn", "dcn", "prm"):
        model, batch = case(tmp_path, name)
        batch["prior"].requires_grad_()
        model.logits(batch)[1, 0].backward()
        cross_slot = batch["prior"].grad[1, 1].abs()
        assert cross_slot > 0 if name == "prm" else cross_slot == 0


def test_cross_network_matches_hand_computed_vector_recurrence():
    network = CrossNetwork(2, 2)
    with torch.no_grad():
        network.weights[0].weight.copy_(torch.tensor([[1., 2.]]))
        network.weights[1].weight.copy_(torch.tensor([[-1., .5]]))
        network.biases[0].copy_(torch.tensor([.1, -.2]))
        network.biases[1].copy_(torch.tensor([.3, .4]))
    x = torch.tensor([[[1., 3.]]])
    first = torch.tensor([[[8.1, 23.8]]])
    second = torch.tensor([[[12.2, 35.6]]])
    torch.testing.assert_close(network(x), torch.cat((x, first, second), -1))


def test_extr_residual_structure_masks_and_causality():
    torch.manual_seed(4)
    block = EXTRAttentionBlock(8, heads=2, attention_dropout=0, ffn_dropout=0)
    query, context = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    valid = torch.ones(2, 3, dtype=torch.bool)
    baseline = block(query, context, valid, valid, causal=True)
    changed = context.clone()
    changed[:, -1] += torch.randn(2, 8) * 100
    actual = block(query, changed, valid, valid, causal=True)
    torch.testing.assert_close(actual[:, :2], baseline[:, :2], rtol=0, atol=0)
    assert not torch.allclose(actual[:, 2], baseline[:, 2])
    valid[:, -1] = False
    masked = block(query, context, valid, valid)
    torch.testing.assert_close(block(query, changed, valid, valid), masked, rtol=0, atol=0)
    assert masked[:, -1].count_nonzero() == 0
    for layer in block.modules():
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
    expected = F.layer_norm(query, (8,), eps=1e-8) * valid[..., None]
    torch.testing.assert_close(block(query, context, valid, valid), expected)


def test_attention_initialization_uses_linear_fan_in_and_keeps_delta_visible():
    torch.manual_seed(37)
    pool = TargetContextPool(3, 64)
    expected_std = (2 / (64 + 64)) ** .5
    for layer in (pool.q, pool.k, pool.v):
        assert .85 < layer.weight.std().item() / expected_std < 1.15
    x = torch.randn(32, 3, 64)
    weights = (pool.q(x) @ pool.k(x).transpose(-1, -2) / 8).softmax(-1)
    delta = weights @ pool.v(x)
    ratio = delta.square().mean().sqrt() / x.square().mean().sqrt()
    assert .1 < ratio < 3


def test_kuaishou_pool_residual_and_empty_history_formula():
    pool = TargetContextPool(2, 2, scale=1)
    with torch.no_grad():
        for layer in (pool.q, pool.k, pool.v):
            layer.weight.copy_(torch.eye(2))
        pool.pool.weight.copy_(torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]]))
    target = torch.tensor([[[1., 0.]]])
    context = torch.tensor([[[1., 2.], [3., 4.]]])
    weights = torch.tensor([1., 3.]).softmax(0)
    delta = weights[0] * context[:, 0] + weights[1] * context[:, 1]
    torch.testing.assert_close(pool(target, context, torch.ones(1, 2, dtype=torch.bool)),
                               (context.sum(1) + 2 * delta)[:, None])
    assert pool(target, context, torch.zeros(1, 2, dtype=torch.bool)).count_nonzero() == 0


@pytest.mark.parametrize("name", ["kuaishou", "pier", "prs"])
def test_invalid_history_cannot_change_current_scores(tmp_path, name):
    model, batch = case(tmp_path, name)
    changed = Interaction({key: batch[key].clone() for key in batch.columns})
    for key in ("hist_item", "hist_category"):
        changed[key][0, 3:] = 9
    torch.testing.assert_close(model.logits(changed)[0], model.logits(batch)[0], rtol=0, atol=0)
    batch["pages"].zero_()
    assert torch.isfinite(model.logits(batch)).all()


def test_pier_candidates_match_sample_and_permutation_loops(tmp_path):
    model, batch = case(tmp_path, "pier", eval_all_perms=True)
    cate, history, dense, pages = model._inputs(batch)
    orders = model.retriever.full_permutations
    actual = model.candidate_logits(cate, history, dense, pages, orders)
    expected = []
    for b in range(2):
        sample = []
        for order in orders:
            value, _ = model.evaluator(cate[b:b+1, order], model.oam(history[b:b+1])[0], dense[b:b+1, order], pages[b:b+1],
                                       model.oam, model.intra_poi, model.target_attention)
            sample.append(value[0])
        expected.append(torch.stack(sample))
    torch.testing.assert_close(actual, torch.stack(expected))
    pred = model.predict(batch)
    best = pred.candidates.scores.mean(-1).argmax(-1)
    torch.testing.assert_close(pred.order, pred.candidates.orders[torch.arange(2), best])
    assert pred.extra["all"].shape == (2, 6, 3)
    assert not any("generator" in name for name, _ in model.named_parameters())


def test_prs_supervises_raw_beam_logits_with_transported_labels(tmp_path, monkeypatch):
    model, batch = case(tmp_path, "prs", use_contrastive_loss=False)
    batch["mask"].fill_(1)
    monkeypatch.setattr(model, "candidate_logits", lambda cate, hist, prior, pages, order:
                        4 * select_by_order(prior.squeeze(-1), order))
    actual = model.loss_components(batch)["supervised"]
    # Gathering logits/labels/masks together leaves this additive BCE invariant.
    expected = masked_bce_with_logits(4 * batch["prior"], batch["label"], batch["mask"], "nonzero_mean")
    torch.testing.assert_close(actual, expected)
    wrong = F.binary_cross_entropy_with_logits((4 * batch["prior"]).sigmoid(), batch["label"])
    assert not torch.allclose(actual, wrong)


@pytest.mark.parametrize("name,index", [("nar", 1), ("seq2slate", 2)])
def test_reference_single_generator_selection_and_batch_mean_loss(tmp_path, name, index):
    model, batch = case(tmp_path, name)
    inputs = model._inputs(batch)
    full_orders, full_aux = MultiGERerankModel._generate(model, *inputs, teacher_forcing=False)
    selected, aux = model._generate(*inputs, teacher_forcing=False)
    torch.testing.assert_close(selected, full_orders[:, index:index+1])
    torch.testing.assert_close(aux["candidate_dense"], full_aux["candidate_dense"][:, index:index+1])
    terms = model.loss_components(batch)
    _, train = MultiGERerankModel._generate(model, *inputs, labels=batch["label"], label_mask=batch["mask"], teacher_forcing=True)
    torch.testing.assert_close(terms["gen.generative"], train["generative"].loss)
    assert model.predict(batch).candidates.scores.shape == (2, 1, 3)


def test_model_configuration_errors_fail_early(tmp_path):
    with pytest.raises(ValueError, match="partition"):
        case(tmp_path, "extr", natural_indices=[0, 1], ad_indices=[1, 2])
    with pytest.raises(ValueError, match="divisible"):
        case(tmp_path, "extr", num_heads=3)
    with pytest.raises(ValueError, match="requires hist"):
        case(tmp_path, "pier", hist_cate_group=None)
    with pytest.raises(ValueError, match="submodularity"):
        case(tmp_path, "nar", submodularity_mode="lc")
    with pytest.warns(UserWarning, match="unsupported"):
        case(tmp_path, "dnn", typo_width=10)


def test_pier_submodular_appends_farthest_non_retrieved_candidate(tmp_path):
    model, batch = case(tmp_path, "pier_submodular")
    cate, history, dense, pages = model._inputs(batch)
    distance = model.retriever(cate, history, pages)
    orders = model._top_orders(cate, distance)
    ids = model.retriever.top_k(distance, model.top_k)
    perms = model.retriever.full_permutations
    top = perms[ids]
    mean = select_by_order(cate, top).mean(1)
    for b in range(2):
        available = [i for i in range(len(perms)) if i not in ids[b].tolist()]
        farthest = max(available, key=lambda i: (cate[b, perms[i]] - mean[b]).square().sum().item())
        torch.testing.assert_close(orders[b, -1], perms[farthest])
    torch.testing.assert_close(orders[:, :2], top)
    pred = model.predict(batch)
    assert pred.candidates.scores.shape == (2, 3, 3)
    assert pred.extra["candidate_lc_index"].shape == (2,)
    model.top_k = len(perms)
    assert model._top_orders(cate, distance).shape == (2, 6, 3)


@pytest.mark.parametrize("name,count,batch_sum", [("mge_metric", 3, True), ("mge_cosine_metric", 4, False)])
def test_metric_variant_supervises_best_greedy_candidate(tmp_path, name, count, batch_sum):
    model, batch = case(tmp_path, name)
    cate, history, dense, pages = model._inputs(batch)
    orders, aux = model._generate(cate, history, dense, pages, teacher_forcing=False)
    assert orders.shape == (2, count, 3)
    logits = model.candidate_logits(cate.detach(), history, dense.detach(), pages, orders,
                                     candidate_dense=aux["candidate_dense"])
    candidates = logits.sigmoid() * select_by_order(batch["mask"], orders)
    best = candidates.mean(-1).argmax(-1)
    expected = []
    for b in range(2):
        order = orders[b, best[b]]
        expected.append(F.binary_cross_entropy_with_logits(logits[b, best[b]], batch["label"][b, order],
                                                           weight=batch["mask"][b, order], reduction="sum"))
    mask = select_by_order(batch["mask"], orders)[torch.arange(2), best]
    expected = sum(expected) / mask.count_nonzero()
    terms = model.loss_components(batch)
    torch.testing.assert_close(terms["evaluator"], expected)
    _, train = model._generate(cate, history, dense, pages, labels=batch["label"], label_mask=batch["mask"], teacher_forcing=True)
    torch.testing.assert_close(terms["gen.generative"], train["generative"].loss)
    pred = model.predict(batch)
    assert torch.isfinite(pred.extra["candidate_mean_l2"]).all()


def test_named_three_and_four_generator_presets(tmp_path):
    for name, mode, count in (("mge3", "none", 3), ("mge_cosine", "cosine", 4)):
        model, batch = case(tmp_path, name)
        assert model.submodularity_mode == mode
        assert model.candidate_dense_mode == "generator"
        assert model.predict(batch).candidates.scores.shape == (2, count, 3)
````

## File: tests/rosen/models/rerank/test_shared_scoring.py
````python
import pytest
import torch

from rosen.models.rerank import MultiGERerankModel, SlateRerankModel
from rosen.models.rerank.components.scoring import EncodedSlate, score_candidate_logits, score_candidate_probs
from tests.rosen.models.rerank.test_reference_models import case


def test_pier_predict_embeds_the_inputs_once(tmp_path, monkeypatch):
    model, batch = case(tmp_path, 'pier')
    inputs, calls = model._inputs, []

    def tracked(inter):
        calls.append(inter)
        return inputs(inter)

    monkeypatch.setattr(model, '_inputs', tracked)
    with torch.no_grad():
        pred = model.eval().predict(batch)
    assert calls == [batch]
    torch.testing.assert_close(pred.scores, model.logits(batch).sigmoid())


def test_multige_inherits_the_slate_input_and_loss_contract():
    assert issubclass(MultiGERerankModel, SlateRerankModel)
    for name in ('_inputs', '_mask', '_view', 'calculate_loss'):
        assert getattr(MultiGERerankModel, name) is getattr(SlateRerankModel, name)


@pytest.mark.parametrize('name', ['mge3', 'nar', 'seq2slate', 'mge_metric'])
def test_skipping_diagnostics_preserves_weights_topk_and_final_ranking(tmp_path, name):
    full, batch = case(tmp_path, name, eval_all_perms=True)
    fast, _ = case(tmp_path, name, eval_all_perms=True, eval_diagnostic_pools=False)
    assert list(full.state_dict()) == list(fast.state_dict())
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, fast.state_dict()[key], rtol=0, atol=0)
    calls = []
    score = fast._score_candidates

    def tracked(*args, **kwargs):
        calls.append(args[5].shape)
        return score(*args, **kwargs)

    fast._score_candidates = tracked
    with torch.no_grad():
        expected = full.eval().predict(batch)
        actual = fast.eval().predict(batch)
    assert len(calls) == 1
    assert 'lastk' in expected.extra and 'all' in expected.extra
    assert not {'lastk', 'randk', 'beamk', 'all'} & actual.extra.keys()
    torch.testing.assert_close(actual.candidates.scores, expected.candidates.scores, rtol=0, atol=0)
    torch.testing.assert_close(actual.candidates.orders, expected.candidates.orders, rtol=0, atol=0)
    torch.testing.assert_close(actual.chosen_scores, expected.chosen_scores, rtol=0, atol=0)
    torch.testing.assert_close(actual.scores, expected.scores, rtol=0, atol=0)
    torch.testing.assert_close(actual.order, expected.order, rtol=0, atol=0)


@pytest.mark.parametrize('candidate_dense', [False, True])
def test_common_probabilities_match_individual_calls_with_chunk_tails_and_gradients(tmp_path, candidate_dense):
    model, batch = case(tmp_path, 'mge3')
    cate, history, dense, pages = model._inputs(batch)
    orders = torch.tensor([[[2, 0, 1], [1, 2, 0], [0, 1, 2]],
                           [[0, 2, 1], [2, 1, 0], [1, 0, 2]]])
    prior = torch.rand(2, 3, 3, 1, requires_grad=True) if candidate_dense else None
    common = dict(scorer=model._score_encoded, chunk_size=2, candidate_dense=prior)
    encoded = EncodedSlate(cate, dense, batch['mask'], model._history_context(history, pages))
    logits = score_candidate_logits(encoded, orders, **common)
    valid = batch['mask'].gather(1, orders.flatten(1)).reshape_as(orders)
    result = score_candidate_probs(encoded, orders, **common)
    torch.testing.assert_close(result, logits.sigmoid() * valid, rtol=0, atol=0)
    # Shared and row-wise order forms have the same semantics for identical rows.
    shared = orders[0]
    common['candidate_dense'] = None
    one = score_candidate_probs(encoded, shared, **common)
    other = score_candidate_probs(encoded, shared[None].expand(2, -1, -1), **common)
    torch.testing.assert_close(one, other, rtol=0, atol=0)
    result.sum().backward()
    assert model.evaluator.output.weight.grad.abs().sum() > 0
    if prior is not None:
        assert torch.isfinite(prior.grad).all() and prior.grad.abs().sum() > 0
````

## File: workspace/kr_time_budget/pipeline/architecture_examples.py
````python
"""KR1K examples: scalar multi-task supervision and native six-item reranking."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rosen.core.enum_type import FeatureSource
from rosen.data.dataset import ParquetStreamDataset
from rosen.emb import AuxField, SchemaManager, SlateLabel
from rosen.emb.schema import MinMaxDenseSetting, SeqGroupConfig, SequenceSetting, SparseEmbSetting
from rosen.evaluate import BatchResolver, FieldScorer, Prediction, RankingEvaluator, SlateValueEvaluator
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.models.base import BaseModel
from rosen.models.rerank import RERANK_MODEL_REGISTRY, build_rerank_model, resolve_rerank_config


WORKSPACE = Path(__file__).resolve().parents[1]
CATEGORIES = ("user_id", "video_id", "author_id", "tab")
LABELS = ("is_click", "long_view")
POIS, PAGES = 6, 2


def atoms():
    return [SparseEmbSetting(name, FeatureSource.USER_ID if name == "user_id" else FeatureSource.ITEM,
                             embedding_dim=8) for name in CATEGORIES]


def pointwise_manager(path):
    return SchemaManager([*atoms(), MinMaxDenseSetting("duration_s", FeatureSource.ITEM)],
                         path, label_fields=LABELS, aux_fields=[
                             AuxField("event_id", dtype="int64"), AuxField("request_id", dtype="int64"),
                             AuxField("slot", dtype="int64"), AuxField("user_id_raw", dtype="int64"),
                         ])


def rerank_manager(path):
    shared = atoms()
    current = SeqGroupConfig("slate", "slate_len", POIS)
    history = SeqGroupConfig("history", "history_len", PAGES * POIS)
    settings = [SequenceSetting(f"cur_{s.field_name}", s, current) for s in shared]
    settings += [SequenceSetting(f"hist_{s.field_name}", s, history) for s in shared]
    return SchemaManager(settings, path, label_fields=[SlateLabel(name, POIS, "mask") for name in LABELS],
                         aux_fields=[
                             AuxField("request_id", dtype="int64"), AuxField("user_id_raw", dtype="int64"),
                             AuxField("page_mask", max_len=PAGES), AuxField("prior_long_view", max_len=POIS),
                         ])


class KRPointwiseMultiTaskModel(BaseModel):
    """One feature vector per exposure; independent CTR and long-view heads."""
    def __init__(self, manager):
        super().__init__(manager)
        self.view = self.omni_embedding.whole
        self.shared = nn.Sequential(nn.Linear(self.view.embedding_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU())
        self.heads = nn.ModuleDict({"ctr": nn.Linear(32, 1), "long_view": nn.Linear(32, 1)})
        self.apply_xavier_initialization()

    def logits(self, interaction):
        hidden = self.shared(self.view(interaction))
        return {name: head(hidden).squeeze(-1) for name, head in self.heads.items()}

    def calculate_loss(self, interaction):
        logits = self.logits(interaction)
        terms = {"ctr": F.binary_cross_entropy_with_logits(logits["ctr"], interaction["is_click"].float()),
                 "long_view": F.binary_cross_entropy_with_logits(logits["long_view"], interaction["long_view"].float())}
        if self.lsp.active:
            for name, loss in terms.items():
                self.lsp.add(f"task.{name}", loss)
        return sum(terms.values())

    def predict(self, interaction):
        logits = self.logits(interaction)
        return Prediction(scores={"is_click": logits["ctr"].sigmoid(), "long_view": logits["long_view"].sigmoid()})


def reranker_config(*, top_k=4, mode=None, candidate_dense_mode=None, model_name="mge"):
    config = dict(slate_cate_group="slate", hist_cate_group="history", slate_dense_field="prior_long_view",
        page_mask_field="page_mask", label_field="long_view", poi_num=POIS, page_num=PAGES,
        mlp_a=(32, 16), mlp_b=(32, 16),
    )
    if model_name in ("pier", "prs", "pier_submodular", "nar", "seq2slate", "mge", "lc_mge",
                      "mge3", "mge_cosine", "mge_metric", "mge_cosine_metric"):
        config.update(top_k=top_k, eval_all_perms=False, candidate_chunk_size=16)
    config = resolve_rerank_config(model_name, **config)
    config.setdefault("loss_reduction", "valid_mean")
    if model_name in ("mge", "lc_mge", "mge3", "mge_cosine", "mge_metric", "mge_cosine_metric", "nar", "seq2slate"):
        config["hidden_dim"] = 32
        if mode is not None:
            config["submodularity_mode"] = mode
        if candidate_dense_mode is not None:
            config["candidate_dense_mode"] = candidate_dense_mode
        config.setdefault("submodularity_mode", "lc")
        config.setdefault("candidate_dense_mode", "prior")
        config["loss_reduction"] = "nonzero_mean" if config["candidate_dense_mode"] == "generator" else "valid_mean"
    return config


def make_reranker(manager, *, top_k=4, mode=None, candidate_dense_mode=None, model_name="mge"):
    name = model_name
    return build_rerank_model(manager, name, **reranker_config(
        top_k=top_k, mode=mode, candidate_dense_mode=candidate_dense_mode, model_name=name))


def loader(path, manager, batch_size):
    dataset = ParquetStreamDataset(path, manager, batch_size=batch_size, shuffle=False, drop_last=False)
    return DataLoader(dataset, batch_size=None, num_workers=0)


def evaluators(task, log_path=None, *, manager=None, rerank_model="mge"):
    result = EvaluatorManager(manager, log_path=log_path, title=f"KR1K_{task}", uid_field="user_id_raw")
    if task == "pointwise":
        result.register("ctr", PointwiseEvaluator("auc", "logloss", "gauc"), target="is_click")
        result.register("long_view", PointwiseEvaluator("auc", "logloss", "gauc"), target="long_view")
    else:
        result.register("long_view", PointwiseEvaluator("auc", "logloss", "gauc"))
        result.register("slate", RankingEvaluator("ndcg@6", "pcoc"), resolver=BatchResolver(view="score_only"))
        result.register("chosen_order", RankingEvaluator("ndcg@6"))
        result.register("pointwise_prior", RankingEvaluator("ndcg@6"),
                        resolver=BatchResolver(pred_scorer=FieldScorer("prior_long_view"), view="score_only"))
        if RERANK_MODEL_REGISTRY[rerank_model.lower()].outputs_candidate_pool:
            result.register("candidates", SlateValueEvaluator("cand_mean", "cand_max", reduction="mean"),
                            resolver=BatchResolver.for_candidates())
    return result
````

## File: workspace/kr_time_budget/pipeline/prepare_architecture_examples.py
````python
"""Read-only reuse of v3 membership; keep feedback exclusively in supervision."""
import gc
import json
from pathlib import Path

import polars as pl

from rosen.datasets.kuairand import KuaiRandDataset
from rosen.utils.shuffle import write_shuffled_parquet
from .architecture_examples import CATEGORIES, LABELS, PAGES, POIS, pointwise_manager, rerank_manager


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def flat_events(source, video_path, split, limit_requests=None):
    """Exactly the event membership of existing supervised six-item requests."""
    requests = pl.scan_parquet(source / "requests_base.parquet").filter(pl.col("split") == split).select(
        "request_id", "session_id", "req_idx", "tab", "session_n", "session_date", "end_date")
    if limit_requests is not None:
        requests = requests.sort("request_id").head(limit_requests)
    members = pl.scan_parquet(source / "request_events.parquet").select("event_id", "request_id", "event_idx")
    events = pl.scan_parquet(source / "events.parquet").select("event_id", "user_id", "video_id", "duration_ms", *LABELS)
    videos = pl.scan_csv(video_path).select("video_id", "author_id")
    return (members.join(requests, on="request_id", how="inner")
            .join(events, on="event_id", how="inner")
            .join(videos, on="video_id", how="left")
            .with_columns(pl.col("author_id").fill_null(-1), pl.col("user_id").alias("user_id_raw"),
                          ((pl.col("event_idx") - pl.col("session_n") % POIS) % POIS).cast(pl.Int8).alias("slot"),
                          (pl.col("duration_ms").clip(0, 600000) / 1000).cast(pl.Float32).alias("duration_s"))
            .select("event_id", "request_id", "session_id", "req_idx", "session_date", "end_date", "slot", "user_id_raw",
                    *CATEGORIES, "duration_s", *LABELS).sort(["request_id", "slot"]))


def group_slates(events):
    """Only earlier complete pages in the same session; pack valid pages left."""
    grouped = events.sort(["request_id", "slot"]).group_by("request_id", maintain_order=True).agg(
        pl.col("session_id").first(), pl.col("req_idx").first(), pl.col("user_id_raw").first(),
        *[pl.col(name).alias(f"cur_{name}") for name in CATEGORIES],
        *[pl.col(name) for name in LABELS], pl.col("prior_ctr"), pl.col("prior_long_view"),
        pl.col("event_id").alias("event_ids"),
    ).sort(["session_id", "req_idx"])
    previous_count = pl.col("request_id").cum_count().over("session_id") - 1
    return grouped.with_columns(
        *[pl.concat_list([pl.col(f"cur_{name}").shift(offset).over("session_id").fill_null([])
                          for offset in range(PAGES, 0, -1)]).alias(f"hist_{name}") for name in CATEGORIES],
        pl.concat_list([(previous_count > index).cast(pl.Float32) for index in range(PAGES)]).alias("page_mask"),
        pl.lit([1.] * POIS).alias("mask"),
    )


def prepare_pointwise(source, output, video_path=None, limit_requests=None, seed=2026):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info["limit_requests"] != limit_requests or info["seed"] != seed or info["source"] != str(source.resolve()):
            raise ValueError("Dataset contract changed; choose a fresh output directory.")
        return info
    manager = pointwise_manager(output / "schema")
    stats = {}
    video_path = video_path or KuaiRandDataset.VIDEO_FEATURES
    for split in ("train", "valid", "test"):
        raw_path = output / f"{split}.raw.parquet"
        if not raw_path.exists():
            flat_events(source, video_path, split, limit_requests).sink_parquet(raw_path, compression="zstd")
        raw = pl.scan_parquet(raw_path)
        audit = raw.select(pl.len().alias("rows"), pl.col("event_id").n_unique().alias("events"),
                           pl.col("request_id").n_unique().alias("requests"),
                           *[pl.col(name).mean().alias(name) for name in LABELS]).collect().row(0, named=True)
        if audit["rows"] != audit["events"] or audit["rows"] != audit["requests"] * POIS:
            raise ValueError(f"Invalid six-slot event membership: {split} {audit}")
        if split == "train":
            manager.fit(raw)
        encoded = output / f"{split}.ordered.parquet"
        manager.transform(raw).sink_parquet(encoded, compression="zstd")
        if split == "train":
            write_shuffled_parquet(encoded, output / "train.parquet", seed=seed, bucket_count=16)
        stats[split] = audit
        print("POINT_DATA", split, audit, flush=True)
        gc.collect()
    info = dict(source=str(source.resolve()), video_path=str(Path(video_path).resolve()), seed=seed,
                limit_requests=limit_requests, splits=stats, history="earlier complete six-item pages only",
                labels={"ctr": "raw is_click", "long_view": "raw long_view"},
                split="unchanged v3 sessions: train end<=20220506, valid 20220507, test 20220508",
                source_files={p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                              for p in [source / "requests_base.parquet", source / "request_events.parquet", source / "events.parquet"]})
    dump(manifest, info)
    return info


def prepare_rerank(point_data, priors, output, seed=2026):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        raise FileExistsError("Rerank data is tied to a specific trained prior; choose a fresh output.")
    manager = rerank_manager(output / "schema")
    stats = {}
    for split in ("train", "valid", "test"):
        raw = pl.scan_parquet(point_data / f"{split}.raw.parquet")
        scores = pl.scan_parquet(priors / f"{split}.parquet")
        grouped = group_slates(raw.join(scores, on="event_id", how="left", validate="1:1"))
        raw_path = output / f"{split}.raw.parquet"
        grouped.sink_parquet(raw_path, compression="zstd")
        raw = pl.scan_parquet(raw_path)
        audit = raw.select(pl.len().alias("requests"),
                           pl.col("prior_long_view").list.eval(pl.element().is_null()).list.any().sum().alias("missing_priors"),
                           (pl.col("hist_video_id").list.len() / POIS).mean().alias("mean_history_pages")).collect().row(0, named=True)
        expected = json.loads((point_data / "manifest.json").read_text())["splits"][split]["requests"]
        if audit["requests"] != expected or audit["missing_priors"]:
            raise ValueError(f"Rerank rows or priors do not align: {audit}")
        if split == "train":
            manager.fit(raw)
        encoded = output / f"{split}.ordered.parquet"
        manager.transform(raw).sink_parquet(encoded, compression="zstd")
        if split == "train":
            write_shuffled_parquet(encoded, output / "train.parquet", seed=seed, bucket_count=16)
        stats[split] = audit
        print("SLATE_DATA", split, audit, flush=True)
        gc.collect()
    info = dict(source=str(point_data.resolve()), prior_source=str(priors.resolve()), seed=seed, splits=stats,
                poi_num=POIS, page_num=PAGES, primary_label="long_view", prior="frozen pointwise long_view probability",
                history="previous two complete requests in same session; head remainder excluded; valid pages packed left",
                masking="all six current exposures valid; page_mask tracks complete history pages")
    dump(manifest, info)
    return info
````

## File: workspace/kr_time_budget/pipeline/run_architecture_examples.py
````python
"""Explicit full-train -> validation -> checkpoint -> full-test examples."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import polars as pl
import pyarrow.parquet as pq
import torch

from rosen.core.train import EarlyStopper, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer
from rosen.models.rerank import RERANK_MODEL_REGISTRY
from rosen.utils.expr_tensor import ExprTensor
from rosen.utils.optimize import split_model_params_by_decay
from .architecture_examples import (
    WORKSPACE, KRPointwiseMultiTaskModel, evaluators, loader, make_reranker, pointwise_manager, rerank_manager,
    reranker_config,
)
from .prepare_architecture_examples import dump, prepare_pointwise, prepare_rerank


class Tee:
    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream

    def write(self, text):
        self.terminal.write(text)
        self.stream.write(text)
        self.stream.flush()

    def flush(self):
        self.terminal.flush()
        self.stream.flush()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_early_stopper(task, *, patience=2):
    return EarlyStopper(patience=patience,
                        preferred_evaluators=("chosen_order",) if task == "rerank" else ("long_view",),
                        preferred_metrics=("ndcg@6",) if task == "rerank" else ("auc",))


def train_example(task, data, output, args):
    output.mkdir(parents=True)
    manager = (pointwise_manager if task == "pointwise" else rerank_manager)(data / "schema")
    manager.load_schema()
    torch.manual_seed(args.seed)
    model_name = getattr(args, "rerank_model", "mge")
    model = KRPointwiseMultiTaskModel(manager) if task == "pointwise" else make_reranker(
        manager, top_k=args.top_k, mode=args.mode, candidate_dense_mode=args.candidate_dense_mode, model_name=model_name)
    if task == "rerank":
        dump(output / "model_config.json", {
            "model": model_name,
            "schema": str((data / "schema").resolve()),
            "kwargs": reranker_config(top_k=args.top_k, mode=args.mode, candidate_dense_mode=args.candidate_dense_mode,
                                      model_name=model_name),
        })
    batch_size = args.point_batch_size if task == "pointwise" else args.rerank_batch_size
    loaders = {split: loader(data / ("train.parquet" if split == "train" else f"{split}.ordered.parquet"),
                             manager, batch_size) for split in ("train", "valid", "test")}
    metadata = json.loads((data / "manifest.json").read_text())
    counts = {split: stats["rows" if task == "pointwise" else "requests"] for split, stats in metadata["splits"].items()}
    epochs = getattr(args, "epochs", 1)
    cfg = SimpleNamespace(device=args.device, max_epochs=epochs, train_log_interval=100,
                          ckpt_dir=str(output / "checkpoints"), experiment_name=f"new_architecture_{task}", dataset_name="KR1K_v3_six_item")
    optimizer = torch.optim.AdamW(split_model_params_by_decay(model, weight_decay=1e-6), lr=.001, weight_decay=1e-6)
    stopper = make_early_stopper(task, patience=max(2, epochs + 1))
    evaluator = evaluators(task, output / "evaluation.log", rerank_model=model_name)
    trainer = BaseTrainer(model, optimizer, manager,
                          TrainerDataLoaders(loaders["train"], loaders["valid"], loaders["test"]),
                          TrainerComponents(evaluator_manager=evaluator, early_stopper=stopper), cfg)
    first = next(iter(loaders["train"])).to(args.device)
    shapes = {field: list(first[field].shape) for field in first.columns}
    if task == "rerank":
        cate, hist, dense, pages = model._inputs(first)
        shapes.update(stacked_slate=list(cate.shape), folded_history=list(hist.shape))
    dump(output / "batch_shapes.json", shapes)
    print("SHAPES", task, shapes, flush=True)
    del first
    started = time.monotonic()
    valid = trainer.run()
    train_seconds = time.monotonic() - started
    expected_steps = math.ceil(counts["train"] / batch_size) * epochs
    if trainer.global_step != expected_steps or trainer.epoch != epochs:
        raise RuntimeError(f"Partial training: {trainer.global_step} vs {expected_steps}")
    checkpoint = trainer.best_checkpoint_path
    if checkpoint is None:
        raise RuntimeError("No best checkpoint was saved")
    trained_steps = trainer.global_step
    trainer.load_checkpoint(checkpoint)
    valid = trainer.evaluate_epoch()
    started = time.monotonic()
    test = trainer.evaluate_test()
    test_seconds = time.monotonic() - started
    result = dict(task=task, status="FULL_TRAIN_FULL_TEST" if args.limit_requests is None else "SMOKE_PARTIAL",
                  seed=args.seed, data=str(data.resolve()), split_counts=counts, epochs=epochs,
                  optimizer="AdamW", learning_rate=.001, weight_decay=1e-6,
                  zero_decay="nn.Embedding, biases, norm, scalar/vector and embedding-like free parameters",
                  physical_batch=batch_size, effective_batch=batch_size, gradient_accumulation=1,
                  optimizer_steps=trained_steps, expected_optimizer_steps=expected_steps,
                  device=args.device, threads=args.threads, precision="float32", scheduler=None, ema=False,
                  warm_start=None, selection="max valid chosen_order.ndcg@6" if task == "rerank" else "max valid long_view.auc",
                  checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_hash(checkpoint),
                  train_order_sha256=file_hash(data / "train.parquet"),
                  train_and_valid_seconds=train_seconds, test_seconds=test_seconds, valid=valid, test=test,
                  last_step_loss_ledger=model.lsp.summary(),
                  loss_diagnostics={key: float(value) for key, value in getattr(model, "loss_diagnostics", {}).items()},
                  schema=str((data / "schema").resolve()),
                  rerank_model=model_name if task == "rerank" else None,
                  mode=getattr(model, "submodularity_mode", None) if task == "rerank" else "shared_bottom_two_heads",
                  top_k=getattr(model, "top_k", None) if task == "rerank" else None,
                  candidate_dense_mode=getattr(model, "candidate_dense_mode", None) if task == "rerank" else None,
                  loss_reduction=model.loss_reduction if task == "rerank" else None)
    dump(output / "metrics.json", result)
    print("FINAL_METRICS", json.dumps(result, ensure_ascii=False), flush=True)
    return trainer, loaders, result


@torch.no_grad()
def export_pointwise(trainer, point_data, output, batch_size):
    """Frozen full-train model generates priors. Held-out labels never enter X."""
    output.mkdir(parents=True)
    trainer.model.eval()
    for split in ("train", "valid", "test"):
        batches = loader(point_data / f"{split}.ordered.parquet", trainer.manager, batch_size)
        rows = 0
        with pq.ParquetWriter(output / f"{split}.parquet", schema=pl.DataFrame({
            "event_id": pl.Series([], dtype=pl.Int64), "prior_ctr": pl.Series([], dtype=pl.Float32),
            "prior_long_view": pl.Series([], dtype=pl.Float32)}).to_arrow().schema, compression="zstd") as writer:
            for batch in batches:
                batch = batch.to(trainer.cfg.device)
                predictions = trainer.model.predict(batch)
                frame = pl.DataFrame({"event_id": batch["event_id"].cpu().numpy(),
                                      "prior_ctr": predictions.scores["is_click"].cpu().numpy(),
                                      "prior_long_view": predictions.scores["long_view"].cpu().numpy()}).with_columns(pl.col("event_id").cast(pl.Int64))
                writer.write_table(frame.to_arrow())
                rows += len(batch)
        print("FROZEN_PRIORS", split, rows, flush=True)
    dump(output / "manifest.json", {"point_checkpoint": str(trainer.best_checkpoint_path.resolve()),
                                   "training_priors": "in-sample predictions from full train model; no OOF claim",
                                   "heldout_priors": "frozen train-only model, no label features or further fitting"})


@torch.no_grad()
def export_rerank_test(trainer, batches, output):
    trainer.model.eval()
    writer = None
    rows = 0
    try:
        for batch in batches:
            batch = batch.to(trainer.cfg.device)
            predictions = trainer.model.predict(batch)
            order = predictions.order
            if not torch.equal(order.sort(-1).values, torch.arange(trainer.model.poi_num, device=order.device).expand_as(order)):
                raise RuntimeError("Generated order is not a permutation")
            frame = pl.DataFrame({"request_id": batch["request_id"].cpu().numpy(),
                                  "user_id_raw": batch["user_id_raw"].cpu().numpy(),
                                  "is_click": batch["is_click"].cpu().tolist(), "long_view": batch["long_view"].cpu().tolist(),
                                  "scores": predictions.scores.cpu().tolist(), "out_index": order.cpu().tolist(),
                                  "chosen_scores": predictions.resolve_chosen_scores().cpu().tolist(),
                                  "prior_long_view": batch["prior_long_view"].cpu().tolist()})
            if writer is None:
                writer = pq.ParquetWriter(output, frame.to_arrow().schema, compression="zstd")
            writer.write_table(frame.to_arrow())
            rows += len(batch)
    finally:
        if writer is not None:
            writer.close()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=WORKSPACE / "data/v3")
    parser.add_argument("--data", type=Path, default=WORKSPACE / "data/architecture_examples")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--point-batch-size", type=int, default=8192)
    parser.add_argument("--rerank-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--mode", choices=("none", "cosine", "lc", "lc_soft"), default=None)
    parser.add_argument("--rerank-model", choices=tuple(RERANK_MODEL_REGISTRY), default="mge")
    parser.add_argument("--candidate-dense-mode", choices=("prior", "generator"), default=None,
                        help="generator matches LC_MGE's generated-candidate dense features; prior preserves old runs")
    parser.add_argument("--limit-requests", type=int, help="Only for explicitly marked smoke runs; per split")
    args = parser.parse_args(argv)
    if min(args.threads, args.point_batch_size, args.rerank_batch_size, args.top_k) <= 0:
        parser.error("Thread, batch and retrieval counts must be positive")
    if args.limit_requests is not None and args.data == WORKSPACE / "data/architecture_examples":
        parser.error("Smoke runs require a distinct --data path")
    output = args.output or WORKSPACE / "runs/architecture_examples" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    torch.set_num_threads(args.threads)
    terminal_out, terminal_err = sys.stdout, sys.stderr
    with (output / "logs/train.log").open("w", buffering=1) as stream:
        sys.stdout, sys.stderr = Tee(terminal_out, stream), Tee(terminal_err, stream)
        try:
            print("RUN", vars(args) | {"output": str(output), "torch": torch.__version__, "python": platform.python_version()}, flush=True)
            # A symbolic sanity check records that both labels remain additive
            # objectives; it has no runtime involvement in model training.
            a, b = ExprTensor(torch.tensor(1.), "BCE_CTR"), ExprTensor(torch.tensor(2.), "BCE_LONG_VIEW")
            print("POINTWISE_OBJECTIVE", (a + b).expr, flush=True)
            prepare_pointwise(args.source, args.data / "pointwise", limit_requests=args.limit_requests, seed=args.seed)
            point, _, point_metrics = train_example("pointwise", args.data / "pointwise", output / "pointwise", args)
            export_pointwise(point, args.data / "pointwise", output / "priors", args.point_batch_size)
            del point
            gc.collect()
            slate_data = args.data / f"rerank_{output.name}"
            prepare_rerank(args.data / "pointwise", output / "priors", slate_data, seed=args.seed)
            rerank, batches, rerank_metrics = train_example("rerank", slate_data, output / "rerank", args)
            exported = export_rerank_test(rerank, batches["test"], output / "rerank/test_predictions.parquet")
            if exported != rerank_metrics["split_counts"]["test"]:
                raise RuntimeError("Partial rerank test export")
            dump(output / "summary.json", {"pointwise": point_metrics, "rerank": rerank_metrics,
                                           "source_hashes": {str(p): file_hash(p) for p in [Path(__file__),
                                               Path(__file__).with_name("architecture_examples.py"),
                                               Path(__file__).with_name("prepare_architecture_examples.py")]},
                                           "comparison": "engineering examples; different model/optimizer-step budgets, not a causal A/B",
                                           "data_boundary": "same offline backward six-item v3 grouping; not known simultaneous exposures or online reward"})
        finally:
            sys.stdout, sys.stderr = terminal_out, terminal_err
    print("COMPLETED", output)


if __name__ == "__main__":
    main()
````

## File: workspace/kr_time_budget/tests/test_architecture_examples.py
````python
import json
import math
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.evaluate import BatchResolver, CandidatePool, EvalBatch, Prediction, EvalContext, SlateValueEvaluator
from rosen.data.padding import FeatureContext, TensorFormatter
from rosen.data.dataset import ParquetStreamDataset
from rosen.emb import AuxField, SchemaManager, SlateLabel
from workspace.kr_time_budget.pipeline.architecture_examples import (
    CATEGORIES, KRPointwiseMultiTaskModel, RankingEvaluator, evaluators, make_reranker,
    pointwise_manager, rerank_manager,
)
from workspace.kr_time_budget.pipeline.prepare_architecture_examples import group_slates


def raw_events():
    records = []
    for request, session, index in ((1, 10, 1), (2, 10, 2), (3, 10, 3), (4, 20, 1)):
        for slot in range(6):
            records.append(dict(request_id=request, session_id=session, req_idx=index, slot=slot,
                                event_id=request * 10 + slot, user_id=1, user_id_raw=1,
                                video_id=request * 10 + slot, author_id=slot, tab=1, duration_s=float(slot + 1),
                                is_click=slot % 2, long_view=(slot + 1) % 2,
                                prior_ctr=.1 * slot, prior_long_view=.1 * (5 - slot)))
    return pl.DataFrame(records)


def interaction(frame, manager):
    encoded = manager.transform(frame.lazy()).collect()
    context = FeatureContext.from_manager(manager)
    formatter = TensorFormatter(context)
    return formatter.format({name: encoded[name].to_numpy() for name in context.read_col_names})


def test_six_item_history_is_ordered_packed_and_session_bounded():
    raw = raw_events().sample(fraction=1, shuffle=True, seed=3)
    grouped = group_slates(raw.lazy()).collect().sort("request_id")
    assert grouped["cur_video_id"].to_list() == [[i * 10 + j for j in range(6)] for i in range(1, 5)]
    assert grouped["page_mask"].to_list() == [[0., 0.], [1., 0.], [1., 1.], [0., 0.]]
    assert grouped["hist_video_id"].to_list() == [[], list(range(10, 16)), list(range(10, 16)) + list(range(20, 26)), []]
    assert grouped["is_click"].to_list() == [[0, 1, 0, 1, 0, 1]] * 4
    assert grouped["long_view"].to_list() == [[1, 0, 1, 0, 1, 0]] * 4
    assert all(len(row) == 6 for row in grouped["mask"].to_list())


def test_scalar_multitask_uses_both_labels_and_explicit_routes(tmp_path):
    raw = raw_events()
    manager = pointwise_manager(tmp_path / "point")
    manager.fit(raw.lazy())
    batch = interaction(raw, manager)
    model = KRPointwiseMultiTaskModel(manager)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.heads["ctr"].bias.fill_(math.log(3))
        model.heads["long_view"].bias.fill_(-math.log(3))
    predictions = model.predict(batch)
    torch.testing.assert_close(predictions.scores["is_click"], torch.full((24,), .75))
    torch.testing.assert_close(predictions.scores["long_view"], torch.full((24,), .25))
    assert model.LABELS == ("is_click", "long_view")
    loss = model.calculate_loss(batch)
    expected = -math.log(.75) - math.log(.25)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert model.heads["ctr"].bias.grad.item() > 0
    assert model.heads["long_view"].bias.grad.item() < 0
    evaluator = evaluators("pointwise", manager=manager)
    evaluator.collect(batch, predictions)
    assert set(evaluator.summary()) == {"ctr", "long_view"}


@pytest.mark.parametrize("candidate_dense_mode", ["prior", "generator"])
def test_native_rerank_schema_and_loss_choose_long_view_not_first_label(tmp_path, candidate_dense_mode):
    from workspace.kr_time_budget.pipeline.architecture_examples import loader
    grouped = group_slates(raw_events().lazy()).collect()
    manager = rerank_manager(tmp_path / "schema")
    manager.fit(grouped.lazy())
    encoded = manager.transform(grouped.lazy()).collect()
    path = tmp_path / "encoded.parquet"
    encoded.write_parquet(path)
    batch = next(iter(loader(path, manager, 4)))
    assert set(batch.columns) == set(manager.fields())
    assert batch['request_id'].dtype == batch['user_id_raw'].dtype == torch.int64
    assert manager.get_aux_spec('prior_long_view') == AuxField('prior_long_view', max_len=6)
    assert manager.get_aux_spec('page_mask') == AuxField('page_mask', max_len=2)
    model = make_reranker(manager, candidate_dense_mode=candidate_dense_mode)
    assert manager.label_field == "is_click" and model.LABEL == "long_view"
    assert manager.uid_field is None
    assert manager.mask_fields == ("mask",)
    assert len(model.omni_embedding.emb_modules) == len(CATEGORIES)
    current, history, prior, pages = model._inputs(batch)
    assert current.shape == (4, 6, 4, 8) and history.shape == (4, 2, 6, 4, 8)
    torch.testing.assert_close(prior[..., 0], batch["prior_long_view"])
    logits, _ = model.evaluator(current, model.oam(history)[0], prior, pages, model.oam, model.intra_poi, model.target_attention)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["long_view"])
    torch.testing.assert_close(model.loss_components(batch)["evaluator"], expected)
    model.calculate_loss(batch).backward()
    with torch.no_grad():
        predictions = model.predict(batch)
    evaluator = evaluators("rerank", manager=manager)
    evaluator.collect(batch, predictions)
    results = evaluator.summary()
    assert {name: set(metrics) for name, metrics in results.items()} == {
        "long_view": {"auc", "logloss", "gauc"}, "slate": {"ndcg@6", "pcoc"},
        "chosen_order": {"ndcg@6"}, "pointwise_prior": {"ndcg@6"},
        "candidates": {"cand_mean", "cand_max"},
    }


@pytest.mark.parametrize('width,pages', [(3, 4), (7, 1)])
def test_prediction_reader_follows_schema_names_and_lengths_without_redeclarations(tmp_path, width, pages):
    from workspace.kr_time_budget.pipeline.predict_lc_mge import prediction_batches
    manager = SchemaManager([], tmp_path / 'schema', label_fields=SlateLabel('outcome', width, 'valid_slots'),
                            aux_fields=[AuxField('new_prior', width, padding_side='left', pad_val=-1.),
                                        AuxField('new_pages', pages), AuxField('raw_id', dtype='int64')])
    frame = pl.DataFrame({'outcome': [[1.], [0.]], 'valid_slots': [[1.], [1., 1.]],
                          'new_prior': [[.25], [.5, .75]], 'new_pages': [[1.], []],
                          'raw_id': [2**53 + 1, 2**53 + 2]})
    training = tmp_path / 'training.parquet'
    frame.write_parquet(training)
    inference = tmp_path / 'inference.parquet'
    frame.drop('outcome').write_parquet(inference)
    train, = list(ParquetStreamDataset(training, manager, batch_size=2, shuffle=False, drop_last=False))
    predict, = list(prediction_batches(inference, manager, 2))
    assert set(predict.columns) == set(train.columns) - set(manager.label_fields)
    for field in predict.columns:
        torch.testing.assert_close(predict[field], train[field], rtol=0, atol=0)
    assert predict['new_prior'].shape == (2, width)
    assert predict['new_pages'].shape == (2, pages)
    assert predict['new_prior'][0].tolist() == [-1.] * (width - 1) + [.25]
    assert predict['raw_id'].tolist() == [2**53 + 1, 2**53 + 2]


def test_lc_mge_run_reloads_configuration_and_predicts_without_labels(tmp_path):
    from workspace.kr_time_budget.pipeline.predict_lc_mge import export_predictions, load_reranker, prediction_batches
    from workspace.kr_time_budget.pipeline.run_architecture_examples import train_example

    data = tmp_path / "data"
    data.mkdir()
    raw = group_slates(raw_events().lazy()).collect()
    manager = rerank_manager(data / "schema")
    manager.fit(raw.lazy())
    encoded = manager.transform(raw.lazy()).collect()
    for name in ("train", "valid.ordered", "test.ordered"):
        encoded.write_parquet(data / f"{name}.parquet")
    (data / "manifest.json").write_text(json.dumps({
        "splits": {split: {"requests": 4} for split in ("train", "valid", "test")}}))
    args = SimpleNamespace(seed=2026, top_k=2, mode="lc", candidate_dense_mode="generator",
                           point_batch_size=8, rerank_batch_size=2, device="cpu", threads=1, limit_requests=4)
    run = tmp_path / "run"
    trainer, batches, result = train_example("rerank", data, run, args)
    loaded = load_reranker(run)
    assert loaded.candidate_dense_mode == "generator" and loaded.submodularity_mode == "lc"
    assert loaded.loss_reduction == "nonzero_mean"
    assert result["status"] == "SMOKE_PARTIAL" and result["optimizer_steps"] == 2
    batch = next(iter(batches["test"]))
    with torch.no_grad():
        expected = trainer.model.predict(batch)
        restored = loaded.predict(batch)
    torch.testing.assert_close(restored.scores, expected.scores, rtol=0, atol=0)
    torch.testing.assert_close(restored.order, expected.order, rtol=0, atol=0)
    for key in expected.extra:
        torch.testing.assert_close(restored.extra[key], expected.extra[key], rtol=0, atol=0)
    unlabeled = data / "unlabeled.parquet"
    encoded.drop("is_click", "long_view", "prior_ctr").write_parquet(unlabeled)
    inferred = next(prediction_batches(unlabeled, loaded.manager, 2))
    assert set(inferred.columns) == set(batch.columns) - set(manager.label_fields)
    for field in inferred.columns:
        torch.testing.assert_close(inferred[field], batch[field], rtol=0, atol=0)
    output = tmp_path / "predictions.parquet"
    assert export_predictions(loaded, unlabeled, output, batch_size=2) == 4
    frame = pl.read_parquet(output)
    torch.testing.assert_close(torch.tensor(frame["scores"].to_list()[:2]), expected.scores)
    assert frame["out_index"].to_list()[:2] == expected.order.tolist()
    with pytest.raises(FileExistsError):
        export_predictions(loaded, unlabeled, output)


def test_custom_order_evaluator_measures_selected_order_and_prior_separately():
    labels = torch.tensor([[1., 0., 0., 0., 0., 0.]])
    batch = Interaction({"prior_long_view": torch.tensor([[1., 0., 0., 0., 0., 0.]])})
    predictions = Prediction(order=torch.tensor([[5, 4, 3, 2, 1, 0]]))
    from rosen.evaluate import FieldScorer, RankingEvaluator
    selected, prior = RankingEvaluator("ndcg@6"), RankingEvaluator("ndcg@6")
    batch["label"] = labels
    context = EvalContext(predictions, batch, "label")
    selected.collect(BatchResolver().resolve(context, mask=torch.ones_like(labels)))
    prior.collect(BatchResolver(pred_scorer=FieldScorer("prior_long_view"), view="score_only").resolve(context, mask=torch.ones_like(labels)))
    assert selected.summary()["ndcg@6"] == pytest.approx(1 / math.log2(7))
    assert prior.summary()["ndcg@6"] == 1


def test_custom_candidate_evaluator_transports_mask_with_each_order():
    evaluator = SlateValueEvaluator("cand_mean", "cand_max", reduction="mean")
    # Slot zero is masked. Its high score moves to different candidate positions.
    pred = Prediction(candidates=CandidatePool(scores=torch.tensor([[[99., .2, .4], [.4, 99., .2]]]),
                                               orders=torch.tensor([[[0, 1, 2], [2, 0, 1]]])))
    context = EvalContext(pred, Interaction({"label": torch.zeros(1, 3)}), "label")
    evaluator.collect(BatchResolver.for_candidates().resolve(context, mask=torch.tensor([[0., 1., 1.]])))
    assert evaluator.summary() == pytest.approx({"cand_mean": .3, "cand_max": .3})


@pytest.mark.parametrize('outputs_pool', [False, True])
def test_candidate_evaluator_registration_uses_model_capability_for_new_registry_entries(monkeypatch, outputs_pool):
    from rosen.models.rerank import RERANK_MODEL_REGISTRY, RerankModel

    class NewReranker(RerankModel):
        outputs_candidate_pool = outputs_pool

    monkeypatch.setitem(RERANK_MODEL_REGISTRY, 'new_external_model', NewReranker)
    registry = evaluators('rerank', rerank_model='NEW_EXTERNAL_MODEL').registry
    assert ('candidates' in registry) == outputs_pool
````

## File: workspace/kr_time_budget/回复/新架构示例/Prediction迁移.md
````markdown
# Prediction / EvalBatch 接口迁移验收

本页保留早前字典迁移的历史验收口径。当前 API 以 [用法](用法.md) 为准：主候选为
CandidatePool、选中得分为 chosen_scores，旧 extra 主键和旧评估包装器均已删除；下文旧接口描述只用于还原当时验收。

本次替换预测与评估接口，未重新训练，也未修改特征、监督、参数结构或训练损失。
复用 [原始完整训练](实验报告.md) 的两个最佳 checkpoint，在同一时间切分的完整 test
上运行新接口，逐条对比原先导出的分数和最终排列。

| 任务 | test 规模 | 核对结果 |
| --- | ---: | --- |
| Pointwise CTR + 长播 | 293,388 曝光 | 两个预测 head 均逐值完全相等（rtol=atol=0） |
| MultiGE 长播重排 | 48,898 屏 / 293,388 视频 | 原列表分数、最终排列均逐值完全相等 |

所有原有 test 指标差值小于 1e-12：CTR AUC 0.749291，Pointwise 长播 AUC 0.756063，
重排长播 AUC 0.755150，最终排列 NDCG@6 0.470777。冻结先验排序 NDCG@6 仍为
0.486740，没有新增排序收益。候选均分现在归独立 `candidates` 评估组，名称为
`topk_mean / best_topk_mean`；不作为真实反馈收益。

数据沿用原 v3 六视频分屏：按会话结束日期 train≤2022-05-06、valid=05-07、test=05-08。
原 checkpoint 来自完整训练集各 1 epoch、独立 valid 选择；本次只复测完整 test，FP32/CPU/4线程，
无选参、无训练。此结论是接口数值保持，不是新增模型的结构因果实验。

## 当时的代码变化（历史）

- 模型返回 Prediction：多任务 scores 的 key 就是标签名；MultiGE 声明 target=long_view，
  scores 原槽位 `[B,6]`、order `[B,6]`，候选分数及索引放 extra。
- Manager 接入 schema，register 只需 target，自动解析 label/mask，全局声明 raw UID。
- 内置 OrderEvaluator 计算最终排列 NDCG；SlateValueEvaluator 只保留有监督指标。
- PriorEvaluator / MGECandidateEvaluator 留在实验层，统一 `collect(EvalBatch)`。
- Trainer/EMA 在实际预测返回后包装旧单任务 Tensor，不改写显式 Prediction 的目标。

代码与用法见 [用法.md](用法.md)。旧 `pred_key/label_field/mask_field` 注册、自定义
collect_batch 与直接四参数 collect 需要迁移；旧对象可显式用 LegacyEvaluatorAdapter 包装。
旧单任务模型 Tensor 输出及原始 KR1K 的 register 写法继续可用。

## 验证证据

- [完整 test 复测脚本](verify_prediction_contract.py)
- [逐条对照日志](../../logs/prediction_contract_audit.log)
- [复测指标、checkpoint 与源码哈希](../../runs/prediction_contract_20261117/audit.json)
- [原完整训练 run](../../runs/architecture_full_20261117/summary.json)
- [原入口源码快照](../../runs/architecture_full_20261117/source_snapshot/architecture_examples.py)
- 框架及 workspace 全量回归：303 passed；日志位于仓库 `logs/prediction_contract_regression.log`。

原 run 的训练日志、指标、checkpoint 与预测文件均保留。原 summary 中三份入口的 SHA-256
可对照 `source_snapshot/` 校验；当前入口已迁移，不能再把当前文件当成历史运行时的源码。

复测命令（默认输出目录已存在时，请指定新的 `--output`）：

```bash
uv run python -m workspace.kr_time_budget.回复.新架构示例.verify_prediction_contract \
  --output workspace/kr_time_budget/runs/prediction_contract_recheck
```
````

## File: workspace/kr_time_budget/回复/新架构示例/verify_prediction_contract.py
````python
"""Audit the new output contract against both frozen full-run test predictions.

No training or fitting. Historical results and source snapshots remain untouched.
Run from the repository root:
uv run python -m workspace.kr_time_budget.回复.新架构示例.verify_prediction_contract
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import polars as pl
import torch

from workspace.kr_time_budget.pipeline.architecture_examples import (
    KRPointwiseMultiTaskModel, evaluators, loader, make_reranker, pointwise_manager, rerank_manager,
)

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify(run, output):
    summary = json.loads((run / 'summary.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    result = {'training': 'none; reuse original full-train 1-epoch best checkpoints',
              'source_run': str(run.resolve()), 'tasks': {}}
    for task in ('pointwise', 'rerank'):
        old = summary[task]
        checkpoint = Path(old['checkpoint'])
        assert sha(checkpoint) == old['checkpoint_sha256']
        manager = (pointwise_manager if task == 'pointwise' else rerank_manager)(Path(old['schema']))
        manager.load_schema()
        model = KRPointwiseMultiTaskModel(manager) if task == 'pointwise' else make_reranker(manager, top_k=old['top_k'], mode=old['mode'])
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        del state
        gc.collect()
        model.eval()
        reference = pl.read_parquet(run / ('priors/test.parquet' if task == 'pointwise' else 'rerank/test_predictions.parquet'))
        evaluation = evaluators(task, output / f'{task}.log', manager=manager)
        count, started = 0, time.monotonic()
        with torch.no_grad():
            for inter in loader(Path(old['data']) / 'test.ordered.parquet', manager, old['physical_batch']):
                pred = model.predict(inter)
                expected = reference.slice(count, len(inter))
                id_field = 'event_id' if task == 'pointwise' else 'request_id'
                assert inter[id_field].tolist() == expected[id_field].to_list()
                if task == 'pointwise':
                    for label, prior in (('is_click', 'prior_ctr'), ('long_view', 'prior_long_view')):
                        torch.testing.assert_close(pred.scores[label], torch.tensor(expected[prior].to_list()), rtol=0, atol=0)
                else:
                    torch.testing.assert_close(pred.scores, torch.tensor(expected['scores'].to_list()), rtol=0, atol=0)
                    torch.testing.assert_close(pred.order, torch.tensor(expected['out_index'].to_list()), rtol=0, atol=0)
                evaluation.collect(inter, pred)
                count += len(inter)
        assert count == old['split_counts']['test'] == reference.height
        metrics = evaluation.summary()
        for group, values in old['test'].items():
            for metric, expected in values.items():
                if group == 'slate' and metric in ('topk_ctr', 'best_topk_ctr'):
                    actual = metrics['candidates']['topk_mean' if metric == 'topk_ctr' else 'best_topk_mean']
                else:
                    actual = metrics[group][metric]
                assert abs(actual - expected) < 1e-12, (task, group, metric, actual, expected)
        result['tasks'][task] = dict(rows=count, exact_prediction_match=True,
                                    metrics_match_tolerance=1e-12, metrics=metrics, seconds=time.monotonic()-started,
                                    checkpoint_sha256=old['checkpoint_sha256'])
        print(task, count, 'exact predictions; all metrics match', flush=True)
        del model, manager, evaluation, reference
        gc.collect()
    paths = [Path('src/rosen/core/prediction.py'), *Path('src/rosen/evaluate').glob('*.py'),
             Path('src/rosen/models/rerank/mge.py'), Path('workspace/kr_time_budget/pipeline/architecture_examples.py')]
    result['source_hashes'] = {str(p): sha(p) for p in paths}
    (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('PASS', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=WORKSPACE/'runs/architecture_full_20261117')
    parser.add_argument('--output', type=Path, default=WORKSPACE/'runs/prediction_contract_20261117')
    args = parser.parse_args()
    verify(args.run, args.output)
````

## File: workspace/kr_time_budget/回复/新架构示例/实验报告.md
````markdown
# KR1K 新架构示例：双标签 Pointwise 与六视频重排

本报告记录原始接口的完整训练。当前源码已迁移 Prediction/EvalBatch，见 [接口迁移复测](Prediction迁移.md)；历史入口源码保存在该 run 的 source_snapshot/，原始结果不覆盖。

两个示例均完成完整训练集 1 轮、完整 valid 和完整 test；没有行数截断或训练步上限。
本次是新架构用法与真实数据闭环验收，不替换历史预算实验，也不构成模型结构的配对因果对照。

## 完整 test 指标

| 任务 | AUC | User GAUC | LogLoss |
| --- | ---: | ---: | ---: |
| Pointwise CTR | 0.749291 | 0.602066 | 0.566823 |
| Pointwise 长播 | 0.756063 | 0.625234 | 0.487465 |
| MultiGE 长播 | 0.755150 | 0.638018 | 0.495800 |

| 长播列表口径 | NDCG@6 |
| --- | ---: |
| 冻结 Pointwise 先验排序 | 0.486740 |
| OCPM 对原列表打分后排序 | 0.486105 |
| 最终生成器选定排列 | 0.470777 |

最终排列相对冻结先验的 NDCG@6 差为 **-0.015963**。本轮没有取得重排排序收益，不能把训练闭环跑通等同于重排有效。

最终选定排列的 NDCG 由导出的 48,898 屏标签与 out_index 独立复算一致。其中 31,744 屏至少有一个长播正例；无正例屏按 0 计入全量平均。

候选打分均值 topk_ctr=0.249077，best_topk_ctr=0.256410，CB=0.971038。这里沿用 MultiGE 历史键名 ctr_*，实际监督和概率语义均为 long_view，这些估计均分不是真实重排点击/长播收益。

## 数据与切分

复用旧预算实验 data/v3 的去重事件、request_events 和 requests_base。仍为每用户会话倒切 6 视频，保留原先的会话筛选与 top-5 场景，不重新选择人群或分屏。

| 切分 | 日期口径 | Pointwise 曝光数 | 重排屏数 |
| --- | --- | ---: | ---: |
| train | session end ≤ 2022-05-06 | 10,330,470 | 1,721,745 |
| valid | 2022-05-07 单日会话 | 347,232 | 57,872 |
| test | 2022-05-08 单日会话 | 293,388 | 48,898 |

逐视频 Label 为原始 is_click 与 long_view，不从播放时长另设阈值，也不使用 y_continue。用户/视频/作者/tab 的词表和视频长度归一化仅 fit train。X 不包含当前反馈、观看时长、未来长度或继续标签。

历史只取同一 session 内之前两屏完整 6 视频，按从旧到新拍扁为 12 个 item，有效页靠左打包，page_mask 对齐；会话头部不足 6 个视频的余量不纳入本例历史。这是相对旧预算模型 hist_tags 的显式变化，非旧模型数值等价迁移。

Pointwise 训练完成后冻结 checkpoint，对 train/valid/test 生成先验。train 先验为同训练集模型的in-sample 预测，未做 OOF；valid/test 完全使用冻结模型，不再拟合。原数据三个 parquet 的大小和 mtime 在报告审计时保持不变。

## 新架构怎样接起来

1. Pointwise：BaseModel + 字符串多标签，shared-bottom 两个输出 head；calculate_loss 为两个 BCE 的和，LossPort 分别登记；predict 返回 ctr/long_view 字典，register 用 pred_key/label_field 显式配对。
2. 重排：两个 SlateLabel 共用 mask，四个属性的当前序列/历史序列共享四张原子词表；stack 得到 [B,6,4,8]，模型折叠历史为 [B,2,6,4,8]。使用公共 MultiGERerankModel，label_field 显式选 long_view，即使 manager.label_field 是 is_click 也不依赖 Trainer 拆标签。
3. 标准 Evaluator 接收 mask 展平计算曝光 AUC，SlateValueEvaluator 计算列表指标；OrderEvaluator 实现 collect_batch，直接读取 out_index 计算最终排列 NDCG，模型不写 eval 胶水。

## 训练协议

| 项目 | Pointwise | MultiGE 重排 |
| --- | --- | --- |
| Physical batch | 8192 曝光 | 512 屏（最多 3072 曝光） |
| Optimizer steps | 1262 | 3363 |
| 优化器 | AdamW，lr=0.001 | AdamW，lr=0.001 |
| Weight decay | 普通权重 1e-6，Embedding/bias/norm/vector 为 0 | 相同分组规则 |
| 精度/设备 | FP32 / cpu / 4 线程 | FP32 / cpu / 4 线程 |
| Seed / accumulation / EMA / scheduler | 2026 / 1 / 关闭 / 无 | 2026 / 1 / 关闭 / 无 |
| 训练及完整 valid 用时 | 75.7s | 482.2s |
| 完整 test 用时 | 1.3s | 12.5s |

训练 parquet 使用 seed=2026 的物理 target-sort shuffle，drop_last=False；读取侧不再 shuffle。两者各完整 1 epoch 后由完整 valid long_view.auc 选择唯一 checkpoint，然后加载该 checkpoint 测完整 test。没有 warm-start，重排只把前一模型预测作为冻结输入；测试集不参与选参。

重排 submodularity_mode=lc，hidden=32，MLP_A=(32,16)，MLP_B=(32,16)，top_k=4。全部 720 个排列仍参与粗筛，但 eval_all_perms=False，不导出全排列 OCPM 分数。LC 为离散无梯度统计项，可导致账本总损失为负；不据此判断训练崩溃。BCE 使用 valid_mean；原 TF 的初始化/GRU/独立 top-k head 等差异见框架 RERANK 文档。

两种模型的输入、任务与更新次数不同，上表不支持把 AUC 差归因于某个单一结构。六视频分屏仍是依赖整场长度的离线代理，不是已知同时曝光请求；当前标签仅评价日志内排序，没有反事实曝光校正或在线收益证据。

## 入口与证据

```bash
uv run python -m workspace.kr_time_budget.run_architecture_examples
```

默认完整数据、两任务各 1 epoch；输出创建到独立时间戳目录。已存在的 run 不覆盖。小规模诊断须显式提供 --limit-requests 和独立 --data，结果标为 SMOKE_PARTIAL。

- [模型、Schema 与评估声明](../../pipeline/architecture_examples.py)
- [数据准备与历史对齐](../../pipeline/prepare_architecture_examples.py)
- [训练与导出入口](../../pipeline/run_architecture_examples.py)
- [专项测试](../../tests/test_architecture_examples.py)
- [本报告生成器](write_report.py) · [验收数据](验收数据.json)
- [完整训练日志](../../runs/architecture_full_20261117/logs/train.log)
- [总结果与配置](../../runs/architecture_full_20261117/summary.json)
- [Pointwise 指标](../../runs/architecture_full_20261117/pointwise/metrics.json) · [重排指标](../../runs/architecture_full_20261117/rerank/metrics.json)
- [完整 test 重排预测](../../runs/architecture_full_20261117/rerank/test_predictions.parquet)

复算报告命令：

```bash
uv run python 'workspace/kr_time_budget/回复/新架构示例/write_report.py' --run '/Users/lxogri/Developer/pyprojects/time-budget/workspace/kr_time_budget/runs/architecture_full_20261117'
```
````

## File: workspace/kr_time_budget/回复/新架构示例/用法.md
````markdown
当前接口与历史结果边界见 [2026-11-19 审计修复](审计修复.md)。本页用法已同步类型化候选池。

# 从两个 KR1K 例子看 Prediction / EvalBatch

[模型与评估源码](../../pipeline/architecture_examples.py) ·
[准备数据](../../pipeline/prepare_architecture_examples.py) ·
[完整训练入口](../../run_architecture_examples.py) ·
[接口迁移复测](Prediction迁移.md)

特征归 EmbSetting，监督归 SchemaManager；模型返回 Prediction，评估器收集 EvalBatch。
修改特征或编码规则时，为 `--data` 选择新目录，避免复用旧 schema。

## 1. Pointwise：同一曝光的 CTR 与长播

```python
manager = SchemaManager(
    [*categorical_settings, MinMaxDenseSetting("duration_s", FeatureSource.ITEM)],
    schema_dir=point_schema,
    label_fields=["is_click", "long_view"],
    aux_fields=[AuxField("event_id", dtype="int64"), AuxField("request_id", dtype="int64"),
                AuxField("slot", dtype="int64"), AuxField("user_id_raw", dtype="int64")],
)
model = KRPointwiseMultiTaskModel(manager)
# predict 返回 Prediction(scores={"is_click": Tensor[B], "long_view": Tensor[B]})
# 两项 loss 仍是独立 BCE，通过 LossPort 分别登记。

evaluation = EvaluatorManager(manager, uid_field="user_id_raw")
evaluation.register("ctr", PointwiseEvaluator("auc", "logloss", "gauc"), target="is_click")
evaluation.register("long_view", PointwiseEvaluator("auc", "logloss", "gauc"), target="long_view")
```

预测 Mapping 的 key 与标签同名，一个 target 完成配对。`user_id_raw` 全局声明一次，避免
OOV 编码将不同用户合并成 GAUC 同组。多标签没有默认 target 时，框架报歧义错误。

## 2. 重排：六视频一屏、前两屏历史

```python
current = SeqGroupConfig("slate", "slate_len", max_len=6)
history = SeqGroupConfig("history", "history_len", max_len=12)
settings = [SequenceSetting(f"cur_{s.field_name}", s, current) for s in shared_atoms]
settings += [SequenceSetting(f"hist_{s.field_name}", s, history) for s in shared_atoms]
manager = SchemaManager(settings, slate_schema, label_fields=[
    SlateLabel("is_click", 6, "mask"), SlateLabel("long_view", 6, "mask"),
], aux_fields=[AuxField("prior_long_view", max_len=6), AuxField("page_mask", max_len=2),
               AuxField("request_id", dtype="int64"), AuxField("user_id_raw", dtype="int64")])
model = MultiGERerankModel(
    manager, slate_cate_group="slate", hist_cate_group="history",
    slate_dense_field="prior_long_view", page_mask_field="page_mask",
    label_field="long_view", poi_num=6, page_num=2, submodularity_mode="lc",
    top_k=4, eval_all_perms=False, mlp_a=(32, 16), mlp_b=(32, 16), hidden_dim=32,
    loss_reduction="valid_mean",
)
# predict 返回 Prediction(scores=..., order=..., chosen_scores=..., candidates=..., target="long_view", extra=...)
evaluation = EvaluatorManager(manager, uid_field="user_id_raw")
evaluation.register("long_view", PointwiseEvaluator("auc", "logloss", "gauc"))
evaluation.register("slate", RankingEvaluator("ndcg@6", "pcoc"), resolver=BatchResolver(view="score_only"))
evaluation.register("chosen_order", RankingEvaluator("ndcg@6"))
evaluation.register("pointwise_prior", RankingEvaluator("ndcg@6"),
                    resolver=BatchResolver(pred_scorer=FieldScorer("prior_long_view"), view="score_only"))
if model.outputs_candidate_pool:
    evaluation.register("candidates", SlateValueEvaluator("cand_mean", "cand_max", reduction="mean"),
                        resolver=BatchResolver.for_candidates())
```

当前和历史引用同一批原子对象，共享词表及 embedding；历史存储仍拍扁，进入模型再折屏。
模型明确选择 long_view，即使 manager 首个标签是 is_click，评估也从 Prediction.target
继承长播，并自动查询对应 SlateLabel.mask_field。

| 内容 | 形状 |
| --- | --- |
| 当前分类属性 stack | `[B,6,4,8]` |
| 历史 stack 后折屏 | `[B,2,6,4,8]` |
| 标签 / mask / 冻结先验 | 各 `[B,6]` |
| 历史页 page_mask | `[B,2]` |
| pred.scores | 原始槽位长播概率 `[B,6]` |
| pred.order | 输出位置 → 输入槽位 `[B,6]` |
| pred.chosen_scores | 与最终 order 对齐的选中得分 `[B,6]` |
| pred.candidates.scores / orders | 各 `[B,4,6]`，类型化主候选分数及对应排列 |

RankingEvaluator 通过 score_only 评价原序分数排序，默认视图评价最终生成排列。
SlateValueEvaluator 评价已对齐整刷或主候选的价值，输入 EvalBatch 只有张量。
自定义来源继承 BaseScorer，在 EvalContext 中读取所需原始字段，再由 Resolver 投射。
候选指标将 mask 按每个候选排列搬到对应坐标后求均值，属于模型估计分，不是真实收益。
导出调用 pred.resolve_chosen_scores()，候选池缺显式选中得分直接报错；子类诊断使用 pred.with_extra()。

## 运行与验证

```bash
uv run python -m workspace.kr_time_budget.run_architecture_examples
```

默认两个任务各完整训练 1 epoch，valid 选 checkpoint，重载后测完整 test。
[原始实验报告](实验报告.md) 保留旧接口运行时的证据；[Prediction 迁移报告](Prediction迁移.md)
复用同一 checkpoint 测完整 test，核对分数、排列和指标。历史预算实验与原始 KR1K 示例不改写。
本次类型化验证为 848 项测试及 11 类候选模型同权重合成 batch 的逐值对照，未重跑完整数据集实验。
````

## File: workspace/kr_time_budget/run_architecture_examples.py
````python
"""Top-level entry: uv run python -m workspace.kr_time_budget.run_architecture_examples."""
from workspace.kr_time_budget.pipeline.run_architecture_examples import main

if __name__ == "__main__":
    main()
````
