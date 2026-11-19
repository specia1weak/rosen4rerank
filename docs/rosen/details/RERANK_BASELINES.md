# LC_MGE 参考目录的独立重排模型

参考根目录为 `references/rerank/参考代码/LC_MGE_code/rerank_model_dir/`。
以下按其中 15 个 `model.py` 目录列出构建入口；预设复用同构网络，有独立公式/训练/候选
行为的模型提供独立类。保留既有 `mge` 默认行为，不改变 MSR 或通用 Trainer。

## 源码对应

| 参考目录 | 构建名 | rosen 实现及行为 |
| --- | --- | --- |
| `DNN_model` | `dnn` | [DNN](../../../src/rosen/models/rerank/dnn.py)：逐 item SiLU MLP_A，拼先验，MLP_B 输出 |
| `DCN_model` | `dcn` | [DCN](../../../src/rosen/models/rerank/dcn.py)：向量 cross recurrence，拼每层状态和原输入后打分 |
| `prm_model` | `prm` | [PRM](../../../src/rosen/models/rerank/prm.py)：逐 item 编码、跨 item 单头 QKV attention、共享输出 head |
| `extr_model` | `extr` | [EXTR](../../../src/rosen/models/rerank/extr.py)：自然项 self-attention、广告项 causal target-attention、列表 head |
| `kuaishou_model` | `kuaishou` | [KuaiShou](../../../src/rosen/models/rerank/kuaishou.py)：target 对当前列表/历史列表残差池化 |
| `pier_model` | `pier` | [PIER](../../../src/rosen/models/rerank/pier.py)：PPML/SimHash 检索，OCPM 从 top-k 选排列 |
| `prs_model` | `prs` | [PRS](../../../src/rosen/models/rerank/prs.py)：首个曝光率 beam 候选的监督，保留 PIER 候选选择 |
| `NAR_model` | `nar` | [NAR](../../../src/rosen/models/rerank/nar.py)：训练三个辅助生成器，仅采用 Listwise 生成候选 |
| `seq2slate_model` | `seq2slate` | [Seq2Slate](../../../src/rosen/models/rerank/seq2slate.py)：相同辅助训练，仅采用 GRU-pointer 候选 |
| `pier_model_submodularity_metric` | `pier_submodular` | [PIER 多样性](../../../src/rosen/models/rerank/pier_submodular.py)：追加距离 top-k embedding 均值最远的未选排列 |
| `MultiG_E_model` | `mge3` | [MultiGE](../../../src/rosen/models/rerank/mge.py)：三生成器，无第四分支 |
| `MultiG_E_Submodularity_Model` | `mge_cosine` | 同一 MultiGE 实现，第四生成器使用 cosine coverage reward |
| `MultiG_E_Submodularity_Model_lc` | `lc_mge` | 同一 MultiGE 实现，第四生成器使用 teacher loss，离散 LC 单独记录诊断 |
| `MultiG_E_model_metric` | `mge_metric` | [MultiGE metric](../../../src/rosen/models/rerank/mge_metric.py)：监督最佳生成候选，AR loss 对 B 求均值，输出多样性统计 |
| `MultiG_E_Submodularity_model_metric` | `mge_cosine_metric` | 同文件 SubmodularMultiGEMetricRerankModel，第四 cosine 分支，AR loss 对 B 求均值 |

## 构建与数据

```python
from rosen.models.rerank import build_rerank_model

model = build_rerank_model(
    manager, "prm",  # 替换为上表构建名
    slate_cate_group="slate", slate_dense_field="prior",
    hist_cate_group="history", page_mask_field="page_mask",
    label_field="label", poi_num=5, page_num=5,
)
```

当前分类属性为共享原子 setting 的 K 长度 SequenceSetting，历史属性为 P*K 长度
SequenceSetting；折叠在模型内完成。`prior` 为 `[B,K]` / `[B,K,1]`；标签为 SlateLabel，
可关联有效槽 mask `[B,K]`。`page_mask` 为 `[B,P]`，由 schema 或 Dataset extras 显式读取。
所有分数为输入槽位坐标；`order` 将输出位置映射到输入槽位。无需新增 Trainer 协议。

DNN/DCN/PRM/EXTR 只需要当前列表，可省略所有历史参数；KuaiShou、PIER 和生成器模型
要求历史视图，缺失时提前报错。DNN/DCN/PRM/EXTR/KuaiShou 不枚举排列，不受 K<=7 限制。
PIER/MultiGE/NAR/Seq2Slate 的粗筛仍枚举 K!，保留 K<=7；关闭 `eval_all_perms` 只关闭
全部排列的 OCPM 输出。工厂沿 MRO 过滤构造关键字，未知参数告警。

PIER/MultiGE 及其子类声明 outputs_candidate_pool=True，主候选输出为 CandidatePool，
选中分数为 Prediction.chosen_scores；其他直接打分模型通过 resolve_chosen_scores() 按 order gather。
使用 BatchResolver.for_candidates() 注册主候选评价，诊断统计通过 with_extra() 扩展。

EXTR 默认自然项为偶数槽 `0,2,4,...`、广告项为奇数槽；可用 `natural_indices` 与
`ad_indices` 指定，二者必须非空且恰好划分所有槽位。属性展平宽度须整除 `num_heads`
（默认 4）。Attention 采用 QKV、residual+LayerNorm、residual FFN，FFN 后无额外 LN，
没有 PyTorch MultiheadAttention 的额外输出投影。attention/FFN dropout 默认 .1/.2。

KuaiShou 对每个当前 item 作为 target，分别读取当前列表与 P*K 历史上下文；P=1 对应
参考实现的一组 K 历史 item。历史先验可由 `hist_dense_field` 指定 `[B,P*K]` /
`[B,P*K,1]`；未指定则明确使用零历史先验。当前列表作为 order context，历史按页 mask
过滤；与参考代码“单 target + 外部 order 列表”的输入组织不同，便于统一列表契约。

## 公式与有意修正

- DCN 使用 `x_next = x0*(x_l @ w_l) + b_l + x_l`。保留参考 `y_cross` 所有层状态与
  原输入再次拼接的行为；修正原始 outer product 后错误 reshape 成原宽度的问题。
- PRM 修复 `K*width+1` 与 `K*(width+1)` 混淆，保持 `[B,K,W]` 给共享 head。
- PRM/EXTR/KuaiShou 的无效 key 在 softmax 前屏蔽；residual 后重新施加 query/context
  mask，全无效 context 严格返回零 attention 权重，不让 padding 影响其他有效 item。
- PIER 原列表与所有候选复用同一个已受监督 OCPM 输出层；按 sample-major 历史对齐，
  修复 TF 候选 head 未受监督、tile 历史错位的问题。默认对比权重 .01，可关闭。
- PRS 用 beam 候选的**原始 logits**监督，并将标签/mask 随排列移动；不复制 TF 的
  “概率再次作为 logits”及标签仍处于旧坐标的问题。
- PIER 多样性从未被 top-k 选中的排列中，追加与 top-k embedding 均值的平方 L2 距离
  最大者；对比项同时使用扩展后的 top-k。若 top-k 已覆盖全排列，则不再追加。
  最终索引从扩展候选取，修复参考代码选到新增项后仍从原 top-k 索引表 gather 的越界。
- NAR/Seq2Slate 不是另写一种名为 Transformer 的架构。参考源码两者只有候选选择的
  一行差别，均训练 Pointwise、Listwise、GRU 三辅助项和 evaluator；本实现保留这个结构，
  修复后 AR loss 按步骤求和、每步使用 batch mean，避免大 batch 与尾批量纲变化。推理始终 greedy，输出合法排列。
- 两个 MultiGE metric 版本确实改变训练目标，不能只当作日志别名。其 evaluator 监督
  最佳生成候选，用原始 logits、随候选移动的标签/mask，并使用合法 greedy 候选替代
  训练期可能重复的 teacher argmax 记录。两个版本统一使用 AR batch mean，均保留
  辅助 BCE 的监督 mask，不复制 Cosine metric 源码漏 mask 的问题。
- metric 输出 `candidate_mean_l2` / `candidate_max_l2`、平均/最大 cosine distance，
  只统计不同生成列表对；`candidate_lc_index` 精确使用现有 LC 原索引关系矩阵定义。
  不复制参考按排序后的 item 值计算的伪 Kendall 计数，也不假装它与上述 LC 等价。

所有新线性层按二维 Xavier 初始化；cross weight、QKV、pool、GRU 都没有额外隐藏的
三维参数轴。新模型不插入既有 MultiGE 的构造顺序。这里保留结构意图与可追踪修正，
不宣称 TF 权重、GRU 门控、初始化或完整论文效果等价。

## 可执行验证与调用方入口

KR 入口新增 `--rerank-model`，覆盖上表全部名字；例：

```bash
uv run python -m workspace.kr_time_budget.run_architecture_examples \
  --rerank-model pier --limit-requests 32 \
  --data workspace/kr_time_budget/data/pier_smoke_new \
  --output workspace/kr_time_budget/runs/pier_smoke_new
```

去掉 `--limit-requests` 可运行完整数据；同一入口依次完成冻结先验、列表数据、训练、valid
选 checkpoint、完整 test 和预测导出。模型构造配置随 run 保存；原独立
`workspace.kr_time_budget.predict_lc_mge` 入口也可从这些配置恢复其他重排模型。

每个独立类均有原生 Trainer 真实更新、loss 账本、checkpoint 与无标签预测验证；另有
cross 手算、permutation equivariance、跨 item 梯度、EXTR residual/causal/padding、
KuaiShou target pooling、历史隔离、PIER 逐排列评分、PRS 监督坐标、生成器选择、多样性
追加与 metric 监督公式对照。见 [测试](../../../tests/rosen/models/rerank/test_reference_models.py)。
15 个入口的真实 KR 小批验收见
[独立模型补全](../../../workspace/kr_time_budget/回复/新架构示例/独立模型补全.md)。

2026-11-19 审计修复：pointwise 统一原槽位坐标，生成 dense 特征随候选重排；metric 主监督
恢复当前 embedding 梯度，移除被覆盖的原序 evaluator 前向；PIER/PRS/PIERSubmodular 明确拒绝
K<N 的裁判输入。旧训练结果不代表这些修复后的模型；参见 [审计记录](../request_execution/changes/20261119_rerank-review-fixes.md)。
