# 重排模型

通过 `build_rerank_model(manager, name, **kwargs)` 构建；所有模型均实现原生
`calculate_loss` / `predict`，返回 `Prediction`。原序分数在 scores，最终排列在 order；
候选池模型将主候选放在 CandidatePool(scores, orders)，选中得分放在 chosen_scores，extra 只放诊断。
测试在仓库 `tests/rosen/models/rerank/`。

| 入口 | 模型类 | 源码 |
| --- | --- | --- |
| `dnn` | DNNRerankModel | [dnn.py](dnn.py) |
| `dcn` | DCNRerankModel | [dcn.py](dcn.py) |
| `prm` | PRMRerankModel | [prm.py](prm.py) |
| `extr` | EXTRRerankModel | [extr.py](extr.py) |
| `kuaishou` | KuaiShouRerankModel | [kuaishou.py](kuaishou.py) |
| `pier` | PIERRerankModel | [pier.py](pier.py) |
| `prs` | PRSRerankModel | [prs.py](prs.py) |
| `nar` | NARRerankModel | [nar.py](nar.py) |
| `seq2slate` | Seq2SlateRerankModel | [seq2slate.py](seq2slate.py) |
| `pier_submodular` | PIERSubmodularRerankModel | [pier_submodular.py](pier_submodular.py) |
| `mge_metric` | MultiGEMetricRerankModel | [mge_metric.py](mge_metric.py) |
| `mge_cosine_metric` | SubmodularMultiGEMetricRerankModel | [mge_metric.py](mge_metric.py) |
| `mge3` / `mge_cosine` / `lc_mge` | MultiGERerankModel 的三/四生成器预设 | [mge.py](mge.py) |
| `mge` | MultiGERerankModel，保留既有默认值 | [mge.py](mge.py) |

独立模型共享 [slate.py](slate.py) 的 schema、输入和监督契约；PIER 与 MultiGE 共享 OAM、
PPML/SimHash、生成器等算子。PRS 改变 beam 候选监督，NAR/Seq2Slate 改变生成器选择，
metric 版本改变候选监督与诊断；这些都具有真实实现行为。

全部内置模型共用 SlateRerankModel.score_permutations，先编码当前物品和固定上下文，
再按 candidate_chunk_size 分块评分。候选 mask 是显式参数，不向 Interaction 注入控制键。
OCPMRerankModel 为 PIER/MultiGE 共享预编码历史页；KuaiShou 共享历史物品及 K/V。
新模型实现 _encode_items/_encode_context/_score_encoded，不能覆写 logits 或 score_permutations。

```python
from rosen.models.rerank import build_rerank_model

model = build_rerank_model(
    manager, "pier",
    slate_cate_group="slate", hist_cate_group="history",
    slate_dense_field="prior", page_mask_field="page_mask",
    label_field="label", poi_num=5, page_num=5,
    top_k=20, eval_all_perms=False,
)
```

数据声明、参考源码逐目录对应、公式与迁移修正见
[独立模型说明](../../../../docs/rosen/details/RERANK_BASELINES.md)；
MultiGE 完整契约见 [RERANK](../../../../docs/rosen/details/RERANK.md)。

MultiGE 族现统一继承 SlateRerankModel；输入、mask、loss 收集与其他模型共用，
候选评分统一使用 components/scoring.py。outputs_candidate_pool 声明是否输出主候选池；
通过 BatchResolver.for_candidates() 评价它，resolve_chosen_scores() 获取与最终 order 对齐的分数。
eval_diagnostic_pools=False 仅保留主候选池，
省去 lastk/randk/beamk/all 的评分，默认开启以保留既有输出。MultiGE 族、PIER/PRS/PIERSubmodular 及 EXTR 的裁判要求
完整列表；其他通用 Slate 模型支持右补空槽的 K<N 评分。

当前实现已修复生成器坐标与 metric 主监督梯度，统一 AR batch mean；默认配置由 presets.py
统一管理。旧实验结果不能当作修复版指标，详见 [审计修复记录](../../../../docs/rosen/request_execution/changes/20261119_rerank-review-fixes.md)。
