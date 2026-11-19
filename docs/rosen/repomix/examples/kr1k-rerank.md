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
