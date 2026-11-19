# 重排审计清单完整修复

## 背景和范围

用户要求全部修复梯度、生成器坐标、OAM 部分列表、空视图、Resolver、选优导出、配置默认、
早停、重复候选评估、TF 损失量纲、离散 LC、Interaction 索引、阶段日志和 checkpoint 路径问题。
本轮改变训练与预测语义；不再要求与第二部分整理前的损失/输出数值等价。
历史数据、checkpoint、benchmark 指标及其原源码快照保持原样。

## 逐项落实

| 审计项 | 实现与验证 |
| --- | --- |
| Metric 当前 slate 梯度被切断 | 候选主监督使用带梯度 cate/history/dense；冷启动 page_mask 全零时保留的 cate 梯度非零，embedding 表梯度非零 |
| Metric 无用的原序 evaluator 前向 | 抽取 generator loss 和 contrastive helper，不再先计算再覆盖原序 evaluator loss；输入只计算一次，greedy 选序在 no_grad 下执行 |
| Teacher/greedy 两遍生成 | 保留必要的两种语义：teacher forcing 训练与合法 greedy 候选选择不能混用；不把 teacher argmax 重复记录当排列 |
| Pointwise 坐标错位 | 直接在原始 dense 上打分，logits、labels/mask 和 order 均为原槽位坐标；Pointwise/Listwise 的 candidate_dense 同步 gather 到候选位置 |
| PIER/PRS/PIERSubmodular 部分列表 | PIER 声明 requires_full_slate=True，子类继承；裁判明确拒绝 K<N，避免 OAM 无 mask 空槽污染 |
| predict 与 score_permutations mask 不一致 | PIER/MultiGE 完整候选裁判同样乘按排列投射的 slot_mask；prior 模式数值与 predict 候选分对照 |
| 空 domain 变全选 | _filter_settings 区分 None 与空集合，显式空 include_fields/target_sources 产生空视图；单个 FeatureSource 也可用 |
| Resolver 拒绝共享候选 | slate 使用 normalize_orders 全契约；固定共享候选通过 orders=[M,K]/[1,M,K] 注册，行过滤不切共享常量，M=B 也不歧义 |
| Resolver 猜测已选分数 | 删除 PredictionScorer 特判，Prediction/Field/LabelScorer 统一显式 alignment=candidates/selected；默认二维来源只能是候选池 |
| Ranking 跳过形状校验 | 所有候选来源宽度一致，ranking 的 scores/labels/ref_scores 必须对齐，NDCG-only 也不能绕过 |
| 导出硬编码 mean argmax | 模型选优时输出 extra.chosen_scores，predict_lc_mge 直接消费；测试故意让该结果与 mean argmax 不同 |
| 变体默认三处分裂 | presets.py 唯一默认表，工厂、直接子类及 KR 入口共用 resolve_rerank_config；显式覆盖始终保留 |
| Benchmark 静默覆盖且协议缺配置 | CLI 用 None 表示未指定；15 个命名模型默认逐个检查，protocol.json 写 effective_model_configs 和训练语义 |
| 早停错位 | KR rerank 选择 valid chosen_order.ndcg@6，pointwise 选择 long_view.auc，测试令二者反向移动确认选对轮次 |
| 手写候选 evaluator/resolver | 删除两类，KR 直接注册核心 SlateValueEvaluator 与 BatchResolver/PredictionScorer，候选指标键改为 cand_mean/cand_max |
| AR batch sum/mean 混量纲 | NAR/Seq2Slate/metric 全部每步 batch mean、按步骤求和；单请求与重复三份的 loss/gradient 对照一致 |
| 无梯度 LC 混入 loss | exact LC 只记录 loss_diagnostics，KR metrics.json 持久化；soft LC 移除固定 reference-reference 常数，独立 gen.coverage 为可微优化项 |
| Interaction Tensor.tolist | 直接按 Tensor 在列所在设备切片，numpy 索引转换为 Tensor；测试禁止调用 tolist，覆盖布尔/整型/uint32 |
| valid/test 日志不分阶段 | summary 新增 stage，BaseTrainer 传 valid/test；日志检查两者同时带 epoch/step |
| checkpoint 文件名猜测 | Trainer 记录真实 best_checkpoint_path/last_checkpoint_path，KR/ML1M 直接使用；不规则轮次步数测试保存和恢复正确路径 |

## 兼容及实验结论边界

- 已选二维分数应显式注册 PredictionScorer(alignment="selected") 或 FieldScorer(..., alignment="selected")。
  不再接受上一轮按 K<N 猜坐标的行为；默认候选来源宽度错配会报错。
- Prediction 的所有字段仍要求首维为 batch；共享候选属于 Resolver 配置，不塞进 Prediction.extra
  冒充逐行数据。逐请求 extra.topk_order 继续支持 [B,M,K]。
- 原 checkpoint 的参数名和形状可加载，但生成器输入坐标、候选特征与损失已有修复，
  不能宣称恢复旧实现的预测/训练语义。要重新比较效果必须使用独立目录重训同口径 baseline。
- 原 20261118 KR benchmark 和 ML1M 结果保持历史记录，不作为本轮修复后指标。
  第二部分 verification.json 的逐张量等价结论仅针对当时的源码快照。
- 全量 benchmark 本轮没有重跑。没有改 train/valid/test 切分，也没有使用 held-out 反馈训练。

## 验证与交接

```sh
uv run pytest tests/rosen workspace/ml1m-rerank/tests \
  workspace/kr_time_budget/tests/test_architecture_examples.py \
  workspace/kr_time_budget/tests/test_rerank_benchmark_report.py \
  workspace/kr_time_budget/tests/test_rerank_audit.py -q --tb=short
```

结果：**615 passed**。完整输出写入 [测试日志](../../../../logs/20261119_rerank_audit_tests.log)。包含真实小表 KR 训练、
checkpoint 重载、label-free parquet 导出、ML1M DNN/AR 链路和全部 tests/rosen。
新增重点测试在 test_audit_regressions.py、test_explicit_score_coordinates.py、test_empty_views.py、
test_interaction_row_index.py、test_stage_checkpoint.py 和 KR test_rerank_audit.py。

文档同步至 details/EVALUATE.md、details/RERANK.md、details/RERANK_BASELINES.md、TRAINING_FLOW.md。
当前完整源码见 workspace/kr_time_budget/回复/新架构示例/重排链路代码审阅.md；
原始研究结果和报告不覆盖，新增审计说明链接供后续使用者区分。
