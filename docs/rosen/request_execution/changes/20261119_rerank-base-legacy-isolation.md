# 重排第二部分：输出坐标、公共模型基类与旧 Embedding 隔离

## 背景和定位

用户要求修复已选 [B,K] 分数在 slate 视图中的二次 gather，清理 Prediction/Interaction，
统一 MultiGE 与 Slate 的输入/监督基础设施，抽取公共候选评分，并隔离旧 Embedding 类。
实现不重训、不改变 schema 状态格式，不修改已有实验的数据和 checkpoint。

## 设计决策

- Resolver 在评分前确定原始 N。PredictionScorer 的二维输出宽度等于 K 且 K<N 时，
  视为已按单个 order 对齐，保留原值；宽度 N 仍按 order gather。
  小下标 [3,2,1,0] 和大下标 [9,7,4,0] 共用该规则，不再使用 order.max() 猜坐标。
  原始分数只取一次；标签/参考/掩码仍独立投射。单个已选 [B,K] 不广播成多个候选。
- Prediction.get_scores/get_order 统一执行任务查找：无目标报 ValueError，缺任务报列出
  可用任务名的 KeyError，显式 target 优先于 Prediction.target。EvalContext 使用同一入口。
- Interaction 只改文档，删掉“测试必须同用户连续且正例在前”的错误约束，转换和切片实现不改。
- MultiGERerankModel 继承 SlateRerankModel，删除重复 _view/_inputs/_mask/calculate_loss，
  新增 logits。子模块名及初始化顺序保持不变。
- components/scoring.py 增加 score_candidate_probs，所有 MultiGE 候选概率路径复用
  score_candidate_logits，保留 candidate_dense 坐标、样本/历史对齐、分块和 mask 语义。
- 增加 eval_diagnostic_pools=True；False 时只算 topk，跳过 lastk/randk/beamk/all。
  默认行为及原输出键保持不变。
- 按本轮用户规格，MultiGE 族 requires_full_slate=True，完整 OAM 维度保持固定；
  不沿用上一轮右补空槽的 MultiGE 部分列表裁判行为。EXTR 同样要求完整列表，
  通用 Slate 模型仍支持 K<N。没有把“评估层支持 N≥K”误写成每个生成器都支持动态长度。
- 十个旧 Embedding/Profile 类以相同类体移到 emb/legacy.py；emblayer.py 仅定义四个当前类，
  从 658 行减到 438 行。旧 import/pickle 路径经模块 __getattr__ 延迟转发，不增加训练模块。
- AuxField、schema 导出、原始/转换 parquet 读取及 TensorFormatter 均保留完整实现。
  补齐 get_aux_spec 可选契约；未在 schema/base.py 或 padding.py 留占位代码。

## 影响与兼容

当前 workspace 无需迁移。MultiGE 子模块的 state_dict 键名和值保持不变；完整列表的
训练和推理数值严格一致。唯一主动收紧的是 MultiGE K<N 裁判输入，按用户方案明确报错。
MultiGE/NAR/Seq2Slate 仍是固定长度生成；诊断关闭会省略诊断 extra 字段，属于显式选择。

## 验证

1. 修改前保存 12 组 MultiGE/NAR/Seq2Slate/metric 配置的快照，包含四种分量模式、两种
   candidate_dense 和 contrastive loss；修改后逐个 torch.equal 对比 state_dict 键/值、
   模块路径、初始化结束 RNG、所有 loss term、全部参数梯度、scores/order/extra，全部通过。
2. AST 比较验证 legacy.py 十个类体与迁移前逐个一致。
3. 核心 tests/rosen、KR 调用及 ML1M 链路共 **560 passed**；含小下标二次 gather、
   Mapping 错误、diagnostic 开关、候选分块尾批/梯度与旧导入测试。

日志：[第二部分测试日志](../../../../logs/20261119_rerank_refactor_part2_tests.log)。
数值验证和原源码快照：[verify_refactor.py](../../../../workspace/kr_time_budget/回复/新架构示例/第二部分/verify_refactor.py)，
[verification.json](../../../../workspace/kr_time_budget/回复/新架构示例/第二部分/verification.json)。

```sh
uv run pytest tests/rosen workspace/ml1m-rerank/tests \
  workspace/kr_time_budget/tests/test_architecture_examples.py \
  workspace/kr_time_budget/tests/test_rerank_benchmark_report.py -q --tb=short
uv run python workspace/kr_time_budget/回复/新架构示例/第二部分/verify_refactor.py verify
```

## 文档及后续

行为与局限同步至 details/EVALUATE.md、details/RERANK.md、details/EMB.md。
单文件原始源码包增加 legacy.py，包含完整 schema/base.py 和 padding.py。
NAR/Seq2Slate 独立轻量化及动态长度 OAM 不属于本轮范围。
