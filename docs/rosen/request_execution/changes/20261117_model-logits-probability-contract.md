# MSR / unified 训练原始分与预测概率分离

## 背景与定位

Prediction 为预测补充语义，不要求每个模型手工构造容器。原有多数 MSR 和 unified
单任务模型却把 raw logits 放在 predict 中；Crocodile/HAMUR 返回概率，存在同名异义，
直接计算 LogLoss 时口径错误。训练又调用 predict，使统一预测语义容易误改训练目标。

## 设计决策

- MSRModel 提供唯一的 `predict(interaction) = logits(interaction).sigmoid()`，返回 Tensor[B]。
- MSR 和 unified 子类实现 logits，训练直接消费原始分；复杂辅助损失保留原有 forward 路径。
- Crocodile / HAMUR 保留概率 BCE 和原有 HAMUR clamp，不顺带替换训练目标。
- HAMUR 骨干在最终 sigmoid 之前返回原始分，覆盖 MLP 2/7 层、DCN、WideDeep 和 adapter 开关。
- Trainer 统一将单任务 Tensor 包装为 Prediction 并绑定模型 LABEL。多任务、排列输出显式使用 Prediction。
- ObservationAuditWorkflow 为获得真实 delta-logit，明确读取模型 logits，再做一次 sigmoid。
- generative 的 GAN/扩散/scaler 及 utils 张量组件不是二元任务预测器，不套用 sigmoid/Prediction。

## 重排归属核对

可复用重排实现全部位于 `src/rosen/models/rerank/`：RerankModel、MultiGERerankModel、
attention/retrieval/generators，以及注册工厂。三生成器、cosine、lc、lc_soft 由同一模型配置覆盖。
KR 示例 `make_reranker` 直接构造公共模型；workspace 仅保留字段配置、数据准备、训练入口与自定义指标。
历史预算实验与输出继续保留原位置。unified 中 ScoreRerankMixFormer/ScoreConditionedLowRankRerank
处理内部 attention score，输出仍为单曝光 CTR，不是候选 slate 的排列模型，因此保持 unified 归属。

## 影响与迁移

模型参数名、形状、初始化及优化目标保持不变。大多数内置模型 predict 从 logits 改为概率：

```python
raw = model.logits(inter)       # 训练 BCEWithLogits / 诊断
prob = model.predict(inter)     # 评估概率，禁止重复 sigmoid
```

外部依赖旧 predict 原始分的代码改调 logits。HAMUR 直接调用底层骨干的脚本也需遵循新原始分契约。
扩展 MSR 的示例规范已同步修改。历史数据、checkpoint 和实验产物无需重新生成；历史 LogLoss
若曾使用 raw logits，不能与修正后的概率口径直接比较，需用旧 checkpoint 重新评估。

## 验证

- MSR 注册表 33 配置，加 HAMUR 8 个骨干/adapter 配置，共 41 例。
- unified 7 个源码模块中的 42 个具体模型，覆盖 train/eval 共 84 次配对检查，含额外 logit 分支。
- 修改前保存同 seed 参数状态哈希、预测、loss、逐参数梯度及 LossPort 账本；修改后精确比较，
  rtol=atol=0。两组初始化、loss、梯度、账本全部相同；概率等于旧 logits 的 sigmoid，
  Crocodile/HAMUR 则与旧概率逐值相同。
- 日志：`logs/msr_logits_comparison.log`、`logs/unified_logits_comparison.log`。
  临时基线张量仅为本次审计材料，路径记录不进入公共源码包。
- 持续回归：`tests/rosen/models/msr/test_prediction_contract.py`、
  `tests/rosen/models/test_unified_prediction_contract.py`；验证训练不调用 predict、概率语义、
  默认 Trainer + LogLoss。ObservationAudit 额外测试防止重复 sigmoid。
- 全量命令 `uv run pytest -q`，结果 444 passed，397 warnings（5.99s），日志 `logs/model_prediction_regression.log`。

## 后续维护

新增单任务 MSR/unified 模型实现 logits，继承 predict；多任务/重排用 Prediction 表达语义。
详见 `docs/rosen/details/MODELS.md`、`docs/rosen/ADD_MSR_MODEL_AND_VERIFY.md`。
不要依靠张量维度或预测数值范围猜测 logits/probability。
