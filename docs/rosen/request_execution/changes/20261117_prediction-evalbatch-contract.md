# Prediction 与 EvalBatch 统一预测评估契约

## 背景与定位

上一版多任务/重排用裸字典输出，register 再用 pred_key/label_field/mask_field 手动连线；
收集存在 collect/collect_batch 两套签名，Manager 和 Decorator 按能力探测分发。
这些是输出语义缺失、接口重复和模型私有指标进入核心评估器的问题。

## 设计决策

- core.Prediction 声明 scores/order/full_scores/target/extra；冻结属性与 Mapping 键，
  校验张量 Batch 对齐。extra 只接受批张量，支持统一布尔切片与设备迁移。
- evaluate.EvalBatch 为不可变上下文视图，含完整预测/Interaction 与当前目标已解析字段。
- Manager 绑定 SchemaManager，target 同时选标签与同名预测，mask 从 SlateLabel 推导；
  多标签歧义、声明冲突、字段缺失均报错，不按第一个预测 key 猜目标。
- 所有 evaluator 继承 BaseEvaluator，统一 collect(EvalBatch)/summary()/clear()；
  summary 的业务 TypeError 直接传播。日志 epoch/step 归 Manager。
- 保留组合 Evaluator，避免把全库 Top-K 能力变成 Point-wise 别名。
- 新增内置 OrderEvaluator，验证合法排列并评价最终顺序；SlateEvaluator 只保留有监督指标。
  MGE 候选指标、冻结先验基准均留 KR 实验层，实现同一个契约。
- Trainer 在实际模型预测后包装裸 Tensor，兼容 EMA teacher、predict_step override；
  CutAuditTrainer 一并迁移。TopKEvaluator 在局部副本上屏蔽历史，禁止写共享分数。

## 影响与迁移

数据、词表、模型参数名、损失和 checkpoint 张量未变；旧单任务 Tensor 模型、原始 KR1K
注册和混合 Evaluator 可继续运行。上一版字典 predict、pred_key/label_field/mask_field
注册以及 collect_batch/四参数收集需要迁移。显式 LegacyEvaluatorAdapter 支持旧四参数对象，
不处理槽位 mask，不在主链路自动探测签名。LogDecorator 的 epoch/step 日志改用 Manager。

两份 KR 新示例已迁移，原始 kuairand1k.py 原文未改。原完整训练 run 保存三份执行入口快照，
不覆盖原结果；当前代码的用法与历史实验报告分开记录。

## 验证

- 全量 `uv run pytest -q`：303 passed（含既有 MSR、Trainer、CutAudit、数据与 workspace 测试）。
- 目标优先级/冲突、字段缺失、掩码展平、DomainFilter、完整上下文、独立切片、冻结容器、
  Order 与 score NDCG 区别、Top-K 不修改输入、显式 Adapter、TypeError 传播均覆盖。
- BaseTrainer / EMATrainer 训练、早停、真实更新与 checkpoint 回环通过；额外验证 teacher
  裸 Tensor 包装使用 teacher.LABEL，而非 student.LABEL。
- 复用两个完整训练 1 epoch 最佳 checkpoint 测完整 test：293,388 曝光与 48,898 屏；
  新旧分数和排列逐值完全一致，全部指标差 <1e-12，无重新训练或数据截断。
- 日志：`logs/prediction_contract_regression.log`、
  `workspace/kr_time_budget/logs/prediction_contract_audit.log`。

## 后续维护

详细契约见 [EVALUATE](../../details/EVALUATE.md)，实际用法、复测与口径见
[KR Prediction 迁移报告](../../../../workspace/kr_time_budget/回复/新架构示例/Prediction迁移.md)。
新增特殊评估优先在实验层继承 BaseEvaluator；候选奖励不会再以模型名称进入核心 SlateEvaluator。
当前 Prediction 只支持批张量 extra；嵌套结构和任意静态元数据若以后确有需求，需显式定义切片语义。
