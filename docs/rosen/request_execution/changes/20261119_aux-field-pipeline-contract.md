# AuxField 在 KR / ML1M 训练与独立推理中闭环

## 背景与定位

SchemaManager、FeatureContext 和 TensorFormatter 已支持 AuxField，但 KR、ML1M 的正式重排
入口仍在 loader 中重复声明 extra_col_names 和浮点列表 formatter。KR 独立推理又手写一份列名
及长度，字段改名或调整 padding 时容易与训练脱节。这属于已有抽象未被主路径复用造成的重复。

## 设计决策

- KR `pointwise_manager` 声明 event_id/request_id/slot/user_id_raw 为 int64 AuxField。
- KR `rerank_manager` 声明 request_id/user_id_raw 为 int64，page_mask/prior_long_view 为
  固定长度 float32 AuxField，长度分别引用 PAGES/POIS。
- ML1M `schema` 声明相应的 request_id/user_id_raw/prior/page_mask。
- loader 只向 ParquetStreamDataset 传 manager，由 FeatureContext 自动推导读取和格式化。
- FeatureContext 两个工厂增加关键字 `include_labels=True`。独立推理显式传 False，只从
  读入/输出列排除全部 schema 标签，保留 SlateLabel 关联 mask 及 AuxField 的格式化规则。
- KR `predict_lc_mge.prediction_batches` 使用上述公共工厂，删除私有输入清单和 formatter。
- KR prior_ctr 继续保存在派生 parquet 中；当前重排模型、评估与导出不消费它，不再加载入 batch。

AuxField 继续只声明无需拟合/嵌入的输入；feature_meta.json 仍仅保存已拟合特征状态，加载时
由同一个 manager 工厂重新提供 AuxField 与监督声明，不引入新的持久化格式或兼容分支。

## 影响范围与迁移

框架修改限于 `src/rosen/data/padding.py` 的 FeatureContext 工厂。默认训练行为不变。
示范修改在 KR architecture_examples/run_architecture_examples/predict_lc_mge、ML1M pipeline/data，
以及其测试和 KR 现用的 verify_prediction_contract 审计入口。

KR loader 的唯一签名改为 `loader(path, manager, batch_size)`；移除无用途的 task 参数，
仓库内当前调用已迁移。外部调用者需删除该参数；没有兼容别名。历史 runs/source_snapshot
保持原样。依赖当前 KR batch 中 prior_ctr 的外部调用应按自己的实际输入需求在 manager 工厂
正式声明 AuxField；默认示范不加载未使用的 CTR 先验。

现有 parquet、schema 及 checkpoint 无需重建或重训；本次不改变模型、优化或评估口径。

## 验证

以下命令在仓库根目录执行：

```bash
uv run pytest -q tests/rosen/emb tests/rosen/data \
  workspace/kr_time_budget/tests/test_architecture_examples.py \
  workspace/kr_time_budget/tests/test_rerank_audit.py workspace/ml1m-rerank/tests \
  > logs/aux_field_contract_pytest.log 2>&1
uv run python -m workspace.kr_time_budget.回复.新架构示例.verify_aux_field_contract \
  > logs/aux_field_contract_real_data.log 2>&1
```

- 105 项测试通过，覆盖原始/已编码辅助输入、标签排除但保留共享 mask、左右 padding、字段
  改名和变长、超过 2^53 的 int64 ID、schema 重载，以及两条管线的模型训练与预测 smoke。
- KR 使用现有 lc_mge_complete_20261117 checkpoint、其数据目录的 test.ordered/test.unlabeled
  parquet，共 32 条请求，CPU FP32、batch=8。无训练/拟合；两种读取路径的输入、scores 和
  order 均严格逐元素相等。这是已有局部工程 smoke，不代表 full train/full eval 实验结论。
- ML1M 使用 user_requests6_v1 的现有 schema 与 test.ordered 前 16 条请求，只验证读取。
  训练 dataset 与无标签 context 的输入严格相等；prior=[16,6]、page_mask=[16,2]，ID 为 int64。
- 修改的 Python 模块通过 compileall 检查。

日志：[pytest](../../../../logs/aux_field_contract_pytest.log)、
[真实数据读取与 checkpoint 验证](../../../../logs/aux_field_contract_real_data.log)。

## 后续维护

正式输入只在各 manager 工厂声明一次。新增先验或改变序列长度时修改 AuxField，与模型维度、
数据生成规则同步；训练读取和独立推理格式化不再各自维护清单。
公共接口见 [DATA](../../details/DATA.md)、[EMB](../../details/EMB.md) 和 [RERANK](../../details/RERANK.md)。
实验侧说明与可复现验证脚本见
[AuxField 契约闭环](../../../../workspace/kr_time_budget/回复/新架构示例/AuxField契约闭环.md)。
