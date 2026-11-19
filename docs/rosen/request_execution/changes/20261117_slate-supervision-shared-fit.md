# 20261117 列表监督与同源序列拟合（Step 2）

## 背景与问题定位

Step 1 保留了 embedding 字段轴，但标签列表仍走通用 formatter，关联 mask 不会自动进入
Parquet 扫描字段。多个 SequenceSetting 共享未单独注册的原子 setting 时，还会重复收集
统计量，包装器的 fitted 状态可能不一致。此次补监督数据链路与共享拟合，不提前决定评估方案。

## 设计决策

- `emb.schema.base` 增加不可变 `SlateLabel`，从 `rosen.emb.schema` 和 `rosen.emb` 导出。
  校验非空标签名、正整数长度、左右 padding 方向及可选 mask 名。
- `SchemaManager.label_fields` 接受标量字符串和 SlateLabel 的混合声明，公开的
  `label_field` / `label_fields` 仍是字符串或字符串元组，去重保序。冲突的同名声明报错。
- `get_label_spec` 对标量和未知字段返回 None。`mask_fields` 去重保序，自动进入
  `fields()` 以及 raw 读取列集合。共享 mask 的长度与 padding 方向在 manager 构造时校验。
- 列表标签和 mask 绑定现有 float32 列表 formatter；标签用声明的 pad_val，mask 补 0。
  沿用 formatter 的取尾部截断规则；整行 null、NaN、空列表填充为 padding。
  已注册的 mask formatter 不覆盖，`extra_col_formatters` 仍拥有最终覆盖权。
- 两种内存模式均按原子对象身份去重，对每个目标只收集/解析一次统计量；来源严格按
  settings 顺序取第一个。原子在顶层时同样遵循顺序。fit/save/load 同步所有序列的 fitted 状态。
  `prepare_data()` 使用同一个去重收集方法，避免两条入口行为分叉。
- BaseModel 提供 LABEL、LABELS、DOMAIN 和 get_label_mask：已提供 mask 时转换 dtype，
  否则返回与标签同形状、同设备的全 1。没有默认标签且未指定标签时抛 ValueError。
  继续支持旧的仅标量 schema 测试替身，不要求它们实现新增查询方法。

## 影响范围与兼容性

标签属于监督声明，不进入特征 setting 或序列组。没有修改 BaseTrainer、EvaluatorManager、
Evaluator、MSR 模型或模型注册。没有新增重排模型、排列、loss 或评估策略。

现有脚本的标量标签索引、schema JSON 格式和 embedding 共享关系保持兼容。SlateLabel
由调用方在构造 manager 时声明，不写进现有 feature_meta.json；加载旧特征状态不会覆盖
当前调用方的监督声明。现有数据和历史结果无需迁移或重跑。

共享目标采用第一个来源是本次明确的约定，**不是合并多个序列的词表/统计量**。
后续来源独有的稀疏取值会按原有 OOV 规则处理，数值也不会扩大 min/max。需要覆盖全部
来源时，应先用所需训练数据显式拟合原子，再将其交给各序列共享；改变拟合来源后应使用
新的 schema 目录，避免复用已有 feature_meta.json。

## 验证

新增 `tests/rosen/emb/test_slate_label_and_shared_fit.py`：32 个参数化用例，覆盖：

- 声明校验、不可变性、混合/单独标签、名称去重、未知字段查询和共享 mask 冲突。
- prepared/raw 两条真实 parquet 读取链路，检查 mask 自动扫描、超长截断、左右 padding、
  null/空列表和标量输出；检查 formatter 显式覆盖优先级。
- sparse/dense × 两种内存模式 × 原子未注册/排在首位/排在末位；统计表达式和解析各调用
  一次、第一来源口径、save/load 的特征状态与编码结果一致、同源稀疏序列只创建一张表。
- 预拟合目标、已有 schema 的加载分支、新增共享包装器状态同步。
- BaseModel 监督属性与 mask 的显式/回退路径，形状、dtype 和设备一致（CPU 及可用的 MPS）。

执行结果：

- `uv run pytest -q tests/rosen/emb/test_slate_label_and_shared_fit.py`：32 passed，20 warnings。
- `uv run pytest -q`：243 passed，381 warnings，5.89 秒，包含此前的 211 个测试。
- compileall 和 `git diff --check` 通过。
- [专项日志](../../../../logs/slate_supervision/targeted-tests.log)、
  [全量日志](../../../../logs/slate_supervision/pytest.log) 按仓库规则仅保留在本地。

## 后续可修改点

多个来源的合并拟合、监督规格持久化、严格缺失 mask 策略可按后续模型需求独立扩展。
本次不推导有效曝光，不自动展平标签，也不让维度决定评估语义。
Step 3 与具体模型另行实施。

接口说明见 [API_REFERENCE](../../API_REFERENCE.md)，使用说明见 [EMB](../../details/EMB.md)。
