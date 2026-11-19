# EvaluatorManager 注册任务与流水线拆分

## 需求与实现

按用户要求将大 collect 拆成职责明确的步骤，保持原异常类型和文本。
当前文件已有主要拆分实现，本轮沿用并补齐边界、调用方、验证与文档。

- 注册项使用 frozen=True 的 _EvalTask，取代 dict；summary/clear 同步使用属性访问。
- _resolve_target 在默认推断后对不依赖 target 的 Resolver 提前返回，避免重复 required 条件。
- _filter_rows 统一 bool / BoolTensor[B] 过滤，空结果返回 (None,None)。
- _resolve_mask 根据显式 mask_field 或 schema label spec 取值。
- _resolve_users 保持 task→Manager→schema 的 UID 优先级及显式缺列报错行为。
- collect 只保留上述高层调用与 Resolver→Evaluator 流水线。
- filter_fn 标注为 Callable，支持普通函数；显式 falsey Resolver 不被默认实例替换。
- 补齐 bool True + 空 batch 的跳过行为，避免空数据进入目标或字段解析。
- 依赖内部 registry 字典的测试改为 task.resolver 属性；未保留旧字典兼容层。

## 验证

- 新增冻结任务、覆盖注册、显式 Resolver、空 batch、非法 filter、Mapping 同步切片、
  mask 覆盖/缺列、UID 优先级/缺列、自定义目标提前返回、原异常完整文本测试。
- 从上轮导出的单文件提取旧 manager，与本轮在 1,392 种组合下逐例比较：
  目标推断/冲突、行过滤/空 batch、自定义来源、UID/mask 取值的结果、异常类型和 args 均相同。
  日志 logs/evaluate_manager_tasks_differential.log。
- 完整回归 641 passed，logs/evaluate_manager_tasks_regression.log。
  Prediction 当前拒绝裸字典的报错已由其他编辑改为 Prediction fields，对应旧测试同步匹配
  当前公共错误前缀；本轮不修改 Prediction 或其错误文本。
- git diff --check、compileall，单文件源码导出逐段校验。

## 影响

公开 register/collect/summary/clear 行为不变，内部 registry 条目由 dict 改为冻结 task。
模型、训练器、数据集、评价指标未修改，没有启动训练。契约文档与源码阅读包同步更新。
