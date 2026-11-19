# 删除旧评价接口适配器

用户明确不需要兼容非 BaseEvaluator 的旧四参数评价器。

- 删除 LegacyEvaluatorAdapter 类及公共导出。
- EvaluatorManager.register 继续强制 BaseEvaluator，错误提示统一指向 collect(EvalBatch)。
- 更新注册拒绝测试、公开导出集合及当前文档；历史改动记录保留原文。
- 自定义评价器直接实现 BaseEvaluator 的 collect、summary、clear，不保留旧接口转发。

验证：评估专项测试及 git diff --check；日志 logs/evaluate_remove_legacy_adapter.log。
更新单文件架构源码导出，未运行训练。
