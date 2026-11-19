# 解析器原始候选宽度与默认整页视图修正

## 问题与实现

用户审阅指出 Mapping 预测与无 mask、无 scores 的纯 order 输入，可能在 slate 投射后
丢失原始 N，导致已合法的选择被 Resolver 误拒绝。

- order 校验前置到 Scorer 执行之前，候选宽度从原 mask 或原始来源获取，不从投射后的
  targets [B,K] / 标量 [B] 反推。Mapping 通过当前 target 选定原始分数槽位。
- BaseScorer.num_candidates(context) 提供不执行打分的原宽度查询。FieldScorer 读原字段宽度；
  ModelScorer 用模型声明，LookupScorer 用全排列表宽度，解决两边都返回整刷标量的场景。
  自定义标量来源若没有原始张量可查询，可以覆盖该方法，不需要重复跑裁判模型。
- Manager 未指定 Resolver 时，对 SlateEvaluator（含子类）默认 slate，其余默认 ranking；
  显式 Resolver 保持优先。slate 无 order 时直接评价原列表，有 order 时投射整页。
- SlotEvaluator 的预测分与标签统一为 CPU float64，覆盖 fp16/bf16 输入。
- in_slate_auc 的二值校验原本已由底层 InSlateAUC 执行，保留单处校验，新增单独注册的回归。
- 明确 reduce_slate 是支持外部候选 [B,M,K] 的公共辅助方法；collect 只接收 [B]/[B,K]，
  不把候选轴隐式当独立请求，新增正反用例。

## 验证

- 评估专项 110 passed，logs/evaluate_resolver_edge_cases.log。
- 完整回归 602 passed，logs/evaluate_resolver_edge_regression.log。
- 新增无 mask 的 order-only / Mapping、自定义标量目标、Lookup/Model 裁判元数据路径、
  非法选择在打分前失败、默认/显式 Resolver 优先级、半精度与二值标签、3D 聚合边界测试。
- 文档与单文件源码导出同步，compileall、git diff --check。

不改模型或训练协议，不启动训练；新默认只影响未显式指定 Resolver 的整页评价。
