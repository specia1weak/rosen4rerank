# 需求执行规范入口

这个目录给 `AGENTS.md` 使用，负责把用户的自然语言需求落成可执行流程。

`AGENTS.md` 只做路由；本目录文档负责说明每类需求的边界、步骤、产物和汇报格式。

## 独立包与实验项目的边界

本仓库只迁入 `src/rosen/`、核心测试和框架文档。下文的 `src/custom_models/`、`@examples/` 是调用方实验项目的目录约定，未随本包迁入；执行实验请求时按用户指定项目的实际布局定位，不把这些目录当成本包已有文件。`workspace/` 基于调用方实验根目录，默认是启动时工作目录，可通过 `ROSEN_ROOT_DIR` 覆盖。

研究记录统一写到所属项目的 `research/`：`plans/` 放方案，`handoffs/` 放交接，`findings/` 放实验结论与经验总结。目录和文件命名约定见 [`research/README.md`](../../../research/README.md)。

## 需求类型

- [项目架构优化 / 代码设计](01_architecture_or_code_design.md)
- [新数据集接入 / Parquet 化](02_new_dataset_preparation.md)
- [新模型编写 + 数据集实验](03_new_model_experiment.md)
- [开放课题自主探索](04_autonomous_research.md)
- [历史实验审计 / 结论整理](05_historical_experiment_audit.md)

架构级改动记录放在 [changes/](changes/)。

## 通用执行原则

- 先判断用户真正要的是设计、实现、实验、审计，还是结论。
- 先读现有代码和产物，再做修改或重跑。
- 涉及长任务时，明确日志路径并记录进度。
- 涉及实验结论时，必须说明数据、切分、训练口径、评估口径和日志路径。
- 涉及源码修改时，保持最小变更，不顺手重构无关模块。
- 新增模型分支或修改模块时，不能只检查 shape、forward 和梯度；必须注意可学习 token/embedding 与主表征的尺度是否匹配、带额外参数轴的初始化是否压缩尺度、残差路径是否符合设计，以及随机数和训练口径是否被连带改变。验证方式按模块风险选择，不要求机械套固定步骤。
- 声称“单变量对照”前，必须用代码或 artifact 证明共有参数、数据顺序、batch、optimizer、训练步数和评估 checkpoint 口径一致；只设置相同 seed 不等于初始化相同。
- 如果 fixed-checkpoint 干预的输出变化与重训差异在数量级上明显对不上，优先查实现和实验协议，不要直接把差异解释成机制有效或无效。
- 需要沉淀人类可读结论时写入 `research/`，raw artifact 留在 `workspace/` 或 `logs/`。
