# Agent Guide

`rosen` 是一个面向多场景推荐实验的 PyTorch 研究框架。它把原始数据处理、schema/特征编码、parquet 流式 dataset、模型、trainer、evaluator 和实验脚本串成一条显式链路，目标是让一次实验能被读懂、复现、比较和审计。

这个文件只负责**路由用户需求**。每类需求的边界约束、详细步骤、验证方式和文档写回要求，统一放在 [`docs/rosen/request_execution/`](docs/rosen/request_execution/README.md)。处理用户请求时，先判断需求类型，再读对应执行规范。

## 用户需求路由

### 1. 项目架构优化 / 代码设计

用户想优化 rosen 架构、整理代码结构、消除坏味道、判断某个逻辑是否应该上游或替代旧实现。不允许把test代码放到src下面

执行规范：[`docs/rosen/request_execution/01_architecture_or_code_design.md`](docs/rosen/request_execution/01_architecture_or_code_design.md)

### 2. 新数据集接入 / Parquet 化

用户希望处理历史未接入的新数据集，或把 raw 数据转成 rosen 可训练的 parquet/profile/sequence 数据集。

执行规范：[`docs/rosen/request_execution/02_new_dataset_preparation.md`](docs/rosen/request_execution/02_new_dataset_preparation.md)

### 3. 新模型编写 + 数据集实验

用户希望新增或修改模型，并用数据集训练、评估、汇报 AUC。未指定数据集时，默认使用最近一次上下文中实际使用的数据集。

执行规范：[`docs/rosen/request_execution/03_new_model_experiment.md`](docs/rosen/request_execution/03_new_model_experiment.md)

### 4. 开放课题自主探索

用户给出研究课题、机制假设或实验方向，希望 agent 自主拆解、设计实验、跑结果并判断正负信号。

执行规范：[`docs/rosen/request_execution/04_autonomous_research.md`](docs/rosen/request_execution/04_autonomous_research.md)

### 5. 历史实验审计 / 结论整理

用户希望检查已有实验数据、比较历史 run、确认 full train/full eval、抽取 AUC 或整理阶段性结论。方案文档、交接文档和经验总结应按用户指定的实验项目定位，不默认指向迁移前项目。

执行规范：[`docs/rosen/request_execution/05_historical_experiment_audit.md`](docs/rosen/request_execution/05_historical_experiment_audit.md)

## 执行约定

- 不兼容旧代码：接口调整直接删除旧名称、旧参数和旧取值，并统一迁移仓库内调用；不得添加兼容别名、fallback、双参数入口或仅告警仍可用的弃用层。一个概念只保留一个规范名。
- 使用 `uv` 执行项目命令：`uv run python ...`、`uv run pytest ...`。
- 本包维护命令默认在仓库根目录运行。实验命令在调用方项目根目录运行；`change_root_workdir()` 使用 `rosen.experiment.ROOT_DIR`，默认是启动时的工作目录，可通过 `ROSEN_ROOT_DIR` 指定。
- 长训练、长预处理、需要持续观察的任务显式写日志到 `workspace/.../logs/` 或 `logs/`。
- 统计和审计优先用 Polars 扫已有 parquet/log，不要为了看一个数字改处理脚本。
- 涉及结论时，必须说明数据、切分、训练口径、评估口径和日志路径。

## 目录速记

- `src/rosen/`：框架核心。
- `tests/`：核心测试；不允许把 test 代码放到 `src/` 下。
- `docs/rosen/`：框架文档；`request_execution/` 为需求执行规范。
- `logs/`：维护记录；实验日志也可写到调用方项目的 `workspace/` 或 `logs/`。
- `workspace/`：调用方实验根目录下的生成数据集、schema、训练日志和观测输出。
- `research/`：人类可读的研究记录；`plans/` 放方案，`handoffs/` 放交接，`findings/` 放实验结论与经验总结。详见 [`research/README.md`](research/README.md)。
- `src/custom_models/`、`@examples/`：原实验项目的模型和入口，未随本包迁入；涉及实验时按调用方项目的实际目录定位。
- 数据源根目录通过 `ROSEN_DATASOURCE_ROOT` 指定；可保存原始文件或信息无损、实验无关、易于二次派生的 canonical 数据，普通实验默认只读。

## 关键坑位

- `feature_meta.json` 是 `SchemaManager.fit(...)` 的 schema 状态；同一 workdir 下可能直接复用旧 schema。
- `ParquetStreamDataset` 本身产出 batch；外层 `DataLoader` 必须用 `batch_size=None`。
- `BaseTrainer.evaluate_epoch()` 默认调用 `model.predict()`；评估语义由模型自己保证。
- train 已经物理 shuffle 时，训练侧通常不需要大 shuffle buffer；valid/test 不要 shuffle。
- 用户只要 AUC 时，优先给 `overall.auc` 和日志路径，不要用大段 observatory 输出淹没结论。

## 当前 KR 实验的目录约定

用户明确要求当前探索阶段全部集中到 `workspace/kr_time_budget/`：实验代码在 `pipeline/`（避免遮蔽标准库 `code`），测试在 `tests/`，数据在 `data/`，训练结果在 `runs/`，总入口在该目录顶层。报告、实验说明及补充回复按对应 `Argon-plans/` 的序号与标题归档到 `回复/<序号及标题>/`，例如第10份计划放 `回复/10决定/`；同一计划的后续报告放相同目录，报告生成脚本也必须写入该位置，并维护相对链接。不要将这些实验模型提前迁入 `src/custom_models/`，也不要将本实验报告分散写到 `research/`。该用户要求优先于上述通用目录建议。入口：[`workspace/kr_time_budget/README.md`](workspace/kr_time_budget/README.md)。
