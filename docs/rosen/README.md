# rosen 文档入口

`rosen` 是一个围绕多场景推荐训练流程搭建的 PyTorch 工程，当前源码核心位于 [`src/rosen`](../../src/rosen)。

它的设计重点不是把实验流程藏进黑盒，而是让你尽量能在一个 `.py` 文件里看清楚：

`原始数据 -> schema/编码 -> dataset -> Interaction -> model -> trainer -> evaluator`

如果一个抽象层会让这条链路变得更难看懂，它通常就不值得存在。这也是为什么当前项目更偏向“显式数据流”和“单文件实验入口”，而不是层层封装。

## 运行与迁移约定

长任务应保留明确的运行命令、工作目录和日志路径，并记录进度及退出状态。
需要恢复的训练应验证 checkpoint、optimizer state、global step 和数据游标/已消费 batch；不能只恢复模型权重后从数据开头重复训练。

本仓库是独立的 `rosen` 核心包。文档中沿用的 `src/custom_models/`、`@examples/` 属于原实验项目，未随本包迁入；使用时按调用方项目的实际布局定位。当前 API 以 `src/rosen/` 为准。
研究记录统一使用 [`research/`](../../research/README.md)：`plans/` 放方案，`handoffs/` 放交接，`findings/` 放实验结论与经验总结。
需求执行规范见 [`request_execution/`](request_execution/README.md)。

## 先看这个理念

- 设计理念与边界：[`DESIGN_PHILOSOPHY.md`](DESIGN_PHILOSOPHY.md)
- 一张分层图看整体：[`ARCHITECTURE.md`](ARCHITECTURE.md)

## 我现在属于哪类读者

- 我刚 clone 仓库，需要配置 Python、uv、GPU 和本机 datasource
  - 先读仓库根目录的 [`README.md`](../../README.md)
- 我第一次接触 rosen，只想 10 分钟跑通一条最小链路
  - 先读 [`QUICKSTART_BEGINNER.md`](QUICKSTART_BEGINNER.md)
- 我已经知道大概链路，想看一版更接近真实工程入口的 quickstart
  - 读 [`QUICKSTART.md`](QUICKSTART.md)
- 我主要想调参，想知道参数从哪来、怎么覆盖、怎么透传给模型
  - 读 [`PARAMS_AND_EXPERIMENTS.md`](PARAMS_AND_EXPERIMENTS.md)
- 我想贡献一个新模型，或者判断模型该放进核心框架还是 `custom_models`
  - 先读 [`CONTRIBUTING_MODEL.md`](CONTRIBUTING_MODEL.md)
  - 再读 [`ADD_MSR_MODEL_AND_VERIFY.md`](ADD_MSR_MODEL_AND_VERIFY.md)
- 我想理解数据在框架里怎么流动
  - 读 [`DATA_FLOW.md`](DATA_FLOW.md)
  - 需要保留完整核心源码及 KR1K 示例的阅读包：[`repomix/rosen-core.md`](repomix/rosen-core.md)，附 [CTR + 长播多任务](repomix/kr1k-multitask.md)、[六视频重排](repomix/kr1k-rerank.md)、[原始 KR1K](repomix/kr1k-original.md)；导读和重新生成命令见 [`repomix/README.md`](repomix/README.md)
- 我想理解训练、评估、早停、checkpoint 怎么串起来
  - 读 [`TRAINING_FLOW.md`](TRAINING_FLOW.md)
- 我已经知道自己要找哪个类或模块，只想按源码查入口
  - 读 [`API_REFERENCE.md`](API_REFERENCE.md)
  - 再看 `details/*`

## 先记住这些真实约定

- datasource 保存原始数据或信息无损、实验无关的 canonical 数据，实验时只读；实验相关 schema、过滤 split、profile、shuffle、日志和 checkpoint 写入 `workspace/`。
- `ROSEN_DATASOURCE_ROOT` 指定本机数据源根；公共脚本不应提交机器绝对路径。
- 真正稳定的特征定义类是 `SparseEmbSetting`、`MultiSparseSetting`、`QuantileEmbSetting`、`MinMaxDenseSetting`、`VectorDenseSetting`、`SequenceSetting`。
- `SchemaManager.fit()` 会把 schema 持久化到 `work_dir/feature_meta.json`。同一个 `work_dir` 下如果元数据已存在，再次 `fit()` 会直接加载旧 schema。
- `ParquetStreamDataset` 和 `RawParquetStreamDataset` 都是 `IterableDataset`。如果你再包一层 `DataLoader`，必须使用 `batch_size=None`。
- `BaseTrainer.evaluate_epoch()` 默认走 `model.predict()`，所以评估语义由 `predict()` 决定。
- `ParamManager` 的优先级是 `命令行已知字段 > build(...) 已知字段 > dataclass 默认值`。
- 未知命令行参数会进入 `cfg.extras`，因此模型私有参数可以直接用 CLI 透传。
- `@examples/ml-1m` 下有不少历史实验稿，含旧类名，不应当当作当前 API 依据。
- 原实验项目的 `@examples/kuairand-1k/kuairan1k_top5.py` 是进阶实验入口，未随本包迁入；本包内的最小链路见 [`QUICKSTART_BEGINNER.md`](QUICKSTART_BEGINNER.md)。

## 推荐阅读顺序

1. 先看 [`DESIGN_PHILOSOPHY.md`](DESIGN_PHILOSOPHY.md)，建立“单文件可读、显式数据流”的心智模型。
2. 再看 [`QUICKSTART_BEGINNER.md`](QUICKSTART_BEGINNER.md)，先跑通最小链路。
3. 已经能跑通之后，再读 [`DATA_FLOW.md`](DATA_FLOW.md) 和 [`TRAINING_FLOW.md`](TRAINING_FLOW.md)。
4. 需要调参时看 [`PARAMS_AND_EXPERIMENTS.md`](PARAMS_AND_EXPERIMENTS.md)。
5. 需要扩展模型时看 [`CONTRIBUTING_MODEL.md`](CONTRIBUTING_MODEL.md) 和 [`ADD_MSR_MODEL_AND_VERIFY.md`](ADD_MSR_MODEL_AND_VERIFY.md)。
6. 需要深入模块细节时，再进入 `details/*`。

## 文档地图

### 按读者目标组织

- [`DESIGN_PHILOSOPHY.md`](DESIGN_PHILOSOPHY.md)
- [`QUICKSTART_BEGINNER.md`](QUICKSTART_BEGINNER.md)
- [`QUICKSTART.md`](QUICKSTART.md)
- [`PARAMS_AND_EXPERIMENTS.md`](PARAMS_AND_EXPERIMENTS.md)
- [`CONTRIBUTING_MODEL.md`](CONTRIBUTING_MODEL.md)
- [`DATA_FLOW.md`](DATA_FLOW.md)
- [`TRAINING_FLOW.md`](TRAINING_FLOW.md)

### 按源码模块组织

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`API_REFERENCE.md`](API_REFERENCE.md)
- [`details/CORE.md`](details/CORE.md)
- [`details/DATA.md`](details/DATA.md)
- [`details/EMB.md`](details/EMB.md)
- [`details/EVALUATE.md`](details/EVALUATE.md)
- [`details/MODELS.md`](details/MODELS.md)
- [`details/RERANK.md`](details/RERANK.md)：MultiGE、列表算子、共享梯度与参考实现差异。
- [`details/EXPERIMENT.md`](details/EXPERIMENT.md)
- [`details/UTILS.md`](details/UTILS.md)
- [`details/DATASETS.md`](details/DATASETS.md)
