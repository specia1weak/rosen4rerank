# 设计理念

`rosen` 的核心目标不是做一个“什么都自动化”的训练黑盒，而是让实验作者在一个 `.py` 文件里看清楚完整流向：

`raw data -> split -> fit -> transform -> dataset -> Interaction -> model -> trainer -> evaluator`

这份文档只讲理念和边界，不讲每个模块的 API 细节。

## 1. 单文件可读优先

推荐的实验入口通常是一个相对完整的单文件脚本，而不是很多层封装之后只剩一句 `run_experiment(cfg)`。

原因很简单：

- 推荐系统实验的核心成本往往不在训练循环本身，而在数据列选择、切分规则、编码方式、评估口径。
- 如果这些关键逻辑被隐藏到太多间接层里，读者会更难判断实验到底做了什么。
- 对排错和复现实验来说，能在一个文件里看清数据从哪来、最后怎么进入模型，比“API 看起来更优雅”更重要。

## 2. 框架封装重复劳动，不隐藏实验逻辑

`rosen` 主要帮你复用这些重复基础设施：

- `SchemaManager` 管特征拟合、编码、落盘与复用。
- `ParquetStreamDataset` / `RawParquetStreamDataset` 管流式读取和 batch 生成。
- `BaseModel` / `MSRModel` 管 embedding 与多场景模型的共同接口。
- `BaseTrainer` / `EvaluatorManager` / `EarlyStopper` 管训练循环、评估和早停。

但它不会强行替你隐藏：

- 原始数据怎么 join
- train/valid/test 怎么切
- 哪些列进入模型
- domain 列如何解释
- 模型私有参数如何透传

这些恰恰是实验真正有差异的地方。

## 3. 显式数据流优先于“魔法”

当前推荐的主链路是显式写出来的：

```python
train_raw, valid_raw, test_raw = ...
manager.fit(train_raw)
train_lf = manager.transform(train_raw)
valid_lf = manager.transform(valid_raw)
train_path, valid_path, test_path = manager.save_as_dataset(...)
train_ds = ParquetStreamDataset(train_path, manager, ...)
model = build_model(manager, num_domains, cfg.model, **cfg.extras)
trainer = BaseTrainer(...)
trainer.run()
```

这样做的优点是：

- 出问题时容易定位是哪一段坏了
- 文档可以和代码天然对齐
- 实验复现时不需要先记住一套复杂约定

对应地，像 `prepare_data()` 这类更自动化的入口虽然存在，但不作为主推荐路径。

## 4. 参数系统服务于实验入口，而不是反过来

`ParamManager` 的定位不是做一个庞大的配置语言，而是给单文件实验脚本一个统一参数入口。

因此当前参数系统有两个很实际的取舍：

- 已知字段走 `dataclass`，适合稳定通用配置。
- 未知字段进入 `cfg.extras`，适合模型私有参数或脚本局部扩展参数。

这让实验作者可以继续保持扁平 CLI：

```bash
python run.py --model shavq_h2_ste_v1 --warmup_samples 8192
```

而不需要强制引入嵌套配置语法。

## 5. rosen 核心与项目实验层要分开看

这个仓库里有两层模型扩展路径：

- `src/rosen/models/msr`
  - 更偏框架内、通用、可复用的模型实现
- `src/custom_models`
  - 更偏项目实验、变体试验、论文复现和快速迭代

这不是重复设计，而是有意分层：

- 核心框架追求稳定接口和公共抽象
- 实验层追求快速试验和低心智负担

## 6. 什么时候该继续抽象，什么时候不该

一个抽象层值得存在，至少要满足下面两条中的一条：

- 它稳定消除了重复劳动
- 它让数据流更清楚，而不是更模糊

如果一个抽象层只是“把很多真实逻辑藏起来”，那通常不值得进入主路径。

## 7. 推荐的阅读方式

- 第一次接触时，先看 [`QUICKSTART_BEGINNER.md`](QUICKSTART_BEGINNER.md)。
- 想理解全链路时，看 [`DATA_FLOW.md`](DATA_FLOW.md) 和 [`TRAINING_FLOW.md`](TRAINING_FLOW.md)。
- 想理解模块分层时，看 [`ARCHITECTURE.md`](ARCHITECTURE.md)。
- 想查具体 API 时，看 [`API_REFERENCE.md`](API_REFERENCE.md) 和 `details/*`。
