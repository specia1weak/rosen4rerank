# 历史实验审计 / 结论整理

## 适用场景

用户希望检查已有实验结果、比较多个 run、判断历史结论是否成立。

## 先读什么

- `workspace/` 中对应 run 的 log、csv、summary、parquet 元数据
- `logs/` 中 grid search 或批量实验日志
- `research/handoffs/`、`research/plans/`、`research/findings/`
- `docs/rosen/PARAMS_AND_EXPERIMENTS.md`
- `docs/rosen/TRAINING_FLOW.md`

## 边界约束

- 审计默认不改代码、不重跑，除非用户要求。
- 不要把不同数据、不同切分、不同 seed、不同 eval split 的结果直接比较。
- 不要把 valid AUC、test AUC、partial AUC 混成一个数。
- 用户只要核心指标时，不要用大段观测输出淹没结论。
- 如果拿不到信息，直接说拿不到，不要猜。

## 需要做的事情

### 定位历史产物

先找到实验留下的真实证据：workspace run 目录、日志、csv、summary、checkpoint、config、交接文档或经验总结。

不要只根据文件名判断实验内容，关键参数和指标应从日志或配置中确认。

### 还原实验口径

审计的重点是可比性。需要确认数据集、切分、模型、seed、batch size、训练轮数、eval split、是否 full train/full eval，以及是否 student/EMA。

如果口径不一致，应该先说明不可直接比较，再给谨慎解读。

### 抽取核心指标

根据用户问题抽最小必要指标。通常优先给 `overall.auc`，必要时补充 `overall.gauc`、domain 指标、loss、速度或资源信息。

如果用户明确只要 AUC，就只给 AUC、口径和路径。

### 判断结论是否成立

结论应基于同口径对照，而不是单个 run 的绝对值。需要区分：

- full 还是 partial。
- valid 还是 test。
- student 还是 EMA。
- 单 seed 还是多 seed。
- smoke 还是正式实验。

### 汇报和写回

汇报要简洁，先回答用户问题，再给证据路径和风险说明。如果这次审计形成阶段性结论，应写回 `research/`。

## 文档写回要求

如果用户要求“整理”或审计结论具有长期价值，需要写到 `research/`：

- 实验集合。
- 筛选和比较口径。
- 指标表。
- 结论。
- 不可比项。
- 后续建议。

## 完成标准

- 用户能立即看到核心 AUC 或核心结论。
- 每个数都有来源路径。
- 明确说明 full/partial、valid/test、student/EMA。
