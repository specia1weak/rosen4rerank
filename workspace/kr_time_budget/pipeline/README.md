# 当前代码从这里读

第19份：[check_weibull_behavior.py](check_weibull_behavior.py)只读已有test及checkpoint，统计经验停止屏数、提取首屏预算参数并计算全局/分桶曲线；[report_weibull_behavior.py](report_weibull_behavior.py)生成[核对报告](../回复/19韦伯/核对报告.md)。不重训，不改变旧模型或预测；仅预算参数读取需要一次首屏预算网前向。

[run_uid_zero.py](run_uid_zero.py)复用逐epoch训练链路，仅将内存batch编码后`user_id`置0，两塔/Fusion均生效；原始用户ID仍用于指标。与已有`epochwise5`逐seed逐epoch配对，结果在`runs/epochwise5_uid_zero`，见[实验说明](../回复/18拍板/UID置0实验说明.md)。

该消融已完成18条轨迹、90次完整test，见[报告与对照曲线](../回复/18拍板/UID置0实验报告.md)。重载checkpoint推理时同样要调用`zero_uid_inputs`，不可直接传原UID；[report_uid_zero.py](report_uid_zero.py)只读生成报告及独立审计。

新增[run_epochwise_exit.py](run_epochwise_exit.py)：沿用六配置×三seed，从头连续训练5轮，每轮末直接评估完整valid/test，第5轮固定停止，不恢复best。已完成90次完整test，见[实验报告与曲线](../回复/18拍板/逐epoch5轮实验报告.md)、[实验说明](../回复/18拍板/逐epoch5轮实验说明.md)。独立产物在[runs/epochwise5](../runs/epochwise5)，与下文预算内best入口分开；[report_epochwise_exit.py](report_epochwise_exit.py)只读生成报告。

当前主线使用明确命名的文件。已有版本文件用于历史实验复现，暂不移动或删除：旧run保存了其路径和源码哈希。

| 阅读顺序 | 文件 | 职责 |
| --- | --- | --- |
| 1 | [kr_request_schema.py](kr_request_schema.py) | 54个字段的业务分类、编码与两塔字段列表 |
| 2 | [exit_model.py](exit_model.py) | 独立Omni、EmbView、单/混合威布尔、pace、乘积BCE |
| 3 | [exit_training.py](exit_training.py) | v3读取、优化器分组、原生Trainer、验证与checkpoint、预测导出 |
| 4 | [run_exit_experiment.py](run_exit_experiment.py) | 单配置单seed的完整运行入口，按顺序串联上述代码 |

```text
data/v3 + train-fit schema
  -> load_batches：原生ParquetStreamDataset -> Interaction
  -> KRExitModel
       budget_view(34字段) -> budget_net -> Weibull gamma
       content_view(48字段) [+ pace] -> content_net -> stay
       prediction = gamma * stay
  -> JointExitTrainer：一个BCE -> backward -> clip(10) -> 一个AdamW
  -> 完整valid选择预算内best；epoch之间不恢复best或重置优化器
  -> 整条训练轨迹结束后，重载预算内best -> 完整test -> 预测与指标
```

共享背景30字段包括用户画像及`tab/day_of_week/is_weekend/start_hour`，两塔都读取，但Embedding参数独立。预算参数网另读4个预算先验；内容网另读18个反馈/列表字段。当前六个配置定义集中在`exit_training.EXPERIMENTS`。双塔四个配置均为仅屏数预算；`_pace`为内容/Fusion额外追加一个原始时钟派生数值。

模型中`stage=2`和`prior_stay`是历史兼容元数据/缓冲；它们不代表两阶段训练或辅助损失。为保持初始化和checkpoint结构，Ours保留未启用的Fusion head，Fusion保留未启用的预算/内容head；仅屏数配置保留不参与最终损失的分钟head。

## 数据与历史实验

- 数据生成：[build_requests_v3.py](build_requests_v3.py)，默认读取既有v3，不在训练时重新fit。
- 第13份矩阵原入口：[run_plan13_ablation.py](run_plan13_ablation.py)；历史模型：[time_budget_v10.py](time_budget_v10.py)。
- 历史报告生成：[report_plan13_ablation.py](report_plan13_ablation.py)。继续使用原入口核对已有run，避免新代码改写其源码证据。
- `run_exit_v*.py`、`time_budget_v*.py`、`report_v*.py`为各轮实验记录，不是当前主线的依赖入口。

当前单次运行示例（已检查语法和CLI；本轮未启动全量重训）：

```bash
uv run python -m workspace.kr_time_budget.pipeline.run_exit_experiment --experiment mixture_step_pace --epochs 1
```

默认生成独立`runs/current/<时间戳>_<配置>_<seed>/`，日志自动写入`logs/train.log`。不覆盖旧run。`--epochs 3`保留1轮和3轮预算内的best；`metrics.json`使用原生Evaluator的嵌套`overall.auc/overall.logloss`，历史报告中的扁平`overall.auc/logloss`结构未改写。本入口不提供断点续训；预算快照用于评估，不能把所选best参数与轨迹末尾的优化器状态拼接续训。

本轮整理范围与后续诊断安排见[第14份代码整理说明](../回复/14定稿/代码整理说明.md)。

当前模型已通过76项测试及一次真实数据诊断；[诊断结果和断点说明](../回复/14定稿/单次运行诊断.md)。诊断脚本是[debug_current_run.py](../debug_current_run.py)，历史模型仅在其新旧回归比较部分导入。

第16份新增[backtest_components.py](backtest_components.py)，只读关联既有test预测与当屏事件，计算组件回测；[report_component_backtest.py](report_component_backtest.py)生成[回测报告](<../回复/16 回测/回测报告.md>)。不参与训练、checkpoint选择或模型推理。

第21份辅助Loss独立放在[auxiliary_exit_model.py](auxiliary_exit_model.py)，继承当前模型并保持前向和所有参数结构不变；[run_auxiliary_exit.py](run_auxiliary_exit.py)复用逐epoch Trainer，训练单/混合+pace的左NLL、右BCE、双辅助三组。基线直接复用UID置0实验，不改写历史源文件。[实验说明](<../回复/21新 Loss/实验说明.md>)列出标签、会话归一化和口径。

```bash
uv run python -m workspace.kr_time_budget.pipeline.run_auxiliary_exit
uv run python -m workspace.kr_time_budget.pipeline.report_auxiliary_exit
```

新结果写入`runs/plan21_auxiliary/`，报告归档`回复/21新 Loss/`。训练目录已存在时入口拒绝覆盖；只生成已有分析的图表可给报告入口加`--report-only`。

用户要求降低辅助强度后，新增[limited_auxiliary_exit.py](limited_auxiliary_exit.py)：每项系数≤.1且辅助合计每步≤主Loss10%，缩放系数停止梯度。入口[run_limited_auxiliary.py](run_limited_auxiliary.py)，只读报告[report_limited_auxiliary.py](report_limited_auxiliary.py)；输出`runs/plan21_auxiliary_cap10/`。主模型、旧辅助模型和旧run的字节保持不变。定义见[弱辅助实验说明](<../回复/21新 Loss/弱辅助实验说明.md>)。
