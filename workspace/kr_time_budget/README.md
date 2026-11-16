# KR 时间预算实验

**报告与回复统一从[回复目录](回复/README.md)查看。** 后续汇报从[汇报素材导航](回复/汇报素材导航.md)开始，按数据、模型、结果、诊断、下一步串联材料。

| 要看什么 | 入口 |
| --- | --- |
| 全部回复，按计划归档 | [回复索引](回复/README.md) |
| 新增：逐epoch测test，固定训练5轮 | [实验报告与曲线](回复/18拍板/逐epoch5轮实验报告.md) · [训练日志](runs/epochwise5/logs/train.log) |
| UID字段置0，配对重训5轮 | [报告与曲线](回复/18拍板/UID置0实验报告.md) · [训练日志](runs/epochwise5_uid_zero/logs/train.log) |
| 第19份：真实行为Weibull与模型 | [核对报告及图表](回复/19韦伯/核对报告.md) · [只读日志](runs/plan19_weibull_check/logs/check.log) |
| 第21份：左NLL、右BCE辅助监督 | [实验报告与曲线](<回复/21新 Loss/实验报告.md>) · [实验说明](<回复/21新 Loss/实验说明.md>) · [训练日志](runs/plan21_auxiliary/logs/train.log) |
| 第21份补充：辅助合计≤主Loss10% | [弱辅助报告与曲线](<回复/21新 Loss/弱辅助实验报告.md>) · [弱辅助说明](<回复/21新 Loss/弱辅助实验说明.md>) · [训练日志](runs/plan21_auxiliary_cap10/logs/train.log) |
| 汇报材料顺序与结论边界 | [汇报素材导航](回复/汇报素材导航.md) |
| 定稿实验，含User-AUC及单/混合威布尔 | [第14份定稿报告](回复/14定稿/定稿实验报告.md) |
| 组件与当屏反馈的回测 | [第16份回测报告](<回复/16 回测/回测报告.md>) |
| 方向A与后续实验 | [第17、18份审阅](回复/18拍板/17与18审阅.md) · [后续实验](回复/18拍板/方案A后续实验.md) |
| 当前代码、训练入口与运行说明 | [pipeline/README.md](pipeline/README.md) |
| IDE断点调试 | [单次运行诊断](回复/14定稿/单次运行诊断.md) · [debug_current_run.py](debug_current_run.py) |

```text
kr_time_budget/
├── 回复/        # 报告、审阅、说明、历史归档
├── pipeline/    # 模型、处理、训练和报告生成代码
├── tests/       # 实验测试
├── data/        # 数据、Schema及构建记录
├── runs/        # checkpoint、预测、指标与运行日志
├── logs/        # 维护和验证记录
└── debug_*.py   # IDE调试入口
```

命令在仓库根目录使用`uv run`执行。已有完整训练产物无需重跑；复现和新实验入口见[代码说明](pipeline/README.md)。旧报告文件迁移表见[文档整理记录](回复/文档整理记录.json)。
