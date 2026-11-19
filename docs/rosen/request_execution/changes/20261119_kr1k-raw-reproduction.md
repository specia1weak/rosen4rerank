# 20261119 KR1K 原始 CSV 完整复现交付

## 问题

之前验证了已有 aligned parquet 的训练复现，却未验证新机器只有原始 CSV 时的完整入口。
仓库 README 还指向被 Git 忽略的 workspace 文档，一项历史对齐测试也从 workspace 导入代码。

## 修改

- run_nxt_continue 增加 `--source_dir`，显式调用已有 prepare_data，从 CSV 到训练/Test 一次完成。
  传入原始目录时，数据输出必须为空，不自动复用旧 schema。
- 默认产物为 `nxt_continue_v1_raw`，避免将独立 CSV 重建与冻结 v6 的 aligned 数据混称。
- load_data 在创建 SchemaManager 前检查完整数据文件，缺失时给出原始 CSV 生成命令。
- 预处理 manifest 记录原始日志/用户/物品 CSV 的 SHA256、预处理代码 hash 和运行时版本。
- 测试改为导入已提交的 experiments 参考实现；新 CLI 回归在不含 workspace 的源码副本中执行。
- 数据根目录默认改为当前目录下 datasource，删除开发者本机绝对路径；保留原有环境变量读取语义。
- 仓库 README 和 experiments/kuairand/README 提供原始 CSV 文件清单、单命令、输出和实测指标。

## 验证与结果

本地原始 CSV 全量重新处理到新目录，无旧 parquet、schema 或 checkpoint 输入：
Train/Valid/Test 为 1,721,745 / 57,872 / 48,898 刷，Train 截至 2022-05-06，Valid/Test
分别为 05-07/05-08 的完整场。当前刷反馈后预测下一刷，CPU、seed=2026、batch=4096，
完整训练 1 epoch / 421 steps，按 Valid AUC 选 checkpoint，完整 Test，无 EMA。

- Valid overall AUC：0.7092394366129005。
- Test overall AUC：0.6978418303966081。
- 原始数据：11,756,073 行，去重后 11,661,848 行。
- 日志：`workspace/kr_time_budget/data/nxt_continue_v1_from_raw_20261119/logs/prepare.log`、
  `workspace/kr_time_budget/runs/nxt_continue_v1/from_raw_20261119/logs/train.log`。

又在 `/tmp/rosen-kr1k-delivery-20261119` 只复制仓库源码，从没有 workspace 的状态运行同一 CLI，
完整重新生成并训练得到相同的 0.6978418303966081。该目录中核心和 KuaiRand 测试共 **802 passed**。
原始文件 hash、代码 hash 与版本已写入 manifest。核验日志与结果复制到
`workspace/kr_time_budget/runs/nxt_continue_v1/raw_csv_delivery_20261119/`，测试日志在
`logs/kr1k_clean_checkout_tests.log`。

原始重建后的物理行序、pace 浮点计算路径与历史 frozen v6 不同；schema 元数据相同，
按 request_id 对齐后 Valid/Test 只有 pace 值有差异。不能称为精确复现历史 0.7002676773；
该成绩是从原始 CSV 得到约 0.7 的可交付结果。后续看 [完整运行说明](../../../../experiments/kuairand/README.md)。
