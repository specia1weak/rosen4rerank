# KR1K：从原始 CSV 到继续预测 AUC

在仓库根目录运行。只需要原始 KuaiRand-1K CSV，不需要任何旧 workspace、parquet、schema 或 checkpoint。

```bash
uv sync --locked --group dev
POLARS_MAX_THREADS=4 uv run python -m experiments.kuairand.run_nxt_continue \
  --source_dir /path/to/KuaiRand-1K/data \
  --experiment_name raw_weak_epoch1 \
  --device cpu
```

把 `--source_dir` 换成本机原始 CSV 目录。目录应包含：

```text
log_random_4_22_to_5_08_1k.csv
log_standard_4_08_to_4_21_1k.csv
log_standard_4_22_to_5_08_1k.csv
user_features_1k.csv
video_features_basic_1k.csv
```

命令会重新处理 CSV、切场/组刷、划分数据、只在 Train 拟合 schema，生成 parquet，再完整训练 1 epoch，
根据 Valid AUC 保存并加载 checkpoint，最后完整评估 Test。默认 batch=4096、seed=2026、CPU 4 threads，
UID 置零，普通双 Weibull、当刷反馈与 pace、弱双辅助监督；不使用 EMA。

输出：

```text
workspace/kr_time_budget/data/nxt_continue_v1_raw/
  manifest.json           # 原始文件 SHA256、代码 hash、依赖版本、切分与行数
  schema/
  train.parquet
  valid.parquet
  test.parquet
  logs/prepare.log
workspace/kr_time_budget/runs/nxt_continue_v1/raw_weak_epoch1/
  result.json             # test.overall.auc 为最终 Test AUC
  metrics.csv
  checkpoints/
  logs/train.log
```

数据目录必须为空，实验目录必须不存在；不会覆盖历史数据或偷偷加载已有 schema。
要重新从 CSV 跑一遍，同时换数据目录和实验名：

```bash
POLARS_MAX_THREADS=4 uv run python -m experiments.kuairand.run_nxt_continue \
  --source_dir /path/to/KuaiRand-1K/data \
  --data_dir workspace/kr_time_budget/data/raw_repeat \
  --experiment_name raw_repeat --device cpu
```

已有这条入口生成的数据时，可省略 `--source_dir`，只传 `--data_dir` 重新训练。
单独预处理也可运行：

```bash
POLARS_MAX_THREADS=4 uv run python -m experiments.kuairand.nxt_continue_v1 \
  --source /path/to/KuaiRand-1K/data \
  --output workspace/kr_time_budget/data/nxt_continue_v1_raw
```

## 实测口径

2026-11-19 使用本地原始 CSV，在全新输出目录重新生成并完整训练：

| 项目 | 数值 |
| --- | ---: |
| Train 刷数 | 1,721,745 |
| Valid 刷数 | 57,872 |
| Test 刷数 | 48,898 |
| 训练步数 | 421 |
| Valid overall AUC | 0.7092394366 |
| Test overall AUC | **0.6978418304** |

预测时点为当前刷反馈已知后、下一刷出现前；每六条事件重建一刷。首尾场排除，Train 场结束截至
2022-05-06，Valid 完整场在 05-07，Test 完整场在 05-08。历史先验严格早于查询日，Valid/Test 冻结至 05-06。
原始日志共 11,756,073 行，全列去重后 11,661,848 行。准备过程曾在 16 GB 内存机器完成。

历史 **0.7002676773** 使用冻结 v6 词表与物理行序；本次独立生成的行序及 pace 浮点计算路径不同，不能将其写成
精确复现旧值。这里交付的是已经实测约 0.7 AUC 的完整原始数据链路。

同一天又在只复制仓库源码、没有旧 workspace 的独立目录中执行上述单命令，完整预处理与训练后
得到相同 **0.6978418304**。核对 schema 元数据一致，按 request_id 对齐后 Valid/Test 只有 pace
浮点值存在差异。该干净目录中 `uv run pytest tests experiments/kuairand/tests -q`：**802 passed**。

### 为什么与旧值不同

已完成配对完整重训：固定同一初始化、seed、batch、421 步、loss、优化器与切分，
分别只交换物理行序和 pace。

| 物理行序 | pace 数值路径 | Test AUC |
| --- | --- | ---: |
| 旧 | 旧 | 0.700267677335224 |
| 旧 | 新 | 0.7002676706538306 |
| 新 | 旧 | 0.6978418237152146 |
| 新 | 新 | 0.6978418303966081 |

按 request_id 对齐后，所有数据列除 pace 浮点舍入外逐值一致，schema 一致。
旧行序来自历史先验 join 未固定的排列，新链路在 join 后排序再 shuffle；
相同 seed 对不同输入排列不会得到相同 batch。Train 172 万行仅 1 行位置相同，从首步更新就分叉。
换回旧行序恢复旧 AUC；pace 的影响约 6.7e-9。本次约 0.002426 的差异由训练行序造成。

当前编码入口明确先固定 session_id、req_idx_raw 顺序再 shuffle，并有打乱原始输入的回归测试，
避免未来 join 顺序变化再静默改变训练协议。详见
[审计记录](../../docs/rosen/request_execution/changes/20261119_kr1k-order-reproduction-audit.md)。

测试：

```bash
uv run pytest tests experiments/kuairand/tests -q
```

`test_nxt_continue_from_raw_cli.py` 会复制源码到没有 workspace 的临时目录，执行完整 CLI（小样本），
检查数据生成、训练、Test 结果及原始文件 hash。所有生成数据与训练产物留在本地，不提交 Git。
