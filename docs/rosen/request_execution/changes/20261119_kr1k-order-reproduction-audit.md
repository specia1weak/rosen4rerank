# 20261119 KR1K 复现 AUC 差异排查

## 问题与原因

旧 aligned AUC=0.700267677335224，新 raw CSV 重建 AUC=0.6978418303966081。
初次交付只列出行序和 pace 差异，没有做变量隔离，未达到复现审计标准。

完整按 request_id 对齐 Train/Valid/Test，除了 pace 浮点舍入外所有列一致；schema 状态相同。
旧 v3 先验 join/group_by 后没有固定顺序，features_raw 首行 request_id 为 1656482；
新链路最后显式 sort(session_id, req_idx)，首行 request_id=0。同 seed 的 shuffle
产生不同训练批次，Train 1,721,745 行中仅一行位置相同。

新增两次完整配对重训，四组初始化 hash、训练配置、421 步和 split 相同：

| 行序 | pace | Valid AUC | Test AUC |
| --- | --- | ---: | ---: |
| 旧 | 旧 | 0.7108117689543617 | 0.700267677335224 |
| 旧 | 新 | 0.7108117858970244 | 0.7002676706538306 |
| 新 | 旧 | 0.7092394341925199 | 0.6978418237152146 |
| 新 | 新 | 0.7092394366129005 | 0.6978418303966081 |

因此本次差异来自行序，pace 贡献约 6.7e-9。Train pace 最大差为 4.77e-7、平均绝对差约 1.73e-8，
不能用这个精度差来解释约 0.002426 AUC 的变化。

首步核对与正式 run 相同初始化，旧行序 loss=0.3126157522，新行序 loss=0.3185522139；
保持旧行序仅换 pace 的首步参数最大差约 2.33e-10，换行序则约 0.00999834。
使用正式 Trainer 的 10.0 梯度裁剪阈值。

## 修复与验证

_encode_requests 内显式按 session_id、req_idx_raw 排序后再编码/切分/shuffle；
request_id 分配前的 join 使用 maintain_order=left。回归将 raw 随机打乱，要求三份 parquet
的值和物理行序及 schema 全部一致。保留新链路确定性，不复制旧 hash join 的偶然排列。

修复后再次从原始 CSV 全量生成到 `nxt_continue_v1_canonical_20261119` 并完整训练，
`raw_canonical_audit_20261119` 的 data、初始化 hash、valid/test 全部指标和步数与之前独立 raw 链路完全一致，
Test AUC 仍为 0.6978418303966081。`uv run pytest tests experiments/kuairand/tests -q`：
**803 passed**；日志 `logs/kr1k_order_audit_full_tests.log`。

完整协议：KR1K 当前刷反馈已知后预测下一刷，Train 截至 2022-05-06，Valid 05-07、Test 05-08；
CPU 4 threads、seed=2026、batch=4096、完整 1 epoch/421 步，按 Valid AUC 选择后完整 Test，无 EMA。
本次是同初始化/同协议的复现差异排查，不推广为多 seed 优劣结论。

原始日志：`workspace/kr_time_budget/runs/nxt_continue_v1/audit_old_order_raw_pace_20261119/logs/train.log`、
`workspace/kr_time_budget/runs/nxt_continue_v1/audit_raw_order_old_pace_20261119/logs/train.log`。
逐列和首步核对 JSON：`workspace/kr_time_budget/runs/nxt_continue_v1/reproduction_difference_20261119/`。
脚本：`workspace/kr_time_budget/回复/新架构示例/复现差异核查/`。
公开说明：[实验 README](../../../../experiments/kuairand/README.md)。
