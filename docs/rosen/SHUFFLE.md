# Shuffle 策略

这篇文档记录 rosen 当前推荐的物理 shuffle 方案，尤其是大 parquet 数据集、profile/sequence 数据集和 full train 场景。

## 1. 为什么需要物理 shuffle

很多原始数据或中间 parquet 都天然按用户、时间、source partition 排序。如果训练阶段只是顺序扫描 parquet，会出现：

- 一个 batch 里用户或时间高度集中
- source partition 顺序泄漏
- train loss 看似平稳但泛化受偏置影响
- DataLoader shuffle buffer 需要很大内存才能缓解

所以推荐在数据准备阶段把 train 做成物理 shuffle，训练时再用轻量读取。

## 2. 当前推荐：`target_sort`

`target_sort` 的核心逻辑：

```text
每一行 -> stable shuffle_id = hash(seed, file_idx, row_idx)
       -> 用 shuffle_id 的高位做 range target
       -> 每个 target 内按完整 shuffle_id 排序
       -> 按 target 顺序拼接或作为 shard 目录读取
```

对 `target_count` 为 2 的幂时，这等价于外部 random-key sort 的分桶版本。它避免了旧 append-bucket 方案里“source part B 总是在 source part A 后面”的顺序泄漏。

## 3. 和旧策略的区别

| 策略 | 行为 | 风险 | 适用 |
|---|---|---|---|
| `exact` global sort | 全表随机 key sort | 内存/时间开销大 | 小数据 sanity check |
| `target_sort` | range target + target 内 sort | I/O 增加，首次物化慢 | 默认推荐 |
| `bucket_chunk` | hash bucket + chunk shuffle | 不是严格 global，可能残留局部顺序 | fallback |
| `append_bucket` | train rows 直接 append 到 bucket parquet | source partition 顺序泄漏 | 兼容旧数据 |

## 4. 代码入口

单文件 checkpoint shuffle：

```python
manager.make_shuffled_checkpoint(
    lf,
    file_name="whole_shuffled.parquet",
    seed=2026,
    bucket_count=128,
    mode="target_sort",
)
```

split train/valid/test 时对 train 做 shuffle：

```python
write_split_bucketed_main(
    main_paths,
    output_dir / "main",
    train_filter=...,
    valid_filter=...,
    test_filter=...,
    train_bucket_count=1024,
    train_shuffle_mode="target_sort",
)
```

## 5. 参数建议

- 小/中等非序列数据：`bucket_count=128`
- 大型非序列数据：`bucket_count=256` 或 `512`
- profile/sequence 大数据：通常用 `user_partitions * 4`，例如 `256 * 4 = 1024`
- `bucket_count` 优先使用 2 的幂，这样 range partition 语义最干净
- `seed` 必须写入 manifest 或 run log，保证可复现

## 6. 训练侧建议

如果 train 已经物理 shuffle：

- `ParquetStreamDataset(..., shuffle=False)` 可以接受
- 不需要很大的 shuffle buffer
- 多 worker 的收益取决于 CPU 解码和磁盘吞吐
- valid/test 不要 shuffle，保持评估确定性

如果 train 没有物理 shuffle：

- 需要显式说明这是临时或 smoke 跑法
- 不要把结果当作 full train 结论

## 7. 验证标准

新 shuffle 策略替代旧逻辑前至少做：

- 小表 correctness：行数不变、列不丢、临时列不泄漏
- 文件级检查：target 文件数、manifest、临时目录清理
- 代表性实验：用同一模型对比旧策略或 exact/global shuffle
- 指标审计：报告 overall AUC/GAUC 和必要 domain 指标

已跑通的参考结果：

```text
KuaiRand Top5 non-seq EMA SharedBottom
shuffle = target_sort, bucket_count = 128
student overall.auc = 0.786300
EMA0.995 overall.auc = 0.787551
```

这说明 `target_sort` 在该数据集上接近 global shuffle 结果，可以作为默认推荐策略保留。

## 8. 常见问题

### GPU 空着是不是没跑

不一定。`target_sort` 的首次物化主要是 CPU/NFS I/O，GPU 会在训练真正开始后才被占用。长预处理要记录进度日志，避免看起来像“挂住”。

### 为什么不是只在每个 target 内 sample shuffle

因为如果 source parts 串行 append 到 target，target 内仍可能保留 source part 块顺序。按 `shuffle_id` 排序才能消除这个隐患。

### target 之间需要随机顺序吗

如果 target 是 hash range，高位 range + target 内 sort + target 顺序拼接就是 random-key sort，不需要再随机 target 顺序。随机 target 顺序反而会破坏严格 key range 语义，但通常对 SGD 影响不大。
