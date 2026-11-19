# 20261119 Log1PDenseSetting

## 背景与设计

新增无需统计量的对数压缩数值特征，沿用现有特征垂直切片，在
`emb/schema/numerical.py` 添加 `Log1PDenseSetting(field_name, source)`。
它继承 BaseNumericalSetting，复用 DenseFormatter 与序列 formatter；不修改 dataset 或 embedding 分发。

使用 Polars Float64 `log1p`，最后转 Float32。null/不可转换数值补 0；严格保持数学语义：
`-1 < x < 0` 输出负数，`x == -1` 为 -inf，`x < -1` 为 NaN，不自动截断负数，NaN/inf 保留。
构造后 is_fitted=True，fit 不扫描统计量；可用于 SequenceSetting 的元素，支持已有截断/padding。

## 影响范围

新增 EmbType.LOG1P_DENSE 和 schema 公开导出，独立类型防止加载已有同名 MinMax 字段的状态。
同步将 AutoMTL 与 HAMUR 的 Dense 类型判断接入 LOG1P_DENSE，避免误分到稀疏特征分支。
新增能力不改变已有 setting。若将旧字段从 MinMax 改为 Log1P，应使用新 schema 目录并重新编码原始数据；
旧编码 parquet 不应再次作为原始输入做 log1p。API 与语义详见 [EMB](../../details/EMB.md)。

## 验证

新增测试覆盖与 math.log1p 的数值一致性（含 1e-12、1e100、负数和缺失值）、数学定义域、
两种 fit 模式无需数据扫描、JSON/manager 保存加载、拒绝误载 MinMax，以及 scalar/sequence 经两条
Parquet dataset 到 OmniEmbLayer 的正确值和形状，以及 AutoMTL/HAMUR 的数值特征路由和前向。

`uv run pytest tests -q`：748 passed（32 条已有警告），其中新增 7 项测试。
编译检查与 `git diff --check` 通过。日志：[log1p_dense_setting_tests.log](../../../../logs/log1p_dense_setting_tests.log)。

## 后续事项

无额外迁移要求；由使用者选择需要对数压缩的字段。
