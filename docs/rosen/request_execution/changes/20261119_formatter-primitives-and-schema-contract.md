# 20261119 Formatter 原语下沉与 SchemaManager 查询契约

## 背景与问题定位

原先 schema setting 的 formatter 依赖 `data.padding`，后者又需要 SchemaManager。
为规避循环依赖，setting 使用函数内导入和 TYPE_CHECKING；padding 同时重复组装标签、mask、
AuxField 的规则，并用 getattr 接纳不完整的 manager。

两处转换语义不一致：MultiSparse 的 Polars 转换保留标签头部，而 tensor formatter 保留尾部；
SequenceSetting 的 truncate_mode 未传入元素和时间 formatter，直接格式化超长编码序列时总是保留尾部。

## 设计决策

- 按用户选定的方案 B，将 ColumnFormatter、各具体 formatter、padding 原语和方向类型放入
  `core/formatters.py`，不依赖业务层。`emb/schema/` 直接导入这些原语。
- AuxField、SlateLabel 与 EmbSetting 统一提供 get_formatters；整数 AuxField 的 pad_val 转成 int。
- SchemaManager 统一提供 fields、raw_fields、get_formatters。字段列表去重保序，前两者支持
  include_labels=False。formatter 汇总保留已有 feature mask 规则，标签本身可以覆盖，AuxField 随后加入，
  FeatureContext 的显式额外 formatter 最后覆盖。
- data.padding 只编排 FeatureContext / TensorFormatter。按用户本轮明确要求，保留原公共 formatter
  类、类型及常量的 re-export；这是对通用“不保留兼容层”约定的明确例外，规范定义只有一份。
- FeatureContext 对列名和额外 formatter 映射做只读快照；仍引用原 manager，未深度冻结 manager 或
  自定义 formatter。MappingProxyType 配合 __reduce__ 支持 pickle 和 DataLoader spawn。
- 保留 from_raw_manager：RawParquetStreamDataset 在线 transform 依赖它，用 raw_fields 读取原始列，
  用 fields 选择转换后的输出，避免提前读取尚未生成的 history_len 等列。
- 元素和时间列同步使用 SequenceSetting.truncate_mode；MultiSparse 标签固定 head，嵌套外层与内层独立截断。

## 影响范围与迁移

- 修改核心 formatter、emb/schema/{base,categorical,numerical,sequence}、emb/manager、emb/emblayer、
  data/padding 与 data/dataset，并更新 ARCHITECTURE、API_REFERENCE、details/DATA。
- emb 中 TYPE_CHECKING 和对 data.padding 的反向导入清零；移除的是本链路用于规避循环的函数内导入。
  其他模块的类型依赖、已有 legacy 隔离加载并不因本次拆分自动消失，未将本次结果宣称为全仓无懒加载。
- 自定义 EmbSetting 若覆写 get_sequence_formatters，需要接受 truncate_mode 参数；自定义 manager / 测试桩
  需要提供 fields、raw_fields 和 get_formatters 的完整调用契约，不再使用 getattr fallback。
- 已经由 Polars 按相同规格截断的数据不受方向修复影响；直接格式化的超长 MultiSparse、head 序列及对应
  时间列结果会改变。依赖旧错误行为的历史结果应按新语义重评，必要时重训，不能直接混作同口径比较。
- 不改变 schema 状态文件格式；不要求普遍重新生成已正确截断的 parquet。

## 测试情况

- 现有 emb/data 测试：75 passed。
- 新增原语与跨层契约回归：41 passed，覆盖 head/tail、left/right、内外层截断、空/null/零维缺失值、
  零长度输出、非法方向、整数精度、formatter 优先级、只读快照和 pickle。
- 同一小表验证直接 formatter、Polars transform、RawParquetStreamDataset 和 ParquetStreamDataset
  的稀疏/稠密/嵌套序列及时间列结果一致；原始表不含生成的长度列。
- 实际以 DataLoader(batch_size=None, num_workers=1, multiprocessing_context="spawn") 读取一次 batch。
- 独立进程验证两种 import 顺序、core-first 导入、旧路径对象一致性及 get_type_hints 可解析。
- 完整核心测试：`uv run pytest tests -q`，741 passed（32 条既有依赖弃用/Transformer 提示）；日志见
  [pytest.log](../../../../logs/formatter_refactor/pytest.log)。

## 后续事项

磁盘 ETL 从 manager 拆分属于独立改动，本次未实施。继续维护本链路先看
[数据契约](../../details/DATA.md) 与 [架构说明](../../ARCHITECTURE.md)。
