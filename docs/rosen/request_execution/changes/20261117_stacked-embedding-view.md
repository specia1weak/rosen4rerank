# 20261117 保留字段轴的 embedding 输出

## 背景与问题定位

重排模型需要候选列表的 `[B, K, F, D]` 表征，其他按字段建模的模型也需要 `[B, F, D]`。
原有视图支持拼接或字典输出，调用方需要自行拆分或堆叠字段。先补通用的输出布局，避免
在 embedding 层引入任务分支，也避免提前绑定评估方案。

## 设计决策

- `OmniEmbLayer` 新增 `split_by="stack"`，对各字段 `[B, ..., D]` 输出沿 `dim=-2`
  堆叠，得到 `[B, ..., F, D]`；`EmbView` 与 `SeqGroupView` 直接复用。
- 字段顺序沿用原 source / setting 遍历顺序，和默认拼接相同；include/exclude 仅筛选。
- stack 与默认拼接一样排除未显式 include 的序列字段；序列组视图显式选择本组字段。
- 所有实际 tensor shape 必须一致。错误信息包含字段名及 shape，便于调用方拆分异宽字段
  或不同长度的序列。不自动广播、投影、折叠页面或改变序列长度。
- `EmbView.num_fields` 返回当前字段数；`field_dim` 返回共有字段宽度，空视图或异宽字段报错。
  原有 `embedding_dim`、`field_segments`、`field_slices` 继续描述拼接布局。
- 空视图前向保持返回 `None`。没有新增参数、初始化步骤或 embedding 表。

## 影响范围

仅增加 `OmniEmbLayer` 输出模式与视图元数据。旧的默认、name、source 输出行为保持不变，
旧式 `BoleEmbLayer` / `SideEmb` 不扩展。现有 MSR 模型和脚本无需迁移。
数据格式、schema、训练和评估流程不变，不需要重新生成数据或重跑历史实验。
此前尝试的 Eval 草稿已撤回，不属于本次改动。

## 验证

- 新增 `tests/rosen/emb/test_stacked_view.py`，覆盖静态与序列布局、字段顺序、逐字段数值、
  与拼接路径的梯度一致性、padding、过滤、高阶前导维、异宽和异形字段报错、空视图。
- `uv run pytest -q tests/rosen/emb`：10 passed，2 warnings。
- `uv run pytest -q`：211 passed，362 warnings，4.97 秒，包含原 204 个测试。
- `uv run python -m compileall -q src/rosen/emb/emblayer.py tests/rosen/emb/test_stacked_view.py` 通过。
- `git diff --check` 通过。
- 本地日志：[embedding 测试](../../../../logs/stack_embedding/embedding-tests.log)、
  [全量回归](../../../../logs/stack_embedding/pytest.log)。日志按仓库规则保留在本地，不提交。

## 后续可修改点

后续按实际模型需求增加监督声明和模型工具；页面折叠、排列、loss 与评估语义均不在本次
输出布局中固定。具体指标和自定义评估执行能力另行设计。

公共接口见 [API_REFERENCE](../../API_REFERENCE.md)，使用方式见 [EMB](../../details/EMB.md)。
