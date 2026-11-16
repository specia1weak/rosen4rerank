# 项目架构优化 / 代码设计

## 适用场景

用户认为 rosen 框架有坏味道，或有明确意向对代码架构进行调整。

这类需求不是“直接按用户第一句话改代码”，而是先一起把设计想清楚。agent 需要理解现有结构，定位问题，和用户反复讨论后再落地。

## 先读什么

- `docs/rosen/DESIGN_PHILOSOPHY.md`
- `docs/rosen/ARCHITECTURE.md`
- `docs/rosen/API_REFERENCE.md`
- 相关源码目录：`src/rosen/`、`src/custom_models/`、`@examples/`
- 如果涉及数据顺序或 shuffle，再读 `docs/rosen/request_execution/02_new_dataset_preparation.md`

## 边界约束

- 不要为了单次实验污染核心框架。
- 不要在没有验证的情况下替换旧逻辑。
- 不要删除用户或历史实验留下的 workspace/logs 产物。
- 不要把 `@examples/` 里的一次性脚本直接复制进 `src/rosen/`。
- 这类任务以架构正确性为优先，不必强求兼容所有旧代码；如果不兼容，要明确说明影响范围和迁移代价。
- 如果改动影响范围较大，要先提醒用户，不要静默大改。

## 需要做的事情

### 查看并理解项目结构

先读代码和目录，不要凭印象判断。需要定位问题属于哪一类：

- 冗余：同一逻辑散落在多个脚本或模块。
- 重复：已有公共工具未被复用，或多个实现语义近似但行为不同。
- 依赖错乱：低层依赖高层、实验层污染框架层、循环导入、注册入口不清。
- 抽象错误：为了单个实验建了公共抽象，或应该公共化的能力仍困在脚本里。
- 行为不透明：数据流、训练流、评估口径或 shuffle 语义无法被清楚追踪。

### 与用户反复讨论设计

这类需求要走“反问 -> 建议 -> 反问”的节奏。

agent 应该主动提出设计选项和取舍，而不是直接替用户拍板。讨论时重点确认：

- 用户真正不满的是可读性、复用性、正确性、性能，还是实验可比性。
- 这次改动应放在核心框架、`custom_models`、`@examples`，还是只写文档。
- 是否允许破坏旧接口。
- 是否需要保留 fallback。
- 什么验证能证明新设计可接受。

### 代码编写与测试

设计达成一致后再写代码。实现时以架构清晰为优先，不必为了兼容历史临时路径牺牲新结构。

测试至少要覆盖“还能正常运行”：

- import / compile 检查。
- 小输入或小表 correctness。
- 代表性 example smoke。
- 如果改动影响训练或数据处理，尽量跑一次相关实验的轻量验证。
- 如果用户要求上线替代旧逻辑，需要补一组更接近真实场景的验证。

### 新模型和模块修改的必须注意事项

用户要求编写模型、增加分支、替换模块或修正已有实现时，基础的 import、shape、finite loss 和非零梯度只能证明“路径能运行”，不能证明实现语义正确，也不能证明新旧实验可比较。下面是必须保持警觉的事项，不要求每次机械执行同一套步骤；应根据模块形态、实验风险和异常信号选择足够的验证。

#### 可学习 token 和 embedding 不要被尺度淹没

- 如果设置了可学习 Token、Query、prompt、position/domain/type embedding 或新的 embedding 表，应检查其初始化尺度与它将要相加、拼接或参与 attention 的主 embedding/hidden state 是否处于合理量级。
- 尺度过小会让新表示在 residual、LayerNorm 前后几乎不可见；尺度过大会淹没主 embedding，导致模型看似快速收敛但实际被新分支主导。
- 不要假设后续 LayerNorm 一定能修复融合前的尺度失衡。对逐元素相加、attention logit、gate 和 residual，进入算子前的相对尺度本身就会改变方向、饱和度或梯度。
- 可按需要比较初始化时和训练后的 RMS、norm、attention/logit贡献或 `new/base` ratio；重点是确认量级合理，而不是要求所有模型输出同一个固定阈值。

#### 参数初始化必须按真实参数语义检查

- 不要因为张量 shape 正确就默认初始化正确。`xavier_uniform_`、`kaiming_*` 等函数对二维线性权重和三维/四维卷积权重采用不同的 fan-in/fan-out 解释。
- 设计新模块时尤其要注意三维参数。带显式 expert、domain、slot 或 token 轴的参数，例如 `[num_tokens, input_dim, hidden_dim]`，如果语义是多份独立线性层，直接对整个三维张量调用 Xavier 可能把额外轴按卷积权重解释，从而显著压低尺度；这类情况通常应逐 token/expert/domain 对二维切片初始化。
- 可以通过实际 `std`、RMS 或输出 delta 与理论 fan-in/fan-out 的期望尺度做针对性核对。只测输出非零无法发现缩小几十倍的分支。
- 对 residual 新分支，要同时记录 base state RMS、delta RMS 和 `delta/base` ratio；比例异常小时，先查初始化、norm 和乘法 scale，不要直接解释成模型主动学成 identity。

#### 残差保留必须符合设计意图

- 修改部分路径时，要回读真实表达式，确认原表示应该保留几次、在哪个子层前后保留。常见错误既包括漏掉 residual，也包括为了“稳定”额外多加一次 residual，导致信息被重复注入。
- 对多子层 block，要确认每一层读的状态符合预期。例如第二个子层究竟应读取 mixing 后的 `Z1`、原始 `Z0`，还是裸 delta，不能只凭 shape 判断。
- 对分支融合、write-back、memory refresh、skip connection 和 final readout，同样要确认 residual 的接收者与写回位置；局部路径正确不代表完整数据流正确。
- 验证可以是回读表达式、可手算小 tensor、branch-off/identity 对照或状态 OBS，按风险选择即可；核心是避免残差遗漏、重复和接错位置。

#### 相同 seed 不等于共有参数相同

- 新增模块会消费随机数。如果它插在已有模块之前初始化，即使 seed 相同，后续共有模块也会得到完全不同的权重。
- 做 paired architecture comparison 时，应保持共有模块的构造顺序不变，把新模块放在所有共有模块之后，或使用隔离的 RNG generator/forked RNG。
- 对强调严格配对的实验，应考虑增加 state-dict 配对检查：排除新结构独有参数后，B/C 同名同 shape 的共有 tensor 是否逐元素相等。不能只比较参数数量、均值或 seed 字符串。
- 如果模型支持把新增支路关闭，初始化后在相同 batch 上应验证 `new_model(branch_off)` 与 baseline logits 精确或在明确容差内一致。若不一致，先定位初始化、数据流或默认开关差异。

#### off/identity 路径要与残差语义一致

- `off` 应明确表示跳过整个新 block、只关闭其中一个算子，还是把 delta 置零；不同语义不得复用同一个名字。
- residual block 应按需要核对执行顺序，例如 `Z1 = Z0 + Mix(Norm(Z0))`、`Z2 = Z1 + FFN(Norm(Z1))`，并确认第二个子层读取的是 `Z1`，不是 `Z0` 或裸 delta。
- padding/mask 应在 residual 后重新施加，且无效 token 必须严格归零；还要验证改变相邻 episode/slot 不会越界影响当前单元。
- fixed-checkpoint off 只能证明当前 checkpoint 是否使用分支，不能替代同初始化、同训练协议的重训对照。

#### 优化协议属于实现正确性的一部分

- 修改 physical batch 会改变每 epoch optimizer step 数、梯度噪声、Adam moment 和 parquet bucket 尾批组成。若没有梯度累计和学习率/step 对齐，不能把它称为纯显存工程变化。
- 新旧模型比较时要显式记录 physical batch、effective batch、gradient accumulation、train rows、optimizer steps、learning rate、scheduler、warm start 和 EMA。
- 如果因为 OOM 改了 batch、token dim、序列长度或 cap，必须同步补同口径 baseline，或把结果标成 screening/non-causal comparison。

#### 数量级不匹配是优先排错信号

- 如果重训 AUC 差达到 `1e-3`，但 fixed-checkpoint 关闭新分支只引起 `1e-5` AUC 或极小 `|delta logit|`，不能直接说重训差来自该分支的推理贡献。
- 优先检查共有参数初始化、batch/step、数据顺序、checkpoint lineage、eval variant 和随机数消耗，再讨论机制。
- 发现这些问题后，旧结果应明确标记 `INVALID_FOR_CAUSAL_COMPARISON`，不要在结论中继续使用“有效/无效”的因果措辞。

### 报告情况并写回文档

最终汇报要说明：

- 原问题是什么。
- 设计方案是什么。
- 改了哪些文件。
- 测试了什么，结果如何。
- 有哪些影响范围和风险。
- 后续如果有人继续维护，应看哪个文档。

如果这是架构层面的长期改动，必须写回文档，不要只在聊天里解释。

### 大范围影响提醒

如果改动涉及公共 API、数据处理默认行为、训练/eval 语义、model registry、schema 保存/加载、shuffle 策略等范围，要明确提醒用户：

- 哪些旧脚本可能受影响。
- 是否需要迁移。
- 是否需要重跑历史实验或重新生成数据。
- 是否需要保留一个临时 fallback。

## 架构改动记录

凡是需求 1 下产生的架构级改动，都需要在 `docs/rosen/request_execution/changes/` 下写一份改动记录。

文件命名：

```text
YYYYMMDD_小标题.md
```

例如：

```text
20260717_target-sort-shuffle.md
20260717_schema-manager-cleanup.md
```

记录内容至少包括：

- 背景：为什么要改。
- 问题定位：冗余、重复、依赖错乱、抽象错误或其他。
- 设计决策：采用了什么方案，放在哪一层。
- 影响范围：可能影响哪些模块、脚本、数据或实验。
- 测试情况：做了哪些验证。
- 后续事项：还需要清理、迁移或观察什么。

## 文档写回要求

如果这是可长期保留的框架能力，需要补文档：

- 需求执行过程和架构改动记录放 `docs/rosen/request_execution/`。
- 框架知识和公共设计说明放 `docs/rosen/`。
- 实验结论和经验沉淀放 `research/`。

## 完成标准

- 用户能知道新设计为什么成立。
- 代码里没有明显一次性补丁残留。
- 至少有一个可复现验证结果。
- 如果影响范围大，用户已被明确提醒。
- 架构级改动已有 `changes` 记录。
- 新增或替换模型模块时，已经根据风险关注可学习表示尺度、额外参数轴初始化、残差保留、共有参数和优化协议；不能只以 smoke 能运行作为完成依据。
