# 新模型编写 + 数据集实验

## 适用场景

用户希望新增模型、修改模型结构，或用某个数据集跑实验看效果。

## 先读什么

- `docs/rosen/CONTRIBUTING_MODEL.md`
- `docs/rosen/ADD_MSR_MODEL_AND_VERIFY.md`
- `docs/rosen/PARAMS_AND_EXPERIMENTS.md`
- `docs/rosen/TRAINING_FLOW.md`
- 现有模型：`src/rosen/models/msr/`、`src/custom_models/`
- 现有实验入口：`@examples/<dataset>/`

## 边界约束

- 实验性模型优先放 `src/custom_models/`，模型稳定后再考虑进入核心 registry。
- 如果用户没有指定数据集，默认使用最近一次上下文中实际使用的数据集。
- 默认用 full train + test 汇报结果，除非用户明确要求 smoke、valid 或 partial。
- 不要把模型改动和数据切分、训练参数改动混在一起，除非用户明确要求。
- 长实验需要有可追踪日志。

## 需要做的事情

### 理解模型意图和实验变量

先弄清楚用户想验证的是一个结构想法、一个机制假设，还是一个工程实现。需要明确新模型相对 baseline 改了什么，以及哪些参数必须保持一致。

如果用户只说“跑一下”，agent 应默认对齐最近使用的数据集和训练口径，并在汇报中说明这个默认选择。

### 选择模型落点

实验性模型优先进入 `src/custom_models/`。只有当模型接口稳定、有复用价值，并且不强依赖当前任务假设时，才考虑放入 `src/rosen/models/msr/`。

实现时应尽量复用已有 `MSRModel`、embedding view、head、loss 和 evaluator 约定，不要为了一个模型改乱公共训练链路。

### 一切从简洁开始

需要加入obs的模型，请复制原模型代码之后，做一个新Cls然后实现相关逻辑；尽量保持基准实现的简洁。

同时避免参数堆砌：避免一个基准模型内部的私有函数过多，避免无需调参的参数过多（例如一个Cls的初始化中的一些参数，明明后续实验基本无需调整，还当做一个参数存放）

### 编写模型并做 smoke

正式训练前要先确认模型能被注册、构造、前向、计算 loss，并能吃下当前数据集的 batch。

smoke 的目标不是得到指标，而是排除 import、shape、dtype、domain 字段、label 字段、loss backward 等基础错误。

同时尝试使用 `rosen.utils.expr_tensor`（源码：`src/rosen/utils/expr_tensor.py`）查看表达式是否符合预期。

smoke 前必须注意以下问题，但验证方式按当前模块风险选择，不要求机械执行固定流程：

- 如果增加可学习 Token、Query 或 embedding 表，注意其初始化和主 embedding/hidden state 是否同量级，避免新表示被主干淹没或反过来淹没主干；不要默认后续 LayerNorm 会自动消除融合前的尺度问题。
- 如果新模块使用三维或带 token/expert/domain/slot 轴的参数，注意整体 Xavier/Kaiming 可能按错误的 fan 语义压低尺度；应结合真实参数语义判断是否需要逐二维切片初始化。
- 确认部分路径的 residual 保留符合设计：既不要漏掉应有 residual，也不要额外重复保留；确认每个子层读取和写回的是预期状态。
- 对严格配对实验，考虑同 seed 构建 baseline 和新模型，排除新模型独有参数后比较共有 state-dict tensor，避免新增模块通过 RNG 消耗改变后续共有模块初始化。
- 如果有 branch-off/identity 开关，在相同 batch 上比较 baseline logits 和新模型 off logits；默认目标是精确相等，无法精确时必须解释容差来源。
- 检查新增分支的输入、输出和 residual delta RMS，不能只看梯度非零。梯度非零但 delta 比主干小几十倍，通常仍需排查初始化或 scale。
- 测试放在 `tests/`，不允许把 test 代码放到 `src/`。

### 正式对照需要关注的口径

除非历史协议明确规定其他值，新的推荐实验必须显式使用
`weight_decay=1e-6`，不得依赖 PyTorch optimizer 的默认值。Weight decay
只施加到普通矩阵/张量权重；`nn.Embedding`、bias、归一化参数和
embedding-like free vector 必须进入零衰减组。实验入口应复用
`rosen.utils.optimize.split_model_params_by_decay(...)`，并把实际
weight decay 和排除策略写入结果元数据。若为复现历史协议而使用其他值，必须
显式传参和记录，不能静默省略。

声称 A/B 是结构单变量对照前，需要关注下面这些口径。高风险或论文级实验应尽量在日志或 manifest 中留证；普通探索可按实际风险选取必要项，不要求为了填表而填表。

```text
dataset / split / schema
train file与物理顺序
train rows
physical batch
gradient accumulation
effective batch
optimizer与learning rate
scheduler
optimizer steps
epoch
seed
共有参数初始化hash或逐tensor比较结果
warm-start/checkpoint lineage
EMA
valid/test rows与eval variants
```

特别注意：

- physical batch 改变后，即使 train rows 和 epoch 不变，optimizer steps 通常也会改变；没有 accumulation 对齐时必须补同 batch baseline。
- token dim、序列长度、Episode cap、Query 数和字段集合都属于模型/数据变量，不能作为“仅为解决 OOM”的静默修改。
- 相同 seed 只表示 RNG 起点相同；只要新增模块改变构造顺序，共有参数就可能不再相同。
- fixed-checkpoint intervention 与 retrain comparison 回答不同问题，前者不能代替后者。

### 结果异常时先做实现审计

遇到远超预期的正负差时，不要立即形成机制结论。下面是常见排查角度，不规定固定顺序：

- baseline 与新模型首批数据是否相同；
- branch-off 后初始 logits 是否与 baseline 相同；
- 共有参数是否逐元素同初始化；
- 新参数初始化 std 是否符合理论值；
- physical/effective batch、optimizer steps 和学习率是否一致；
- full eval 是否加载了本 run 的正确 checkpoint；
- fixed-checkpoint `|delta logit|` 是否与重训指标差处在合理数量级。

如果实验声称严格因果对照，但共有初始化、训练协议或 checkpoint lineage 无法对齐，应将结果标为：

```text
INVALID_FOR_CAUSAL_COMPARISON
```

可以保留日志用于排错，但不得写成“模块有效/无效”的经验结论。

### 运行实验并观察过程

full 实验要有明确 run name 和日志路径。训练过程中需要能回答用户当前在做什么：训练、评估、数据准备、还是等待资源。

如果出现 partial AUC，要明确它是不是最终结果，不要提前当作结论。

### 汇报结果并写回文档

汇报应以指标和路径为核心，尤其是 `overall.auc`。如果有 domain evaluator，也应给出 domain 结果或说明路径。

如果实验形成可复用结论，需要写到 `research/`；如果模型设计稳定，再考虑补充到 `docs/rosen/`。

## 文档写回要求

如果实验形成可复用结论：

- 结果和解释写入 `research/findings/` 或对应课题目录。
- 只把稳定模型设计写入 `docs/rosen/`。
- raw log 保留在 `workspace/` 或 `logs/`。

## 完成标准

- 模型实现可复现，基础 smoke 通过。
- 如果用户要正式实验，应有 full train + eval 结果。
- 汇报包含具体 AUC、对照口径和日志路径。
- 对照实验已证明共有初始化和训练协议一致；若不一致，汇报必须明确标成非因果 screening 或无效对照。
