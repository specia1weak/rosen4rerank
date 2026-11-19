# models/

列表重排子包 `rosen.models.rerank` 的模型与独立注册入口见 [RERANK](RERANK.md)。
其 `MultiGERerankModel` 复用 BaseModel、Trainer 与评估路由，通过配置选择三生成器、
Cosine、LC 或可微 LC；不并入下方多场景模型注册表。

`models/` 是当前仓库最宽的模块。真正有训练入口意义的，主要是 `BaseModel`、`MSRModel`、`MODEL_REGISTRY`、`DomainTowerHead` 和一批多场景模型实现。

## 1. 基础模型

### `BaseModel`

源码在 [`src/rosen/models/base.py`](../../../src/rosen/models/base.py)。

```python
class BaseModel(nn.Module):
    def __init__(self, manager: SchemaManager):
        self.manager = manager
        self.omni_embedding = OmniEmbLayer(manager=manager)
```

它负责自动挂载统一 embedding 层，其他逻辑由子类自己实现。

## 2. `MSRModel`

源码在 [`src/rosen/models/msr/base.py`](../../../src/rosen/models/msr/base.py)。

```python
class MSRModel(BaseModel):
    def __init__(self, manager, num_domains, **kwargs):
        ...

    @classmethod
    def from_manager(cls, manager, num_domains, **kwargs):
        ...
```

### 核心约定

绝大多数 MSR 实现都会在 `__init__` 里设置：

- `self.DOMAIN = self.manager.domain_field`
- `self.LABEL = self.manager.label_field`
- `self.input_view = self.omni_embedding.whole`

### 训练 logits 与预测概率

所有内置 MSR 注册变体以及 `models/unified/` 的单任务 CTR 模型采用同一边界：

```python
raw = model.logits(interaction)       # Tensor[B]，未归一化，供训练/诊断
probabilities = model.predict(interaction)  # Tensor[B]，等于 raw.sigmoid()
```

子类实现 `logits(interaction)`，基类 MSRModel 只实现一次 `predict`。
单任务模型不需要构造 Prediction；Trainer 在统一评估边界包装 Tensor 并绑定 model.LABEL。
需要多头、最终排列或附加输出的模型才显式返回 Prediction。

训练损失不调用 predict。带辅助损失的模型保留原有 `_forward_dict`/标签条件路径，
避免重复 forward 或丢失辅助项。Crocodile 与 HAMUR 保留原来的概率 BCE；HAMUR 也保留
clamp 语义，但直接从 logits 生成概率计算损失，绝不再依赖预测接口。
HAMUR 底层网络已在最终 sigmoid 之前输出 raw logits；MLP/DCN/WideDeep 及 adapter 开关均覆盖。
forward 的参数签名仍由具体骨干定义；`logits(interaction)` 是统一的 Interaction 入口。

迁移外部代码时：原先 `BCEWithLogits(model.predict(inter), y)` 改成 `model.logits(inter)`；
原先对 predict 结果再 sigmoid 的代码要删除额外 sigmoid。旧 checkpoint 的参数名与形状不变，
但大多数 MSR 的 predict 从 raw logits 改为概率，因此历史直接消费 predict 的脚本需核对口径。

同样适用于 MixFormer、SASRec、RankMixer、TokenMixer、HyFormer、HeadedCTR 和 UniRank 系列，
包括额外 logit 分支的子类；这些模型继续继承 MSRModel 的概率预测接口。
`ObservationAuditWorkflow` 要求显式提供 logits(interaction)，直接读取原始分后计算一次 sigmoid。

`models/generative/` 的 GAN、扩散及 scaler 是生成/变换组件，不属于二元打分接口；
它们的 forward/sample/denoise 不添加 sigmoid 或 Prediction 包装。

### `from_manager()`

如果 `__init__` 允许 `**kwargs`，就直接透传。
如果不允许，它会过滤掉不支持的参数，并发出 warning。

## 3. `DomainTowerHead`

源码在 [`src/rosen/models/msr/components/heads.py`](../../../src/rosen/models/msr/components/heads.py)。

```python
head = DomainTowerHead(
    num_domains=3,
    input_dim=128,
    hidden_dims=(64,),
)
```

它内部其实是 `MultiScenarioContainer(num_domains, lambda: MLP(..., 1))`。

## 4. 主要骨干思路

当前仓库已经没有单独的 `models/backbone/` 目录。常用共享底座与骨干实现主要分布在 `models/msr/` 及其子目录里。

### `sharedbottom`

最基础的共享底座，所有 domain 共用底部表征，再接 domain-specific head。

### `mmoe`

共享专家 + gate 路由。

### `ple` / `ple_v1`

shared/specific experts 的层级式组合。

### `star`

通过参数融合得到 domain-specific 特化表达。

## 5. `MODEL_REGISTRY`

源码在 [`src/rosen/models/msr/__init__.py`](../../../src/rosen/models/msr/__init__.py)。

当前注册名包括：

```text
sharedbottom
sharedbottom_less
sharedbottom_plus
mmoe
ple
ple_v1
star
satrans
m3oe
m3oe_v1
m3oe_v2
m2m
ppnet
epnet
pepnet
crocodile
pareto
hierrec
automtl
riple
hamur
sdsp_ple
sdsp_mmoe
ssim
multiemb
```

### `build_model()`

```python
model = build_model(manager, num_domains=5, model_cls="mmoe")
```

也可以直接传类：

```python
model = build_model(manager, num_domains=5, model_cls=MMoEModel)
```

## 6. 主要模型族

### `sharedbottom`

最简单的多场景基线，shared backbone + per-domain tower。

### `mmoe`

门控混合专家，每个 domain 有独立 gate。

### `ple`

分层 shared/specific experts。

### `star`

参数融合式场景建模。

### `satrans`

SATrans 风格的场景自适应特征交互模型。实现位于
`src/rosen/models/msr/satrans/`，核心逻辑是将 Rosen 的 field-wise
embedding 组织为 `[batch, field, dim]`，再用 domain embedding 生成 MetaNet
参数去调制 self-attention 的 Q/K/V 侧交互。默认 `meta_mode="QK"`，
`head_type="shared_linear"`，更接近原 SATrans 的共享线性读出；也支持
`head_type="domain_tower"` 作为 Rosen 风格的多场景读出增强。

### `m3oe`

合并门控混合专家，有 `m3oe_v1` / `m3oe_v2` 变体。

### `m2m`

Transformer + meta gate 风格。

### `ppnet` / `epnet` / `pepnet`

PEPNet 家族的三个入口名。实现统一在 `src/rosen/models/msr/pepnet/` 下。

### `crocodile`

多场景模型族的一部分，已注册在 `MODEL_REGISTRY` 中。

### `pareto`

与 Pareto / 多目标优化相关的模型入口。

### `hierrec`

层次化场景建模。

### `automtl`

自动化多任务/多场景架构搜索。

### `riple`

共享/专属专家加辅助约束的多场景模型。

### `hamur`

超网络适配器风格的多场景建模。

### `sdsp_ple` / `sdsp_mmoe`

带 domain-specific decomposition 思路的多场景模型入口。

### `ssim`

场景共享实例建模方向的多场景模型入口。

### `multiemb`

多 embedding 路径的多场景模型入口。

## 7. 模型工具

源码在 [`src/rosen/models/utils`](../../../src/rosen/models/utils)。

### `MultiScenarioContainer`

给每个 domain 一份独立子网络，前向时按 `domain_ids` 选择对应输出。

### `MultiTaskContainer`

给每个 task 一份独立子网络，返回字典输出。

### `MLP`

当前推荐使用的通用 MLP 实现。

### `DNN`

偏 DeepCTR 风格的兼容版本。

### `FeatureBifurcator` / `BifurcatedMLP`

用于把输出拆成“bias + fluctuation”两路。

## 8. 使用层面的真实建议

- 新 MSR 模型优先继承 `MSRModel`。
- `predict()` 和 `calculate_loss()` 要和 `BaseTrainer` 的默认评估逻辑对齐。
- 大多数模型默认输出 raw logits，而不是 sigmoid 后概率。
- 如果模型使用序列特征，通常需要显式从 `omni_embedding.seq_groups[...]` 取，不要期待 `whole` 自动帮你展开序列语义。
