# 贡献一个模型

这篇文档面向“想新增模型、做模型变体、判断代码该落在哪一层”的读者。

如果你只想看“怎样把一个核心 MSR 模型接入 `MODEL_REGISTRY`”，可以继续读 [`ADD_MSR_MODEL_AND_VERIFY.md`](ADD_MSR_MODEL_AND_VERIFY.md)。这篇文档先帮你判断应该走哪条路。

## 1. 先判断模型该放哪里

当前仓库里有两条常见路径：

### 路径 A：放进 `src/rosen/models/msr`

适合：

- 接口稳定
- 对项目外也有复用价值
- 能进入当前 `MSRModel` 抽象体系
- 你愿意维护更长期的公共 API 兼容性

这是“框架核心模型”路径。

### 路径 B：放进 `src/custom_models`

适合：

- 实验性模型
- 某篇论文的多个局部变体
- 只在当前项目数据或任务里有意义
- 需要快速迭代，不想立即收敛成框架公共抽象

这是“项目实验层模型”路径。

当前仓库里这条路径很常见，入口在 [`src/custom_models/__init__.py`](../../src/custom_models/__init__.py)。

## 2. 当前推荐策略

如果你一开始拿不准，优先放 `src/custom_models`。

原因是：

- 实验阶段的模型通常会反复改接口
- 很多变体只是共享同一个模型类，再通过名字或 extra 参数区分
- 过早塞进核心框架，后面更容易把公共抽象弄脏

等模型形态稳定、复用边界清楚之后，再考虑提升进 `rosen/models/msr`。

## 3. 一个最小模型至少要满足什么

无论你放核心层还是实验层，当前多场景模型通常至少要满足：

- 继承 `MSRModel`
- 能从 `Interaction` 里取到 domain 字段
- 实现 `predict()`
- 实现 `calculate_loss()`

最小形状一般类似：

```python
class MyModel(MSRModel):
    def __init__(self, manager, num_domains, hidden_dim=128, **kwargs):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        ...

    def predict(self, interaction):
        ...

    def calculate_loss(self, interaction):
        ...
```

## 4. 模型参数是怎么接进来的

推荐理解是：

- 通用配置放 dataclass 字段
- 模型私有参数从 `cfg.extras` 进来
- 脚本里用 `build_model(..., **cfg.extras)` 或 `MODEL_KWARGS + cfg.extras`

这意味着你在模型构造函数里只需要定义真正关心的参数，其他多余参数会由 `MSRModel.from_manager()` 过滤。

## 5. 如果走核心层路径

核心层的最短流程是：

1. 在 `src/rosen/models/msr/<your_model>/` 下实现模型
2. 在 `src/rosen/models/msr/__init__.py` 中注册进 `MODEL_REGISTRY`
3. 在实验脚本里设置 `cfg.model = "<your_name>"`
4. 通过 `build_model(manager, num_domains, cfg.model, **cfg.extras)` 构造
5. 用一个现有 example 验证能否完整跑通

这一条的细节说明见 [`ADD_MSR_MODEL_AND_VERIFY.md`](ADD_MSR_MODEL_AND_VERIFY.md)。

## 6. 如果走 `custom_models` 路径

当前更常见的流程是：

1. 在 `src/custom_models/<your_model>/` 下实现模型
2. 在 [`src/custom_models/__init__.py`](../../src/custom_models/__init__.py) 加入 `CUSTOM_MODEL_REGISTRY`
3. 在实验脚本里调用：

```python
from custom_models import CUSTOM_MODEL_REGISTRY
from rosen.models.msr import update_register

update_register(**CUSTOM_MODEL_REGISTRY)
```

4. 把 `cfg.model` 设成你的注册名
5. 用现有 example 跑通

这样做的好处是：

- 不污染核心框架默认注册表
- 一个模型类可以挂多个实验别名
- 更适合快速试验变体

## 7. 核心层与实验层的边界怎么判断

更适合提升到核心层的信号：

- 已经被多个脚本复用
- 参数形态基本稳定
- 没有强项目特定假设
- 你愿意为它维护文档和长期兼容性

更适合继续留在实验层的信号：

- 名字还在频繁变化
- 一个类挂很多实验变体别名
- 很依赖当前数据集列定义
- 还在快速试论文想法

## 8. 推荐的验证顺序

不管模型放哪，第一轮验证至少做这些：

1. `build_model(...)` 能找到你的名字并成功实例化
2. 一个 batch 能跑通 `predict()`
3. 一个 batch 能跑通 `calculate_loss()`
4. `trainer.train_epoch()` 能跑
5. `trainer.evaluate_epoch()` 能跑

这几步比一开始就上整套长训练更有效。

## 9. 和参数系统配合时的建议

- 不要急着把每个模型私参都写进 dataclass
- 优先用 `extras` 做透传
- 如果某些参数已经在很多脚本里稳定存在，再考虑提升为 dataclass 字段

这能让模型贡献路径和实验路径保持一致，不会每加一个开关都去改公共配置类。

## 10. 继续阅读

- 核心 MSR 模型接入细节：[`ADD_MSR_MODEL_AND_VERIFY.md`](ADD_MSR_MODEL_AND_VERIFY.md)
- 模型模块总览：[`details/MODELS.md`](details/MODELS.md)
- 参数透传逻辑：[`PARAMS_AND_EXPERIMENTS.md`](PARAMS_AND_EXPERIMENTS.md)
