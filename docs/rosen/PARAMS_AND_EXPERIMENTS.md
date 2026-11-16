# 参数与实验入口

这篇文档面向“主要任务是调参、批量实验、给模型透传私有参数”的读者。

重点不是类结构，而是参数到底怎么流动。

## 1. 先记住一条规则

`ParamManager` 当前的真实优先级是：

`命令行已知字段 > build(...) 已知字段 > dataclass 默认值`

同时：

- 未知字段不会报错
- 未知 `build(...)` 参数会进入 `cfg.extras`
- 未知命令行参数也会进入 `cfg.extras`

因此，`cfg.extras` 就是当前框架里承载“模型私有参数”的标准位置。

## 2. 参数来源有哪几层

### dataclass 默认值

这是最底层的初始值：

```python
@dataclass
class DemoConfig(ConfigBase):
    model: str = "mmoe"
    batch_size: int = 1024
```

### `build(...)` 传入参数

适合脚本内部覆盖默认值：

```python
cfg = ParamManager(DemoConfig).build(model="ple", batch_size=2048)
```

### 命令行已知字段

如果字段已经在 dataclass 里声明，CLI 会覆盖上面两层：

```bash
python run.py --model star --batch_size 4096
```

### `extras`

如果参数不在 dataclass 里，它会进入 `cfg.extras`。

这正是模型私有参数的主要入口。

## 3. 未知命令行参数现在怎么解析

当前支持三种常见写法：

```bash
python run.py --warmup_samples 8192
python run.py --dropout_rate=0.1
python run.py --specific_quantized_fusion
```

对应结果大致是：

```python
cfg.extras == {
    "warmup_samples": 8192,
    "dropout_rate": 0.1,
    "specific_quantized_fusion": True,
}
```

解析规则是：

- `--k v` -> `extras["k"] = parsed(v)`
- `--k=v` -> `extras["k"] = parsed(v)`
- `--flag` -> `extras["flag"] = True`
- `true/false/null/none` 会转成 `True/False/None`
- 其余值优先尝试 `ast.literal_eval`
- 解析失败则保留为字符串

同时，CLI 里的同名 extra 会覆盖 `build(...)` 传进来的 extra。

## 4. 一个完整的覆盖例子

```python
cfg = ParamManager(DemoConfig).build(
    model="shavq_h2_ste_v1",
    warmup_samples=4096,
    residual_diversity_weight=0.0,
)
```

如果命令行再传：

```bash
python run.py --warmup_samples 8192 --residual_diversity_weight 0.01
```

最终结果会是：

```python
cfg.model == "shavq_h2_ste_v1"
cfg.extras["warmup_samples"] == 8192
cfg.extras["residual_diversity_weight"] == 0.01
```

也就是：

`CLI extra > build(...) extra`

## 5. 推荐的模型构造写法

如果你的脚本没有模型别名默认参数，最直接的写法就是：

```python
model = build_model(manager, num_domains, cfg.model, **cfg.extras)
```

如果脚本里还维护了一份 `MODEL_KWARGS` 作为“按 model name 选择的代码默认值”，推荐这样写：

```python
model_kwargs = MODEL_KWARGS.get(cfg.model, {}).copy()
model_kwargs.update(cfg.extras)
model = build_model(manager, num_domains, cfg.model, **model_kwargs)
```

这条规则等价于：

`CLI extra > build() extra > MODEL_KWARGS 默认值`

## 6. 为什么不强制引入 `model_kwargs.xxx`

当前框架的倾向是保持扁平 CLI：

```bash
python run.py --model shavq_h2_ste_v1 --warmup_samples 8192
```

而不是强制写成：

```bash
python run.py --model_kwargs.warmup_samples 8192
```

原因是当前实验入口以“单文件清晰可读”为主，模型私有参数本质上就是实验脚本的一部分，用 `extras` 透传足够直接，也更符合现有工程习惯。

## 7. `GridSearchEngine` 怎么和这套参数系统配合

`GridSearchEngine` 的行为很简单：它会把参数字典转成 CLI：

```python
{
    "model": "ple",
    "warmup_samples": 8192,
}
```

会变成：

```bash
python your_script.py --model ple --warmup_samples 8192
```

因此它天然支持 unknown extras。

例如：

```python
engine.run(
    param_space={
        "model": ["shavq_h2_ste_v1", "shavq_h2_ste_v1_r001_fuse"],
        "warmup_samples": [4096, 8192],
    },
    available_gpus=[0, 1],
)
```

## 8. 哪些参数应该放在 dataclass，哪些应该留在 `extras`

更适合放 dataclass 的：

- batch size
- learning rate
- seed
- dataset name
- model name
- train log interval
- 任何你认为“对很多脚本都稳定存在”的字段

更适合留在 `extras` 的：

- 某个模型变体独有的开关
- 还在快速试验中的局部超参
- 只对个别实验脚本有意义的参数

## 9. 常见误解

### 误解一：未知 CLI 参数会被丢掉

当前不会。它们会进入 `cfg.extras`。

### 误解二：`cfg.some_key` 一定意味着这是 dataclass 字段

不一定。`ConfigBase.__getattr__` 会去 `extras` 里找未声明字段，所以：

```python
cfg.warmup_samples
```

可能只是 `cfg.extras["warmup_samples"]` 的代理访问。

### 误解三：参数框架会自动判断哪些 extra 该给模型

不会。当前推荐做法是脚本自己决定：

- 直接 `**cfg.extras`
- 或者 `MODEL_KWARGS + cfg.extras`

下游 `MSRModel.from_manager()` 会根据构造函数签名过滤不匹配参数。

## 10. 继续阅读

- 想看源码层入口：[`details/EXPERIMENT.md`](details/EXPERIMENT.md)
- 想看真实脚本如何落地：[`@examples/kuairand-1k/kuairan1k_top5.py`](../../@examples/kuairand-1k/kuairan1k_top5.py)
