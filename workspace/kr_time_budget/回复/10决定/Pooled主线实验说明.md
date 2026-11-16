# Argon 10：Pooled概率预测主线

执行`Argon-plans/10决定.md`：主指标为Pooled AUC与LogLoss，辅助指标为按相同`req_idx_raw`分组的Intra-Step GAUC；User-GAUC只保留诊断，不用于Loss或checkpoint选择。

## 本轮改动与固定项

以Argon 9的`flexible_shape`为起点，先验证解除尺度硬截断这一最小改动。分钟K=2混合威布尔留作后续独立架构实验，本轮不同时改变分布族或预算网输出维度。

| 配置 | 分钟尺度 | 屏数尺度 | 其他模型部分 |
| --- | --- | --- | --- |
| capped | exp(clamp(raw,−2,4.5)+1.6) | 同左 | 复现flexible_shape |
| minute_uncapped | exp(raw+1.6) | 保留原截断 | 不变 |
| both_uncapped | exp(raw+1.6) | exp(raw+1.6) | 不变 |
| fusion | 不使用预算路径 | 不使用预算路径 | 全54字段MLP对照 |

解除截断同时去掉原上下界，不偷偷换成更高硬上界。shape仍为`clamp(0.3+softplus(raw),max=5)`；预算MLP仍输出4维。固定分钟步长1分钟、屏数步长1屏，屏数自变量为`req_idx_raw−1`。浮点溢出时直接报错，避免静默截断。

两套独立OmniEmbedding：预算参数网34字段、内容塔48字段，公共背景各自编码。54字段及v3数据预处理不变，不增加动态内容反馈进预算网。最终只优化`BCE(gamma_time*gamma_step*stay,y_continue)`，同一次backward/optimizer.step更新两塔，无内容辅助Loss、预热、冻结或detach。

## 配对实验协议

- seed固定为2026、2027、2028；每seed运行上述4组，共12组。
- KR 1K data/v3：train为20220408–20220506、1,721,745条；valid为20220507、57,872条；test为20220508、48,898条。
- 每组完整train 1epoch、421步，batch4096，无累积，复用相同train物理顺序；valid/test不shuffle。
- AdamW：预算lr=.005，内容/Fusion lr=.001；普通矩阵weight_decay=1e-6，Embedding/bias/一维参数零衰减，无scheduler/EMA，CPU4线程。
- 每seed所有配置的完整state初始化逐tensor相同。内容bias固定3.0；Fusion该未使用内容头的bias不影响其预测。
- 每组step0/106/211/421完整valid，按LogLoss选checkpoint，再做完整test。无User-GAUC选择步骤。
- 独立重载所选checkpoint，核对全部test预测。seed2026的capped/Fusion还须逐值复现已有flexible_shape/Fusion预测。

## 观测与判断

报告三seed均值与样本标准差，以及逐seed配对差值；不把seed2026或test上的最好配置挑出来当总体结论。Intra-Step GAUC按有效组样本数加权，同时报告有效组数与覆盖量。补充Brier、固定20等宽概率桶ECE、总体平均预测偏差、固定屏号桶内真实继续率与平均预测，供概率校准诊断。

记录每个时钟lambda/shape、gamma分布、超过旧上界的比例，以及超过旧上界后是否仍有非零尺度梯度。解除截断不等于分钟时钟一定变得有效；观察无截断时模型最终学到什么。保留真实分钟回退统计，累计S不增，不再要求条件gamma随时钟递减。

代码：[模型](../../pipeline/time_budget_v9.py)、[训练入口](../../pipeline/run_exit_v9_pooled.py)、[测试](../../tests/test_v9.py)。结果统一写入[主线实验报告](Pooled主线实验报告.md)，日志在[runs/exit_v9_pooled/logs/train.log](../../runs/exit_v9_pooled/logs/train.log)。

```bash
uv run python -u -m workspace.kr_time_budget.pipeline.run_exit_v9_pooled
```
