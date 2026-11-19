# 示例一：KR1K Pointwise CTR + 长播

这个附件保留实际跑过的源码，而非抽出的伪代码。阅读顺序：

1. `pipeline/architecture_examples.py`：`pointwise_manager` 声明两个标量标签；`KRPointwiseMultiTaskModel` 共享 backbone、两个独立 head；`Prediction.scores` 直接以标签名为 key，`evaluators("pointwise")` 仅指定 target 配对。
2. `pipeline/prepare_architecture_examples.py`：`flat_events` 复用旧六视频实验的人群、切分及逐视频标签；`prepare_pointwise` 仅 fit train、物理 shuffle train。
3. `pipeline/run_architecture_examples.py`：`train_example("pointwise", ...)` 复用 BaseTrainer，完整训练 1 轮、valid 选择 checkpoint、加载后测完整 test；`export_pointwise` 冻结输出先验给后续重排。
4. `tests/test_architecture_examples.py` 与 `回复/新架构示例/实验报告.md`：语义校验、实际规模、指标及边界。

上述相对路径均位于 `workspace/kr_time_budget/`。共享模块也包含第二个重排示例，是同一次实际运行的完整依赖，未为导出而另写一份模型。

在仓库根目录运行两个示例的共同入口：

```bash
uv run python -m workspace.kr_time_budget.run_architecture_examples
```

输入为已准备好的 `workspace/kr_time_budget/data/v3/` 及 KR1K 视频侧表；原始数据根由 `ROSEN_DATASOURCE_ROOT` 指定。默认输出到新的时间戳目录，不覆盖历史 run。要单独理解第一阶段，沿 `main` 中准备 Pointwise → 训练 → 导出先验三步阅读；该 CLI 会接着跑重排。

本次完整 test：CTR AUC 0.749291，长播 AUC 0.756063。完整训练 10,330,470 曝光，test 293,388 曝光；按会话结束日期切分 train≤2022-05-06、valid=05-07、test=05-08。日志位于 `workspace/kr_time_budget/runs/architecture_full_20261117/logs/train.log`。

更多形状与 API 用法见同包 `回复/新架构示例/用法.md`；框架实现见配套 `rosen-core.md`。

当前接口迁移复测见同包 `Prediction迁移.md`：完整 test 双头分数与原 checkpoint 输出逐值相同。
