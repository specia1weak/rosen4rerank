# LC_MGE 主模型与训练/推理链路补全

## 背景与定位

用户指定以 `references/rerank/参考代码/LC_MGE_code` 为对照补全 LC_MGE 主模型及使用链路。
本次不迁移该目录内其他基线。

已有 MultiGE 框架实现，但生成候选进入 OCPM 的 dense 特征统一取原始先验，遗漏参考 LC
主实现传入生成器输出分数的路径；BCE 缺少非零权重个数分母；独立推理未保存构造配置。

## 设计与影响范围

- `models/rerank/mge.py` 增加显式 `candidate_dense_mode`，`generator` 保留参考记录坐标、
  对分数 detach，与候选轴同步分块。其余候选族继续使用原先验。
- `base.py` 增加 `nonzero_mean` BCE，保持已有两种 reduction 的含义。
- 工厂增加 `lc_mge` 预设，复用同一模型类；`mge` 默认与参数结构保留。
- KR 实验代码仍放 `workspace/kr_time_budget/pipeline/`。训练保存模型构造配置，新增
  从配置、schema、checkpoint 恢复的无标签 parquet 推理入口；测试放对应 `tests/`。

显式选择 `generator` 可能改变最终排列，历史结果必须记录该选项。既有 checkpoint 的
state_dict 结构不变，但恢复时须正确传入行为配置。旧 run 没有新增配置文件，独立推理
入口要求新流程保存的 `model_config.json`，不猜测旧 run 的参数。无公共 Trainer、数据
默认行为或 MSR 注册表修改；无需重新生成旧数据。

## 验证

- 专项 `uv run pytest -q tests/rosen/models/rerank/test_mge_rerank.py workspace/kr_time_budget/tests/test_architecture_examples.py`：39 passed。
- `uv run pytest -q`：486 passed，512 warnings；无失败。
- 隔离的真实 KR partial 流程：train/valid/test 各 32 请求，重排完整 1 epoch、4 step；
  训练→valid 选择→恢复 checkpoint→test→32 排列导出成功。独立无标签推理逐值恢复 scores/order。
- 日志：[专项](../../../../logs/lc_mge_completion/targeted.log)、[全量](../../../../logs/lc_mge_completion/pytest.log)。

生成器分数、原先验与历史样本的对齐使用数值对照验证；保持既有初始化顺序、残差结构和
共享参数，新增路径无可学习参数，不改变优化器训练协议。

## 后续与文档

本次验证是结构移植和运行验证。TF 权重转换、跨框架数值等价、Avito 复现实验尚未执行；
离散 LC 无梯度的性质与修正后的共享 OCPM head 均已明示。
说明见 [RERANK](../../details/RERANK.md)，实际数据、切分、训练/评估口径、命令与运行日志见
[LC_MGE补全](../../../../workspace/kr_time_budget/回复/新架构示例/LC_MGE补全.md)。
