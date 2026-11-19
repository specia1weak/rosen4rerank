# Step 4：MultiGE 重排模型与独立组件

## 背景与定位

Step 1–3 已提供字段堆叠、列表监督与显式多头评估路由；本步增加具体模型层能力。
将多生成器、排列操作和共享注意力放在 `models/rerank`，不让重排任务进入通用数据、
embedding、Trainer 或 Evaluator 的条件分支。本步没有继续改动这些公共模块。

参考仓库内 `research/projects/LC_MGE_code/rerank_model_dir/` 的三个 TF model.py，
实现一个参数化的 `MultiGERerankModel`，避免三份同构代码。

## 设计与文件

- `models/rerank/base.py`：RerankModel 与四个公共张量算子、签名内省工厂。
- `components/attention.py`：向量化 OAM、IntraPOI、样本对齐的 PageTargetAttention。
- `components/retrieval.py`：位置池化/逐元素 sign 距离、穷举曝光率粗筛；常量均属 buffer。
- `components/generators.py`：独立编码器、Pointwise/Listwise/AR、共享 PointerHead、
  detached Cosine 与离散/可微 LC 统计。
- `mge.py`：四种模式、同一 OCPM 原列表与候选打分、分块、辅助目标及 LossPort 账本。
- `rerank/__init__.py`：独立 RERANK_MODEL_REGISTRY，`mge` 入口与 build_rerank_model。

共享 attention 仅注册一份，生成器只 detach 输入而保留共享网络的梯度。
只有 evaluator 目标（含可选 contrastive 的历史兴趣路径）能更新 embedding。
新可选模块置于共有模块之后初始化，buffer 使用局部 RNG。
没有新增模型自定义评估协议，模型返回 Tensor Mapping 交给既有注册路由。

## 明确的口径与参考实现差异

1. 默认 masked BCE mean 除 B*K，valid_mean 除权重和。TF 默认除非零权重个数，
   二值 mask 时才与 valid_mean 一致；没有把任务单中的 mean 误称为 TF 等价。
2. LC 保留原始索引比较关系矩阵的六对差异总数，离散、无梯度；lc_soft 使用
   expected-position 软关系平方差，是另外一个有梯度的实验目标。
3. Teacher forcing 固定选择 t，但记录各步 argmax 可以重复；预测固定 greedy，输出合法排列。
4. 按 B/M 广播或 repeat_interleave 修正历史错位；统一用 K 初始化解码集合，允许 P!=K。
   全无效历史兴趣返回零。
5. 候选复用同一个已训练 OCPM 输出 head，不保留参考代码单独命名的 top-k head。
6. 保留 Pointwise 的历史坐标约定：先对 SimHash top-one 的先验打分，再把排序位置
   直接应用原列表。候选有效 mask 随排列移动，屏均分分母固定 K。
7. PyTorch GRUCell、二维 Xavier 与框架 embedding 初始化不等价于 TF1 参数/门控/
   预训练恢复流程。没有完成 TF 权重转换或跨框架数值复刻。

## 验证

新增 `tests/rosen/models/rerank/test_mge_rerank.py`，25 个参数化用例：

- 非连续历史折叠、全排列上限、任意尾部 gather 数值和梯度、masked BCE 两种分母及缺失标签。
- 显式属性/POI 循环公式核对 OAM 残差与投影，二维参数初始化及 attention delta/base RMS 检查。
- B>1/M>1 历史页对齐和全无效页；PPML/逐元素 hash 距离、曝光率排序逐排列对照。
- LC 六对关系矩阵与 Python 双循环数值一致（含训练重复索引），Cosine 数值和 detach，
  lc_soft 对概率及第四生成器参数具有非零梯度。
- 四种模式的合法推理排列、输出概率和尺寸、最优生成器选择；训练重复记录与逐步损失数值。
- 所有生成器损失均不能更新 embedding 或先验输入，能更新共享 OAM；evaluator 能更新 embedding。
- 分块 OCPM 与单样本单排列计算一致，改变 chunk 大小结果一致；共有初始化完全一致。
- 无标签预测、关闭全部排列打分、工厂告警；buffer 保存和参数不重复注册。
- K=5、P=2 的合成 Interaction DataLoader，BaseTrainer/EMATrainer 全流程，
  显式 Point-wise 与 Slate 路由、LossPort 子项求和、零 LR 稳定早停、随后非零 LR 真实更新，
  checkpoint 参数/buffer/预测恢复及 EMA deepcopy 引用对齐。

`uv run pytest -q`：**295 passed，381 warnings，5.43 秒**，包含此前 270 个测试。
专项 **25 passed**；compileall 与 `git diff --check` 通过。
本地日志：[专项](../../../../logs/mge_rerank/targeted.log)、[全量](../../../../logs/mge_rerank/pytest.log)。

## 影响与后续

既有 MSR 入口、数据格式和训练流程不变，没有历史数据迁移或重训要求。
新增模型的先验分/历史页 mask 需由 schema 或 Dataset extra columns 明确读取；
候选收益依赖模型估计，不是线上点击率。全排列最多 K=7，关闭 ctr_all 也不取消阶乘粗筛。
后续可独立替换 retrieval、生成器奖励或候选 mask/归一化，不需改变 Trainer 契约。

这里只做合成数据的结构和训练闭环验证，没有跑真实 Avito 数据或报告论文效果。
公共契约与完整使用说明见 [RERANK](../../details/RERANK.md)。
