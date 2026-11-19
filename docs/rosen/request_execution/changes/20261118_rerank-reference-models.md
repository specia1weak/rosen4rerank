# 参考目录独立 rerank 模型补全

## 背景与问题

此前仅有 `MultiGERerankModel` 和模式预设；用户要求将参考目录其余模型实际补进
`src/rosen/models/rerank`，而非只增加别名或改已有 LC 逻辑。
对 `LC_MGE_code/rerank_model_dir` 中 15 个 model.py 逐类核对，确定独立结构、辅助目标、
候选选择及 metric 版本的真实训练差别。

## 设计与文件

- 独立类与文件：`dnn.py`、`dcn.py`、`prm.py`、`extr.py`、`kuaishou.py`、`pier.py`、
  `prs.py`、`nar.py`、`seq2slate.py`、`pier_submodular.py`、`mge_metric.py`。
- `slate.py`：共有 schema 视图、列表输入、历史折叠、mask、监督损失与 Prediction 契约。
- `components/scoring.py`：候选 raw logits 分块打分；`components/diversity.py`：明确
  定义的每样本多样性统计；QKV/生成器/PPML 继续复用原有组件。
- 注册上述具体类，增加 `mge3`/`mge_cosine` 精确模式预设，保留 `mge`/`lc_mge`。
  工厂沿 MRO 收集各构造层接受的关键字，支持继承式配置并对未知项告警。
- 调用方 KR `--rerank-model` 覆盖所有入口；配置、候选评估器与无标签导出按模型类型
  选择，仍在 `workspace/kr_time_budget`。测试未放入 src。

## 实现取舍与影响

修复 DCN/PRM 错误 reshape、PRS/metric 概率作为 logits 与监督坐标错位、无效 context
进入 attention、PIER 多样性新增候选的最终索引越界。metric 训练采用合法 greedy 候选，
不复制可能重复 item 的 teacher argmax 记录与无效伪 Kendall 计数。
这些修正影响与原 TF 的逐值比较，不能声称严格复制原始 bug。

EXTR 按自然/广告组构造注意力，组可配置；KuaiShou 将单 target+外部 order 上下文适配为
逐槽 target+当前列表，并显式支持历史先验。这些输入组织差异已写入公共说明。
没有新增可学习 token/三维独立 expert 参数；二维 Xavier、残差顺序与 padding 隔离有
针对性检查。既有 MultiGE 初始化顺序、参数布局与默认行为保留。

无公共 Trainer、schema 格式、数据默认处理或 MSR 注册修改；无需迁移已有数据或
checkpoint。新模型需要各自构造配置与训练，不能将旧 MultiGE 权重直接当作它们的权重。
参考源码未训练的独立 top-k head 不保留，所有候选使用已训练 OCPM head。

## 验证与后续

测试覆盖九个基础独立模型及三个 metric/多样性类的真实优化、Trainer、loss 账本、
checkpoint/deepcopy、无标签预测；公式/行为检查包括 cross、permutation equivariance、
跨槽梯度、EXTR 因果与残差、KuaiShou target pool、历史隔离、PIER 候选、PRS 标签、
NAR/Seq2Slate 选择、metric 最佳候选及初始化尺度。

15 个参考入口逐个完成 KR 小批 train/valid/test 各 32 请求、1 epoch、4 optimizer step，
新实例严格加载 checkpoint 后无标签预测的 scores/order 与原导出逐值一致。
最终 `uv run pytest -q`：**519 passed，512 warnings**；compileall、`git diff --check` 及
新增文档相对链接检查通过。日志：[全量](../../../../logs/rerank_reference_models/pytest.log)。
详细训练与评估口径、日志和验收脚本见
[独立模型补全](../../../../workspace/kr_time_budget/回复/新架构示例/独立模型补全.md)。

源码导航：[rerank/README](../../../../src/rosen/models/rerank/README.md)；
完整目录对应、构造参数及修正：[RERANK_BASELINES](../../details/RERANK_BASELINES.md)。
没有运行全量效果比较、TF checkpoint 转换或 Avito 复现；后续效果实验需统一训练协议。
