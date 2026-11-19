# rosen 源码与 KR1K 示例阅读包

四份导出文件可分别发送，也可一起阅读：

| 文件 | 内容 |
| --- | --- |
| [rosen-core.md](rosen-core.md) | 当前框架核心链路，含 stack、SlateLabel、共享拟合、Prediction/EvalBatch、完整 MSR、unified 与 rerank 子包 |
| [kr1k-multitask.md](kr1k-multitask.md) | 新示例一：逐曝光 CTR + 长播双任务，含模型、数据准备、训练、评估及专项测试 |
| [kr1k-rerank.md](kr1k-rerank.md) | 新示例二：六视频一屏、两屏历史的 MultiGE 长播重排，含完整模型组件与实验链路 |
| [kr1k-original.md](kr1k-original.md) | 原始 KR1K 示例 `docs/rosen/details/kuairand1k.py`，原文保留，附数据路径适配器 |

入选文件完整保留，不删除注释、空行或函数体，不做代码压缩。这些是阅读包，不是可单独运行的源码发行包；运行时使用仓库中的原始路径。两个新示例共用准备和训练模块，两个附件都保留这些完整文件，任务阅读入口分别见包内导读。

## 数据怎么走

```text
实验脚本读取、join、过滤原始表（Polars LazyFrame）
  -> 显式切分 train / valid / test，或 SchemaManager.split_dataset()
  -> SchemaManager.fit(train_raw)：拟合词表、数值统计量，保存 schema
  -> SchemaManager.transform(split)：按 schema 编码各 split
  -> SchemaManager.save_as_dataset()：写入 parquet
  -> ParquetStreamDataset：扫描、worker 分片、可选 shuffle、组 batch
  -> TensorFormatter：按字段 formatter 转 tensor、截断和 padding
  -> Interaction：字段名 -> batch tensor
  -> OmniEmbLayer：按字段/source/序列组取 embedding
  -> 模型 calculate_loss() / predict()
  -> Trainer：更新参数、评估、早停、checkpoint
```

原始数据 join、过滤、切分规则和特征选择由实验脚本明确决定。`EmbSetting` 子类定义每个字段的 fit、transform、tensor formatter 和 embedding 取值方式；`SchemaManager` 负责组织这些规则。

`fit()` 发现 `schema_dir/feature_meta.json` 已存在时直接加载。schema 保存各 setting 状态，词表另存到相对路径 `vocab/*.parquet`。共享同一个原子 setting 的序列只拟合一次，以 settings 顺序中第一来源为准。`fields()` 汇总 setting 输出列，以及时间、label、mask、domain 字段；dataset 还支持通过 `extra_col_names` / `extra_col_formatters` 保留额外列。

标签通过字符串或 `SlateLabel` 声明，`label_field` / `label_fields` 始终返回字符串。列表标签与 mask 由 `TensorFormatter` 按声明截断、padding；它们不进入特征序列组。

已编码 parquet 走 `ParquetStreamDataset`；原始 parquet 需要边读边编码时走 `RawParquetStreamDataset`。两者已经产出 batch，外层 `DataLoader` 使用 `batch_size=None`。验证、测试通常设置 `shuffle=False, drop_last=False`，避免丢失尾批。

profile 路径由 `ProfileDatasetBundle` 读取 manifest、schema 和 split。`ProfileParquetStreamDataset` 通过 `ProfileSequenceStore` 按 owner 和 cursor 取历史前缀，再按时间规则过滤、按 max_len 截断，最后格式化为 `Interaction`。这条路径复用侧表中的历史，避免每行重复存储整段历史。

## batch 之后怎么走

`BaseModel` 持有 manager、标签属性、`OmniEmbLayer`、`LossPort` 和 `ObservationPort`。`OmniEmbLayer.whole` 汇合普通用户/物品/交互特征；序列通过 `seq_groups` 的视图单独读取，`split_by="stack"` 保留字段轴 `[B, ..., F, D]`。`SharedBottomModel` 是按 domain 选 tower 的多场景例子；所有内置 MSR/unified 子类统一提供 logits(inter)，基类 predict 返回 sigmoid 概率 Tensor，无需每个模型构造 Prediction；KR1K 多任务例子则共享 backbone，独立预测 CTR 与长播，两者语义不同。

`BaseTrainer` 的默认 step 是 `zero_grad -> calculate_loss -> lsp.settle -> backward -> clip_grad_norm -> optimizer.step`。模型实现 `CustomTrainStepProtocol.custom_train_step(batch, ctx)` 时接管该 step 的优化过程，并负责结算 loss、返回标量 Tensor。epoch/eval 周期方法由 `BaseTrainerWithLife` 解释；EMA 和两阶段运行也有独立实现。

默认评估调用实际模型的 `predict()`，将裸 Tensor 包装成 `Prediction(scores=..., target=model.LABEL)`，
显式 Prediction 则原样保留，覆盖 EMA teacher。Manager 自动绑定 schema，`register(target=...)`
或 Prediction.target 同时确定标签与同名预测，并查询 SlateLabel mask。多标签未指定目标时明确报错。
PointwiseEvaluator 展平有效曝光；RankingEvaluator 按候选分数或最终排列评价排序，SlateValueEvaluator 评价已对齐的整刷价值。

所有 evaluator 统一继承 BaseEvaluator，实现 `collect(EvalBatch)`、`summary()`、`clear()`。
EvalContext 保存已过滤的 Prediction/Interaction，BatchResolver 通过 Scorer 解析成只有 Tensor 的 EvalBatch。
KR1K 根据 outputs_candidate_pool 能力使用 BatchResolver.for_candidates() 读取 CandidatePool，
对齐标签/mask 后交给 SlateValueEvaluator。chosen_scores 为一等字段，导出统一使用 resolve_chosen_scores()；extra 仅作诊断。
RetrievalEvaluator 消费 full_scores，并在副本上屏蔽历史；默认 Trainer 不自动调用 full_sort_predict。
旧字典输出与 collect_batch 路由需迁移；自定义评价器统一继承 BaseEvaluator，实现 collect(EvalBatch)。
接口只提供规范名；旧类名、targets/target_scorer 参数及 candidates 视图已删除。
Scorer 自己声明 schema/target 依赖，同一 collect 复用过滤和 CPU 搬运；评价器按 batch 缓存数组。
裁判统一走 SlateRerankModel 的预编码/分块路径，候选 mask 显式传入；OCPM 历史页和 KuaiShou 历史 K/V 在分块前预编码，Interaction 不再注入控制字段。

checkpoint 保存模型、主 optimizer、epoch、global_step，以及模型通过 `TrainerStateProtocol` 提供的额外训练状态。`run()` 根据验证指标执行早停和 best 保存；不会自动重载 best 后测试，需要入口显式安排。

## 源码阅读顺序

下列路径均为仓库相对路径，可在导出文件中搜索对应的 `## File:`：

1. `src/rosen/emb/manager.py`、`emb/schema/*.py`、`emb/split.py`：字段规则、编码、切分、持久化。
2. `src/rosen/data/dataset.py`、`data/padding.py`、`data/profile.py`、`core/interaction.py`：parquet 到 batch，以及 profile 历史装配。
3. `src/rosen/emb/emblayer.py`、`models/base.py`、`models/msr/sharedbottom.py`：batch 到 embedding 和模型输出。
4. `src/rosen/core/trainers/base.py`、`core/train/{context,hooks,loss}.py`：优化步骤和扩展契约。
5. `src/rosen/core/prediction.py`、`src/rosen/evaluate/{batch,manager,evaluator,metrics}.py`：语义预测、统一评估上下文与指标。
6. `src/rosen/experiment/{config,param}.py`：工作目录、配置和 CLI 参数。
7. `src/rosen/models/rerank/{base,mge}.py` 与 `components/*.py`：列表算子、共享注意力、粗筛、生成器、OCPM 与梯度边界。

核心包附完整 MSR、unified 与 rerank 子包、公共 logits/predict 契约及相关回归测试。观测附 BaseModel 使用的 port 及直接消费 logits 的审计路径。KR1K 实验代码放在独立附件，数据、日志、checkpoint 不打包。

## 三个示例怎么读

- [多任务导读](examples/kr1k-multitask.md)：先看 `pointwise_manager`、`KRPointwiseMultiTaskModel`，再看 `evaluators("pointwise")` 与 `train_example`。
- [重排导读](examples/kr1k-rerank.md)：先看 `rerank_manager`、`make_reranker`、`group_slates`，再看 MultiGE、RankingEvaluator 与 SlateValueEvaluator。
- [原始示例导读](examples/kr1k-original.md)：原始单标签、多场景脚本保持原样，历史切分用法按实际代码说明。

历史两个新示例已各跑完整训练集 1 轮、完整 valid/test；早前 Prediction 迁移以原 checkpoint 复测完整 test，详情见附件报告。最终排列 NDCG@6 低于冻结先验，未取得排序收益。本次候选池类型化通过完整测试和 11 类候选模型的同权重合成 batch 逐值核对，未重跑完整数据集实验；候选预测均分不代表真实反馈收益。

## 重新生成

在仓库根目录运行（本次使用 Repomix 1.18.1）：

```bash
uv run python docs/rosen/repomix/rebuild.py
```

需要本机 `repomix` CLI（本次 1.18.1）。脚本按四份配置重新生成，逐文件核对代码块与源文件内容，并写入 [manifest.json](manifest.json) 记录文件清单、字节数和 SHA-256。只生成核心包仍可运行 `repomix --config docs/rosen/repomix/repomix.config.json`。

选取范围以本目录 `repomix*.config.json` 为准。Repomix 使用 Markdown 代码围栏包装完整文件，可能规范化文件首尾空白；文件内部的注释、空行和代码保持原样。修改源码后重新生成，不直接编辑生成的四份阅读包。
