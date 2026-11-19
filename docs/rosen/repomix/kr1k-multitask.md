This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where content has been formatted for parsing in markdown style.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
- Pay special attention to the Repository Description. These contain important context and guidelines specific to this project.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: docs/rosen/repomix/examples/kr1k-multitask.md, docs/rosen/repomix/repomix.multitask.config.json, docs/rosen/repomix/README.md, workspace/kr_time_budget/run_architecture_examples.py, workspace/kr_time_budget/pipeline/architecture_examples.py, workspace/kr_time_budget/pipeline/prepare_architecture_examples.py, workspace/kr_time_budget/pipeline/run_architecture_examples.py, workspace/kr_time_budget/tests/test_architecture_examples.py, workspace/kr_time_budget/回复/新架构示例/用法.md, workspace/kr_time_budget/回复/新架构示例/实验报告.md, src/rosen/datasets/kuairand.py, workspace/kr_time_budget/回复/新架构示例/Prediction迁移.md, workspace/kr_time_budget/回复/新架构示例/verify_prediction_contract.py
- Files matching these patterns are excluded: **/__pycache__/**, **/*.pyc, **/.DS_Store, docs/rosen/repomix/rosen-core.md, docs/rosen/repomix/kr1k-*.md, docs/rosen/repomix/manifest.json
- Files matching default ignore patterns are excluded
- Content has been formatted for parsing in markdown style

# User Provided Header
新示例一：KR1K Pointwise CTR + 长播双标签多任务。实际运行源码完整保留；共享模块也含后续重排阶段。 先读 docs/rosen/repomix/examples/kr1k-multitask.md。附件为阅读包，运行请使用仓库源文件路径。

# Directory Structure
````
docs/
  rosen/
    repomix/
      examples/
        kr1k-multitask.md
      README.md
      repomix.multitask.config.json
src/
  rosen/
    datasets/
      kuairand.py
workspace/
  kr_time_budget/
    pipeline/
      architecture_examples.py
      prepare_architecture_examples.py
      run_architecture_examples.py
    tests/
      test_architecture_examples.py
    回复/
      新架构示例/
        Prediction迁移.md
        verify_prediction_contract.py
        实验报告.md
        用法.md
    run_architecture_examples.py
````

# Files

## File: docs/rosen/repomix/examples/kr1k-multitask.md
````markdown
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
````

## File: docs/rosen/repomix/README.md
````markdown
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
````

## File: docs/rosen/repomix/repomix.multitask.config.json
````json
{
  "output": {
    "filePath": "docs/rosen/repomix/kr1k-multitask.md",
    "style": "markdown",
    "parsableStyle": true,
    "headerText": "新示例一：KR1K Pointwise CTR + 长播双标签多任务。实际运行源码完整保留；共享模块也含后续重排阶段。 先读 docs/rosen/repomix/examples/kr1k-multitask.md。附件为阅读包，运行请使用仓库源文件路径。",
    "fileSummary": true,
    "directoryStructure": true,
    "removeComments": false,
    "removeEmptyLines": false,
    "compress": false,
    "git": {
      "sortByChanges": false
    }
  },
  "include": [
    "docs/rosen/repomix/examples/kr1k-multitask.md",
    "docs/rosen/repomix/repomix.multitask.config.json",
    "docs/rosen/repomix/README.md",
    "workspace/kr_time_budget/run_architecture_examples.py",
    "workspace/kr_time_budget/pipeline/architecture_examples.py",
    "workspace/kr_time_budget/pipeline/prepare_architecture_examples.py",
    "workspace/kr_time_budget/pipeline/run_architecture_examples.py",
    "workspace/kr_time_budget/tests/test_architecture_examples.py",
    "workspace/kr_time_budget/回复/新架构示例/用法.md",
    "workspace/kr_time_budget/回复/新架构示例/实验报告.md",
    "src/rosen/datasets/kuairand.py",
    "workspace/kr_time_budget/回复/新架构示例/Prediction迁移.md",
    "workspace/kr_time_budget/回复/新架构示例/verify_prediction_contract.py"
  ],
  "ignore": {
    "useGitignore": false,
    "useDotIgnore": false,
    "useDefaultPatterns": true,
    "customPatterns": [
      "**/__pycache__/**",
      "**/*.pyc",
      "**/.DS_Store",
      "docs/rosen/repomix/rosen-core.md",
      "docs/rosen/repomix/kr1k-*.md",
      "docs/rosen/repomix/manifest.json"
    ]
  },
  "security": {
    "enableSecurityCheck": true
  }
}
````

## File: src/rosen/datasets/kuairand.py
````python
import polars as pl
from rosen.datasets.base import DatasetBase
import pprint

class KuaiRandDataset(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "KuaiRand"
    DATA_KUAI_1K = BASE_DIR / "KuaiRand-1K" / "data"
    STD_LOG_FORMER_DATA = DATA_KUAI_1K / "log_standard_4_08_to_4_21_1k.csv"
    STD_LOG_FORMER_DATA_P1 = STD_LOG_FORMER_DATA
    STD_LOG_FORMER_DATA_P2 = DATA_KUAI_1K / "log_standard_4_22_to_5_08_1k.csv"
    RAND_LOG_FORMER_DATA = DATA_KUAI_1K / "log_random_4_22_to_5_08_1k.csv"
    USER_FEATURES = DATA_KUAI_1K / "user_features_1k.csv"
    VIDEO_FEATURES = DATA_KUAI_1K / "video_features_basic_1k.csv"

# ==========================================
# 执行与测试示例
# ==========================================
if __name__ == "__main__":
    # 为了防止路径不存在报错，这里用 try-except 包裹文件读取逻辑
    # 请确保你的实际路径中存在这些 CSV 文件
    from rosen.datasets.overview import get_general_info, get_head_info, get_group_stats
    try:
        # 读取三个文件 (这里假设存在且格式标准)
        df_log = pl.read_csv(KuaiRandDataset.STD_LOG_FORMER_DATA_P2)
        df_user = pl.read_csv(KuaiRandDataset.USER_FEATURES)
        df_video = pl.read_csv(KuaiRandDataset.VIDEO_FEATURES)

        datasets = {
            "Log Standard": df_log,
            "User Features": df_user,
            "Video Features": df_video
        }

        # 1. 测试通用信息打印
        print("=== 一、通用信息统计 ===")
        for name, df in datasets.items():
            info = get_general_info(df)
            print(f"\n[{name}] 数据集统计:")
            # 使用 pprint 打印字典，结构清晰且不会省略数据
            pprint.pprint(info, depth=4, compact=False)

        # 2. 测试分组统计函数 (以 df_log 中的某个假设列为例，例如 'is_click')
        print("\n=== 二、分组统计 ===")
        # 这里假设 df_log 有一个 'is_click' 列，你可以替换为你实际想看的列名
        test_col = "tab"
        group_stats = get_group_stats(df_log, test_col)
        print(f"对 '{test_col}' 的分组统计结果:")
        pprint.pprint(group_stats)

        # 3. 测试 Head(3) 视图
        print("\n=== 三、Head(3) 完整数据及类型视图 ===")
        head_view = get_head_info(df_user)  # 以 User Features 为例
        print("[User Features] 前三行完整数据:")
        pprint.pprint(head_view)

    except FileNotFoundError as e:
        print(f"文件未找到，请检查路径: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
````

## File: workspace/kr_time_budget/pipeline/architecture_examples.py
````python
"""KR1K examples: scalar multi-task supervision and native six-item reranking."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rosen.core.enum_type import FeatureSource
from rosen.data.dataset import ParquetStreamDataset
from rosen.emb import AuxField, SchemaManager, SlateLabel
from rosen.emb.schema import MinMaxDenseSetting, SeqGroupConfig, SequenceSetting, SparseEmbSetting
from rosen.evaluate import BatchResolver, FieldScorer, Prediction, RankingEvaluator, SlateValueEvaluator
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.models.base import BaseModel
from rosen.models.rerank import RERANK_MODEL_REGISTRY, build_rerank_model, resolve_rerank_config


WORKSPACE = Path(__file__).resolve().parents[1]
CATEGORIES = ("user_id", "video_id", "author_id", "tab")
LABELS = ("is_click", "long_view")
POIS, PAGES = 6, 2


def atoms():
    return [SparseEmbSetting(name, FeatureSource.USER_ID if name == "user_id" else FeatureSource.ITEM,
                             embedding_dim=8) for name in CATEGORIES]


def pointwise_manager(path):
    return SchemaManager([*atoms(), MinMaxDenseSetting("duration_s", FeatureSource.ITEM)],
                         path, label_fields=LABELS, aux_fields=[
                             AuxField("event_id", dtype="int64"), AuxField("request_id", dtype="int64"),
                             AuxField("slot", dtype="int64"), AuxField("user_id_raw", dtype="int64"),
                         ])


def rerank_manager(path):
    shared = atoms()
    current = SeqGroupConfig("slate", "slate_len", POIS)
    history = SeqGroupConfig("history", "history_len", PAGES * POIS)
    settings = [SequenceSetting(f"cur_{s.field_name}", s, current) for s in shared]
    settings += [SequenceSetting(f"hist_{s.field_name}", s, history) for s in shared]
    return SchemaManager(settings, path, label_fields=[SlateLabel(name, POIS, "mask") for name in LABELS],
                         aux_fields=[
                             AuxField("request_id", dtype="int64"), AuxField("user_id_raw", dtype="int64"),
                             AuxField("page_mask", max_len=PAGES), AuxField("prior_long_view", max_len=POIS),
                         ])


class KRPointwiseMultiTaskModel(BaseModel):
    """One feature vector per exposure; independent CTR and long-view heads."""
    def __init__(self, manager):
        super().__init__(manager)
        self.view = self.omni_embedding.whole
        self.shared = nn.Sequential(nn.Linear(self.view.embedding_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU())
        self.heads = nn.ModuleDict({"ctr": nn.Linear(32, 1), "long_view": nn.Linear(32, 1)})
        self.apply_xavier_initialization()

    def logits(self, interaction):
        hidden = self.shared(self.view(interaction))
        return {name: head(hidden).squeeze(-1) for name, head in self.heads.items()}

    def calculate_loss(self, interaction):
        logits = self.logits(interaction)
        terms = {"ctr": F.binary_cross_entropy_with_logits(logits["ctr"], interaction["is_click"].float()),
                 "long_view": F.binary_cross_entropy_with_logits(logits["long_view"], interaction["long_view"].float())}
        if self.lsp.active:
            for name, loss in terms.items():
                self.lsp.add(f"task.{name}", loss)
        return sum(terms.values())

    def predict(self, interaction):
        logits = self.logits(interaction)
        return Prediction(scores={"is_click": logits["ctr"].sigmoid(), "long_view": logits["long_view"].sigmoid()})


def reranker_config(*, top_k=4, mode=None, candidate_dense_mode=None, model_name="mge"):
    config = dict(slate_cate_group="slate", hist_cate_group="history", slate_dense_field="prior_long_view",
        page_mask_field="page_mask", label_field="long_view", poi_num=POIS, page_num=PAGES,
        mlp_a=(32, 16), mlp_b=(32, 16),
    )
    if model_name in ("pier", "prs", "pier_submodular", "nar", "seq2slate", "mge", "lc_mge",
                      "mge3", "mge_cosine", "mge_metric", "mge_cosine_metric"):
        config.update(top_k=top_k, eval_all_perms=False, candidate_chunk_size=16)
    config = resolve_rerank_config(model_name, **config)
    config.setdefault("loss_reduction", "valid_mean")
    if model_name in ("mge", "lc_mge", "mge3", "mge_cosine", "mge_metric", "mge_cosine_metric", "nar", "seq2slate"):
        config["hidden_dim"] = 32
        if mode is not None:
            config["submodularity_mode"] = mode
        if candidate_dense_mode is not None:
            config["candidate_dense_mode"] = candidate_dense_mode
        config.setdefault("submodularity_mode", "lc")
        config.setdefault("candidate_dense_mode", "prior")
        config["loss_reduction"] = "nonzero_mean" if config["candidate_dense_mode"] == "generator" else "valid_mean"
    return config


def make_reranker(manager, *, top_k=4, mode=None, candidate_dense_mode=None, model_name="mge"):
    name = model_name
    return build_rerank_model(manager, name, **reranker_config(
        top_k=top_k, mode=mode, candidate_dense_mode=candidate_dense_mode, model_name=name))


def loader(path, manager, batch_size):
    dataset = ParquetStreamDataset(path, manager, batch_size=batch_size, shuffle=False, drop_last=False)
    return DataLoader(dataset, batch_size=None, num_workers=0)


def evaluators(task, log_path=None, *, manager=None, rerank_model="mge"):
    result = EvaluatorManager(manager, log_path=log_path, title=f"KR1K_{task}", uid_field="user_id_raw")
    if task == "pointwise":
        result.register("ctr", PointwiseEvaluator("auc", "logloss", "gauc"), target="is_click")
        result.register("long_view", PointwiseEvaluator("auc", "logloss", "gauc"), target="long_view")
    else:
        result.register("long_view", PointwiseEvaluator("auc", "logloss", "gauc"))
        result.register("slate", RankingEvaluator("ndcg@6", "pcoc"), resolver=BatchResolver(view="score_only"))
        result.register("chosen_order", RankingEvaluator("ndcg@6"))
        result.register("pointwise_prior", RankingEvaluator("ndcg@6"),
                        resolver=BatchResolver(pred_scorer=FieldScorer("prior_long_view"), view="score_only"))
        if RERANK_MODEL_REGISTRY[rerank_model.lower()].outputs_candidate_pool:
            result.register("candidates", SlateValueEvaluator("cand_mean", "cand_max", reduction="mean"),
                            resolver=BatchResolver.for_candidates())
    return result
````

## File: workspace/kr_time_budget/pipeline/prepare_architecture_examples.py
````python
"""Read-only reuse of v3 membership; keep feedback exclusively in supervision."""
import gc
import json
from pathlib import Path

import polars as pl

from rosen.datasets.kuairand import KuaiRandDataset
from rosen.utils.shuffle import write_shuffled_parquet
from .architecture_examples import CATEGORIES, LABELS, PAGES, POIS, pointwise_manager, rerank_manager


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def flat_events(source, video_path, split, limit_requests=None):
    """Exactly the event membership of existing supervised six-item requests."""
    requests = pl.scan_parquet(source / "requests_base.parquet").filter(pl.col("split") == split).select(
        "request_id", "session_id", "req_idx", "tab", "session_n", "session_date", "end_date")
    if limit_requests is not None:
        requests = requests.sort("request_id").head(limit_requests)
    members = pl.scan_parquet(source / "request_events.parquet").select("event_id", "request_id", "event_idx")
    events = pl.scan_parquet(source / "events.parquet").select("event_id", "user_id", "video_id", "duration_ms", *LABELS)
    videos = pl.scan_csv(video_path).select("video_id", "author_id")
    return (members.join(requests, on="request_id", how="inner")
            .join(events, on="event_id", how="inner")
            .join(videos, on="video_id", how="left")
            .with_columns(pl.col("author_id").fill_null(-1), pl.col("user_id").alias("user_id_raw"),
                          ((pl.col("event_idx") - pl.col("session_n") % POIS) % POIS).cast(pl.Int8).alias("slot"),
                          (pl.col("duration_ms").clip(0, 600000) / 1000).cast(pl.Float32).alias("duration_s"))
            .select("event_id", "request_id", "session_id", "req_idx", "session_date", "end_date", "slot", "user_id_raw",
                    *CATEGORIES, "duration_s", *LABELS).sort(["request_id", "slot"]))


def group_slates(events):
    """Only earlier complete pages in the same session; pack valid pages left."""
    grouped = events.sort(["request_id", "slot"]).group_by("request_id", maintain_order=True).agg(
        pl.col("session_id").first(), pl.col("req_idx").first(), pl.col("user_id_raw").first(),
        *[pl.col(name).alias(f"cur_{name}") for name in CATEGORIES],
        *[pl.col(name) for name in LABELS], pl.col("prior_ctr"), pl.col("prior_long_view"),
        pl.col("event_id").alias("event_ids"),
    ).sort(["session_id", "req_idx"])
    previous_count = pl.col("request_id").cum_count().over("session_id") - 1
    return grouped.with_columns(
        *[pl.concat_list([pl.col(f"cur_{name}").shift(offset).over("session_id").fill_null([])
                          for offset in range(PAGES, 0, -1)]).alias(f"hist_{name}") for name in CATEGORIES],
        pl.concat_list([(previous_count > index).cast(pl.Float32) for index in range(PAGES)]).alias("page_mask"),
        pl.lit([1.] * POIS).alias("mask"),
    )


def prepare_pointwise(source, output, video_path=None, limit_requests=None, seed=2026):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info["limit_requests"] != limit_requests or info["seed"] != seed or info["source"] != str(source.resolve()):
            raise ValueError("Dataset contract changed; choose a fresh output directory.")
        return info
    manager = pointwise_manager(output / "schema")
    stats = {}
    video_path = video_path or KuaiRandDataset.VIDEO_FEATURES
    for split in ("train", "valid", "test"):
        raw_path = output / f"{split}.raw.parquet"
        if not raw_path.exists():
            flat_events(source, video_path, split, limit_requests).sink_parquet(raw_path, compression="zstd")
        raw = pl.scan_parquet(raw_path)
        audit = raw.select(pl.len().alias("rows"), pl.col("event_id").n_unique().alias("events"),
                           pl.col("request_id").n_unique().alias("requests"),
                           *[pl.col(name).mean().alias(name) for name in LABELS]).collect().row(0, named=True)
        if audit["rows"] != audit["events"] or audit["rows"] != audit["requests"] * POIS:
            raise ValueError(f"Invalid six-slot event membership: {split} {audit}")
        if split == "train":
            manager.fit(raw)
        encoded = output / f"{split}.ordered.parquet"
        manager.transform(raw).sink_parquet(encoded, compression="zstd")
        if split == "train":
            write_shuffled_parquet(encoded, output / "train.parquet", seed=seed, bucket_count=16)
        stats[split] = audit
        print("POINT_DATA", split, audit, flush=True)
        gc.collect()
    info = dict(source=str(source.resolve()), video_path=str(Path(video_path).resolve()), seed=seed,
                limit_requests=limit_requests, splits=stats, history="earlier complete six-item pages only",
                labels={"ctr": "raw is_click", "long_view": "raw long_view"},
                split="unchanged v3 sessions: train end<=20220506, valid 20220507, test 20220508",
                source_files={p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                              for p in [source / "requests_base.parquet", source / "request_events.parquet", source / "events.parquet"]})
    dump(manifest, info)
    return info


def prepare_rerank(point_data, priors, output, seed=2026):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        raise FileExistsError("Rerank data is tied to a specific trained prior; choose a fresh output.")
    manager = rerank_manager(output / "schema")
    stats = {}
    for split in ("train", "valid", "test"):
        raw = pl.scan_parquet(point_data / f"{split}.raw.parquet")
        scores = pl.scan_parquet(priors / f"{split}.parquet")
        grouped = group_slates(raw.join(scores, on="event_id", how="left", validate="1:1"))
        raw_path = output / f"{split}.raw.parquet"
        grouped.sink_parquet(raw_path, compression="zstd")
        raw = pl.scan_parquet(raw_path)
        audit = raw.select(pl.len().alias("requests"),
                           pl.col("prior_long_view").list.eval(pl.element().is_null()).list.any().sum().alias("missing_priors"),
                           (pl.col("hist_video_id").list.len() / POIS).mean().alias("mean_history_pages")).collect().row(0, named=True)
        expected = json.loads((point_data / "manifest.json").read_text())["splits"][split]["requests"]
        if audit["requests"] != expected or audit["missing_priors"]:
            raise ValueError(f"Rerank rows or priors do not align: {audit}")
        if split == "train":
            manager.fit(raw)
        encoded = output / f"{split}.ordered.parquet"
        manager.transform(raw).sink_parquet(encoded, compression="zstd")
        if split == "train":
            write_shuffled_parquet(encoded, output / "train.parquet", seed=seed, bucket_count=16)
        stats[split] = audit
        print("SLATE_DATA", split, audit, flush=True)
        gc.collect()
    info = dict(source=str(point_data.resolve()), prior_source=str(priors.resolve()), seed=seed, splits=stats,
                poi_num=POIS, page_num=PAGES, primary_label="long_view", prior="frozen pointwise long_view probability",
                history="previous two complete requests in same session; head remainder excluded; valid pages packed left",
                masking="all six current exposures valid; page_mask tracks complete history pages")
    dump(manifest, info)
    return info
````

## File: workspace/kr_time_budget/pipeline/run_architecture_examples.py
````python
"""Explicit full-train -> validation -> checkpoint -> full-test examples."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import polars as pl
import pyarrow.parquet as pq
import torch

from rosen.core.train import EarlyStopper, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer
from rosen.models.rerank import RERANK_MODEL_REGISTRY
from rosen.utils.expr_tensor import ExprTensor
from rosen.utils.optimize import split_model_params_by_decay
from .architecture_examples import (
    WORKSPACE, KRPointwiseMultiTaskModel, evaluators, loader, make_reranker, pointwise_manager, rerank_manager,
    reranker_config,
)
from .prepare_architecture_examples import dump, prepare_pointwise, prepare_rerank


class Tee:
    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream

    def write(self, text):
        self.terminal.write(text)
        self.stream.write(text)
        self.stream.flush()

    def flush(self):
        self.terminal.flush()
        self.stream.flush()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_early_stopper(task, *, patience=2):
    return EarlyStopper(patience=patience,
                        preferred_evaluators=("chosen_order",) if task == "rerank" else ("long_view",),
                        preferred_metrics=("ndcg@6",) if task == "rerank" else ("auc",))


def train_example(task, data, output, args):
    output.mkdir(parents=True)
    manager = (pointwise_manager if task == "pointwise" else rerank_manager)(data / "schema")
    manager.load_schema()
    torch.manual_seed(args.seed)
    model_name = getattr(args, "rerank_model", "mge")
    model = KRPointwiseMultiTaskModel(manager) if task == "pointwise" else make_reranker(
        manager, top_k=args.top_k, mode=args.mode, candidate_dense_mode=args.candidate_dense_mode, model_name=model_name)
    if task == "rerank":
        dump(output / "model_config.json", {
            "model": model_name,
            "schema": str((data / "schema").resolve()),
            "kwargs": reranker_config(top_k=args.top_k, mode=args.mode, candidate_dense_mode=args.candidate_dense_mode,
                                      model_name=model_name),
        })
    batch_size = args.point_batch_size if task == "pointwise" else args.rerank_batch_size
    loaders = {split: loader(data / ("train.parquet" if split == "train" else f"{split}.ordered.parquet"),
                             manager, batch_size) for split in ("train", "valid", "test")}
    metadata = json.loads((data / "manifest.json").read_text())
    counts = {split: stats["rows" if task == "pointwise" else "requests"] for split, stats in metadata["splits"].items()}
    epochs = getattr(args, "epochs", 1)
    cfg = SimpleNamespace(device=args.device, max_epochs=epochs, train_log_interval=100,
                          ckpt_dir=str(output / "checkpoints"), experiment_name=f"new_architecture_{task}", dataset_name="KR1K_v3_six_item")
    optimizer = torch.optim.AdamW(split_model_params_by_decay(model, weight_decay=1e-6), lr=.001, weight_decay=1e-6)
    stopper = make_early_stopper(task, patience=max(2, epochs + 1))
    evaluator = evaluators(task, output / "evaluation.log", rerank_model=model_name)
    trainer = BaseTrainer(model, optimizer, manager,
                          TrainerDataLoaders(loaders["train"], loaders["valid"], loaders["test"]),
                          TrainerComponents(evaluator_manager=evaluator, early_stopper=stopper), cfg)
    first = next(iter(loaders["train"])).to(args.device)
    shapes = {field: list(first[field].shape) for field in first.columns}
    if task == "rerank":
        cate, hist, dense, pages = model._inputs(first)
        shapes.update(stacked_slate=list(cate.shape), folded_history=list(hist.shape))
    dump(output / "batch_shapes.json", shapes)
    print("SHAPES", task, shapes, flush=True)
    del first
    started = time.monotonic()
    valid = trainer.run()
    train_seconds = time.monotonic() - started
    expected_steps = math.ceil(counts["train"] / batch_size) * epochs
    if trainer.global_step != expected_steps or trainer.epoch != epochs:
        raise RuntimeError(f"Partial training: {trainer.global_step} vs {expected_steps}")
    checkpoint = trainer.best_checkpoint_path
    if checkpoint is None:
        raise RuntimeError("No best checkpoint was saved")
    trained_steps = trainer.global_step
    trainer.load_checkpoint(checkpoint)
    valid = trainer.evaluate_epoch()
    started = time.monotonic()
    test = trainer.evaluate_test()
    test_seconds = time.monotonic() - started
    result = dict(task=task, status="FULL_TRAIN_FULL_TEST" if args.limit_requests is None else "SMOKE_PARTIAL",
                  seed=args.seed, data=str(data.resolve()), split_counts=counts, epochs=epochs,
                  optimizer="AdamW", learning_rate=.001, weight_decay=1e-6,
                  zero_decay="nn.Embedding, biases, norm, scalar/vector and embedding-like free parameters",
                  physical_batch=batch_size, effective_batch=batch_size, gradient_accumulation=1,
                  optimizer_steps=trained_steps, expected_optimizer_steps=expected_steps,
                  device=args.device, threads=args.threads, precision="float32", scheduler=None, ema=False,
                  warm_start=None, selection="max valid chosen_order.ndcg@6" if task == "rerank" else "max valid long_view.auc",
                  checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_hash(checkpoint),
                  train_order_sha256=file_hash(data / "train.parquet"),
                  train_and_valid_seconds=train_seconds, test_seconds=test_seconds, valid=valid, test=test,
                  last_step_loss_ledger=model.lsp.summary(),
                  loss_diagnostics={key: float(value) for key, value in getattr(model, "loss_diagnostics", {}).items()},
                  schema=str((data / "schema").resolve()),
                  rerank_model=model_name if task == "rerank" else None,
                  mode=getattr(model, "submodularity_mode", None) if task == "rerank" else "shared_bottom_two_heads",
                  top_k=getattr(model, "top_k", None) if task == "rerank" else None,
                  candidate_dense_mode=getattr(model, "candidate_dense_mode", None) if task == "rerank" else None,
                  loss_reduction=model.loss_reduction if task == "rerank" else None)
    dump(output / "metrics.json", result)
    print("FINAL_METRICS", json.dumps(result, ensure_ascii=False), flush=True)
    return trainer, loaders, result


@torch.no_grad()
def export_pointwise(trainer, point_data, output, batch_size):
    """Frozen full-train model generates priors. Held-out labels never enter X."""
    output.mkdir(parents=True)
    trainer.model.eval()
    for split in ("train", "valid", "test"):
        batches = loader(point_data / f"{split}.ordered.parquet", trainer.manager, batch_size)
        rows = 0
        with pq.ParquetWriter(output / f"{split}.parquet", schema=pl.DataFrame({
            "event_id": pl.Series([], dtype=pl.Int64), "prior_ctr": pl.Series([], dtype=pl.Float32),
            "prior_long_view": pl.Series([], dtype=pl.Float32)}).to_arrow().schema, compression="zstd") as writer:
            for batch in batches:
                batch = batch.to(trainer.cfg.device)
                predictions = trainer.model.predict(batch)
                frame = pl.DataFrame({"event_id": batch["event_id"].cpu().numpy(),
                                      "prior_ctr": predictions.scores["is_click"].cpu().numpy(),
                                      "prior_long_view": predictions.scores["long_view"].cpu().numpy()}).with_columns(pl.col("event_id").cast(pl.Int64))
                writer.write_table(frame.to_arrow())
                rows += len(batch)
        print("FROZEN_PRIORS", split, rows, flush=True)
    dump(output / "manifest.json", {"point_checkpoint": str(trainer.best_checkpoint_path.resolve()),
                                   "training_priors": "in-sample predictions from full train model; no OOF claim",
                                   "heldout_priors": "frozen train-only model, no label features or further fitting"})


@torch.no_grad()
def export_rerank_test(trainer, batches, output):
    trainer.model.eval()
    writer = None
    rows = 0
    try:
        for batch in batches:
            batch = batch.to(trainer.cfg.device)
            predictions = trainer.model.predict(batch)
            order = predictions.order
            if not torch.equal(order.sort(-1).values, torch.arange(trainer.model.poi_num, device=order.device).expand_as(order)):
                raise RuntimeError("Generated order is not a permutation")
            frame = pl.DataFrame({"request_id": batch["request_id"].cpu().numpy(),
                                  "user_id_raw": batch["user_id_raw"].cpu().numpy(),
                                  "is_click": batch["is_click"].cpu().tolist(), "long_view": batch["long_view"].cpu().tolist(),
                                  "scores": predictions.scores.cpu().tolist(), "out_index": order.cpu().tolist(),
                                  "chosen_scores": predictions.resolve_chosen_scores().cpu().tolist(),
                                  "prior_long_view": batch["prior_long_view"].cpu().tolist()})
            if writer is None:
                writer = pq.ParquetWriter(output, frame.to_arrow().schema, compression="zstd")
            writer.write_table(frame.to_arrow())
            rows += len(batch)
    finally:
        if writer is not None:
            writer.close()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=WORKSPACE / "data/v3")
    parser.add_argument("--data", type=Path, default=WORKSPACE / "data/architecture_examples")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--point-batch-size", type=int, default=8192)
    parser.add_argument("--rerank-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--mode", choices=("none", "cosine", "lc", "lc_soft"), default=None)
    parser.add_argument("--rerank-model", choices=tuple(RERANK_MODEL_REGISTRY), default="mge")
    parser.add_argument("--candidate-dense-mode", choices=("prior", "generator"), default=None,
                        help="generator matches LC_MGE's generated-candidate dense features; prior preserves old runs")
    parser.add_argument("--limit-requests", type=int, help="Only for explicitly marked smoke runs; per split")
    args = parser.parse_args(argv)
    if min(args.threads, args.point_batch_size, args.rerank_batch_size, args.top_k) <= 0:
        parser.error("Thread, batch and retrieval counts must be positive")
    if args.limit_requests is not None and args.data == WORKSPACE / "data/architecture_examples":
        parser.error("Smoke runs require a distinct --data path")
    output = args.output or WORKSPACE / "runs/architecture_examples" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    torch.set_num_threads(args.threads)
    terminal_out, terminal_err = sys.stdout, sys.stderr
    with (output / "logs/train.log").open("w", buffering=1) as stream:
        sys.stdout, sys.stderr = Tee(terminal_out, stream), Tee(terminal_err, stream)
        try:
            print("RUN", vars(args) | {"output": str(output), "torch": torch.__version__, "python": platform.python_version()}, flush=True)
            # A symbolic sanity check records that both labels remain additive
            # objectives; it has no runtime involvement in model training.
            a, b = ExprTensor(torch.tensor(1.), "BCE_CTR"), ExprTensor(torch.tensor(2.), "BCE_LONG_VIEW")
            print("POINTWISE_OBJECTIVE", (a + b).expr, flush=True)
            prepare_pointwise(args.source, args.data / "pointwise", limit_requests=args.limit_requests, seed=args.seed)
            point, _, point_metrics = train_example("pointwise", args.data / "pointwise", output / "pointwise", args)
            export_pointwise(point, args.data / "pointwise", output / "priors", args.point_batch_size)
            del point
            gc.collect()
            slate_data = args.data / f"rerank_{output.name}"
            prepare_rerank(args.data / "pointwise", output / "priors", slate_data, seed=args.seed)
            rerank, batches, rerank_metrics = train_example("rerank", slate_data, output / "rerank", args)
            exported = export_rerank_test(rerank, batches["test"], output / "rerank/test_predictions.parquet")
            if exported != rerank_metrics["split_counts"]["test"]:
                raise RuntimeError("Partial rerank test export")
            dump(output / "summary.json", {"pointwise": point_metrics, "rerank": rerank_metrics,
                                           "source_hashes": {str(p): file_hash(p) for p in [Path(__file__),
                                               Path(__file__).with_name("architecture_examples.py"),
                                               Path(__file__).with_name("prepare_architecture_examples.py")]},
                                           "comparison": "engineering examples; different model/optimizer-step budgets, not a causal A/B",
                                           "data_boundary": "same offline backward six-item v3 grouping; not known simultaneous exposures or online reward"})
        finally:
            sys.stdout, sys.stderr = terminal_out, terminal_err
    print("COMPLETED", output)


if __name__ == "__main__":
    main()
````

## File: workspace/kr_time_budget/tests/test_architecture_examples.py
````python
import json
import math
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.evaluate import BatchResolver, CandidatePool, EvalBatch, Prediction, EvalContext, SlateValueEvaluator
from rosen.data.padding import FeatureContext, TensorFormatter
from rosen.data.dataset import ParquetStreamDataset
from rosen.emb import AuxField, SchemaManager, SlateLabel
from workspace.kr_time_budget.pipeline.architecture_examples import (
    CATEGORIES, KRPointwiseMultiTaskModel, RankingEvaluator, evaluators, make_reranker,
    pointwise_manager, rerank_manager,
)
from workspace.kr_time_budget.pipeline.prepare_architecture_examples import group_slates


def raw_events():
    records = []
    for request, session, index in ((1, 10, 1), (2, 10, 2), (3, 10, 3), (4, 20, 1)):
        for slot in range(6):
            records.append(dict(request_id=request, session_id=session, req_idx=index, slot=slot,
                                event_id=request * 10 + slot, user_id=1, user_id_raw=1,
                                video_id=request * 10 + slot, author_id=slot, tab=1, duration_s=float(slot + 1),
                                is_click=slot % 2, long_view=(slot + 1) % 2,
                                prior_ctr=.1 * slot, prior_long_view=.1 * (5 - slot)))
    return pl.DataFrame(records)


def interaction(frame, manager):
    encoded = manager.transform(frame.lazy()).collect()
    context = FeatureContext.from_manager(manager)
    formatter = TensorFormatter(context)
    return formatter.format({name: encoded[name].to_numpy() for name in context.read_col_names})


def test_six_item_history_is_ordered_packed_and_session_bounded():
    raw = raw_events().sample(fraction=1, shuffle=True, seed=3)
    grouped = group_slates(raw.lazy()).collect().sort("request_id")
    assert grouped["cur_video_id"].to_list() == [[i * 10 + j for j in range(6)] for i in range(1, 5)]
    assert grouped["page_mask"].to_list() == [[0., 0.], [1., 0.], [1., 1.], [0., 0.]]
    assert grouped["hist_video_id"].to_list() == [[], list(range(10, 16)), list(range(10, 16)) + list(range(20, 26)), []]
    assert grouped["is_click"].to_list() == [[0, 1, 0, 1, 0, 1]] * 4
    assert grouped["long_view"].to_list() == [[1, 0, 1, 0, 1, 0]] * 4
    assert all(len(row) == 6 for row in grouped["mask"].to_list())


def test_scalar_multitask_uses_both_labels_and_explicit_routes(tmp_path):
    raw = raw_events()
    manager = pointwise_manager(tmp_path / "point")
    manager.fit(raw.lazy())
    batch = interaction(raw, manager)
    model = KRPointwiseMultiTaskModel(manager)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.heads["ctr"].bias.fill_(math.log(3))
        model.heads["long_view"].bias.fill_(-math.log(3))
    predictions = model.predict(batch)
    torch.testing.assert_close(predictions.scores["is_click"], torch.full((24,), .75))
    torch.testing.assert_close(predictions.scores["long_view"], torch.full((24,), .25))
    assert model.LABELS == ("is_click", "long_view")
    loss = model.calculate_loss(batch)
    expected = -math.log(.75) - math.log(.25)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert model.heads["ctr"].bias.grad.item() > 0
    assert model.heads["long_view"].bias.grad.item() < 0
    evaluator = evaluators("pointwise", manager=manager)
    evaluator.collect(batch, predictions)
    assert set(evaluator.summary()) == {"ctr", "long_view"}


@pytest.mark.parametrize("candidate_dense_mode", ["prior", "generator"])
def test_native_rerank_schema_and_loss_choose_long_view_not_first_label(tmp_path, candidate_dense_mode):
    from workspace.kr_time_budget.pipeline.architecture_examples import loader
    grouped = group_slates(raw_events().lazy()).collect()
    manager = rerank_manager(tmp_path / "schema")
    manager.fit(grouped.lazy())
    encoded = manager.transform(grouped.lazy()).collect()
    path = tmp_path / "encoded.parquet"
    encoded.write_parquet(path)
    batch = next(iter(loader(path, manager, 4)))
    assert set(batch.columns) == set(manager.fields())
    assert batch['request_id'].dtype == batch['user_id_raw'].dtype == torch.int64
    assert manager.get_aux_spec('prior_long_view') == AuxField('prior_long_view', max_len=6)
    assert manager.get_aux_spec('page_mask') == AuxField('page_mask', max_len=2)
    model = make_reranker(manager, candidate_dense_mode=candidate_dense_mode)
    assert manager.label_field == "is_click" and model.LABEL == "long_view"
    assert manager.uid_field is None
    assert manager.mask_fields == ("mask",)
    assert len(model.omni_embedding.emb_modules) == len(CATEGORIES)
    current, history, prior, pages = model._inputs(batch)
    assert current.shape == (4, 6, 4, 8) and history.shape == (4, 2, 6, 4, 8)
    torch.testing.assert_close(prior[..., 0], batch["prior_long_view"])
    logits, _ = model.evaluator(current, model.oam(history)[0], prior, pages, model.oam, model.intra_poi, model.target_attention)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["long_view"])
    torch.testing.assert_close(model.loss_components(batch)["evaluator"], expected)
    model.calculate_loss(batch).backward()
    with torch.no_grad():
        predictions = model.predict(batch)
    evaluator = evaluators("rerank", manager=manager)
    evaluator.collect(batch, predictions)
    results = evaluator.summary()
    assert {name: set(metrics) for name, metrics in results.items()} == {
        "long_view": {"auc", "logloss", "gauc"}, "slate": {"ndcg@6", "pcoc"},
        "chosen_order": {"ndcg@6"}, "pointwise_prior": {"ndcg@6"},
        "candidates": {"cand_mean", "cand_max"},
    }


@pytest.mark.parametrize('width,pages', [(3, 4), (7, 1)])
def test_prediction_reader_follows_schema_names_and_lengths_without_redeclarations(tmp_path, width, pages):
    from workspace.kr_time_budget.pipeline.predict_lc_mge import prediction_batches
    manager = SchemaManager([], tmp_path / 'schema', label_fields=SlateLabel('outcome', width, 'valid_slots'),
                            aux_fields=[AuxField('new_prior', width, padding_side='left', pad_val=-1.),
                                        AuxField('new_pages', pages), AuxField('raw_id', dtype='int64')])
    frame = pl.DataFrame({'outcome': [[1.], [0.]], 'valid_slots': [[1.], [1., 1.]],
                          'new_prior': [[.25], [.5, .75]], 'new_pages': [[1.], []],
                          'raw_id': [2**53 + 1, 2**53 + 2]})
    training = tmp_path / 'training.parquet'
    frame.write_parquet(training)
    inference = tmp_path / 'inference.parquet'
    frame.drop('outcome').write_parquet(inference)
    train, = list(ParquetStreamDataset(training, manager, batch_size=2, shuffle=False, drop_last=False))
    predict, = list(prediction_batches(inference, manager, 2))
    assert set(predict.columns) == set(train.columns) - set(manager.label_fields)
    for field in predict.columns:
        torch.testing.assert_close(predict[field], train[field], rtol=0, atol=0)
    assert predict['new_prior'].shape == (2, width)
    assert predict['new_pages'].shape == (2, pages)
    assert predict['new_prior'][0].tolist() == [-1.] * (width - 1) + [.25]
    assert predict['raw_id'].tolist() == [2**53 + 1, 2**53 + 2]


def test_lc_mge_run_reloads_configuration_and_predicts_without_labels(tmp_path):
    from workspace.kr_time_budget.pipeline.predict_lc_mge import export_predictions, load_reranker, prediction_batches
    from workspace.kr_time_budget.pipeline.run_architecture_examples import train_example

    data = tmp_path / "data"
    data.mkdir()
    raw = group_slates(raw_events().lazy()).collect()
    manager = rerank_manager(data / "schema")
    manager.fit(raw.lazy())
    encoded = manager.transform(raw.lazy()).collect()
    for name in ("train", "valid.ordered", "test.ordered"):
        encoded.write_parquet(data / f"{name}.parquet")
    (data / "manifest.json").write_text(json.dumps({
        "splits": {split: {"requests": 4} for split in ("train", "valid", "test")}}))
    args = SimpleNamespace(seed=2026, top_k=2, mode="lc", candidate_dense_mode="generator",
                           point_batch_size=8, rerank_batch_size=2, device="cpu", threads=1, limit_requests=4)
    run = tmp_path / "run"
    trainer, batches, result = train_example("rerank", data, run, args)
    loaded = load_reranker(run)
    assert loaded.candidate_dense_mode == "generator" and loaded.submodularity_mode == "lc"
    assert loaded.loss_reduction == "nonzero_mean"
    assert result["status"] == "SMOKE_PARTIAL" and result["optimizer_steps"] == 2
    batch = next(iter(batches["test"]))
    with torch.no_grad():
        expected = trainer.model.predict(batch)
        restored = loaded.predict(batch)
    torch.testing.assert_close(restored.scores, expected.scores, rtol=0, atol=0)
    torch.testing.assert_close(restored.order, expected.order, rtol=0, atol=0)
    for key in expected.extra:
        torch.testing.assert_close(restored.extra[key], expected.extra[key], rtol=0, atol=0)
    unlabeled = data / "unlabeled.parquet"
    encoded.drop("is_click", "long_view", "prior_ctr").write_parquet(unlabeled)
    inferred = next(prediction_batches(unlabeled, loaded.manager, 2))
    assert set(inferred.columns) == set(batch.columns) - set(manager.label_fields)
    for field in inferred.columns:
        torch.testing.assert_close(inferred[field], batch[field], rtol=0, atol=0)
    output = tmp_path / "predictions.parquet"
    assert export_predictions(loaded, unlabeled, output, batch_size=2) == 4
    frame = pl.read_parquet(output)
    torch.testing.assert_close(torch.tensor(frame["scores"].to_list()[:2]), expected.scores)
    assert frame["out_index"].to_list()[:2] == expected.order.tolist()
    with pytest.raises(FileExistsError):
        export_predictions(loaded, unlabeled, output)


def test_custom_order_evaluator_measures_selected_order_and_prior_separately():
    labels = torch.tensor([[1., 0., 0., 0., 0., 0.]])
    batch = Interaction({"prior_long_view": torch.tensor([[1., 0., 0., 0., 0., 0.]])})
    predictions = Prediction(order=torch.tensor([[5, 4, 3, 2, 1, 0]]))
    from rosen.evaluate import FieldScorer, RankingEvaluator
    selected, prior = RankingEvaluator("ndcg@6"), RankingEvaluator("ndcg@6")
    batch["label"] = labels
    context = EvalContext(predictions, batch, "label")
    selected.collect(BatchResolver().resolve(context, mask=torch.ones_like(labels)))
    prior.collect(BatchResolver(pred_scorer=FieldScorer("prior_long_view"), view="score_only").resolve(context, mask=torch.ones_like(labels)))
    assert selected.summary()["ndcg@6"] == pytest.approx(1 / math.log2(7))
    assert prior.summary()["ndcg@6"] == 1


def test_custom_candidate_evaluator_transports_mask_with_each_order():
    evaluator = SlateValueEvaluator("cand_mean", "cand_max", reduction="mean")
    # Slot zero is masked. Its high score moves to different candidate positions.
    pred = Prediction(candidates=CandidatePool(scores=torch.tensor([[[99., .2, .4], [.4, 99., .2]]]),
                                               orders=torch.tensor([[[0, 1, 2], [2, 0, 1]]])))
    context = EvalContext(pred, Interaction({"label": torch.zeros(1, 3)}), "label")
    evaluator.collect(BatchResolver.for_candidates().resolve(context, mask=torch.tensor([[0., 1., 1.]])))
    assert evaluator.summary() == pytest.approx({"cand_mean": .3, "cand_max": .3})


@pytest.mark.parametrize('outputs_pool', [False, True])
def test_candidate_evaluator_registration_uses_model_capability_for_new_registry_entries(monkeypatch, outputs_pool):
    from rosen.models.rerank import RERANK_MODEL_REGISTRY, RerankModel

    class NewReranker(RerankModel):
        outputs_candidate_pool = outputs_pool

    monkeypatch.setitem(RERANK_MODEL_REGISTRY, 'new_external_model', NewReranker)
    registry = evaluators('rerank', rerank_model='NEW_EXTERNAL_MODEL').registry
    assert ('candidates' in registry) == outputs_pool
````

## File: workspace/kr_time_budget/回复/新架构示例/Prediction迁移.md
````markdown
# Prediction / EvalBatch 接口迁移验收

本页保留早前字典迁移的历史验收口径。当前 API 以 [用法](用法.md) 为准：主候选为
CandidatePool、选中得分为 chosen_scores，旧 extra 主键和旧评估包装器均已删除；下文旧接口描述只用于还原当时验收。

本次替换预测与评估接口，未重新训练，也未修改特征、监督、参数结构或训练损失。
复用 [原始完整训练](实验报告.md) 的两个最佳 checkpoint，在同一时间切分的完整 test
上运行新接口，逐条对比原先导出的分数和最终排列。

| 任务 | test 规模 | 核对结果 |
| --- | ---: | --- |
| Pointwise CTR + 长播 | 293,388 曝光 | 两个预测 head 均逐值完全相等（rtol=atol=0） |
| MultiGE 长播重排 | 48,898 屏 / 293,388 视频 | 原列表分数、最终排列均逐值完全相等 |

所有原有 test 指标差值小于 1e-12：CTR AUC 0.749291，Pointwise 长播 AUC 0.756063，
重排长播 AUC 0.755150，最终排列 NDCG@6 0.470777。冻结先验排序 NDCG@6 仍为
0.486740，没有新增排序收益。候选均分现在归独立 `candidates` 评估组，名称为
`topk_mean / best_topk_mean`；不作为真实反馈收益。

数据沿用原 v3 六视频分屏：按会话结束日期 train≤2022-05-06、valid=05-07、test=05-08。
原 checkpoint 来自完整训练集各 1 epoch、独立 valid 选择；本次只复测完整 test，FP32/CPU/4线程，
无选参、无训练。此结论是接口数值保持，不是新增模型的结构因果实验。

## 当时的代码变化（历史）

- 模型返回 Prediction：多任务 scores 的 key 就是标签名；MultiGE 声明 target=long_view，
  scores 原槽位 `[B,6]`、order `[B,6]`，候选分数及索引放 extra。
- Manager 接入 schema，register 只需 target，自动解析 label/mask，全局声明 raw UID。
- 内置 OrderEvaluator 计算最终排列 NDCG；SlateValueEvaluator 只保留有监督指标。
- PriorEvaluator / MGECandidateEvaluator 留在实验层，统一 `collect(EvalBatch)`。
- Trainer/EMA 在实际预测返回后包装旧单任务 Tensor，不改写显式 Prediction 的目标。

代码与用法见 [用法.md](用法.md)。旧 `pred_key/label_field/mask_field` 注册、自定义
collect_batch 与直接四参数 collect 需要迁移；旧对象可显式用 LegacyEvaluatorAdapter 包装。
旧单任务模型 Tensor 输出及原始 KR1K 的 register 写法继续可用。

## 验证证据

- [完整 test 复测脚本](verify_prediction_contract.py)
- [逐条对照日志](../../logs/prediction_contract_audit.log)
- [复测指标、checkpoint 与源码哈希](../../runs/prediction_contract_20261117/audit.json)
- [原完整训练 run](../../runs/architecture_full_20261117/summary.json)
- [原入口源码快照](../../runs/architecture_full_20261117/source_snapshot/architecture_examples.py)
- 框架及 workspace 全量回归：303 passed；日志位于仓库 `logs/prediction_contract_regression.log`。

原 run 的训练日志、指标、checkpoint 与预测文件均保留。原 summary 中三份入口的 SHA-256
可对照 `source_snapshot/` 校验；当前入口已迁移，不能再把当前文件当成历史运行时的源码。

复测命令（默认输出目录已存在时，请指定新的 `--output`）：

```bash
uv run python -m workspace.kr_time_budget.回复.新架构示例.verify_prediction_contract \
  --output workspace/kr_time_budget/runs/prediction_contract_recheck
```
````

## File: workspace/kr_time_budget/回复/新架构示例/verify_prediction_contract.py
````python
"""Audit the new output contract against both frozen full-run test predictions.

No training or fitting. Historical results and source snapshots remain untouched.
Run from the repository root:
uv run python -m workspace.kr_time_budget.回复.新架构示例.verify_prediction_contract
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import polars as pl
import torch

from workspace.kr_time_budget.pipeline.architecture_examples import (
    KRPointwiseMultiTaskModel, evaluators, loader, make_reranker, pointwise_manager, rerank_manager,
)

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify(run, output):
    summary = json.loads((run / 'summary.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    result = {'training': 'none; reuse original full-train 1-epoch best checkpoints',
              'source_run': str(run.resolve()), 'tasks': {}}
    for task in ('pointwise', 'rerank'):
        old = summary[task]
        checkpoint = Path(old['checkpoint'])
        assert sha(checkpoint) == old['checkpoint_sha256']
        manager = (pointwise_manager if task == 'pointwise' else rerank_manager)(Path(old['schema']))
        manager.load_schema()
        model = KRPointwiseMultiTaskModel(manager) if task == 'pointwise' else make_reranker(manager, top_k=old['top_k'], mode=old['mode'])
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        del state
        gc.collect()
        model.eval()
        reference = pl.read_parquet(run / ('priors/test.parquet' if task == 'pointwise' else 'rerank/test_predictions.parquet'))
        evaluation = evaluators(task, output / f'{task}.log', manager=manager)
        count, started = 0, time.monotonic()
        with torch.no_grad():
            for inter in loader(Path(old['data']) / 'test.ordered.parquet', manager, old['physical_batch']):
                pred = model.predict(inter)
                expected = reference.slice(count, len(inter))
                id_field = 'event_id' if task == 'pointwise' else 'request_id'
                assert inter[id_field].tolist() == expected[id_field].to_list()
                if task == 'pointwise':
                    for label, prior in (('is_click', 'prior_ctr'), ('long_view', 'prior_long_view')):
                        torch.testing.assert_close(pred.scores[label], torch.tensor(expected[prior].to_list()), rtol=0, atol=0)
                else:
                    torch.testing.assert_close(pred.scores, torch.tensor(expected['scores'].to_list()), rtol=0, atol=0)
                    torch.testing.assert_close(pred.order, torch.tensor(expected['out_index'].to_list()), rtol=0, atol=0)
                evaluation.collect(inter, pred)
                count += len(inter)
        assert count == old['split_counts']['test'] == reference.height
        metrics = evaluation.summary()
        for group, values in old['test'].items():
            for metric, expected in values.items():
                if group == 'slate' and metric in ('topk_ctr', 'best_topk_ctr'):
                    actual = metrics['candidates']['topk_mean' if metric == 'topk_ctr' else 'best_topk_mean']
                else:
                    actual = metrics[group][metric]
                assert abs(actual - expected) < 1e-12, (task, group, metric, actual, expected)
        result['tasks'][task] = dict(rows=count, exact_prediction_match=True,
                                    metrics_match_tolerance=1e-12, metrics=metrics, seconds=time.monotonic()-started,
                                    checkpoint_sha256=old['checkpoint_sha256'])
        print(task, count, 'exact predictions; all metrics match', flush=True)
        del model, manager, evaluation, reference
        gc.collect()
    paths = [Path('src/rosen/core/prediction.py'), *Path('src/rosen/evaluate').glob('*.py'),
             Path('src/rosen/models/rerank/mge.py'), Path('workspace/kr_time_budget/pipeline/architecture_examples.py')]
    result['source_hashes'] = {str(p): sha(p) for p in paths}
    (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('PASS', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=WORKSPACE/'runs/architecture_full_20261117')
    parser.add_argument('--output', type=Path, default=WORKSPACE/'runs/prediction_contract_20261117')
    args = parser.parse_args()
    verify(args.run, args.output)
````

## File: workspace/kr_time_budget/回复/新架构示例/实验报告.md
````markdown
# KR1K 新架构示例：双标签 Pointwise 与六视频重排

本报告记录原始接口的完整训练。当前源码已迁移 Prediction/EvalBatch，见 [接口迁移复测](Prediction迁移.md)；历史入口源码保存在该 run 的 source_snapshot/，原始结果不覆盖。

两个示例均完成完整训练集 1 轮、完整 valid 和完整 test；没有行数截断或训练步上限。
本次是新架构用法与真实数据闭环验收，不替换历史预算实验，也不构成模型结构的配对因果对照。

## 完整 test 指标

| 任务 | AUC | User GAUC | LogLoss |
| --- | ---: | ---: | ---: |
| Pointwise CTR | 0.749291 | 0.602066 | 0.566823 |
| Pointwise 长播 | 0.756063 | 0.625234 | 0.487465 |
| MultiGE 长播 | 0.755150 | 0.638018 | 0.495800 |

| 长播列表口径 | NDCG@6 |
| --- | ---: |
| 冻结 Pointwise 先验排序 | 0.486740 |
| OCPM 对原列表打分后排序 | 0.486105 |
| 最终生成器选定排列 | 0.470777 |

最终排列相对冻结先验的 NDCG@6 差为 **-0.015963**。本轮没有取得重排排序收益，不能把训练闭环跑通等同于重排有效。

最终选定排列的 NDCG 由导出的 48,898 屏标签与 out_index 独立复算一致。其中 31,744 屏至少有一个长播正例；无正例屏按 0 计入全量平均。

候选打分均值 topk_ctr=0.249077，best_topk_ctr=0.256410，CB=0.971038。这里沿用 MultiGE 历史键名 ctr_*，实际监督和概率语义均为 long_view，这些估计均分不是真实重排点击/长播收益。

## 数据与切分

复用旧预算实验 data/v3 的去重事件、request_events 和 requests_base。仍为每用户会话倒切 6 视频，保留原先的会话筛选与 top-5 场景，不重新选择人群或分屏。

| 切分 | 日期口径 | Pointwise 曝光数 | 重排屏数 |
| --- | --- | ---: | ---: |
| train | session end ≤ 2022-05-06 | 10,330,470 | 1,721,745 |
| valid | 2022-05-07 单日会话 | 347,232 | 57,872 |
| test | 2022-05-08 单日会话 | 293,388 | 48,898 |

逐视频 Label 为原始 is_click 与 long_view，不从播放时长另设阈值，也不使用 y_continue。用户/视频/作者/tab 的词表和视频长度归一化仅 fit train。X 不包含当前反馈、观看时长、未来长度或继续标签。

历史只取同一 session 内之前两屏完整 6 视频，按从旧到新拍扁为 12 个 item，有效页靠左打包，page_mask 对齐；会话头部不足 6 个视频的余量不纳入本例历史。这是相对旧预算模型 hist_tags 的显式变化，非旧模型数值等价迁移。

Pointwise 训练完成后冻结 checkpoint，对 train/valid/test 生成先验。train 先验为同训练集模型的in-sample 预测，未做 OOF；valid/test 完全使用冻结模型，不再拟合。原数据三个 parquet 的大小和 mtime 在报告审计时保持不变。

## 新架构怎样接起来

1. Pointwise：BaseModel + 字符串多标签，shared-bottom 两个输出 head；calculate_loss 为两个 BCE 的和，LossPort 分别登记；predict 返回 ctr/long_view 字典，register 用 pred_key/label_field 显式配对。
2. 重排：两个 SlateLabel 共用 mask，四个属性的当前序列/历史序列共享四张原子词表；stack 得到 [B,6,4,8]，模型折叠历史为 [B,2,6,4,8]。使用公共 MultiGERerankModel，label_field 显式选 long_view，即使 manager.label_field 是 is_click 也不依赖 Trainer 拆标签。
3. 标准 Evaluator 接收 mask 展平计算曝光 AUC，SlateValueEvaluator 计算列表指标；OrderEvaluator 实现 collect_batch，直接读取 out_index 计算最终排列 NDCG，模型不写 eval 胶水。

## 训练协议

| 项目 | Pointwise | MultiGE 重排 |
| --- | --- | --- |
| Physical batch | 8192 曝光 | 512 屏（最多 3072 曝光） |
| Optimizer steps | 1262 | 3363 |
| 优化器 | AdamW，lr=0.001 | AdamW，lr=0.001 |
| Weight decay | 普通权重 1e-6，Embedding/bias/norm/vector 为 0 | 相同分组规则 |
| 精度/设备 | FP32 / cpu / 4 线程 | FP32 / cpu / 4 线程 |
| Seed / accumulation / EMA / scheduler | 2026 / 1 / 关闭 / 无 | 2026 / 1 / 关闭 / 无 |
| 训练及完整 valid 用时 | 75.7s | 482.2s |
| 完整 test 用时 | 1.3s | 12.5s |

训练 parquet 使用 seed=2026 的物理 target-sort shuffle，drop_last=False；读取侧不再 shuffle。两者各完整 1 epoch 后由完整 valid long_view.auc 选择唯一 checkpoint，然后加载该 checkpoint 测完整 test。没有 warm-start，重排只把前一模型预测作为冻结输入；测试集不参与选参。

重排 submodularity_mode=lc，hidden=32，MLP_A=(32,16)，MLP_B=(32,16)，top_k=4。全部 720 个排列仍参与粗筛，但 eval_all_perms=False，不导出全排列 OCPM 分数。LC 为离散无梯度统计项，可导致账本总损失为负；不据此判断训练崩溃。BCE 使用 valid_mean；原 TF 的初始化/GRU/独立 top-k head 等差异见框架 RERANK 文档。

两种模型的输入、任务与更新次数不同，上表不支持把 AUC 差归因于某个单一结构。六视频分屏仍是依赖整场长度的离线代理，不是已知同时曝光请求；当前标签仅评价日志内排序，没有反事实曝光校正或在线收益证据。

## 入口与证据

```bash
uv run python -m workspace.kr_time_budget.run_architecture_examples
```

默认完整数据、两任务各 1 epoch；输出创建到独立时间戳目录。已存在的 run 不覆盖。小规模诊断须显式提供 --limit-requests 和独立 --data，结果标为 SMOKE_PARTIAL。

- [模型、Schema 与评估声明](../../pipeline/architecture_examples.py)
- [数据准备与历史对齐](../../pipeline/prepare_architecture_examples.py)
- [训练与导出入口](../../pipeline/run_architecture_examples.py)
- [专项测试](../../tests/test_architecture_examples.py)
- [本报告生成器](write_report.py) · [验收数据](验收数据.json)
- [完整训练日志](../../runs/architecture_full_20261117/logs/train.log)
- [总结果与配置](../../runs/architecture_full_20261117/summary.json)
- [Pointwise 指标](../../runs/architecture_full_20261117/pointwise/metrics.json) · [重排指标](../../runs/architecture_full_20261117/rerank/metrics.json)
- [完整 test 重排预测](../../runs/architecture_full_20261117/rerank/test_predictions.parquet)

复算报告命令：

```bash
uv run python 'workspace/kr_time_budget/回复/新架构示例/write_report.py' --run '/Users/lxogri/Developer/pyprojects/time-budget/workspace/kr_time_budget/runs/architecture_full_20261117'
```
````

## File: workspace/kr_time_budget/回复/新架构示例/用法.md
````markdown
当前接口与历史结果边界见 [2026-11-19 审计修复](审计修复.md)。本页用法已同步类型化候选池。

# 从两个 KR1K 例子看 Prediction / EvalBatch

[模型与评估源码](../../pipeline/architecture_examples.py) ·
[准备数据](../../pipeline/prepare_architecture_examples.py) ·
[完整训练入口](../../run_architecture_examples.py) ·
[接口迁移复测](Prediction迁移.md)

特征归 EmbSetting，监督归 SchemaManager；模型返回 Prediction，评估器收集 EvalBatch。
修改特征或编码规则时，为 `--data` 选择新目录，避免复用旧 schema。

## 1. Pointwise：同一曝光的 CTR 与长播

```python
manager = SchemaManager(
    [*categorical_settings, MinMaxDenseSetting("duration_s", FeatureSource.ITEM)],
    schema_dir=point_schema,
    label_fields=["is_click", "long_view"],
    aux_fields=[AuxField("event_id", dtype="int64"), AuxField("request_id", dtype="int64"),
                AuxField("slot", dtype="int64"), AuxField("user_id_raw", dtype="int64")],
)
model = KRPointwiseMultiTaskModel(manager)
# predict 返回 Prediction(scores={"is_click": Tensor[B], "long_view": Tensor[B]})
# 两项 loss 仍是独立 BCE，通过 LossPort 分别登记。

evaluation = EvaluatorManager(manager, uid_field="user_id_raw")
evaluation.register("ctr", PointwiseEvaluator("auc", "logloss", "gauc"), target="is_click")
evaluation.register("long_view", PointwiseEvaluator("auc", "logloss", "gauc"), target="long_view")
```

预测 Mapping 的 key 与标签同名，一个 target 完成配对。`user_id_raw` 全局声明一次，避免
OOV 编码将不同用户合并成 GAUC 同组。多标签没有默认 target 时，框架报歧义错误。

## 2. 重排：六视频一屏、前两屏历史

```python
current = SeqGroupConfig("slate", "slate_len", max_len=6)
history = SeqGroupConfig("history", "history_len", max_len=12)
settings = [SequenceSetting(f"cur_{s.field_name}", s, current) for s in shared_atoms]
settings += [SequenceSetting(f"hist_{s.field_name}", s, history) for s in shared_atoms]
manager = SchemaManager(settings, slate_schema, label_fields=[
    SlateLabel("is_click", 6, "mask"), SlateLabel("long_view", 6, "mask"),
], aux_fields=[AuxField("prior_long_view", max_len=6), AuxField("page_mask", max_len=2),
               AuxField("request_id", dtype="int64"), AuxField("user_id_raw", dtype="int64")])
model = MultiGERerankModel(
    manager, slate_cate_group="slate", hist_cate_group="history",
    slate_dense_field="prior_long_view", page_mask_field="page_mask",
    label_field="long_view", poi_num=6, page_num=2, submodularity_mode="lc",
    top_k=4, eval_all_perms=False, mlp_a=(32, 16), mlp_b=(32, 16), hidden_dim=32,
    loss_reduction="valid_mean",
)
# predict 返回 Prediction(scores=..., order=..., chosen_scores=..., candidates=..., target="long_view", extra=...)
evaluation = EvaluatorManager(manager, uid_field="user_id_raw")
evaluation.register("long_view", PointwiseEvaluator("auc", "logloss", "gauc"))
evaluation.register("slate", RankingEvaluator("ndcg@6", "pcoc"), resolver=BatchResolver(view="score_only"))
evaluation.register("chosen_order", RankingEvaluator("ndcg@6"))
evaluation.register("pointwise_prior", RankingEvaluator("ndcg@6"),
                    resolver=BatchResolver(pred_scorer=FieldScorer("prior_long_view"), view="score_only"))
if model.outputs_candidate_pool:
    evaluation.register("candidates", SlateValueEvaluator("cand_mean", "cand_max", reduction="mean"),
                        resolver=BatchResolver.for_candidates())
```

当前和历史引用同一批原子对象，共享词表及 embedding；历史存储仍拍扁，进入模型再折屏。
模型明确选择 long_view，即使 manager 首个标签是 is_click，评估也从 Prediction.target
继承长播，并自动查询对应 SlateLabel.mask_field。

| 内容 | 形状 |
| --- | --- |
| 当前分类属性 stack | `[B,6,4,8]` |
| 历史 stack 后折屏 | `[B,2,6,4,8]` |
| 标签 / mask / 冻结先验 | 各 `[B,6]` |
| 历史页 page_mask | `[B,2]` |
| pred.scores | 原始槽位长播概率 `[B,6]` |
| pred.order | 输出位置 → 输入槽位 `[B,6]` |
| pred.chosen_scores | 与最终 order 对齐的选中得分 `[B,6]` |
| pred.candidates.scores / orders | 各 `[B,4,6]`，类型化主候选分数及对应排列 |

RankingEvaluator 通过 score_only 评价原序分数排序，默认视图评价最终生成排列。
SlateValueEvaluator 评价已对齐整刷或主候选的价值，输入 EvalBatch 只有张量。
自定义来源继承 BaseScorer，在 EvalContext 中读取所需原始字段，再由 Resolver 投射。
候选指标将 mask 按每个候选排列搬到对应坐标后求均值，属于模型估计分，不是真实收益。
导出调用 pred.resolve_chosen_scores()，候选池缺显式选中得分直接报错；子类诊断使用 pred.with_extra()。

## 运行与验证

```bash
uv run python -m workspace.kr_time_budget.run_architecture_examples
```

默认两个任务各完整训练 1 epoch，valid 选 checkpoint，重载后测完整 test。
[原始实验报告](实验报告.md) 保留旧接口运行时的证据；[Prediction 迁移报告](Prediction迁移.md)
复用同一 checkpoint 测完整 test，核对分数、排列和指标。历史预算实验与原始 KR1K 示例不改写。
本次类型化验证为 848 项测试及 11 类候选模型同权重合成 batch 的逐值对照，未重跑完整数据集实验。
````

## File: workspace/kr_time_budget/run_architecture_examples.py
````python
"""Top-level entry: uv run python -m workspace.kr_time_budget.run_architecture_examples."""
from workspace.kr_time_budget.pipeline.run_architecture_examples import main

if __name__ == "__main__":
    main()
````
