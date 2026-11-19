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
- Only files matching these patterns are included: docs/rosen/repomix/examples/kr1k-original.md, docs/rosen/repomix/repomix.original.config.json, docs/rosen/repomix/README.md, docs/rosen/details/kuairand1k.py, src/rosen/datasets/kuairand.py
- Files matching these patterns are excluded: **/__pycache__/**, **/*.pyc, **/.DS_Store, docs/rosen/repomix/rosen-core.md, docs/rosen/repomix/kr1k-*.md, docs/rosen/repomix/manifest.json
- Files matching default ignore patterns are excluded
- Content has been formatted for parsing in markdown style

# User Provided Header
原始 KR1K 示例：docs/rosen/details/kuairand1k.py 原文保留；历史验证输入等语义见包内导读。 先读 docs/rosen/repomix/examples/kr1k-original.md。附件为阅读包，运行请使用仓库源文件路径。

# Directory Structure
````
docs/
  rosen/
    details/
      kuairand1k.py
    repomix/
      examples/
        kr1k-original.md
      README.md
      repomix.original.config.json
src/
  rosen/
    datasets/
      kuairand.py
````

# Files

## File: docs/rosen/details/kuairand1k.py
````python
"""
KuaiRand 1K example using the standalone rosen package and built-in models.

Run from the project root:
    uv run python docs/rosen/details/kuairand1k.py --model crocodile

Set ROSEN_DATASOURCE_ROOT to the directory containing KuaiRand/KuaiRand-1K/data.
Generated data and logs are written beneath rosen.experiment.WORKSPACE.

This example demonstrates:
    - SchemaManager with SparseEmbSetting / MultiSparseSetting
    - fit -> transform -> save_as_dataset
    - ParquetStreamDataset
    - EvaluatorManager + DomainFilter
    - build_model() with rosen's built-in MODEL_REGISTRY
"""
import time
from dataclasses import dataclass, replace
from typing import Dict

import torch

from rosen.core.train import EarlyStopper, TwoStageTrainingProtocol
from rosen.core.train.context import TrainerDataLoaders, TrainerComponents
from rosen.core.trainers import BaseTrainerWithLife, TwoStageRunner
from rosen.data.dataset import ParquetStreamDataset
from rosen.emb.schema import SparseEmbSetting, MultiSparseSetting
from rosen.emb import SchemaManager
import polars as pl
from rosen.core.enum_type import FeatureSource
from rosen.evaluate.evaluator import PointwiseEvaluator

from rosen.evaluate.manager import EvaluatorManager, DomainFilter
from rosen.experiment.param import ConfigBase, ParamManager
from rosen.models.base import BaseModel
from rosen.models.msr import build_model
from rosen.utils.optimize import split_params_by_decay
from rosen.experiment import change_root_workdir

change_root_workdir()

@dataclass
class KuairandConfig(ConfigBase):
    dataset_name: str = "kuairand-rand"
    seed: int = 2026
    device: str = "cuda"
    max_epochs: int = 1
    ckpt_dir: str = "" # 不保存ckpt

    batch_size: int = 4096
    id_emb: int = 32
    side_emb: int = 32
    shuffle_buffer_size: int = 2000000
    train_log_interval: int = 200

    model: str = "crocodile"
    log_name: str = "test.log"
    weight_decay: float = 1e-6
    train_end_date: int = 20220506
    valid_dates: str = "20220507"
    test_dates: str = "20220508"
    derive_date_from_time_ms: bool = False



MODEL_KWARGS: Dict[str, Dict[str, object]] = {
    "ple": dict(
        num_levels=2,
        num_shared_experts=1,
    ),
    "sharedbottom_codebook": dict(
        num_tokens = 64,
        codebook_dim = 16,
        codebook_decay = 0.99,
        quantize_weight = 0.02
    )
}

pm = ParamManager(KuairandConfig)
cfg: KuairandConfig = pm.build(experiment_name=KuairandConfig.model)
cfg.experiment_name = cfg.model

print(cfg)
time.sleep(2)


class KuairandTrainer(BaseTrainerWithLife):
    def __init__(self, model: BaseModel, optimizer: torch.optim.Optimizer, manager: SchemaManager,
                 loaders: TrainerDataLoaders, components: TrainerComponents, cfg: ConfigBase):
        super().__init__(model, optimizer, manager, loaders, components, cfg)

    def train_epoch(self):
        if self.epoch != 0:
            self.model.omni_embedding.reinitialize_large_vocab_embeddings(1001, init_std=1e-2)
            print("重初始化")

        super().train_epoch()



if __name__ == '__main__':
    from rosen.utils.task_chain import auto_queue
    auto_queue()

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, cfg.id_emb, min_freq=10, use_oov=True)
    item_setting = SparseEmbSetting("video_id", FeatureSource.ITEM_ID, cfg.id_emb, min_freq=10, use_oov=True)
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("tab", FeatureSource.INTERACTION, cfg.side_emb, padding_zero=False, use_oov=False, use_null=False),

        SparseEmbSetting("user_active_degree", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("is_live_streamer", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("is_video_author", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("friend_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("register_days_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("follow_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("fans_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("is_lowactive_period", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat0", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat1", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat2", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat3", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat4", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat5", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat6", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat7", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat8", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat9", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat10", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat11", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat12", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat13", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat14", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat15", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat16", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat17", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),

        SparseEmbSetting("author_id", FeatureSource.ITEM, cfg.id_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("music_id", FeatureSource.ITEM, cfg.id_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("video_type", FeatureSource.ITEM, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("music_type", FeatureSource.ITEM, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("visible_status", FeatureSource.ITEM, cfg.side_emb, min_freq=10, use_oov=True),

        MultiSparseSetting(
            "tag", FeatureSource.ITEM,cfg.side_emb, max_tag_len=5,
            is_string_format=True, separator=",", min_freq=10, use_oov=True
        ),
    ]
    from rosen.experiment import WORKSPACE
    WORKDIR = WORKSPACE / cfg.dataset_name
    manager = SchemaManager(settings_list, WORKDIR, time_field="time_ms", label_fields="is_click", domain_fields="tab")
    from rosen.datasets.kuairand import KuaiRandDataset

    user_lf = pl.scan_csv(KuaiRandDataset.USER_FEATURES)
    item_lf = pl.scan_csv(KuaiRandDataset.VIDEO_FEATURES)
    inter_lf = pl.scan_csv([KuaiRandDataset.STD_LOG_FORMER_DATA_P1, KuaiRandDataset.STD_LOG_FORMER_DATA_P2, KuaiRandDataset.RAND_LOG_FORMER_DATA])
    whole_lf: pl.LazyFrame = inter_lf.join(item_lf, on="video_id", how="left").join(user_lf, on="user_id", how="left")

    ## 排除部分场景数据
    top_5_df = whole_lf.group_by("tab").len().sort("len", descending=True).head(5).select("tab").collect()
    top_5_list = top_5_df.get_column("tab").to_list()
    whole_lf = whole_lf.filter(pl.col("tab").is_in(top_5_list))
    print(top_5_list)

    if cfg.derive_date_from_time_ms:
        event_time = pl.from_epoch("time_ms", time_unit="ms").dt.convert_time_zone("Asia/Shanghai")
        whole_lf = whole_lf.with_columns(
            event_time.dt.strftime("%Y%m%d").cast(pl.Int32).alias("date")
        )

    ## 增加新特征
    parsed_date = pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
    # whole_lf = whole_lf.with_columns(
    #     parsed_date.dt.weekday().alias("day_of_week"),
    #     parsed_date.dt.weekday().is_in([6, 7]).cast(pl.UInt8).alias("is_weekend"),
    #     (pl.col("hourmin").cast(pl.Int32) // 100).cast(pl.UInt8).alias("hour")
    # )

    whole_lf = manager.make_shuffled_checkpoint(
        whole_lf,
        file_name="whole_shuffled.parquet",
        redo=False,
        seed=cfg.seed,
    )
    print("join表checkpoint完成")
    ## 处理中
    parse_dates = lambda value: [int(part.strip()) for part in str(value).split(",") if part.strip()]
    valid_dates = parse_dates(cfg.valid_dates)
    test_dates = parse_dates(cfg.test_dates)
    train_raw = whole_lf.filter(pl.col("date") <= cfg.train_end_date)
    valid_raw = whole_lf.filter(pl.col("date").is_in(valid_dates))
    test_raw = whole_lf.filter(pl.col("date").is_in(test_dates))
    print(
        f"date split: train<= {cfg.train_end_date}, "
        f"valid={valid_dates}, test={test_dates}"
    )

    manager.fit(train_raw, low_memory=True)
    train_lf = manager.transform(train_raw)
    valid_lf = manager.transform(valid_raw)
    test_lf = manager.transform(test_raw)
    print(train_lf.select("date").head(5).collect())
    print(valid_lf.select("date").head(5).collect())
    train_path, valid_path, test_path = manager.save_as_dataset(train_lf, valid_lf, test_lf)
    print("架构编译成功，可供调用。")
    # ======================== 模型在这里 ======================================== #
    # num_domains = manager.get_setting(manager.domain_field).vocab_size
    num_domains = 5
    model_kwargs = MODEL_KWARGS.get(cfg.model, {}).copy()
    if getattr(cfg, "extras", None):
        model_kwargs.update(cfg.extras)
    print(model_kwargs)
    model = build_model(manager, num_domains, cfg.model, **model_kwargs)
    # ======================== 数据处理完成 准备trainer信息 ======================== #
    ps_dataset = ParquetStreamDataset(train_path, manager, batch_size=cfg.batch_size, shuffle=False, drop_last=False) # 文件已全局打乱
    ps_valid   = ParquetStreamDataset(test_path, manager, batch_size=4096 * 2, shuffle=False, drop_last=False) # 不能被shuffle
    # ======================== Trainer 准备 =======================#
    def trainer_factory(current_model, stage=None):
        run_cfg = cfg
        if stage is not None and cfg.ckpt_dir:
            run_cfg = replace(cfg, ckpt_dir=f"{cfg.ckpt_dir}/{stage.value}")
        evaluator_manager = EvaluatorManager(
            log_path=WORKDIR / cfg.log_name,
            title=cfg.experiment_name,
        )
        evaluator_manager.register("overall", PointwiseEvaluator("auc", "gauc"))
        for domain in range(num_domains):
            evaluator_manager.register(
                f"domain{domain}",
                PointwiseEvaluator("auc", "gauc"),
                filter_fn=DomainFilter(manager.domain_field, domain),
            )

        params = split_params_by_decay(
            current_model.named_parameters(),
            weight_decay=cfg.weight_decay,
            no_decay_keywords=["embedding"],
        )
        return KuairandTrainer(
            current_model,
            torch.optim.Adam(params, lr=1e-3),
            manager,
            TrainerDataLoaders(train=ps_dataset, valid=ps_valid),
            TrainerComponents(
                evaluator_manager=evaluator_manager,
                early_stopper=EarlyStopper(),
            ),
            run_cfg,
        )

    if isinstance(model, TwoStageTrainingProtocol):
        TwoStageRunner(trainer_factory).run(model)
    else:
        trainer_factory(model).run()
````

## File: docs/rosen/repomix/examples/kr1k-original.md
````markdown
# 原始 KR1K 示例

附件原文收录当前仓库已有的 `docs/rosen/details/kuairand1k.py`，并附 `src/rosen/datasets/kuairand.py` 数据路径适配器。原始脚本未因本次整理而修改。

它展示单个 `is_click` 标签、多场景 `tab`、SchemaManager、ParquetStreamDataset、按 domain 注册 evaluator，以及模型工厂和两阶段 Trainer 用法。它不是 CTR + 长播的多任务示例，也不是六视频重排入口。

注意原文中的历史行为：脚本虽然生成 train/valid/test 三个文件，`ps_valid` 实际读取 `test_path` 作为 Trainer 的验证输入。因此它不作为本轮新示例的验证/测试协议参考；新的两个示例明确使用独立 valid 选 checkpoint，再加载测 test。

原脚本默认设备为 CUDA，并在模块顶层解析参数、设置工作目录；阅读或语法检查无需 import 执行。数据根由 `ROSEN_DATASOURCE_ROOT` 指定。更多框架依赖查配套 `rosen-core.md` 或完整仓库。
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

## File: docs/rosen/repomix/repomix.original.config.json
````json
{
  "output": {
    "filePath": "docs/rosen/repomix/kr1k-original.md",
    "style": "markdown",
    "parsableStyle": true,
    "headerText": "原始 KR1K 示例：docs/rosen/details/kuairand1k.py 原文保留；历史验证输入等语义见包内导读。 先读 docs/rosen/repomix/examples/kr1k-original.md。附件为阅读包，运行请使用仓库源文件路径。",
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
    "docs/rosen/repomix/examples/kr1k-original.md",
    "docs/rosen/repomix/repomix.original.config.json",
    "docs/rosen/repomix/README.md",
    "docs/rosen/details/kuairand1k.py",
    "src/rosen/datasets/kuairand.py"
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
