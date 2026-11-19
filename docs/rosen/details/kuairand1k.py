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
