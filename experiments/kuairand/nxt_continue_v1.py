"""KR 离开预测模型（rosen 架构重写版）。

数据入口：uv run python -m experiments.kuairand.nxt_continue_v1
默认产物：workspace/kr_time_budget/data/nxt_continue_v1_raw/{train,valid,test}.parquet
加载：manager, loaders = load_data(DEFAULT_DATA_DIR)
pace 在预处理时写为 Float32 列 log1p(elapsed_min_raw / req_idx_raw)，模型直接读取。
完整 CSV -> 训练入口与说明见 experiments/kuairand/README.md。
prepare_baseline_data 仅用于已有历史 v6 产物的审计，不是新机器的前置条件。
当前刷四项反馈作为 Float32 辅助列进入内容塔；默认加载时编码 UID=0。

核心分解：
- TimeBudgetExitModel: P(continue) = gamma(t | bg, prior) * stay(bg, content[, pace])
  - 两塔独立 OmniEmbLayer（同名背景字段同初值出发，训练参数互不共享）
  - WeibullHead 统一支持 K=1 单威布尔与 K>=2 混合威布尔（按存活后验更新条件存活率）
  - 可消融辅助监督：左塔终屏离散 NLL（按 requests_per_session 校正为会话均值）+ 右塔直连 BCE
- FusionExitModel: 全特征单塔基线对照
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from rosen.core.interaction import Interaction
from rosen.core.enum_type import FeatureSource as FS
from rosen.data.dataset import ParquetStreamDataset
from rosen.datasets.kuairand import KuaiRandDataset
from rosen.emb import AuxField, SchemaManager
from rosen.emb.schema import MinMaxDenseSetting, MultiSparseSetting, SparseEmbSetting
from rosen.emb.emblayer import EmbView, OmniEmbLayer
from rosen.models.base import BaseModel


# 数据处理：raw CSV -> 六视频 Request -> 严格历史特征 -> train-fit schema / parquet。
USER_CATEGORY_FIELDS = [
    "user_active_degree", "is_live_streamer", "is_video_author",
    "friend_user_num_range", "register_days_range", "follow_user_num_range", "fans_user_num_range",
] + [f"onehot_feat{i}" for i in range(18)]
USER_PROFILE_FIELDS = ["user_id"] + USER_CATEGORY_FIELDS
SESSION_CONTEXT_FIELDS = ["tab", "day_of_week", "is_weekend", "start_hour"]
SHARED_FIELDS = USER_PROFILE_FIELDS + SESSION_CONTEXT_FIELDS
USER_BUDGET_PRIOR_FIELDS = ["user_hist_session_min", "user_hist_req_cnt"]
DAY_BUDGET_PRIOR_FIELDS = ["day_prev_sess_cnt", "day_prev_elapsed_min"]
BUDGET_PRIOR_FIELDS = USER_BUDGET_PRIOR_FIELDS + DAY_BUDGET_PRIOR_FIELDS
SESSION_CLOCK_FIELDS = ["elapsed_min", "req_idx"]
HISTORY_NUMERIC_FIELDS = [
    "prev1_click_cnt", "prev1_longview_cnt", "prev1_pos_cnt", "sess_cum_click_rate",
    "sess_cum_longview_rate", "sess_cum_pos_rate", "sess_cum_hate_cnt",
]
CONTENT_HISTORY_FIELDS = ["is_first_req"] + HISTORY_NUMERIC_FIELDS + ["hist_tags"]
SLATE_NUMERIC_FIELDS = [
    "slate_prior_ctr_mean", "slate_prior_longview_mean", "slate_dur_mean_min",
    "slate_unique_tag_cnt", "slate_unique_author_cnt",
]
CONTENT_SLATE_FIELDS = ["slate_tags", "slate_video_types", "slate_music_types"] + SLATE_NUMERIC_FIELDS + ["slate_has_rand"]
BUDGET_PARAMETER_FIELDS = SHARED_FIELDS + BUDGET_PRIOR_FIELDS
CONTENT_TOWER_FIELDS = SHARED_FIELDS + CONTENT_HISTORY_FIELDS + CONTENT_SLATE_FIELDS
FEATURES = (
    USER_PROFILE_FIELDS + USER_BUDGET_PRIOR_FIELDS + SESSION_CONTEXT_FIELDS
    + DAY_BUDGET_PRIOR_FIELDS + SESSION_CLOCK_FIELDS + CONTENT_HISTORY_FIELDS + CONTENT_SLATE_FIELDS
)
TRAIN_END = 20220506
SPLITS = ("train", "valid", "test")
DEFAULT_DATA_DIR = Path("workspace/kr_time_budget/data/nxt_continue_v1_raw")
FEEDBACK_FIELDS = ["current_click_rate", "current_longview_rate", "current_positive_rate", "current_hate_rate"]


def build_schema(schema_dir: Path | str) -> SchemaManager:
    """54 个嵌入字段；pace 和公式原始量作为有类型的列随 batch 读取。"""
    settings = [SparseEmbSetting("user_id", FS.USER_ID, 16, min_freq=10, use_oov=True)]
    settings += [SparseEmbSetting(name, FS.USER, 8, min_freq=10, use_oov=True) for name in USER_CATEGORY_FIELDS]
    settings += [
        SparseEmbSetting(name, FS.INTERACTION, 8, padding_zero=name != "tab", use_null=name != "tab")
        for name in SESSION_CONTEXT_FIELDS
    ]
    settings += [SparseEmbSetting(name, FS.INTERACTION, 4) for name in ("is_first_req", "slate_has_rand")]
    settings += [
        MultiSparseSetting(name, FS.ITEM, dim, max_tag_len=cap, is_string_format=True, separator=",", agg="sum")
        for name, dim, cap in (
            ("hist_tags", 16, 15), ("slate_tags", 16, 12),
            ("slate_video_types", 8, 6), ("slate_music_types", 8, 6),
        )
    ]
    settings += [
        MinMaxDenseSetting(name, FS.INTERACTION, min_val=0.0, max_val=1.0)
        for name in BUDGET_PRIOR_FIELDS + SESSION_CLOCK_FIELDS + HISTORY_NUMERIC_FIELDS + SLATE_NUMERIC_FIELDS
    ]
    aux = [AuxField(name) for name in ["pace", "elapsed_min_raw", *FEEDBACK_FIELDS]]
    aux += [AuxField(name, dtype="int64") for name in (
        "request_id", "session_id", "user_id_raw", "tab_raw", "req_idx_raw",
    )]
    return SchemaManager(settings, str(schema_dir), label_fields="y_continue", domain_fields="tab", aux_fields=aux)


def _sessionize(events: pl.DataFrame) -> pl.DataFrame:
    events = events.sort("user_id", "time_ms", "source_order", "source_row").with_columns(
        pl.col("play_time_ms").clip(0, 300000).alias("play_clip")
    )
    gap = pl.col("time_ms") - (
        pl.col("time_ms").shift(1).over("user_id") + pl.col("play_clip").shift(1).over("user_id")
    )
    events = events.with_columns((gap.is_null() | (gap > 1800000)).cast(pl.Int64).alias("new_session"))
    events = events.with_columns(pl.col("new_session").cum_sum().alias("session_id"))
    return events.with_columns(
        pl.len().over("session_id").alias("session_n"),
        pl.int_range(pl.len()).over("session_id").alias("event_idx"),
        pl.col("time_ms").first().over("session_id").alias("session_start_ms"),
        pl.col("session_id").first().over("user_id").alias("first_session"),
        pl.col("session_id").last().over("user_id").alias("last_session"),
    ).with_columns(
        ((pl.col("event_idx") - (pl.col("session_n") % 6).cast(pl.Int64)) // 6 + 1).alias("req_idx")
    )


def _build_sessions(events: pl.DataFrame) -> pl.DataFrame:
    sessions = events.group_by("user_id", "session_id", maintain_order=True).agg(
        pl.col("date").first().alias("session_date"), pl.col("date").max().alias("end_date"),
        (pl.col("hourmin").first() // 100).alias("start_hour"), pl.col("tab").first(),
        pl.col("time_ms").first().alias("session_start_ms"),
        pl.col("time_ms").last().alias("session_last_time_ms"),
        ((pl.col("time_ms").last() - pl.col("time_ms").first()) / 60000).alias("session_min"),
        (pl.len() // 6).cast(pl.Float64).alias("session_req_cnt"), pl.len().alias("session_n"),
        pl.col("first_session").first(), pl.col("last_session").first(),
    ).sort("session_id")
    return sessions.with_columns(
        (pl.col("session_id").cum_count().over("user_id", "session_date") - 1).alias("day_prev_sess_cnt"),
        (pl.col("session_min").cum_sum().over("user_id", "session_date") - pl.col("session_min")).alias("day_prev_elapsed_min"),
    )


def _aggregate_requests(events: pl.DataFrame) -> pl.DataFrame:
    """头部不足六条的 chunk 0 仅入历史；当前反馈仅供后续 Request 使用。"""
    requests = events.group_by("user_id", "session_id", "req_idx", maintain_order=True).agg(
        pl.len().alias("n"), pl.col("video_id").alias("slate_items"),
        pl.col("tag").str.join(",").alias("slate_tags"),
        pl.col("tag").filter((pl.col("is_click") == 1) | (pl.col("long_view") == 1)).str.join(",").alias("engaged_tags"),
        pl.col("video_type").str.join(",").alias("slate_video_types"),
        pl.col("music_type").str.join(",").alias("slate_music_types"),
        (pl.col("duration_ms").clip(3000, 600000).mean() / 60000).alias("slate_dur_mean_min"),
        pl.col("author_id").filter(pl.col("author_id") >= 0).n_unique().alias("slate_unique_author_cnt"),
        pl.col("is_rand").max().alias("slate_has_rand"),
        pl.col("is_click").sum().alias("click_n"), pl.col("long_view").sum().alias("long_n"),
        pl.col("positive").sum().alias("pos_n"), pl.col("is_hate").sum().alias("hate_n"),
        pl.col("time_ms").last().alias("request_time_ms"),
        ((pl.col("time_ms").last() - pl.col("session_start_ms").first()) / 60000).alias("elapsed_min"),
    ).sort("session_id", "req_idx")
    counts = ["n", "click_n", "long_n", "pos_n", "hate_n"]
    requests = requests.with_columns(
        *[(pl.col(c).cum_sum().over("session_id") - pl.col(c)).alias("past_" + c) for c in counts]
    ).with_columns(
        *[pl.col(c).shift(1).over("session_id").fill_null(0).alias(out) for c, out in (
            ("click_n", "prev1_click_cnt"), ("long_n", "prev1_longview_cnt"), ("pos_n", "prev1_pos_cnt"),
        )],
        *[(pl.col("past_" + c) / pl.col("past_n").clip(lower_bound=1)).alias(out) for c, out in (
            ("click_n", "sess_cum_click_rate"), ("long_n", "sess_cum_longview_rate"), ("pos_n", "sess_cum_pos_rate"),
        )],
        pl.col("past_hate_n").alias("sess_cum_hate_cnt"),
        (pl.col("req_idx") == 1).cast(pl.Int8).alias("is_first_req"),
        pl.concat_str([
            pl.col("engaged_tags").shift(2).over("session_id").fill_null(""),
            pl.col("engaged_tags").shift(1).over("session_id").fill_null(""),
        ], separator=",").alias("hist_tags"),
        (pl.col("req_idx") < pl.col("req_idx").max().over("session_id")).cast(pl.Int8).alias("y_continue"),
        pl.col("slate_tags").str.split(",").list.eval(
            pl.element().filter((pl.element() != "") & (pl.element() != "__MISSING__"))
        ).list.n_unique().alias("slate_unique_tag_cnt"),
        pl.col("req_idx").cast(pl.Int64).alias("req_idx_raw"),
        pl.col("elapsed_min").cast(pl.Float32).alias("elapsed_min_raw"),
        (pl.col("elapsed_min") / pl.col("req_idx").clip(lower_bound=1)).log1p().cast(pl.Float32).alias("pace"),
        *[(pl.col(source).cast(pl.Float32) / 6).alias(field) for source, field in zip(
            ["click_n", "long_n", "pos_n", "hate_n"], FEEDBACK_FIELDS)],
    )
    return requests.filter(pl.col("req_idx") > 0).drop(
        "engaged_tags", *counts, *["past_" + c for c in counts]
    )


def _past_daily(query: pl.DataFrame, observations: pl.DataFrame, keys: list[str], values: list[str]) -> pl.DataFrame:
    """只统计查询日之前的数据；valid/test 的查询日统一冻结为 20220507。"""
    daily = observations.group_by(keys + ["date"]).agg(
        pl.len().cast(pl.Float64).alias("prior_n"), *[pl.col(v).sum().alias("sum_" + v) for v in values]
    ).sort(keys + ["date"])
    sums = ["prior_n"] + ["sum_" + v for v in values]
    daily = daily.with_columns(*[
        (pl.col(c).cum_sum().over(keys) if keys else pl.col(c).cum_sum()).alias(c) for c in sums
    ])
    return query.sort("date").join_asof(
        daily.sort("date"), on="date", by=keys or None, strategy="backward",
        allow_exact_matches=False, check_sortedness=False,
    ).with_columns(pl.col(sums).fill_null(0))


def _user_priors(query: pl.DataFrame, sessions: pl.DataFrame) -> pl.DataFrame:
    values = ["session_min", "session_req_cnt"]
    observations = sessions.select("user_id", pl.col("end_date").alias("date"), *values)
    own = _past_daily(query, observations, ["user_id"], values)
    global_priors = _past_daily(query.select("request_id", "date"), observations, [], values).select(
        "request_id", *[(pl.col("sum_" + c) / pl.col("prior_n").clip(lower_bound=1)).alias("global_" + c) for c in values]
    )
    return own.join(global_priors, on="request_id", validate="1:1").select(
        "request_id", pl.col("prior_n").alias("user_prior_n"),
        *[((pl.col("sum_" + c) + 5 * pl.col("global_" + c)) / (pl.col("prior_n") + 5)).alias(out)
          for c, out in zip(values, USER_BUDGET_PRIOR_FIELDS)],
    )


def _item_priors(query: pl.DataFrame, observations: pl.DataFrame) -> pl.DataFrame:
    values = ["is_click", "long_view"]
    per_item = _past_daily(query.explode("slate_items", empty_as_null=True).rename({"slate_items": "video_id"}), observations, ["video_id"], values)
    global_priors = _past_daily(query.select("request_id", "date"), observations, [], values).select(
        "request_id", *[(pl.col("sum_" + c) / pl.col("prior_n").clip(lower_bound=1)).alias("global_" + c) for c in values]
    )
    per_item = per_item.join(global_priors, on="request_id").with_columns(*[
        ((pl.col("sum_" + c) + 20 * pl.col("global_" + c)) / (pl.col("prior_n") + 20)).alias("quality_" + c) for c in values
    ])
    return per_item.group_by("request_id").agg(
        pl.col("quality_is_click").mean().alias("slate_prior_ctr_mean"),
        pl.col("quality_long_view").mean().alias("slate_prior_longview_mean"),
    )


def _encode_requests(raw: pl.DataFrame, output: Path, seed: int) -> dict:
    raw = raw.with_columns(*[pl.col(c).clip(lower_bound=0).log1p().cast(pl.Float32) for c in BUDGET_PRIOR_FIELDS])
    manager = build_schema(output / "schema")
    manager.fit(raw.filter(pl.col("split") == "train").lazy())
    stats = {}
    for split in SPLITS:
        frame = manager.transform(raw.filter(pl.col("split") == split).lazy()).select(manager.fields()).collect()
        if not frame.height:
            raise ValueError(f"Empty {split} split")
        if split == "train":
            frame = frame.sample(fraction=1, shuffle=True, seed=seed)
        if frame.null_count().sum_horizontal().item():
            raise ValueError(f"Null values in encoded {split}")
        frame.write_parquet(output / f"{split}.parquet", row_group_size=65536)
        stats[split] = {
            "rows": frame.height, "sessions": frame["session_id"].n_unique(),
            "users": frame["user_id_raw"].n_unique(), "positive_rate": frame["y_continue"].mean(),
            "requests_per_session": frame.height / frame["session_id"].n_unique(),
            "request_order_sha256": hashlib.sha256(frame["request_id"].to_numpy().tobytes()).hexdigest(),
        }
        print("ENCODED", split, stats[split], flush=True)
    return stats


def _prepare_data(source: Path, output: Path, log_paths: list[Path], seed: int) -> dict:
    frames, sources = [], []
    for i, path in enumerate(log_paths):
        frame = pl.read_csv(path).with_row_index("source_row").with_columns(pl.lit(i).alias("source_order"))
        frames.append(frame)
        sources.append({"path": str(path.resolve()), "rows": frame.height, "bytes": path.stat().st_size})
        print("READ", path.name, frame.height, flush=True)
    events = pl.concat(frames)
    del frames
    raw_rows = events.height
    raw_columns = [c for c in events.columns if c not in ("source_row", "source_order")]
    events = _sessionize(events.unique(subset=raw_columns, maintain_order=True)).with_row_index("event_id")
    events.write_parquet(output / "events.parquet", row_group_size=65536)
    sessions = _build_sessions(events)
    sessions.write_parquet(output / "sessions.parquet")
    usable = sessions.filter(
        (pl.col("session_id") != pl.col("first_session")) & (pl.col("session_id") != pl.col("last_session"))
        & (pl.col("session_n") >= 6)
    )
    top_tabs = usable.filter(pl.col("end_date") <= TRAIN_END).group_by("tab").len().sort(
        ["len", "tab"], descending=[True, False]
    ).head(5)["tab"].to_list()
    usable = usable.filter(pl.col("tab").is_in(top_tabs)).with_columns(
        pl.when(pl.col("end_date") <= TRAIN_END).then(pl.lit("train"))
        .when((pl.col("session_date") == 20220507) & (pl.col("end_date") == 20220507)).then(pl.lit("valid"))
        .when((pl.col("session_date") == 20220508) & (pl.col("end_date") == 20220508)).then(pl.lit("test"))
        .otherwise(pl.lit("excluded")).alias("split")
    )
    boundary_excluded = usable.filter(pl.col("split") == "excluded").height
    usable = usable.filter(pl.col("split") != "excluded")
    video = pl.read_csv(
        source / "video_features_basic_1k.csv", columns=["video_id", "author_id", "tag", "video_type", "music_type"],
        schema_overrides={"tag": pl.String, "video_type": pl.String},
    )
    selected = events.join(usable.select("session_id"), on="session_id", how="semi").sort("session_id", "event_idx")
    selected = selected.join(video, on="video_id", how="left", validate="m:1", maintain_order="left").with_columns(
        pl.col("tag").fill_null("__MISSING__"), pl.col("video_type").fill_null("__MISSING__"),
        pl.col("music_type").cast(pl.Int64).cast(pl.String).fill_null("__MISSING__"), pl.col("author_id").fill_null(-1),
        pl.any_horizontal(*[pl.col(c) == 1 for c in ("is_like", "is_comment", "is_follow", "is_forward")]).cast(pl.Int8).alias("positive"),
    )
    requests = _aggregate_requests(selected).join(usable, on=["user_id", "session_id"], how="left", validate="m:1").with_row_index("request_id")
    selected.select("event_id", "user_id", "session_id", "req_idx", "event_idx").join(
        requests.select("session_id", "req_idx", "request_id"), on=["session_id", "req_idx"], how="left",
    ).write_parquet(output / "request_events.parquet")
    del selected, video
    users = pl.read_csv(source / "user_features_1k.csv").select("user_id", *USER_CATEGORY_FIELDS).with_columns(
        pl.col("is_live_streamer").eq(1).cast(pl.Int8),
        *[pl.col(f"onehot_feat{i}").cast(pl.Int64).fill_null(-1) for i in range(18)],
    )
    requests = requests.join(users, on="user_id", how="left", validate="m:1").with_columns(
        pl.col("user_id").alias("user_id_raw"), pl.col("tab").alias("tab_raw"),
        pl.col("session_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d").dt.weekday().alias("day_of_week"),
        (pl.col("session_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
    )
    query = requests.select(
        "request_id", "user_id",
        pl.when(pl.col("split") == "train").then(pl.col("session_date")).otherwise(pl.lit(TRAIN_END + 1)).alias("date"),
    )
    requests = requests.join(_user_priors(query, sessions), on="request_id", validate="1:1")
    observations = events.filter(pl.col("date") <= TRAIN_END).select("video_id", "date", "is_click", "long_view")
    item_query = query.select("request_id", "date").join(requests.select("request_id", "slate_items"), on="request_id", validate="1:1")
    requests = requests.join(_item_priors(item_query, observations), on="request_id", validate="1:1").sort("session_id", "req_idx")
    requests.write_parquet(output / "features_raw.parquet", row_group_size=65536)
    deduplicated_rows = events.height
    del events, observations, item_query, query, users
    stats = _encode_requests(requests, output, seed)
    return {
        "sources": sources, "raw_rows": raw_rows, "deduplicated_rows": deduplicated_rows,
        "sessions_total": sessions.height, "sessions_retained": usable.height,
        "boundary_sessions_excluded": boundary_excluded, "top_tabs": top_tabs, "seed": seed, "splits": stats,
        "protocol": {
            "dedup": "all raw CSV columns, before sessionization",
            "session": "idle > 30min after previous clipped play (0..300000ms); user first/last session excluded",
            "request": "six events; head remainder is history-only chunk 0",
            "split": "train ends <= 20220506; valid entirely 20220507; test entirely 20220508; crossing held-out boundaries excluded",
            "priors": "strictly previous calendar days; valid/test frozen through 20220506; user/item smoothing 5/20",
            "clock": "elapsed_min_raw = (current screen last timestamp - session first timestamp)/60000; no play_time added",
            "pace": "Float32 log1p(elapsed_min_raw / req_idx_raw), materialized before encoding and shuffle",
            "content": "current static slate, prior-request history and four current-request feedback rates; decision after feedback",
            "schema": "fit train only; budget priors log1p; other dense values passthrough; pace/clock/IDs typed auxiliary columns",
            "ordering": "train physical shuffle; valid/test session_id, req_idx ascending",
        },
    }


def _source_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare_data(source: Path | str, output: Path | str, *, seed: int = 2026) -> Path:
    """独立从 KR-1K raw CSV 构建数据；写日志并拒绝复用非空输出目录。"""
    source, output = Path(source), Path(output)
    log_paths = sorted(source.glob("log_*.csv"))
    if not log_paths:
        raise FileNotFoundError(f"No log_*.csv found in {source}")
    for name in ("video_features_basic_1k.csv", "user_features_1k.csv"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    print("数据处理日志：", output / "logs/prepare.log", flush=True)
    with (output / "logs/prepare.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        input_files = [*log_paths, source / "video_features_basic_1k.csv", source / "user_features_1k.csv"]
        print("HASH raw CSV inputs", flush=True)
        input_records = [_source_record(path) for path in input_files]
        manifest = _prepare_data(source, output, log_paths, seed)
        manifest["build"] = {
            "kind": "raw_csv", "inputs": input_records,
            "preprocessor_sha256": _source_record(Path(__file__))["sha256"],
            "python": platform.python_version(), "polars": pl.__version__, "torch": torch.__version__,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print("DATA READY", output, flush=True)
    return output


def prepare_baseline_data(source: Path | str, output: Path | str) -> Path:
    """将冻结的 v6 编码数据物化为本入口的列契约，保留旧词表、成员和物理行序。

    这是显式的历史基线对齐入口；不重新 fit、shuffle 或编码，也不修改源数据。
    pace 以旧 torch Float32 算式逐批计算后落盘，训练时只读 pace 列。
    """
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["version"] != 6 or manifest["feedback_fields"] != FEEDBACK_FIELDS:
        raise ValueError("Baseline alignment requires v6_current_feedback")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    with (output / "logs/prepare.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        shutil.copytree(source / "schema", output / "schema")
        manager = build_schema(output / "schema")
        manager.load_schema()
        stats, hashes = {}, {}
        for split in SPLITS:
            path = source / f"{split}.parquet"
            frame = pl.read_parquet(path)
            minutes = torch.from_numpy(frame["elapsed_min"].to_numpy().copy()).float()
            screens = torch.from_numpy(frame["req_idx_raw"].to_numpy().copy()).float()
            pace = torch.log1p(minutes.clamp(min=0) / screens.clamp(min=1)).numpy()
            result = frame.with_columns(
                pl.col("elapsed_min").cast(pl.Float32).alias("elapsed_min_raw"), pl.Series("pace", pace),
            ).select(manager.fields())
            assert result["request_id"].equals(frame["request_id"])
            result.write_parquet(output / f"{split}.parquet", row_group_size=65536)
            hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            stats[split] = {
                "rows": result.height, "sessions": result["session_id"].n_unique(),
                "users": result["user_id_raw"].n_unique(), "positive_rate": result["y_continue"].mean(),
                "requests_per_session": result.height / result["session_id"].n_unique(),
                "request_order_sha256": hashlib.sha256(result["request_id"].to_numpy().tobytes()).hexdigest(),
            }
            print(split, stats[split], flush=True)
        aligned = {
            "source": str(source.resolve()), "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
            "source_parquet_sha256": hashes, "splits": stats,
            "protocol": {"schema": "frozen v6 train-fit copy", "order": "source physical order retained exactly",
                         "pace": "materialized torch Float32 log1p(elapsed_min / req_idx_raw)",
                         "feedback_fields": FEEDBACK_FIELDS, "source_unchanged": True},
        }
        (output / "manifest.json").write_text(json.dumps(aligned, indent=2) + "\n")
    return output


def load_data(data_dir: Path | str, *, batch_size: int = 4096, zero_uid: bool = True) -> tuple[SchemaManager, dict[str, DataLoader]]:
    """读取预处理产物；ParquetStreamDataset 已产出 batch，pace 自动随 schema 加载。"""
    data_dir = Path(data_dir)
    required = ["manifest.json", "schema/feature_meta.json", *[f"{s}.parquet" for s in SPLITS]]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete prepared dataset at {data_dir}: missing {', '.join(missing)}. "
            "Generate it from raw CSV with: uv run python -m experiments.kuairand.run_nxt_continue "
            f"--source_dir /path/to/KuaiRand-1K/data --data_dir {data_dir} --device cpu"
        )
    manager = build_schema(data_dir / "schema")
    manager.load_schema()
    loaders = {}
    for split in SPLITS:
        source = pl.scan_parquet(data_dir / f"{split}.parquet")
        if zero_uid:
            source = source.with_columns(pl.lit(0, dtype=pl.Int64).alias("user_id"))
        loaders[split] = DataLoader(ParquetStreamDataset(
            source, manager, batch_size=batch_size, shuffle=False, drop_last=False,
        ), batch_size=None)
    return manager, loaders


def _masked_bce_from_prob(
    prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    per_item = F.binary_cross_entropy(prob.clamp(eps, 1.0 - eps), target, reduction="none")
    return (per_item * mask).mean() / mask.mean().clamp(min=torch.finfo(mask.dtype).tiny)


def _masked_bce_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    per_item = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per_item * mask).sum() / mask.sum().clamp(min=1.0)


@dataclass(frozen=True)
class WeibullSurvival:
    """单个时钟分支的存活统计结果（便于训练反传及后续接入 ObservationPort）。"""

    gamma: torch.Tensor          # 条件存活率 S(t + delta) / S(t), shape (B,)
    survival: torch.Tensor       # 累积存活率 S(t), shape (B,)
    log_survival: torch.Tensor   # float64 log S(t), shape (B,)
    prior: torch.Tensor          # 先验分量权重 pi_k, shape (B, K)
    posterior: torch.Tensor      # t 时刻后验权重 p_k(t), shape (B, K)
    scale: torch.Tensor          # float64 尺度参数 lambda_k, shape (B, K)
    shape: torch.Tensor          # float64 形状参数 k_k, shape (B, K)


class WeibullHead(nn.Module):
    """普通 K>=1 混合或结构化 K=2 存活头，支持条件继续率与离散终屏 NLL。

    structured 限制 k0∈[0.3,1]、k1∈[1.2,4.5]，lambda1=lambda0+delta。
    coupled_bathtub 使用 S=(pi0*exp(-H0)+pi1)*exp(-H1)，避免早退态反超长尾。
    这些约束提供三阶段、双峰的表达能力，不保证每组参数都呈现双峰。
    """

    def __init__(
        self, in_features: int, num_components: int = 1, capped: bool = True,
        structured: bool = False, coupled_bathtub: bool = True,
    ) -> None:
        super().__init__()
        if num_components < 1:
            raise ValueError(f"num_components must be >= 1, got {num_components}")
        if structured and num_components != 2:
            raise ValueError("structured Weibull requires num_components == 2")
        self.num_components = int(num_components)
        self.capped = bool(capped)
        self.structured = bool(structured)
        self.coupled_bathtub = bool(coupled_bathtub)

        out_features = 2 if self.num_components == 1 else 3 * self.num_components
        self.proj = nn.Linear(in_features, out_features)
        if self.num_components > 1:
            self._init_mixture_symmetry_breaking()

    def _init_mixture_symmetry_breaking(self) -> None:
        with torch.no_grad():
            if self.structured:
                self.proj.weight.zero_()
                self.proj.bias.copy_(self.proj.bias.new_tensor([-0.3, 0.0, -0.3, -0.6, -0.4, 0.4]))
                return
            shifts = torch.linspace(-math.log(2.0), math.log(2.0), self.num_components)
            for idx, shift in enumerate(shifts):
                self.proj.bias[idx * 2].add_(shift)
            gate_start = 2 * self.num_components
            self.proj.weight[gate_start:].zero_()
            self.proj.bias[gate_start:].zero_()

    def raw(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)

    def _parse(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = raw.shape[0]
        k = self.num_components
        if k == 1:
            params = raw.unsqueeze(-2)  # (B, 1, 2)
            log_pi = raw.new_zeros((batch_size, 1), dtype=torch.float64)
        elif self.structured:
            r = raw.double()
            scale1 = torch.exp(math.log(0.6) + math.log(25.0 / 0.6) * torch.sigmoid(r[:, 0]))
            shape1 = 0.30 + 0.70 * torch.sigmoid(r[:, 1])
            delta_scale = torch.exp(math.log(3.0) + math.log(200.0 / 3.0) * torch.sigmoid(r[:, 2]))
            scale = torch.stack([scale1, scale1 + delta_scale], dim=-1)
            shape = torch.stack([shape1, 1.20 + 3.30 * torch.sigmoid(r[:, 3])], dim=-1)
            # 每个 logit 在 [-2.5, 2.5]；两分量最小权重约 0.0067，并非 0.075。
            log_pi = F.log_softmax(2.5 * torch.tanh(r[:, 4:6] / 2.5), dim=-1)
            return scale, shape, log_pi
        else:
            params = raw[:, : 2 * k].reshape(batch_size, k, 2)
            log_pi = F.log_softmax(raw[:, 2 * k :].double(), dim=-1)

        log_scale = params[..., 0].clamp(-2.0, 4.5) if self.capped else params[..., 0]
        scale = torch.exp(log_scale + 1.6).double()
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise FloatingPointError("Weibull scale must be finite and positive")
        shape = (0.3 + F.softplus(params[..., 1])).clamp(max=5.0).double()
        return scale, shape, log_pi

    @staticmethod
    def _hazard_and_increment(
        scale: torch.Tensor,
        shape: torch.Tensor,
        elapsed: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = elapsed.double().clamp(min=0.0).unsqueeze(-1)
        d = delta.double().clamp(min=0.0).unsqueeze(-1)
        safe_t = t.clamp(min=1e-12)

        base = (safe_t / scale).pow(shape)
        hazard = torch.where(t > 0, base, torch.zeros_like(base))
        inc_pos = base * torch.expm1(shape * torch.log1p(d / safe_t))
        inc_zero = torch.where(d > 0, (d.clamp(min=1e-12) / scale).pow(shape), torch.zeros_like(base))
        increment = torch.where(t > 0, inc_pos, inc_zero)
        return hazard, increment

    def _effective_hazard_and_increment(self, scale, shape, elapsed, delta):
        if not self.structured:
            # 普通单/混合 Weibull 保留原计算路径及浮点运算顺序。
            return self._hazard_and_increment(scale, shape, elapsed, delta)
        t = elapsed.double().clamp(min=0.0).unsqueeze(-1)
        d = delta.double().clamp(min=0.0).unsqueeze(-1)
        safe_t = torch.where(t > 0, t, torch.ones_like(t))
        log_h = shape * (safe_t.log() - scale.log())
        hazard = torch.where(t > 0, log_h.clamp(max=20.0).exp(), torch.zeros_like(scale))
        exponent = shape * torch.log1p(d / safe_t)
        inc_pos = (hazard * torch.expm1(exponent.clamp(max=20.0))).clamp(max=1e8)
        inc_zero = (shape * (d.clamp(min=1e-12).log() - scale.log())).clamp(max=20.0).exp()
        increment = torch.where(d > 0, torch.where(t > 0, inc_pos, inc_zero), torch.zeros_like(scale))
        if self.coupled_bathtub:
            hazard = torch.stack([hazard[:, 0] + hazard[:, 1], hazard[:, 1]], dim=-1)
            increment = torch.stack([increment[:, 0] + increment[:, 1], increment[:, 1]], dim=-1)
        return hazard, increment

    def survival(
        self,
        raw: torch.Tensor,
        elapsed: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
    ) -> WeibullSurvival:
        if delta is None:
            delta = torch.ones_like(elapsed)
        scale, shape, log_pi = self._parse(raw)
        hazard, increment = self._effective_hazard_and_increment(scale, shape, elapsed, delta)
        log_s_k = -hazard

        if self.num_components == 1:
            log_s = log_s_k.squeeze(-1)
            log_gamma = -increment.squeeze(-1)
            posterior = torch.ones_like(log_s_k)
        else:
            log_s = torch.logsumexp(log_pi + log_s_k, dim=-1)
            log_post = log_pi + log_s_k - log_s.unsqueeze(-1)
            log_gamma = torch.logsumexp(log_post - increment, dim=-1)
            posterior = log_post.exp()

        return WeibullSurvival(
            gamma=log_gamma.exp().to(raw.dtype),
            survival=log_s.exp().to(raw.dtype),
            log_survival=log_s,
            prior=log_pi.exp(),
            posterior=posterior,
            scale=scale,
            shape=shape,
        )

    def discrete_log_mass(self, raw: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """稳定计算 log P(L = l) = log[S(l-1) - S(l)]，支持 L=1 的 S(0)=1 边界。"""
        length_d = length.double()
        if length_d.ndim != 1 or length_d.shape[0] != raw.shape[0]:
            raise ValueError(f"Shape mismatch: raw={tuple(raw.shape)}, length={tuple(length_d.shape)}")
        if not torch.isfinite(length_d).all() or torch.any(length_d < 1) or torch.any(length_d != length_d.round()):
            raise ValueError("Complete positive integer screen counts required")

        scale, shape, log_pi = self._parse(raw)
        t = length_d - 1.0
        hazard, increment = self._effective_hazard_and_increment(scale, shape, t, torch.ones_like(t))
        prob_step = (-torch.expm1(-increment)).clamp(min=torch.finfo(torch.float64).tiny)
        log_mass_k = -hazard + torch.log(prob_step)

        if self.num_components == 1:
            return log_mass_k.squeeze(-1)
        return torch.logsumexp(log_pi + log_mass_k, dim=-1)


@dataclass(frozen=True)
class ExitOutput:
    """双塔离开预测模型的完整前向输出。"""

    prediction: torch.Tensor
    gamma: torch.Tensor
    stay: torch.Tensor
    stay_logit: torch.Tensor
    log_survival: torch.Tensor
    step_raw: torch.Tensor
    step_survival: WeibullSurvival
    time_survival: Optional[WeibullSurvival] = None


class _BaseExitModel(BaseModel):
    """KR 离开预测模型公共基类：schema 校验、label/mask 契约、pace 拼接与 LossPort 记账。"""

    def __init__(
        self,
        manager: SchemaManager,
        *,
        add_pace: bool = False,
        label_field: Optional[str] = None,
    ) -> None:
        actual = {s.field_name for s in manager.settings}
        expected = set(FEATURES)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"{type(self).__name__} requires the KR 54-field schema "
                f"(missing={missing}, extra={extra})"
            )
        super().__init__(manager)
        self.add_pace = bool(add_pace)
        self.label_field = label_field or self.LABEL or "y_continue"

    def _with_pace(self, features: torch.Tensor, interaction: Interaction) -> torch.Tensor:
        if not self.add_pace:
            return features
        return torch.cat([features, interaction["pace"].unsqueeze(-1)], dim=-1)

    def _target_and_mask(self, interaction: Interaction) -> tuple[torch.Tensor, torch.Tensor]:
        target = interaction[self.label_field].float()
        mask = self.get_label_mask(interaction, label_field=self.label_field, dtype=target.dtype)
        return target, mask

    def _finalize_loss(self, entries: list[tuple[str, torch.Tensor, float]]) -> torch.Tensor:
        if self.lsp.active:
            for name, value, weight in entries:
                self.lsp.add(name, value, weight=weight)
            return self.lsp.total()
        return sum(
            (value * weight for _, value, weight in entries),
            start=entries[0][1].new_zeros(()),
        )


class TimeBudgetExitModel(_BaseExitModel):
    """双塔预算存活 x 内容留存模型，原生集成 LossPort 与可消融辅助监督。"""

    def __init__(
        self,
        manager: SchemaManager,
        *,
        num_components: int = 1,
        step_only: bool = True,
        add_pace: bool = False,
        left_weight: float = 0.0,
        right_weight: float = 0.0,
        requests_per_session: float = 1.0,
        content_init_bias: float = 3.0,
        label_field: Optional[str] = None,
        structured: bool = False,
        coupled_bathtub: bool = True,
        detach_budget_in_joint: bool = False,
        max_aux_ratio: Optional[float] = None,
    ) -> None:
        if left_weight < 0 or right_weight < 0:
            raise ValueError("Auxiliary loss weights must be non-negative")
        if requests_per_session <= 0:
            raise ValueError("requests_per_session must be positive")
        if left_weight and not step_only:
            raise ValueError("left_session_nll requires step_only=True")
        if max_aux_ratio is not None and (not math.isfinite(max_aux_ratio) or not 0 < max_aux_ratio <= .1):
            raise ValueError("max_aux_ratio must be in (0, 0.1]")

        super().__init__(manager, add_pace=add_pace, label_field=label_field)
        self.num_components = int(num_components)
        self.step_only = bool(step_only)
        self.left_weight = float(left_weight)
        self.right_weight = float(right_weight)
        self.requests_per_session = float(requests_per_session)
        self.detach_budget_in_joint = bool(detach_budget_in_joint)
        self.max_aux_ratio = max_aux_ratio

        # 1. 独立预算 OmniEmbLayer：同名背景字段从同一初值出发，训练参数互不共享
        budget_settings = self.omni_embedding.get_settings(include_fields=BUDGET_PARAMETER_FIELDS)
        self.budget_embedding = OmniEmbLayer(emb_settings=budget_settings)
        for name, table in self.budget_embedding.emb_modules.items():
            table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
        self.budget_view = EmbView(self.budget_embedding, include_fields=BUDGET_PARAMETER_FIELDS)
        self.content_view = EmbView(self.omni_embedding, include_fields=CONTENT_TOWER_FIELDS)

        # 2. 左塔（预算塔）：按时钟配置只建活跃 head，不分配死参数
        self.budget_encoder = nn.Sequential(
            nn.Linear(self.budget_view.embedding_dim, 64),
            nn.SiLU(),
        )
        # 历史初始化先抽两个时钟的四个通道，内容网在其后初始化。
        # 保留新架构的独立 WeibullHead，只将原始抽样映射到活跃通道。
        original_head = nn.Linear(64, 4)

        # 3. 右塔（内容塔）
        self.content_net = nn.Sequential(
            nn.Linear(self.content_view.embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.content_net[-1].bias, content_init_bias)
        # 旧模型初始化顺序包含一个未参与双塔预测的 Fusion MLP。
        # 只重放随机抽样，不保存模块或增加模型参数。
        nn.Sequential(nn.Linear(self.omni_embedding.whole.embedding_dim, 64), nn.SiLU(),
                      nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        with torch.random.fork_rng():
            self.step_head = WeibullHead(64, self.num_components, structured=structured, coupled_bathtub=coupled_bathtub)
            self.time_head: Optional[WeibullHead] = (
                None if self.step_only else WeibullHead(64, self.num_components, structured=structured, coupled_bathtub=coupled_bathtub)
            )
            self.feedback_projection = nn.Linear(len(FEEDBACK_FIELDS), 64, bias=False)
        nn.init.zeros_(self.feedback_projection.weight)
        if not structured:
            with torch.no_grad():
                for head, clock in [(self.step_head, 1), (self.time_head, 0)]:
                    if head is None:
                        continue
                    for component, shift in enumerate(torch.linspace(-math.log(2), math.log(2), self.num_components)
                                                       if self.num_components > 1 else [0.]):
                        target = slice(component * 2, component * 2 + 2)
                        head.proj.weight[target].copy_(original_head.weight[clock * 2:clock * 2 + 2])
                        head.proj.bias[target].copy_(original_head.bias[clock * 2:clock * 2 + 2])
                        head.proj.bias[component * 2].add_(shift)
                    if self.num_components > 1:
                        head.proj.weight[2 * self.num_components:].zero_()
                        head.proj.bias[2 * self.num_components:].zero_()
        if self.add_pace:
            original = self.content_net[0]
            with torch.random.fork_rng():
                expanded = nn.Linear(original.in_features + 1, original.out_features)
            with torch.no_grad():
                expanded.weight[:, :-1].copy_(original.weight)
                expanded.weight[:, -1].zero_()
                expanded.bias.copy_(original.bias)
            self.content_net[0] = expanded

    @property
    def budget_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [self.budget_embedding, self.budget_encoder, self.step_head]
        if self.time_head is not None:
            modules.append(self.time_head)
        return tuple(modules)

    def forward_components(self, interaction: Interaction) -> ExitOutput:
        budget_h = self.budget_encoder(self.budget_view(interaction))
        step_raw = self.step_head.raw(budget_h)
        step = (interaction["req_idx_raw"].float() - 1.0).clamp(min=0.0)
        step_surv = self.step_head.survival(step_raw, step)

        if self.time_head is not None:
            time_raw = self.time_head.raw(budget_h)
            elapsed = interaction["elapsed_min_raw"].float().clamp(min=0.0)
            time_surv: Optional[WeibullSurvival] = self.time_head.survival(time_raw, elapsed)
            gamma = time_surv.gamma * step_surv.gamma
            log_s = time_surv.log_survival + step_surv.log_survival
        else:
            time_surv = None
            gamma = step_surv.gamma
            log_s = step_surv.log_survival

        content_x = self._with_pace(self.content_view(interaction), interaction)
        feedback = torch.stack([interaction[name] for name in FEEDBACK_FIELDS], dim=-1)
        hidden = self.content_net[0](content_x) + self.feedback_projection(feedback)
        stay_logit = self.content_net[1:](hidden).squeeze(-1)
        stay = torch.sigmoid(stay_logit)

        return ExitOutput(
            prediction=gamma * stay,
            gamma=gamma,
            stay=stay,
            stay_logit=stay_logit,
            log_survival=log_s,
            step_raw=step_raw,
            step_survival=step_surv,
            time_survival=time_surv,
        )

    def predict(self, interaction: Interaction) -> torch.Tensor:
        return self.forward_components(interaction).prediction

    def calculate_loss(self, interaction: Interaction) -> torch.Tensor:
        out = self.forward_components(interaction)
        target, mask = self._target_and_mask(interaction)

        joint_pred = (out.gamma.detach() * out.stay
                      if self.detach_budget_in_joint and self.left_weight > 0 else out.prediction)
        joint = _masked_bce_from_prob(joint_pred, target, mask)
        entries: list[tuple[str, torch.Tensor, float]] = [("joint_bce", joint, 1.0)]
        auxiliary = joint.new_zeros((), dtype=torch.float64)

        if self.left_weight:
            terminal = (target == 0) & (mask > 0)
            if terminal.any():
                nll_sum = -self.step_head.discrete_log_mass(
                    self.step_head.raw(self.budget_encoder(self.budget_view(interaction[terminal]))),
                    interaction["req_idx_raw"][terminal],
                ).sum()
                left = nll_sum * self.requests_per_session / mask.sum().clamp(min=1.0)
            else:
                left = joint.new_zeros(())
            auxiliary = auxiliary + self.left_weight * left

        if self.right_weight:
            right = _masked_bce_from_prob(out.stay, target, mask)
            auxiliary = auxiliary + self.right_weight * right

        if self.max_aux_ratio is not None:
            cap_scale = torch.minimum(torch.ones_like(auxiliary),
                self.max_aux_ratio * joint.detach().double() / auxiliary.detach().clamp(min=torch.finfo(torch.float64).tiny))
            auxiliary = cap_scale * auxiliary
        entries.append(("auxiliary", auxiliary, 1.0))

        return self._finalize_loss(entries)


class FusionExitModel(_BaseExitModel):
    """全特征单塔基线对照，与 TimeBudgetExitModel 共享同一 schema 与 LossPort 契约。"""

    def __init__(
        self,
        manager: SchemaManager,
        *,
        add_pace: bool = False,
        label_field: Optional[str] = None,
    ) -> None:
        super().__init__(manager, add_pace=add_pace, label_field=label_field)
        self.fusion_view = self.omni_embedding.whole
        fusion_in = self.fusion_view.embedding_dim + (1 if self.add_pace else 0)
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        with torch.random.fork_rng():
            self.feedback_projection = nn.Linear(len(FEEDBACK_FIELDS), 64, bias=False)
        nn.init.zeros_(self.feedback_projection.weight)

    def logits(self, interaction: Interaction) -> torch.Tensor:
        x = self._with_pace(self.fusion_view(interaction), interaction)
        feedback = torch.stack([interaction[name] for name in FEEDBACK_FIELDS], dim=-1)
        hidden = self.fusion_net[0](x) + self.feedback_projection(feedback)
        return self.fusion_net[1:](hidden).squeeze(-1)

    def predict(self, interaction: Interaction) -> torch.Tensor:
        return torch.sigmoid(self.logits(interaction))

    def calculate_loss(self, interaction: Interaction) -> torch.Tensor:
        target, mask = self._target_and_mask(interaction)
        loss = _masked_bce_from_logits(self.logits(interaction), target, mask)
        return self._finalize_loss([("joint_bce", loss, 1.0)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KR next-continue v1 数据预处理")
    parser.add_argument("--source", type=Path, default=KuaiRandDataset.DATA_KUAI_1K)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    print("数据已生成：", prepare_data(args.source, args.output))
