"""nxt_continue_v1 的 raw -> parquet -> Interaction 链路。"""
from datetime import datetime, timezone
import json

import polars as pl
import pytest
import torch

from experiments.kuairand.nxt_continue_v1 import (
    BUDGET_PRIOR_FIELDS,
    FEATURES,
    FEEDBACK_FIELDS,
    USER_CATEGORY_FIELDS,
    FusionExitModel,
    TimeBudgetExitModel,
    _aggregate_requests,
    _encode_requests,
    _item_priors,
    _sessionize,
    _user_priors,
    load_data,
    prepare_data,
)


@pytest.fixture
def raw_source(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    rows = []
    for uid in (1, 2):
        for date in (20220501, 20220505, 20220506, 20220507, 20220508, 20220509):
            start = int(datetime.strptime(str(date), "%Y%m%d").replace(tzinfo=timezone.utc, hour=10).timestamp() * 1000)
            count = 13 if date == 20220505 else 12
            for i in range(count):
                rows.append(dict(
                    user_id=uid, video_id=i % 6 + 1, date=date, hourmin=1000 + i // 6,
                    time_ms=start + i * 10000, play_time_ms=5000, duration_ms=30000,
                    tab=1, is_click=i % 2, long_view=i % 3 == 0, is_like=i % 4 == 0,
                    is_comment=0, is_follow=0, is_forward=0, is_hate=0, is_rand=0,
                ))
    # CSV 内重复事件必须先去重，再切场和组屏。
    rows.append(rows[-1].copy())
    pl.DataFrame(rows).with_columns(pl.col(pl.Boolean).cast(pl.Int64)).write_csv(source / "log_fixture.csv")
    pl.DataFrame({
        "video_id": list(range(1, 7)), "author_id": list(range(11, 17)),
        "tag": ["1,2", "2", "3", "4", "5", "6"], "video_type": ["NORMAL"] * 6, "music_type": [1] * 6,
    }).write_csv(source / "video_features_basic_1k.csv")
    profiles = [{"user_id": uid, **{c: 1 for c in USER_CATEGORY_FIELDS}} for uid in (1, 2)]
    pl.DataFrame(profiles).write_csv(source / "user_features_1k.csv")
    return source


def test_preprocessing_materializes_pace_and_models_consume_the_column(raw_source, tmp_path):
    output = prepare_data(raw_source, tmp_path / "processed")
    raw = pl.read_parquet(output / "features_raw.parquet")
    assert raw.group_by("split").len().sort("split")["len"].to_list() == [4, 8, 4]
    assert raw["pace"].dtype == pl.Float32
    expected = (raw["elapsed_min_raw"] / raw["req_idx_raw"]).log1p()
    torch.testing.assert_close(torch.tensor(raw["pace"].to_numpy()), torch.tensor(expected.to_numpy()).float())
    events = pl.read_parquet(output / "events.parquet")
    assert events.height == 146
    # 时钟不加终视频播放时长，head remainder 也在 session 起点内。
    first = raw.filter((pl.col("split") == "train") & (pl.col("req_idx_raw") == 1)).sort("session_id").row(0, named=True)
    assert first["elapsed_min_raw"] == pytest.approx(1.0)
    assert first["prev1_click_cnt"] == 0
    assert raw.filter(pl.col("split") == "test")["session_date"].unique().to_list() == [20220508]
    for split in ("train", "valid", "test"):
        frame = pl.read_parquet(output / f"{split}.parquet")
        assert frame["pace"].dtype == pl.Float32
        if split != "train":
            assert frame.equals(frame.sort("session_id", "req_idx_raw"))
    manager, loaders = load_data(output, batch_size=4)
    assert len(manager.settings) == len(FEATURES) == 54
    assert manager.get_aux_spec("pace").dtype == "float32"
    batch = next(iter(loaders["train"]))
    assert batch["pace"].dtype == torch.float32
    assert batch["req_idx_raw"].dtype == torch.int64
    for model_class in (FusionExitModel, TimeBudgetExitModel):
        model = model_class(manager, add_pace=True)
        # 模型必须取物化列；故意让此列与 raw 时钟不一致以检测隐式重算。
        batch["pace"] = torch.full_like(batch["pace"], 7.25)
        joined = model._with_pace(torch.zeros(len(batch["pace"]), 2), batch)
        assert torch.equal(joined[:, -1], batch["pace"])
        loss = model.calculate_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(FileExistsError):
        prepare_data(raw_source, output)


def test_encoding_canonicalizes_order_before_seeded_shuffle(raw_source, tmp_path):
    original = prepare_data(raw_source, tmp_path / "original")
    raw = pl.read_parquet(original / "features_raw.parquet")
    reordered = tmp_path / "reordered"
    reordered.mkdir()
    _encode_requests(raw.sample(fraction=1, shuffle=True, seed=71), reordered, seed=2026)
    for split in ("train", "valid", "test"):
        expected = pl.read_parquet(original / f"{split}.parquet")
        actual = pl.read_parquet(reordered / f"{split}.parquet")
        assert actual.equals(expected)
    assert json.loads((original / "schema/feature_meta.json").read_text()) == json.loads(
        (reordered / "schema/feature_meta.json").read_text()
    )


def test_request_feedback_is_historical_and_pace_ignores_play_duration():
    events = pl.DataFrame({
        "user_id": [1] * 13, "video_id": list(range(13)), "time_ms": [i * 10000 for i in range(13)],
        "source_row": list(range(13)), "source_order": [0] * 13, "play_time_ms": [5000] * 13,
        "duration_ms": [30000] * 13, "tag": [str(i) for i in range(13)],
        "video_type": ["NORMAL"] * 13, "music_type": ["1"] * 13, "author_id": [1] * 13,
        "is_click": [1] * 13, "long_view": [1] * 13, "positive": [1] * 13, "is_hate": [0] * 13, "is_rand": [0] * 13,
    })
    ordered = _sessionize(events)
    before = _aggregate_requests(ordered)
    changed = ordered.with_columns(
        pl.when(pl.col("req_idx") == 1).then(0).otherwise(pl.col("is_click")).alias("is_click"),
        pl.when(pl.col("req_idx") == 1).then(0).otherwise(pl.col("long_view")).alias("long_view"),
        pl.when(pl.col("req_idx") == 1).then(0).otherwise(pl.col("positive")).alias("positive"),
        pl.lit(300000).alias("play_clip"),
    )
    after = _aggregate_requests(changed)
    # 当前反馈是允许的决策输入，但不能写回本刷的历史字段。
    assert before.filter(pl.col("req_idx") == 1).drop(FEEDBACK_FIELDS).equals(
        after.filter(pl.col("req_idx") == 1).drop(FEEDBACK_FIELDS))
    assert before["current_click_rate"].to_list() == [1., 1.]
    assert after["current_click_rate"].to_list() == [0., 1.]
    assert before["prev1_click_cnt"].to_list() == [1, 6]
    assert after["prev1_click_cnt"].to_list() == [1, 0]
    assert before["y_continue"].to_list() == [1, 0]
    assert before["elapsed_min_raw"].to_list() == pytest.approx([1, 2])
    assert before["pace"].to_list() == after["pace"].to_list()


def test_priors_do_not_read_query_day_or_future():
    query = pl.DataFrame({"request_id": [0, 1], "user_id": [1, 1], "date": [20220506, 20220507]})
    sessions = pl.DataFrame({
        "user_id": [1, 1, 1], "end_date": [20220505, 20220506, 20220507],
        "session_min": [2.0, 4.0, 99.0], "session_req_cnt": [2.0, 4.0, 99.0],
    })
    original = _user_priors(query, sessions).sort("request_id")
    changed = _user_priors(query, sessions.with_columns(
        pl.when(pl.col("end_date") >= 20220507).then(1000).otherwise(pl.col("session_min")).alias("session_min")
    )).sort("request_id")
    assert original.equals(changed)
    assert original[BUDGET_PRIOR_FIELDS[0]].to_list() == [2.0, 3.0]
    item_query = query.select("request_id", "date").with_columns(pl.lit([1, 1, 1, 1, 1, 1]).alias("slate_items"))
    observations = pl.DataFrame({"video_id": [1, 1, 1], "date": [20220505, 20220506, 20220507], "is_click": [0, 1, 1], "long_view": [0, 1, 1]})
    prior = _item_priors(item_query, observations).sort("request_id")
    future_changed = _item_priors(item_query, observations.with_columns(
        pl.when(pl.col("date") >= 20220507).then(0).otherwise(pl.col("is_click")).alias("is_click")
    )).sort("request_id")
    assert prior.equals(future_changed)
    assert prior["slate_prior_ctr_mean"].to_list() == [0.0, 0.5]
