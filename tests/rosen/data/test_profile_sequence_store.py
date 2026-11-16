from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rosen.data import ProfileSequenceStore
from rosen.emb.schema import ProfileSeqGroupConfig


def _manager(history_time_filter: str, max_len: int = 3):
    group = ProfileSeqGroupConfig(
        group_name="click_history",
        seq_len_field_name="seq_len",
        max_len=max_len,
        time_field_name="time_ms_seq",
        owner_field_name="user_id",
        anchor_field_name="time_ms",
        history_time_filter=history_time_filter,
    )
    settings = [
        SimpleNamespace(
            is_sequence_setting=True,
            group_name=group.group_name,
            group=group,
            seq_len_field_name=group.seq_len_field_name,
            max_len=group.max_len,
            field_name=field,
            time_field_name=group.time_field_name,
            truncate_mode="tail",
        )
        for field in ("video_id_seq", "tab_seq")
    ]
    return SimpleNamespace(settings=settings, uid_field="user_id")


def _profile_root(tmp_path):
    group_dir = tmp_path / "click_history"
    group_dir.mkdir()
    table = pa.table({
        "user_id": pa.array([1], type=pa.int64()),
        "video_id_seq": pa.array([[10, 11, 12, 13, 14]], type=pa.list_(pa.int64())),
        "tab_seq": pa.array([[1, 2, 3, 4, 5]], type=pa.list_(pa.int64())),
        "time_ms_seq": pa.array(
            [[100, 150, 200, 300, 300]],
            type=pa.list_(pa.int64()),
        ),
    })
    pq.write_table(table, group_dir / "part-000.parquet")
    return tmp_path


def _batch():
    return {
        "user_id": np.asarray([1]),
        "seq_len": np.asarray([5]),
        "time_ms": np.asarray([300]),
    }


def test_profile_store_filters_before_tail_truncation(tmp_path) -> None:
    store = ProfileSequenceStore(
        _manager("drop_same_ts", max_len=3),
        _profile_root(tmp_path),
    )

    batch = store.inject(_batch())

    assert batch["seq_len"].tolist() == [3]
    assert batch["video_id_seq"][0].tolist() == [10, 11, 12]
    assert batch["tab_seq"][0].tolist() == [1, 2, 3]
    assert batch["time_ms_seq"][0].tolist() == [100, 150, 200]


def test_profile_store_supports_cursor_and_same_timestamp_policies(tmp_path) -> None:
    profile_root = _profile_root(tmp_path)

    cursor_batch = ProfileSequenceStore(
        _manager("cursor_order", max_len=3),
        profile_root,
    ).inject(_batch())
    same_time_batch = ProfileSequenceStore(
        _manager("same_ts_only", max_len=1),
        profile_root,
    ).inject(_batch())

    assert cursor_batch["video_id_seq"][0].tolist() == [12, 13, 14]
    assert same_time_batch["video_id_seq"][0].tolist() == [14]
    assert same_time_batch["seq_len"].tolist() == [1]
