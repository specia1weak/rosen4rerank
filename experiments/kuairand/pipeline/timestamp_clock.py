"""KR当前分钟定义：纯事件时间戳差，不加任何播放时长。

仍沿用已有会话/六视频屏成员。当前屏观测时刻取该屏最后一条事件时间戳，
会话起点包含头部历史片段；这是屏末事件观测时刻，不是估计播放结束时刻。
会话切分属于独立协议，本模块不改写历史session_id。
"""
import json
from pathlib import Path

import polars as pl
from torch.utils.data import DataLoader

from rosen.data.dataset import ParquetStreamDataset
from experiments.kuairand.pipeline.kr_request_schema import EXTRA_COLUMNS


CLOCK = 'screen_last_event_timestamp_minus_session_first_timestamp; no_play_duration'


def elapsed_minutes(timestamp='request_time_ms', origin='session_start_ms'):
    return ((pl.col(timestamp) - pl.col(origin)) / 60000).alias('elapsed_min')


def timestamp_boundaries(events):
    """events只需ID/时间列；max时间戳与已按时间排序的末条事件一致。"""
    sessions = events.group_by('session_id').agg(
        pl.col('time_ms').min().alias('session_start_ms'),
        pl.col('time_ms').max().alias('session_last_time_ms'),
    ).with_columns(
        ((pl.col('session_last_time_ms')-pl.col('session_start_ms'))/60000).alias('session_min'))
    requests = (events.group_by('session_id', 'req_idx').agg(
        pl.col('time_ms').max().alias('request_time_ms'))
        .join(sessions.select('session_id', 'session_start_ms'), on='session_id', validate='m:1')
        .with_columns(elapsed_minutes()))
    return sessions, requests


def load_timestamp_batches(data, manager, row_limits=None):
    """读取明确标记的新口径，不冒充历史v3契约；不重新fit或log1p。"""
    data = Path(data)
    manifest = json.loads((data/'manifest.json').read_text())
    if manifest['version'] != 4 or manifest['protocol']['clock'] != CLOCK:
        raise ValueError('Requires KR v4 timestamp-only minute data')
    setting = manager.get_setting('elapsed_min')
    if (setting.min_val, setting.max_val) != (0., 1.):
        raise ValueError('Minute inputs must be unscaled Float32 timestamp differences')
    batches = {}
    for split in ['train', 'valid', 'test']:
        frame = pl.scan_parquet(data/f'{split}.parquet').with_columns(pl.col('elapsed_min').alias('elapsed_min_raw'))
        if row_limits is not None and split in row_limits:
            frame = frame.head(row_limits[split])
        dataset = ParquetStreamDataset(frame, manager, batch_size=4096, shuffle=False,
            drop_last=False, extra_col_names=EXTRA_COLUMNS+['elapsed_min_raw'])
        batches[split] = list(DataLoader(dataset, batch_size=None, num_workers=0))
    return batches
