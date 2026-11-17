"""ML1M已确认的Session口径：30分钟空闲、一条评分一步。

只处理会话和顺序，不定义模型特征、训练标签、负样本或train/test切分。
同用户相邻评分间隔>1800秒开启新场；=1800秒仍留在原场。
同秒评分按ratings.dat原始行序稳定排列，这不是可观测的真实先后。
elapsed_min仅为评分时间戳差，不是观影分钟；首步为0。
所有原始评分保留，不过滤单步/零跨度场；用户末场只标观察边界。
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from rosen.datasets.base import DatasetBase

WORKSPACE = Path(__file__).resolve().parents[3] / 'workspace/ml1m'
GAP_SECONDS = 1800


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ratings(path):
    return pl.read_csv(path, separator=':', has_header=False,
        new_columns=['user_id', '_pad1', 'movie_id', '_pad2', 'rating', '_pad3', 'timestamp'],
        schema_overrides={c: pl.Int64 for c in ['user_id', 'movie_id', 'rating', 'timestamp']}
    ).select('user_id', 'movie_id', 'rating', 'timestamp').with_row_index('event_id').with_columns(
        pl.col('event_id').cast(pl.Int64))


def sessionize(ratings):
    ordered = ratings.sort('user_id', 'timestamp', 'event_id').with_columns(
        pl.col('timestamp').diff().over('user_id').alias('previous_gap_seconds'))
    ordered = ordered.with_columns(
        (pl.col('previous_gap_seconds').is_null() | (pl.col('previous_gap_seconds') > GAP_SECONDS))
        .alias('is_session_start'))
    events = ordered.with_columns(
        pl.col('is_session_start').cast(pl.Int64).cum_sum().alias('session_id'))
    events = events.with_columns(
        pl.col('event_id').cum_count().over('session_id').cast(pl.Int64).alias('step_idx'),
        ((pl.col('timestamp') - pl.col('timestamp').first().over('session_id')) / 60.)
        .alias('elapsed_min'),
        (pl.col('timestamp') == pl.col('timestamp').shift(1).over('user_id')).fill_null(False)
        .alias('same_timestamp_as_previous'))
    sessions = events.group_by('session_id', maintain_order=True).agg(
        pl.col('user_id').first(), pl.len().cast(pl.Int64).alias('steps'),
        pl.col('timestamp').first().alias('start_timestamp'),
        pl.col('timestamp').last().alias('last_rating_timestamp'),
        pl.col('event_id').first().alias('first_event_id'),
        pl.col('event_id').last().alias('last_event_id'),
        pl.col('timestamp').n_unique().alias('distinct_timestamps'))
    sessions = sessions.with_columns(
        ((pl.col('last_rating_timestamp') - pl.col('start_timestamp')) / 60.).alias('span_min'),
        (pl.col('session_id') == pl.col('session_id').first().over('user_id')).alias('first_observed_session'),
        (pl.col('session_id') == pl.col('session_id').last().over('user_id')).alias('last_observed_session'))
    return events, sessions


def audit(ratings, events, sessions):
    original_columns = ratings.columns
    assert events.sort('event_id').select(original_columns).equals(ratings.sort('event_id'))
    assert events.height == ratings.height and events['event_id'].n_unique() == ratings.height
    assert sessions['steps'].sum() == ratings.height
    assert events['is_session_start'].sum() == sessions.height
    first = events.filter(pl.col('step_idx') == 1)
    assert first.height == sessions.height and first['elapsed_min'].sum() == 0
    assert events.filter(~pl.col('is_session_start') & (pl.col('previous_gap_seconds') > GAP_SECONDS)).is_empty()
    assert events.filter(pl.col('is_session_start') & pl.col('previous_gap_seconds').is_not_null()
                         & (pl.col('previous_gap_seconds') <= GAP_SECONDS)).is_empty()
    assert sessions['last_observed_session'].sum() == ratings['user_id'].n_unique()
    # 独立NumPy差分计算边界和步号，避免只检查自身聚合结果。
    expected = ratings.sort('user_id', 'timestamp', 'event_id')
    u, t = expected['user_id'].to_numpy(), expected['timestamp'].to_numpy()
    starts = np.r_[True, (np.diff(u) != 0) | (np.diff(t) > GAP_SECONDS)]
    session_ids = np.cumsum(starts)
    indices = np.flatnonzero(starts)
    sizes = np.diff(np.r_[indices, len(t)])
    steps = np.arange(len(t)) - np.repeat(indices, sizes) + 1
    np.testing.assert_array_equal(events['session_id'], session_ids)
    np.testing.assert_array_equal(events['step_idx'], steps)
    np.testing.assert_array_equal(sessions['steps'], sizes)
    return dict(original_rows_and_values_preserved=True, independent_numpy_session_and_step_exact=True,
                every_first_step_elapsed_zero=True, thresholds_and_observation_flags_verified=True)


def build(source, output):
    hashes = {str(p.resolve()): sha(p) for p in [source, Path(__file__)]}
    ratings = read_ratings(source)
    events, sessions = sessionize(ratings)
    checks = audit(ratings, events, sessions)
    for frame, name in [(events, 'events'), (sessions, 'sessions')]:
        path = output / f'{name}.parquet'
        frame.write_parquet(path, row_group_size=65536)
        assert pl.read_parquet(path).equals(frame)
    stats = dict(ratings=events.height, users=events['user_id'].n_unique(), sessions=sessions.height,
        mean_steps=float(sessions['steps'].mean()), median_steps=float(sessions['steps'].median()),
        p90_steps=float(sessions['steps'].quantile(.9, interpolation='linear')),
        singleton_sessions=sessions.filter(pl.col('steps') == 1).height,
        zero_span_sessions=sessions.filter(pl.col('span_min') == 0).height,
        tied_adjacent_pairs=int(events['same_timestamp_as_previous'].sum()),
        last_observed_sessions=int(sessions['last_observed_session'].sum()))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), version=1,
        protocol=dict(inactivity_seconds=GAP_SECONDS, boundary='strictly greater',
            step='one rating', order=['user_id', 'timestamp', 'event_id (original zero-based line index)'],
            same_second_order='source line order only; no claim of true chronological ordering',
            elapsed_min='(current rating timestamp - first rating timestamp in session) / 60',
            filters=None, split=None, label=None, schema=None, training=False,
            endpoint='last observed rating, not a known physical exit time',
            boundary_flags='first/last observed sessions are markers, not censoring/exit labels'),
        statistics=stats, checks=checks, source_hashes=hashes,
        output_sha256={name: sha(output/f'{name}.parquet') for name in ['events', 'sessions']})
    assert all(sha(p) == h for p, h in hashes.items())
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
        default=DatasetBase.SYSTEM_DATA_DIR/'MovieLens/ml-1m/ratings.dat')
    parser.add_argument('--output', type=Path, default=WORKSPACE/'data/sessions_30min_v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'logs').mkdir()
    with (args.output/'logs/build.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        build(args.source, args.output)
    print('完成：', args.output/'manifest.json')


if __name__ == '__main__':
    main()
