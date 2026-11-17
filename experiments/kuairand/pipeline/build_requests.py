"""KR session/request reconstruction. datasource is read-only."""
import argparse
import json
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold
from rosen.datasets.kuairand import KuaiRandDataset

ROOT = (Path(__file__).resolve().parents[3] / 'workspace/kr_time_budget/data/v1')
SOURCE = KuaiRandDataset.DATA_KUAI_1K


def sessionize(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(['user_id', 'time_ms', 'source_order', 'source_row'])
    df = df.with_columns((pl.col('play_time_ms').clip(0, 300000)).alias('play_clip'))
    gap = pl.col('time_ms') - (pl.col('time_ms').shift(1).over('user_id') + pl.col('play_clip').shift(1).over('user_id'))
    df = df.with_columns((gap.is_null() | (gap > 1800000)).cast(pl.Int64).alias('new_session'))
    df = df.with_columns(pl.col('new_session').cum_sum().alias('session_id'))
    return df.with_columns(
        pl.len().over('session_id').alias('session_n'),
        pl.int_range(pl.len()).over('session_id').alias('event_idx'),
        pl.col('time_ms').first().over('session_id').alias('session_start_ms'),
        pl.col('session_id').first().over('user_id').alias('first_session'),
        pl.col('session_id').last().over('user_id').alias('last_session'),
    ).with_columns(
        ((pl.col('event_idx').cast(pl.Int64) - (pl.col('session_n') % 6).cast(pl.Int64)) // 6 + 1).alias('req_idx')
    )


def aggregate_requests(events: pl.DataFrame) -> pl.DataFrame:
    """req_idx=0 holds the head remainder and participates only in history."""
    df = events.group_by(['user_id', 'session_id', 'req_idx'], maintain_order=True).agg(
        pl.len().alias('n_items'),
        pl.col('video_id').alias('slate_items'), pl.col('tag').alias('slate_tags'),
        pl.col('dur_s').alias('slate_durations'),
        pl.col('dur_s').sum().alias('slate_dur_sum_s'),
        pl.col('tag').n_unique().alias('slate_unique_tag_cnt'),
        pl.col('author_id').n_unique().alias('slate_unique_author_cnt'),
        pl.col('is_rand').max().alias('slate_has_rand'),
        pl.col('time_ms').first().alias('request_start_ms'),
        ((pl.col('time_ms').last() - pl.col('session_start_ms').first() + pl.col('play_clip').last()) / 1000).alias('elapsed_end_s_DIAGNOSTIC'),
        ((pl.col('time_ms').first() - pl.col('session_start_ms').first()) / 1000).alias('elapsed_time_s'),
        (pl.col('play_time_ms').clip(lower_bound=0).sum() / 1000).alias('curr_watch_s'),
        pl.col('comp').sum().alias('comp_sum'), pl.col('long_view').sum().alias('long_sum'),
        pl.col('pos').sum().alias('pos_sum'), pl.col('is_hate').sum().alias('hate_sum'),
        *[pl.col(c).first() for c in ['date', 'session_date', 'session_hour', 'session_tab', 'session_start_ms', 'session_wall_s', 'session_requests', 'day_prev_sess_cnt', 'day_prev_time_s']],
    ).sort(['session_id', 'req_idx'])
    for c in ['curr_watch_s', 'comp_sum', 'pos_sum', 'hate_sum', 'n_items']:
        df = df.with_columns((pl.col(c).cum_sum().over('session_id') - pl.col(c)).alias('past_' + c))
    df = df.with_columns(
        pl.col('curr_watch_s').shift(1).over('session_id').fill_null(0).alias('prev1_play_s'),
        (pl.col('comp_sum') / pl.col('n_items')).shift(1).over('session_id').fill_null(0).alias('prev1_comp_rate'),
        (pl.col('long_sum') / pl.col('n_items')).shift(1).over('session_id').fill_null(0).alias('prev1_long_view_rate'),
        pl.col('pos_sum').shift(1).over('session_id').fill_null(0).alias('prev1_pos_cnt'),
        (pl.col('past_n_items') == 0).cast(pl.Int8).alias('is_first_req'),
        (pl.col('past_comp_sum') / pl.col('past_n_items').clip(lower_bound=1)).alias('sess_cum_comp_rate'),
        (pl.col('past_pos_sum') / pl.col('past_n_items').clip(lower_bound=1)).alias('sess_cum_pos_rate'),
        pl.col('past_hate_sum').alias('sess_cum_hate_cnt'),
        pl.col('curr_watch_s').reverse().cum_sum().reverse().over('session_id').alias('remaining_watch_s'),
        (pl.col('req_idx') < pl.col('req_idx').max().over('session_id')).cast(pl.Int8).alias('y_continue'),
    )
    # The previous two chunks, including a possible head remainder; <=12 events.
    for raw, out in [('slate_items', 'history_items'), ('slate_tags', 'history_tags')]:
        df = df.with_columns(pl.concat_list([
            pl.col(raw).shift(2).over('session_id').fill_null([]),
            pl.col(raw).shift(1).over('session_id').fill_null([]),
        ]).alias(out))
    return df.filter(pl.col('req_idx') > 0).with_columns(
        (pl.col('remaining_watch_s') - pl.col('curr_watch_s')).alias('future_watch_s')
    ).drop(['comp_sum', 'long_sum', 'pos_sum', 'hate_sum'] + ['past_' + c for c in ['curr_watch_s', 'comp_sum', 'pos_sum', 'hate_sum', 'n_items']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--output', type=Path, default=ROOT)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'requests.parquet').exists() or (out / 'manifest.json').exists():
        raise FileExistsError(f'{out} already contains a dataset; use a fresh --output to avoid stale fold schemas')
    paths = sorted(args.source.glob('log_*.csv'))
    cols = ['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'play_time_ms', 'duration_ms', 'tab', 'is_like', 'is_comment', 'is_forward', 'is_follow', 'is_hate', 'long_view', 'is_rand']
    frames = []
    for i, p in enumerate(paths):
        f = pl.read_csv(p, columns=cols).with_row_index('source_row').with_columns(pl.lit(i).alias('source_order'))
        print(p.name, f.height, flush=True)
        frames.append(f)
    df = pl.concat(frames)
    del frames
    raw_rows = df.height
    # Deduplicate the modeled raw projection; omitted raw feedback columns are
    # not part of this key. Record this rule rather than calling it byte-exact.
    df = df.unique(subset=cols, maintain_order=True)
    dedup_rows = df.height
    df = sessionize(df)
    sessions = df.group_by(['user_id', 'session_id'], maintain_order=True).agg(
        pl.col('date').first().alias('session_date'),
        (pl.col('hourmin').first() // 100).alias('session_hour'),
        pl.col('tab').first().alias('session_tab'),
        ((pl.col('time_ms').last() - pl.col('time_ms').first() + pl.col('play_clip').last()) / 1000).alias('session_wall_s'),
        (pl.len() // 6).alias('session_requests'),
        pl.col('date').max().cast(pl.String).str.strptime(pl.Date, '%Y%m%d').dt.offset_by('1d').dt.strftime('%Y%m%d').cast(pl.Int64).alias('prior_available_date'),
        pl.col('first_session').first(), pl.col('last_session').first(), pl.len().alias('session_n'),
    ).sort('session_id')
    sessions = sessions.with_columns(
        (pl.col('session_id').cum_count().over(['user_id', 'session_date']) - 1).alias('day_prev_sess_cnt'),
        (pl.col('session_wall_s').cum_sum().over(['user_id', 'session_date']) - pl.col('session_wall_s')).alias('day_prev_time_s'),
    )
    sessions.write_parquet(out / 'sessions.parquet')
    print('sessions', sessions.height, 'events', df.height, flush=True)
    df = df.join(sessions.drop(['session_n', 'first_session', 'last_session']), on=['user_id', 'session_id'], how='left', validate='m:1')
    video = pl.read_csv(args.source / 'video_features_basic_1k.csv', columns=['video_id', 'author_id', 'tag'], schema_overrides={'tag': pl.String})
    assert video['video_id'].n_unique() == video.height
    df = df.join(video, on='video_id', how='left', validate='m:1').with_columns(
        pl.col('tag').fill_null('__MISSING__'), pl.col('author_id').fill_null(-1),
        (pl.col('duration_ms') / 1000).clip(3, 600).alias('dur_s'),
        pl.sum_horizontal('is_like', 'is_comment', 'is_forward', 'is_follow').alias('pos'),
    ).with_columns((pl.col('play_clip') / (pl.col('dur_s') * 1000)).clip(0, 1).alias('comp'))
    # Retain all observed events for strictly earlier-day, training-user priors.
    df.select(['user_id', 'video_id', 'date', 'session_id', 'req_idx', 'comp', 'long_view', 'is_like']).write_parquet(out / 'prior_events.parquet')
    # First/last observed sessions can be left/right censored at observation edges.
    usable = df.filter((pl.col('session_id') != pl.col('first_session')) & (pl.col('session_id') != pl.col('last_session')) & (pl.col('session_n') >= 6))
    req = aggregate_requests(usable)
    del df, usable
    users = pl.read_csv(args.source / 'user_features_1k.csv')
    keep = ['user_id', 'user_active_degree', 'is_live_streamer', 'is_video_author'] + [f'onehot_feat{i}' for i in range(18)] + ['register_days', 'fans_user_num', 'follow_user_num', 'friend_user_num']
    users = users.select(keep).with_columns(
        pl.col('is_live_streamer').eq(1).cast(pl.Int8),
        *[pl.col(f'onehot_feat{i}').fill_null(-1 if i < 12 else 0) for i in range(18)],
    )
    req = req.join(users, on='user_id', how='left', validate='m:1')
    user_values = req['user_id'].to_numpy()
    folds = np.empty(req.height, dtype=np.int8)
    for i, (_, idx) in enumerate(GroupKFold(5).split(user_values, groups=user_values)):
        folds[idx] = i
    req = req.with_columns(pl.Series('fold', folds)).with_row_index('request_id')
    assert req.null_count().sum_horizontal().item() == 0
    assert req.filter(pl.col('n_items') != 6).height == 0
    req.write_parquet(out / 'requests.parquet', row_group_size=65536)
    stats = {
        'raw_rows': raw_rows, 'deduplicated_rows': dedup_rows, 'all_sessions': sessions.height,
        'retained_sessions': req['session_id'].n_unique(), 'request_rows': req.height,
        'users': req['user_id'].n_unique(), 'y_continue_mean': req['y_continue'].mean(),
        'fold_rows': req.group_by('fold').len().sort('fold').to_dicts(),
        'sources': [{'path': str(p.resolve()), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths],
        'protocol': '30min idle gap after clipped preceding play; M=6 reverse chunks; head remainder history; first/last session per user excluded; pre-exposure wall clock; end-clock diagnostic only',
        'history_cap': 12, 'duration_clip_s': [3, 600], 'play_clip_ms_for_boundary_and_completion': [0, 300000],
        'watch_labels': 'sum raw nonnegative play_time_ms / 1000; no upper clip',
    }
    (out / 'manifest.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
