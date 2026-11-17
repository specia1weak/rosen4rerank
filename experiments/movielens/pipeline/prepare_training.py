"""30min/一评分一步的继续预测数据；完整Session按全局时间80/10/10切分。

预测当前电影评分前的下一步评分继续；当前rating仅审计，不进网络。
同秒历史遵从已确认源行序，不宣称真实微观先后。全局cut来自评分时间
80/90分位；跨cut的Session及结束后不足30min观察窗的场不作完整停止监督。
历史按过去日志在线推进，包含valid/test已发生评分；schema只fit train。
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import polars as pl
from rosen.datasets.base import DatasetBase
from experiments.movielens.pipeline.prepare_sessions import WORKSPACE, sha
from experiments.movielens.pipeline.schema import manager_at, FEATURES, BUDGET_FIELDS


def make_features(events, sessions, users, movies):
    s = sessions.sort('user_id', 'start_timestamp', 'session_id').with_columns(
        (pl.col('session_id').cum_count().over('user_id') - 1).cast(pl.Int64).alias('_past_n'),
        (pl.col('steps').cum_sum().over('user_id') - pl.col('steps')).alias('_past_steps'),
        (pl.col('span_min').cum_sum().over('user_id') - pl.col('span_min')).alias('_past_minutes'),
        pl.col('last_rating_timestamp').shift(1).over('user_id').alias('_previous_end'))
    s = s.with_columns(
        (pl.col('_past_steps') / pl.col('_past_n').clip(lower_bound=1)).log1p().alias('past_session_steps_log'),
        (pl.col('_past_minutes') / pl.col('_past_n').clip(lower_bound=1)).log1p().alias('past_session_min_log'),
        pl.col('_past_n').log1p().alias('past_session_count_log'),
        ((pl.col('start_timestamp')-pl.col('_previous_end'))/60.).fill_null(0).log1p().alias('previous_session_gap_log'),
        pl.from_epoch('start_timestamp', time_unit='s').dt.hour().cast(pl.String).alias('start_hour'),
        pl.from_epoch('start_timestamp', time_unit='s').dt.weekday().cast(pl.String).alias('start_weekday'))
    f = events.join(s.select('session_id', 'steps', 'start_timestamp', 'last_rating_timestamp',
        *[c for c in BUDGET_FIELDS if c not in ['user_id', 'gender', 'age', 'occupation', 'zip3']]),
        on='session_id', validate='m:1', maintain_order='left')
    f = f.join(users, on='user_id', validate='m:1', maintain_order='left').join(
        movies, on='movie_id', validate='m:1', maintain_order='left')
    assert f['gender'].null_count() == 0 and f['genres'].null_count() == 0
    f = f.sort('user_id', 'timestamp', 'event_id').with_columns(
        pl.col('genres').shift(1).over('session_id').fill_null('').alias('previous_genres'),
        (pl.col('rating').shift(1).over('session_id').fill_null(0)/5.).alias('previous_rating_scaled'),
        ((pl.col('rating').cum_sum().over('session_id')-pl.col('rating')) /
         (pl.col('step_idx')-1).clip(lower_bound=1)/5.).alias('session_previous_rating_mean'),
        (((pl.col('rating')>=4).cast(pl.Int64).cum_sum().over('session_id')-(pl.col('rating')>=4).cast(pl.Int64)) /
         (pl.col('step_idx')-1).clip(lower_bound=1)).alias('session_previous_positive_rate'),
        (pl.col('elapsed_min').shift(1).over('session_id').fill_null(0) /
         (pl.col('step_idx')-1).clip(lower_bound=1)).log1p().alias('pace_previous'),
        pl.col('step_idx').log1p().alias('step_log'), pl.col('elapsed_min').log1p().alias('elapsed_log'),
        ((pl.col('release_year')-1900)/110.).clip(0,1).alias('release_year_scaled'),
        (pl.col('step_idx')<pl.col('steps')).cast(pl.Float32).alias('y_continue'),
        pl.col('event_id').alias('request_id'), pl.col('user_id').alias('user_id_raw'),
        pl.col('step_idx').alias('req_idx_raw'), pl.col('elapsed_min').alias('elapsed_min_raw'),
        pl.lit(0).alias('tab_raw'), pl.lit(0).alias('tab'), pl.col('rating').alias('rating_audit'))
    return f


def build(output):
    base = WORKSPACE/'data/sessions_30min_v1'
    source = DatasetBase.SYSTEM_DATA_DIR/'MovieLens/ml-1m'
    tracked = [base/n for n in ['events.parquet', 'sessions.parquet', 'manifest.json']]
    tracked += [source/'users.dat', source/'movies.dat', Path(__file__), Path(__file__).with_name('schema.py')]
    hashes = {str(p.resolve()): sha(p) for p in tracked}
    events, sessions = pl.read_parquet(base/'events.parquet'), pl.read_parquet(base/'sessions.parquet')
    users = pl.read_csv(source/'users.dat', separator=':', has_header=False,
        schema_overrides={'column_9':pl.String}).select(
        pl.col('column_1').alias('user_id'), pl.col('column_3').alias('gender'),
        pl.col('column_5').cast(pl.String).alias('age'), pl.col('column_7').cast(pl.String).alias('occupation'),
        pl.col('column_9').str.split('-').list.first().str.zfill(5).str.slice(0,3).alias('zip3'))
    # latin1电影标题只用于提取年份，不把标题文本编码成特征。
    movie_rows = [line.split('::') for line in (source/'movies.dat').read_text(encoding='latin-1').splitlines()]
    movies = pl.DataFrame(movie_rows, schema=['movie_id', 'title', 'genres'], orient='row').with_columns(
        pl.col('movie_id').cast(pl.Int64), pl.col('title').str.extract(r'\((\d{4})\)$',1).cast(pl.Int64).fill_null(1900).alias('release_year')).drop('title')
    f = make_features(events, sessions, users, movies)
    cut1, cut2 = [int(events['timestamp'].quantile(q, interpolation='nearest')) for q in [.8,.9]]
    end = int(events['timestamp'].max())
    # 结束后观察满30分钟，才把没有下一条评分记作活动停止。
    selections = {
        'train': pl.col('last_rating_timestamp')+1800 < cut1,
        'valid': (pl.col('start_timestamp')>=cut1)&(pl.col('last_rating_timestamp')+1800<cut2),
        'test': (pl.col('start_timestamp')>=cut2)&(pl.col('last_rating_timestamp')+1800<end)}
    split_frames = {k:f.filter(v) for k,v in selections.items()}
    manager = manager_at(output/'schema')
    manager.fit(split_frames['train'].lazy())
    split_meta = {}
    seen = set()
    for name, raw in split_frames.items():
        assert raw.height > 0
        ids = set(raw['session_id'].to_list()); assert not ids & seen; seen |= ids
        g = raw.group_by('session_id').agg(pl.len().alias('n'), pl.col('steps').first().alias('length'),
            pl.col('y_continue').sum().alias('continuations'),
            *[pl.col(c).n_unique().alias(c) for c in BUDGET_FIELDS])
        assert (g['n']==g['length']).all() and (g['continuations']==g['length']-1).all()
        assert g.select(pl.all().exclude('session_id','n','length','continuations').max()).row(0) == (1,)*len(BUDGET_FIELDS)
        raw.write_parquet(output/f'{name}_raw.parquet')
        encoded = manager.transform(raw.lazy()).collect()
        if name == 'train':
            encoded = encoded.sample(fraction=1., shuffle=True, seed=2026)
        else:
            encoded = encoded.sort('timestamp','user_id_raw','request_id')
        encoded.write_parquet(output/f'{name}.parquet', row_group_size=65536)
        split_meta[name] = dict(rows=raw.height, sessions=len(ids), users=raw['user_id'].n_unique(),
            positive_rate=float(raw['y_continue'].mean()), mean_steps=raw.height/len(ids),
            timestamp_min=int(raw['timestamp'].min()), timestamp_max=int(raw['timestamp'].max()),
            uid_unseen_users=len(set(raw['user_id'].to_list())-set(split_frames['train']['user_id'].to_list())),
            order_sha256=sha(output/f'{name}.parquet'))
        print('SPLIT', name, json.dumps(split_meta[name]), flush=True)
    excluded = sessions.filter(~pl.col('session_id').is_in(list(seen)))
    excluded.write_parquet(output/'excluded_sessions.parquet')
    meta = dict(source_hashes=hashes, protocol=dict(session='30min idle strictly greater; one rating per step',
        ties='original source line order, no claim true within-second order',
        target='next rating exists within this session; current rating excluded from input',
        context_clock='UTC; actual user timezone unknown', split='global rating timestamp quantiles80/90; complete sessions with 30min end observation; crossing/near-boundary excluded',
        cutoff_timestamps=[cut1,cut2,end], cutoff_utc=[datetime.fromtimestamp(t,timezone.utc).isoformat() for t in [cut1,cut2,end]],
        history='online expanding past events including earlier valid/test; same-second source order',
        priors='past completed user sessions only; cold defaults0; no population future prior',
        schema_fit='train only', train_shuffle='physical fixed permutation seed2026', uid='encoded input zero at load'),
        splits=split_meta, excluded_sessions=excluded.height, excluded_rows=int(excluded['steps'].sum()),
        features=FEATURES, budget_fields=BUDGET_FIELDS, current_rating_input=False)
    assert all(sha(p)==h for p,h in hashes.items())
    (output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORKSPACE/'data/continuation_v1')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False); (args.output/'logs').mkdir()
    with (args.output/'logs/build.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
        build(args.output)
    print('完成：',args.output/'manifest.json')
