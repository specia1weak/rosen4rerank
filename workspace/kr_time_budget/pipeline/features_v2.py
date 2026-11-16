"""V2: own strictly past session history for every user; fit only inner train."""
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from rosen.emb import SchemaManager
from rosen.emb.schema import MinMaxDenseSetting, SparseEmbSetting, SequenceSetting, SeqGroupConfig, VectorDenseSetting
from rosen.core.enum_type import FeatureSource as FS
from workspace.kr_time_budget.pipeline.features import B1, B2, B3, B4, CATS, NUMS, LOG_COLS, past_daily

STATIC_BUDGET_NUMS = B1 + ['hour_sin', 'hour_cos', 'is_weekend', 'day_prev_sess_cnt', 'day_prev_time_s']
EXTRAS = ['request_id', 'user_id', 'session_id', 'y_continue', 'elapsed_minutes', 'delta_minutes', 'req_idx_raw', 'intra_step_group', 'active_step_group']


def manager_v2(path, fit=False):
    settings = [MinMaxDenseSetting(c, FS.INTERACTION) for c in NUMS]
    settings += [SparseEmbSetting(c, FS.USER, embedding_dim=4) for c in CATS]
    for name, length in [('slate_tags', 6), ('history_tags', 12)]:
        settings.append(SequenceSetting(name, SparseEmbSetting(name, FS.ITEM, embedding_dim=8), SeqGroupConfig(name, name + '_len', length)))
    settings.append(VectorDenseSetting('slate_durations', FS.INTERACTION, embedding_dim=6))
    manager = SchemaManager(settings, str(path), label_fields='y_continue')
    if not fit:
        manager.load_schema()
    return manager


def own_session_priors(query, sessions, train_users, smoothing=5):
    """Session history of held-out users is inference-time information, not fit data."""
    # Source v1 stored end_date+1; undo that delay. past_daily itself excludes
    # the same day, so a completed session from yesterday is available today.
    sess = sessions.with_columns(pl.col('prior_available_date').cast(pl.String).str.strptime(pl.Date, '%Y%m%d').dt.offset_by('-1d').dt.strftime('%Y%m%d').cast(pl.Int64).alias('date'))
    values = ['session_wall_s', 'session_requests']
    own = past_daily(query, sess, ['user_id'], values)
    global_past = past_daily(query.select('request_id', 'date'), sess.join(train_users, on='user_id', how='semi'), [], values)
    global_past = global_past.select('request_id', *[(pl.col('sum_' + c) / pl.col('prior_n').clip(lower_bound=1)).alias('global_' + c) for c in values])
    return own.join(global_past, on='request_id', validate='1:1').select('request_id', pl.col('prior_n').alias('own_prior_n'), *[((pl.col('sum_' + c) + smoothing * pl.col('global_' + c)) / (pl.col('prior_n') + smoothing)).alias(out) for c, out in zip(values, ['user_hist_session_s', 'user_hist_session_requests'])])


def prepare_fold_v2(source, output, fold, seed=2026):
    out = output / f'fold_{fold}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        saved = json.loads((out / 'manifest.json').read_text())
        if saved['seed'] != seed or saved['source'] != str(source.resolve()):
            raise ValueError('Existing fold has another source/seed')
        return manager_v2(out / 'schema'), out
    req = pl.read_parquet(source / 'requests.parquet')
    outer_users = req.filter(pl.col('fold') != fold)['user_id'].unique().sort().to_numpy()
    shuffled_users = np.random.default_rng(seed + fold).permutation(outer_users)
    valid_users = shuffled_users[:int(np.ceil(len(outer_users) * .1))].tolist()
    req = req.with_columns(pl.when(pl.col('fold') == fold).then(pl.lit('test')).when(pl.col('user_id').is_in(valid_users)).then(pl.lit('valid')).otherwise(pl.lit('train')).alias('split'))
    train_users = req.filter(pl.col('split') == 'train').select('user_id').unique()
    obs = pl.read_parquet(source / 'prior_events.parquet').join(train_users, on='user_id', how='semi')
    vals = ['comp', 'long_view', 'is_like']
    q = req.select('request_id', 'date', 'slate_items').explode('slate_items').rename({'slate_items': 'video_id'})
    q = past_daily(q, obs, ['video_id'], vals)
    glob = past_daily(req.select('request_id', 'date'), obs, [], vals).select('request_id', *[(pl.col('sum_' + v) / pl.col('prior_n').clip(lower_bound=1)).alias('global_' + v) for v in vals])
    q = q.join(glob, on='request_id').with_columns(*[((pl.col('sum_' + v) + 20 * pl.col('global_' + v)) / (pl.col('prior_n') + 20)).alias('prior_' + v) for v in vals])
    agg = q.group_by('request_id').agg(*[getattr(pl.col('prior_' + v), a)().alias('prior_' + v + '_' + a) for v in vals for a in ['mean', 'min', 'max']])
    req = req.join(agg, on='request_id', validate='1:1')
    del obs, q, glob, agg
    sessions = pl.read_parquet(source / 'sessions.parquet')
    # Query at session start day, so user/background parameters stay fixed even
    # in sessions that cross midnight.
    prior = own_session_priors(req.select('request_id', 'user_id', pl.col('session_date').alias('date')), sessions, train_users)
    req = req.join(prior, on='request_id', validate='1:1')
    active_rank = pl.struct('user_active_degree', 'req_idx_bin').rank('dense')
    req = req.with_columns(
        (pl.col('session_hour') * (2 * np.pi / 24)).sin().alias('hour_sin'),
        (pl.col('session_hour') * (2 * np.pi / 24)).cos().alias('hour_cos'),
        (pl.col('session_date').cast(pl.String).str.strptime(pl.Date, '%Y%m%d').dt.weekday() >= 6).cast(pl.Float32).alias('is_weekend'),
        pl.col('elapsed_end_s_DIAGNOSTIC').log1p().alias('elapsed_log'),
        (pl.col('elapsed_end_s_DIAGNOSTIC') / 60).cast(pl.Float32).alias('elapsed_minutes'),
        pl.lit(1., dtype=pl.Float32).alias('delta_minutes'),
        pl.col('req_idx').cast(pl.Int64).alias('req_idx_raw'),
        pl.col('req_idx').cast(pl.Int64).alias('intra_step_group'),
        pl.when(pl.col('req_idx') <= 5).then(pl.col('req_idx')).when(pl.col('req_idx') <= 10).then(6).when(pl.col('req_idx') <= 20).then(7).when(pl.col('req_idx') <= 50).then(8).otherwise(9).alias('req_idx_bin'),
        pl.when(pl.col('session_tab').is_in([0, 1, 2, 4, 6])).then(pl.col('session_tab')).otherwise(99).alias('session_tab'),
        pl.col('slate_durations').list.eval(pl.element().log1p() / 6.4),
    ).with_columns(active_rank.cast(pl.Int64).alias('active_step_group'))
    # Persist unscaled priors with split tags for train/test variation auditing.
    req.select('request_id', 'user_id', 'session_id', 'split', 'own_prior_n', 'user_hist_session_s', 'user_hist_session_requests').write_parquet(out / 'user_prior_audit.parquet')
    prior_stats = req.group_by('split').agg(pl.len().alias('rows'), (pl.col('own_prior_n') > 0).mean().alias('own_history_coverage'), pl.col('user_hist_session_s').std().alias('history_std_s'), pl.col('user_hist_session_s').n_unique().alias('history_unique')).to_dicts()
    # End-of-screen clock need not increase in raw KR for every adjacent chunk;
    # record it; never silently replace Leader's formula by cummax.
    timed = req.sort(['session_id', 'req_idx_raw']).with_columns(pl.col('elapsed_minutes').diff().over('session_id').alias('elapsed_diff'))
    clock_decreases = timed.filter(pl.col('elapsed_diff') < -1e-6).height
    del timed
    req = req.with_columns(*[pl.col(c).clip(lower_bound=0).log1p().alias(c) for c in LOG_COLS])
    mgr = manager_v2(out / 'schema', fit=True)
    mgr.fit(req.filter(pl.col('split') == 'train').lazy())
    keep = list(dict.fromkeys(mgr.fields() + EXTRAS))
    stats = {}
    for split in ['train', 'valid', 'test']:
        part = mgr.transform(req.filter(pl.col('split') == split).lazy()).select(keep).collect()
        assert part.null_count().sum_horizontal().item() == 0
        if split == 'train':
            part = part.sample(fraction=1, shuffle=True, seed=seed)
        part.write_parquet(out / f'{split}.parquet', row_group_size=65536)
        stats[split] = {'rows': part.height, 'users': sorted(part['user_id'].unique().to_list()), 'request_order_sha256': hashlib.sha256(part['request_id'].to_numpy().tobytes()).hexdigest()}
        print('V2 data', fold, split, part.height, flush=True)
    meta = {'fold': fold, 'source': str(source.resolve()), 'seed': seed, 'splits': stats, 'user_prior_stats': prior_stats, 'clock_adjacent_decreases': clock_decreases, 'protocol': 'own strictly earlier completed sessions for all users, m=5 train population smoothing; item priors and schema inner-train only; end-of-screen elapsed, delta=1min; no item-ID features', 'static_budget_numeric': STATIC_BUDGET_NUMS}
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2))
    return mgr, out
