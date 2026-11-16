"""Fold-local, strictly earlier-calendar-day quality and budget priors."""
import json
from pathlib import Path
import polars as pl
from rosen.emb import SchemaManager
from rosen.emb.schema import MinMaxDenseSetting, SparseEmbSetting, SequenceSetting, SeqGroupConfig, VectorDenseSetting
from rosen.core.enum_type import FeatureSource as FS

B1 = ['register_days', 'fans_user_num', 'follow_user_num', 'friend_user_num', 'user_hist_session_s', 'user_hist_session_requests']
B2 = ['hour_sin', 'hour_cos', 'is_weekend', 'day_prev_sess_cnt', 'day_prev_time_s', 'elapsed_log', 'req_idx']
B3 = ['is_first_req', 'prev1_play_s', 'prev1_comp_rate', 'prev1_long_view_rate', 'prev1_pos_cnt', 'sess_cum_comp_rate', 'sess_cum_pos_rate', 'sess_cum_hate_cnt']
B4 = ['slate_dur_sum_s', 'slate_unique_tag_cnt', 'slate_unique_author_cnt', 'slate_has_rand'] + ['prior_' + c + '_' + a for c in ['comp', 'long_view', 'is_like'] for a in ['mean', 'min', 'max']]
CATS = ['user_active_degree', 'is_live_streamer', 'is_video_author'] + [f'onehot_feat{i}' for i in range(12)] + ['session_tab']
B1 += [f'onehot_feat{i}' for i in range(12, 18)]
NUMS = B1 + B2 + B3 + B4
LOG_COLS = ['register_days', 'fans_user_num', 'follow_user_num', 'friend_user_num', 'user_hist_session_s', 'user_hist_session_requests', 'day_prev_sess_cnt', 'day_prev_time_s', 'req_idx', 'prev1_play_s', 'prev1_pos_cnt', 'sess_cum_hate_cnt', 'slate_dur_sum_s']
EXTRAS = ['request_id', 'user_id', 'session_id', 'y_continue', 'elapsed_minutes', 'delta_minutes', 'curr_watch_s', 'future_watch_s', 'remaining_watch_s']


def past_daily(query, observations, keys, values):
    """Cumulative sums/count through the previous day; current day excluded."""
    by = keys + ['date']
    daily = observations.group_by(by).agg(pl.len().cast(pl.Float64).alias('prior_n'), *[pl.col(v).sum().alias('sum_' + v) for v in values]).sort(by)
    sums = ['prior_n'] + ['sum_' + v for v in values]
    if keys:
        daily = daily.with_columns(*[pl.col(c).cum_sum().over(keys).alias(c) for c in sums])
    else:
        daily = daily.with_columns(*[pl.col(c).cum_sum().alias(c) for c in sums])
    return query.sort('date').join_asof(daily.sort('date'), on='date', by=keys or None, strategy='backward', allow_exact_matches=False, check_sortedness=False).with_columns(pl.col(sums).fill_null(0))


def make_manager(path: Path, req=None):
    settings = [MinMaxDenseSetting(c, FS.INTERACTION) for c in NUMS]
    settings += [SparseEmbSetting(c, FS.USER, embedding_dim=4) for c in CATS]
    for name, cap in [('slate_items', 6), ('slate_tags', 6), ('history_items', 12), ('history_tags', 12)]:
        settings.append(SequenceSetting(name, SparseEmbSetting(name, FS.ITEM, embedding_dim=8, min_freq=5 if 'items' in name else 1), SeqGroupConfig(name, name + '_len', cap)))
    settings.append(VectorDenseSetting('slate_durations', FS.INTERACTION, embedding_dim=6))
    mgr = SchemaManager(settings, str(path), label_fields='y_continue')
    if req is None:
        mgr.load_schema()
    return mgr


def prepare_fold(root: Path, fold: int, seed=2026):
    out = root / f'fold_{fold}'
    out.mkdir(exist_ok=True)
    if (out / 'manifest.json').exists():
        saved = json.loads((out / 'manifest.json').read_text())
        if saved['seed'] != seed:
            raise ValueError('Fold physical shuffle uses another seed; use a fresh dataset directory')
        return make_manager(out / 'schema'), out
    req = pl.read_parquet(root / 'requests.parquet')
    train_users = req.filter(pl.col('fold') != fold).select('user_id').unique()
    test_users = req.filter(pl.col('fold') == fold)['user_id'].unique().to_list()
    obs = pl.read_parquet(root / 'prior_events.parquet').join(train_users, on='user_id', how='semi')
    # Priors for each slate video, using only training users and days < request day.
    q = req.select('request_id', 'date', 'slate_items').explode('slate_items').rename({'slate_items': 'video_id'})
    vals = ['comp', 'long_view', 'is_like']
    q = past_daily(q, obs, ['video_id'], vals)
    global_q = past_daily(req.select('request_id', 'date'), obs, [], vals).select('request_id', *[(pl.col('sum_' + v) / pl.col('prior_n').clip(lower_bound=1)).alias('global_' + v) for v in vals])
    q = q.join(global_q, on='request_id', how='left').with_columns(*[((pl.col('sum_' + v) + 20 * pl.col('global_' + v)) / (pl.col('prior_n') + 20)).alias('prior_' + v) for v in vals])
    agg = q.group_by('request_id').agg(*[getattr(pl.col('prior_' + v), a)().alias('prior_' + v + '_' + a) for v in vals for a in ['mean', 'min', 'max']])
    req = req.join(agg, on='request_id', how='left')
    del obs, q, agg
    # Training users: their own prior-day sessions; held-out users: training-population fallback.
    sess = pl.read_parquet(root / 'sessions.parquet').join(train_users, on='user_id', how='semi').rename({'prior_available_date': 'date'})
    vals = ['session_wall_s', 'session_requests']
    uq = past_daily(req.select('request_id', 'date', 'user_id'), sess, ['user_id'], vals)
    gq = past_daily(req.select('request_id', 'date'), sess, [], vals).select('request_id', *[(pl.col('sum_' + v) / pl.col('prior_n').clip(lower_bound=1)).alias('global_' + v) for v in vals])
    uq = uq.join(gq, on='request_id').select('request_id', *[pl.when(pl.col('prior_n') > 0).then(pl.col('sum_' + v) / pl.col('prior_n')).otherwise(pl.col('global_' + v)).alias(n) for v, n in zip(vals, ['user_hist_session_s', 'user_hist_session_requests'])])
    req = req.join(uq, on='request_id')
    req = req.with_columns(
        (pl.col('session_hour') * (2 * 3.141592653589793 / 24)).sin().alias('hour_sin'),
        (pl.col('session_hour') * (2 * 3.141592653589793 / 24)).cos().alias('hour_cos'),
        (pl.col('session_date').cast(pl.String).str.strptime(pl.Date, '%Y%m%d').dt.weekday() >= 6).cast(pl.Float32).alias('is_weekend'),
        pl.col('elapsed_time_s').log1p().alias('elapsed_log'),
        (pl.col('elapsed_time_s') / 60).cast(pl.Float32).alias('elapsed_minutes'),
        (pl.col('slate_dur_sum_s') / 60).cast(pl.Float32).alias('delta_minutes'),
        pl.when(pl.col('session_tab').is_in([0, 1, 2, 4, 6])).then(pl.col('session_tab')).otherwise(99).alias('session_tab'),
        pl.col('slate_durations').list.eval(pl.element().log1p() / 6.4),
    ).with_columns(*[pl.col(c).clip(lower_bound=0).log1p().alias(c) for c in LOG_COLS])
    mgr = make_manager(out / 'schema', req)
    mgr.fit(req.filter(pl.col('fold') != fold).lazy())
    keep = list(dict.fromkeys(mgr.fields() + EXTRAS))
    for split, pred in [('train', pl.col('fold') != fold), ('test', pl.col('fold') == fold)]:
        part = mgr.transform(req.filter(pred).lazy()).select(keep).collect()
        if split == 'train':
            part = part.sample(fraction=1, shuffle=True, seed=seed)
        part.write_parquet(out / f'{split}.parquet', row_group_size=65536)
        print('fold', fold, split, part.height, flush=True)
    meta = {'fold': fold, 'train_rows': req.filter(pl.col('fold') != fold).height, 'test_rows': req.filter(pl.col('fold') == fold).height, 'test_users': sorted(test_users), 'priors': 'training-users only; strictly earlier calendar days; item smoothing strength=20; unseen users use prior-day training population', 'features': {'block1': B1 + CATS[:-1], 'block2': B2 + ['session_tab'], 'block3': B3 + ['history_items', 'history_tags'], 'block4': B4 + ['slate_items', 'slate_tags', 'slate_durations']}, 'seed': seed}
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2))
    return mgr, out
