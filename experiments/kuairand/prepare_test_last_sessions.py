"""重做Test尾场：追加有完整空闲观察窗口的last_session，未知结束另存删失表。

不修改旧Train/Valid/Test或Schema。保留旧Test前缀，新增request_id从历史最大值后分配。
日志最后事件时间作为保守观察截止；沿用 previous.play_clip 修正的30分钟空闲规则。

uv run python -m experiments.kuairand.prepare_test_last_sessions
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import polars as pl
import torch

from rosen.datasets.kuairand import KuaiRandDataset
from experiments.kuairand.nxt_continue_v1 import (
    BUDGET_PRIOR_FIELDS, DEFAULT_DATA_DIR, FEEDBACK_FIELDS, USER_CATEGORY_FIELDS,
    _aggregate_requests, _item_priors, _user_priors, build_schema,
)

DEFAULT_OUTPUT = Path('workspace/kr_time_budget/data/nxt_continue_v1_test_tail')
IDLE_MS = 30 * 60 * 1000


def classify_test_tails(sessions: pl.DataFrame, ends: pl.DataFrame, *, cutoff_ms: int, tabs: list[int]) -> pl.DataFrame:
    """只改Test尾场；cutoff-last_play_end严格>30分钟才确认完整结束。"""
    return sessions.join(ends, on='session_id', validate='1:1').filter(
        (pl.col('session_date') == 20220508) & (pl.col('end_date') == 20220508)
        & (pl.col('session_id') == pl.col('last_session'))
        & (pl.col('session_id') != pl.col('first_session'))
        & (pl.col('session_n') >= 6) & pl.col('tab').is_in(tabs)
    ).with_columns(
        (pl.lit(cutoff_ms) - pl.col('last_event_ms') - pl.col('last_play_clip')).alias('followup_idle_ms'),
    ).with_columns((pl.col('followup_idle_ms') > IDLE_MS).alias('end_observed')).sort('session_id')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(parent: Path, history: Path, raw_source: Path, output: Path) -> dict:
    tracked = [parent / f'{s}.parquet' for s in ['train', 'valid', 'test']]
    tracked += [parent / 'manifest.json', *[p for p in (parent / 'schema').rglob('*') if p.is_file()]]
    before_hashes = {str(p.resolve()): sha(p) for p in tracked}
    events_path = history / 'v3/events.parquet'
    sessions_path = history / 'v4_timestamp/sessions.parquet'
    events = pl.scan_parquet(events_path)
    # 最大已观测时间比假设日志一定覆盖到午夜更保守，但仍依赖源日志连续覆盖的假设。
    cutoff = events.select(pl.col('time_ms').max()).collect().item()
    ends = events.group_by('session_id').agg(
        pl.col('time_ms').max().alias('last_event_ms'),
        pl.col('play_clip').sort_by('event_idx').last().alias('last_play_clip'),
    ).collect()
    sessions = pl.read_parquet(sessions_path)
    tabs = json.loads((history / 'v3/structure_manifest.json').read_text())['top_tabs']
    candidates = classify_test_tails(sessions, ends, cutoff_ms=cutoff, tabs=tabs)
    candidates.write_parquet(output / 'tail_sessions.parquet')
    print('TAILS', candidates.group_by('end_observed').len().to_dicts(), flush=True)
    selected = events.join(candidates.select('session_id').lazy(), on='session_id', how='semi').sort('session_id', 'event_idx').collect()
    video = pl.read_csv(raw_source / 'video_features_basic_1k.csv',
        columns=['video_id', 'author_id', 'tag', 'video_type', 'music_type'], schema_overrides={'tag': pl.String, 'video_type': pl.String})
    selected = selected.join(video, on='video_id', how='left', validate='m:1', maintain_order='left').with_columns(
        pl.col('tag').fill_null('__MISSING__'), pl.col('video_type').fill_null('__MISSING__'),
        pl.col('music_type').cast(pl.Int64).cast(pl.String).fill_null('__MISSING__'), pl.col('author_id').fill_null(-1),
        pl.any_horizontal(*[pl.col(c) == 1 for c in ['is_like', 'is_comment', 'is_follow', 'is_forward']]).cast(pl.Int8).alias('positive'),
    )
    max_request = max(pl.scan_parquet(parent / f'{s}.parquet').select(pl.col('request_id').max()).collect().item() for s in ['train','valid','test'])
    requests = _aggregate_requests(selected).join(candidates, on=['user_id','session_id'], validate='m:1').sort('session_id','req_idx')
    requests = requests.with_row_index('request_id', offset=max_request + 1)
    users = pl.read_csv(raw_source / 'user_features_1k.csv').select('user_id', *USER_CATEGORY_FIELDS).with_columns(
        pl.col('is_live_streamer').eq(1).cast(pl.Int8),
        *[pl.col(f'onehot_feat{i}').cast(pl.Int64).fill_null(-1) for i in range(18)],
    )
    requests = requests.join(users, on='user_id', how='left', validate='m:1').with_columns(
        pl.col('user_id').alias('user_id_raw'), pl.col('tab').alias('tab_raw'),
        pl.lit(7, dtype=pl.Int8).alias('day_of_week'), pl.lit(1, dtype=pl.Int8).alias('is_weekend'),
        pl.lit('test').alias('split'),
    )
    query = requests.select('request_id', 'user_id', pl.lit(20220507, dtype=pl.Int64).alias('date'))
    requests = requests.join(_user_priors(query, sessions), on='request_id', validate='1:1')
    observations = events.filter(pl.col('date') <= 20220506).select('video_id','date','is_click','long_view').collect()
    item_query = query.select('request_id','date').join(requests.select('request_id','slate_items'), on='request_id', validate='1:1')
    requests = requests.join(_item_priors(item_query, observations), on='request_id', validate='1:1').sort('session_id','req_idx')
    requests = requests.with_columns(
        pl.when(~pl.col('end_observed') & (pl.col('y_continue') == 0)).then(None).otherwise(pl.col('y_continue')).alias('y_continue'),
    )
    requests.write_parquet(output / 'tail_requests_raw.parquet')
    shutil.copytree(parent / 'schema', output / 'schema')
    manager = build_schema(output / 'schema')
    manager.load_schema()
    transformed = manager.transform(requests.with_columns(
        *[pl.col(c).clip(lower_bound=0).log1p().cast(pl.Float32) for c in BUDGET_PRIOR_FIELDS]
    ).lazy()).select(manager.fields()).collect()
    # 保持旧基线的Float32 pace运算顺序，预处理时物化，不在模型里计算。
    minutes = torch.tensor(transformed['elapsed_min'].to_numpy(), dtype=torch.float32)
    screens = torch.tensor(transformed['req_idx_raw'].to_numpy(), dtype=torch.float32)
    transformed = transformed.with_columns(pl.Series('pace', torch.log1p(minutes / screens).numpy()))
    transformed = transformed.join(requests.select('request_id','end_observed'), on='request_id', validate='1:1', maintain_order='left')
    original = pl.read_parquet(parent / 'test.parquet')
    tails = transformed.filter(pl.col('end_observed')).drop('end_observed').cast(original.schema)
    censored = transformed.filter(~pl.col('end_observed')).drop('end_observed').cast(original.schema)
    assert tails.null_count().sum_horizontal().item() == 0
    assert censored['y_continue'].null_count() == candidates.filter(~pl.col('end_observed')).height
    tails.write_parquet(output / 'test_added_complete.parquet')
    censored.write_parquet(output / 'test_censored.parquet')
    primary = pl.concat([original, tails])
    assert primary.head(original.height).equals(original)
    assert primary['request_id'].n_unique() == primary.height
    primary.write_parquet(output / 'test.parquet', row_group_size=65536)
    # 删失场中非末刷的继续标签是可观测的，单独保留用于逐请求分类补充评估。
    known = pl.concat([primary, censored.filter(pl.col('y_continue').is_not_null())])
    known.write_parquet(output / 'test_known_requests.parquet')
    for split in ['train','valid']:
        shutil.copy2(parent / f'{split}.parquet', output / f'{split}.parquet')
    manifest = json.loads((parent / 'manifest.json').read_text())
    manifest['parent'] = str(parent.resolve())
    stats = dict(rows=primary.height, sessions=primary['session_id'].n_unique(), users=primary['user_id_raw'].n_unique(),
        positive_rate=primary['y_continue'].mean(), requests_per_session=primary.height/primary['session_id'].n_unique(),
        request_order_sha256=hashlib.sha256(primary['request_id'].to_numpy().tobytes()).hexdigest())
    manifest['splits']['test'] = stats
    manifest['test_tail_revision'] = dict(
        cutoff_ms=cutoff, cutoff_local=datetime.fromtimestamp(cutoff/1000,ZoneInfo('Asia/Shanghai')).isoformat(),
        candidate_sessions=candidates.height, complete_added_sessions=tails['session_id'].n_unique(), complete_added_rows=tails.height,
        censored_sessions=censored['session_id'].n_unique(), censored_rows=censored.height,
        known_censored_continue_rows=censored['y_continue'].is_not_null().sum(),
        policy='Test only: last_session complete if cutoff-last_event-clip(last_play,0,300000)>1800000; strict >; otherwise censor',
        coverage_assumption='last globally observed event is a conservative common observation horizon; source export coverage is assumed continuous, not independently certified',
        censored_label='unknown terminal y_continue=null; no full-length NLL; previous continue labels retained separately',
        unchanged='Train/Valid byte-exact; original Test prefix exact; schema and all historical files preserved',
        other_filters='first-session exclusion, >=6 events, frozen top5 tabs, full-session May8 and reverse-six grouping retained',
    )
    manifest['protocol'] = {**manifest['protocol'], 'test_tail': manifest['test_tail_revision']['policy']}
    manifest['source_hashes'] = before_hashes
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert all(sha(p) == digest for p, digest in before_hashes.items())
    assert sha(output/'train.parquet') == sha(parent/'train.parquet')
    assert sha(output/'valid.parquet') == sha(parent/'valid.parquet')
    print('COMPLETE', json.dumps(manifest['test_tail_revision']), flush=True)
    return manifest


def prepare_test_last_sessions(parent=DEFAULT_DATA_DIR, *, history=Path('workspace/kr_time_budget/data'),
                               raw_source=KuaiRandDataset.DATA_KUAI_1K, output=DEFAULT_OUTPUT):
    parent, history, raw_source, output = map(Path, (parent, history, raw_source, output))
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    print('预处理日志：', output/'logs/prepare.log', flush=True)
    with (output/'logs/prepare.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        return build(parent, history, raw_source, output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    prepare_test_last_sessions(output=args.output)
