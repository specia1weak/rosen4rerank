"""从既有v3成员生成v4纯时间戳分钟，重算关联分钟先验，保留行序/标签/词表。

会话ID、六视频屏、split全部沿用，不改sessionize的历史空闲判定。
仅修改elapsed_min、user_hist_session_min、day_prev_elapsed_min及其上游原始量。
旧数据/源码只读。新目录中复制冻结词表；数值设置固定[0,1]透传。
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import gc
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import polars as pl
import torch

from experiments.kuairand.pipeline.build_requests_v3 import user_priors
from experiments.kuairand.pipeline.exit_training import WORKSPACE, EXPERIMENTS
from experiments.kuairand.pipeline.kr_request_schema import request_manager
from experiments.kuairand.pipeline.exit_model import KRExitModel, pace_feature
from experiments.kuairand.pipeline.timestamp_clock import CLOCK, timestamp_boundaries, load_timestamp_batches

CHANGED = ['elapsed_min', 'user_hist_session_min', 'day_prev_elapsed_min']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_columns(frame, updates, keys, columns):
    """在有唯一键的更新表上左连接，保留原始行序与列序。"""
    order = frame.columns
    updated = frame.drop(columns).join(updates.select(*keys, *columns), on=keys,
        how='left', validate='m:1', maintain_order='left')
    assert updated.height == frame.height
    assert updated.select(columns).null_count().sum_horizontal().item() == 0
    return updated.select(order)


def build(source, output):
    torch.set_num_threads(4)
    tracked = [source/name for name in ['events.parquet', 'sessions.parquet', 'requests_base.parquet',
        'features_raw.parquet', 'manifest.json', 'train.parquet', 'valid.parquet', 'test.parquet']]
    tracked += list((source/'schema').rglob('*'))
    tracked += [Path(__file__), Path(__file__).with_name('timestamp_clock.py'),
                Path(__file__).with_name('build_requests_v3.py')]
    hashes = {str(p.resolve()):sha(p) for p in tracked if p.is_file()}
    old_manifest = json.loads((source/'manifest.json').read_text())
    assert old_manifest['version'] == 3
    events = pl.scan_parquet(source/'events.parquet').select('session_id', 'req_idx', 'time_ms')
    s_lazy, r_lazy = timestamp_boundaries(events)
    bounds = s_lazy.collect()
    clocks = r_lazy.filter(pl.col('req_idx') > 0).collect()
    old_sessions = pl.read_parquet(source/'sessions.parquet')
    sessions = replace_columns(old_sessions, bounds, ['session_id'], ['session_min'])
    sessions = sessions.with_columns(
        (pl.col('session_min').cum_sum().over(['user_id', 'session_date'])-pl.col('session_min')).alias('day_prev_elapsed_min'))
    sessions = sessions.join(bounds.drop('session_min'), on='session_id', validate='1:1', maintain_order='left')
    sessions.write_parquet(output/'sessions.parquet')
    print('TIMESTAMP_BOUNDARIES', bounds.height, clocks.height, flush=True)
    raw = pl.read_parquet(source/'features_raw.parquet')
    raw = replace_columns(raw, clocks, ['session_id','req_idx'], ['elapsed_min'])
    raw = replace_columns(raw, sessions, ['session_id'], ['session_min','day_prev_elapsed_min'])
    query = raw.select('request_id','user_id',
        pl.when(pl.col('split')=='train').then(pl.col('session_date')).otherwise(pl.lit(20220507)).alias('date'))
    priors = user_priors(query, sessions)
    original_priors = raw.select('request_id','user_prior_n','user_hist_req_cnt').join(
        priors, on='request_id', suffix='_new', validate='1:1')
    for col in ['user_prior_n','user_hist_req_cnt']:
        np.testing.assert_allclose(original_priors[col],original_priors[col+'_new'],atol=1e-12,rtol=1e-12)
    raw = replace_columns(raw, priors, ['request_id'], ['user_hist_session_min'])
    clocks_meta = clocks.select('session_id','req_idx','session_start_ms','request_time_ms')
    raw = raw.join(clocks_meta, on=['session_id','req_idx'], how='left', validate='1:1', maintain_order='left')
    raw.write_parquet(output/'features_raw.parquet',row_group_size=65536)
    base = pl.read_parquet(source/'requests_base.parquet')
    base = replace_columns(base, raw, ['request_id'], ['elapsed_min','session_min','day_prev_elapsed_min'])
    base = base.join(raw.select('request_id','session_start_ms','request_time_ms'), on='request_id',
        how='left', validate='1:1', maintain_order='left')
    base.write_parquet(output/'requests_base.parquet',row_group_size=65536)
    del base, priors, query, original_priors
    gc.collect()
    shutil.copytree(source/'schema',output/'schema')
    manager = request_manager(output/'schema', load=True)
    for col in CHANGED:
        setting = manager.get_setting(col)
        assert (setting.min_val,setting.max_val)==(0.,1.)
    encoded_updates = raw.select('request_id', pl.col('elapsed_min').cast(pl.Float32),
        *[pl.col(col).clip(lower_bound=0).log1p().cast(pl.Float32) for col in CHANGED[1:]])
    splits = {}
    for split in ['train','valid','test']:
        old = pl.read_parquet(source/f'{split}.parquet')
        new = replace_columns(old,encoded_updates,['request_id'],CHANGED)
        assert new.select(pl.exclude(CHANGED)).equals(old.select(pl.exclude(CHANGED)))
        assert new.schema == old.schema
        new.write_parquet(output/f'{split}.parquet',row_group_size=65536)
        ordered = new.sort('session_id','req_idx_raw')
        delta = ordered.select(pl.col('elapsed_min').diff().over('session_id').alias('d'))['d']
        assert delta.min() >= 0
        order_hash=hashlib.sha256(new['request_id'].to_numpy().tobytes()).hexdigest()
        assert order_hash==old_manifest['splits'][split]['request_order_sha256']
        splits[split]={**old_manifest['splits'][split], 'backwards_adjacent_pairs':int((delta<0).sum()),
            'changed_rows':{col:int((new[col]!=old[col]).sum()) for col in CHANGED},
            'sha256':sha(output/f'{split}.parquet')}
        print('SPLIT',split,json.dumps(splits[split]),flush=True)
        del old,new,ordered
    manifest={**old_manifest, 'version':4, 'parent_data':str(source.resolve()),
        'parent_manifest_sha256':sha(source/'manifest.json'), 'splits':splits,
        'protocol':{**old_manifest['protocol'],'clock':CLOCK,
            'minute_definition':'(screen_last_event.time_ms - session_first_event.time_ms)/60000',
            'session_min':'(session_last_event.time_ms - session_first_event.time_ms)/60000',
            'content_elapsed':'masked direct clock; pace uses timestamp-only elapsed_min/req_idx',
            'session_membership':'same v3 IDs; previous clipped-play idle-gap boundary retained',
            'minute_priors':'recomputed using timestamp-only session_min; same strict day/frozen-eval boundaries'},
        'changed_encoded_fields':CHANGED, 'all_other_encoded_values_exact':True,
        'schema':'copied frozen v3 train-fit vocab; dense fixed [0,1] passthrough; no refit',
        'source_hashes':hashes}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # 一个真实batch的联合前后向检查，不执行optimizer.step，不产生新模型成绩。
    batches = load_timestamp_batches(output,manager,row_limits={'train':4096,'valid':1,'test':1})
    batch=batches['train'][0].interaction
    batch['user_id']=torch.zeros_like(batch['user_id'])
    torch.manual_seed(2026)
    model=KRExitModel(manager,**EXPERIMENTS['mixture_step_pace'])
    loss=model.calculate_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    expected=torch.log1p(batch['elapsed_min_raw'].float()/batch['req_idx_raw'].float())
    assert torch.equal(pace_feature(batch),expected)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    audit={'source_hashes_unchanged':True,'changed_encoded_fields':CHANGED,
        'all_other_encoded_values_and_row_order_exact':True,'all_clocks_nondecreasing':True,
        'train_fit_schema_frozen_copy':True,'batch_rows':len(batch['y_continue']),
        'finite_joint_loss_and_backward':True,'optimizer_steps':0,'full_training_done':False,
        'smoke_loss_not_test_metric':float(loss.detach())}
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (WORKSPACE/'data/current.json').write_text(json.dumps({'data':'v4_timestamp', 'version':4,
        'clock':CLOCK, 'historical_data':'v3', 'new_model_results_available':False},indent=2)+'\n')
    print('READY',json.dumps(audit),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=WORKSPACE/'data/v3')
    parser.add_argument('--output',type=Path,default=WORKSPACE/'data/v4_timestamp')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'logs').mkdir()
    print('日志：',args.output/'logs/build.log',flush=True)
    with (args.output/'logs/build.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
        build(args.source,args.output)
    print('完成：',args.output/'audit.json',flush=True)


if __name__=='__main__':
    main()
