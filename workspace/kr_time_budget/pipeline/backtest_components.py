"""Plan16只读组件回测：已有test预测关联当屏真实反馈，不训练或选择checkpoint。

uv run python -m workspace.kr_time_budget.pipeline.backtest_components

当前反馈从request_events的event_id精确回连events，不假设requests_base含当屏计数。
正方向固定为高stay对应高当屏反馈，不在看到负相关后翻转分数。
观察相关性不能认定组件纯度/因果识别；pace与观看时长共享当前耗时来源。
"""
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/v3'
PREDICTIONS = ROOT/'runs/plan13_ablation'
MODELS = ['single_step', 'single_step_pace', 'mixture_step', 'mixture_step_pace', 'fusion', 'fusion_pace']
TARGETS = ['curr_longview_rate', 'curr_click_rate', 'curr_watch_min', 'curr_watch_clip_min', 'curr_mean_watch_fraction']
SCOPES = ['all', 'exit', 'continue', 'first_or_30plus', 'first', '30plus']


def feedback_from_events(members, events):
    """输入可为LazyFrame；聚合当屏6个事件，保留顺序供独立关联核验。"""
    joined = members.join(events, on='event_id', how='inner', validate='1:1')
    return joined.group_by('request_id').agg(
        pl.len().alias('event_count'), pl.col('event_id').n_unique().alias('unique_events'),
        pl.col('video_id').sort_by('event_idx').alias('observed_items'),
        pl.col('session_id').first(), pl.col('user_id').first(), pl.col('req_idx').first(),
        pl.col('is_click').sum().alias('curr_click_cnt'),
        pl.col('long_view').sum().alias('curr_longview_cnt'),
        (pl.col('play_time_ms').clip(lower_bound=0).sum()/60000).alias('curr_watch_min'),
        (pl.col('play_clip').sum()/60000).alias('curr_watch_clip_min'),
        (pl.col('duration_ms').clip(3000, 600000).sum()/60000).alias('curr_duration_min'),
        (pl.col('play_time_ms').clip(lower_bound=0)/pl.col('duration_ms').clip(3000,600000)).clip(0,1).mean().alias('curr_mean_watch_fraction'),
        ((pl.col('time_ms').sort_by('event_idx').last()-pl.col('session_start_ms').first()
          +pl.col('play_clip').sort_by('event_idx').last())/60000).alias('recomputed_elapsed_min'),
    ).with_columns(
        (pl.col('curr_longview_cnt')/6).alias('curr_longview_rate'),
        (pl.col('curr_click_cnt')/6).alias('curr_click_rate'),
        (pl.col('curr_longview_cnt')>0).cast(pl.Int8).alias('curr_any_longview'),
        (pl.col('curr_click_cnt')>0).cast(pl.Int8).alias('curr_any_click'))


def build_feedback():
    test = pl.read_parquet(DATA/'test.parquet', columns=['request_id', 'session_id', 'user_id_raw', 'req_idx_raw', 'y_continue'])
    ids = test.select('request_id').lazy()
    membership = (pl.scan_parquet(DATA/'request_events.parquet').filter(pl.col('request_id').is_not_null())
        .join(ids, on='request_id', how='semi').select('event_id','request_id','event_idx'))
    events = pl.scan_parquet(DATA/'events.parquet').select('event_id','video_id','session_id','user_id','req_idx',
        'is_click','long_view','play_time_ms','play_clip','duration_ms','time_ms','session_start_ms')
    feedback = feedback_from_events(membership, events).sort('request_id').collect()
    assert feedback.height == feedback['request_id'].n_unique() == 48898
    assert (feedback['event_count'] == 6).all() and (feedback['unique_events'] == 6).all()
    assert feedback['event_count'].sum() == 293388
    assert feedback.select(pl.all().is_null().sum()).sum_horizontal().item() == 0
    base = pl.scan_parquet(DATA/'requests_base.parquet').filter(pl.col('split')=='test').select(
        'request_id','slate_items','elapsed_min','prev1_click_cnt','prev1_longview_cnt','slate_has_rand',
        'slate_dur_mean_min','start_hour','day_of_week').collect()
    check = feedback.join(base, on='request_id', validate='1:1').join(test, on='request_id', validate='1:1', suffix='_test')
    assert (check['session_id'] == check['session_id_test']).all()
    assert (check['user_id'] == check['user_id_raw']).all()
    assert (check['req_idx'] == check['req_idx_raw']).all()
    assert check['observed_items'].equals(check['slate_items'])
    np.testing.assert_allclose(check['recomputed_elapsed_min'],check['elapsed_min'],rtol=0,atol=1e-12)
    # 独立交叉核对：当前反馈应等于下一屏的prev1，而非错连到本屏prev1。
    previous = feedback.select('session_id', (pl.col('req_idx')+1).alias('req_idx'),
                              pl.col('curr_click_cnt').alias('prev_click'),pl.col('curr_longview_cnt').alias('prev_long'))
    shifted = check.join(previous, on=['session_id','req_idx'], how='inner', validate='1:1')
    assert (shifted['prev1_click_cnt'] == shifted['prev_click']).all()
    assert (shifted['prev1_longview_cnt'] == shifted['prev_long']).all()
    feedback = check.drop('session_id_test','slate_items','observed_items','user_id','req_idx',
                          'prev1_click_cnt','prev1_longview_cnt','elapsed_min').sort('request_id')
    return feedback, dict(rows=feedback.height,events=293388,events_per_request=6,
        ordered_video_ids_exact=True,elapsed_recomputed_exact=True,metadata_and_labels_exact=True,
        previous_screen_feedback_crosscheck_rows=shifted.height)


def pearson(x, y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    a,b=x-x.mean(),y-y.mean()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denominator) if denominator else None


def grouped_spearman(frame, scores, targets, groups):
    """组内平均rank相关，再按有效组请求数加权；常量分数/目标组无定义而排除。"""
    ranked=frame.select(*groups,*scores,*targets).with_columns(
        *[pl.col(col).rank('average').over(groups).alias(col) for col in scores+targets])
    pairs=[(score,target) for score in scores for target in targets]
    ranges={name:f'range_{i}' for i,name in enumerate(scores+targets)}
    result=ranked.group_by(groups).agg(pl.len().alias('n'),
        *[pl.corr(score,target).alias(str(i)) for i,(score,target) in enumerate(pairs)],
        *[(pl.col(name).max()-pl.col(name).min()).alias(column) for name,column in ranges.items()])
    records={}
    for i,pair in enumerate(pairs):
        # 大常量组的浮点Pearson实现偶尔会返回有限数；不能只靠is_finite识别无定义组。
        valid=result.filter(pl.col(str(i)).is_finite() & (pl.col(ranges[pair[0]])>0) & (pl.col(ranges[pair[1]])>0))
        n=valid['n'].sum()
        records[pair]=dict(value=float((valid[str(i)]*valid['n']).sum()/n) if n else None,
                           eligible_rows=int(n),eligible_groups=valid.height,total_groups=result.height)
    return records


def user_auc(frame, scores, label):
    """与现有User-GAUC一致：剔除单标签用户，按请求数加权，ties取平均rank。"""
    ranked=frame.select('user_id_raw',label,*scores).with_columns(
        *[pl.col(col).rank('average').over('user_id_raw').alias(col) for col in scores])
    groups=ranked.group_by('user_id_raw').agg(pl.len().alias('n'),pl.col(label).sum().alias('pos'),
        *[(pl.col(col)*pl.col(label)).sum().alias(col) for col in scores])
    valid=groups.filter((pl.col('pos')>0)&(pl.col('pos')<pl.col('n')))
    n=valid['n'].sum()
    result={}
    for score in scores:
        auc=(valid[score]-valid['pos']*(valid['pos']+1)/2)/(valid['pos']*(valid['n']-valid['pos']))
        result[score]=dict(value=float((auc*valid['n']).sum()/n) if n else None,
            eligible_rows=int(n),eligible_users=valid.height,total_users=groups.height)
    return result


def subset(frame, name):
    if name=='all': return frame
    if name=='exit': return frame.filter(pl.col('y_continue')==0)
    if name=='continue': return frame.filter(pl.col('y_continue')==1)
    if name=='first_or_30plus': return frame.filter((pl.col('req_idx_raw')==1)|(pl.col('req_idx_raw')>=30))
    if name=='first': return frame.filter(pl.col('req_idx_raw')==1)
    if name=='30plus': return frame.filter(pl.col('req_idx_raw')>=30)
    raise ValueError(name)


def metrics_for_scope(frame, scores, budget, seed, scope):
    data=subset(frame,scope)
    if data.is_empty(): raise ValueError(f'Empty subset {scope}')
    group_metrics={name:grouped_spearman(data,scores,TARGETS,keys) for name,keys in
                   [('within_step',['req_idx_raw']),('within_user',['user_id_raw']),
                    ('within_user_step',['user_id_raw','req_idx_raw'])]}
    rows=[]
    for score in scores:
        for target in TARGETS:
            value=pearson(rankdata(data[score].to_numpy()),rankdata(data[target].to_numpy()))
            rows.append(dict(budget=budget,seed=seed,scope=scope,score=score,target=target,metric='spearman',
                             value=value,eligible_rows=data.height,eligible_groups=1,total_groups=1))
            for name,metrics in group_metrics.items():
                rows.append(dict(budget=budget,seed=seed,scope=scope,score=score,target=target,metric=name,
                                 **metrics[(score,target)]))
    for label in ['curr_any_longview','curr_any_click']:
        users=user_auc(data,scores,label)
        for score in scores:
            rows.append(dict(budget=budget,seed=seed,scope=scope,score=score,target=label,metric='auc',
                value=float(roc_auc_score(data[label],data[score])) if data[label].n_unique()==2 else None,
                eligible_rows=data.height,eligible_groups=1,total_groups=1))
            value=users[score]
            rows.append(dict(budget=budget,seed=seed,scope=scope,score=score,target=label,metric='user_auc',
                value=value['value'],eligible_rows=value['eligible_rows'],eligible_groups=value['eligible_users'],total_groups=value['total_users']))
    return rows


def aligned_predictions(feedback,budget,seed,source_hashes):
    frame=feedback
    scores=[]
    base=None
    for model in MODELS:
        path=PREDICTIONS/f'{seed}/{model}/{budget}ep/predictions.parquet'
        pred=pl.read_parquet(path)
        source_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert pred.height==pred['request_id'].n_unique()==48898
        metadata=pred.select('request_id','session_id','user_id_raw','tab_raw','req_idx_raw','y_continue','elapsed_min_raw')
        if base is None: base=metadata
        else: assert base.equals(metadata)
        columns=['prediction'] if model.startswith('fusion') else ['prediction','content_stay','gamma','log_survival']
        named={col:f'{model}/{col}' for col in columns}
        frame=frame.join(pred.select(pl.col('request_id').cast(pl.UInt32),*[pl.col(col).alias(named[col]) for col in columns]),
                         on='request_id',how='left',validate='1:1')
        scores.extend(named.values())
    verify=frame.select('request_id','session_id','user_id_raw','req_idx_raw','y_continue').sort('request_id')
    expected=base.select(pl.col('request_id').cast(pl.UInt32),'session_id','user_id_raw','req_idx_raw','y_continue').sort('request_id')
    assert verify.equals(expected)
    rawbase=pl.read_parquet(DATA/'features_raw.parquet',columns=['request_id','slate_prior_ctr_mean','slate_prior_longview_mean'])
    frame=frame.join(rawbase,on='request_id',how='left',validate='1:1').with_columns(
        pl.col('req_idx_raw').cast(pl.Float64).alias('baseline/step'),
        pl.col('recomputed_elapsed_min').alias('baseline/elapsed'),
        (pl.col('recomputed_elapsed_min').clip(lower_bound=0)/pl.col('req_idx_raw').clip(lower_bound=1)).log1p().alias('baseline/pace'),
        pl.col('slate_prior_ctr_mean').alias('baseline/prior_ctr'),
        pl.col('slate_prior_longview_mean').alias('baseline/prior_longview'))
    scores += ['baseline/step','baseline/elapsed','baseline/pace','baseline/prior_ctr','baseline/prior_longview']
    assert all(frame[col].null_count()==0 and frame[col].is_finite().all() for col in scores+TARGETS)
    return frame,scores


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    (output/'logs').mkdir()
    source_paths=[DATA/'request_events.parquet',DATA/'events.parquet',DATA/'requests_base.parquet',
                  DATA/'features_raw.parquet',DATA/'test.parquet',DATA/'manifest.json',PREDICTIONS/'config.json',
                  PREDICTIONS/'summary.json',PREDICTIONS/'audit.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    with (output/'logs/run.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
        print('READ_ONLY_BACKTEST_START',flush=True)
        feedback,audit=build_feedback()
        feedback.write_parquet(output/'feedback.parquet')
        print('FEEDBACK_AUDIT',json.dumps(audit),flush=True)
        rows=[]
        for budget in [3,1]:
            for seed in [2026,2027,2028]:
                frame,scores=aligned_predictions(feedback,budget,seed,hashes)
                for scope in SCOPES:
                    rows.extend(metrics_for_scope(frame,scores,budget,seed,scope))
                    print('METRICS',budget,seed,scope,subset(frame,scope).height,flush=True)
                # 单份对齐表方便逐请求debug，不改原有预测。
                if budget==3 and seed==2026: frame.write_parquet(output/'aligned_2026_3ep.parquet')
        metrics=pl.DataFrame(rows)
        metrics.write_parquet(output/'metrics.parquet')
        aggregates=metrics.group_by('budget','scope','score','target','metric').agg(
            pl.col('value').mean().alias('mean'),pl.col('value').std(ddof=1).alias('seed_std'),
            pl.col('value').is_not_null().sum().alias('valid_seeds'),
            pl.col('eligible_rows').min().alias('eligible_rows_min'),pl.col('eligible_rows').max().alias('eligible_rows_max'))
        aggregates.sort('budget','scope','score','target','metric').write_parquet(output/'aggregates.parquet')
        # 深度桶中的真实反馈，显示跨屏号相关的基线来源。
        buckets=feedback.with_columns(pl.when(pl.col('req_idx_raw')==1).then(pl.lit('1'))
            .when(pl.col('req_idx_raw')==2).then(pl.lit('2')).when(pl.col('req_idx_raw')==3).then(pl.lit('3'))
            .when(pl.col('req_idx_raw')<=5).then(pl.lit('4–5')).when(pl.col('req_idx_raw')<=10).then(pl.lit('6–10'))
            .when(pl.col('req_idx_raw')<=20).then(pl.lit('11–20')).when(pl.col('req_idx_raw')<=50).then(pl.lit('21–50'))
            .otherwise(pl.lit('51+')).alias('step_bucket')).group_by('step_bucket').agg(
                pl.len().alias('rows'),pl.col('y_continue').mean().alias('continue_rate'),
                *[pl.col(target).mean().alias(target) for target in TARGETS])
        buckets.write_parquet(output/'depth_feedback.parquet')
        for relative,digest in hashes.items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative
        manifest=dict(status='PASS',data='KR1K v3 test=20220508; 48898 requests / 855 users',
            inference='reuse plan13 saved predictions, no retraining or model inference',
            original_training='full train1721745, valid57872; 3 continuous epochs/1263 updates; budgets1/3; seeds2026/2027/2028; best full-valid LogLoss',
            feedback_audit=audit,scopes={name:subset(feedback,name).height for name in SCOPES},
            targets=TARGETS,scores=scores,metric_rows=metrics.height,source_hashes=hashes,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            direction='higher score is hypothesized to mean higher current engagement; no sign flip',
            caveats=['observational diagnostic, not causal purity identification',
                     'pace shares current-screen timing information with watch targets',
                     'terminal conditioning and session reconstruction can induce selection effects'])
        (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        print('COMPLETE',metrics.height,flush=True)
    print(output/'manifest.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/plan16_component_backtest_checked')
    args=parser.parse_args()
    run(args.output)


if __name__=='__main__':
    main()
