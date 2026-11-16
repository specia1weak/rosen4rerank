"""Ordered raw events -> requests -> temporal features -> native rosen schema."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import polars as pl
from rosen.datasets.kuairand import KuaiRandDataset
from workspace.kr_time_budget.pipeline.build_requests import sessionize
from workspace.kr_time_budget.pipeline.features import past_daily
from workspace.kr_time_budget.pipeline.kr_request_schema import (request_manager, FEATURES, USER_CATEGORY_FIELDS, LOG1P_FIELDS, EXTRA_COLUMNS)

ROOT = Path('workspace/kr_time_budget/data/v3')
TRAIN_END = 20220506


def build_sessions(events):
    sessions = events.group_by(['user_id', 'session_id'], maintain_order=True).agg(
        pl.col('date').first().alias('session_date'), pl.col('date').max().alias('end_date'),
        (pl.col('hourmin').first() // 100).alias('start_hour'), pl.col('tab').first().alias('tab'),
        ((pl.col('time_ms').last() - pl.col('time_ms').first() + pl.col('play_clip').last()) / 60000).alias('session_min'),
        (pl.len() // 6).cast(pl.Float64).alias('session_req_cnt'), pl.len().alias('session_n'),
        pl.col('first_session').first(), pl.col('last_session').first(),
    ).sort('session_id')
    return sessions.with_columns(
        (pl.col('session_id').cum_count().over(['user_id','session_date']) - 1).alias('day_prev_sess_cnt'),
        (pl.col('session_min').cum_sum().over(['user_id','session_date']) - pl.col('session_min')).alias('day_prev_elapsed_min'))


def aggregate(events):
    """Head remainder is chunk 0: history only, never a supervised sample."""
    g = events.group_by(['user_id','session_id','req_idx'], maintain_order=True).agg(
        pl.len().alias('n'), pl.col('video_id').alias('slate_items'),
        pl.col('tag').str.join(',').alias('slate_tags'),
        pl.col('tag').filter((pl.col('is_click') == 1) | (pl.col('long_view') == 1)).str.join(',').alias('engaged_tags'),
        pl.col('video_type').str.join(',').alias('slate_video_types'),
        pl.col('music_type').str.join(',').alias('slate_music_types'),
        (pl.col('duration_ms').clip(3000,600000).mean() / 60000).alias('slate_dur_mean_min'),
        pl.col('author_id').filter(pl.col('author_id') >= 0).n_unique().alias('slate_unique_author_cnt'),
        pl.col('is_rand').max().alias('slate_has_rand'),
        pl.col('is_click').sum().alias('click_n'), pl.col('long_view').sum().alias('long_n'),
        pl.col('positive').sum().alias('pos_n'), pl.col('is_hate').sum().alias('hate_n'),
        ((pl.col('time_ms').last() - pl.col('session_start_ms').first() + pl.col('play_clip').last()) / 60000).alias('elapsed_min'),
    ).sort(['session_id','req_idx'])
    for col in ['n','click_n','long_n','pos_n','hate_n']:
        g = g.with_columns((pl.col(col).cum_sum().over('session_id')-pl.col(col)).alias('past_'+col))
    g = g.with_columns(
        *[pl.col(c).shift(1).over('session_id').fill_null(0).alias(out) for c,out in
          [('click_n','prev1_click_cnt'),('long_n','prev1_longview_cnt'),('pos_n','prev1_pos_cnt')]],
        *[(pl.col('past_'+c)/pl.col('past_n').clip(lower_bound=1)).alias(out) for c,out in
          [('click_n','sess_cum_click_rate'),('long_n','sess_cum_longview_rate'),('pos_n','sess_cum_pos_rate')]],
        pl.col('past_hate_n').alias('sess_cum_hate_cnt'), (pl.col('req_idx')==1).cast(pl.Int8).alias('is_first_req'),
        pl.concat_str([pl.col('engaged_tags').shift(2).over('session_id').fill_null(''),
                       pl.col('engaged_tags').shift(1).over('session_id').fill_null('')],separator=',').alias('hist_tags'),
        (pl.col('req_idx') < pl.col('req_idx').max().over('session_id')).cast(pl.Int8).alias('y_continue'),
        pl.col('slate_tags').str.split(',').list.eval(pl.element().filter((pl.element()!='') & (pl.element()!='__MISSING__'))).list.n_unique().alias('slate_unique_tag_cnt'),
    )
    return g.filter(pl.col('req_idx')>0).drop(['engaged_tags','n','click_n','long_n','pos_n','hate_n']+['past_'+c for c in ['n','click_n','long_n','pos_n','hate_n']])


def user_priors(query, sessions):
    obs = sessions.select('user_id',pl.col('end_date').alias('date'),'session_min','session_req_cnt')
    own = past_daily(query,obs,['user_id'],['session_min','session_req_cnt'])
    glob = past_daily(query.select('request_id','date'),obs,[],['session_min','session_req_cnt']).select('request_id',
        *[(pl.col('sum_'+c)/pl.col('prior_n').clip(lower_bound=1)).alias('global_'+c) for c in ['session_min','session_req_cnt']])
    return own.join(glob,on='request_id',validate='1:1').select('request_id',pl.col('prior_n').alias('user_prior_n'),
        *[((pl.col('sum_'+c)+5*pl.col('global_'+c))/(pl.col('prior_n')+5)).alias(out) for c,out in
          [('session_min','user_hist_session_min'),('session_req_cnt','user_hist_req_cnt')]])


def item_priors(query, obs):
    vals=['is_click','long_view']
    per = past_daily(query.explode('slate_items').rename({'slate_items':'video_id'}),obs,['video_id'],vals)
    glob = past_daily(query.select('request_id','date'),obs,[],vals).select('request_id',
        *[(pl.col('sum_'+c)/pl.col('prior_n').clip(lower_bound=1)).alias('global_'+c) for c in vals])
    per=per.join(glob,on='request_id').with_columns(
        *[((pl.col('sum_'+c)+20*pl.col('global_'+c))/(pl.col('prior_n')+20)).alias('quality_'+c) for c in vals])
    return per.group_by('request_id').agg(
        pl.col('quality_is_click').mean().alias('slate_prior_ctr_mean'),
        pl.col('quality_long_view').mean().alias('slate_prior_longview_mean'),
        pl.col('prior_n').mean().alias('item_prior_support_mean'),
        (pl.col('prior_n')==0).mean().alias('item_prior_cold_fraction'))


def encode(raw, out):
    raw=raw.with_columns(*[pl.col(c).clip(lower_bound=0).log1p().cast(pl.Float32) for c in LOG1P_FIELDS])
    mgr=request_manager(out/'schema')
    mgr.fit(raw.filter(pl.col('split')=='train').lazy())
    keep=list(dict.fromkeys(mgr.fields()+EXTRA_COLUMNS))
    stats={}
    for split in ['train','valid','test']:
        frame=mgr.transform(raw.filter(pl.col('split')==split).lazy()).select(keep).collect()
        if split=='train': frame=frame.sample(fraction=1,shuffle=True,seed=2026)
        assert frame.null_count().sum_horizontal().item()==0
        frame.write_parquet(out/f'{split}.parquet',row_group_size=65536)
        stats[split]={'rows':frame.height,'users':frame['user_id_raw'].n_unique(),
                      'positive_rate':frame['y_continue'].mean(),
                      'request_order_sha256':hashlib.sha256(frame['request_id'].to_numpy().tobytes()).hexdigest()}
        print('ENCODED',split,stats[split],flush=True)
    return stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=KuaiRandDataset.DATA_KUAI_1K);ap.add_argument('--output',type=Path,default=ROOT)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        print('Already complete',out,flush=True);return
    if not (out/'events.parquet').exists():
        frames=[];sources=[]
        for i,p in enumerate(sorted(args.source.glob('log_*.csv'))):
            f=pl.read_csv(p).with_row_index('source_row').with_columns(pl.lit(i).alias('source_order'))
            frames.append(f);sources.append({'path':str(p),'rows':f.height,'bytes':p.stat().st_size})
            print('READ',p.name,f.height,flush=True)
        events=pl.concat(frames);del frames;cols=[c for c in events.columns if c not in ['source_row','source_order']]
        raw_n=events.height
        events=events.unique(subset=cols,maintain_order=True)
        events=sessionize(events).with_row_index('event_id')
        events.write_parquet(out/'events.parquet',row_group_size=65536)
        (out/'source_manifest.json').write_text(json.dumps({'sources':sources,'raw_rows':raw_n,'deduplicated_rows':events.height,'dedup':'all 19 raw columns'},indent=2))
        print('EVENTS',events.height,flush=True);del events;gc.collect()
    if not (out/'requests_base.parquet').exists():
        events=pl.read_parquet(out/'events.parquet');sessions=build_sessions(events)
        sessions.write_parquet(out/'sessions.parquet')
        usable=sessions.filter((pl.col('session_id')!=pl.col('first_session'))&(pl.col('session_id')!=pl.col('last_session'))&(pl.col('session_n')>=6))
        # Choose scenarios using training-range sessions only, never test counts.
        top=usable.filter(pl.col('end_date')<=TRAIN_END).group_by('tab').len().sort(['len','tab'],descending=[True,False]).head(5)['tab'].to_list()
        usable=usable.filter(pl.col('tab').is_in(top)).with_columns(
            pl.when(pl.col('end_date')<=TRAIN_END).then(pl.lit('train'))
            .when((pl.col('session_date')==20220507)&(pl.col('end_date')==20220507)).then(pl.lit('valid'))
            .when((pl.col('session_date')==20220508)&(pl.col('end_date')==20220508)).then(pl.lit('test')).otherwise(pl.lit('boundary_excluded')).alias('split'))
        excluded=usable.filter(pl.col('split')=='boundary_excluded').height
        usable=usable.filter(pl.col('split')!='boundary_excluded')
        events=events.join(usable.select('session_id'),on='session_id',how='semi').sort(['session_id','event_idx'])
        video=pl.read_csv(args.source/'video_features_basic_1k.csv',columns=['video_id','author_id','tag','video_type','music_type'],schema_overrides={'tag':pl.String})
        events=events.join(video,on='video_id',how='left',validate='m:1',maintain_order='left').with_columns(
            pl.col('tag').fill_null('__MISSING__'),pl.col('video_type').fill_null('__MISSING__'),
            pl.col('music_type').cast(pl.Int64).cast(pl.String).fill_null('__MISSING__'),pl.col('author_id').fill_null(-1),
            pl.any_horizontal(*[pl.col(c)==1 for c in ['is_like','is_comment','is_follow','is_forward']]).cast(pl.Int8).alias('positive'))
        req=aggregate(events).join(usable,on=['user_id','session_id'],how='left',validate='m:1').with_row_index('request_id')
        # Persist event membership, including request-0 remainder, without duplicating event payloads.
        events.select('event_id','user_id','session_id','req_idx','event_idx').join(req.select('session_id','req_idx','request_id'),on=['session_id','req_idx'],how='left').write_parquet(out/'request_events.parquet')
        del events,video;gc.collect()
        users=pl.read_csv(args.source/'user_features_1k.csv').select('user_id',*USER_CATEGORY_FIELDS).with_columns(
            pl.col('is_live_streamer').eq(1).cast(pl.Int8),
            *[pl.col(f'onehot_feat{i}').cast(pl.Int64).fill_null(-1) for i in range(18)])
        req=req.join(users,on='user_id',how='left',validate='m:1').with_columns(
            pl.col('user_id').alias('user_id_raw'),pl.col('tab').alias('tab_raw'),pl.col('req_idx').alias('req_idx_raw'),
            pl.col('req_idx').alias('intra_step_group'),
            pl.col('session_date').cast(pl.String).str.strptime(pl.Date,'%Y%m%d').dt.weekday().alias('day_of_week'),
            (pl.col('session_date').cast(pl.String).str.strptime(pl.Date,'%Y%m%d').dt.weekday()>=6).cast(pl.Int8).alias('is_weekend'),
        ).with_columns(pl.struct('user_active_degree','req_idx').rank('dense').alias('active_step_group'))
        req.write_parquet(out/'requests_base.parquet')
        (out/'structure_manifest.json').write_text(json.dumps({'top_tabs':top,'sessions_total':sessions.height,'sessions_retained':usable.height,'boundary_sessions_excluded':excluded,'requests':req.height},indent=2))
        print('REQUESTS',req.height,'sessions',usable.height,'top',top,flush=True);del req;gc.collect()
    req=pl.read_parquet(out/'requests_base.parquet');sessions=pl.read_parquet(out/'sessions.parquet')
    # All held-out priors are frozen at May 6 end; train queries exclude the entire query day.
    query=req.select('request_id','user_id',pl.when(pl.col('split')=='train').then(pl.col('session_date')).otherwise(pl.lit(20220507)).alias('date'))
    req=req.join(user_priors(query,sessions),on='request_id',validate='1:1')
    print('USER PRIORS complete',flush=True)
    obs=pl.read_parquet(out/'events.parquet',columns=['video_id','date','is_click','long_view']).filter(pl.col('date')<=TRAIN_END)
    item_query=query.select('request_id','date').join(req.select('request_id','slate_items'),on='request_id',validate='1:1')
    req=req.join(item_priors(item_query,obs),on='request_id',validate='1:1')
    del obs;gc.collect();print('ITEM PRIORS complete',flush=True)
    req.write_parquet(out/'features_raw.parquet',row_group_size=65536)
    splits=encode(req,out)
    train_users=req.filter(pl.col('split')=='train')['user_id'].unique()
    audit={s:{'unseen_users':req.filter((pl.col('split')==s)&~pl.col('user_id').is_in(train_users.implode()))['user_id'].n_unique(),
              'user_history_coverage':req.filter(pl.col('split')==s).select((pl.col('user_prior_n')>0).mean()).item(),
              'item_cold_fraction':req.filter(pl.col('split')==s)['item_prior_cold_fraction'].mean()} for s in ['train','valid','test']}
    tags={c:{'cap':cap,'truncated_fraction':req.select((pl.col(c).str.split(',').list.eval(pl.element().filter(pl.element()!='')).list.len()>cap).mean()).item()} for c,cap in [('slate_tags',12),('hist_tags',15)]}
    meta={'version':3,'features':FEATURES,'splits':splits,'audit':audit,'tags':tags,
          'protocol':{'train':'session ends <=20220506','valid':'whole session on 20220507','test':'whole session on 20220508','session_tab':'first event','top_tabs':'top5 train sessions',
                      'train_priors':'strictly earlier calendar days; users completed sessions; items all observed exposures','eval_priors':'frozen through 20220506','user_smoothing':5,'item_smoothing':20,
                      'positive':'any like/comment/follow/forward per event','tags':'repetitions retained; chronological order; head truncation; sum aggregation; independent hist/slate vocab and tables',
                      'head_remainder':'chunk0 history only; participates in previous-chunk stats and last2chunk tags','duration_clip_ms':[3000,600000],'clock':'Leader screen end; delta1min',
                      'content_elapsed':'full-vector content path; excluded only from budget parameter net'},
          'source':json.loads((out/'source_manifest.json').read_text()),'structure':json.loads((out/'structure_manifest.json').read_text())}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2));print('DATA READY',json.dumps(meta['splits']),flush=True)

if __name__=='__main__':main()
