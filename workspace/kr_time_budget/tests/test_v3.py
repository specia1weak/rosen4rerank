import polars as pl
import pytest
import torch
from rosen.data.dataset import arrow_table_to_batch_dict
from rosen.data.padding import FeatureContext, TensorFormatter
from rosen.core.train.two_stage import StageSnapshot, StageReport
from workspace.kr_time_budget.pipeline.build_requests_v3 import aggregate, user_priors, item_priors
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, FEATURES, USER_CATEGORY_FIELDS, BUDGET_PARAMETER_FIELDS
from workspace.kr_time_budget.pipeline.time_budget_v3 import ExitModelV3
from workspace.kr_time_budget.pipeline.run_exit import state_hash


def test_past_priors_exclude_current_and_future_and_snapshot():
    q=pl.DataFrame({'request_id':[0,1],'user_id':[1,1],'date':[20220505,20220507]})
    sess=pl.DataFrame({'user_id':[1,1,1],'end_date':[20220504,20220506,20220508],
                       'session_min':[10.,20.,10000.],'session_req_cnt':[2.,4.,2000.]})
    r=user_priors(q,sess).sort('request_id')
    assert r['user_hist_session_min'].to_list()==[10.,15.]
    obs=pl.DataFrame({'video_id':[1,1,1],'date':[20220504,20220506,20220508],'is_click':[0,1,1],'long_view':[1,0,1]})
    query=q.select('request_id','date').with_columns(pl.lit([1]*6).alias('slate_items'))
    p=item_priors(query,obs).sort('request_id')
    assert p['slate_prior_ctr_mean'].to_list()==[0.,.5]
    assert p['slate_prior_longview_mean'].to_list()==[1.,.5]


def test_history_only_previous_chunks_tags_are_tokens():
    n=18
    e=pl.DataFrame({'user_id':[1]*n,'session_id':[1]*n,'req_idx':[1]*6+[2]*6+[3]*6,
        'video_id':list(range(n)),'tag':['20,43']*n,'video_type':['NORMAL']*n,'music_type':['1']*n,
        'duration_ms':[60000]*n,'author_id':[1]*n,'is_rand':[0]*n,'is_click':[1]*6+[0]*12,
        'long_view':[0]*n,'positive':[1]*6+[0]*12,'is_hate':[0]*n,'time_ms':[i*1000 for i in range(n)],
        'session_start_ms':[0]*n,'play_clip':[1000]*n})
    r=aggregate(e)
    assert r['y_continue'].to_list()==[1,1,0]
    assert r['prev1_click_cnt'].to_list()==[0,6,0]
    assert r['sess_cum_click_rate'].to_list()==[0.,1.,.5]
    assert r['slate_unique_tag_cnt'].to_list()==[2]*3
    assert r['is_first_req'].to_list()==[1,0,0]
    changed=e.with_columns(pl.when(pl.col('req_idx')>=2).then(1).otherwise(pl.col('is_click')).alias('is_click'))
    a=aggregate(changed)
    assert a['hist_tags'][:2].to_list()==r['hist_tags'][:2].to_list()
    assert a['sess_cum_click_rate'][:2].to_list()==r['sess_cum_click_rate'][:2].to_list()


@pytest.fixture
def model_batch(tmp_path):
    mgr=request_manager(tmp_path)
    data={s.field_name: ([0.,1.] if not s.requires_embedding_module else ['1','2']) for s in mgr.settings}
    for c in ['hist_tags','slate_tags']:data[c]=['20,43','20,67']
    for c in ['slate_video_types','slate_music_types']:data[c]=['1,1','1,2']
    data['y_continue']=[0,1]
    raw=pl.DataFrame(data)
    # min_freq=10 is deliberate in production; repeat here to exercise real user vocab.
    raw=pl.concat([raw]*6)
    mgr.fit(raw.lazy())
    frame=mgr.transform(raw.lazy()).collect()
    b=TensorFormatter(FeatureContext.from_manager(mgr)).format(arrow_table_to_batch_dict(frame.to_arrow())).interaction
    torch.manual_seed(2026)
    return ExitModelV3(mgr),b,mgr


def test_schema_native_multitag_and_parameter_ownership(model_batch):
    m,b,mgr=model_batch
    assert len(FEATURES)==54
    assert mgr.get_setting('slate_tags').vocab['20']!=mgr.get_setting('slate_tags').vocab['43']
    assert b['slate_tags'].shape[-1]==12
    setting=mgr.get_setting('slate_tags')
    vector=m.omni_embedding(b,split_by='name')['slate_tags']
    native=setting.compute_tensor(b,m.omni_embedding.emb_modules)
    assert torch.equal(vector,native)
    names=[name for name,_ in m.named_parameters(remove_duplicate=False)]
    assert len(names)==len({id(p) for _,p in m.named_parameters(remove_duplicate=False)})
    hashes=[]
    for mode in ['content','budget','fusion','ours']:
        torch.manual_seed(2026);hashes.append(state_hash(ExitModelV3(mgr,mode)))
    assert len(set(hashes))==1


def test_budget_static_and_stage_transition_freezes_every_budget_parameter(model_batch):
    m,b,mgr=model_batch
    raw=m.budget_raw(b)
    changed={c:(v if c in BUDGET_PARAMETER_FIELDS else v*0) for c,v in b.items()}
    assert torch.equal(raw,m.budget_raw(changed))
    m.calculate_loss(b).backward()
    assert all(p.grad is None for p in m.content_net.parameters())
    snapshot=StageSnapshot.capture(m,StageReport(1,1,{},'student',{}))
    stage2=m.initialize_stage_two(snapshot)
    assert stage2 is not m
    before=stage2.budget_components(b)[0].detach().clone()
    opt=torch.optim.AdamW([p for p in stage2.parameters() if p.requires_grad],lr=.001,weight_decay=0)
    pred,gamma,stay,_,_=stage2.components(b)
    assert torch.equal(pred,gamma*stay) and not gamma.requires_grad
    stage2.calculate_loss(b).backward();opt.step()
    assert torch.equal(before,stage2.budget_components(b)[0])
    assert all(not p.requires_grad for module in stage2.budget_modules for p in module.parameters())
