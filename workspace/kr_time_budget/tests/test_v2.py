import numpy as np
import polars as pl
import pytest
import torch
from torch.nn import functional as F
from workspace.kr_time_budget.pipeline.features import B1, B2, B3, B4, NUMS, CATS
from workspace.kr_time_budget.pipeline.features_v2 import manager_v2, own_session_priors, STATIC_BUDGET_NUMS
from workspace.kr_time_budget.pipeline.run_exit import state_hash
from workspace.kr_time_budget.pipeline.run_exit_v2 import scores
from rosen.data.dataset import arrow_table_to_batch_dict
from rosen.data.padding import FeatureContext, TensorFormatter
from rosen.utils.optimize import split_model_params_by_decay
from workspace.kr_time_budget.pipeline.time_budget_v2 import ExitModelV2


@pytest.fixture
def model_batch(tmp_path):
    data={c:[.1,.9] for c in NUMS}
    data.update({c:['a','b'] for c in CATS})
    data.update(slate_tags=[[1,2,3,4,5,6]]*2,history_tags=[[1,2],[3]],slate_durations=[[.1]*6,[.2]*6],y_continue=[0,1])
    raw=pl.DataFrame(data)
    mgr=manager_v2(tmp_path,fit=True);mgr.fit(raw.lazy())
    frame=mgr.transform(raw.lazy()).collect()
    batch=TensorFormatter(FeatureContext.from_manager(mgr)).format(arrow_table_to_batch_dict(frame.to_arrow())).interaction
    batch.update(elapsed_minutes=torch.tensor([1.,20.]),delta_minutes=torch.ones(2))
    torch.manual_seed(2026)
    return ExitModelV2(mgr),batch,mgr


def test_heldout_users_use_own_past_sessions_with_m5_smoothing():
    sessions=pl.DataFrame({'user_id':[1,1,2,2],'prior_available_date':[20220102,20220106,20220102,20220106],'session_wall_s':[10.,10000.,100.,90000.],'session_requests':[1.,1000.,10.,9000.]})
    query=pl.DataFrame({'request_id':[0,1,2],'user_id':[1,2,3],'date':[20220103]*3})
    train=pl.DataFrame({'user_id':[1]})
    r=own_session_priors(query,sessions,train).sort('request_id')
    assert r['own_prior_n'].to_list()==[1.,1.,0.]
    assert r['user_hist_session_s'].to_list()==[10.,25.,10.]
    changed=sessions.with_columns(pl.when(pl.col('prior_available_date')==20220106).then(1e9).otherwise(pl.col('session_wall_s')).alias('session_wall_s'))
    assert own_session_priors(query,changed,train).sort('request_id').equals(r)


def test_v2_initialization_tag_only_and_bias(model_batch):
    model,b,mgr=model_batch
    hashes=[]
    for mode in ['content','budget','fusion','ours']:
        torch.manual_seed(2026);m=ExitModelV2(mgr,mode)
        hashes.append(state_hash(m))
    assert len(set(hashes))==1
    assert model.content_net[-1].bias.item()==pytest.approx(2.6)
    assert not any('items' in s.field_name for s in mgr.settings)
    assert not set(map(id,model.budget_encoder.parameters()))&set(map(id,model.content_encoder.parameters()))
    for p in model.content_encoder.emb_modules.values():
        assert .05 < p.embedding.weight.std().item() < .2


def test_budget_routes_only_session_start_features_and_is_monotone(model_batch):
    model,b,mgr=model_batch
    baseline=model.budget_raw(b)
    changed=dict(b)
    for c in NUMS:
        if c not in STATIC_BUDGET_NUMS:
            changed[c]=torch.ones_like(b[c])*100
    changed.update(slate_tags=torch.zeros_like(b['slate_tags']),history_tags=torch.zeros_like(b['history_tags']),slate_durations=torch.ones_like(b['slate_durations'])*100,elapsed_minutes=torch.tensor([100.,1000.]))
    assert torch.equal(model.budget_raw(changed),baseline)
    samples={k:v[:1].expand((7,)+v.shape[1:]) for k,v in b.items()}
    samples['elapsed_minutes']=torch.tensor([0.,1.,5.,10.,30.,60.,1000.])
    raw=model.budget_raw(samples)
    assert torch.allclose(raw,raw[:1].expand_as(raw))
    gamma,s,log_s=model.budget_components(samples)
    assert (log_s.diff()<=0).all()
    assert torch.isfinite(gamma).all()


def test_stage2_content_updates_cannot_move_budget(model_batch):
    model,b,mgr=model_batch
    model.set_stage(1)
    model.calculate_loss(b).backward()
    assert all(p.grad is None for p in model.content_net.parameters())
    model.zero_grad(set_to_none=True)
    model.set_stage(2)
    before=model.budget_components(b)[0].detach().clone()
    pred,gamma,stay,_,_=model.components(b)
    assert torch.equal(pred,gamma*stay)
    assert not gamma.requires_grad
    assert .85 < stay.mean().item() < .98
    opt=torch.optim.AdamW(split_model_params_by_decay(model,weight_decay=1e-6),lr=.001)
    model.calculate_loss(b).backward()
    assert all(p.grad is None for p in model.budget_encoder.parameters())
    assert all(p.grad is None for p in model.budget_net.parameters())
    assert any(p.grad is not None for p in model.content_net.parameters())
    opt.step()
    assert torch.equal(before,model.budget_components(b)[0])


def test_intra_step_auc_controls_exact_progress():
    df=pl.DataFrame({'y_continue':[0,1,0,1], 'prediction':[.1,.9,.2,.8], 'user_id':[0,0,1,1], 'intra_step_group':[1,1,2,2], 'active_step_group':[0,0,1,1]})
    assert scores(df)['intra_step_gauc']['auc']==1.
    flat=df.with_columns(pl.Series('prediction',[.1,.1,.2,.2]))
    assert scores(flat)['intra_step_gauc']['auc']==.5
