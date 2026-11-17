import polars as pl
import pytest
import torch
from experiments.kuairand.tests.fixtures import model_batch
from experiments.kuairand.pipeline.current_feedback import (
    FEEDBACK_FIELDS,feedback_rows,CurrentFeedbackExitModel,CurrentFeedbackFusion)
from experiments.kuairand.pipeline.limited_auxiliary_exit import LimitedAuxiliaryExitModel
from experiments.kuairand.pipeline.exit_model import KRExitModel


def test_current_aggregation_uses_own_events_and_binary_union():
    d=pl.DataFrame(dict(request_id=[1]*6+[2]*6,is_click=[1,1,0,0,0,0]+[0]*6,
        long_view=[1,0,0,0,0,0]+[1]*6,is_like=[1]+[0]*11,is_comment=[1]+[0]*11,
        is_follow=[0]*12,is_forward=[0]*12,is_hate=[0]*5+[1]+[0]*6))
    a=feedback_rows(d).sort('request_id')
    assert a['feedback_event_count'].to_list()==[6,6]
    assert a[FEEDBACK_FIELDS[0]][0]==pytest.approx(2/6)
    assert a[FEEDBACK_FIELDS[2]][0]==pytest.approx(1/6) # like+comment on one event counted once
    changed=d.with_columns(pl.when(pl.col('request_id')==2).then(1).otherwise(pl.col('is_click')).alias('is_click'))
    assert feedback_rows(changed).filter(pl.col('request_id')==1).equals(a.filter(pl.col('request_id')==1))


@pytest.mark.parametrize('kind',['single','mixture','fusion'])
def test_branch_initial_identity_budget_isolation_and_first_gradient(model_batch,kind):
    _,raw,manager=model_batch;n=len(raw['y_continue'])
    b=dict(raw,req_idx_raw=torch.arange(n)+1,elapsed_min_raw=torch.arange(n).float()+1,
        **{c:torch.full((n,),.5) for c in FEEDBACK_FIELDS})
    kwargs=dict(mode='fusion' if kind=='fusion' else 'ours',add_pace=True,step_only=kind!='fusion',mixture=kind=='mixture')
    torch.manual_seed(2026)
    new=(CurrentFeedbackFusion(manager,**kwargs) if kind=='fusion' else
        CurrentFeedbackExitModel(manager,**kwargs,left_weight=.1,right_weight=.1,requests_per_session=15.))
    torch.manual_seed(2026)
    old=(KRExitModel(manager,**kwargs) if kind=='fusion' else
        LimitedAuxiliaryExitModel(manager,**kwargs,left_weight=.1,right_weight=.1,requests_per_session=15.))
    for name,t in old.state_dict().items():assert torch.equal(t,new.state_dict()[name])
    assert torch.equal(new.predict(b),old.predict(b))
    assert torch.equal(new.calculate_loss(b),old.calculate_loss(b))
    new.calculate_loss(b).backward()
    assert new.feedback_projection.weight.grad.norm()>0
    with torch.no_grad():new.feedback_projection.weight.fill_(.1)
    changed=dict(b,**{c:torch.zeros(n) for c in FEEDBACK_FIELDS})
    assert not torch.equal(new.predict(b),new.predict(changed))
    if kind!='fusion':
        assert torch.equal(new.budget_raw(b),new.budget_raw(changed))
        assert torch.equal(new.budget_components(b)[0],new.budget_components(changed)[0])
    new.use_current_feedback=False
    assert torch.equal(new.predict(b),old.predict(b))
    assert torch.equal(new.predict(b),new.predict(changed))
