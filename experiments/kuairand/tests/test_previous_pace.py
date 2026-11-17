import numpy as np
import polars as pl
import pytest
import torch
from experiments.kuairand.pipeline.previous_pace import add_previous_clock,previous_pace
from experiments.kuairand.tests.fixtures import model_batch


def test_previous_lookup_respects_sessions_and_shuffled_order():
    f=pl.DataFrame(dict(session_id=[1,2,1,2,1],req_idx_raw=[3,1,1,2,2],
                       elapsed_min=[9.,4.,2.,7.,5.],request_id=[13,21,11,22,12]))
    out=add_previous_clock(f)
    np.testing.assert_array_equal(out['prev_elapsed_min_raw'],[5,0,0,4,2])
    np.testing.assert_array_equal(out['prev_req_idx_raw'],[2,0,0,1,1])
    assert out.select(f.columns).equals(f)


def test_changing_current_or_future_clock_cannot_change_current_previous_pace():
    f=pl.DataFrame(dict(session_id=[1,1,1],req_idx_raw=[1,2,3],elapsed_min=[2.,5.,9.]))
    a=add_previous_clock(f)
    b=add_previous_clock(f.with_columns(pl.when(pl.col('req_idx_raw')>=2).then(1000.).otherwise(pl.col('elapsed_min')).alias('elapsed_min')))
    assert a['prev_elapsed_min_raw'][1]==b['prev_elapsed_min_raw'][1]==2.
    expected=np.log1p([0,2,2.5])
    batch={c:torch.tensor(a[c].to_numpy()) for c in ['prev_elapsed_min_raw','prev_req_idx_raw']}
    np.testing.assert_allclose(previous_pace(batch),expected,rtol=1e-6)


def test_incomplete_previous_request_rejected():
    with pytest.raises(ValueError,match='missing previous'):
        add_previous_clock(pl.DataFrame(dict(session_id=[1],req_idx_raw=[2],elapsed_min=[2.])))


@pytest.mark.parametrize('mixture',[False,True])
def test_model_previous_pace_ignores_current_clock_with_nonzero_pace_weight(model_batch,mixture):
    from experiments.kuairand.pipeline.previous_pace import PreviousPaceExitModel
    from experiments.kuairand.pipeline.limited_auxiliary_exit import LimitedAuxiliaryExitModel
    from experiments.kuairand.pipeline.exit_training import state_hash
    _,raw,manager=model_batch
    n=len(raw['y_continue'])
    batch=dict(raw,elapsed_min_raw=torch.arange(n).float()+5,req_idx_raw=torch.arange(n)+1,
        prev_elapsed_min_raw=torch.arange(n).float(),prev_req_idx_raw=torch.arange(n))
    torch.manual_seed(2026)
    previous=PreviousPaceExitModel(manager,step_only=True,mixture=mixture,add_pace=True,
        left_weight=.1,right_weight=.1,pace_scope='previous')
    torch.manual_seed(2026)
    current=LimitedAuxiliaryExitModel(manager,step_only=True,mixture=mixture,add_pace=True,
        left_weight=.1,right_weight=.1)
    assert state_hash(previous)==state_hash(current)
    # 初始化新pace列为0可能掩盖路由错误，故主动给非零权重再验证。
    with torch.no_grad():previous.content_net[0].weight[:,-1].fill_(.2)
    changed=dict(batch,elapsed_min_raw=batch['elapsed_min_raw']+10000,elapsed_min=batch['elapsed_min_raw']+10000)
    assert torch.equal(previous.predict(batch),previous.predict(changed))
    previous.calculate_loss(batch).backward()
    assert torch.isfinite(previous.content_net[0].weight.grad).all()
