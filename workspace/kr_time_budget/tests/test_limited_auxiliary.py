"""弱辅助上限、stop-gradient及关闭时精确回归。"""
import pytest
import torch
from rosen.core.interaction import Interaction
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.limited_auxiliary_exit import cap_auxiliary, LimitedAuxiliaryExitModel
from workspace.kr_time_budget.pipeline.exit_model import KRExitModel
from workspace.kr_time_budget.pipeline.exit_training import state_hash, joint_parameter_groups


@pytest.mark.parametrize('auxiliary',[0., .001, .02, .4, 10000.])
def test_cap_value_and_detached_ratio_preserve_main_and_aux_gradients(auxiliary):
    main=torch.tensor(.22,dtype=torch.float64,requires_grad=True)
    aux=torch.tensor(auxiliary,dtype=torch.float64,requires_grad=True)
    weighted,scale=cap_auxiliary(main,aux)
    total=main+weighted
    assert 0 <= weighted <= .1*main+1e-14
    assert not scale.requires_grad
    total.backward()
    assert main.grad.item()==1.
    assert aux.grad.item()==scale.item()
    if auxiliary>0:
        assert aux.grad>0  # 不可让aux/aux自行相消而关闭辅助梯度。
    if auxiliary<=.022:
        assert scale==1


@pytest.mark.parametrize('weights',[(.1,0.),(0.,.1),(.1,.1)])
@pytest.mark.parametrize('mixture',[False,True])
def test_actual_loss_limits_combined_terms_both_heads_receive_gradients(model_batch,weights,mixture):
    _,raw,manager=model_batch
    n=len(raw['y_continue'])
    batch=Interaction(dict(raw,user_id=torch.zeros_like(raw['user_id']),
        elapsed_min_raw=raw['elapsed_min'],req_idx_raw=torch.arange(n)+1))
    model=LimitedAuxiliaryExitModel(manager,step_only=True,mixture=mixture,add_pace=True,
        left_weight=weights[0],right_weight=weights[1],requests_per_session=15.7426761)
    prediction=model.predict(batch).detach().clone()
    terms=model.loss_terms(batch)
    loss=model.calculate_loss(batch)
    row=model.last_loss_terms
    assert torch.equal(prediction,model.predict(batch))
    assert row['auxiliary_main_ratio']<=.1+1e-12
    assert row['effective_left_weight']<=.1 and row['effective_right_weight']<=.1
    assert abs(float(loss.detach())-row['joint_bce']-row['auxiliary_weighted'])<1e-12
    # 对比显式冻结系数的参考梯度，检验主Loss本身未被一起缩放。
    params=[model.budget_net[-1].weight,model.content_net[-1].weight]
    actual=torch.autograd.grad(loss,params)
    expected=terms['joint_bce']+row['effective_left_weight']*terms['left_session_nll']+row['effective_right_weight']*terms['right_bce']
    reference=torch.autograd.grad(expected,params)
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-8)
        assert torch.isfinite(a).all() and a.abs().sum()>0


@pytest.mark.parametrize('mixture',[False,True])
def test_zero_ratio_is_exact_old_bce_adam_update(model_batch,mixture):
    _,raw,manager=model_batch
    batch=Interaction(dict(raw,elapsed_min_raw=raw['elapsed_min'],
        req_idx_raw=torch.arange(len(raw['y_continue']))+1))
    models=[]
    for cls in [KRExitModel,LimitedAuxiliaryExitModel]:
        torch.manual_seed(2026)
        kwargs=dict(left_weight=.1,right_weight=.1,max_aux_ratio=0.) if cls is LimitedAuxiliaryExitModel else {}
        models.append(cls(manager,step_only=True,mixture=mixture,add_pace=True,**kwargs))
    assert state_hash(models[0])==state_hash(models[1])
    losses=[]
    for model in models:
        optimizer=torch.optim.AdamW(joint_parameter_groups(model),weight_decay=0)
        loss=model.calculate_loss(batch);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step();losses.append(loss)
    assert torch.equal(losses[0],losses[1])
    assert state_hash(models[0])==state_hash(models[1])


def test_reject_invalid_cap_and_nonfinite_loss():
    for value in [-.1,.11,float('nan')]:
        with pytest.raises(ValueError):
            cap_auxiliary(torch.tensor(1.),torch.tensor(1.),value)
    with pytest.raises(FloatingPointError):
        cap_auxiliary(torch.tensor(1.),torch.tensor(float('nan')))
