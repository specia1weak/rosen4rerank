import pytest
import torch
from rosen.core.train.two_stage import StageSnapshot, StageReport
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.time_budget_v3 import ExitModelV3
from workspace.kr_time_budget.pipeline.time_budget_v4 import ExitModelV4, single_weibull_survival, weibull_parameters
from workspace.kr_time_budget.pipeline.run_exit import state_hash


def test_ifr_survival_matches_definition_and_both_curves_decrease():
    t=torch.tensor([0.,.01,.1,1.,10.,100.,1000.])
    raw=torch.tensor([[0.,-2.]]).expand(len(t),-1).clone().requires_grad_()
    d=torch.ones_like(t)
    gamma,s,log_s=single_weibull_survival(raw,t,d)
    scale,shape=weibull_parameters(raw)
    expected=(-((t.double()+1)/scale.double()).pow(shape.double())+(t.double()/scale.double()).pow(shape.double())).exp()
    assert torch.allclose(gamma.double(),expected,atol=3e-8,rtol=1e-6)
    assert (shape>1).all() and (gamma.diff()<=0).all() and (log_s.diff()<=0).all()
    gamma.sum().backward();assert torch.isfinite(raw.grad).all()
    same,_,_=single_weibull_survival(raw.detach(),t,torch.zeros_like(t))
    assert torch.equal(same,torch.ones_like(t))


def test_large_elapsed_has_no_cancellation_rebound():
    t=torch.tensor([0.,1.,100.,10000.,1000000.])
    for raw_scale in [-100.,0.,100.]:
        raw=torch.tensor([[raw_scale,100.]]).expand(len(t),-1)
        gamma,s,logs=single_weibull_survival(raw,t,torch.ones_like(t))
        assert torch.isfinite(gamma).all() and torch.isfinite(logs).all()
        assert (gamma.diff()<=0).all() and (logs.diff()<=0).all()


def test_common_initialization_and_unchanged_baselines(model_batch):
    _,b,manager=model_batch
    hashes=[]
    for mode in ['content','budget','fusion','ours']:
        torch.manual_seed(2026);old=ExitModelV3(manager,mode)
        torch.manual_seed(2026);new=ExitModelV4(manager,mode)
        for name,tensor in old.state_dict().items():
            if name.startswith('budget_net.2.'):continue
            assert torch.equal(tensor,new.state_dict()[name]),name
        hashes.append(state_hash(new))
        if mode in ['content','fusion']:assert torch.equal(old.predict(b),new.predict(b))
    assert len(set(hashes))==1


def test_zero_transition_and_frozen_budget_with_trainable_content(model_batch):
    _,b,manager=model_batch
    model=ExitModelV4(manager)
    warm=model.predict(b).detach()
    assert (warm<=model.prior_stay).all()
    assert model.prior_stay.item()==pytest.approx(.93086158)
    model.calculate_loss(b).backward()
    assert all(p.grad is None for p in model.content_net.parameters())
    snapshot=StageSnapshot.capture(model,StageReport(1,1,{},'student',{}))
    stage2=model.initialize_stage_two(snapshot)
    assert torch.equal(warm,stage2.predict(b))
    assert torch.equal(stage2.content_net[-1].weight,torch.zeros_like(stage2.content_net[-1].weight))
    before=stage2.budget_components(b)[0].detach().clone()
    opt=torch.optim.AdamW([p for p in stage2.parameters() if p.requires_grad],lr=.001,weight_decay=0)
    loss=stage2.calculate_loss(b);loss.backward()
    assert stage2.content_net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for module in stage2.budget_modules for p in module.parameters())
    opt.step();assert torch.equal(before,stage2.budget_components(b)[0])
