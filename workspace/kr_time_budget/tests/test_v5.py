import pytest
import torch
from rosen.core.train.two_stage import StageSnapshot, StageReport
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.time_budget_v4 import ExitModelV4
from workspace.kr_time_budget.pipeline.time_budget_v5 import ExitModelV5
from workspace.kr_time_budget.pipeline.run_exit import state_hash


def test_only_prior_buffer_changes_at_initialization_and_baselines_match(model_batch):
    _,b,manager=model_batch
    hashes=[]
    for mode in ['content','budget','fusion','ours']:
        torch.manual_seed(2026);old=ExitModelV4(manager,mode)
        torch.manual_seed(2026);new=ExitModelV5(manager,mode)
        for name,tensor in old.state_dict().items():
            if name!='prior_stay':assert torch.equal(tensor,new.state_dict()[name]),name
        assert new.prior_stay.item()==pytest.approx(.97068777)
        hashes.append(state_hash(new))
        if mode in ['content','fusion']:assert torch.equal(old.predict(b),new.predict(b))
    assert len(set(hashes))==1


def test_ours_content_clock_invariance_and_fusion_clock_visibility(model_batch):
    _,b,manager=model_batch
    model=ExitModelV5(manager)
    changed=dict(b,elapsed_min=b['elapsed_min']+20,req_idx=b['req_idx']+10)
    assert torch.equal(model.content_x(b),model.content_x(changed))
    assert torch.equal(model.budget_raw(b),model.budget_raw(changed))
    assert not torch.equal(model.budget_components(b)[0],model.budget_components(changed)[0])
    fusion=ExitModelV5(manager,'fusion')
    assert not torch.equal(fusion.content_x(b),fusion.content_x(changed))
    # Dynamic feedback still passes through Ours: only the two requested fields are masked.
    feedback=dict(b,prev1_click_cnt=b['prev1_click_cnt']+1)
    assert not torch.equal(model.content_x(b),model.content_x(feedback))


def test_prior35_exact_transition_and_frozen_budget(model_batch):
    _,b,manager=model_batch
    model=ExitModelV5(manager)
    warm=model.predict(b).detach()
    assert (warm<=model.prior_stay).all()
    snapshot=StageSnapshot.capture(model,StageReport(1,1,{},'student',{}))
    product=model.initialize_stage_two(snapshot)
    assert product.content_net[-1].bias.item()==pytest.approx(3.5)
    assert torch.equal(warm,product.predict(b))
    before=product.budget_components(b)[0].detach().clone()
    opt=torch.optim.AdamW([p for p in product.parameters() if p.requires_grad],lr=.001,weight_decay=0)
    product.calculate_loss(b).backward()
    assert product.content_net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for module in product.budget_modules for p in module.parameters())
    opt.step();assert torch.equal(before,product.budget_components(b)[0])
