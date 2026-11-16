import pytest
import torch
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.time_budget_v8 import DualBranchModel, joint_parameter_groups
from workspace.kr_time_budget.pipeline.time_budget_v9 import (
    PooledTimeBudgetModel, weibull_parameters, single_weibull_survival,
)


@pytest.fixture
def inputs(model_batch):
    _, b, manager = model_batch
    return dict(b, elapsed_min_raw=b['elapsed_min'].clone(), req_idx_raw=(b['req_idx']+1).long()), manager


def test_capped_v8_exact_state_prediction_loss_and_gradients(inputs):
    b, manager = inputs
    torch.manual_seed(2026)
    old = DualBranchModel(manager, shape_floor=.3)
    torch.manual_seed(2026)
    new = PooledTimeBudgetModel(manager, scale_caps='both')
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    assert torch.equal(old.predict(b), new.predict(b))
    old.calculate_loss(b).backward()
    new.calculate_loss(b).backward()
    for name, p in old.named_parameters():
        other = dict(new.named_parameters())[name]
        assert (p.grad is None) == (other.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, other.grad)


def test_uncapping_removes_both_plateaus_without_changing_inside_values():
    raw = torch.tensor([[-3., -1.], [0., -1.], [4., -1.], [6., -1.]], dtype=torch.float64, requires_grad=True)
    capped = weibull_parameters(raw, True)[0]
    uncapped = weibull_parameters(raw, False)[0]
    assert torch.equal(capped[1:3], uncapped[1:3])
    assert uncapped[0] < capped[0] and uncapped[-1] > capped[-1]
    dc = torch.autograd.grad(capped.sum(), raw, retain_graph=True)[0][:, 0]
    du = torch.autograd.grad(uncapped.sum(), raw)[0][:, 0]
    assert dc[0] == 0 and dc[-1] == 0 and (du > 0).all()


@pytest.mark.parametrize('caps', ['both', 'step', 'none'])
def test_same_initialization_single_product_and_both_tower_updates(inputs, caps):
    b, manager = inputs
    torch.manual_seed(2026)
    control = PooledTimeBudgetModel(manager, scale_caps='both')
    torch.manual_seed(2026)
    model = PooledTimeBudgetModel(manager, scale_caps=caps)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in control.state_dict().items())
    assert model.is_capped('time') == (caps == 'both')
    assert model.is_capped('step') == (caps != 'none')
    p, gamma, stay, _, _ = model.components(b)
    assert torch.equal(p, gamma*stay)
    assert torch.equal(model.calculate_loss(b), torch.nn.functional.binary_cross_entropy(
        p.clamp(1e-6, 1-1e-6), b['y_continue'].float()))
    watched = [model.budget_net[-1].weight, model.content_net[-1].weight,
               model.budget_embedding.emb_modules['user_id'].embedding.weight,
               model.omni_embedding.emb_modules['user_id'].embedding.weight]
    before = [x.detach().clone() for x in watched]
    model.calculate_loss(b).backward()
    assert all(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in watched)
    torch.optim.AdamW(joint_parameter_groups(model, .005), weight_decay=0).step()
    assert all(not torch.equal(old, new) for old, new in zip(before, watched))


def test_uncapped_survival_origin_extreme_clock_and_gradient():
    raw = torch.tensor([[6., -2.]]*5, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0., 1., 100., 879., 10000.], dtype=torch.float64)
    gamma, s, log_s = single_weibull_survival(raw, t, torch.ones_like(t), False)
    scale, shape = weibull_parameters(raw, False)
    direct = torch.exp((t/scale).pow(shape)-((t+1)/scale).pow(shape))
    torch.testing.assert_close(gamma, direct, rtol=1e-10, atol=1e-10)
    assert (log_s[1:] < log_s[:-1]).all()
    assert (gamma[1:] > gamma[:-1]).all()  # DFR allows rising conditional survival.
    (gamma.sum()+s.sum()).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[:, 0].abs().sum() > 0
