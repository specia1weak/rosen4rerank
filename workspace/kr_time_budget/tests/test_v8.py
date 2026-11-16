import pytest
import torch
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.time_budget_v7 import TimeBudgetModel
from workspace.kr_time_budget.pipeline.kr_request_schema import SHARED_FIELDS
from workspace.kr_time_budget.pipeline.time_budget_v8 import (
    DualBranchModel, single_weibull_survival, weibull_parameters, joint_parameter_groups,
)


@pytest.fixture
def inputs(model_batch):
    _, b, manager = model_batch
    return dict(b, elapsed_min_raw=b['elapsed_min'].clone(), req_idx_raw=(b['req_idx']+1).long()), manager


def test_legacy_reproduces_v7_state_predictions_loss_and_gradients(inputs):
    batch, manager = inputs
    torch.manual_seed(2026)
    old = TimeBudgetModel(manager)
    torch.manual_seed(2026)
    new = DualBranchModel(manager, scale_mapping='legacy', content_bias=3.5)
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
    assert torch.equal(old.predict(batch), new.predict(batch))
    assert torch.equal(old.calculate_loss(batch), new.calculate_loss(batch))
    old.calculate_loss(batch).backward()
    new.calculate_loss(batch).backward()
    for name, parameter in old.named_parameters():
        other = dict(new.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, other.grad), name


@pytest.mark.parametrize('floor', [1.05, .3])
def test_survival_matches_direct_formula_and_finite_derivatives_at_origin(floor):
    raw = torch.tensor([[0., -2.]]*6, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0., .01, .5, 1., 10., 100.], dtype=torch.float64)
    gamma, survival, log_s = single_weibull_survival(raw, t, torch.ones_like(t), 'log', floor)
    scale, shape = weibull_parameters(raw, 'log', floor)
    torch.testing.assert_close(scale, torch.full_like(scale, torch.exp(torch.tensor(1.6, dtype=torch.float64))))
    direct_log = (t/scale).pow(shape)-((t+1)/scale).pow(shape)
    torch.testing.assert_close(gamma.log(), direct_log, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(survival, log_s.exp())
    assert (log_s[1:] < log_s[:-1]).all()
    assert ((gamma[1:] < gamma[:-1]) if floor > 1 else (gamma[1:] > gamma[:-1])).all()
    (gamma.sum()+survival.sum()).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    zero_raw = raw.detach().clone().requires_grad_(True)
    zero_gamma, _, _ = single_weibull_survival(zero_raw, t, torch.zeros_like(t), 'log', floor)
    assert torch.equal(zero_gamma, torch.ones_like(t))
    zero_gamma.sum().backward()
    assert torch.isfinite(zero_raw.grad).all()


@pytest.mark.parametrize('floor,bounded,bias', [(1.05, False, 3.5), (1.05, False, 3.), (.3, False, 3.), (1.05, True, 3.)])
def test_single_joint_loss_updates_both_towers_and_lr_decay_ownership(inputs, floor, bounded, bias):
    batch, manager = inputs
    model = DualBranchModel(manager, shape_floor=floor, content_bias=bias, bounded_content=bounded)
    expected = model.components(batch)[1]*model.components(batch)[2]
    assert torch.equal(model.predict(batch), expected)
    assert torch.equal(model.calculate_loss(batch), torch.nn.functional.binary_cross_entropy(
        expected.clamp(1e-6, 1-1e-6), batch['y_continue'].float()))
    groups = joint_parameter_groups(model, .005)
    ownership = {id(p): group for group in groups for p in group['params']}
    for module in model.budget_modules:
        assert all(ownership[id(p)]['lr'] == .005 for p in module.parameters())
    assert all(ownership[id(p)]['lr'] == .001 for p in model.content_net.parameters())
    assert ownership[id(model.budget_net[0].weight)]['weight_decay'] == 1e-6
    assert ownership[id(model.budget_net[0].bias)]['weight_decay'] == 0
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            assert ownership[id(module.weight)]['weight_decay'] == 0
    watched = [model.budget_net[-1].weight, model.content_net[-1].weight,
               model.budget_embedding.emb_modules['user_id'].embedding.weight,
               model.omni_embedding.emb_modules['user_id'].embedding.weight]
    before = [p.detach().clone() for p in watched]
    model.calculate_loss(batch).backward()
    assert all(p.requires_grad and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in watched)
    torch.optim.AdamW(groups, weight_decay=0).step()
    assert all(not torch.equal(old, p) for old, p in zip(before, watched))
    assert set(model.budget_view.field_slices).intersection(model.content_view.field_slices) == set(SHARED_FIELDS)


def test_bounded_content_has_explicit_range_and_same_initial_tensors(inputs):
    _, manager = inputs
    torch.manual_seed(2026)
    plain = DualBranchModel(manager)
    torch.manual_seed(2026)
    bounded = DualBranchModel(manager, bounded_content=True)
    assert all(torch.equal(value, bounded.state_dict()[name]) for name, value in plain.state_dict().items())
    logits = torch.tensor([-1e6, 3., 1e6], requires_grad=True)
    stay = bounded.content_stay(logits)
    torch.testing.assert_close(stay, torch.sigmoid(torch.tensor([2.5, 3., 3.5])))
    stay[1].backward()
    assert logits.grad[1] > 0
