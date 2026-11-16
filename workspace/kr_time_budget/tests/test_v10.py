import pytest
import torch
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.time_budget_v9 import PooledTimeBudgetModel
from workspace.kr_time_budget.pipeline.time_budget_v10 import (
    MixtureTimeBudgetModel, mixture_parts, mixture_weibull_survival, pace_feature,
)
from workspace.kr_time_budget.pipeline.time_budget_v8 import joint_parameter_groups


@pytest.fixture
def inputs(model_batch):
    _, b, manager = model_batch
    return dict(b, elapsed_min_raw=b['elapsed_min'].clone(), req_idx_raw=(b['req_idx']+1).long()), manager


def test_mixture_is_survival_ratio_not_fixed_prior_average():
    raw = torch.tensor([[-1., -.3, 2., .2, .3, -.3]]*6, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0., .1, 1., 5., 20., 50.], dtype=torch.float64)
    parts = mixture_parts(raw, t)
    scale, shape, pi = parts['scale'], parts['shape'], parts['prior']
    s0 = torch.exp(-(t[:, None]/scale).pow(shape))
    s1 = torch.exp(-((t[:, None]+1)/scale).pow(shape))
    expected = (pi*s1).sum(1)/(pi*s0).sum(1)
    torch.testing.assert_close(parts['gamma'], expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(parts['posterior'], pi*s0/(pi*s0).sum(1, keepdim=True))
    fixed_prior_ratio = (pi*parts['component_log_gamma'].exp()).sum(1)
    assert (fixed_prior_ratio[1:]-parts['gamma'][1:]).abs().max() > .01
    assert (parts['log_survival'][1:] < parts['log_survival'][:-1]).all()
    parts['gamma'].sum().backward()
    assert torch.isfinite(raw.grad).all() and (raw.grad.abs().sum(0) > 0).all()


def test_equal_components_reduce_to_single_and_extreme_origin_finite():
    raw = torch.tensor([[0., -2., 0., -2., .2, -.2]]*4, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0., 1., 100., 10000.], dtype=torch.float64)
    gamma, s, log_s = mixture_weibull_survival(raw, t)
    p = mixture_parts(raw, t)
    torch.testing.assert_close(gamma, p['component_log_gamma'][:, 0].exp(), rtol=1e-9, atol=1e-11)
    assert torch.isfinite(log_s).all() and torch.isfinite(gamma).all()
    (gamma.sum()+s.sum()).backward()
    assert torch.isfinite(raw.grad).all()


def test_single_reproduces_v9_and_common_initialization(inputs):
    b, manager = inputs
    torch.manual_seed(2026)
    old = PooledTimeBudgetModel(manager, scale_caps='both')
    torch.manual_seed(2026)
    new = MixtureTimeBudgetModel(manager)
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    assert torch.equal(old.predict(b), new.predict(b))
    torch.manual_seed(2026)
    mixed = MixtureTimeBudgetModel(manager, mixture=True)
    for k, v in new.state_dict().items():
        if k not in {'budget_net.2.weight', 'budget_net.2.bias'}:
            assert torch.equal(v, mixed.state_dict()[k]), k
    torch.manual_seed(2026)
    pace = MixtureTimeBudgetModel(manager, mixture=True, add_pace=True)
    torch.testing.assert_close(mixed.predict(b), pace.predict(b), rtol=1e-6, atol=1e-7)
    assert pace.content_net[0].weight[:, -1].abs().sum() == 0


@pytest.mark.parametrize('step_only,add_pace', [(False, False), (True, False), (True, True)])
def test_joint_mixture_loss_gate_and_independent_tower_updates(inputs, step_only, add_pace):
    b, manager = inputs
    model = MixtureTimeBudgetModel(manager, mixture=True, step_only=step_only, add_pace=add_pace)
    prediction, gamma, stay, _, _ = model.components(b)
    assert torch.equal(model.calculate_loss(b), torch.nn.functional.binary_cross_entropy(
        (gamma*stay).clamp(1e-6, 1-1e-6), b['y_continue'].float()))
    params = [model.budget_net[-1].weight, model.content_net[-1].weight,
              model.budget_embedding.emb_modules['user_id'].embedding.weight,
              model.omni_embedding.emb_modules['user_id'].embedding.weight]
    before = [p.detach().clone() for p in params]
    model.calculate_loss(b).backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in params)
    head_grad = model.budget_net[-1].weight.grad
    assert head_grad[10:12].abs().sum() > 0  # screen mixture gates learn on first batch.
    assert (head_grad[:6].abs().sum() == 0) if step_only else (head_grad[:6].abs().sum() > 0)
    if add_pace:
        assert model.content_net[0].weight.grad[:, -1].abs().sum() > 0
    torch.optim.AdamW(joint_parameter_groups(model, .005), weight_decay=0).step()
    assert all(not torch.equal(p, q) for p, q in zip(before, params))
    changed = dict(b, elapsed_min_raw=b['elapsed_min_raw']+10)
    if step_only:
        assert torch.equal(model.budget_components(b)[0], model.budget_components(changed)[0])
    assert torch.equal(model.budget_raw(b), model.budget_raw(changed))


def test_pace_uses_only_current_raw_columns(inputs):
    b, _ = inputs
    expected = torch.log1p(b['elapsed_min_raw']/b['req_idx_raw'])
    assert torch.equal(pace_feature(b), expected)
    changed = dict(b, elapsed_min=b['elapsed_min']+100, req_idx=b['req_idx']+100,
                   y_continue=1-b['y_continue'])
    assert torch.equal(pace_feature(b), pace_feature(changed))
