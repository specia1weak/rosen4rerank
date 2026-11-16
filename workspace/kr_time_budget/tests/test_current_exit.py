"""当前主线的语义与历史等价性；不以shape/非零梯度代替公式检查。"""
import numpy as np
import pytest
import torch
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.exit_model import KRExitModel, mixture_parts
from workspace.kr_time_budget.pipeline.exit_training import EXPERIMENTS, joint_parameter_groups
from workspace.kr_time_budget.pipeline.kr_request_schema import (
    BUDGET_PARAMETER_FIELDS, CONTENT_TOWER_FIELDS, CONTENT_EXCLUSIVE_FIELDS,
    BUDGET_PRIOR_FIELDS, SHARED_FIELDS,
)
from workspace.kr_time_budget.pipeline.time_budget_v10 import MixtureTimeBudgetModel
from workspace.kr_time_budget.pipeline.time_budget_v8 import joint_parameter_groups as old_groups


@pytest.fixture
def inputs(model_batch):
    _, batch, manager = model_batch
    return dict(batch, elapsed_min_raw=batch['elapsed_min'].clone(),
                req_idx_raw=(batch['req_idx']+1).long()), manager


@pytest.mark.parametrize('name', list(EXPERIMENTS))
def test_current_model_exact_initial_forward_gradient_and_update(inputs, name):
    batch, manager = inputs
    options = EXPERIMENTS[name]
    torch.manual_seed(2026)
    old = MixtureTimeBudgetModel(manager, **options)
    torch.manual_seed(2026)
    current = KRExitModel(manager, **options)
    assert old.state_dict().keys() == current.state_dict().keys()
    assert all(torch.equal(value, current.state_dict()[key]) for key, value in old.state_dict().items())
    assert torch.equal(old.predict(batch), current.predict(batch))
    assert torch.equal(old.calculate_loss(batch), current.calculate_loss(batch))
    old.calculate_loss(batch).backward()
    current.calculate_loss(batch).backward()
    for (key, a), (other, b) in zip(old.named_parameters(), current.named_parameters()):
        assert key == other
        assert (a.grad is None) == (b.grad is None), key
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad), key
    a_groups, b_groups = old_groups(old, .005), joint_parameter_groups(current)
    for a, b in zip(a_groups, b_groups):
        assert {k: v for k, v in a.items() if k != 'params'} == {k: v for k, v in b.items() if k != 'params'}
    for model, groups in [(old, a_groups), (current, b_groups)]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        torch.optim.AdamW(groups, weight_decay=0).step()
    assert all(torch.equal(value, current.state_dict()[key]) for key, value in old.state_dict().items())


@pytest.mark.parametrize('pace', [False, True])
def test_routes_and_independent_embedding_ownership(inputs, pace):
    batch, manager = inputs
    model = KRExitModel(manager, mixture=True, step_only=True, add_pace=pace)
    assert set(model.budget_view.field_slices) == set(BUDGET_PARAMETER_FIELDS)
    assert set(model.content_view.field_slices) == set(CONTENT_TOWER_FIELDS)
    assert set(model.budget_view.field_slices) & set(model.content_view.field_slices) == set(SHARED_FIELDS)
    for field in SHARED_FIELDS:
        assert model.budget_embedding.emb_modules[field] is not model.omni_embedding.emb_modules[field]
    content_changed = dict(batch, **{field: torch.zeros_like(batch[field]) for field in CONTENT_EXCLUSIVE_FIELDS})
    assert torch.equal(model.budget_raw(batch), model.budget_raw(content_changed))
    prior_changed = dict(batch, **{field: batch[field]+100 for field in BUDGET_PRIOR_FIELDS})
    assert torch.equal(model.network_view(batch, model.content_view), model.network_view(prior_changed, model.content_view))
    labels_changed = dict(batch, y_continue=1-batch['y_continue'])
    assert torch.equal(model.predict(batch), model.predict(labels_changed))
    model.calculate_loss(batch).backward()
    # step_only保留兼容head，分钟行梯度为0；两塔活跃用户表都有梯度。
    assert model.budget_net[-1].weight.grad[:6].abs().sum() == 0
    assert model.budget_net[-1].weight.grad[6:].abs().sum() > 0
    assert model.content_net[-1].weight.grad.abs().sum() > 0
    assert model.budget_embedding.emb_modules['user_id'].embedding.weight.grad.abs().sum() > 0
    assert model.omni_embedding.emb_modules['user_id'].embedding.weight.grad.abs().sum() > 0


def test_mixture_matches_independent_numpy_survival_ratio():
    raw = np.array([[0.1, -2., 1.2, 1., .4, -.8],
                    [-4., .5, 6., -1., -2., 3.],
                    [1., -4., 3., -.5, .2, .2]], dtype=np.float64)
    t = np.array([0., 7., 1000.])
    parts = mixture_parts(torch.tensor(raw), torch.tensor(t))
    scale = np.exp(np.clip(raw[:, [0, 2]], -2., 4.5)+1.6)
    shape = np.minimum(.3+np.logaddexp(0., raw[:, [1, 3]]), 5.)
    logits = raw[:, 4:6]
    log_pi = logits-np.logaddexp.reduce(logits, axis=1)[:, None]
    log_s0 = np.logaddexp.reduce(log_pi-(t[:, None]/scale)**shape, axis=1)
    log_s1 = np.logaddexp.reduce(log_pi-((t[:, None]+1)/scale)**shape, axis=1)
    np.testing.assert_allclose(parts['log_survival'].detach(), log_s0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(parts['gamma'].detach(), np.exp(log_s1-log_s0), atol=1e-12, rtol=1e-12)
    posterior = np.exp(log_pi-(t[:, None]/scale)**shape-log_s0[:, None])
    np.testing.assert_allclose(parts['posterior'].detach(), posterior, atol=1e-12, rtol=1e-12)
    assert not np.allclose(posterior[1:], np.exp(log_pi)[1:])


def test_mixture_gradcheck_at_origin_and_positive_step():
    raw = torch.tensor([[.1, -1., 1., .5, .2, -.3],
                        [.5, .2, 2., -2., -.4, .1]], dtype=torch.double, requires_grad=True)
    t = torch.tensor([0., 17.], dtype=torch.double)
    def output(value):
        parts = mixture_parts(value, t)
        return torch.cat([parts['gamma'], parts['log_survival']])
    assert torch.autograd.gradcheck(output, (raw,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_decreasing_survival_does_not_require_decreasing_gamma():
    # 两个同分量的shape<1例子：S下降但gamma上升，这在当前允许的模型族中合法。
    raw = torch.tensor([[1., -2., 1., -2., 0., 0.]], dtype=torch.double).repeat(30, 1)
    result = mixture_parts(raw, torch.arange(30, dtype=torch.double))
    assert (torch.diff(result['log_survival']) < 0).all()
    assert (torch.diff(result['gamma']) > 0).all()
