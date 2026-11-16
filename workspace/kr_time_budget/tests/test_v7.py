import pytest
import torch
from rosen.core.train.two_stage import StageSnapshot, StageReport
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.kr_request_schema import (
    BUDGET_PARAMETER_FIELDS, CONTENT_TOWER_FIELDS, SHARED_FIELDS, FEATURES,
)
from workspace.kr_time_budget.pipeline.time_budget_v4 import single_weibull_survival
from workspace.kr_time_budget.pipeline.time_budget_v7 import TimeBudgetModel


@pytest.fixture
def inputs(model_batch):
    _, batch, manager = model_batch
    batch = dict(batch, elapsed_min_raw=batch['elapsed_min'].clone(),
                 req_idx_raw=(batch['req_idx'] + 1).long())
    return batch, manager


def test_native_views_and_independent_tables(inputs):
    batch, manager = inputs
    model = TimeBudgetModel(manager)
    assert set(model.budget_view.field_slices) == set(BUDGET_PARAMETER_FIELDS)
    assert set(model.content_view.field_slices) == set(CONTENT_TOWER_FIELDS)
    assert set(model.fusion_view.field_slices) == set(FEATURES)
    assert set(model.budget_view.field_slices) & set(model.content_view.field_slices) == set(SHARED_FIELDS)
    for view in [model.budget_view, model.content_view, model.fusion_view]:
        vector = view(batch)
        fields = view(batch, split_by='name')
        assert vector.shape[1] == view.embedding_dim
        for name, section in view.field_slices.items():
            assert torch.equal(vector[:, section], fields[name])
    for name, table in model.budget_embedding.emb_modules.items():
        budget = table.embedding.weight
        content = model.omni_embedding.emb_modules[name].embedding.weight
        assert budget.data_ptr() != content.data_ptr()
        assert torch.equal(budget, content)
    params = list(model.named_parameters(remove_duplicate=False))
    assert len(params) == len({id(p) for _, p in params})


def test_routing_and_raw_clock_separation(inputs):
    batch, manager = inputs
    model = TimeBudgetModel(manager)
    changed = dict(batch, elapsed_min=batch['elapsed_min']+100, req_idx=batch['req_idx']+100,
                   user_hist_session_min=batch['user_hist_session_min']+10)
    assert torch.equal(model.content_view(batch), model.content_view(changed))
    assert not torch.equal(model.budget_view(batch), model.budget_view(changed))
    content_changed = dict(batch, slate_dur_mean_min=batch['slate_dur_mean_min']+1)
    assert torch.equal(model.budget_raw(batch), model.budget_raw(content_changed))
    assert not torch.equal(model.content_view(batch), model.content_view(content_changed))
    shared_changed = dict(batch, start_hour=batch['start_hour'].flip(0))
    assert not torch.equal(model.budget_view(batch), model.budget_view(shared_changed))
    assert not torch.equal(model.content_view(batch), model.content_view(shared_changed))
    g0 = model.budget_components(batch)[0]
    network_clock_changed = dict(batch, elapsed_min=batch['elapsed_min']+100, req_idx=batch['req_idx']+100)
    assert torch.equal(g0, model.budget_components(network_clock_changed)[0])
    for name in ['elapsed_min_raw', 'req_idx_raw']:
        changed = dict(batch, **{name: batch[name]+10})
        assert not torch.equal(g0, model.budget_components(changed)[0])
    with pytest.raises(KeyError, match='elapsed_min_raw'):
        model.predict({k: v for k, v in batch.items() if k != 'elapsed_min_raw'})


@pytest.mark.parametrize('clock', ['time', 'dual'])
def test_survival_formula_monotonicity_and_both_head_gradients(inputs, clock):
    batch, manager = inputs
    model = TimeBudgetModel(manager, clock=clock)
    # Repeat one fixed background; both coordinates increase on a grid, including origin.
    fixed = {name: value[:1].expand_as(value).clone() for name, value in batch.items()}
    fixed['elapsed_min_raw'] = torch.linspace(0, 100, len(batch['y_continue']))
    fixed['req_idx_raw'] = torch.arange(1, len(batch['y_continue'])+1)
    raw = model.budget_raw(fixed)
    gamma, survival, log_s = model.budget_components(fixed)
    gt, _, lt = single_weibull_survival(raw[:, :2], fixed['elapsed_min_raw'], torch.ones_like(gamma))
    if clock == 'dual':
        gr, _, lr = single_weibull_survival(raw[:, 2:], fixed['req_idx_raw']-1, torch.ones_like(gamma))
        assert torch.equal(gamma, gt*gr)
        assert torch.equal(log_s, lt+lr)
    else:
        assert torch.equal(gamma, gt) and torch.equal(log_s, lt)
    assert (gamma[1:] <= gamma[:-1]).all() and (log_s[1:] <= log_s[:-1]).all()
    assert torch.isfinite(gamma).all() and ((gamma >= 0) & (gamma <= 1)).all()
    torch.testing.assert_close(survival.double(), log_s.exp(), rtol=1e-6, atol=1e-7)
    model.calculate_loss(batch).backward()
    grad = model.budget_net[-1].weight.grad
    assert torch.isfinite(grad).all() and grad[:2].abs().sum() > 0
    if clock == 'dual':
        assert grad[2:].abs().sum() > 0
    else:
        assert grad[2:].abs().sum() == 0


def test_shared_initialization_all_modes_and_single_dual_identity(inputs):
    _, manager = inputs
    states = []
    for mode in ['ours', 'budget', 'content', 'fusion']:
        for clock in ['time', 'dual']:
            torch.manual_seed(2026)
            states.append(TimeBudgetModel(manager, mode, clock).state_dict())
    for state in states[1:]:
        assert all(torch.equal(value, state[name]) for name, value in states[0].items())


@pytest.mark.parametrize('mode', ['content', 'fusion', 'budget'])
def test_baseline_modes_use_their_own_objectives(inputs, mode):
    batch, manager = inputs
    model = TimeBudgetModel(manager, mode=mode)
    if mode == 'content':
        expected = torch.sigmoid(model.content_net(model.content_view(batch)).squeeze(-1))
        head = model.content_net[-1]
    elif mode == 'fusion':
        expected = torch.sigmoid(model.fusion_net(model.fusion_view(batch)).squeeze(-1))
        head = model.fusion_net[-1]
    else:
        expected = model.budget_components(batch)[0] * model.prior_stay
        head = model.budget_net[-1]
    assert torch.equal(model.predict(batch), expected)
    model.calculate_loss(batch).backward()
    assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('clock', ['time', 'dual'])
def test_exact_stage_transition_budget_freeze_and_second_step_gradient(inputs, clock):
    batch, manager = inputs
    model = TimeBudgetModel(manager, clock=clock, training_strategy='two_stage')
    before = model.predict(batch).detach()
    snapshot = StageSnapshot.capture(model, StageReport(1, 1, {}, 'student', {}))
    product = model.initialize_stage_two(snapshot)
    assert torch.equal(before, product.predict(batch))
    frozen = {k: v.clone() for k, v in product.budget_embedding.state_dict().items()}
    gamma = product.budget_components(batch)[0].detach().clone()
    optimizer = torch.optim.AdamW([p for p in product.parameters() if p.requires_grad], lr=.001, weight_decay=0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        product.calculate_loss(batch).backward()
        assert product.content_net[-1].weight.grad.abs().sum() > 0
        if step == 1:
            assert product.content_net[0].weight.grad.abs().sum() > 0
        assert all(p.grad is None for module in product.budget_modules for p in module.parameters())
        optimizer.step()
    assert torch.equal(gamma, product.budget_components(batch)[0])
    assert all(torch.equal(value, product.budget_embedding.state_dict()[name]) for name, value in frozen.items())
    assert not torch.equal(product.content_view(batch), model.content_view(batch))


@pytest.mark.parametrize('clock', ['time', 'dual'])
def test_joint_product_loss_updates_both_towers_from_first_batch(inputs, clock):
    batch, manager = inputs
    model = TimeBudgetModel(manager, clock=clock)
    assert model.training_strategy == 'joint'
    model.set_stage(2)
    prediction, gamma, stay, _, _ = model.components(batch)
    assert gamma.requires_grad and stay.requires_grad
    assert torch.equal(model.predict(batch), gamma*stay)
    expected_loss = torch.nn.functional.binary_cross_entropy(
        (gamma*stay).clamp(1e-6, 1-1e-6), batch['y_continue'].float())
    assert torch.equal(model.calculate_loss(batch), expected_loss)
    gamma.retain_grad()
    stay.retain_grad()
    expected_loss.backward()
    assert gamma.grad.abs().sum() > 0 and stay.grad.abs().sum() > 0
    watched = [model.budget_net[0].weight, model.budget_net[-1].weight,
               model.content_net[0].weight, model.content_net[-1].weight,
               model.budget_embedding.emb_modules['user_id'].embedding.weight,
               model.omni_embedding.emb_modules['user_id'].embedding.weight]
    for parameter in watched:
        assert parameter.requires_grad and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    before = [p.detach().clone() for p in watched]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, watched))
    with pytest.raises(ValueError, match='ordinary Trainer'):
        model.should_finish_stage_one(StageReport(1, 1, {}, 'student', {}))
