"""The new single-step cells use the same product objective and pace route."""
import pytest
import torch
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.time_budget_v10 import MixtureTimeBudgetModel
from workspace.kr_time_budget.pipeline.time_budget_v8 import joint_parameter_groups


@pytest.mark.parametrize('pace', [False, True])
def test_single_step_cells_clock_independence_and_joint_update(model_batch, pace):
    _, b, manager = model_batch
    b = dict(b, elapsed_min_raw=b['elapsed_min'].clone(), req_idx_raw=(b['req_idx']+1).long())
    model = MixtureTimeBudgetModel(manager, mixture=False, step_only=True, add_pace=pace)
    gamma = model.budget_components(b)[0]
    changed = dict(b, elapsed_min_raw=b['elapsed_min_raw']+10)
    assert torch.equal(gamma, model.budget_components(changed)[0])
    assert torch.equal(model.budget_raw(b), model.budget_raw(changed))
    content = model.network_view(b, model.content_view)
    altered_content = model.network_view(changed, model.content_view)
    assert (not torch.equal(content, altered_content)) == pace
    p, gamma, stay, _, _ = model.components(b)
    assert torch.equal(model.calculate_loss(b), torch.nn.functional.binary_cross_entropy(
        (gamma*stay).clamp(1e-6, 1-1e-6), b['y_continue'].float()))
    watched = [model.budget_net[-1].weight, model.content_net[-1].weight,
        model.budget_embedding.emb_modules['user_id'].embedding.weight,
        model.omni_embedding.emb_modules['user_id'].embedding.weight]
    before = [v.detach().clone() for v in watched]
    model.calculate_loss(b).backward()
    assert model.budget_net[-1].weight.grad[:2].abs().sum() == 0
    assert model.budget_net[-1].weight.grad[2:].abs().sum() > 0
    assert all(torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0 for v in watched)
    if pace:
        assert model.content_net[0].weight.grad[:, -1].abs().sum() > 0
    torch.optim.AdamW(joint_parameter_groups(model, .005), weight_decay=0).step()
    assert all(not torch.equal(old, value) for old, value in zip(before, watched))
