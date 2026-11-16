import pytest
import torch
from rosen.core.train.two_stage import StageSnapshot, StageReport
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.time_budget_v5 import ExitModelV5
from workspace.kr_time_budget.pipeline.time_budget_v6_probe import ExitModelV6Probe, CONTENT_MASKS, HISTORY_PROXY_FIELDS


@pytest.mark.parametrize('variant', list(CONTENT_MASKS))
def test_probe_exact_initial_state_transition_and_frozen_budget(model_batch, variant):
    _, b, manager = model_batch
    warm = ExitModelV5(manager)
    snapshot = StageSnapshot.capture(warm, StageReport(1, 1, {}, 'student', {}))
    baseline = warm.initialize_stage_two(snapshot)
    probe = ExitModelV6Probe(manager, variant).initialize_stage_two(snapshot)
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, probe.state_dict()[name]), name
    assert torch.equal(warm.predict(b), probe.predict(b))
    gamma = probe.budget_components(b)[0].detach().clone()
    optimizer = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=.001, weight_decay=0)
    probe.calculate_loss(b).backward()
    assert probe.content_net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for module in probe.budget_modules for p in module.parameters())
    optimizer.step()
    assert torch.equal(gamma, probe.budget_components(b)[0])


def test_probe_routing_preserves_previous_feedback_and_removes_only_requested_fields(model_batch):
    _, b, manager = model_batch
    baseline = ExitModelV5(manager)
    control = ExitModelV6Probe(manager, 'control')
    control.load_state_dict(baseline.state_dict())
    assert torch.equal(control.content_x(b), baseline.content_x(b))
    restored = ExitModelV6Probe(manager, 'restore_clocks')
    changed = dict(b, elapsed_min=b['elapsed_min']+10, req_idx=b['req_idx']+5)
    assert not torch.equal(restored.content_x(b), restored.content_x(changed))
    assert torch.equal(control.content_x(b), control.content_x(changed))
    masked = ExitModelV6Probe(manager, 'mask_history')
    changed = dict(b)
    for name in HISTORY_PROXY_FIELDS:
        changed[name] = torch.zeros_like(b[name])
    assert torch.equal(masked.content_x(b), masked.content_x(changed))
    feedback = dict(b, prev1_click_cnt=b['prev1_click_cnt']+1)
    assert not torch.equal(masked.content_x(b), masked.content_x(feedback))
