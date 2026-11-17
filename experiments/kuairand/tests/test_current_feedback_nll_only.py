import pytest
import torch
from experiments.kuairand.tests.fixtures import model_batch
from experiments.kuairand.pipeline.current_feedback import FEEDBACK_FIELDS, CurrentFeedbackExitModel
from experiments.kuairand.pipeline.current_feedback_nll_only import CurrentFeedbackNLLOnly


@pytest.mark.parametrize('mixture', [False, True])
def test_detach_and_nll_gradient_routes_and_actual_weight(model_batch, mixture):
    _, raw, manager = model_batch
    n = len(raw['y_continue'])
    b = dict(raw, req_idx_raw=torch.arange(n)+1, elapsed_min_raw=torch.arange(n).float()+1,
        **{c: torch.full((n,), .5) for c in FEEDBACK_FIELDS})
    b['y_continue'] = torch.zeros(n)
    models = []
    for weight in [.1, 1.]:
        torch.manual_seed(2026)
        models.append(CurrentFeedbackNLLOnly(manager, mode='ours', mixture=mixture, step_only=True,
            add_pace=True, nll_weight=weight, requests_per_session=15.))
    a, z = models
    assert all(torch.equal(v, z.state_dict()[k]) for k, v in a.state_dict().items())
    torch.manual_seed(2026)
    base = CurrentFeedbackExitModel(manager, mode='ours', mixture=mixture, step_only=True,
        add_pace=True, left_weight=.1, right_weight=.1, requests_per_session=15.)
    assert torch.equal(a.predict(b), base.predict(b))
    budget = list(a.budget_embedding.parameters())+list(a.budget_net.parameters())
    content = list(a.omni_embedding.parameters())+list(a.content_net.parameters())+list(a.feedback_projection.parameters())
    terms = a.loss_terms(b)
    assert all(g is None for g in torch.autograd.grad(terms['joint_bce'], budget, allow_unused=True, retain_graph=True))
    assert all(g is None for g in torch.autograd.grad(terms['left_session_nll'], content, allow_unused=True, retain_graph=True))
    left_grad = torch.autograd.grad(terms['left_session_nll'], budget, allow_unused=True)
    assert sum(float(g.norm()) for g in left_grad if g is not None) > 0
    a.calculate_loss(b).backward(); z.calculate_loss(b).backward()
    for p, q in zip(a.budget_net.parameters(), z.budget_net.parameters()):
        assert torch.allclose(q.grad, 10*p.grad, atol=1e-6, rtol=1e-5)
    for p, q in zip(a.content_net.parameters(), z.content_net.parameters()):
        assert torch.equal(p.grad, q.grad)
    assert a.feedback_projection.weight.grad.norm() > 0
    assert a.last_loss_terms['right_main_ratio'] <= .1+1e-12
    assert z.last_loss_terms['effective_left_weight'] == 1.
