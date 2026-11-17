import torch
from experiments.kuairand.tests.fixtures import model_batch
from experiments.kuairand.pipeline.current_feedback import FEEDBACK_FIELDS
from experiments.kuairand.pipeline.current_feedback_strong import StrongCurrentFeedback


def test_strong_sum_and_joint_budget_gradient(model_batch):
    _, raw, manager = model_batch
    n = len(raw['y_continue'])
    b = dict(raw, req_idx_raw=torch.arange(n)+1, elapsed_min_raw=torch.arange(n).float()+1,
        **{c: torch.full((n,), .5) for c in FEEDBACK_FIELDS})
    m = StrongCurrentFeedback(manager, mode='ours', step_only=True, add_pace=True, requests_per_session=15.)
    terms = m.loss_terms(b)
    assert torch.equal(m.calculate_loss(b), terms['joint_bce']+terms['left_session_nll']+terms['right_bce'])
    grads = torch.autograd.grad(terms['joint_bce'], list(m.budget_net.parameters()), allow_unused=True)
    assert sum(float(g.norm()) for g in grads if g is not None) > 0
    assert m.left_weight == m.right_weight == 1.
