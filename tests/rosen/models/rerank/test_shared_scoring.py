import pytest
import torch

from rosen.models.rerank import MultiGERerankModel, SlateRerankModel
from rosen.models.rerank.components.scoring import EncodedSlate, score_candidate_logits, score_candidate_probs
from tests.rosen.models.rerank.test_reference_models import case


def test_pier_predict_embeds_the_inputs_once(tmp_path, monkeypatch):
    model, batch = case(tmp_path, 'pier')
    inputs, calls = model._inputs, []

    def tracked(inter):
        calls.append(inter)
        return inputs(inter)

    monkeypatch.setattr(model, '_inputs', tracked)
    with torch.no_grad():
        pred = model.eval().predict(batch)
    assert calls == [batch]
    torch.testing.assert_close(pred.scores, model.logits(batch).sigmoid())


def test_multige_inherits_the_slate_input_and_loss_contract():
    assert issubclass(MultiGERerankModel, SlateRerankModel)
    for name in ('_inputs', '_mask', '_view', 'calculate_loss'):
        assert getattr(MultiGERerankModel, name) is getattr(SlateRerankModel, name)


@pytest.mark.parametrize('name', ['mge3', 'nar', 'seq2slate', 'mge_metric'])
def test_skipping_diagnostics_preserves_weights_topk_and_final_ranking(tmp_path, name):
    full, batch = case(tmp_path, name, eval_all_perms=True)
    fast, _ = case(tmp_path, name, eval_all_perms=True, eval_diagnostic_pools=False)
    assert list(full.state_dict()) == list(fast.state_dict())
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, fast.state_dict()[key], rtol=0, atol=0)
    calls = []
    score = fast._score_candidates

    def tracked(*args, **kwargs):
        calls.append(args[5].shape)
        return score(*args, **kwargs)

    fast._score_candidates = tracked
    with torch.no_grad():
        expected = full.eval().predict(batch)
        actual = fast.eval().predict(batch)
    assert len(calls) == 1
    assert 'lastk' in expected.extra and 'all' in expected.extra
    assert not {'lastk', 'randk', 'beamk', 'all'} & actual.extra.keys()
    torch.testing.assert_close(actual.candidates.scores, expected.candidates.scores, rtol=0, atol=0)
    torch.testing.assert_close(actual.candidates.orders, expected.candidates.orders, rtol=0, atol=0)
    torch.testing.assert_close(actual.chosen_scores, expected.chosen_scores, rtol=0, atol=0)
    torch.testing.assert_close(actual.scores, expected.scores, rtol=0, atol=0)
    torch.testing.assert_close(actual.order, expected.order, rtol=0, atol=0)


@pytest.mark.parametrize('candidate_dense', [False, True])
def test_common_probabilities_match_individual_calls_with_chunk_tails_and_gradients(tmp_path, candidate_dense):
    model, batch = case(tmp_path, 'mge3')
    cate, history, dense, pages = model._inputs(batch)
    orders = torch.tensor([[[2, 0, 1], [1, 2, 0], [0, 1, 2]],
                           [[0, 2, 1], [2, 1, 0], [1, 0, 2]]])
    prior = torch.rand(2, 3, 3, 1, requires_grad=True) if candidate_dense else None
    common = dict(scorer=model._score_encoded, chunk_size=2, candidate_dense=prior)
    encoded = EncodedSlate(cate, dense, batch['mask'], model._history_context(history, pages))
    logits = score_candidate_logits(encoded, orders, **common)
    valid = batch['mask'].gather(1, orders.flatten(1)).reshape_as(orders)
    result = score_candidate_probs(encoded, orders, **common)
    torch.testing.assert_close(result, logits.sigmoid() * valid, rtol=0, atol=0)
    # Shared and row-wise order forms have the same semantics for identical rows.
    shared = orders[0]
    common['candidate_dense'] = None
    one = score_candidate_probs(encoded, shared, **common)
    other = score_candidate_probs(encoded, shared[None].expand(2, -1, -1), **common)
    torch.testing.assert_close(one, other, rtol=0, atol=0)
    result.sum().backward()
    assert model.evaluator.output.weight.grad.abs().sum() > 0
    if prior is not None:
        assert torch.isfinite(prior.grad).all() and prior.grad.abs().sum() > 0
