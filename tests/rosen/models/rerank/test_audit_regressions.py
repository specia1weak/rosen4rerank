"""Semantic checks for the review findings, including cold-start gradient ownership."""
import pytest
import torch
from torch.nn import functional as F

from rosen.core.interaction import Interaction
from rosen.models.rerank import select_by_order
from tests.rosen.models.rerank.test_reference_models import case


@pytest.mark.parametrize('name', ['mge_metric', 'mge_cosine_metric'])
def test_metric_main_loss_trains_current_embedding_without_history(tmp_path, name, monkeypatch):
    model, batch = case(tmp_path, name, candidate_chunk_size=10)
    batch['pages'].zero_()
    captured, calls = {}, []
    inputs = model._inputs
    generate = model._generate

    def tracked_inputs(inter):
        value = inputs(inter)
        value[0].retain_grad()
        captured['cate'] = value[0]
        calls.append('inputs')
        return value

    def tracked_generate(*args, **kwargs):
        calls.append((kwargs['teacher_forcing'], torch.is_grad_enabled()))
        return generate(*args, **kwargs)

    judge_calls = []
    handle = model.evaluator.register_forward_hook(lambda *args: judge_calls.append(True))
    monkeypatch.setattr(model, '_inputs', tracked_inputs)
    monkeypatch.setattr(model, '_generate', tracked_generate)
    terms = model.loss_components(batch)
    handle.remove()
    terms['evaluator'].backward()
    assert calls == ['inputs', (True, True), (False, False)]
    assert len(judge_calls) == 1  # No discarded original-slate OCPM pass.
    assert captured['cate'].grad.abs().sum() > 0
    for table in model.omni_embedding.emb_modules.values():
        assert table.embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.generative_generator.parameters())


@pytest.mark.parametrize('name', ['mge3', 'mge_cosine', 'nar', 'seq2slate', 'mge_metric', 'mge_cosine_metric'])
def test_generator_objectives_are_batch_means_including_tail_batches(tmp_path, name):
    model, batch = case(tmp_path, name)
    single = batch[:1]
    duplicate = Interaction({key: single[key].repeat(3, *([1] * (single[key].ndim-1))) for key in single})
    small = model.loss_components(single)
    large = model.loss_components(duplicate)
    for key in small:
        torch.testing.assert_close(small[key], large[key], rtol=1e-5, atol=1e-6)
    gradient_small = torch.autograd.grad(small['gen.generative'], model.pointer_head.output.weight, retain_graph=True)[0]
    gradient_large = torch.autograd.grad(large['gen.generative'], model.pointer_head.output.weight)[0]
    torch.testing.assert_close(gradient_small, gradient_large, rtol=1e-4, atol=1e-6)


def test_pointwise_logits_labels_and_orders_share_original_slot_coordinates(tmp_path, monkeypatch):
    model, batch = case(tmp_path, 'mge3')
    with torch.no_grad():
        model.pointwise_generator.linear.weight.fill_(2.)
        model.pointwise_generator.linear.bias.fill_(-.5)
    # Deliberately make retrieved source order nonidentity; it must not relabel pointwise items.
    monkeypatch.setattr(model.retriever, 'top_k', lambda distance, k: torch.full((2, k), 5, dtype=torch.long))
    inputs = model._inputs(batch)
    orders, aux = model._generate(*inputs, teacher_forcing=False)
    expected_logits = batch['prior'] * 2 - .5
    expected_order = expected_logits.argsort(dim=-1, descending=True, stable=True)
    torch.testing.assert_close(aux['pointwise_logits'], expected_logits)
    torch.testing.assert_close(orders[:, 0], expected_order)
    torch.testing.assert_close(aux['candidate_dense'][:, 0, :, 0], expected_logits.sigmoid().gather(1, expected_order))
    expected_loss = F.binary_cross_entropy_with_logits(expected_logits, batch['label'],
                                                       weight=batch['mask'], reduction='sum') / batch['mask'].count_nonzero()
    torch.testing.assert_close(model.loss_components(batch)['gen.pointwise'], expected_loss)


def test_exact_lc_is_diagnostic_and_not_a_constant_optimization_offset(tmp_path):
    model, batch = case(tmp_path, 'lc_mge', submodularity_mode='lc')
    terms = model.loss_components(batch)
    _, aux = model._generate(*model._inputs(batch), labels=batch['label'], label_mask=batch['mask'], teacher_forcing=True)
    assert 'lc_exact' in model.loss_diagnostics
    assert not model.loss_diagnostics['lc_exact'].requires_grad
    torch.testing.assert_close(terms['gen.submodularity'], aux['submodularity'].loss)
    assert 'gen.coverage' not in terms
    assert terms['gen.submodularity'] >= 0


def test_soft_coverage_objective_contains_only_differentiable_pairs(tmp_path):
    from rosen.models.rerank.components.generators import lc_soft_score
    probs = torch.rand(2, 3, 3, requires_grad=True)
    orders = torch.tensor([[[0, 1, 2], [2, 1, 0]]] * 2)
    value = lc_soft_score(probs, orders)
    # Explicit soft/reference-only comparison; no reference/reference constant.
    steps = torch.arange(3.)
    expected_positions = (probs * steps[None, :, None]).sum(1)
    soft = torch.sigmoid(expected_positions[:, None, :] - expected_positions[:, :, None])
    inverse = orders.argsort(-1)
    ref = (inverse[..., None, :] > inverse[..., :, None]).float()
    expected = ((soft[:, None] - ref).square() * (~torch.eye(3, dtype=torch.bool))).sum((-1, -2, -3))
    torch.testing.assert_close(value, expected)
    value.sum().backward()
    assert probs.grad.abs().sum() > 0


@pytest.mark.parametrize('name', ['pier', 'prs', 'pier_submodular', 'mge3', 'mge_metric'])
def test_judge_and_prediction_agree_on_masked_candidate_scores(tmp_path, name):
    model, batch = case(tmp_path, name, candidate_dense_mode='prior') if name.startswith('mge') else case(tmp_path, name)
    model.eval()
    with torch.no_grad():
        pred = model.predict(batch)
        judged = model.score_permutations(batch, pred.candidates.orders)
    torch.testing.assert_close(judged, pred.candidates.scores)
    selected = (pred.candidates.orders == pred.order[:, None]).all(-1).long().argmax(-1)
    torch.testing.assert_close(pred.chosen_scores, pred.candidates.scores[torch.arange(2), selected])
