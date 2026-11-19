"""Typed primary candidates and explicit selected scores survive every output operation."""
from dataclasses import FrozenInstanceError

import pytest
import torch

from rosen.core.prediction import CandidatePool, Prediction


def pool():
    return CandidatePool(
        scores=torch.tensor([[[.7, .4, .2], [.5, .3, .1]], [[.6, .5, .2], [.4, .2, .1]]]),
        orders=torch.tensor([[[1, 2, 0], [2, 1, 0]], [[0, 2, 1], [1, 0, 2]]]),
    )


def test_candidate_selection_is_per_row_and_preserves_gradient_ownership():
    candidates = pool()
    candidates.scores.requires_grad_()
    order, scores = candidates.select(torch.tensor([0, 1]))
    assert (candidates.batch_size, candidates.num_candidates, candidates.slate_size) == (2, 2, 3)
    torch.testing.assert_close(order, torch.tensor([[1, 2, 0], [1, 0, 2]]))
    torch.testing.assert_close(scores, torch.tensor([[.7, .4, .2], [.4, .2, .1]]))
    scores.sum().backward()
    torch.testing.assert_close(candidates.scores.grad, torch.tensor([[[1., 1., 1.], [0., 0., 0.]],
                                                                    [[0., 0., 0.], [1., 1., 1.]]]))


@pytest.mark.parametrize('scores,orders', [
    (torch.ones(2, 3), torch.zeros(2, 1, 3, dtype=torch.long)),
    (torch.ones(2, 1, 3, dtype=torch.long), torch.zeros(2, 1, 3, dtype=torch.long)),
    (torch.ones(2, 1, 3), torch.zeros(2, 1, 3)),
    (torch.ones(2, 1, 3), torch.zeros(2, 3, dtype=torch.long)),
    (torch.ones(2, 2, 3), torch.zeros(2, 1, 3, dtype=torch.long)),
    (torch.ones(2, 0, 3), torch.zeros(2, 0, 3, dtype=torch.long)),
    (torch.ones(2, 1, 0), torch.zeros(2, 1, 0, dtype=torch.long)),
])
def test_candidate_pool_rejects_invalid_shapes_dtypes_and_empty_axes(scores, orders):
    with pytest.raises(ValueError):
        CandidatePool(scores=scores, orders=orders)


@pytest.mark.parametrize('indices', [[0, 1], torch.zeros(2), torch.tensor([[0], [1]]),
                                    torch.tensor([-1, 0]), torch.tensor([0, 2])])
def test_candidate_selection_rejects_invalid_indices(indices):
    with pytest.raises(ValueError):
        pool().select(indices)


def test_output_updates_filtering_and_transfer_preserve_all_typed_fields():
    candidates = pool()
    order, chosen = candidates.select(torch.tensor([0, 1]))
    original = Prediction(scores=torch.ones(2, 3), order=order, chosen_scores=chosen,
                          candidates=candidates, full_scores=torch.ones(2, 5), target='y',
                          extra={'previous': torch.tensor([3., 4.])})
    enriched = original.with_extra({'diag': torch.tensor([1., 2.])}, previous=torch.tensor([5., 6.]))
    assert enriched.candidates is candidates and enriched.chosen_scores is chosen
    assert enriched.scores is original.scores and enriched.full_scores is original.full_scores
    assert set(original.extra) == {'previous'} and original.extra['previous'].tolist() == [3., 4.]
    assert enriched.extra['previous'].tolist() == [5., 6.]
    with pytest.raises(TypeError):
        enriched.extra['bad'] = torch.ones(2)
    with pytest.raises(FrozenInstanceError):
        candidates.scores = torch.ones(2, 2, 3)
    selected = enriched[torch.tensor([False, True])].to('cpu')
    assert selected.batch_size == 1 and selected.target == 'y'
    assert selected.candidates.orders.dtype == selected.order.dtype == torch.long
    assert selected.candidates.scores.device.type == 'cpu'
    torch.testing.assert_close(selected.candidates.scores, candidates.scores[1:])
    torch.testing.assert_close(selected.candidates.orders, candidates.orders[1:])
    torch.testing.assert_close(selected.chosen_scores, chosen[1:])
    torch.testing.assert_close(selected.full_scores, original.full_scores[1:])
    assert selected.extra['diag'].tolist() == [2.]
    empty = enriched[torch.zeros(2, dtype=torch.bool)]
    assert empty.batch_size == empty.candidates.batch_size == 0
    assert empty.resolve_chosen_scores().shape == (0, 3)


def test_candidates_alone_define_batch_size_and_support_empty_row_selection():
    pred = Prediction(candidates=pool())
    assert pred.batch_size == 2 and pred.to('cpu').candidates.batch_size == 2
    assert pred[torch.zeros(2, dtype=torch.bool)].batch_size == 0
    with pytest.raises(ValueError, match='boolean'):
        pred.candidates[torch.ones(2)]
    with pytest.raises(TypeError, match='CandidatePool'):
        Prediction(candidates={'scores': torch.ones(2, 3)})
    with pytest.raises(ValueError, match='batch dimension'):
        Prediction(scores=torch.ones(3, 3), candidates=pool())


def test_chosen_score_resolution_distinguishes_direct_scores_from_candidate_judging():
    scores = torch.tensor([[.1, .9, .3], [.8, .2, .5]])
    candidates = pool()
    order, chosen = candidates.select(torch.tensor([0, 1]))
    pred = Prediction(scores=scores, order=order, chosen_scores=chosen, candidates=candidates)
    assert pred.resolve_chosen_scores() is chosen
    assert not torch.allclose(chosen, scores.gather(-1, order))
    with pytest.raises(ValueError, match='explicit chosen_scores'):
        Prediction(scores=scores, order=order, candidates=candidates).resolve_chosen_scores()
    direct = Prediction(scores=scores, order=order[:, :2])
    assert direct.get_chosen_scores() is None
    torch.testing.assert_close(direct.resolve_chosen_scores(), scores.gather(-1, order[:, :2]))
    with pytest.raises(ValueError, match='both scores and order'):
        Prediction(scores=scores).resolve_chosen_scores()
    with pytest.raises(ValueError, match='slot scores'):
        Prediction(scores=torch.ones(2), order=order).resolve_chosen_scores()


@pytest.mark.parametrize('kwargs', [
    {'chosen_scores': torch.ones(2, 3)},
    {'order': torch.zeros(2, 3, dtype=torch.long), 'chosen_scores': torch.ones(2, 2)},
    {'order': torch.zeros(2, 3, dtype=torch.long), 'chosen_scores': torch.ones(2, 3, dtype=torch.long)},
    {'order': torch.zeros(2, 3, dtype=torch.long), 'chosen_scores': torch.ones(2)},
    {'order': {'click': torch.zeros(2, 3, dtype=torch.long)}, 'chosen_scores': torch.ones(2, 3)},
    {'order': {'click': torch.zeros(2, 3, dtype=torch.long)}, 'chosen_scores': {'watch': torch.ones(2, 3)}},
    {'order': {'click': torch.zeros(2, 3, dtype=torch.long)}, 'chosen_scores': {'click': torch.ones(2, 2)}},
    {'order': torch.zeros(2, 3, dtype=torch.long), 'chosen_scores': {'click': torch.ones(2, 2)}},
])
def test_chosen_scores_require_shape_and_task_alignment(kwargs):
    with pytest.raises(ValueError, match='chosen_scores'):
        Prediction(**kwargs)


@pytest.mark.parametrize('shared_order', [False, True])
def test_multitask_chosen_scores_are_frozen_and_resolve_explicit_or_declared_task(shared_order):
    order = torch.tensor([[1, 0], [0, 1]])
    click, watch = torch.tensor([[.2, .8], [.5, .5]]), torch.ones(2, 2)
    pred = Prediction(order=order if shared_order else {'click': order, 'watch': order},
                      chosen_scores={'click': click, 'watch': watch}, target='watch')
    assert pred.resolve_chosen_scores() is watch and pred.get_chosen_scores('click') is click
    with pytest.raises(TypeError):
        pred.chosen_scores['new'] = click
    with pytest.raises(KeyError, match='available.*click.*watch'):
        pred.get_chosen_scores('missing')
    filtered = pred.with_extra(diag=torch.ones(2))[torch.tensor([True, False])]
    torch.testing.assert_close(filtered.resolve_chosen_scores('click'), click[:1])
