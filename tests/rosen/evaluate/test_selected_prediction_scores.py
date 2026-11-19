import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.evaluate import (
    BatchResolver, EvalContext, FieldScorer, Prediction, PredictionScorer, SlateValueEvaluator,
)


@pytest.mark.parametrize('mapping', [False, True])
@pytest.mark.parametrize('order', [torch.tensor([[3, 2, 1, 0]]), torch.tensor([[9, 7, 4, 0]])])
def test_selected_scores_are_not_reordered_again_even_with_only_small_indices(mapping, order):
    labels = torch.arange(10.).reshape(1, 10)
    scores = torch.tensor([[.8, .2, .9, .1]])
    pred = Prediction(chosen_scores={'click': scores} if mapping else scores,
                      order={'click': order} if mapping else order)
    context = EvalContext(pred, Interaction({'click': labels}), 'click')
    mask = torch.tensor([[1, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    resolved = BatchResolver(view='slate', pred_scorer=PredictionScorer('chosen_scores')).resolve(context, mask=mask)
    torch.testing.assert_close(resolved.scores, scores)
    torch.testing.assert_close(resolved.labels, labels.gather(1, order))
    torch.testing.assert_close(resolved.mask, mask.gather(1, order))
    evaluator = SlateValueEvaluator('mae', reduction='dcg')
    evaluator.collect(resolved)
    discounts = 1 / torch.log2(torch.arange(4).double() + 2)
    expected = abs(((scores - resolved.labels) * resolved.mask * discounts).sum().item())
    assert evaluator.summary()['mae'] == pytest.approx(expected)


def test_direct_input_scores_still_reorder_in_partial_and_full_views():
    for order in (torch.tensor([[3, 2, 1, 0]]), torch.tensor([[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])):
        values = torch.arange(10.).reshape(1, 10)
        result = BatchResolver(view='slate').resolve(EvalContext(
            Prediction(scores=values, order=order), Interaction({'label': values}), 'label'))
        torch.testing.assert_close(result.scores, values.gather(1, order))


def test_static_field_scores_remain_in_candidate_coordinates():
    values = torch.arange(10.).reshape(1, 10)
    order = torch.tensor([[3, 2, 1, 0]])
    context = EvalContext(Prediction(scores=torch.ones(1, 4), order=order),
                          Interaction({'label': values, 'prior': values / 10}), 'label')
    result = BatchResolver(view='slate', pred_scorer=FieldScorer('prior')).resolve(context)
    torch.testing.assert_close(result.scores, values.gather(1, order) / 10)


def test_prediction_scorer_runs_once_and_selected_width_does_not_supply_fake_pool_width():
    class CountingPrediction(PredictionScorer):
        calls = 0

        def score(self, context, order=None):
            self.calls += 1
            return super().score(context, order=order)

    scorer = CountingPrediction('chosen_scores')
    order = torch.tensor([[3, 2, 1, 0]])
    context = EvalContext(Prediction(chosen_scores=torch.ones(1, 4), order=order),
                          Interaction({'label': torch.arange(10.).reshape(1, 10)}), 'label')
    result = BatchResolver(view='slate', pred_scorer=scorer).resolve(context)
    assert scorer.calls == 1 and result.scores.shape == (1, 4)


def test_local_scores_cannot_be_broadcast_as_unrelated_candidate_slate_scores():
    context = EvalContext(Prediction(chosen_scores=torch.ones(1, 4), order=torch.tensor([[3, 2, 1, 0]]), extra={
        'orders': torch.tensor([[[0, 1, 2, 3], [9, 8, 7, 6]]])}),
        Interaction({'label': torch.ones(1, 10)}), 'label')
    with pytest.raises(ValueError, match='Selected scores'):
        BatchResolver(view='slate', order_field='extra.orders', pred_scorer=PredictionScorer('chosen_scores')).resolve(context)


@pytest.mark.parametrize('field', ['scores', 'order'])
def test_prediction_mapping_errors_report_available_tasks_and_allow_declared_default(field):
    value = torch.ones(2, 4) if field == 'scores' else torch.arange(4).expand(2, -1)
    pred = Prediction(**{field: {'click': value, 'like': value}})
    getter = getattr(pred, 'get_' + field)
    with pytest.raises(ValueError, match='click.*like.*target'):
        getter()
    with pytest.raises(KeyError, match='available.*click.*like'):
        getter('missing')
    declared = Prediction(**{field: {'click': value}}, target='click')
    assert getattr(declared, 'get_' + field)() is value
    assert EvalContext(declared).prediction_slot(field) is value
