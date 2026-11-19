"""Selected slate and candidate-pool evaluation never use original OCPM scores."""
from types import SimpleNamespace

import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.evaluate import (
    BaseEvaluator, BatchResolver, CandidatePool, DomainFilter, EvalContext,
    EvaluatorManager, FieldScorer, Prediction, PredictionScorer, SlateValueEvaluator,
)


def prediction():
    pool = CandidatePool(scores=torch.tensor([[[.2, .8], [.4, .6]], [[.1, .9], [.3, .7]]]),
                         orders=torch.tensor([[[2, 1], [0, 3]], [[1, 3], [2, 0]]]))
    order, chosen = pool.select(torch.tensor([1, 0]))
    return Prediction(scores=torch.full((2, 4), .01), order=order, chosen_scores=chosen,
                      candidates=pool, target='label')


@pytest.mark.parametrize('field', ['chosen_scores', 'candidates.scores'])
def test_typed_selected_fields_reject_candidate_alignment(field):
    assert PredictionScorer(field).alignment == 'selected'
    with pytest.raises(ValueError, match='selected-order'):
        PredictionScorer(field, alignment='candidates')


def test_default_slate_and_explicit_chosen_scoring_use_order_aligned_predictions():
    pred = prediction()
    labels, mask = torch.arange(8.).reshape(2, 4), torch.ones(2, 4)
    context = EvalContext(pred, Interaction({'label': labels}), 'label')
    for resolver in (BatchResolver(view='slate'),
                     BatchResolver(view='slate', pred_scorer=PredictionScorer('chosen_scores'))):
        batch = resolver.resolve(context, mask=mask)
        torch.testing.assert_close(batch.scores, pred.chosen_scores)
        torch.testing.assert_close(batch.labels, labels.gather(-1, pred.order))
        assert not torch.allclose(batch.scores, pred.scores.gather(-1, pred.order))
    original = BatchResolver(view='score_only').resolve(context)
    torch.testing.assert_close(original.scores, pred.scores)
    with pytest.raises(ValueError, match='original-order scores'):
        BatchResolver(view='slate', pred_scorer=PredictionScorer('scores')).resolve(context)
    missing = Prediction(scores=pred.scores, order=pred.order, candidates=pred.candidates)
    with pytest.raises(ValueError, match='explicit chosen_scores'):
        BatchResolver(view='slate').resolve(EvalContext(missing, context.inter, 'label'))


def test_chosen_scorer_routes_direct_model_gather_through_prediction_contract():
    pred = Prediction(scores=torch.tensor([[.1, .9, .3]]), order=torch.tensor([[2, 0]]))
    batch = BatchResolver(view='slate', pred_scorer=PredictionScorer('chosen_scores')).resolve(
        EvalContext(pred, Interaction({'label': torch.tensor([[1., 0., 1.]])}), 'label'))
    torch.testing.assert_close(batch.scores, torch.tensor([[.3, .1]]))


def test_candidate_factory_filters_every_typed_field_and_projects_labels_mask_and_users():
    class Capture(BaseEvaluator):
        def collect(self, batch):
            self.batch = batch

        def summary(self):
            return {'rows': self.batch.batch_size}

        def clear(self):
            self.batch = None

    pred = prediction()
    inter = Interaction({'label': torch.tensor([[0., 1., 2., 3.], [4., 5., 6., 7.]]),
                         'mask': torch.tensor([[1, 0, 1, 1], [0, 1, 1, 1]]),
                         'uid': torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
                         'keep': torch.tensor([0, 1])})
    manager = EvaluatorManager(SimpleNamespace(label_fields=('label',)), uid_field='uid')
    capture = Capture()
    manager.register('pool', capture, DomainFilter('keep', 1), resolver=BatchResolver.for_candidates(mask_field='mask'))
    manager.collect(inter, pred)
    batch = capture.batch
    torch.testing.assert_close(batch.scores, pred.candidates.scores[1:])
    assert batch.labels.tolist() == [[[5., 7.], [6., 4.]]]
    assert batch.mask.tolist() == [[[1, 1], [1, 0]]]
    assert batch.users.tolist() == [[[21, 23], [22, 20]]]
    assert batch.order is None and batch.scores.device.type == 'cpu'
    with pytest.raises(ValueError, match='Prediction.candidates'):
        BatchResolver.for_candidates().resolve(EvalContext(Prediction(scores=pred.scores), inter, 'label'))


def test_multitask_chosen_scores_select_tasks_and_ambiguous_tasks_fail_before_collection():
    order = torch.tensor([[1, 0]])
    pred = Prediction(order={'click': order, 'watch': order},
                      chosen_scores={'click': torch.tensor([[.8, .2]]), 'watch': torch.tensor([[.7, .9]])},
                      target='watch')
    inter = Interaction({'click': torch.tensor([[0., 1.]]), 'watch': torch.tensor([[1., 1.]])})
    manager = EvaluatorManager(SimpleNamespace(label_fields=('click', 'watch')))
    manager.register('click', SlateValueEvaluator('mae', reduction='mean'), target='click')
    manager.register('watch', SlateValueEvaluator('mae', reduction='mean'), target='watch')
    manager.collect(inter, pred)
    assert manager.summary() == {'click': {'mae': pytest.approx(0., abs=1e-7)}, 'watch': {'mae': pytest.approx(.2)}}
    no_target = Prediction(order=pred.order, chosen_scores=pred.chosen_scores)
    manager = EvaluatorManager()
    manager.register('ambiguous', SlateValueEvaluator('mae'))
    with pytest.raises(ValueError, match='Mapping predictions require an explicit target'):
        manager.collect(inter, no_target)


@pytest.mark.parametrize('candidates', [False, True])
def test_single_task_chosen_scores_and_candidate_pool_reject_conflicting_targets(candidates):
    pred = prediction()
    # These resolvers do not consume an ordinary scores/order Tensor.
    if candidates:
        pred = Prediction(candidates=pred.candidates, target='watch')
        resolver = BatchResolver.for_candidates(label_scorer=FieldScorer('value'))
    else:
        pred = Prediction(order=pred.order, chosen_scores=pred.chosen_scores, target='watch')
        resolver = BatchResolver(view='score_only', label_scorer=FieldScorer('value'),
                                 pred_scorer=PredictionScorer('chosen_scores'))
    manager = EvaluatorManager()
    manager.register('click', SlateValueEvaluator('mae'), target='click', resolver=resolver)
    with pytest.raises(ValueError, match='conflicts'):
        manager.collect(Interaction({'value': torch.ones(2, 4)}), pred)


def test_diagnostic_extra_sources_remain_independent_of_the_primary_pool():
    pred = prediction().with_extra(diag_scores=torch.tensor([[[.9, .1]], [[.4, .6]]]),
                                   diag_orders=torch.tensor([[[1, 0]], [[3, 2]]]))
    inter = Interaction({'label': torch.arange(8.).reshape(2, 4)})
    result = BatchResolver(pred_scorer=PredictionScorer('extra.diag_scores', alignment='selected'),
                           order_field='extra.diag_orders', view='slate').resolve(EvalContext(pred, inter, 'label'))
    torch.testing.assert_close(result.scores, pred.extra['diag_scores'])
    torch.testing.assert_close(result.labels, torch.tensor([[[1., 0.]], [[7., 6.]]]))
