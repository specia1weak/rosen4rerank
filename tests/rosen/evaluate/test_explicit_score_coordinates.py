import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.evaluate import BatchResolver, EvalContext, FieldScorer, Prediction, PredictionScorer


def context(batch_size=3):
    labels = torch.arange(10.).expand(batch_size, -1)
    return EvalContext(Prediction(scores=labels / 10), Interaction({'label': labels}), 'label')


@pytest.mark.parametrize('shape', ['2d', '3d'])
@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_fixed_shared_candidates_work_independently_of_batch_size(shape, batch_size):
    orders = torch.tensor([[9, 2, 4], [0, 3, 7]])
    shared = orders if shape == '2d' else orders[None]
    ctx = context(batch_size)
    result = BatchResolver(view='slate', orders=shared).resolve(ctx, mask=torch.ones(batch_size, 10))
    assert result.labels.shape == (batch_size, 2, 3)
    torch.testing.assert_close(result.labels, ctx.inter['label'][:, orders])
    torch.testing.assert_close(result.scores, ctx.pred.scores[:, orders])
    assert result.mask.shape == result.labels.shape


@pytest.mark.parametrize('order', [torch.tensor([[9, 0], [3, 2]]), torch.tensor([[[9, 0], [3, 2]]])])
def test_custom_resolver_shared_orders_use_the_scorer_normalization_contract(order):
    class Shared(BatchResolver):
        def _extract_order(self, ctx):
            return order
    result = Shared(view='slate').resolve(context(3))
    assert result.labels.shape == (3, 2, 2)


@pytest.mark.parametrize('scorer_kind', ['prediction', 'field'])
def test_selected_coordinates_are_explicit_for_all_static_sources_including_k_equals_n(scorer_kind):
    order = torch.tensor([[3, 2, 1, 0]])
    values = torch.tensor([[.7, .1, .8, .2]])
    ctx = EvalContext(Prediction(scores=values, order=order), Interaction({'label': torch.ones(1, 4), 'score': values}), 'label')
    create = (lambda alignment: PredictionScorer(alignment=alignment)) if scorer_kind == 'prediction' else (
        lambda alignment: FieldScorer('score', alignment=alignment))
    selected = BatchResolver(view='slate', pred_scorer=create('selected')).resolve(ctx)
    pool = BatchResolver(view='slate', pred_scorer=create('candidates')).resolve(ctx)
    torch.testing.assert_close(selected.scores, values)
    torch.testing.assert_close(pool.scores, values.gather(1, order))


def test_ranking_rejects_mismatched_unused_scores_even_when_only_ndcg_is_requested():
    ctx = EvalContext(Prediction(scores=torch.ones(1, 4), order=torch.tensor([[3, 2, 1, 0]])),
                      Interaction({'label': torch.ones(1, 10)}), 'label')
    with pytest.raises(ValueError, match='align'):
        BatchResolver().resolve(ctx)


def test_slate_does_not_guess_short_prediction_scores_are_already_selected():
    ctx = EvalContext(Prediction(scores=torch.ones(1, 4), order=torch.tensor([[3, 2, 1, 0]])),
                      Interaction({'label': torch.ones(1, 10)}), 'label')
    with pytest.raises(ValueError, match='selected alignment'):
        BatchResolver(view='slate').resolve(ctx)


def test_fixed_shared_orders_survive_manager_row_filtering():
    from rosen.evaluate import BaseEvaluator, EvaluatorManager

    class Capture(BaseEvaluator):
        def collect(self, batch):
            self.batch = batch
        def summary(self):
            return {}
        def clear(self):
            pass

    evaluator = Capture()
    em = EvaluatorManager()
    shared = torch.tensor([[9, 2], [7, 4]])
    em.register('slice', evaluator, lambda inter: torch.tensor([True, False, True]),
                resolver=BatchResolver(view='slate', orders=shared), target='label')
    ctx = context(3)
    em.collect(ctx.inter, ctx.pred)
    assert evaluator.batch.labels.shape == (2, 2, 2)
    torch.testing.assert_close(evaluator.batch.labels[:, 0], torch.tensor([[9., 2.], [9., 2.]]))
