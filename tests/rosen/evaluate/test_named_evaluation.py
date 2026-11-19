"""Numerical and routing checks for the named evaluation contracts."""
from dataclasses import fields, replace
import importlib

import numpy as np
import polars as pl
import pytest
import torch
from sklearn.metrics import roc_auc_score

from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager, SlateLabel
from rosen.evaluate import (
    BaseScorer, BatchResolver, CandidatePool, EvalBatch, EvalContext, EvaluatorManager,
    FieldScorer, LabelScorer, PointwiseEvaluator, Prediction, PredictionScorer,
    RankingEvaluator, RetrievalEvaluator, SidecarLookupScorer, SlateValueEvaluator,
)
from rosen.evaluate.metrics import GAUC


def test_labels_are_required_and_dataclass_replace_preserves_other_fields():
    labels = torch.ones(2, 3)
    scores = torch.rand_like(labels)
    batch = EvalBatch(labels=labels, scores=scores)
    assert batch.labels is labels
    assert 'labels' in {f.name for f in fields(batch)}
    updated = replace(batch, labels=torch.zeros_like(labels))
    assert updated.labels.sum() == 0 and updated.scores is scores
    with pytest.raises(TypeError, match='labels'):
        EvalBatch(scores=scores)
    with pytest.raises(ValueError, match='batch dimension'):
        replace(batch, labels=torch.ones(1, 3))


@pytest.mark.parametrize('module_name,removed', [
    ('rosen.evaluate', ('PointWiseEvaluator', 'SlotEvaluator', 'SlateEvaluator', 'TopKEvaluator',
                        'ScoreContext', 'PredScorer', 'TargetScorer', 'LookupScorer', 'ModelScorer')),
    ('rosen.evaluate.evaluator', ('PointWiseEvaluator', 'SlotEvaluator', 'SlateEvaluator', 'TopKEvaluator')),
    ('rosen.evaluate.scorer', ('ScoreContext', 'PredScorer', 'TargetScorer', 'LookupScorer',
                               'ModelScorer', '_orders', '_permute')),
    ('rosen.evaluate.resolver', ('ScoreContext', 'PredScorer', 'TargetScorer', '_orders', '_permute')),
    ('rosen.models.rerank', ('apply_permutation',)),
    ('rosen.models.rerank.base', ('apply_permutation',)),
])
def test_removed_names_are_not_importable(module_name, removed):
    module = importlib.import_module(module_name)
    for name in removed:
        assert not hasattr(module, name), f'{module_name}.{name}'
        with pytest.raises(ImportError):
            exec(f'from {module_name} import {name}', {})


def test_removed_constructor_arguments_properties_and_view_are_rejected():
    with pytest.raises(TypeError, match='targets'):
        EvalBatch(targets=torch.ones(1))
    assert not hasattr(EvalBatch(labels=torch.ones(1)), 'targets')
    with pytest.raises(TypeError, match='target_scorer'):
        BatchResolver(target_scorer=FieldScorer('reward'))
    resolver = BatchResolver()
    assert not hasattr(resolver, 'target_scorer')
    assert not hasattr(resolver, 'requires_target')
    with pytest.raises(ValueError, match='view must be'):
        BatchResolver(view='candidates')
    from rosen.models.rerank import RerankModel
    assert not hasattr(RerankModel, 'apply_permutation')


def test_ranking_partial_order_metrics_use_the_full_pool_and_stable_score_ties():
    labels = torch.tensor([[1., 0., 1., 1., 0.], [0., 0., 0., 0., 0.]])
    order = torch.tensor([[1, 2], [0, 1]])
    evaluator = RankingEvaluator('ndcg@3', 'recall@3', 'hit@3', 'hr@3', 'precision@3', 'mrr@3', 'map@3')
    evaluator.collect(EvalBatch(labels=labels, order=order))
    values = evaluator.summary()
    expected_ndcg = (1 / np.log2(3)) / (1 + 1 / np.log2(3) + .5) / 2
    assert values == pytest.approx({'ndcg@3': expected_ndcg, 'recall@3': 1/3,
                                   'hit@3': 1., 'hr@3': 1., 'precision@3': .25,
                                   'mrr@3': .5, 'map@3': 1/6})
    # Invalid chosen positions keep their output rank and contribute no gain.
    mask = torch.tensor([[0, 1, 1, 1, 1]])
    evaluator = RankingEvaluator('precision@2', 'mrr@2')
    evaluator.collect(EvalBatch(labels=labels[:1], order=torch.tensor([[0, 2]]), mask=mask))
    assert evaluator.summary() == {'precision@2': 1., 'mrr@2': .5}
    evaluator = RankingEvaluator('ndcg@1')
    evaluator.collect(EvalBatch(labels=labels[:1], scores=torch.ones(1, 5)))
    assert evaluator.summary()['ndcg@1'] == 1.


@pytest.mark.parametrize('metric', ['recall@2', 'hit@2', 'precision@2', 'mrr@2', 'map@2', 'ndcg@2'])
def test_cutoff_metrics_reject_repeated_or_out_of_range_orders(metric):
    for order in (torch.tensor([[0, 0]]), torch.tensor([[0, 4]])):
        with pytest.raises(ValueError, match='permutation'):
            RankingEvaluator(metric).collect(EvalBatch(labels=torch.ones(1, 4), order=order))


def test_vectorized_gauc_matches_per_user_sklearn_with_ties():
    rng = np.random.default_rng(719)
    users = rng.integers(0, 30, 400)
    labels = rng.integers(0, 2, 400)
    scores = rng.integers(0, 6, 400) / 5
    total, weighted = 0, 0.
    for uid in np.unique(users):
        rows = users == uid
        if len(np.unique(labels[rows])) == 2:
            weighted += rows.sum() * roc_auc_score(labels[rows], scores[rows])
            total += rows.sum()
    assert GAUC.calculate(labels, scores, users) == pytest.approx(weighted / total)
    assert GAUC.calculate([], [], []) == 0.


def test_candidate_resolver_transports_labels_masks_and_preserves_typed_scores(tmp_path):
    schema = SchemaManager([], tmp_path, label_fields=SlateLabel('label', 4, 'mask'))
    inter = Interaction({'label': torch.tensor([[1., 0., 1., 0.]]), 'mask': torch.tensor([[1, 0, 1, 1]])})
    scores = torch.tensor([[[.2, .8], [.4, .6]]])
    orders = torch.tensor([[[2, 1], [0, 3]]])
    pred = Prediction(order=torch.tensor([[3, 0]]), candidates=CandidatePool(scores=scores, orders=orders))
    resolver = BatchResolver.for_candidates()
    batch = resolver.resolve(EvalContext(pred, inter, 'label'), mask=inter['mask'])
    assert batch.labels.tolist() == [[[1., 0.], [1., 0.]]]
    assert batch.mask.tolist() == [[[1, 0], [1, 1]]]
    torch.testing.assert_close(batch.scores, scores)
    manager = EvaluatorManager(schema)
    manager.register('candidates', SlateValueEvaluator('cand_mean', 'cand_max'), resolver=resolver)
    manager.collect(inter, pred)
    assert manager.summary()['candidates'] == pytest.approx({'cand_mean': .6, 'cand_max': 1.})


@pytest.mark.parametrize('mode,score,value,reference', [('max', 8., 1., 4.), ('mean', 5., 5.5, 2.5)])
def test_candidate_value_ncis_uses_the_same_predicted_selection(mode, score, value, reference):
    batch = EvalBatch(labels=torch.tensor([[[10.], [1.]]]), scores=torch.tensor([[[2.], [8.]]]),
                      ref_scores=torch.tensor([[[1.], [4.]]]))
    evaluator = SlateValueEvaluator('mae', 'pcoc', 'ncis', candidate_reduce=mode)
    evaluator.collect(batch)
    np.testing.assert_array_equal(np.concatenate(evaluator._prediction_batches), [score])
    np.testing.assert_array_equal(np.concatenate(evaluator._value_batches), [value])
    np.testing.assert_array_equal(np.concatenate(evaluator._reference_batches), [reference])
    assert evaluator.summary() == pytest.approx({'mae': abs(score-value), 'pcoc': score/value, 'ncis': value})


def test_candidates_exclude_invalid_rows_and_check_finite_values_before_accumulating():
    scores = torch.tensor([[[2., 4.], [100., 200.]], [[float('nan'), 0.], [0., 0.]]])
    mask = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [0, 0]]])
    evaluator = SlateValueEvaluator('cand_mean', 'cand_max', 'mae', reduction='mean', candidate_reduce='max')
    evaluator.collect(EvalBatch(labels=torch.ones(2, 2), scores=scores, mask=mask))
    assert evaluator.summary() == {'cand_mean': 3., 'cand_max': 3., 'mae': 2.}
    with pytest.raises(ValueError, match='finite'):
        evaluator.collect(EvalBatch(labels=torch.zeros(1), scores=torch.full((1, 1, 2), float('nan'))))
    assert evaluator.summary()['mae'] == 2.


def test_sparse_sidecar_supports_large_pools_and_missing_order_keys(tmp_path):
    path = tmp_path / 'sidecar.parquet'
    pl.DataFrame({'request_id': [1, 1, 1, 2], 'order': [None, [9, 2], [3, 5], [8, 1]],
                  'reward': [2., 7., 4., 8.]}).write_parquet(path)
    scorer = SidecarLookupScorer(path, score_col='reward', order_col='order', num_candidates=10)
    context = EvalContext(Prediction(order=torch.tensor([[9, 2]])), Interaction({'request_id': torch.tensor([1])}))
    assert scorer(context).tolist() == [2.]
    assert scorer(context, torch.tensor([[[9, 2], [3, 5]]])).tolist() == [[7., 4.]]
    assert scorer(context, torch.tensor([[9, 2]])).tolist() == [7.]
    with pytest.raises(KeyError, match='Sidecar'):
        scorer(context, torch.tensor([[0, 1]]))
    with pytest.raises(ValueError, match='permutation'):
        scorer(context, torch.tensor([[10, 1]]))


def test_sparse_sidecar_rejects_duplicate_keys_and_nonfinite_values(tmp_path):
    path = tmp_path / 'bad.parquet'
    for rewards, message in (([1., 2.], 'Duplicate'), ([float('nan'), 1.], 'finite')):
        pl.DataFrame({'request_id': [1, 1], 'order': [[2, 1], [2, 1]], 'scores': rewards}).write_parquet(path)
        with pytest.raises(ValueError, match=message):
            SidecarLookupScorer(path, order_col='order', num_candidates=3)


def test_pointwise_defaults_ignore_unused_order_and_reference_scorers_declare_target(tmp_path):
    class Reference(BaseScorer):
        def needs_target(self, pred):
            return True

        def score(self, context, order=None):
            return context.inter[context.target]

    manager = EvaluatorManager(SchemaManager([], tmp_path, label_fields='label'))
    manager.register('point', PointwiseEvaluator('auc'))
    manager.collect(Interaction({'label': torch.tensor([0., 1.])}),
                    Prediction(scores=torch.tensor([.1, .9]), order=torch.tensor([[0], [0]])))
    assert manager.registry['point'].resolver.view == 'score_only'
    assert manager.summary()['point']['auc'] == 1.
    assert BatchResolver(label_scorer=FieldScorer('reward'), pred_scorer=FieldScorer('score'),
                         ref_scorer=Reference()).needs_target(Prediction(scores=torch.ones(1)))


def test_catalog_recall_and_mrr_with_history_restoring_the_positive(tmp_path):
    manager = EvaluatorManager(SchemaManager([], tmp_path, label_fields='item'), uid_field='uid')
    manager.register('catalog', RetrievalEvaluator('hr@2', 'recall@2', 'mrr@2', history_dict={1: [0, 1]}))
    manager.collect(Interaction({'item': torch.tensor([1, 2]), 'uid': torch.tensor([1, 2])}),
                    Prediction(full_scores=torch.tensor([[.9, .8, .7], [.9, .1, .8]])))
    assert manager.summary()['catalog'] == {'hr@2': 1., 'recall@2': 1., 'mrr@2': .75}
