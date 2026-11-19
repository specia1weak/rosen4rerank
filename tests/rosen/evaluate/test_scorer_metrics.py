"""Independent formula checks for slot and whole-slate evaluation."""
from dataclasses import replace
import math

import numpy as np
import pytest
from sklearn.metrics import log_loss, roc_auc_score
import torch

from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager, SlateLabel
from rosen.evaluate import (
    BatchResolver, EvalBatch, EvaluatorManager, FieldScorer, PointwiseEvaluator, Prediction, SlateValueEvaluator, RankingEvaluator,
)
from rosen.evaluate.metrics import InSlateAUC, METRIC_REGISTRY, RegressionAUC


def make_batch(scores, labels, mask=None, users=None):
    scores, labels = torch.as_tensor(scores, dtype=torch.float64), torch.as_tensor(labels, dtype=torch.float64)
    return EvalBatch(scores=scores, labels=labels,
                     mask=torch.as_tensor(mask) if mask is not None else None,
                     users=torch.as_tensor(users) if users is not None else None)


def cut(batch, selection):
    count = batch.batch_size
    rows = torch.zeros(count, dtype=torch.bool)
    rows[selection] = True
    return EvalBatch(scores=batch.scores[rows], labels=batch.labels[rows],
                     mask=batch.mask[rows] if batch.mask is not None else None,
                     users=batch.users[rows] if batch.users is not None else None,
                     ref_scores=batch.ref_scores[rows] if batch.ref_scores is not None else None)


def test_slot_formulas_mask_ties_and_batch_partition():
    b = make_batch([[.8, .8, .9], [.2, .9, .5], [.7, .6, .8], [.1, .2, .3]],
                   [[1, 0, 9], [0, 1, 0], [1, 1, 1], [0, 0, 0]],
                   mask=[[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]], users=[10, 10, 20, 30])
    metrics = ("auc", "logloss", "gauc", "pcoc", "ndcg@2", "ndcg@6", "in_slate_auc")
    full, split = RankingEvaluator(*metrics), RankingEvaluator(*metrics)
    full.collect(b)
    split.collect(cut(b, slice(0, 1)))
    split.collect(cut(b, slice(1, 4)))
    result = full.summary()
    assert result == pytest.approx(split.summary())
    valid = b.mask > 0
    y, p = b.labels[valid].numpy(), b.scores[valid].numpy()
    assert result["auc"] == pytest.approx(roc_auc_score(y, p))
    assert result["logloss"] == pytest.approx(log_loss(y, p))
    assert result["pcoc"] == pytest.approx(p.sum() / y.sum())
    assert result["in_slate_auc"] == pytest.approx(.75)  # tie=.5, perfect=1, two illegal rows excluded
    group = b.users[:, None].expand_as(b.scores)[valid].numpy() == 10
    assert result["gauc"] == pytest.approx(roc_auc_score(y[group], p[group]))
    for k in (2, 6):
        expected = []
        for ys, ps, ms in zip(b.labels.numpy(), b.scores.numpy(), b.mask.numpy()):
            kept = np.flatnonzero(ms)
            order = kept[np.argsort(-ps[kept], kind="stable")]
            discount = 1 / np.log2(np.arange(min(k, len(kept))) + 2)
            dcg = ((2 ** ys[order[:k]] - 1) * discount).sum()
            ideal = ((2 ** np.sort(ys[kept])[::-1][:k] - 1) * discount).sum()
            expected.append(dcg / ideal if ideal else 0.)
        assert result[f"ndcg@{k}"] == pytest.approx(np.mean(expected))
    assert full.summary() == result
    full.clear()
    assert full.summary() == {}


def test_in_slate_auc_matches_sklearn_for_random_tied_scores_and_masks():
    rng = np.random.default_rng(123)
    labels = rng.integers(0, 2, (100, 6))
    scores = rng.integers(0, 4, (100, 6)).astype(float)
    mask = rng.integers(0, 2, (100, 6)).astype(bool)
    expected = [roc_auc_score(y[m], p[m]) if np.unique(y[m]).size == 2 else np.nan
                for y, p, m in zip(labels, scores, mask)]
    actual = InSlateAUC.calculate(torch.tensor(labels), torch.tensor(scores), torch.tensor(mask))
    np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True, atol=1e-14)


def brute_r_auc(values, scores):
    comparable = [(i, j) for i in range(len(values)) for j in range(len(values)) if values[i] > values[j]]
    return sum(float(scores[i] > scores[j]) + .5 * float(scores[i] == scores[j]) for i, j in comparable) / len(comparable) if comparable else np.nan


def test_r_auc_matches_independent_pair_count_with_both_types_of_ties():
    rng = np.random.default_rng(71)
    for _ in range(70):
        values = rng.integers(0, 4, 20)
        scores = rng.integers(0, 5, 20)
        assert RegressionAUC.calculate(values, scores) == pytest.approx(brute_r_auc(values, scores), abs=1e-14)
    assert RegressionAUC.calculate([1, 2, 3], [4, 4, 4]) == .5
    assert math.isnan(RegressionAUC.calculate([1, 1, 1], [1, 2, 3]))


def test_r_auc_handles_large_input_without_quadratic_pair_storage():
    values = np.arange(100_000)
    assert RegressionAUC.calculate(values, values) == pytest.approx(1.)
    assert RegressionAUC.calculate(values, -values) == pytest.approx(0.)


@pytest.mark.parametrize("reduction", ["sum", "mean", "dcg"])
def test_slate_aggregation_and_all_metrics_match_global_formulas(reduction):
    b = make_batch([[.1, .8, .7], [.2, .6, .9], [.8, .2, .4], [.3, .2, .1]],
                   [[0, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
                   mask=[[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]])
    ref = torch.tensor([[0., .05, 999.], [.1, .1, .1], [.3, .1, .2], [.1, .2, .1]])
    b = replace(b, ref_scores=ref)
    names = ("r_auc", "slate_auc", "pcoc", "rmse", "mae", "ncis")
    full = SlateValueEvaluator(*names, reduction=reduction, ncis_cap=2.)
    split = SlateValueEvaluator(*names, reduction=reduction, ncis_cap=2.)
    full.collect(b)
    for selection in (slice(0, 1), slice(1, 3), slice(3, 4)):
        split.collect(cut(b, selection))
    result = full.summary()
    assert result == pytest.approx(split.summary())
    mask = b.mask.numpy() > 0
    def reduce(t):
        t = np.where(mask, t, 0.)
        if reduction == "dcg":
            return (t / np.log2(np.arange(3) + 2)).sum(-1)
        total = t.sum(-1)
        return total / mask.sum(-1) if reduction == "mean" else total
    pred, value, reference = reduce(b.scores.numpy()), reduce(b.labels.numpy()), reduce(ref.numpy())
    assert result["r_auc"] == pytest.approx(brute_r_auc(value, pred))
    assert result["slate_auc"] == pytest.approx(roc_auc_score(value > 0, pred))
    assert result["pcoc"] == pytest.approx(pred.sum() / value.sum())
    assert result["mae"] == pytest.approx(np.abs(pred - value).mean())
    assert result["rmse"] == pytest.approx(np.sqrt(((pred - value) ** 2).mean()))
    weight = np.minimum(pred / np.maximum(reference, 1e-6), 2.)
    assert result["ncis"] == pytest.approx((weight * value).sum() / weight.sum())
    full.clear()
    assert full.summary() == {}


def test_global_r_auc_and_slate_auc_include_cross_batch_pairs():
    b = make_batch([.1, .2, .9, .8], [0., 0., 2., 2.])
    evaluator = SlateValueEvaluator("r_auc", "slate_auc")
    evaluator.collect(cut(b, slice(0, 2)))
    evaluator.collect(cut(b, slice(2, 4)))
    assert evaluator.summary() == {"r_auc": 1., "slate_auc": 1.}


def test_reduction_accepts_candidate_masks_and_preserves_scalar_values():
    evaluator = SlateValueEvaluator("rmse", reduction="dcg")
    values = torch.tensor([[[2., float("nan"), 4.], [7., 8., 9.]]])
    mask = torch.tensor([[[1, 0, 1], [0, 1, 1]]])
    expected = torch.tensor([[4., 8. / math.log2(3) + 4.5]], dtype=torch.float64)
    torch.testing.assert_close(evaluator.reduce_slate(values, mask), expected)
    shared_mask = torch.tensor([[1, 0, 1]])
    torch.testing.assert_close(evaluator.reduce_slate(values, shared_mask), torch.tensor([[4., 11.5]], dtype=torch.float64))
    scalar = torch.tensor([7.])
    assert evaluator.reduce_slate(scalar, shared_mask) is scalar


def test_scorers_explicitly_override_batch_predictions_and_targets(tmp_path):
    manager = SchemaManager([], tmp_path, label_fields=SlateLabel("label", 2, "mask"))
    inter = Interaction({"label": torch.zeros(2, 2), "other_target": torch.tensor([[0., 1.], [1., 0.]]),
                         "prior": torch.tensor([[.1, .9], [.8, .2]]), "mask": torch.ones(2, 2)})
    evaluators = EvaluatorManager(manager)
    resolver = BatchResolver(pred_scorer=FieldScorer("prior"), label_scorer=FieldScorer("other_target"))
    evaluators.register("slots", RankingEvaluator("auc", "pcoc"), resolver=resolver)
    evaluators.register("value", SlateValueEvaluator("rmse"), resolver=resolver)
    evaluators.collect(inter, Prediction(scores=torch.zeros(2, 2)))
    assert evaluators.summary()["slots"] == {"auc": 1., "pcoc": pytest.approx(1.)}
    assert evaluators.summary()["value"]["rmse"] == pytest.approx(0., abs=1e-7)


def test_masked_sentinels_empty_rows_and_undefined_metrics():
    b = make_batch([[.5, float("nan")], [float("nan"), 8.]], [[0., -9.], [-9., 9.]], mask=[[1, 0], [0, 0]])
    slots = RankingEvaluator("logloss", "pcoc", "in_slate_auc", "ndcg@6")
    slots.collect(b)
    result = slots.summary()
    assert result["logloss"] == pytest.approx(math.log(2))
    assert result["ndcg@6"] == 0
    assert math.isnan(result["pcoc"]) and math.isnan(result["in_slate_auc"])
    slate = SlateValueEvaluator("pcoc", "r_auc", "slate_auc", "mae")
    slate.collect(b)
    result = slate.summary()
    assert math.isnan(result["pcoc"]) and math.isnan(result["r_auc"]) and math.isnan(result["slate_auc"])
    assert result["mae"] == .5
    for evaluator in (slots, slate):
        evaluator.clear()
        evaluator.collect(replace(b, mask=torch.zeros(2, 2)))
        assert evaluator.summary() == {}


def test_ncis_zero_reference_cap_and_zero_weights():
    metric = METRIC_REGISTRY["ncis"]
    assert metric.calculate([1., 3.], [1., .1], y_ref=[0., 1.], cap=2.) == pytest.approx((2 + .3) / 2.1)
    assert math.isnan(metric.calculate([1., 3.], [0., 0.], y_ref=[1., 1.]))
    with pytest.raises(ValueError, match="nonnegative"):
        metric.calculate([1., 3.], [1., -1.], y_ref=[1., 1.])


def test_configuration_and_input_errors_are_explicit():
    with pytest.raises(ValueError, match="Unsupported slot"):
        RankingEvaluator("reward")
    with pytest.raises(ValueError, match="Unsupported slate"):
        SlateValueEvaluator("ndcg@6")
    with pytest.raises(ValueError, match="ref_scores"):
        SlateValueEvaluator("ncis").collect(make_batch([.1], [1.]))
    with pytest.raises(ValueError, match="finite"):
        SlateValueEvaluator("rmse", ncis_cap=float("nan"))
    with pytest.raises(ValueError, match="matching"):
        RankingEvaluator("auc").collect(make_batch([.1, .2], [0, 1]))
    with pytest.raises(ValueError, match="binary"):
        RankingEvaluator("auc").collect(make_batch([[.1, .2]], [[0, 2]]))
    with pytest.raises(ValueError, match="probabilities"):
        RankingEvaluator("logloss").collect(make_batch([[.1, 2.]], [[0, 1]]))
    with pytest.raises(ValueError, match="finite"):
        SlateValueEvaluator("rmse").collect(make_batch([float("nan")], [0]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_slot_low_precision_inputs_are_normalized_before_cpu_metrics(dtype):
    scores = torch.tensor([[.8, .2, .6], [.1, .9, .7]], dtype=dtype)
    labels = torch.tensor([[1., 0., 1.], [0., 1., 0.]], dtype=dtype)
    half, full = RankingEvaluator("ndcg@3", "in_slate_auc"), RankingEvaluator("ndcg@3", "in_slate_auc")
    half.collect(EvalBatch(labels=labels, scores=scores))
    full.collect(EvalBatch(labels=labels.double(), scores=scores.double()))
    assert half.summary() == full.summary()


def test_in_slate_auc_alone_rejects_nonbinary_valid_targets_but_ignores_padding():
    evaluator = RankingEvaluator("in_slate_auc")
    with pytest.raises(ValueError, match="binary"):
        evaluator.collect(make_batch([[.1, .7, .4]], [[0., 2., 1.]]))
    evaluator.clear()
    evaluator.collect(make_batch([[.1, .7, .4]], [[0., 2., 1.]], mask=[[1, 0, 1]]))
    assert evaluator.summary() == {"in_slate_auc": 1.}


def test_collect_rejects_candidate_axis_while_public_reducer_accepts_it():
    evaluator = SlateValueEvaluator("mae")
    values = torch.ones(2, 3, 4)
    assert evaluator.reduce_slate(values, None).shape == (2, 3)
    with pytest.raises(ValueError, match="aligned"):
        evaluator.collect(EvalBatch(labels=values, scores=values))


@pytest.mark.parametrize("cls", [PointwiseEvaluator, RankingEvaluator])
@pytest.mark.parametrize("users", [None, torch.ones(2, 3), torch.ones(2, 1, 4)])
def test_gauc_validates_users_even_when_every_slot_is_masked(cls, users):
    batch = EvalBatch(labels=torch.zeros(2, 4), scores=torch.ones(2, 4),
                      mask=torch.zeros(2, 4), users=users)
    with pytest.raises(ValueError, match="user IDs|User IDs"):
        cls("gauc").collect(batch)


@pytest.mark.parametrize("cls", [PointwiseEvaluator, RankingEvaluator])
@pytest.mark.parametrize("slot_users", [False, True])
def test_gauc_accepts_valid_empty_batch_then_collects_normal_batch(cls, slot_users):
    users = torch.tensor([7, 8])
    if slot_users:
        users = users[:, None].expand(2, 4)
    evaluator = cls("gauc")
    batch = EvalBatch(labels=torch.tensor([[0., 1., 0., 1.]] * 2),
                      scores=torch.tensor([[.1, .9, .2, .8]] * 2), mask=torch.zeros(2, 4), users=users)
    evaluator.collect(batch)
    assert evaluator.summary() == {}
    evaluator.collect(replace(batch, mask=torch.ones(2, 4)))
    assert evaluator.summary() == {"gauc": 1.}
