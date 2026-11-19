"""N-candidate/K-output evaluation, including the full-permutation special case."""
from dataclasses import replace
import math

import numpy as np
import polars as pl
import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager, SlateLabel
import rosen.evaluate as evaluation
from rosen.evaluate import (
    BatchResolver, EvalBatch, EvaluatorManager, FieldScorer, SidecarLookupScorer, EvalContext,
    JudgeModelScorer, PredictionScorer, Prediction, SlateValueEvaluator, RankingEvaluator, LabelScorer,
)
from rosen.evaluate.metrics import ListNDCG
from rosen.evaluate.scorer import normalize_orders, gather_by_order


def example():
    labels = torch.tensor([[1., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 1., 0., 0., 1., 0., 0.]])
    scores = torch.tensor([[.9, .2, .7, .1, .3, .8, .4, .5, .6, .0],
                           [.2, .9, .4, .3, .8, .7, .1, .6, .0, .5]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]])
    order = torch.tensor([[5, 2, 0, 6], [7, 1, 8, 4]])
    return EvalBatch(scores=scores, order=order, labels=labels, mask=mask)


def context(b):
    inter = Interaction({"label": b.labels, "mask": b.mask, "prior": b.scores / 2,
                         "ref": b.scores / 3, "request_id": torch.tensor([19, 71])})
    return EvalContext(Prediction(scores=b.scores, order=b.order), inter, "label")


def expected_ndcg(labels, order, mask, k):
    values = []
    for y, indices, valid in zip(labels.numpy(), order.numpy(), mask.numpy().astype(bool)):
        gain = np.where(valid, 2. ** y - 1, 0.)
        chosen = indices[:k]
        dcg = (gain[chosen] / np.log2(np.arange(len(chosen)) + 2)).sum()
        ideal = np.sort(gain)[::-1][:k]
        idcg = (ideal / np.log2(np.arange(len(ideal)) + 2)).sum()
        values.append(dcg / idcg if idcg else 0.)
    return torch.tensor(values, dtype=torch.float64)


def test_prediction_and_fixed_scorers_select_10_to_4_and_batched_candidates():
    b = example()
    assert b.scores.shape == (2, 10) and b.order.shape == (2, 4)
    orders, single = normalize_orders(b.order, 2, 10)
    assert orders.shape == (2, 1, 4) and single
    for scorer, value in ((PredictionScorer(), b.scores), (LabelScorer(), b.labels), (FieldScorer("prior"), b.scores / 2)):
        torch.testing.assert_close(scorer(context(b), b.order), value.gather(1, b.order))
        candidates = torch.stack((b.order, b.order.flip(-1)), 1)
        torch.testing.assert_close(scorer(context(b), candidates), value[:, None].expand(-1, 2, -1).gather(-1, candidates))
    torch.testing.assert_close(gather_by_order(b.mask, b.order), b.mask.gather(1, b.order))


@pytest.mark.parametrize("k", [1, 4, 6, 20])
def test_ndcg_uses_all_10_candidates_for_ideal_gain(k):
    b = example()
    expected = expected_ndcg(b.labels, b.order, b.mask, k)
    actual = ListNDCG.calculate(b.labels, b.order, mask=b.mask, k=k)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
    for evaluator in (RankingEvaluator(f"ndcg@{k}"),):
        evaluator.collect(b)
        assert evaluator.summary()[f"ndcg@{k}"] == pytest.approx(expected.mean().item())


def test_omitting_a_good_candidate_cannot_become_a_perfect_ndcg():
    y = torch.tensor([[3., 2., 1., 0.]])
    order = torch.tensor([[1, 2]])
    actual = ListNDCG.calculate(y, order, k=2).item()
    assert actual == pytest.approx((3 + 1 / math.log2(3)) / (7 + 3 / math.log2(3)))
    assert actual < 1


def test_score_sorting_and_explicit_order_share_one_ndcg_path():
    b = example()
    full = b.scores.masked_fill(b.mask == 0, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    for k in (1, 4, 6, 10):
        implicit = ListNDCG.calculate(b.labels, full, b.mask, k=k)
        explicit = ListNDCG.calculate(b.labels, full[:, :k], mask=b.mask, k=k)
        assert torch.equal(implicit, explicit)
        evaluator = RankingEvaluator(f"ndcg@{k}")
        evaluator.collect(BatchResolver(view="score_only").resolve(context(b), mask=b.mask))
        assert evaluator.summary()[f"ndcg@{k}"] == implicit.mean().item()


def test_full_6_to_6_matches_old_scatter_sort_exactly_for_valid_items():
    rng = torch.Generator().manual_seed(91)
    labels = torch.randint(0, 2, (100, 6), generator=rng).double()
    order = torch.stack([torch.randperm(6, generator=rng) for _ in range(100)])
    ranks = torch.arange(6, 0, -1).double().expand_as(labels)
    old_scores = torch.zeros_like(labels).scatter(1, order, ranks)
    old = ListNDCG.calculate(labels, old_scores.argsort(dim=-1, descending=True), k=6)
    new = ListNDCG.calculate(labels, order, k=6)
    assert torch.equal(old, new)
    batch = EvalBatch(labels=labels, order=order)
    evaluator = RankingEvaluator("ndcg@6")
    evaluator.collect(batch)
    assert evaluator.summary()["ndcg@6"] == old.mean().item()


@pytest.mark.parametrize("reduction", ["sum", "mean", "dcg"])
def test_slate_selection_projects_prediction_value_reference_and_mask_together(reduction):
    b = example()
    evaluator = SlateValueEvaluator("mae", "pcoc", "ncis", reduction=reduction, ncis_cap=2.)
    resolved = BatchResolver(view="slate", ref_scorer=FieldScorer("ref")).resolve(context(b), mask=b.mask)
    evaluator.collect(resolved)
    valid = b.mask.gather(1, b.order).numpy() > 0
    def aggregate(value):
        value = np.where(valid, value.gather(1, b.order).numpy().astype(float), 0.)
        if reduction == "dcg":
            return (value / np.log2(np.arange(4) + 2)).sum(-1)
        sums = value.sum(-1)
        return sums / valid.sum(-1) if reduction == "mean" else sums
    pred, target, ref = aggregate(b.scores), aggregate(b.labels), aggregate(context(b).inter["ref"])
    result = evaluator.summary()
    assert result["mae"] == pytest.approx(np.abs(pred - target).mean())
    assert result["pcoc"] == pytest.approx(pred.sum() / target.sum())
    weights = np.minimum(pred / np.maximum(ref, 1e-6), 2.)
    assert result["ncis"] == pytest.approx(np.dot(weights, target) / weights.sum())
    assert b.mask.shape == (2, 10) and b.scores.shape == (2, 10)  # no mutation


def test_slate_excludes_only_rows_with_no_selected_valid_items():
    b = example()
    mask = b.mask.clone()
    mask[0, b.order[0]] = 0
    evaluator = SlateValueEvaluator("mae")
    evaluator.collect(BatchResolver(view="slate").resolve(context(b), mask=mask))
    selected = b.mask[1].gather(0, b.order[1]) > 0
    expected = abs(b.scores[1, b.order[1]][selected].double().sum() - b.labels[1, b.order[1]][selected].double().sum())
    assert evaluator.summary()["mae"] == expected.item()


@pytest.mark.parametrize("indices", [[0, 0, 1, 2], [-1, 0, 1, 2], [0, 1, 2, 10], [], list(range(11))])
def test_bad_selection_rejected_everywhere(indices):
    b = example()
    order = torch.tensor([indices, indices], dtype=torch.long)
    with pytest.raises(ValueError):
        normalize_orders(order, 2, 10)
    with pytest.raises(ValueError):
        gather_by_order(b.scores, order)
    with pytest.raises(ValueError):
        ListNDCG.calculate(b.labels, order, k=4)
    with pytest.raises(ValueError):
        RankingEvaluator("ndcg@4").collect(replace(b, order=order))
    with pytest.raises(ValueError):
        BatchResolver(view="slate").resolve(replace(context(b), pred=Prediction(scores=b.scores, order=order)), mask=b.mask)


def test_only_pcoc_is_supported_and_zero_denominator_is_nan():
    b = example()
    evaluator = RankingEvaluator("pcoc")
    evaluator.collect(b)
    assert math.isfinite(evaluator.summary()["pcoc"])
    evaluator.clear()
    evaluator.collect(replace(b, labels=torch.zeros_like(b.labels)))
    assert math.isnan(evaluator.summary()["pcoc"])
    with pytest.raises(ValueError, match="Unsupported slot"):
        RankingEvaluator("cb")


def test_lookup_explicitly_rejects_partial_order_in_full_permutation_table(tmp_path):
    path = tmp_path / "table.parquet"
    pl.DataFrame({"request_id": [19, 71], "scores": [[.1] * 3, [.2] * 3],
                  "perm_values": [[float(i) for i in range(6)]] * 2}).write_parquet(path)
    scorer = SidecarLookupScorer(path)
    with pytest.raises(ValueError, match="full permutations"):
        scorer(context(example()), torch.tensor([[1, 2], [0, 1]]))


def test_model_scorer_delegates_partial_orders_to_a_capable_judge():
    class Judge(torch.nn.Module):
        poi_num = 10
        def score_permutations(self, inter, orders):
            return gather_by_order(inter["prior"], orders)
    b = example()
    scorer = JudgeModelScorer(Judge())
    torch.testing.assert_close(scorer(context(b), b.order), context(b).inter["prior"].gather(1, b.order))


def test_manager_handles_order_only_predictions_with_10_candidates(tmp_path):
    b = example()
    manager = EvaluatorManager(SchemaManager([], tmp_path, label_fields=SlateLabel("label", 10, "mask")))
    manager.register("ranking", RankingEvaluator("ndcg@4"))
    manager.register("value", SlateValueEvaluator("mae"), resolver=BatchResolver(view="slate", pred_scorer=FieldScorer("prior")))
    manager.collect(context(b).inter, Prediction(order=b.order))
    assert manager.summary()["ranking"]["ndcg@4"] == pytest.approx(expected_ndcg(b.labels, b.order, b.mask, 4).mean().item())
    assert math.isfinite(manager.summary()["value"]["mae"])


def test_removed_wrappers_are_not_exported():
    removed = {"Evaluator", "OrderEvaluator", "LegacySlateEvaluator", "EvaluatorDecorator", "LogDecorator"}
    assert not removed.intersection(evaluation.__all__)
    assert all(not hasattr(evaluation, name) for name in removed)
