"""The only raw-object boundary is Resolver -> pure EvalBatch -> Evaluator."""
from dataclasses import fields
import inspect
import math

import polars as pl
import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager, SlateLabel
from rosen.evaluate import (
    BaseEvaluator, BaseScorer, BatchResolver, DomainFilter, EvalBatch, EvaluatorManager,
    FieldScorer, SidecarLookupScorer, JudgeModelScorer, PointwiseEvaluator, Prediction, EvalContext,
    SlateValueEvaluator, RankingEvaluator, RetrievalEvaluator,
)
from rosen.evaluate.metrics import ListNDCG
from tests.rosen.models.rerank.test_reference_models import case


class Capture(BaseEvaluator):
    def collect(self, batch):
        assert isinstance(batch, EvalBatch)
        assert not hasattr(batch, "inter") and not hasattr(batch, "pred")
        self.batch = batch

    def summary(self):
        return {"rows": self.batch.batch_size}

    def clear(self):
        self.batch = None


def test_resolved_batch_fields_are_only_tensors_and_no_flags_or_scorers():
    assert {f.name for f in fields(EvalBatch)} == {
        "labels", "scores", "ref_scores", "full_scores", "mask", "order", "users"}
    for cls in (RankingEvaluator, SlateValueEvaluator):
        signature = inspect.signature(cls)
        assert "use_order" not in signature.parameters
        assert not any("scorer" in name for name in signature.parameters)
    assert "order" not in inspect.signature(ListNDCG.calculate).parameters
    assert "scores" not in inspect.signature(ListNDCG.calculate).parameters
    with pytest.raises(ValueError, match="Tensor"):
        EvalBatch(labels=torch.zeros(2), scores={"click": torch.zeros(2)})


def test_custom_target_bypasses_multilabel_guessing_and_runs_once_after_filter(tmp_path):
    class WeightedFeedback(BaseScorer):
        calls = 0
        def score(self, context, order=None):
            assert context.inter["domain"].tolist() == [1, 1]
            self.calls += 1
            return .2 * context.inter["click"] + .8 * context.inter["watch"]

    target = WeightedFeedback()
    schema = SchemaManager([], tmp_path, label_fields=["click", "watch"])
    manager = EvaluatorManager(schema, uid_field="uid", log_path=tmp_path / "metrics.log")
    capture = Capture()
    resolver = BatchResolver(label_scorer=target, pred_scorer=FieldScorer("value"),
                             ref_scorer=FieldScorer("ref"), mask_field="valid")
    manager.register("view", capture, DomainFilter("domain", 1), resolver=resolver)
    inter = Interaction({"click": torch.tensor([0., 1., 0.]), "watch": torch.tensor([0., 0., 1.]),
                         "value": torch.tensor([.1, .4, .9]), "ref": torch.tensor([.3, .2, .5]),
                         "domain": torch.tensor([0, 1, 1]), "uid": torch.tensor([10, 20, 30]),
                         "valid": torch.ones(3)})
    manager.collect(inter, Prediction(scores={"unrelated": torch.ones(3)}))
    assert target.calls == 1
    torch.testing.assert_close(capture.batch.labels, torch.tensor([.2, .8]))
    torch.testing.assert_close(capture.batch.scores, torch.tensor([.4, .9]))
    torch.testing.assert_close(capture.batch.ref_scores, torch.tensor([.2, .5]))
    assert capture.batch.users.tolist() == [20, 30]
    assert manager.summary(epoch=2, step=17) == {"view": {"rows": 2}}
    text = (tmp_path / "metrics.log").read_text()
    assert "view.rows" in text and "Epoch: 02" in text and "Step: 17" in text


def test_candidate_ranking_and_slate_views_do_not_double_project_or_cross_contaminate(tmp_path):
    schema = SchemaManager([], tmp_path, label_fields=SlateLabel("label", 4, "valid"))
    manager = EvaluatorManager(schema, uid_field="uid")
    captures = {view: Capture() for view in ("score_only", "ranking", "slate")}
    for view, capture in captures.items():
        manager.register(view, capture, resolver=BatchResolver(view=view, ref_scorer=FieldScorer("ref")))
    inter = Interaction({"label": torch.tensor([[1., 0., 1., 0.]]), "ref": torch.tensor([[.1, .2, .3, .4]]),
                         "valid": torch.tensor([[1, 0, 1, 1]]), "uid": torch.tensor([[7, 8, 9, 10]])})
    pred = Prediction(scores=torch.tensor([[.8, .2, .6, .1]]), order=torch.tensor([[2, 1]]))
    before = {name: inter[name].clone() for name in inter.columns}
    manager.collect(inter, pred)
    assert captures["score_only"].batch.order is None
    assert captures["ranking"].batch.order.tolist() == [[2, 1]]
    assert captures["ranking"].batch.labels.shape == (1, 4)
    slate = captures["slate"].batch
    assert slate.order is None
    assert slate.labels.tolist() == [[1., 0.]] and slate.mask.tolist() == [[1, 0]]
    assert slate.users.tolist() == [[9, 8]]
    torch.testing.assert_close(slate.scores, pred.scores[:, [2, 1]])
    torch.testing.assert_close(slate.ref_scores, inter["ref"][:, [2, 1]])
    assert all(torch.equal(inter[name], value) for name, value in before.items())


def test_full_catalog_resolves_full_scores_without_raw_prediction_access(tmp_path):
    manager = EvaluatorManager(SchemaManager([], tmp_path, label_fields="item"), uid_field="uid")
    manager.register("recall", RetrievalEvaluator("hr@1", "ndcg@2", history_dict={1: [2]}))
    inter = Interaction({"item": torch.tensor([1, 0]), "uid": torch.tensor([1, 2])})
    pred = Prediction(full_scores=torch.tensor([[.1, .8, .9], [.9, .2, .1]]))
    manager.collect(inter, pred)
    assert manager.summary() == {"recall": {"hr@1": 1., "ndcg@2": 1.}}


def test_ncis_uses_resolved_reference_and_pcoc_has_one_zero_rule(tmp_path):
    manager = EvaluatorManager(SchemaManager([], tmp_path, label_fields="value"))
    manager.register("ope", SlateValueEvaluator("ncis", "pcoc"),
                     resolver=BatchResolver(ref_scorer=FieldScorer("propensity")))
    manager.register("calibration", PointwiseEvaluator("pcoc"))
    inter = Interaction({"value": torch.tensor([1., 3.]), "propensity": torch.tensor([0., 1.])})
    manager.collect(inter, Prediction(scores=torch.tensor([1., .1])))
    assert manager.summary()["ope"]["ncis"] == pytest.approx((10 + .3) / 10.1)
    assert manager.summary()["ope"]["pcoc"] == manager.summary()["calibration"]["pcoc"]
    for evaluator in (PointwiseEvaluator("pcoc"), RankingEvaluator("pcoc"), SlateValueEvaluator("pcoc")):
        evaluator.collect(EvalBatch(labels=torch.zeros(1, 2), scores=torch.ones(1, 2)))
        assert math.isnan(evaluator.summary()["pcoc"])


def test_lookup_judge_flows_through_resolver_for_original_and_generated_lists(tmp_path):
    path = tmp_path / "judge.parquet"
    # Full-permutation scalar values, lexicographic indices 0..5.
    pl.DataFrame({"request_id": [29, 11], "scores": [7., 2.],
                  "perm_values": [[float(i + 10) for i in range(6)], [float(i) for i in range(6)]]}).write_parquet(path)
    inter = Interaction({"request_id": torch.tensor([11, 29]), "label": torch.zeros(2, 3)})
    pred = Prediction(order=torch.tensor([[2, 1, 0], [0, 2, 1]]))
    manager = EvaluatorManager(uid_field=None)
    capture = Capture()
    manager.register("judge", capture, resolver=BatchResolver(view="slate", pred_scorer=SidecarLookupScorer(path),
                                                             label_scorer=FieldScorer("label"), mask_field="valid"))
    inter["valid"] = torch.ones(2, 3)
    manager.collect(inter, pred)
    assert capture.batch.scores.tolist() == [5., 11.]
    assert capture.batch.labels.shape == (2, 3)


def test_model_judge_flows_through_slate_resolver_without_feedback_input(tmp_path):
    model, inter = case(tmp_path, "prm")
    judge = JudgeModelScorer(model)
    order = torch.tensor([[2, 0, 1], [1, 2, 0]])
    pred = Prediction(scores=torch.zeros(2, 3), order=order, target="label")
    manager = EvaluatorManager(model.manager)
    capture = Capture()
    manager.register("judge", capture, resolver=BatchResolver(view="slate", pred_scorer=judge))
    manager.collect(inter, pred)
    expected = judge(EvalContext(pred, inter, "label"), order)
    torch.testing.assert_close(capture.batch.scores, expected)
    assert not capture.batch.scores.requires_grad
    assert capture.batch.mask.shape == capture.batch.labels.shape == (2, 3)


@pytest.mark.parametrize("mapping", [False, True])
@pytest.mark.parametrize("with_scores", [False, True])
def test_maskless_slate_uses_original_width_for_order_only_and_mapping_sources(mapping, with_scores):
    labels = torch.tensor([[0., 1., 0., 1., 0.], [1., 0., 0., 1., 1.]])
    scores = torch.tensor([[.1, .9, .3, .7, .2], [.8, .2, .1, .9, .7]])
    order = torch.tensor([[3, 1], [4, 0]])
    pred = Prediction(scores=({"click": scores} if mapping else scores) if with_scores else None,
                      order={"click": order} if mapping else order)
    context = EvalContext(pred, Interaction({"click": labels}), "click")
    resolved = BatchResolver(view="slate").resolve(context)
    assert resolved.order is None and resolved.mask is None
    torch.testing.assert_close(resolved.labels, labels.gather(1, order))
    if with_scores:
        torch.testing.assert_close(resolved.scores, scores.gather(1, order))
    else:
        assert resolved.scores is None


def test_mapping_scores_supply_width_for_custom_scalar_target_without_mask():
    class Value(BaseScorer):
        calls = 0
        def score(self, context, order=None):
            self.calls += 1
            return torch.tensor([2.])
    value = Value()
    scores = torch.tensor([[.1, .2, .3, .4, .5]])
    pred = Prediction(scores={"click": scores}, order={"click": torch.tensor([[4, 2]])})
    result = BatchResolver(view="slate", label_scorer=value).resolve(EvalContext(pred, target="click"))
    assert result.labels.tolist() == [2.] and value.calls == 1
    torch.testing.assert_close(result.scores, scores[:, [4, 2]])


@pytest.mark.parametrize("has_order", [False, True])
@pytest.mark.parametrize("mask", [False, True])
def test_default_slate_resolver_selects_output_or_original_without_manual_view(tmp_path, has_order, mask):
    schema = SchemaManager([], tmp_path, label_fields=SlateLabel("label", 4, "valid") if mask else "label")
    manager = EvaluatorManager(schema)
    evaluator = SlateValueEvaluator("mae", reduction="sum")
    manager.register("value", evaluator)
    labels = torch.tensor([[1., 0., 0., 1.]])
    scores = torch.tensor([[.2, .3, .4, .8]])
    inter = Interaction({"label": labels, "valid": torch.tensor([[1, 1, 0, 1]])})
    order = torch.tensor([[2, 3]]) if has_order else None
    manager.collect(inter, Prediction(scores=scores, order=order))
    selected = torch.arange(4)[None] if order is None else order
    valid = inter["valid"].gather(1, selected).bool() if mask else torch.ones_like(selected, dtype=torch.bool)
    expected = abs((scores.gather(1, selected).double() * valid).sum() -
                   (labels.gather(1, selected).double() * valid).sum()).item()
    assert manager.registry["value"].resolver.view == "slate"
    assert manager.summary()["value"]["mae"] == pytest.approx(expected)
    # An explicit view is authoritative, even for a SlateValueEvaluator subclass.
    resolver = BatchResolver(view="score_only")
    manager.register("original", SlateValueEvaluator("mae"), resolver=resolver)
    assert manager.registry["original"].resolver is resolver


def test_maskless_lookup_judge_validates_before_returning_scalar_values(tmp_path):
    path = tmp_path / "values.parquet"
    pl.DataFrame({"request_id": [5], "scores": [2.], "perm_values": [[float(i) for i in range(6)]]}).write_parquet(path)
    judge = SidecarLookupScorer(path)
    context = EvalContext(Prediction(order=torch.tensor([[2, 0, 1]])),
                           Interaction({"request_id": torch.tensor([5])}))
    result = BatchResolver(view="slate", pred_scorer=judge, label_scorer=judge).resolve(context)
    assert result.scores.tolist() == result.labels.tolist() == [4.]


def test_maskless_model_judge_can_declare_width_without_raw_scores_or_labels(tmp_path):
    class Value(BaseScorer):
        def score(self, context, order=None):
            return torch.ones(context.pred.batch_size)
    model, inter = case(tmp_path, "prm")
    judge = JudgeModelScorer(model)
    inter = Interaction({name: inter[name] for name in inter.columns if name != "label"})
    pred = Prediction(order=torch.tensor([[2, 0, 1], [1, 2, 0]]))
    resolved = BatchResolver(view="slate", pred_scorer=judge, label_scorer=Value()).resolve(EvalContext(pred, inter))
    assert resolved.scores.shape == (2, 3) and resolved.labels.shape == (2,)
    assert not resolved.scores.requires_grad


def test_invalid_maskless_order_is_rejected_before_any_scorer_runs():
    class Value(BaseScorer):
        calls = 0
        def num_candidates(self, context):
            return context.inter["click"].shape[1]
        def score(self, context, order=None):
            self.calls += 1
            return torch.ones(context.pred.batch_size)
    value = Value()
    context = EvalContext(Prediction(order=torch.tensor([[0, 4]])), Interaction({"click": torch.ones(1, 4)}), "click")
    with pytest.raises(ValueError, match="partial permutation"):
        BatchResolver(view="slate", label_scorer=value).resolve(context)
    assert value.calls == 0


@pytest.mark.parametrize("view", ["ranking", "slate"])
def test_unused_local_predictions_do_not_override_full_candidate_source_width(view):
    labels = torch.arange(50.).expand(2, -1)
    stage1 = labels / 50
    order = torch.tensor([[49, 30, 12, 7, 3, 0], [48, 32, 17, 8, 4, 1]])
    context = EvalContext(Prediction(scores=torch.zeros(2, 6), order=order),
                           Interaction({"click": labels, "stage1": stage1}), "click")
    resolved = BatchResolver(view=view, pred_scorer=FieldScorer("stage1")).resolve(context)
    expected = labels if view == "ranking" else labels.gather(1, order)
    torch.testing.assert_close(resolved.labels, expected)
    torch.testing.assert_close(resolved.scores, expected / 50)


def test_each_scorer_declares_only_its_own_source_width_without_columns_attribute():
    from rosen.evaluate import PredictionScorer, LabelScorer
    class Value(BaseScorer):
        def score(self, context, order=None):
            return torch.ones(2)
    # A plain mapping deliberately has no .columns property.
    inter = {"click": torch.ones(2, 50), "stage1": torch.ones(2, 40), "scalar": torch.ones(2)}
    context = EvalContext(Prediction(scores={"click": torch.ones(2, 6)}), inter, "click")
    assert Value().num_candidates(context) is None
    assert PredictionScorer().num_candidates(context) == 6
    assert LabelScorer().num_candidates(context) == 50
    assert FieldScorer("stage1").num_candidates(context) == 40
    assert FieldScorer("scalar").num_candidates(context) is None
    assert LabelScorer().num_candidates(EvalContext(context.pred, inter)) is None
    assert LabelScorer().num_candidates(EvalContext(context.pred, inter, "missing")) is None


def test_order_only_resolver_does_not_construct_unused_prediction_scorer(monkeypatch):
    import rosen.evaluate.resolver as module
    def forbidden():
        raise AssertionError("Unused PredictionScorer was constructed")
    monkeypatch.setattr(module, "PredictionScorer", forbidden)
    context = EvalContext(Prediction(order=torch.tensor([[4, 1]])), {"click": torch.ones(1, 5)}, "click")
    result = BatchResolver(view="slate").resolve(context)
    assert result.scores is None and result.labels.shape == (1, 2)


@pytest.mark.parametrize("explicit_pred", [False, True])
def test_custom_target_cannot_relabel_a_consumed_single_tensor_prediction(tmp_path, explicit_pred):
    from rosen.evaluate import PredictionScorer
    manager = EvaluatorManager(SchemaManager([], tmp_path, label_fields=["click", "like"]))
    resolver = BatchResolver(label_scorer=FieldScorer("click"),
                             pred_scorer=PredictionScorer() if explicit_pred else None)
    manager.register("click", PointwiseEvaluator("auc"), target="click", resolver=resolver)
    inter = Interaction({"click": torch.tensor([0., 1.]), "like": torch.tensor([1., 0.])})
    with pytest.raises(ValueError, match="conflicts"):
        manager.collect(inter, Prediction(scores=torch.tensor([.2, .8]), target="like"))
    manager.collect(inter, Prediction(scores=torch.tensor([.2, .8]), target="click"))
    assert manager.summary()["click"]["auc"] == 1.


def test_custom_sources_can_ignore_unrelated_prediction_target(tmp_path):
    manager = EvaluatorManager(SchemaManager([], tmp_path, label_fields=["click", "like"]))
    manager.register("click", PointwiseEvaluator("auc"), target="click",
                     resolver=BatchResolver(label_scorer=FieldScorer("click"),
                                            pred_scorer=FieldScorer("stage1"), view="score_only"))
    inter = Interaction({"click": torch.tensor([0., 1.]), "stage1": torch.tensor([.2, .8])})
    manager.collect(inter, Prediction(scores=torch.tensor([.9, .1]), target="like"))
    assert manager.summary()["click"]["auc"] == 1.
