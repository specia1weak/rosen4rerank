"""Semantic output, target alignment and evaluator migration contracts."""
import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from sklearn.metrics import log_loss

from rosen.core.interaction import Interaction
from rosen.core.prediction import as_prediction
from rosen.core.train import EMAConfig, EarlyStopper, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer, EMATrainer
from rosen.emb import SchemaManager, SlateLabel
from rosen.evaluate import (
    BaseEvaluator, BatchResolver, DomainFilter, EvalBatch, FieldScorer, EvaluatorManager,
    PointwiseEvaluator, Prediction, RankingEvaluator, RetrievalEvaluator,
)
from rosen.models.base import BaseModel


def batch(labels, scores=None, *, mask=None, users=None, order=None):
    return EvalBatch( scores=scores, order=order, labels=labels, mask=mask, users=users)


def test_prediction_slices_every_slot_preserves_target_and_moves_indices():
    pred = Prediction(scores={"click": torch.arange(3.)}, order={"click": torch.tensor([[1, 0]] * 3)},
                      full_scores=torch.arange(12.).reshape(3, 4), target="click",
                      extra={"candidates": torch.arange(18.).reshape(3, 3, 2), "ids": torch.arange(3)})
    selected = pred[torch.tensor([True, False, True])].to("cpu")
    assert selected.batch_size == 2 and selected.target == "click"
    torch.testing.assert_close(selected.scores["click"], torch.tensor([0., 2.]))
    torch.testing.assert_close(selected.full_scores, pred.full_scores[[0, 2]])
    torch.testing.assert_close(selected.extra["candidates"], pred.extra["candidates"][[0, 2]])
    assert selected.order["click"].dtype == torch.long
    assert pred[torch.zeros(3, dtype=torch.bool)].batch_size == 0
    with pytest.raises(FrozenInstanceError):
        pred.target = "other"
    with pytest.raises(TypeError):
        pred.scores["other"] = torch.ones(3)
    with pytest.raises(ValueError, match="boolean"):
        pred[0]


@pytest.mark.parametrize("kwargs,match", [
    ({"scores": torch.ones(2), "extra": {"wrong": torch.ones(3)}}, "batch dimension"),
    ({"scores": torch.ones(2), "extra": {"constant": torch.tensor(3.)}}, "batch dimension"),
    ({"scores": torch.ones(2), "extra": {"nested": {"x": torch.ones(2)}}}, "Tensor values"),
    ({"order": torch.ones(2, 3)}, "LongTensor"),
    ({"scores": torch.ones(2, 3, 4)}, "scores"),
    ({"full_scores": torch.ones(2)}, "full_scores"),
    ({"scores": torch.ones(2), "target": " "}, "target"),
])
def test_prediction_rejects_ambiguous_or_misaligned_storage(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        Prediction(**kwargs)


def test_raw_tensor_normalization_and_untyped_mapping_rejection():
    tensor = torch.tensor([.1, .9])
    normalized = as_prediction(tensor, target="click")
    assert normalized.scores is tensor and normalized.target == "click"
    assert as_prediction(normalized, target="other") is normalized
    with pytest.raises(TypeError, match=r"predict\(\) must return Prediction or Tensor"):
        as_prediction({"click": tensor})


def test_multitask_routes_one_target_to_prediction_and_label(tmp_path):
    schema = SchemaManager([], tmp_path / "schema", label_fields=["click", "buy"])
    manager = EvaluatorManager(schema, log_path=tmp_path / "metrics.log")
    manager.register("click", PointwiseEvaluator("auc", "logloss"), target="click")
    manager.register("buy", PointwiseEvaluator("auc"), target="buy")
    inter = Interaction({"click": torch.tensor([0., 1., 0., 1.]), "buy": torch.tensor([1., 0., 1., 0.])})
    preds = Prediction(scores={"click": torch.tensor([.1, .9, .2, .8]), "buy": torch.tensor([.1, .9, .2, .8])})
    manager.collect(inter, preds)
    result = manager.summary(epoch=1)
    assert result["click"]["auc"] == 1 and result["buy"]["auc"] == 0
    assert (tmp_path / "metrics.log").is_file()
    manager.clear()
    assert manager.summary() == {"click": {}, "buy": {}}


@pytest.mark.parametrize("slot_users", [False, True])
def test_schema_mask_and_domain_filter_compose_independently(tmp_path, slot_users):
    schema = SchemaManager([], tmp_path, label_fields=SlateLabel("label", 3, "mask"))
    inter = Interaction({"label": torch.tensor([[0., 1., 1.], [0., 1., 1.], [1., 0., 0.]]),
                         "mask": torch.tensor([[1., 1., 0.], [1., 1., 0.], [1., 1., 0.]]),
                         "domain": torch.tensor([0, 1, 1]),
                         "uid": torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]) if slot_users else torch.tensor([1, 2, 3])})
    manager = EvaluatorManager(schema, uid_field="uid")
    evaluator = PointwiseEvaluator("auc", "logloss", "gauc")
    manager.register("domain", evaluator, DomainFilter("domain", 1))
    manager.register("all", PointwiseEvaluator("auc"))
    manager.collect(inter, Prediction(scores=torch.tensor([[.1, .9, .99], [.1, .9, .99], [.9, .1, .2]])))
    assert manager.summary()["domain"]["auc"] == manager.summary()["domain"]["gauc"] == 1.
    assert manager.summary()["all"]["auc"] == 1.


@pytest.mark.parametrize("evaluator", [PointwiseEvaluator("gauc"), RankingEvaluator("gauc")])
def test_gauc_without_users_is_an_explicit_error(evaluator):
    with pytest.raises(ValueError, match="uid_field"):
        evaluator.collect(batch(torch.tensor([[0., 1.]]), torch.tensor([[.1, .9]])))


def test_target_priority_conflicts_missing_fields_and_mask_validation(tmp_path):
    schema = SchemaManager([], tmp_path, label_fields=["click", SlateLabel("long", 2, "mask")])
    inter = Interaction({"click": torch.tensor([0., 1.]), "long": torch.tensor([[0., 1.], [1., 0.]])})
    manager = EvaluatorManager(schema)
    manager.register("point", PointwiseEvaluator("auc"))
    with pytest.raises(ValueError, match="Ambiguous"):
        manager.collect(inter, Prediction(scores={"click": torch.tensor([.1, .9])}))
    manager.register("point", PointwiseEvaluator("auc"), target="click")
    with pytest.raises(ValueError, match="conflicts"):
        manager.collect(inter, Prediction(scores=torch.tensor([.1, .9]), target="long"))
    with pytest.raises(KeyError, match="click"):
        manager.collect(inter, Prediction(scores={"missing": torch.tensor([.1, .9])}))
    manager.register("point", PointwiseEvaluator("auc"), target="missing")
    with pytest.raises(KeyError, match="not declared"):
        manager.collect(inter, Prediction(scores=torch.tensor([.1, .9])))
    manager.register("point", PointwiseEvaluator("auc"), target="long")
    with pytest.raises(KeyError, match="mask"):
        manager.collect(inter, Prediction(scores=torch.ones(2, 2)))
    with pytest.raises(ValueError, match="same shape"):
        PointwiseEvaluator("auc").collect(batch(torch.tensor([0., 1.]), torch.tensor([.1, .9]), mask=torch.ones(2, 2)))


def test_single_label_fallback_but_never_infer_a_mapping_key(tmp_path):
    inter = Interaction({"click": torch.tensor([0., 1.])})
    manager = EvaluatorManager(SchemaManager([], tmp_path, label_fields="click"))
    manager.register("all", PointwiseEvaluator("auc"))
    manager.collect(inter, Prediction(scores=torch.tensor([.1, .9])))
    assert manager.summary()["all"]["auc"] == 1.
    unbound = EvaluatorManager()
    unbound.register("all", PointwiseEvaluator("auc"))
    with pytest.raises(ValueError, match="never guessed"):
        unbound.collect(inter, Prediction(scores={"click": torch.tensor([.1, .9])}))


def test_slate_global_aggregation_and_position_relevance():
    labels = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 1., 0.]])
    scores = torch.tensor([[.9, .1, .2], [.8, .3, .1], [.4, .7, .2]])
    mask = torch.tensor([[1., 1., 0.], [1., 1., 1.], [1., 0., 0.]])
    evaluator = RankingEvaluator("auc", "logloss", "pcoc", "ndcg@3")
    all_at_once = RankingEvaluator("auc", "logloss", "pcoc", "ndcg@3")
    for start, stop in ((0, 2), (2, 3)):
        evaluator.collect(batch(labels[start:stop], scores[start:stop], mask=mask[start:stop]))
    all_at_once.collect(batch(labels, scores, mask=mask))
    assert evaluator.summary() == pytest.approx(all_at_once.summary())
    result = evaluator.summary()
    assert result["pcoc"] == pytest.approx(scores[mask > 0].sum().item() / labels[mask > 0].sum().item())
    assert result["logloss"] == pytest.approx(log_loss(labels[mask > 0], scores[mask > 0]), abs=1e-7)
    assert result["ndcg@3"] == pytest.approx((2 + 1 / math.log2(3)) / 3)
    position = RankingEvaluator("ndcg@3", )
    position.collect(batch(torch.tensor([[2., 1., 0.]]), torch.tensor([[0., 1., 2.]])))
    assert position.summary()["ndcg@3"] == pytest.approx((1 / math.log2(3) + 3 / 2) / (3 + 1 / math.log2(3)))
    zero = RankingEvaluator("pcoc", "ndcg@3")
    zero.collect(batch(torch.zeros(1, 3), torch.ones(1, 3)))
    assert math.isnan(zero.summary()["pcoc"])
    assert zero.summary()["ndcg@3"] == 0.
    with pytest.raises(ValueError, match="Unsupported slot"):
        RankingEvaluator("topk_ctr")


def test_order_is_not_score_sorting_and_mask_uses_input_coordinates():
    labels = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    scores = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    order = torch.tensor([[2, 1, 0], [2, 1, 0]])
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    context = batch(labels, scores, order=order, mask=mask)
    selected, sorted_scores = RankingEvaluator("ndcg@3"), RankingEvaluator("ndcg@3")
    selected.collect(context)
    sorted_scores.collect(batch(labels, scores, mask=mask))
    # Explicit output positions are preserved. A masked first slot has zero gain;
    # it must not silently move the positive item from position 3 to position 2.
    assert selected.summary()["ndcg@3"] == .5
    assert sorted_scores.summary()["ndcg@3"] == 1.
    with pytest.raises(ValueError, match="permutation"):
        selected.collect(batch(labels, order=torch.tensor([[0, 0, 2], [0, 1, 2]])))


def test_empty_masks_rows_and_filters_do_not_collect():
    evaluator = RankingEvaluator("ndcg@2", "pcoc", "auc")
    evaluator.collect(batch(torch.zeros(2, 2), torch.ones(2, 2), mask=torch.zeros(2, 2)))
    assert evaluator.summary() == {}
    manager = EvaluatorManager()
    manager.register("false", PointwiseEvaluator("auc"), lambda inter: False)
    manager.register("empty", PointwiseEvaluator("auc"), lambda inter: torch.zeros(2, dtype=torch.bool))
    manager.collect(None, Prediction(scores=torch.ones(2)))
    assert manager.summary() == {"false": {}, "empty": {}}
    manager.register("bad", PointwiseEvaluator("auc"), lambda inter: torch.ones(2))
    with pytest.raises(ValueError, match="boolean"):
        manager.collect(None, Prediction(scores=torch.ones(2)))


def test_custom_evaluator_receives_full_filtered_context_and_summary_errors_propagate():
    class Capture(BaseEvaluator):
        def collect(self, context):
            self.context = context
        def summary(self):
            return {"size": self.context.batch_size}
        def clear(self):
            pass
    capture = Capture()
    manager = EvaluatorManager(uid_field="uid")
    manager.register("capture", capture, DomainFilter("domain", 1), target="label",
                     resolver=BatchResolver(pred_scorer=FieldScorer("uid")))
    inter = Interaction({"domain": torch.tensor([0, 1]), "uid": torch.tensor([3, 4]), "label": torch.tensor([0., 1.])})
    manager.collect(inter, Prediction(scores=torch.tensor([.1, .9]), extra={"x": torch.tensor([[2.], [5.]])}))
    assert capture.context.scores.item() == capture.context.users.item() == 4
    assert not hasattr(capture.context, "pred") and not hasattr(capture.context, "inter")
    assert manager.summary() == {"capture": {"size": 1}}
    def broken_summary():
        raise TypeError("metric implementation failed")
    capture.summary = broken_summary
    with pytest.raises(TypeError, match="metric implementation failed"):
        manager.summary()


def test_full_catalog_filtering_and_history_never_mutate_shared_scores():
    inter = Interaction({"domain": torch.tensor([0, 1, 1]), "uid": torch.tensor([1, 2, 3]), "item": torch.tensor([0, 1, 0])})
    scores = torch.tensor([[.8, .9], [.1, .9], [.1, .9]])
    before = scores.clone()
    manager = EvaluatorManager(uid_field="uid")
    manager.register("filtered", RetrievalEvaluator("hr@1", history_dict={1: [1]}), DomainFilter("domain", 0), target="item")
    manager.register("masked", RetrievalEvaluator("hr@1", history_dict={1: [1]}), target="item")
    manager.register("all", RetrievalEvaluator("hr@1", "ndcg@2", history_dict={}), target="item")
    manager.collect(inter, Prediction(full_scores=scores))
    assert manager.summary()["filtered"]["hr@1"] == 1.
    assert manager.summary()["masked"]["hr@1"] == pytest.approx(2 / 3)
    assert manager.summary()["all"]["hr@1"] == pytest.approx(1 / 3)
    torch.testing.assert_close(scores, before, rtol=0, atol=0)


def test_registration_rejects_old_non_base_evaluator():
    class Legacy:
        def collect(self, users, labels, preds, full):
            self.count = labels.numel()
        def summary(self):
            return {"count": self.count}
        def clear(self):
            self.count = 0
    manager = EvaluatorManager()
    with pytest.raises(TypeError, match="inherit BaseEvaluator"):
        manager.register("legacy", Legacy())


class SemanticModel(BaseModel):
    def __init__(self, manager, raw=False):
        super().__init__(manager)
        self.raw = raw
        self.weight = torch.nn.Parameter(torch.tensor(1.))
    def calculate_loss(self, inter):
        return self.weight.square()
    def predict(self, inter):
        values = torch.sigmoid(self.weight * inter["x"])
        return values if self.raw else Prediction(scores={"click": values})


@pytest.mark.parametrize("trainer_class", [BaseTrainer, EMATrainer])
@pytest.mark.parametrize("raw", [False, True])
def test_predictions_without_uid_train_eval_early_stop_and_teacher_dispatch(tmp_path, trainer_class, raw):
    schema = SchemaManager([], tmp_path / "schema", label_fields="click")
    model = SemanticModel(schema, raw)
    manager = EvaluatorManager()
    manager.register("overall", PointwiseEvaluator("auc", "logloss"))
    inter = Interaction({"x": torch.tensor([-1., 1.]), "click": torch.tensor([0., 1.])})
    options = {"ema_config": EMAConfig(decays=(.5,), eval_student=True)} if trainer_class is EMATrainer else {}
    trainer = trainer_class(model, torch.optim.SGD(model.parameters(), lr=0.), schema,
                            TrainerDataLoaders([inter], [inter], [inter]),
                            TrainerComponents(evaluator_manager=manager, early_stopper=EarlyStopper(patience=1)),
                            SimpleNamespace(device="cpu", train_log_interval=100, max_epochs=5, ckpt_dir=""), **options)
    assert manager.schema_manager is schema
    assert trainer.run()["overall"]["auc"] == 1.
    assert trainer.epoch == 2
    assert trainer.evaluate_test()["overall"]["auc"] == 1.
    if trainer_class is EMATrainer:
        with torch.no_grad():
            trainer.ema.teachers[.5].weight.fill_(-1.)
        trainer.evaluate_epoch()
        assert trainer.last_eval_metrics_by_checkpoint["student"]["overall"]["auc"] == 1.
        assert trainer.last_eval_metrics_by_checkpoint["ema0p5"]["overall"]["auc"] == 0.


def test_tensor_wrapping_uses_actual_evaluated_models_target(tmp_path):
    schema = SchemaManager([], tmp_path, label_fields=["click", "buy"])
    student, teacher = SemanticModel(schema, raw=True), SemanticModel(schema, raw=True)
    student.LABEL, teacher.LABEL = "click", "buy"
    manager = EvaluatorManager()
    manager.register("overall", PointwiseEvaluator("auc"))
    inter = Interaction({"x": torch.tensor([-1., 1.]), "click": torch.tensor([0., 1.]), "buy": torch.tensor([1., 0.])})
    trainer = BaseTrainer(student, torch.optim.SGD(student.parameters(), lr=0.), schema,
                          TrainerDataLoaders([], [inter]), TrainerComponents(evaluator_manager=manager),
                          SimpleNamespace(device="cpu"))
    assert trainer._evaluate_loader(teacher, [inter])["overall"]["auc"] == 0.
    assert trainer.evaluate_epoch()["overall"]["auc"] == 1.
    with pytest.raises(ValueError, match="different SchemaManager"):
        manager.bind(SchemaManager([], tmp_path / "other", label_fields="click"))
