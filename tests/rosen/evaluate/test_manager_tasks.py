"""Registration tasks and the manager pipeline retain source/exception contracts."""
from dataclasses import FrozenInstanceError, is_dataclass
from types import SimpleNamespace

import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager, SlateLabel
from rosen.evaluate import (
    BaseEvaluator, BatchResolver, EvaluatorManager, FieldScorer, PointwiseEvaluator,
    Prediction, SlateValueEvaluator,
)


class Capture(BaseEvaluator):
    def __init__(self):
        self.batches = []

    def collect(self, batch):
        self.batches.append(batch)

    def summary(self):
        return {"rows": sum(batch.batch_size for batch in self.batches)}

    def clear(self):
        self.batches.clear()


def test_registration_uses_frozen_tasks_and_preserves_explicit_falsey_resolver():
    class FalseyResolver(BatchResolver):
        def __bool__(self):
            return False
    manager = EvaluatorManager()
    resolver = FalseyResolver(view="score_only")
    evaluator = Capture()
    manager.register("capture", evaluator, resolver=resolver, target="click", uid_field="uid")
    task = manager.registry["capture"]
    assert is_dataclass(task) and task.evaluator is evaluator and task.resolver is resolver
    assert task.target == "click" and task.uid_field == "uid"
    with pytest.raises(FrozenInstanceError):
        task.target = "other"
    manager.register("slate", SlateValueEvaluator("mae"))
    assert manager.registry["slate"].resolver.view == "slate"
    manager.register("capture", evaluator, target="buy")
    assert manager.registry["capture"].target == "buy" and task.target == "click"


@pytest.mark.parametrize("filter_fn", [lambda inter: True, lambda inter: False,
                                       lambda inter: torch.zeros(0, dtype=torch.bool)])
def test_empty_batches_skip_resolution_but_still_apply_the_filter(filter_fn):
    class NoResolution(BatchResolver):
        def resolve(self, *args, **kwargs):
            raise AssertionError("Empty rows reached the resolver")
    calls = []
    def tracked(inter):
        calls.append(inter)
        return filter_fn(inter)
    manager = EvaluatorManager()
    capture = Capture()
    manager.register("empty", capture, tracked, resolver=NoResolution())
    inter = Interaction({"click": torch.zeros(0)})
    manager.collect(inter, Prediction(scores=torch.zeros(0)))
    assert calls == [inter] and not capture.batches
    assert manager.summary() == {"empty": {"rows": 0}}


@pytest.mark.parametrize("mask", [torch.ones(2), torch.ones(2, 1, dtype=torch.bool), [True, False]])
def test_filter_errors_retain_exact_type_and_message(mask):
    manager = EvaluatorManager()
    manager.register("invalid", Capture(), lambda inter: mask)
    with pytest.raises(ValueError) as error:
        manager.collect(None, Prediction(scores=torch.zeros(2)))
    assert str(error.value) == "Evaluator filters must return bool or a boolean Tensor[B]."


def test_row_filter_keeps_multitask_orders_and_full_scores_aligned():
    manager = EvaluatorManager(uid_field="uid")
    capture = Capture()
    manager.register("capture", capture, lambda inter: inter["keep"] > 0, target="click")
    inter = Interaction({"click": torch.tensor([[1., 0., 1.], [0., 1., 0.]]),
                         "uid": torch.tensor([17, 29]), "keep": torch.tensor([0, 1])})
    pred = Prediction(scores={"click": torch.arange(6.).reshape(2, 3)},
                      order={"click": torch.tensor([[2, 0], [1, 2]])}, full_scores=torch.arange(10.).reshape(2, 5))
    manager.collect(inter, pred)
    batch, = capture.batches
    assert batch.users.tolist() == [29] and batch.labels.tolist() == [[0., 1., 0.]]
    assert batch.scores.tolist() == [[3., 4., 5.]] and batch.order.tolist() == [[1, 2]]
    assert batch.full_scores.tolist() == [[5., 6., 7., 8., 9.]]
    manager.clear()
    assert manager.summary() == {"capture": {"rows": 0}}


@pytest.mark.parametrize("task_uid,manager_uid,expected", [
    ("task_uid", "manager_uid", [31, 32]),
    (None, "manager_uid", [21, 22]), (None, None, [11, 12]),
])
def test_user_resolution_priority(task_uid, manager_uid, expected):
    schema = SimpleNamespace(label_fields=("click",), uid_field="schema_uid")
    manager = EvaluatorManager(schema, uid_field=manager_uid)
    capture = Capture()
    manager.register("view", capture, uid_field=task_uid)
    inter = Interaction({"click": torch.tensor([0., 1.]), "task_uid": torch.tensor([31, 32]),
                         "manager_uid": torch.tensor([21, 22]), "schema_uid": torch.tensor([11, 12])})
    manager.collect(inter, Prediction(scores=torch.tensor([.1, .9])))
    assert capture.batches[0].users.tolist() == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_uid_keeps_schema_fallback_optional_and_explicit_field_required(explicit):
    schema = SimpleNamespace(label_fields=("click",), uid_field="missing")
    manager = EvaluatorManager(schema, uid_field="missing" if explicit else None)
    capture = Capture()
    manager.register("view", capture)
    inter = Interaction({"click": torch.zeros(2)})
    pred = Prediction(scores=torch.zeros(2))
    if explicit:
        with pytest.raises(KeyError) as error:
            manager.collect(inter, pred)
        assert error.value.args == ("missing",)
    else:
        manager.collect(inter, pred)
        assert capture.batches[0].users is None


def test_explicit_mask_overrides_schema_mask_and_missing_mask_preserves_keyerror(tmp_path):
    schema = SchemaManager([], tmp_path, label_fields=SlateLabel("click", 2, "schema_mask"))
    manager = EvaluatorManager(schema)
    capture = Capture()
    manager.register("view", capture, resolver=BatchResolver(mask_field="explicit_mask"))
    inter = Interaction({"click": torch.ones(1, 2), "schema_mask": torch.ones(1, 2),
                         "explicit_mask": torch.tensor([[0, 1]])})
    pred = Prediction(scores=torch.ones(1, 2))
    manager.collect(inter, pred)
    assert capture.batches[0].mask.tolist() == [[0, 1]]
    inter.drop("explicit_mask")
    with pytest.raises(KeyError) as error:
        manager.collect(inter, pred)
    assert error.value.args == ("explicit_mask",)


@pytest.mark.parametrize("labels,requested,pred_target,mapping,expected_type,message", [
    (("click", "buy"), None, None, True, ValueError,
     "Ambiguous evaluation target: set register(target=...) or Prediction.target."),
    ((), None, None, True, ValueError,
     "Mapping predictions require an explicit target; keys are never guessed."),
    (("click", "buy"), "click", "buy", False, ValueError,
     "Prediction target 'buy' conflicts with registered target 'click'."),
    (("click",), "missing", None, False, KeyError,
     "Target 'missing' is not declared in SchemaManager.label_fields."),
])
def test_target_failures_keep_exact_messages(labels, requested, pred_target, mapping, expected_type, message):
    manager = EvaluatorManager(SimpleNamespace(label_fields=labels))
    manager.register("view", Capture(), target=requested)
    tensor = torch.zeros(2)
    prediction = Prediction(scores={"click": tensor} if mapping else tensor, target=pred_target)
    with pytest.raises(expected_type) as error:
        manager.collect(Interaction({"click": tensor, "buy": tensor}), prediction)
    assert error.value.args == (message,)


def test_custom_sources_return_early_without_schema_checks(tmp_path):
    manager = EvaluatorManager(SchemaManager([], tmp_path, label_fields=["click", "buy"]))
    capture = Capture()
    manager.register("value", capture, target="custom", resolver=BatchResolver(
        pred_scorer=FieldScorer("value"), label_scorer=FieldScorer("feedback"), view="score_only"))
    inter = Interaction({"value": torch.tensor([.2, .9]), "feedback": torch.tensor([0., 1.])})
    manager.collect(inter, Prediction(scores=torch.tensor([.1, .3]), target="buy"))
    assert capture.batches[0].labels.tolist() == [0., 1.]


@pytest.mark.parametrize("target", ["", " ", 7])
def test_registration_target_errors_unchanged(target):
    with pytest.raises(ValueError) as error:
        EvaluatorManager().register("bad", PointwiseEvaluator("auc"), target=target)
    assert str(error.value) == "target must be a non-empty label name."
