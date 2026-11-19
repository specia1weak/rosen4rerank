"""Source declarations, collect-scoped reuse and compact batch accumulation."""
from dataclasses import dataclass
from types import SimpleNamespace
from weakref import ref

import numpy as np
import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.evaluate import (
    BaseEvaluator, BaseScorer, BatchResolver, EvaluatorManager, FieldScorer,
    PointwiseEvaluator, Prediction, SlateValueEvaluator,
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


@pytest.mark.parametrize("source", ["pred_scorer", "label_scorer", "ref_scorer"])
def test_custom_scorer_declares_schema_dependency_without_concrete_type_checks(source):
    class SchemaSource(BaseScorer):
        uses_schema_labels = True

        def score(self, context, order=None):
            return context.inter[context.target]

    scorer = SchemaSource()
    sources = dict(pred_scorer=FieldScorer("value"), label_scorer=FieldScorer("feedback"))
    sources[source] = scorer
    resolver = BatchResolver(view="score_only", **sources)
    assert resolver.uses_schema_labels and resolver.needs_target(Prediction(scores=torch.ones(2)))
    manager = EvaluatorManager(SimpleNamespace(label_fields=("click",)))
    capture = Capture()
    manager.register("value", capture, target="custom", resolver=resolver)
    inter = Interaction({"value": torch.tensor([.2, .9]), "feedback": torch.tensor([0., 1.]),
                         "click": torch.tensor([1., 0.])})
    with pytest.raises(KeyError, match="not declared"):
        manager.collect(inter, Prediction(scores=torch.ones(2)))
    assert not capture.batches
    manager.register("value", capture, target="click", resolver=resolver)
    manager.collect(inter, Prediction(scores=torch.ones(2)))
    assert capture.summary() == {"rows": 2}


@pytest.mark.parametrize("source", ["pred_scorer", "label_scorer", "ref_scorer"])
def test_custom_prediction_source_declares_target_binding_without_field_introspection(source):
    class PredictionSource(BaseScorer):
        def uses_prediction_target(self, pred):
            return True

        def score(self, context, order=None):
            return context.pred.get_scores(context.target)

    sources = dict(pred_scorer=FieldScorer("value"), label_scorer=FieldScorer("feedback"))
    sources[source] = PredictionSource()
    resolver = BatchResolver(view="score_only", **sources)
    manager = EvaluatorManager(SimpleNamespace(label_fields=("click", "buy")))
    capture = Capture()
    manager.register("value", capture, target="click", resolver=resolver)
    inter = Interaction({"value": torch.tensor([.2, .9]), "feedback": torch.tensor([0., 1.])})
    with pytest.raises(ValueError, match="conflicts"):
        manager.collect(inter, Prediction(scores=torch.ones(2), target="buy"))
    assert not capture.batches
    manager.collect(inter, Prediction(scores=torch.ones(2), target="click"))
    assert capture.summary() == {"rows": 2}


@pytest.mark.parametrize("custom_filter", [False, True])
def test_shared_filters_and_cpu_transfers_run_once_per_collect(monkeypatch, custom_filter):
    @dataclass
    class Keep:
        calls: int = 0

        def __call__(self, inter):
            self.calls += 1
            return inter["keep"].bool()

        def __bool__(self):
            # A falsey callable must still be used; identity caching allows unhashable filters.
            return False

    keep = Keep() if custom_filter else None
    manager = EvaluatorManager(SimpleNamespace(label_fields=("label",)))
    left, right = Capture(), Capture()
    manager.register("left", left, keep)
    manager.register("right", right, keep)
    filter_calls, transfer_calls = [], []
    original_filter, original_cpu = manager._filter_rows, torch.Tensor.cpu

    def filter_rows(*args):
        filter_calls.append(True)
        return original_filter(*args)

    def cpu(tensor, *args, **kwargs):
        transfer_calls.append(tensor)
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(manager, "_filter_rows", filter_rows)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    for _ in range(2):
        inter = Interaction({"label": torch.tensor([0., 1., 0.]), "keep": torch.tensor([1, 1, 0])})
        manager.collect(inter, Prediction(scores=torch.tensor([.2, .8, .3])))
        assert left.batches[-1].labels is right.batches[-1].labels
        assert left.batches[-1].scores is right.batches[-1].scores
    assert len(filter_calls) == 2 and len(transfer_calls) == 4
    if custom_filter:
        assert keep.calls == 2 and left.summary() == {"rows": 4}
    else:
        assert left.summary() == {"rows": 6}


def test_filters_and_source_tensors_are_not_confused_across_registrations():
    manager = EvaluatorManager(SimpleNamespace(label_fields=("label",)))
    left, right = Capture(), Capture()
    manager.register("left", left, lambda inter: inter["uid"] == 1)
    manager.register("right", right, lambda inter: inter["uid"] == 2)
    inter = Interaction({"label": torch.tensor([0., 1.]), "uid": torch.tensor([1, 2])})
    manager.collect(inter, Prediction(scores=torch.tensor([.2, .8])))
    assert left.batches[0].labels.item() == 0. and right.batches[0].labels.item() == 1.
    assert left.batches[0].scores.item() == pytest.approx(.2)
    assert right.batches[0].scores.item() == pytest.approx(.8)


def test_transfer_cache_releases_temporary_scorer_sources_between_tasks():
    class TemporarySource(BaseScorer):
        previous = None
        calls = 0

        def score(self, context, order=None):
            if self.previous is not None:
                assert self.previous() is None, "Transfer cache retained a temporary source Tensor"
            tensor = context.inter["value"].clone()
            self.previous = ref(tensor)
            self.calls += 1
            return tensor

    scorer = TemporarySource()
    resolver = BatchResolver(pred_scorer=scorer, label_scorer=FieldScorer("feedback"), view="score_only")
    manager = EvaluatorManager()
    manager.register("left", Capture(), resolver=resolver)
    manager.register("right", Capture(), resolver=resolver)
    manager.collect(Interaction({"value": torch.tensor([.2, .9]), "feedback": torch.tensor([0., 1.])}),
                    Prediction(scores=torch.ones(2)))
    assert scorer.calls == 2 and scorer.previous() is None


def forbid_scalar_lists(*args, **kwargs):
    raise AssertionError("Evaluation caches must not materialize Python scalar lists")


def test_pointwise_batch_arrays_preserve_global_metrics_and_snapshot_inputs(monkeypatch):
    from rosen.evaluate import EvalBatch
    monkeypatch.setattr(torch.Tensor, "tolist", forbid_scalar_lists)
    evaluator = PointwiseEvaluator("auc", "gauc", "logloss", "pcoc")
    assert evaluator.summary() == {}
    for label, probabilities in ((0., [.1, .2]), (1., [.8, .9])):
        labels, scores, users = torch.full((2,), label), torch.tensor(probabilities), torch.ones(2, dtype=torch.long)
        evaluator.collect(EvalBatch(labels=labels, scores=scores, users=users))
        labels.fill_(99.)
        scores.fill_(float("nan"))
        users.fill_(99)
    expected = {"auc": 1., "gauc": 1., "logloss": -np.log([.9, .8, .8, .9]).mean(), "pcoc": 1.}
    assert evaluator.summary() == pytest.approx(expected)
    assert evaluator.summary() == pytest.approx(expected)
    chunks = evaluator._label_batches + evaluator._score_batches + evaluator._user_batches
    assert len(chunks) == 6 and all(isinstance(chunk, np.ndarray) for chunk in chunks)
    assert sum(chunk.nbytes for chunk in chunks) == 4 * 3 * 8
    evaluator.clear()
    assert evaluator.summary() == {} and not evaluator._user_batches


def test_slate_batch_arrays_preserve_global_value_metrics_and_snapshot_inputs(monkeypatch):
    from rosen.evaluate import EvalBatch
    monkeypatch.setattr(torch.Tensor, "tolist", forbid_scalar_lists)
    evaluator = SlateValueEvaluator("r_auc", "mae", "rmse", "pcoc", "ncis")
    assert evaluator.summary() == {}
    for labels, scores, refs in (([1., 4.], [2., 3.], [1., 2.]), ([2., 3.], [1., 4.], [1., 2.])):
        # float64 scalar views can share the input buffer; caches must own a snapshot.
        batch = EvalBatch(labels=torch.tensor(labels, dtype=torch.float64),
                          scores=torch.tensor(scores, dtype=torch.float64),
                          ref_scores=torch.tensor(refs, dtype=torch.float64))
        evaluator.collect(batch)
        for tensor in (batch.labels, batch.scores, batch.ref_scores):
            tensor.fill_(float("nan"))
    expected = {"r_auc": 2/3, "mae": 1., "rmse": 1., "pcoc": 1., "ncis": 32/13}
    assert evaluator.summary() == pytest.approx(expected)
    assert evaluator.summary() == pytest.approx(expected)
    chunks = evaluator._prediction_batches + evaluator._value_batches + evaluator._reference_batches
    assert len(chunks) == 6 and all(isinstance(chunk, np.ndarray) for chunk in chunks)
    assert sum(chunk.nbytes for chunk in chunks) == 4 * 3 * 8
    evaluator.clear()
    assert evaluator.summary() == {} and not evaluator._reference_batches
