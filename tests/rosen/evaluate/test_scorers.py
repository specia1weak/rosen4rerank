"""Scorer contracts: coordinates, sidecar IDs, candidate values and frozen judges."""
from dataclasses import replace
import itertools

import polars as pl
import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.evaluate import (
    BaseEvaluator, BaseScorer, BatchResolver, EvalContext, EvaluatorManager, FieldScorer, SidecarLookupScorer,
    JudgeModelScorer, PredictionScorer, Prediction, SlateValueEvaluator, RankingEvaluator, LabelScorer,
)
from rosen.models.rerank import MultiGERerankModel
from tests.rosen.models.rerank.test_reference_models import case


def batch():
    scores = torch.tensor([[.1, .7, .9], [.8, .3, .4]])
    labels = torch.tensor([[0., 1., 1.], [1., 0., 0.]])
    inter = Interaction({"request_id": torch.tensor([2**60 + 7, 31]), "prior": scores + .01, "label": labels})
    return EvalContext(Prediction(scores=scores), inter=inter, target="label")


@pytest.mark.parametrize("scorer,field", [(PredictionScorer(), "scores"), (LabelScorer(), "labels"),
                                        (FieldScorer("prior"), "prior")])
def test_fixed_scorers_single_and_shared_orders(scorer, field):
    b = batch()
    original = b.inter[field] if field == "prior" else b.pred.scores if field == "scores" else b.inter["label"]
    torch.testing.assert_close(scorer(b), original)
    order = torch.tensor([[2, 0, 1], [1, 2, 0]])
    torch.testing.assert_close(scorer(b, order), original.gather(1, order))
    # B == M: a shared candidate axis must be explicit, never inferred as B.
    shared = order[None]
    expected = torch.stack([original[:, o] for o in order], 1)
    torch.testing.assert_close(scorer(b, shared), expected)
    full = torch.tensor(list(itertools.permutations(range(3))))
    torch.testing.assert_close(scorer(b, full), torch.stack([original[:, o] for o in full], 1))
    individual = torch.stack((full[:2], full[2:4]))
    torch.testing.assert_close(scorer(b, individual),
                               original[:, None].expand(-1, 2, -1).gather(-1, individual))


def test_scalar_fields_are_not_misrepresented_as_slot_permutations():
    b = batch()
    b.inter["value"] = torch.tensor([5., 7.])
    scorer = FieldScorer("value")
    torch.testing.assert_close(scorer(b), torch.tensor([5., 7.]))
    torch.testing.assert_close(scorer(b, torch.tensor([[0, 1, 2], [0, 1, 2]])), torch.tensor([5., 7.]))


@pytest.mark.parametrize("order", [torch.tensor([[0, 0, 2], [0, 1, 2]]),
                                   torch.tensor([[0, 1, 3], [0, 1, 2]])])
def test_invalid_permutations_fail(order):
    with pytest.raises(ValueError, match="permutation"):
        PredictionScorer()(batch(), order)


def test_missing_sources_fail_explicitly():
    b = batch()
    with pytest.raises(ValueError, match="target scores"):
        LabelScorer()(replace(b, target=None))
    with pytest.raises(ValueError, match="batch.inter"):
        FieldScorer("prior")(replace(b, inter=None))
    with pytest.raises(TypeError, match="LongTensor"):
        PredictionScorer()(b, torch.zeros(2, 3))


def test_lookup_all_720_values_and_arbitrary_subsets_match_exact_rows(tmp_path):
    b = batch()
    b = replace(b, pred=Prediction(scores=torch.zeros(2, 6)))
    ids = b.inter["request_id"].tolist()
    # Store reversed IDs to ensure row order does not masquerade as a key join.
    table = pl.DataFrame({"request_id": ids[::-1], "scores": [[.2] * 6, [.9] * 6],
                          "perm_values": [[i + 1000. for i in range(720)], [float(i) for i in range(720)]]})
    path = tmp_path / "judge.parquet"
    table.write_parquet(path)
    scorer = SidecarLookupScorer(path)
    torch.testing.assert_close(scorer(b), torch.tensor([[.9] * 6, [.2] * 6]))
    full = torch.tensor(list(itertools.permutations(range(6))))
    expected = torch.arange(720.).expand(2, -1).clone()
    expected[1] += 1000
    torch.testing.assert_close(scorer(b, full), expected)
    torch.testing.assert_close(scorer(b, full[[56, 719]]), torch.tensor([56., 1719.]))
    subset = full[torch.tensor([[2, 5, 10], [12, 15, 20]])]
    torch.testing.assert_close(scorer(b, subset), torch.tensor([[2., 5., 10.], [1012., 1015., 1020.]]))
    # A filtered batch must use its own request IDs.
    selected = replace(b, pred=Prediction(scores=torch.zeros(1, 6)),
                       inter=b.inter[torch.tensor([False, True])])
    torch.testing.assert_close(scorer(selected, full[None]), expected[1:])


def test_lookup_scalar_scores_missing_ids_and_missing_permutation_values(tmp_path):
    path = tmp_path / "scores.parquet"
    pl.DataFrame({"request_id": [2**60 + 7, 31], "scores": [3., 7.]}).write_parquet(path)
    scorer = SidecarLookupScorer(path)
    torch.testing.assert_close(scorer(batch()), torch.tensor([3., 7.]))
    with pytest.raises(ValueError, match="requires perm_values"):
        scorer(batch(), torch.tensor([[0, 1, 2], [0, 1, 2]]))
    b = batch()
    b.inter["request_id"] = torch.tensor([99, 31])
    with pytest.raises(KeyError, match="99"):
        scorer(b)
    pl.DataFrame({"request_id": [1, 1], "scores": [3., 7.]}).write_parquet(path)
    with pytest.raises(ValueError, match="unique"):
        SidecarLookupScorer(path)


@pytest.mark.parametrize("name", ["dnn", "dcn", "prm", "extr", "kuaishou", "pier", "prs", "nar",
                                  "seq2slate", "pier_submodular", "mge3", "mge_cosine", "mge_metric", "mge_cosine_metric"])
def test_model_scorer_recomputes_supplied_lists_without_feedback_features(tmp_path, name):
    model, inter = case(tmp_path, name)
    scorer = JudgeModelScorer(model)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    unlabeled = Interaction({field: inter[field] for field in inter.columns if field != "label"})
    prediction = model.predict(unlabeled)
    b = EvalContext(prediction, inter=unlabeled)
    torch.testing.assert_close(scorer(b), prediction.scores)
    orders = torch.tensor([[[2, 1, 0], [1, 0, 2]], [[1, 2, 0], [0, 2, 1]]])
    output = scorer(b, orders)
    assert output.shape == (2, 2, 3) and not output.requires_grad
    assert torch.isfinite(output).all()
    for i in range(2):
        fields = {setting.field_name for setting in model.slate_view.selected_settings} | {"prior", "mask"}
        changed = Interaction({field: unlabeled[field].gather(1, orders[:, i]) if field in fields
                               else unlabeled[field] for field in unlabeled.columns})
        # Compare raw scoring on the changed slate, not a gathered original prediction.
        if hasattr(model, "logits"):
            expected = model.logits(changed).sigmoid()
        else:
            expected = model.predict(changed).scores
        expected = expected * changed["mask"]
        torch.testing.assert_close(output[:, i], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(scorer(b, orders[:, 0]), output[:, 0])
    assert all(torch.equal(inter[field], unlabeled[field]) for field in unlabeled.columns)


def test_evaluator_scorer_interfaces_still_support_registered_subclasses():
    with pytest.raises(TypeError, match="abstract"):
        BaseScorer()
    assert isinstance(RankingEvaluator("pcoc"), BaseEvaluator)
    assert isinstance(SlateValueEvaluator("rmse"), BaseEvaluator)

    class SlotMean(RankingEvaluator):
        def collect(self, batch):
            self.value = batch.scores.mean().item()

        def summary(self):
            return {"mean": self.value}

        def clear(self):
            self.value = 0.

    class SlateMean(SlateValueEvaluator):
        def reduce_slate(self, tensor, mask):
            return tensor.mean(-1)

        def collect(self, batch):
            self.value = self.reduce_slate(batch.labels, batch.mask).mean().item()

        def summary(self):
            return {"mean": self.value}

        def clear(self):
            self.value = 0.

    slot = SlotMean()
    slate = SlateMean()
    assert isinstance(slot, BaseEvaluator) and isinstance(slate, BaseEvaluator)
    manager = EvaluatorManager()
    manager.register("slot", slot, target="label", resolver=BatchResolver(pred_scorer=FieldScorer("prior")))
    manager.register("slate", slate, resolver=BatchResolver(label_scorer=FieldScorer("prior")))
    b = batch()
    manager.collect(b.inter, b.pred)
    assert manager.summary()["slot"]["mean"] == pytest.approx(b.inter["prior"].mean().item())
    assert manager.summary()["slate"]["mean"] == pytest.approx(b.inter["prior"].mean().item())
    manager.clear()
    assert slot.value == slate.value == 0
