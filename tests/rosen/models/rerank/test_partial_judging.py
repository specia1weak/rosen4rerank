import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.evaluate import EvalContext, JudgeModelScorer, Prediction
from rosen.models.rerank import MultiGERerankModel, RERANK_MODEL_REGISTRY, select_by_order
from tests.rosen.models.rerank.test_reference_models import NAMES, case


def test_select_by_order_preserves_feature_axes_and_shared_orders():
    values = torch.arange(2 * 10 * 3).reshape(2, 10, 3)
    shared = torch.tensor([[9, 2, 0, 5], [1, 7, 3, 8]])
    selected = select_by_order(values, shared)
    assert selected.shape == (2, 2, 4, 3)
    torch.testing.assert_close(selected[:, 0], values[:, shared[0]])


@pytest.mark.parametrize('name', [n for n in NAMES if n not in ('extr', 'pier', 'prs', 'pier_submodular')
                                 and not issubclass(RERANK_MODEL_REGISTRY[n], MultiGERerankModel)])
def test_partial_judge_matches_explicit_padded_features_and_ignores_unselected_items(tmp_path, name):
    model, batch = case(tmp_path, name)
    model.eval()
    batch['mask'] = torch.ones_like(batch['mask'])
    order = torch.tensor([[[1, 0]], [[1, 0]]])
    before = {key: value.clone() for key, value in batch.interaction.items()}
    with torch.no_grad():
        result = model.score_permutations(batch, order)
        explicit = Interaction({key: value.clone() for key, value in batch.interaction.items()})
        for key in ('cur_item', 'cur_category', 'prior', 'mask'):
            explicit[key] = torch.cat((batch[key][:, [1, 0]], torch.zeros_like(batch[key][:, :1])), 1)
        explicit['slate_len'] = torch.full_like(batch['slate_len'], 2)
        expected = model.logits(explicit).sigmoid()[:, :2]
        torch.testing.assert_close(result[:, 0], expected)
        changed = Interaction({key: value.clone() for key, value in batch.interaction.items()})
        for key in ('cur_item', 'cur_category'):
            changed[key][:, 2] = 7
        changed['prior'][:, 2] = -100
        changed['label'].fill_(-123)
        torch.testing.assert_close(model.score_permutations(changed, order), result)
    assert result.shape == (2, 1, 2) and torch.isfinite(result).all()
    for key, value in before.items():
        torch.testing.assert_close(batch[key], value)
    judge = JudgeModelScorer(model)
    torch.testing.assert_close(judge(EvalContext(Prediction(order=order[:, 0]), batch), order[:, 0]), result[:, 0])


def test_partial_mask_is_applied_even_when_schema_has_no_mask(tmp_path):
    model, batch = case(tmp_path, 'prm')
    # Do not mutate the shared manager: create an unlabeled inference configuration.
    model.LABEL = None
    batch.drop('mask')
    result = model.eval().score_permutations(batch, torch.tensor([[[1]], [[0]]]))
    assert result.shape == (2, 1, 1) and torch.isfinite(result).all()


@pytest.mark.parametrize('name', ['prm', 'mge3'])
def test_judges_reject_nonpermutations_at_the_boundary(tmp_path, name):
    model, batch = case(tmp_path, name)
    for order in (torch.tensor([[[1, 1]], [[0, 1]]]), torch.tensor([[[1, 3]], [[0, 1]]])):
        with pytest.raises(ValueError):
            model.score_permutations(batch, order)


@pytest.mark.parametrize('name', ['extr', 'pier', 'prs', 'pier_submodular', 'mge3', 'mge_cosine', 'nar', 'seq2slate', 'mge_metric'])
def test_fixed_slot_models_explicitly_reject_partial_lists(tmp_path, name):
    model, batch = case(tmp_path, name)
    with pytest.raises(ValueError):
        model.score_permutations(batch, torch.tensor([[[1, 0]], [[1, 0]]]))
