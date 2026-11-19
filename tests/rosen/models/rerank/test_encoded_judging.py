"""One explicit-mask, encoded-context judge path across every rerank model."""
import math

import pytest
import torch

from rosen.core.interaction import Interaction
from rosen.models.rerank import RERANK_MODEL_REGISTRY, SlateRerankModel
from rosen.models.rerank.components.scoring import EncodedSlate, score_candidate_logits
from tests.rosen.models.rerank.test_reference_models import case


ORDERS = torch.tensor([[[2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 1, 0], [0, 2, 1]],
                       [[1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [0, 1, 2]]])


def explicit_candidate(batch, order, *, n=3):
    """Independent schema-field construction; no private keys or model adapters."""
    data = dict(batch.interaction)
    for field in ('cur_item', 'cur_category', 'prior', 'mask'):
        values = batch[field].gather(1, order)
        data[field] = torch.cat((values, values.new_zeros((len(batch), n - order.shape[1]))), dim=1)
    data['slate_len'] = (data['mask'] > 0).sum(-1)
    data.pop('label', None)
    return Interaction(data)


@pytest.mark.parametrize('name', tuple(RERANK_MODEL_REGISTRY))
def test_every_judge_uses_the_same_entry_point_and_matches_explicit_slate_logits(tmp_path, name):
    model, batch = case(tmp_path, name)
    model.eval()
    assert type(model).score_permutations is SlateRerankModel.score_permutations
    assert type(model).logits is SlateRerankModel.logits
    batch['mask'] = torch.tensor([[1., 0., .5], [0., 1., 1.]])
    supported_widths = (3,) if model.requires_full_slate else (1, 2, 3)
    with torch.no_grad():
        for k in supported_widths:
            orders = ORDERS[..., :k]
            expected = []
            for candidate in orders.unbind(dim=1):
                explicit = explicit_candidate(batch, candidate)
                expected.append(model.logits(explicit).sigmoid()[:, :k] * batch['mask'].gather(1, candidate))
            expected = torch.stack(expected, dim=1)
            for chunk_size in (1, 2, 99):
                model.candidate_chunk_size = chunk_size
                torch.testing.assert_close(model.score_permutations(batch, orders), expected, rtol=1e-5, atol=2e-7)
    assert not hasattr(model, '_ordered_slate_interactions')


@pytest.mark.parametrize('name', tuple(RERANK_MODEL_REGISTRY))
def test_items_and_history_are_encoded_once_per_judge_before_candidate_chunks(tmp_path, name, monkeypatch):
    model, batch = case(tmp_path, name)
    model.eval()
    model.candidate_chunk_size = 2
    seen, calls = [], {'slate': 0, 'history': 0, 'history_context': 0}
    forward = model.omni_embedding.forward
    scorer = model._score_encoded

    def lookup(interaction, *args, **kwargs):
        calls['slate'] += 1
        assert interaction is batch
        return forward(interaction, *args, **kwargs)

    # EmbView resolves fields through OmniEmbLayer.forward. Count both views,
    # and separately inspect the fixed history encoder or K/V projections.
    monkeypatch.setattr(model.omni_embedding, 'forward', lookup)
    handles = []
    if model.outputs_candidate_pool:
        history_inputs, history_tensors = model._history_inputs, []

        def history_source(*args):
            history, pages = history_inputs(*args)
            history_tensors.append(history)
            return history, pages

        monkeypatch.setattr(model, '_history_inputs', history_source)

        def track_history(module, args):
            if any(args[0] is history for history in history_tensors):
                calls['history'] += 1
        handles.append(model.oam.register_forward_pre_hook(track_history))
    elif name == 'kuaishou':
        handles.append(model.history_context.k.register_forward_pre_hook(lambda *args: calls.__setitem__('history_context', calls['history_context'] + 1)))

    def score(items, dense, mask, context):
        seen.append((items.shape, mask.clone(), context))
        return scorer(items, dense, mask, context)

    monkeypatch.setattr(model, '_score_encoded', score)
    with torch.no_grad():
        result = model.score_permutations(batch, ORDERS)
    for handle in handles:
        handle.remove()
    assert result.shape == (2, 5, 3) and len(seen) == math.ceil(5 / model.candidate_chunk_size)
    assert [shape[1] for shape, _, _ in seen] == [2, 2, 1]
    assert all(context is seen[0][2] for _, _, context in seen)
    # All candidates share one current view plus one history view only when used.
    assert calls['slate'] == (2 if model.outputs_candidate_pool or name == 'kuaishou' else 1)
    if model.outputs_candidate_pool:
        assert calls['history'] == 1
    if name == 'kuaishou':
        assert calls['history_context'] == 1


@pytest.mark.parametrize('name', ['prm', 'kuaishou'])
def test_partial_scorer_receives_explicit_mask_without_schema_mask_or_hidden_fields(tmp_path, name, monkeypatch):
    model, batch = case(tmp_path, name)
    model.LABEL = None
    batch.drop('mask')
    batch.drop('label')
    model.eval()
    original_fields = set(batch.columns)
    scorer, observed = model._score_encoded, []

    def masked(items, dense, mask, context):
        assert mask.shape == (2, 2, 3)
        torch.testing.assert_close(mask, torch.tensor([[[1., 0., 0.]] * 2] * 2))
        assert items[:, :, 1:].count_nonzero() == dense[:, :, 1:].count_nonzero() == 0
        observed.append(mask)
        return scorer(items, dense, mask, context)

    monkeypatch.setattr(model, '_score_encoded', masked)
    model.candidate_chunk_size = 2
    orders = ORDERS[:, :2, :1]
    with torch.no_grad():
        result = model.score_permutations(batch, orders)
    assert len(observed) == 1 and result.shape == (2, 2, 1)
    assert set(batch.columns) == original_fields


@pytest.mark.parametrize('name', ['prm', 'kuaishou', 'pier', 'mge_metric'])
def test_chunked_judge_gradients_equal_explicit_full_candidate_evaluation(tmp_path, name):
    model, batch = case(tmp_path, name)
    model.eval()
    orders = ORDERS[:, :3]
    model.candidate_chunk_size = 2
    model.score_permutations(batch, orders).sum().backward()
    actual = {key: None if value.grad is None else value.grad.clone() for key, value in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    for candidate in orders.unbind(1):
        explicit = explicit_candidate(batch, candidate)
        mask = batch['mask'].gather(1, candidate)
        (model.logits(explicit).sigmoid() * mask).sum().backward()
    for key, value in model.named_parameters():
        assert (value.grad is None) == (actual[key] is None), key
        if value.grad is not None:
            torch.testing.assert_close(value.grad, actual[key], rtol=1e-4, atol=1e-6, msg=key)


@pytest.mark.parametrize('name', ['prm', 'pier', 'mge3'])
@pytest.mark.parametrize('order', [torch.zeros(2, 1, 3), torch.tensor([[[0, 0, 1]], [[0, 1, 2]]]),
                                   torch.tensor([[[0, 1, 3]], [[0, 1, 2]]]),
                                   torch.tensor([[[-1, 1, 2]], [[0, 1, 2]]]),
                                   torch.zeros(2, 0, 3, dtype=torch.long)])
def test_invalid_orders_fail_before_lookup_or_context_encoding(tmp_path, name, order, monkeypatch):
    model, batch = case(tmp_path, name)

    def forbidden(*args):
        raise AssertionError('Invalid order reached feature/context encoding')

    monkeypatch.setattr(model, 'encode_slate', forbidden)
    with pytest.raises(ValueError):
        model.score_permutations(batch, order)


def test_new_models_must_use_explicit_encoded_scoring_extension_points():
    with pytest.raises(TypeError, match='_score_encoded'):
        class ImplicitJudge(SlateRerankModel):
            def logits(self, interaction):
                return interaction['untracked_control']

    with pytest.raises(TypeError, match='fixed entry points'):
        class ParallelJudge(SlateRerankModel):
            def score_permutations(self, interaction, orders):
                return torch.ones_like(orders)


@pytest.mark.parametrize('name', ['dnn', 'prm', 'kuaishou'])
def test_custom_source_mask_is_selected_once_then_passed_explicitly(tmp_path, name, monkeypatch):
    model, batch = case(tmp_path, name)
    model.eval()
    batch['alternate_validity'] = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    calls = []

    def source_mask(interaction, reference):
        calls.append(interaction)
        return interaction['alternate_validity'].to(reference.dtype)

    monkeypatch.setattr(model, '_mask', source_mask)
    orders = ORDERS[..., :2]
    with torch.no_grad():
        actual = model.score_permutations(batch, orders)
        assert calls == [batch]
        expected = []
        for candidate in orders.unbind(1):
            explicit = explicit_candidate(batch, candidate)
            selected = batch['alternate_validity'].gather(1, candidate)
            explicit['alternate_validity'] = torch.cat((selected, selected.new_zeros((2, 1))), 1)
            expected.append(model.logits(explicit).sigmoid()[:, :2] * selected)
        torch.testing.assert_close(actual, torch.stack(expected, 1), rtol=1e-5, atol=2e-7)


def test_bad_encoded_scorer_shape_is_rejected_instead_of_broadcasting():
    encoded = EncodedSlate(torch.ones(2, 3, 4), torch.ones(2, 3, 1), torch.ones(2, 3))
    with pytest.raises(ValueError, match='scorer must return logits'):
        score_candidate_logits(encoded, ORDERS, scorer=lambda items, dense, mask, context: mask[..., 0])
