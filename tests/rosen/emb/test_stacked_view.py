import pytest
import torch

from rosen.core.enum_type import FeatureSource
from rosen.core.interaction import Interaction
from rosen.emb.emblayer import EmbView, OmniEmbLayer
from rosen.emb.schema import SeqGroupConfig, SequenceSetting, SparseEmbSetting


def _example(category_dim=4):
    item = SparseEmbSetting("item", FeatureSource.ITEM_ID, embedding_dim=4)
    category = SparseEmbSetting("category", FeatureSource.ITEM, embedding_dim=category_dim)
    for setting in (item, category):
        setting._build_vocab_indices(["a", "b", "c"])
    group = SeqGroupConfig("slate", "slate_len", max_len=3)
    omni = OmniEmbLayer(emb_settings=[
        item, category,
        SequenceSetting("slate_item", item, group),
        SequenceSetting("slate_category", category, group),
    ])
    interaction = Interaction({
        "item": torch.tensor([1, 2]),
        "category": torch.tensor([2, 3]),
        "slate_item": torch.tensor([[1, 2, 0], [3, 1, 2]]),
        "slate_category": torch.tensor([[2, 3, 0], [1, 2, 3]]),
        "slate_len": torch.tensor([2, 3]),
    })
    return omni, interaction


@pytest.mark.parametrize("sequence", [False, True])
def test_stack_preserves_field_order_values_and_embedding_gradients(sequence):
    omni, interaction = _example()
    view = omni.seq_groups["slate"] if sequence else EmbView(omni)
    fields = tuple(setting.field_name for setting in view.selected_settings)
    by_name = view(interaction, split_by="name")
    stacked = view(interaction, split_by="stack")
    concatenated = view(interaction)

    assert fields == (("slate_item", "slate_category") if sequence else ("item", "category"))
    assert view.num_fields == 2
    assert view.field_dim == 4
    assert stacked.shape == ((2, 3, 2, 4) if sequence else (2, 2, 4))
    for index, field in enumerate(fields):
        torch.testing.assert_close(stacked.select(-2, index), by_name[field], rtol=0, atol=0)
    torch.testing.assert_close(stacked.flatten(-2), concatenated, rtol=0, atol=0)

    parameters = tuple(omni.parameters())
    stack_grads = torch.autograd.grad(stacked.square().sum(), parameters)
    concat_grads = torch.autograd.grad(concatenated.square().sum(), parameters)
    for stack_grad, concat_grad in zip(stack_grads, concat_grads):
        torch.testing.assert_close(stack_grad, concat_grad, rtol=0, atol=0)
    if sequence:
        assert torch.count_nonzero(stacked[0, 2]) == 0


def test_stack_uses_existing_source_include_and_exclude_filters():
    omni, interaction = _example()
    selected = EmbView(
        omni,
        target_sources=(FeatureSource.ITEM_ID, FeatureSource.ITEM),
        include_fields=("category", "item"),
        exclude_fields=("item",),
    )
    assert selected.num_fields == 1
    assert selected.field_dim == 4
    output = selected(interaction, split_by="stack")
    assert output.shape == (2, 1, 4)
    torch.testing.assert_close(output[:, 0], selected(interaction))


def test_stack_rejects_mixed_field_widths_without_changing_concat():
    omni, interaction = _example(category_dim=2)
    view = omni.whole
    assert view.embedding_dim == 6
    assert view(interaction).shape == (2, 6)
    with pytest.raises(ValueError, match="equal embedding dimensions"):
        _ = view.field_dim
    with pytest.raises(ValueError, match="matching tensor shapes.*item.*category"):
        view(interaction, split_by="stack")


def test_stack_rejects_incompatible_leading_dimensions():
    omni, interaction = _example()
    mixed = EmbView(omni, include_fields=("item", "slate_item"))
    with pytest.raises(ValueError, match="matching tensor shapes.*slate_item"):
        mixed(interaction, split_by="stack")


def test_stack_handles_arbitrary_leading_dimensions():
    omni, _ = _example()
    first, second = omni.settings[:2]
    left = torch.randn(2, 3, 5, 4)
    right = torch.randn(2, 3, 5, 4)
    output = omni.format_output([(first, left), (second, right)], "stack")
    assert output.shape == (2, 3, 5, 2, 4)
    torch.testing.assert_close(output[..., 0, :], left)
    torch.testing.assert_close(output[..., 1, :], right)


def test_empty_stack_view_keeps_none_output_and_has_no_field_dim():
    omni, interaction = _example()
    view = EmbView(omni, exclude_fields=("item", "category", "slate_item", "slate_category"))
    assert view.num_fields == 0
    assert view(interaction, split_by="stack") is None
    with pytest.raises(ValueError, match="non-empty view"):
        _ = view.field_dim
