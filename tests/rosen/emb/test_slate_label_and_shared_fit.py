from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
import torch

from rosen.core.enum_type import FeatureSource
from rosen.core.interaction import Interaction
from rosen.data.dataset import ParquetStreamDataset, RawParquetStreamDataset
from rosen.data.padding import FeatureContext, PaddedFloatSequenceFormatter, TensorFormatter
from rosen.emb import SchemaManager, SlateLabel
from rosen.emb.emblayer import OmniEmbLayer
from rosen.emb.schema import MinMaxDenseSetting, SeqGroupConfig, SequenceSetting, SparseEmbSetting
from rosen.models.base import BaseModel


@pytest.mark.parametrize("kwargs", [
    {"field_name": ""}, {"field_name": " "}, {"max_len": 0}, {"max_len": -1},
    {"max_len": 2.5}, {"max_len": True}, {"padding_side": "middle"}, {"mask_field": ""},
])
def test_slate_label_rejects_invalid_declarations(kwargs):
    with pytest.raises(ValueError):
        SlateLabel(**({"field_name": "click", "max_len": 3} | kwargs))


@pytest.mark.parametrize("labels, expected", [
    ("scalar", ("scalar",)),
    (SlateLabel("click", 3, "mask"), ("click",)),
    ([SlateLabel("click", 3, "mask"), "scalar", SlateLabel("buy", 3, "mask"), "scalar"],
     ("click", "scalar", "buy")),
    (None, ()),
])
def test_label_fields_remain_strings_and_masks_are_deduplicated(tmp_path, labels, expected):
    manager = SchemaManager([], tmp_path, label_fields=labels)
    assert manager.label_fields == expected
    assert manager.label_field == (expected[0] if expected else None)
    assert manager.get_label_spec("scalar") is None
    assert manager.get_label_spec("unknown") is None
    has_mask = "click" in expected
    assert manager.mask_fields == (("mask",) if has_mask else ())
    assert manager.fields() == [*expected, *manager.mask_fields]
    if has_mask:
        assert manager.get_label_spec("click") == SlateLabel("click", 3, "mask")
        with pytest.raises(FrozenInstanceError):
            manager.get_label_spec("click").max_len = 4


@pytest.mark.parametrize("other", [SlateLabel("buy", 4, "mask"), SlateLabel("buy", 3, "mask", "left")])
def test_shared_label_mask_requires_same_layout(tmp_path, other):
    with pytest.raises(ValueError, match="Shared mask.*max_len and padding_side"):
        SchemaManager([], tmp_path, label_fields=[SlateLabel("click", 3, "mask"), other])


def test_conflicting_duplicate_label_declarations_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Conflicting label"):
        SchemaManager([], tmp_path, label_fields=["click", SlateLabel("click", 3)])


@pytest.mark.parametrize("side", ["left", "right"])
def test_list_labels_and_masks_reach_both_parquet_datasets(tmp_path, side):
    manager = SchemaManager([], tmp_path / "schema", label_fields=[
        SlateLabel("click", 3, "mask", side, pad_val=-1.0), "scalar",
        SlateLabel("buy", 3, "mask", side),
    ])
    frame = pl.DataFrame({
        "click": [[1.0, 0.0], [], None, [0.0, 1.0, 0.0, 1.0]],
        "buy": [[0.0, 1.0], [], None, [1.0, 0.0, 1.0, 0.0]],
        "mask": [[1.0, 1.0], [], None, [1.0, 0.0, 1.0, 1.0]],
        "scalar": [1.0, 0.0, 0.0, 1.0],
        "unused": [1, 2, 3, 4],
    })
    path = tmp_path / "labels.parquet"
    frame.write_parquet(path)
    click = [[1, 0, -1], [-1, -1, -1], [-1, -1, -1], [1, 0, 1]]
    mask = [[1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1]]
    if side == "left":
        click[0], mask[0] = [-1, 1, 0], [0, 1, 1]
    for dataset_type in (ParquetStreamDataset, RawParquetStreamDataset):
        dataset = dataset_type(path, manager, batch_size=4, shuffle=False, drop_last=False)
        batch, = list(dataset)
        assert set(batch.columns) == set(manager.fields())
        torch.testing.assert_close(batch["click"], torch.tensor(click, dtype=torch.float32))
        torch.testing.assert_close(batch["mask"], torch.tensor(mask, dtype=torch.float32))
        assert batch["buy"].shape == (4, 3)
        assert batch["scalar"].shape == (4,)
        assert dataset.scanner.read_cols.count("mask") == 1


def test_tensor_formatter_missing_lists_and_explicit_override(tmp_path):
    manager = SchemaManager([], tmp_path, label_fields=SlateLabel("click", 3, "mask"))
    context = FeatureContext.from_manager(manager)
    batch = {"click": np.array([None, np.nan, []], dtype=object),
             "mask": np.array([None, np.nan, []], dtype=object)}
    formatter = TensorFormatter(context)
    output = formatter.format(batch)
    assert torch.equal(output["click"], torch.zeros(3, 3))
    assert torch.equal(output["mask"], torch.zeros(3, 3))
    override = PaddedFloatSequenceFormatter(2, pad_val=7.0)
    custom = TensorFormatter(FeatureContext.from_manager(
        manager, extra_col_formatters={"click": override},
    ))
    assert custom.col_formatters["click"] is override
    assert torch.equal(custom.format(batch)["click"], torch.full((3, 2), 7.0))


def _shared_manager(directory, kind, placement):
    atom = (SparseEmbSetting("atom", FeatureSource.ITEM, embedding_dim=4)
            if kind == "sparse" else MinMaxDenseSetting("atom", FeatureSource.ITEM))
    current = SequenceSetting("current", atom, SeqGroupConfig("cur", "cur_len", 3))
    history = SequenceSetting("history", atom, SeqGroupConfig("hist", "hist_len", 3))
    settings = [current, history]
    if placement == "first":
        settings.insert(0, atom)
    elif placement == "last":
        settings.append(atom)
    return SchemaManager(settings, directory), atom, current, history


@pytest.mark.parametrize("low_memory", [False, True])
@pytest.mark.parametrize("kind", ["sparse", "dense"])
@pytest.mark.parametrize("placement", ["absent", "first", "last"])
def test_shared_target_fits_first_source_once_and_roundtrips(tmp_path, low_memory, kind, placement):
    manager, atom, current, history = _shared_manager(tmp_path, kind, placement)
    raw = (pl.DataFrame({"atom": ["top", "top"], "current": [["a", "b"], ["b"]],
                         "history": [["history_only"], ["a"]]}) if kind == "sparse"
           else pl.DataFrame({"atom": [10.0, 20.0], "current": [[1.0, 3.0], [2.0]],
                              "history": [[-20.0], [50.0]]})).lazy()
    # Count real calls rather than relying on fitted flags alone.
    for setting in manager.settings:
        setting.get_fit_exprs = Mock(wraps=setting.get_fit_exprs)
        setting.parse_fit_result = Mock(wraps=setting.parse_fit_result)
    manager.fit(raw, low_memory=low_memory)
    assert sum(s.get_fit_exprs.call_count for s in manager.settings) == 1
    assert sum(s.parse_fit_result.call_count for s in manager.settings) == 1
    assert atom.is_fitted and current.is_fitted and history.is_fitted
    if kind == "sparse":
        assert set(atom.vocab) == ({"top"} if placement == "first" else {"a", "b"})
    else:
        assert (atom.min_val, atom.max_val) == ((10.0, 20.0) if placement == "first" else (1.0, 3.0))
    encoded = manager.transform(raw).collect()
    omni = OmniEmbLayer(manager=manager)
    assert len(omni.emb_modules) == (1 if kind == "sparse" else 0)
    assert current.embedding_field_name == history.embedding_field_name == atom.embedding_field_name
    manager.save_schema()

    restored, restored_atom, restored_cur, restored_hist = _shared_manager(tmp_path, kind, placement)
    restored.load_schema()
    assert restored_atom.is_fitted and restored_cur.is_fitted and restored_hist.is_fitted
    assert restored_cur.element_setting is restored_hist.element_setting is restored_atom
    assert restored_atom.to_dict() == atom.to_dict()
    assert restored.transform(raw).collect().equals(encoded)
    if kind == "sparse":
        input_tensor = torch.tensor([[1, 1, 0]])
        inter = Interaction({"current": input_tensor, "history": input_tensor})
        outputs = omni(inter, split_by="name", include_fields=("current", "history"))
        torch.testing.assert_close(outputs["current"], outputs["history"], rtol=0, atol=0)
        assert len(OmniEmbLayer(manager=restored).emb_modules) == 1


def test_prefitted_target_and_missing_wrapper_metadata_sync(tmp_path):
    manager, atom, current, history = _shared_manager(tmp_path, "sparse", "absent")
    atom._build_vocab_indices(["a"])
    manager.fit(pl.DataFrame({}).lazy())
    assert current.is_fitted and history.is_fitted
    # A runtime schema may add another wrapper around the loaded target.
    restored, atom2, cur2, hist2 = _shared_manager(tmp_path, "sparse", "absent")
    extra = SequenceSetting("new_history", atom2, SeqGroupConfig("new", "new_len", 3))
    restored.settings.append(extra)
    restored.fit(pl.DataFrame({}).lazy())  # existing feature_meta.json path
    assert atom2.is_fitted and cur2.is_fitted and hist2.is_fitted and extra.is_fitted


def test_base_model_label_properties_and_mask_fallback(tmp_path):
    manager = SchemaManager([], tmp_path, label_fields=[SlateLabel("click", 3, "mask"), "scalar"])
    model = BaseModel(manager)
    assert (model.LABEL, model.LABELS, model.DOMAIN) == ("click", ("click", "scalar"), None)
    for device in (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]):
        labels = torch.zeros(2, 3, device=device)
        mask = torch.tensor([[1, 0, 0], [1, 1, 0]], device=device)
        inter = Interaction({"click": labels, "mask": mask, "scalar": torch.ones(2, device=device)})
        actual = model.get_label_mask(inter, dtype=torch.bool)
        assert actual.dtype == torch.bool and actual.device == labels.device
        assert torch.equal(actual, mask.bool())
        scalar_mask = model.get_label_mask(inter, "scalar", dtype=torch.float16)
        assert scalar_mask.shape == (2,) and scalar_mask.device == labels.device
        assert scalar_mask.dtype == torch.float16 and scalar_mask.sum() == 2
        inter.drop("mask")
        fallback = model.get_label_mask(inter)
        assert fallback.shape == labels.shape and fallback.device == labels.device
        assert fallback.dtype == torch.float32 and fallback.sum() == 6
    empty = BaseModel(SchemaManager([], tmp_path / "empty"))
    with pytest.raises(ValueError, match="requires a label"):
        empty.get_label_mask(Interaction({"x": torch.ones(2)}))
