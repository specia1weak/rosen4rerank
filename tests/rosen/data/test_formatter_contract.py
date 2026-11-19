import pickle
import subprocess
import sys
from dataclasses import FrozenInstanceError

import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from rosen.core.enum_type import FeatureSource
from rosen.core.formatters import IntFormatter, PaddedFloatSequenceFormatter
from rosen.data.dataset import ParquetStreamDataset, RawParquetStreamDataset
from rosen.data.padding import FeatureContext, TensorFormatter
from rosen.emb import AuxField, SchemaManager, SlateLabel
from rosen.emb.schema import (
    MinMaxDenseSetting, MultiSparseSetting, SeqGroupConfig, SequenceSetting, SparseEmbSetting,
)


@pytest.mark.parametrize("mode", ["head", "tail"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("kind", ["sparse", "dense", "multi"])
def test_formatter_and_polars_agree_on_sequence_truncation(tmp_path, mode, side, kind):
    if kind == "dense":
        atom = MinMaxDenseSetting("atom", FeatureSource.ITEM)
        atom.min_val, atom.max_val, atom.is_fitted = 0.0, 10.0, True
        raw_values = [[1., 2., 3., 4.], [5.], []]
        encoded_values = [[.1, .2, .3, .4], [.5], []]
    else:
        atom = (SparseEmbSetting("atom", FeatureSource.ITEM) if kind == "sparse"
                else MultiSparseSetting("atom", FeatureSource.ITEM, max_tag_len=2))
        atom._build_vocab_indices([str(i) for i in range(1, 7)])
        raw_values = ([[1, 2, 3, 4], [5], []] if kind == "sparse"
                      else [[[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], [[5]], []])
        encoded_values = raw_values  # Fitted vocab IDs match these integers.
    seq = SequenceSetting("history", atom, SeqGroupConfig(
        "hist", "history_len", 2, side, time_field_name="history_time",
    ), truncate_mode=mode)
    manager = SchemaManager([seq], tmp_path / "schema")
    raw = pl.DataFrame({"history": raw_values, "history_time": [[10, 20, 30, 40], [50], []]})
    encoded = manager.transform(raw.lazy()).collect()
    direct = TensorFormatter(FeatureContext.from_manager(manager)).format({
        "history": encoded_values,
        "history_time": raw["history_time"].to_numpy(),
        "history_len": [2, 1, 0],
    })
    raw_path, encoded_path = tmp_path / "raw.parquet", tmp_path / "encoded.parquet"
    raw.write_parquet(raw_path)
    encoded.write_parquet(encoded_path)
    for dataset_cls, path in [(RawParquetStreamDataset, raw_path), (ParquetStreamDataset, encoded_path)]:
        dataset = dataset_cls(path, manager, batch_size=3, shuffle=False, drop_last=False)
        batch, = list(dataset)
        for col in manager.fields():
            torch.testing.assert_close(batch[col], direct[col])
    selected = ([10, 20] if mode == "head" else [30, 40])
    assert direct["history_time"][0].tolist() == selected
    if kind == "multi":
        assert direct["history"][0].tolist() == ([[1, 2], [2, 3]] if mode == "head"
                                                  else [[3, 4], [4, 5]])


def test_multisparse_formatter_keeps_head_like_polars(tmp_path):
    setting = MultiSparseSetting("tags", FeatureSource.ITEM, max_tag_len=2)
    setting._build_vocab_indices(["1", "2", "3"])
    manager = SchemaManager([setting], tmp_path)
    encoded = manager.transform(pl.DataFrame({"tags": [[1, 2, 3], [2], []]}).lazy()).collect()
    formatter = setting.get_formatters()["tags"]
    assert formatter.format([[1, 2, 3], [2], []]).tolist() == [[1, 2], [2, 0], [0, 0]]
    torch.testing.assert_close(formatter.format(encoded["tags"].to_numpy()),
                               formatter.format([[1, 2, 3], [2], []]))


def test_manager_field_queries_and_formatter_precedence(tmp_path):
    atom = SparseEmbSetting("atom", FeatureSource.ITEM)
    mask = SparseEmbSetting("mask", FeatureSource.ITEM)
    seq = SequenceSetting("history", atom, SeqGroupConfig("hist", "length", 2))
    manager = SchemaManager([seq, mask], tmp_path, time_field="time", domain_fields="domain",
                            label_fields=[SlateLabel("click", 2, "mask"),
                                          SlateLabel("mask", 2, pad_val=-1.)],
                            aux_fields=AuxField("request", 2, "int64", pad_val=-2.))
    assert manager.fields() == ["history", "length", "mask", "time", "click", "domain", "request"]
    assert manager.raw_fields() == ["history", "mask", "request", "click", "domain", "time"]
    assert manager.fields(include_labels=False) == ["history", "length", "time", "domain", "request"]
    assert manager.raw_fields(include_labels=False) == ["history", "request", "domain", "time"]
    formatters = manager.get_formatters()
    assert formatters["mask"] == PaddedFloatSequenceFormatter(2, pad_val=-1.)
    assert type(formatters["request"].pad_val) is int
    # An existing feature's mask conversion wins over the label's mask default.
    manager = SchemaManager([mask], tmp_path, label_fields=SlateLabel("click", 2, "mask"))
    assert isinstance(manager.get_formatters()["mask"], IntFormatter)


def test_context_snapshots_extras_and_roundtrips_through_spawn(tmp_path):
    manager = SchemaManager([], tmp_path / "schema", aux_fields=AuxField("request", dtype="int64"))
    extras = {"extra": IntFormatter()}
    names = ["request", "extra"]
    context = FeatureContext(manager, names, names, extras)
    extras.clear()
    names.clear()
    assert context.read_col_names == ("request", "extra")
    assert "extra" in context.extra_col_formatters
    with pytest.raises(TypeError):
        context.extra_col_formatters["extra"] = IntFormatter()
    with pytest.raises(FrozenInstanceError):
        context.read_col_names = ()
    restored = pickle.loads(pickle.dumps(context))
    assert restored.extra_col_formatters == context.extra_col_formatters
    with pytest.raises(TypeError):
        restored.extra_col_formatters["other"] = IntFormatter()
    path = tmp_path / "data.parquet"
    pl.DataFrame({"request": [1, 2], "extra": [3, 4]}).write_parquet(path)
    dataset = ParquetStreamDataset(path, manager, extra_col_formatters=dict(context.extra_col_formatters),
                                   batch_size=2, shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=None, num_workers=1, multiprocessing_context="spawn")
    batch, = list(loader)
    assert batch["extra"].tolist() == [3, 4]


@pytest.mark.parametrize("modules", [
    ["rosen.data.padding", "rosen.emb"], ["rosen.emb", "rosen.data.padding"],
    ["rosen.core.formatters", "rosen.emb.schema.base", "rosen.data.padding"],
])
def test_import_order_and_resolved_annotations(modules):
    script = "\n".join(f"import {module}" for module in modules)
    script += "\nfrom typing import get_type_hints\n"
    script += "from rosen.emb.schema import EmbSetting, SequenceSetting\n"
    script += "from rosen.core.formatters import IntFormatter\n"
    script += "from rosen.data.padding import IntFormatter as Reexported\n"
    script += "assert Reexported is IntFormatter\n"
    script += "get_type_hints(EmbSetting.get_formatters)\nget_type_hints(SequenceSetting.get_formatters)\n"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
