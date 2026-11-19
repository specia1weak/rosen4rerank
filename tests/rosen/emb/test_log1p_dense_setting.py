import json
import math

import numpy as np
import polars as pl
import pytest
import torch

from rosen.core.enum_type import FeatureSource
from rosen.data import ParquetStreamDataset, RawParquetStreamDataset
from rosen.data.padding import FeatureContext, TensorFormatter
from rosen.emb import SchemaManager
from rosen.emb.emblayer import OmniEmbLayer
from rosen.emb.schema import Log1PDenseSetting, MinMaxDenseSetting, SeqGroupConfig, SequenceSetting, SparseEmbSetting
from rosen.models.msr.automtl.model import AutoMTLModel
from rosen.models.msr.hamur.common import infer_widedeep_fields
from rosen.models.msr.hamur.model import HAMURModel


def test_log1p_matches_math_and_preserves_small_and_large_inputs():
    setting = Log1PDenseSetting("value", FeatureSource.ITEM)
    values = [0., 1e-12, 1., 9., 1e100, -.5, None]
    output = pl.DataFrame({"value": values}).select(setting.get_transform_expr())
    expected = np.array([math.log1p(x) if x is not None else 0. for x in values], dtype=np.float32)
    assert output["value"].dtype == pl.Float32
    np.testing.assert_allclose(output["value"].to_numpy(), expected, rtol=1e-6, atol=0.)
    assert setting.is_fitted and not setting.requires_embedding_module
    assert setting.get_fit_exprs() == []
    assert setting.get_element_fit_exprs(pl.col("value"), "value") == []


def test_log1p_retains_mathematical_domain_errors():
    setting = Log1PDenseSetting("value", FeatureSource.ITEM)
    output = pl.DataFrame({"value": [-1., -2., float("nan"), float("inf")]}).select(
        setting.get_transform_expr(),
    )["value"].to_numpy()
    assert np.isneginf(output[0])
    assert np.isnan(output[1:3]).all()
    assert np.isposinf(output[3])


@pytest.mark.parametrize("low_memory", [False, True])
def test_log1p_stateless_fit_and_schema_roundtrip(tmp_path, low_memory):
    setting = Log1PDenseSetting("value", FeatureSource.ITEM)
    manager = SchemaManager([setting], tmp_path)
    # A missing parquet source proves fit does not scan data for this setting.
    manager.fit(pl.scan_parquet(tmp_path / "not-needed.parquet"), low_memory=low_memory)
    state, = json.loads(manager.meta_filepath.read_text())
    assert state["type"] == "LOG1P_DENSE"
    assert Log1PDenseSetting.from_dict(state).to_dict() == setting.to_dict()
    restored = SchemaManager([Log1PDenseSetting("value", FeatureSource.ITEM)], tmp_path)
    restored.load_schema()
    raw = pl.DataFrame({"value": [0., 3., 99.]})
    assert restored.transform(raw.lazy()).collect().equals(manager.transform(raw.lazy()).collect())
    with pytest.raises(ValueError, match="类型不匹配"):
        SchemaManager([MinMaxDenseSetting("value", FeatureSource.ITEM)], tmp_path).load_schema()


@pytest.mark.parametrize("mode, side", [("head", "right"), ("tail", "left")])
def test_log1p_scalar_and_sequence_reach_dataset_and_embedding(tmp_path, mode, side):
    atom = Log1PDenseSetting("value", FeatureSource.ITEM)
    sequence = SequenceSetting("history", atom, SeqGroupConfig("hist", "length", 2, side),
                               truncate_mode=mode)
    manager = SchemaManager([atom, sequence], tmp_path / "schema")
    raw = pl.DataFrame({"value": [3., None, 0.], "history": [[0., 3., 7.], [3.], []]})
    manager.fit(raw.lazy())
    assert sequence.is_fitted
    encoded = manager.transform(raw.lazy()).collect()
    raw_path, encoded_path = tmp_path / "raw.parquet", tmp_path / "encoded.parquet"
    raw.write_parquet(raw_path)
    encoded.write_parquet(encoded_path)
    expected_history = ([[0., math.log(4)], [math.log(4), 0.], [0., 0.]] if mode == "head"
                        else [[math.log(4), math.log(8)], [0., math.log(4)], [0., 0.]])
    for dataset_cls, path in [(RawParquetStreamDataset, raw_path), (ParquetStreamDataset, encoded_path)]:
        batch, = list(dataset_cls(path, manager, batch_size=3, shuffle=False, drop_last=False))
        torch.testing.assert_close(batch["value"], torch.tensor([math.log(4), 0., 0.]))
        torch.testing.assert_close(batch["history"], torch.tensor(expected_history))
        embedding = OmniEmbLayer(manager=manager)
        assert len(embedding.emb_modules) == 0
        fields = embedding(batch, split_by="name", include_fields=["value", "history"])
        assert fields["value"].shape == (3, 1)
        assert fields["history"].shape == (3, 2, 1)
        torch.testing.assert_close(fields["history"].squeeze(-1), batch["history"])
    restored_atom = Log1PDenseSetting("value", FeatureSource.ITEM)
    restored_seq = SequenceSetting("history", restored_atom, SeqGroupConfig("hist", "length", 2, side),
                                   truncate_mode=mode)
    restored = SchemaManager([restored_atom, restored_seq], tmp_path / "schema")
    restored.load_schema()
    assert restored_seq.element_setting is restored_atom
    assert restored.transform(raw.lazy()).collect().equals(encoded)


def test_models_recognize_log1p_as_a_dense_feature(tmp_path):
    item = SparseEmbSetting("item", FeatureSource.ITEM, embedding_dim=4)
    item._build_vocab_indices(["1", "2"])
    manager = SchemaManager([item, Log1PDenseSetting("value", FeatureSource.ITEM)], tmp_path,
                            domain_fields="domain", label_fields="label")
    assert infer_widedeep_fields(manager) == (("value",), ("item",))
    frame = manager.transform(pl.DataFrame({"item": [1, 2, 1, 2], "value": [0., 3., 7., 15.],
                                           "domain": [0, 0, 1, 1], "label": [0., 1., 0., 1.]}).lazy()).collect()
    batch = TensorFormatter(FeatureContext.from_manager(manager)).format({
        col: frame[col].to_numpy() for col in manager.fields()
    })
    automtl = AutoMTLModel(manager, num_domains=2, num_experts=2, num_expert_layers=1,
                           expert_num_layers=1, expert_in_features=4, expert_out_features=4,
                           expert_candidate_ops=("Identity", "MLP-16"), tower_hidden_dims=(4,))
    assert automtl.dense_field_names == ["value"] and automtl.dense_input_dim == 1
    assert automtl.sparse_field_names == ["item"]
    hamur = HAMURModel(manager, num_domains=2, backbone="widedeep", use_adapter=False, mlp_dims=(4,))
    assert hamur.wide_dim == 1 and hamur.deep_dim == 4
    for model in (automtl, hamur):
        model.eval()
        logits = model.logits(batch)
        assert logits.shape == (4,) and torch.isfinite(logits).all()
