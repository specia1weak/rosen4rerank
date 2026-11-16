from __future__ import annotations

from rosen.core.enum_type import FeatureSource
from rosen.emb import SchemaManager
from rosen.emb.schema import SeqGroupConfig, SequenceSetting, SparseEmbSetting


def _base_settings():
    item = SparseEmbSetting("item_id", FeatureSource.ITEM_ID, 8)
    tab = SparseEmbSetting("tab", FeatureSource.INTERACTION, 8)
    group = SeqGroupConfig("history", "seq_len", max_len=10)
    return item, tab, group


def test_runtime_manager_can_load_sequence_subset_from_full_schema(tmp_path) -> None:
    item, tab, group = _base_settings()
    item.is_fitted = True
    item.vocab = {"item": 1}
    tab.is_fitted = True
    tab.vocab = {"tab": 0}
    full_manager = SchemaManager(
        [
            item,
            tab,
            SequenceSetting("item_id_seq", item, group),
            SequenceSetting("tab_seq", tab, group),
        ],
        tmp_path,
    )
    full_manager.save_schema()

    runtime_item, runtime_tab, runtime_group = _base_settings()
    runtime_manager = SchemaManager(
        [
            runtime_item,
            runtime_tab,
            SequenceSetting("tab_seq", runtime_tab, runtime_group),
        ],
        tmp_path,
    )
    runtime_manager.load_schema()

    sequence_fields = [
        setting.field_name
        for setting in runtime_manager.settings
        if setting.is_sequence_setting
    ]
    assert sequence_fields == ["tab_seq"]
    assert "item_id_seq" not in runtime_manager.fields()
    assert runtime_manager.get_setting("tab_seq").is_fitted
