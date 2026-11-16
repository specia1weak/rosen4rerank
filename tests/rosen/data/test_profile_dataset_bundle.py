from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import rosen.data.profile as profile_module
from rosen.data import ProfileDatasetBundle, ProfileDatasetLayout


def _make_layout(root: Path) -> ProfileDatasetLayout:
    train_dir = root / "main" / "train.shuffle"
    train_dir.mkdir(parents=True)
    (root / "schema").mkdir()
    (root / "schema" / "feature_meta.json").write_text("{}")
    (root / "profile_seq" / "profiles").mkdir(parents=True)
    (root / "main" / "valid.parquet").touch()
    (root / "main" / "test.parquet").touch()
    manifest = {
        "train_dir": "workspace/old-name/main/train.shuffle",
        "valid_path": "workspace/old-name/main/valid.parquet",
        "test_path": "workspace/old-name/main/test.parquet",
    }
    (root / "main" / "split_manifest.json").write_text(json.dumps(manifest))
    return ProfileDatasetLayout.from_root(root)


def test_layout_resolves_manifest_paths_inside_selected_root(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path / "renamed-dataset")

    assert layout.split_path("train") == layout.root / "main" / "train.shuffle"
    assert layout.split_path("valid") == layout.root / "main" / "valid.parquet"
    assert layout.split_path("test") == layout.root / "main" / "test.parquet"
    layout.validate()


def test_layout_falls_back_to_legacy_single_train_file(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    (root / "main").mkdir(parents=True)
    (root / "main" / "train.global_shuffle.parquet").touch()

    layout = ProfileDatasetLayout.from_root(root)

    assert layout.split_path("train").name == "train.global_shuffle.parquet"


def test_bundle_loader_hides_profile_storage_paths(tmp_path: Path, monkeypatch) -> None:
    layout = _make_layout(tmp_path / "dataset")
    captured = {}

    class FakeProfileDataset:
        def __init__(self, path, manager, **kwargs):
            captured.update(path=path, manager=manager, **kwargs)

        def __iter__(self):
            yield "raw"

    monkeypatch.setattr(profile_module, "ProfileParquetStreamDataset", FakeProfileDataset)
    manager = SimpleNamespace()
    bundle = ProfileDatasetBundle(layout)

    loader = bundle.loader(
        "train",
        manager,
        batch_size=32,
        transform=lambda batch: f"prepared:{batch}",
    )

    assert list(loader) == ["prepared:raw"]
    assert captured["path"] == layout.root / "main" / "train.shuffle"
    assert captured["profile_root"] == layout.root / "profile_seq" / "profiles"
    assert captured["manager"] is manager


def test_bundle_rows_sums_directory_parts(tmp_path: Path) -> None:
    layout = _make_layout(tmp_path / "dataset")
    table = pa.table({"value": [1, 2, 3]})
    pq.write_table(table, layout.split_path("train") / "part-000.parquet")
    pq.write_table(table.slice(0, 2), layout.split_path("train") / "part-001.parquet")

    assert ProfileDatasetBundle(layout).rows("train") == 5


def test_layout_reports_missing_logical_components(tmp_path: Path) -> None:
    layout = ProfileDatasetLayout.from_root(tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="profile store"):
        layout.validate(("train",))


def test_layout_rejects_unknown_split(tmp_path: Path) -> None:
    layout = ProfileDatasetLayout.from_root(tmp_path)

    with pytest.raises(ValueError, match="Unknown profile dataset split"):
        layout.split_path("holdout")
