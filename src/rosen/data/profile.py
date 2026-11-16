from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager

from .dataset import ProfileParquetStreamDataset


_SPLIT_MANIFEST_KEYS = {
    "train": "train_dir",
    "valid": "valid_path",
    "test": "test_path",
}
_SPLIT_FALLBACK_NAMES = {
    "train": ("train.shuffle", "train.parquet", "train.global_shuffle.parquet"),
    "valid": ("valid.parquet",),
    "test": ("test.parquet",),
}


@dataclass(frozen=True)
class ProfileDatasetLayout:
    """Resolve the physical layout of one prepared profile dataset."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProfileDatasetLayout":
        return cls(Path(root).expanduser())

    @property
    def main_dir(self) -> Path:
        return self.root / "main"

    @property
    def manifest_path(self) -> Path:
        return self.main_dir / "split_manifest.json"

    @property
    def schema_dir(self) -> Path:
        return self.root / "schema"

    @property
    def schema_path(self) -> Path:
        return self.schema_dir / "feature_meta.json"

    @property
    def profile_root(self) -> Path:
        return self.root / "profile_seq" / "profiles"

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def split_path(self, split: str) -> Path:
        if split not in _SPLIT_MANIFEST_KEYS:
            raise ValueError(
                f"Unknown profile dataset split {split!r}; "
                f"expected one of {sorted(_SPLIT_MANIFEST_KEYS)}."
            )

        manifest_value = self._manifest().get(_SPLIT_MANIFEST_KEYS[split])
        if manifest_value:
            local_path = self.main_dir / Path(manifest_value).name
            if local_path.exists():
                return local_path

            manifest_path = Path(manifest_value).expanduser()
            if manifest_path.exists():
                return manifest_path

        candidates = [
            self.main_dir / name for name in _SPLIT_FALLBACK_NAMES[split]
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def validate(self, splits: Sequence[str] = ("train", "valid", "test")) -> None:
        required = {
            "dataset root": self.root,
            "schema": self.schema_path,
            "profile store": self.profile_root,
            **{f"{split} split": self.split_path(split) for split in splits},
        }
        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not path.exists()
        ]
        if missing:
            formatted = "\n".join(f"  - {entry}" for entry in missing)
            raise FileNotFoundError(
                f"Profile dataset is incomplete under {self.root}:\n{formatted}"
            )


class _MappedInteractionLoader(Iterable[Interaction]):
    def __init__(
        self,
        source: Iterable[Interaction],
        transform: Callable[[Interaction], Interaction],
    ) -> None:
        self.source = source
        self.transform = transform

    def __iter__(self) -> Iterator[Interaction]:
        for interaction in self.source:
            yield self.transform(interaction)


def _identity_collate(batch):
    return batch


class ProfileDatasetBundle:
    """Load schema and logical splits without exposing storage subdirectories."""

    def __init__(self, layout: ProfileDatasetLayout) -> None:
        self.layout = layout

    @classmethod
    def open(cls, root: str | Path) -> "ProfileDatasetBundle":
        return cls(ProfileDatasetLayout.from_root(root))

    @property
    def root(self) -> Path:
        return self.layout.root

    def load_manager(
        self,
        settings,
        *,
        time_field: str,
        label_fields: str | Sequence[str],
        domain_fields: str | Sequence[str],
    ) -> SchemaManager:
        self.layout.validate(splits=())
        manager = SchemaManager(
            settings,
            str(self.layout.schema_dir),
            time_field=time_field,
            label_fields=label_fields,
            domain_fields=domain_fields,
        )
        manager.load_schema()
        return manager

    def loader(
        self,
        split: str,
        manager: SchemaManager,
        *,
        batch_size: int = 4096,
        shuffle: bool = False,
        drop_last: bool = False,
        shuffle_buffer_size: int = 2_000_000,
        num_workers: int = 0,
        prefetch_factor: int = 1,
        transform: Callable[[Interaction], Interaction] | None = None,
    ) -> Iterable[Interaction]:
        self.layout.validate(splits=(split,))
        source: Iterable[Interaction] = ProfileParquetStreamDataset(
            self.layout.split_path(split),
            manager,
            profile_root=self.layout.profile_root,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            shuffle_buffer_size=shuffle_buffer_size,
        )

        worker_count = max(0, int(num_workers))
        if worker_count > 0:
            kwargs: dict[str, Any] = {
                "batch_size": None,
                "num_workers": worker_count,
                "collate_fn": _identity_collate,
            }
            if prefetch_factor > 0:
                kwargs["prefetch_factor"] = int(prefetch_factor)
            source = DataLoader(source, **kwargs)

        if transform is not None:
            source = _MappedInteractionLoader(source, transform)
        return source

    def rows(self, split: str) -> int:
        path = self.layout.split_path(split)
        parquet_paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        if not parquet_paths or any(not parquet_path.is_file() for parquet_path in parquet_paths):
            raise FileNotFoundError(path)
        return sum(
            pq.ParquetFile(parquet_path).metadata.num_rows
            for parquet_path in parquet_paths
        )
