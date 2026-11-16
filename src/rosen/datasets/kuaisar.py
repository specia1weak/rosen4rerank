from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rosen.datasets.base import DatasetBase


@dataclass(frozen=True)
class KuaiSARPaths:
    """Resolved paths for one extracted KuaiSAR release.

    The adapter deliberately exposes files only. Dataset-specific filtering,
    temporal splitting, and episode construction are experiment decisions and
    belong under ``@examples/`` or ``workspace/``.
    """

    root: Path

    @property
    def rec_interactions(self) -> Path:
        return self.root / "rec_inter.csv"

    @property
    def search_interactions(self) -> Path:
        return self.root / "src_inter.csv"

    @property
    def user_features(self) -> Path:
        return self.root / "user_features.csv"

    @property
    def item_features(self) -> Path:
        return self.root / "item_features.csv"

    @property
    def social_network(self) -> Path:
        return self.root / "social_network.csv"

    def validate(self, required: Iterable[str] = ("rec_interactions",)) -> None:
        """Fail with all missing logical files instead of the first one only."""

        unknown = [name for name in required if not hasattr(type(self), name)]
        if unknown:
            raise ValueError(f"Unknown KuaiSAR path names: {unknown}")

        missing = [
            getattr(self, name)
            for name in required
            if not getattr(self, name).is_file()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "KuaiSAR is not available at the resolved source root. Missing files:\n"
                f"{formatted}\n"
                "Pass an explicit dataset directory or configure "
                "ROSEN_DATASOURCE_ROOT."
            )


class KuaiSARDataset(DatasetBase):
    """Read-only path adapter for the extracted KuaiSAR dataset."""

    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "KuaiSAR" / "KuaiSAR_v2"

    @classmethod
    def paths(cls, root: str | Path | None = None) -> KuaiSARPaths:
        return KuaiSARPaths(
            Path(root).expanduser() if root is not None else cls.BASE_DIR
        )


# Match the existing KuaiRand adapter's convenient class-level file constants.
KuaiSARDataset.REC_INTERACTIONS = KuaiSARDataset.BASE_DIR / "rec_inter.csv"
KuaiSARDataset.SEARCH_INTERACTIONS = KuaiSARDataset.BASE_DIR / "src_inter.csv"
KuaiSARDataset.USER_FEATURES = KuaiSARDataset.BASE_DIR / "user_features.csv"
KuaiSARDataset.ITEM_FEATURES = KuaiSARDataset.BASE_DIR / "item_features.csv"
KuaiSARDataset.SOCIAL_NETWORK = KuaiSARDataset.BASE_DIR / "social_network.csv"
