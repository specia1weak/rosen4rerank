from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Optional

from rosen.utils.observatory.snapshot import ModelSnapshot


class SnapshotStore:
    """Capacity-bounded history whose eviction unit is a complete snapshot."""

    def __init__(self, capacity: int = 1000):
        self.capacity = max(1, int(capacity))
        self._snapshots: deque[ModelSnapshot] = deque(maxlen=self.capacity)

    def commit(self, snapshot: ModelSnapshot) -> ModelSnapshot:
        self._snapshots.append(snapshot)
        return snapshot

    def latest(self) -> Optional[ModelSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def list(
        self,
        *,
        stage: str | None = None,
        phase: str | None = None,
        target: str | None = None,
    ) -> list[ModelSnapshot]:
        rows: Iterable[ModelSnapshot] = self._snapshots
        if stage is not None:
            rows = (row for row in rows if row.stage == stage)
        if phase is not None:
            rows = (row for row in rows if row.phase == phase)
        if target is not None:
            rows = (row for row in rows if row.target == target)
        return list(rows)

    def history(
        self,
        signal_name: str,
        metric: str,
        *,
        stage: str | None = None,
        phase: str | None = None,
        target: str | None = None,
    ) -> list[tuple[int, Any]]:
        rows = []
        for snapshot in self.list(stage=stage, phase=phase, target=target):
            signal = snapshot.signals.get(str(signal_name))
            if signal is None or metric not in signal.metrics:
                continue
            rows.append((snapshot.step, signal.metrics[metric]))
        return rows

    def __len__(self) -> int:
        return len(self._snapshots)


__all__ = ["SnapshotStore"]
