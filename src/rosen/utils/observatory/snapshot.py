from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SignalSnapshot:
    name: str
    kind: str
    role: str
    scope: str
    metrics: Mapping[str, Any]
    coordinates: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelSnapshot:
    snapshot_id: str
    step: int
    epoch: int
    batch_idx: int
    stage: str
    phase: str
    target: str
    model_mode: str
    signals: Mapping[str, SignalSnapshot]
    interval_start: Optional[int] = None
    interval_end: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["signals"] = {name: signal.to_dict() for name, signal in self.signals.items()}
        return payload


__all__ = ["ModelSnapshot", "SignalSnapshot"]
