from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Set, Tuple, Union

import torch


@dataclass(frozen=True)
class AxisSpec:
    name: str
    kind: str = "generic"
    labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationSpec:
    role: str = "tensor"
    axes: Tuple[AxisSpec, ...] = ()
    metrics: Tuple[str, ...] = ("basic", "spectral")
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CapturedObservation:
    name: str
    value: Any
    namespace: str = ""

    @property
    def full_name(self) -> str:
        if not self.namespace:
            return self.name
        return f"{self.namespace}.{self.name}"


class ObservationPort:
    """
    Lightweight per-model trace buffer.

    The port is intentionally not a recorder: it does not compute metrics,
    print reports, or export artifacts. Models use it to mark interesting
    forward-pass tensors, and an external observatory can decide when to
    enable collection and how to interpret the captured trace.
    """

    def __init__(self, namespace: str = "", enabled: bool = False):
        self.namespace = str(namespace)
        self.enabled = bool(enabled)
        self._items: Dict[str, CapturedObservation] = {}
        self._specs: Dict[str, ObservationSpec] = {}
        self._cut_names: Set[str] = set()
        self._watch_names: Set[str] = set()
        self._active_cut: Optional[str] = None
        self._watch_enabled: bool = False
        self._watch_items: Dict[str, Any] = {}
        self._gradient_tap_names: Set[str] = set()
        self._gradient_tap_enabled: bool = False
        self._gradient_items: Dict[str, torch.Tensor] = {}
        self._gradient_handles: list[Any] = []

    @staticmethod
    def _normalize_axes(axes: Iterable[Union[str, AxisSpec]]) -> Tuple[AxisSpec, ...]:
        return tuple(axis if isinstance(axis, AxisSpec) else AxisSpec(name=str(axis), kind=str(axis)) for axis in axes)

    def register(
            self,
            name: str,
            spec: Optional[ObservationSpec] = None,
            *,
            role: str = "tensor",
            axes: Iterable[Union[str, AxisSpec]] = (),
            metrics: Iterable[str] = ("basic", "spectral"),
            metadata: Optional[Mapping[str, Any]] = None,
            overwrite: bool = False,
    ) -> ObservationSpec:
        key = str(name)
        if key in self._specs and not overwrite:
            raise ValueError(f"Observation {key!r} is already registered.")
        if spec is None:
            spec = ObservationSpec(
                role=str(role),
                axes=self._normalize_axes(axes),
                metrics=tuple(str(metric) for metric in metrics),
                metadata=dict(metadata or {}),
            )
        self._specs[key] = spec
        return spec

    def spec(self, name: str) -> Optional[ObservationSpec]:
        return self._specs.get(str(name))

    def clear(self) -> None:
        self._items.clear()

    def cut(self, name: str, value: Any) -> Any:
        """
        Mark a tensor as a one-at-a-time ablation point.

        In normal execution this is an identity function. When a matching cut
        name is active, tensor values are replaced by zeros at this exact point
        in the model's forward path.
        """
        key = str(name)
        self._cut_names.add(key)
        if self._active_cut != key:
            return value
        if isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
        return value

    def watch(
            self,
            name: str,
            value: Any,
            *,
            detach: bool = True,
            clone: bool = False,
    ) -> Any:
        """
        Mark a downstream tensor to summarize during cut ablation search.
        """
        key = str(name)
        self._watch_names.add(key)
        if not self._watch_enabled:
            return value

        stored = value
        if isinstance(value, torch.Tensor):
            if detach:
                stored = value.detach()
            if clone:
                stored = stored.clone()
        self._watch_items[key] = stored
        return value

    def tap_gradient(
            self,
            name: str,
            value: Any,
            *,
            clone: bool = True,
    ) -> Any:
        """
        Mark a tensor as a backward-gradient tap.

        In normal execution this is an identity function. When gradient tapping
        is enabled, a backward hook stores the tensor's gradient under `name`.
        """
        key = str(name)
        self._gradient_tap_names.add(key)
        if not self._gradient_tap_enabled:
            return value
        if not isinstance(value, torch.Tensor) or not value.requires_grad:
            return value

        def store_gradient(grad: torch.Tensor) -> None:
            stored = grad.detach()
            if clone:
                stored = stored.clone()
            self._gradient_items[key] = stored

        self._gradient_handles.append(value.register_hook(store_gradient))
        return value

    def observe(
            self,
            name: str,
            value: Any,
            *,
            role: str = "tensor",
            axes: Iterable[Union[str, AxisSpec]] = (),
            metrics: Iterable[str] = ("basic", "rank"),
            metadata: Optional[Mapping[str, Any]] = None,
            watch: bool = False,
            gradient: bool = False,
            detach: bool = True,
            clone: bool = False,
            gradient_clone: bool = True,
    ) -> Any:
        """
        Declare one stable signal and participate in every active observation
        mode. Outside a collection/watch/gradient scope this remains a cheap
        identity operation.
        """
        key = str(name)
        if self.spec(key) is None:
            self.register(
                key,
                role=role,
                axes=axes,
                metrics=metrics,
                metadata=metadata,
            )
        self.capture(key, value, detach=detach, clone=clone)
        if watch:
            self.watch(key, value, detach=detach, clone=clone)
        if gradient:
            self.tap_gradient(key, value, clone=gradient_clone)
        return value

    @contextmanager
    def cutting(self, name: Optional[str]) -> Iterator["ObservationPort"]:
        previous = self._active_cut
        self._active_cut = None if name is None else str(name)
        try:
            yield self
        finally:
            self._active_cut = previous

    @contextmanager
    def watching(self, enabled: bool = True, clear: bool = True) -> Iterator["ObservationPort"]:
        previous = self._watch_enabled
        if clear:
            self._watch_items.clear()
        self._watch_enabled = bool(enabled)
        try:
            yield self
        finally:
            self._watch_enabled = previous

    @contextmanager
    def tapping_gradients(self, enabled: bool = True, clear: bool = True) -> Iterator["ObservationPort"]:
        previous = self._gradient_tap_enabled
        if clear:
            self.clear_gradients()
        self._gradient_tap_enabled = bool(enabled)
        try:
            yield self
        finally:
            self._gradient_tap_enabled = previous

    def cut_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._cut_names))

    def watch_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._watch_names))

    def gradient_tap_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._gradient_tap_names))

    def clear_cut_watch(self) -> None:
        self._cut_names.clear()
        self._watch_names.clear()
        self._watch_items.clear()
        self._active_cut = None

    def clear_gradients(self) -> None:
        self._gradient_items.clear()
        for handle in self._gradient_handles:
            handle.remove()
        self._gradient_handles.clear()

    def watch_snapshot(self, *, clear: bool = False) -> Dict[str, Any]:
        result = dict(self._watch_items)
        if clear:
            self._watch_items.clear()
        return result

    def gradient_snapshot(self, *, clear: bool = False) -> Dict[str, torch.Tensor]:
        result = dict(self._gradient_items)
        if clear:
            self.clear_gradients()
        return result

    def capture(
            self,
            name: str,
            value: Any,
            *,
            detach: bool = True,
            clone: bool = False,
    ) -> Any:
        if not self.enabled:
            return value

        stored = value
        if isinstance(value, torch.Tensor):
            if detach:
                stored = value.detach()
            if clone:
                stored = stored.clone()

        key = str(name)
        self._items[key] = CapturedObservation(
            name=key,
            value=stored,
            namespace=self.namespace,
        )
        return value

    def capture_many(
            self,
            values: Mapping[str, Any],
            *,
            detach: bool = True,
            clone: bool = False,
    ) -> Mapping[str, Any]:
        if not self.enabled:
            return values
        for name, value in values.items():
            self.capture(name, value, detach=detach, clone=clone)
        return values

    @contextmanager
    def collecting(self, enabled: bool = True, clear: bool = True) -> Iterator["ObservationPort"]:
        previous_enabled = self.enabled
        if clear:
            self.clear()
        self.enabled = bool(enabled)
        try:
            yield self
        finally:
            self.enabled = previous_enabled

    def snapshot(self, *, values_only: bool = True, clear: bool = False) -> Dict[str, Any]:
        if values_only:
            result = {name: item.value for name, item in self._items.items()}
        else:
            result = dict(self._items)
        if clear:
            self.clear()
        return result

    def summary(self, *, clear: bool = False) -> Dict[str, CapturedObservation]:
        """Return the observations captured for the current sampled step."""

        return self.snapshot(values_only=False, clear=clear)

    def clear_summary(self) -> None:
        self.clear()

    def __len__(self) -> int:
        return len(self._items)
