from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from rosen.utils.observatory.port import ObservationPort


_REPRESENTATION_BOUNDARIES = (
    nn.Linear,
    nn.ReLU,
    nn.PReLU,
    nn.GELU,
    nn.SiLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.LayerNorm,
    nn.BatchNorm1d,
)


@dataclass(frozen=True)
class ModuleProbeSpec:
    module_path: str
    prefix: str
    include_root: bool = True
    leaf_policy: str = "representation"
    role: str = "hidden"
    metrics: tuple[str, ...] = ("basic", "rank")
    domain_routed: bool = False


class ModuleProbeRegistry:
    """Attach opt-in forward probes without changing the observed module."""

    def __init__(self) -> None:
        self._specs: list[ModuleProbeSpec] = []
        self._handles: list[object] = []
        self._attached_model_id: int | None = None
        self._domain_context: dict[str, torch.Tensor] = {}

    def add(self, spec: ModuleProbeSpec) -> None:
        self._specs.append(spec)

    @staticmethod
    def _is_selected(module: nn.Module, policy: str) -> bool:
        if policy == "all_leaves":
            return not any(module.children())
        if policy == "linear":
            return isinstance(module, nn.Linear)
        if policy == "representation":
            return isinstance(module, _REPRESENTATION_BOUNDARIES)
        raise ValueError(f"Unknown module probe leaf_policy={policy!r}.")

    @staticmethod
    def _capture_value(
        port: ObservationPort,
        name: str,
        value,
        *,
        role: str,
        metrics: tuple[str, ...],
        metadata: Mapping[str, object],
    ) -> None:
        if isinstance(value, torch.Tensor):
            if port.spec(name) is None:
                port.register(name, role=role, axes=("batch", "feature"), metrics=metrics, metadata=metadata)
            port.capture(name, value, clone=True)
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                if isinstance(child, torch.Tensor):
                    ModuleProbeRegistry._capture_value(
                        port,
                        f"{name}.{key}",
                        child,
                        role=role,
                        metrics=metrics,
                        metadata={**metadata, "output_key": str(key)},
                    )

    def attach(self, model: nn.Module, port: ObservationPort) -> None:
        if self._attached_model_id == id(model):
            return
        self.close()
        for spec in self._specs:
            module = model.get_submodule(spec.module_path) if spec.module_path else model
            if spec.domain_routed:
                def capture_domain_context(_module, inputs, *, _prefix=spec.prefix):
                    if len(inputs) >= 2 and isinstance(inputs[1], torch.Tensor):
                        self._domain_context[_prefix] = inputs[1].detach().long().view(-1)

                self._handles.append(module.register_forward_pre_hook(capture_domain_context))
            for relative_name, child in module.named_modules():
                if not relative_name or not self._is_selected(child, spec.leaf_policy):
                    continue
                signal_name = f"{spec.prefix}.layers.{relative_name}"
                metadata = {
                    "module_path": f"{spec.module_path}.{relative_name}".strip("."),
                    "module_type": type(child).__name__,
                }

                def capture_leaf(
                    _module,
                    _inputs,
                    output,
                    *,
                    _name=signal_name,
                    _meta=metadata,
                    _spec=spec,
                    _relative_name=relative_name,
                ):
                    selected_output = output
                    selected_meta = _meta
                    if _spec.domain_routed and isinstance(output, torch.Tensor):
                        tokens = _relative_name.split(".")
                        if "domain_networks" in tokens:
                            index = tokens.index("domain_networks")
                            if index + 1 < len(tokens):
                                domain = int(tokens[index + 1])
                                domain_ids = self._domain_context.get(_spec.prefix)
                                if domain_ids is not None and output.ndim > 0 and output.size(0) == domain_ids.numel():
                                    mask = domain_ids.to(output.device) == domain
                                    selected_output = output[mask]
                                    selected_meta = {**_meta, "coordinates": {"domain": domain}}
                    self._capture_value(
                        port,
                        _name,
                        selected_output,
                        role=_spec.role,
                        metrics=_spec.metrics,
                        metadata=selected_meta,
                    )

                self._handles.append(child.register_forward_hook(capture_leaf))

            root_metadata = {
                "module_path": spec.module_path,
                "module_type": type(module).__name__,
                "root_output": True,
            }

            def capture_root(_module, _inputs, output, *, _spec=spec, _meta=root_metadata):
                if _spec.include_root:
                    self._capture_value(
                        port,
                        f"{_spec.prefix}.output",
                        output,
                        role=_spec.role,
                        metrics=_spec.metrics,
                        metadata=_meta,
                    )
                self._domain_context.pop(_spec.prefix, None)

            # A root hook is also needed when include_root=False: it defines
            # the lifetime of domain-routing context and prevents one forward
            # call from retaining the previous batch's domain ids.
            self._handles.append(module.register_forward_hook(capture_root))
        self._attached_model_id = id(model)

    def close(self) -> None:
        for handle in self._handles:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self._handles.clear()
        self._domain_context.clear()
        self._attached_model_id = None


__all__ = ["ModuleProbeRegistry", "ModuleProbeSpec"]
