from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import torch


ForwardTraceFn = Callable[[torch.Tensor], Mapping[str, torch.Tensor]]


@dataclass
class FieldZeroingResult:
    field_names: Tuple[str, ...]
    target_names: Tuple[str, ...]
    abs_delta: Dict[str, torch.Tensor]
    signed_delta: Dict[str, torch.Tensor]
    relative_abs_delta: Dict[str, torch.Tensor]
    by_domain_abs_delta: Dict[str, torch.Tensor] = field(default_factory=dict)
    by_domain_signed_delta: Dict[str, torch.Tensor] = field(default_factory=dict)
    domain_labels: Tuple[str, ...] = ()

    def metric(self, name: str) -> Dict[str, torch.Tensor]:
        if name == "abs_delta":
            return self.abs_delta
        if name == "signed_delta":
            return self.signed_delta
        if name == "relative_abs_delta":
            return self.relative_abs_delta
        if name == "by_domain_abs_delta":
            return self.by_domain_abs_delta
        if name == "by_domain_signed_delta":
            return self.by_domain_signed_delta
        raise KeyError(f"Unknown field-zeroing metric: {name!r}")

    def target_matrix(self, metric: str = "abs_delta") -> torch.Tensor:
        values = self.metric(metric)
        rows = [values[target].view(-1) for target in self.target_names]
        if not rows:
            return torch.zeros((len(self.field_names), 0), dtype=torch.float32)
        return torch.stack(rows, dim=1)

    def domain_matrix(self, target: str, metric: str = "by_domain_abs_delta") -> torch.Tensor:
        values = self.metric(metric)
        if target not in values:
            raise KeyError(f"No domain metric for target {target!r}.")
        return values[target]

    def top_fields(self, target: str, k: int = 8, metric: str = "abs_delta") -> Tuple[Tuple[str, float], ...]:
        values = self.metric(metric)[target].view(-1)
        if values.numel() == 0:
            return ()
        topk = min(int(k), int(values.numel()))
        indices = torch.topk(values, k=topk).indices.tolist()
        return tuple((self.field_names[idx], float(values[idx].item())) for idx in indices)


@contextmanager
def _temporary_eval(module):
    if module is None or not hasattr(module, "training"):
        yield
        return

    was_training = bool(module.training)
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def _normalize_field_slices(field_slices: Mapping[str, slice]) -> Tuple[Tuple[str, slice], ...]:
    normalized = []
    for name, field_slice in field_slices.items():
        if not isinstance(field_slice, slice):
            raise TypeError(f"field_slices[{name!r}] must be a slice, got {type(field_slice).__name__}.")
        if field_slice.step not in (None, 1):
            raise ValueError(f"field_slices[{name!r}] must not use a step.")
        normalized.append((str(name), field_slice))
    return tuple(normalized)


def _validate_field_slice(name: str, field_slice: slice, width: int) -> None:
    start = 0 if field_slice.start is None else int(field_slice.start)
    stop = width if field_slice.stop is None else int(field_slice.stop)
    if start < 0 or stop < start or stop > width:
        raise ValueError(f"Invalid slice for field {name!r}: {field_slice!r} with input width {width}.")


def _flatten_target(target: torch.Tensor, batch_size: int, target_name: str) -> torch.Tensor:
    if target.size(0) != batch_size:
        raise ValueError(
            f"Target {target_name!r} has batch size {target.size(0)}, expected {batch_size}."
        )
    if target.ndim == 1:
        return target.float().view(batch_size, 1)
    return target.float().reshape(batch_size, -1)


def _resolve_domain_labels(
        domain_ids: Optional[torch.Tensor],
        num_domains: Optional[int],
        domain_labels: Optional[Iterable[str]],
) -> Tuple[str, ...]:
    if num_domains is None:
        if domain_ids is None or domain_ids.numel() == 0:
            return tuple(str(label) for label in (domain_labels or ()))
        num_domains = int(domain_ids.long().max().item()) + 1

    labels = tuple(str(label) for label in (domain_labels or ()))
    if len(labels) >= int(num_domains):
        return labels[:int(num_domains)]
    return labels + tuple(f"d{idx}" for idx in range(len(labels), int(num_domains)))


def _domain_means(
        per_sample_values: torch.Tensor,
        domain_ids: Optional[torch.Tensor],
        num_domains: Optional[int],
) -> torch.Tensor:
    if domain_ids is None:
        return torch.zeros((0,), dtype=torch.float32)

    flat_domains = domain_ids.detach().long().view(-1).cpu()
    if flat_domains.numel() != per_sample_values.numel():
        raise ValueError(
            f"domain_ids length {flat_domains.numel()} does not match batch size {per_sample_values.numel()}."
        )

    if num_domains is None:
        num_domains = int(flat_domains.max().item()) + 1 if flat_domains.numel() > 0 else 0

    result = []
    for domain_idx in range(int(num_domains)):
        mask = flat_domains == domain_idx
        if mask.any():
            result.append(per_sample_values[mask].mean())
        else:
            result.append(torch.tensor(0.0, dtype=torch.float32))
    return torch.stack(result) if result else torch.zeros((0,), dtype=torch.float32)


@torch.no_grad()
def field_zeroing_intervention(
        x: torch.Tensor,
        field_slices: Mapping[str, slice],
        forward_fn: ForwardTraceFn,
        *,
        targets: Iterable[str],
        domain_ids: Optional[torch.Tensor] = None,
        num_domains: Optional[int] = None,
        domain_labels: Optional[Iterable[str]] = None,
        module=None,
        eps: float = 1e-9,
) -> FieldZeroingResult:
    """
    Zero each input field slice and measure target deltas.

    `forward_fn` receives a masked `x` tensor and must return a mapping from
    target names to tensors whose first dimension is batch. The input `x` is not
    modified.
    """
    if x.ndim < 2:
        raise ValueError(f"field_zeroing_intervention expects x with shape [batch, feature...], got {tuple(x.shape)}.")

    fields = _normalize_field_slices(field_slices)
    target_names = tuple(str(target) for target in targets)
    if not fields:
        raise ValueError("field_zeroing_intervention requires at least one field slice.")
    if not target_names:
        raise ValueError("field_zeroing_intervention requires at least one target.")

    batch_size = int(x.size(0))
    input_width = int(x.size(1))
    for field_name, field_slice in fields:
        _validate_field_slice(field_name, field_slice, input_width)

    abs_delta = {target: [] for target in target_names}
    signed_delta = {target: [] for target in target_names}
    relative_abs_delta = {target: [] for target in target_names}
    by_domain_abs_delta = {target: [] for target in target_names}
    by_domain_signed_delta = {target: [] for target in target_names}

    with _temporary_eval(module):
        base_trace = forward_fn(x)
        base_targets = {
            target: _flatten_target(base_trace[target].detach(), batch_size, target)
            for target in target_names
        }
        base_abs = {
            target: base_targets[target].abs().mean().clamp_min(float(eps)).cpu()
            for target in target_names
        }

        for field_name, field_slice in fields:
            del field_name
            masked_x = x.detach().clone()
            masked_x[:, field_slice] = 0.0
            masked_trace = forward_fn(masked_x)

            for target in target_names:
                masked_target = _flatten_target(masked_trace[target].detach(), batch_size, target)
                delta = (base_targets[target] - masked_target).detach().cpu()
                per_sample_abs = delta.abs().mean(dim=1)
                per_sample_signed = delta.mean(dim=1)
                abs_value = per_sample_abs.mean()
                signed_value = per_sample_signed.mean()

                abs_delta[target].append(abs_value)
                signed_delta[target].append(signed_value)
                relative_abs_delta[target].append(abs_value / base_abs[target])
                by_domain_abs_delta[target].append(_domain_means(per_sample_abs, domain_ids, num_domains))
                by_domain_signed_delta[target].append(_domain_means(per_sample_signed, domain_ids, num_domains))

    packed_abs = {target: torch.stack(values).float() for target, values in abs_delta.items()}
    packed_signed = {target: torch.stack(values).float() for target, values in signed_delta.items()}
    packed_relative = {target: torch.stack(values).float() for target, values in relative_abs_delta.items()}
    packed_domain_abs = {
        target: torch.stack(values).float()
        for target, values in by_domain_abs_delta.items()
        if values and values[0].numel() > 0
    }
    packed_domain_signed = {
        target: torch.stack(values).float()
        for target, values in by_domain_signed_delta.items()
        if values and values[0].numel() > 0
    }

    return FieldZeroingResult(
        field_names=tuple(name for name, _ in fields),
        target_names=target_names,
        abs_delta=packed_abs,
        signed_delta=packed_signed,
        relative_abs_delta=packed_relative,
        by_domain_abs_delta=packed_domain_abs,
        by_domain_signed_delta=packed_domain_signed,
        domain_labels=_resolve_domain_labels(domain_ids, num_domains, domain_labels),
    )
