from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

import torch


LossWeight = float | int | torch.Tensor


@dataclass(frozen=True)
class LossTerm:
    """One named contribution to the active optimization objective."""

    value: torch.Tensor
    weight: LossWeight = 1.0

    @property
    def weighted(self) -> torch.Tensor:
        return self.value * self.weight


class LossPort:
    """Model-owned loss ledger and current-step summary port.

    ``LossPort`` never owns history. The code that owns a backward transaction
    owns a default ``main`` scope, model code may add named terms, and
    ``settle()`` returns the graph-bearing scalar Tensor. Additional named
    scopes are only needed for independent optimization objectives. A detached
    summary remains available until the next ``start_step()`` or an external
    caller clears it.
    """

    def __init__(self) -> None:
        self._scope = ""
        self._terms: OrderedDict[str, LossTerm] = OrderedDict()
        self._stack: list[tuple[str, OrderedDict[str, LossTerm]]] = []
        self._step_context: dict[str, Any] = {}
        self._settlements: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @property
    def active(self) -> bool:
        return bool(self._scope)

    @property
    def scope_name(self) -> str:
        return self._scope or "main"

    @property
    def has_terms(self) -> bool:
        return bool(self._terms)

    @property
    def has_settlements(self) -> bool:
        return bool(self._settlements)

    def start_step(
            self,
            *,
            step: int,
            epoch: int,
            batch_idx: int,
            stage: str,
    ) -> None:
        """Start one logical model step and discard the previous step summary."""

        if self.active:
            raise RuntimeError("LossPort.start_step() cannot run inside an active scope.")
        self._step_context = {
            "step": int(step),
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "stage": str(stage),
        }
        self.clear_summary()

    @contextmanager
    def scope(self, name: str = "main") -> Iterator["LossPort"]:
        scope_name = str(name).strip()
        if not scope_name:
            raise ValueError("LossPort scope name cannot be empty.")
        self._stack.append((self._scope, self._terms))
        self._scope = scope_name
        self._terms = OrderedDict()
        completed = False
        try:
            yield self
            completed = True
        finally:
            unsettled_scope = self._scope
            has_unsettled_terms = bool(self._terms)
            self._scope, self._terms = self._stack.pop()
            if completed and has_unsettled_terms:
                raise RuntimeError(
                    f"Loss scope {unsettled_scope!r} exited with unsettled terms; "
                    "the code that owns the optimization transaction must call settle()."
                )

    def add(self, name: str, value: torch.Tensor, *, weight: LossWeight = 1.0) -> torch.Tensor:
        if not self.active:
            raise RuntimeError("LossPort.add() requires an active LossPort.scope().")
        key = str(name)
        if key in self._terms:
            raise ValueError(f"Loss term {key!r} already exists in scope {self._scope!r}.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Loss term {key!r} must be a Tensor, got {type(value).__name__}.")
        self._terms[key] = LossTerm(value=value, weight=weight)
        return value

    def total(self) -> torch.Tensor:
        if not self.has_terms:
            raise RuntimeError(f"Loss scope {self.scope_name!r} has no terms.")
        return sum(
            (term.weighted for term in self._terms.values()),
            start=next(iter(self._terms.values())).value.new_zeros(()),
        )

    @staticmethod
    def _scalar(value: torch.Tensor | float | int, *, label: str) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{label} must be scalar, got shape {tuple(value.shape)}.")
            return float(value.detach().float().cpu().item())
        return float(value)

    def settle(self, value: torch.Tensor | None = None) -> torch.Tensor:
        """Finalize the active scope, retain a detached summary, and return Tensor."""

        if not self.active:
            raise RuntimeError("LossPort.settle() requires an active LossPort.scope().")
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(
                "Training loss must be a Tensor or None, "
                f"got {type(value).__name__}."
            )
        total = self.total() if value is None else value
        if total.numel() != 1:
            raise ValueError(f"Settled loss must be scalar, got shape {tuple(total.shape)}.")

        scope = self.scope_name
        if scope in self._settlements:
            raise RuntimeError(
                f"Loss scope {scope!r} was settled more than once in the same step; "
                "use distinct scope names for distinct optimization objectives."
            )
        terms = OrderedDict()
        for name, term in self._terms.items():
            raw = self._scalar(term.value, label=f"Loss term {name!r}")
            weight = self._scalar(term.weight, label=f"Loss weight {name!r}")
            terms[name] = {
                "raw": raw,
                "weight": weight,
                "weighted": raw * weight,
            }
        self._settlements[scope] = {
            "total": self._scalar(total, label="Settled loss"),
            "terms": terms,
        }
        self._clear_terms()
        return total

    def _clear_terms(self) -> None:
        self._terms = OrderedDict()

    def summary(self, *, clear: bool = False) -> dict[str, Any]:
        """Return the detached settlements produced by the current model step."""

        result = {
            **self._step_context,
            "settlements": deepcopy(self._settlements),
        }
        if clear:
            self.clear_summary()
        return result

    def clear_summary(self) -> None:
        self._settlements.clear()


__all__ = ["LossPort", "LossTerm", "LossWeight"]
