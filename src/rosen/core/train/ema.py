from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence, Union

import torch


def ema_decay_tag(decay: float) -> str:
    return f"{float(decay):.6g}".replace(".", "p")


def _parse_ema_decays(value: Union[None, float, int, str, Iterable[float]]) -> tuple[float, ...]:
    """Normalize the supported EMA decay configuration forms."""
    if value is None:
        return ()
    if isinstance(value, (float, int)):
        return (float(value),)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = stripped.split(",")
        return _parse_ema_decays(parsed)
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class EMAConfig:
    decays: Sequence[float] | float | str | None = field(default_factory=lambda: (0.995,))
    start_step: int = 0
    update_every: int = 1
    include_buffers: bool = True
    eval_student: bool = True
    eval_ema: bool = True
    primary_checkpoint: str = ""
    backend: str = "teacher"


class EMAController:
    """Exponential moving averages over an arbitrary ``nn.Module``."""

    def __init__(
            self,
            model: torch.nn.Module,
            *,
            decays: Sequence[float],
            start_step: int = 0,
            update_every: int = 1,
            include_buffers: bool = True,
            backend: str = "teacher",
    ):
        self.decays = tuple(float(decay) for decay in decays)
        if not self.decays:
            raise ValueError("EMAController requires at least one decay.")
        for decay in self.decays:
            if not 0.0 <= decay < 1.0:
                raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")

        self.start_step = max(0, int(start_step))
        self.update_every = max(1, int(update_every))
        self.include_buffers = bool(include_buffers)
        self.backend = str(backend).lower()
        if self.backend != "teacher":
            raise ValueError("EMAController currently supports backend='teacher'.")

        self.teachers: Dict[float, torch.nn.Module] = {}
        for decay in self.decays:
            teacher = copy.deepcopy(model)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)
            self.teachers[float(decay)] = teacher

    @torch.no_grad()
    def update(
            self,
            model: torch.nn.Module,
            *,
            step: int,
            state_names: Iterable[str] | None = None,
    ) -> bool:
        step = int(step)
        if step < self.start_step:
            return False
        if (step - self.start_step) % self.update_every != 0:
            return False

        source_state = model.state_dict()
        parameter_names = set(dict(model.named_parameters()).keys())
        selected_names = None if state_names is None else frozenset(str(name) for name in state_names)
        if selected_names is not None:
            unknown = selected_names.difference(source_state)
            if unknown:
                raise KeyError(f"EMA update received unknown state names: {sorted(unknown)}")
        for decay, teacher in self.teachers.items():
            target_state = teacher.state_dict()
            for name, source_value in source_state.items():
                if selected_names is not None and name not in selected_names:
                    continue
                target_value = target_state[name]
                if torch.is_floating_point(target_value):
                    if self.include_buffers or name in parameter_names:
                        target_value.mul_(decay).add_(source_value.detach(), alpha=1.0 - decay)
                    else:
                        target_value.copy_(source_value.detach())
                else:
                    target_value.copy_(source_value.detach())
            teacher.eval()
        return True

    def checkpoint_items(self) -> tuple[tuple[str, float, torch.nn.Module], ...]:
        return tuple(
            (f"ema{ema_decay_tag(decay)}", float(decay), self.teachers[decay])
            for decay in self.decays
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "decays": list(self.decays),
            "start_step": self.start_step,
            "update_every": self.update_every,
            "include_buffers": self.include_buffers,
            "backend": self.backend,
            "teachers": {
                f"ema{ema_decay_tag(decay)}": teacher.state_dict()
                for decay, teacher in self.teachers.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        teachers = state.get("teachers", {})
        if not isinstance(teachers, Mapping):
            raise TypeError("EMA state must contain a mapping under 'teachers'.")
        for decay, teacher in self.teachers.items():
            tag = f"ema{ema_decay_tag(decay)}"
            if tag in teachers:
                teacher.load_state_dict(teachers[tag])
                continue
            legacy_tag = ema_decay_tag(decay)
            if legacy_tag in teachers:
                teacher.load_state_dict(teachers[legacy_tag])


__all__ = [
    "EMAConfig",
    "EMAController",
    "ema_decay_tag",
]
