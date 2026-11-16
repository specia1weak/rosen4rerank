> 历史源码快照，仅供溯源；当前实现以各模块源码为准，实验目录定位已更新。

This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: **/*.py
- Files matching these patterns are excluded: models/**
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
core/
  train/
    __init__.py
    context.py
    early_stopper.py
    ema.py
    hooks.py
    loss.py
    two_stage.py
  trainers/
    audit/
      __init__.py
      cut.py
    __init__.py
    base.py
    ema.py
    two_stage.py
    with_life.py
  __init__.py
  enum_type.py
  interaction.py
data/
  __init__.py
  dataset.py
  padding.py
  profile.py
datasets/
  __init__.py
  aliccp.py
  amazon.py
  amazon2014.py
  base.py
  douban.py
  kuairand.py
  kuaisar.py
  movielens.py
  overview.py
  taac2026.py
emb/
  schema/
    __init__.py
    base.py
    categorical.py
    numerical.py
    sequence.py
    utils.py
  __init__.py
  emblayer.py
  manager.py
  split.py
evaluate/
  __init__.py
  evaluator.py
  manager.py
  metrics.py
experiment/
  __init__.py
  config.py
  engine.py
  param.py
  tracker.py
utils/
  observatory/
    audit/
      __init__.py
      cut.py
      metrics.py
      observation.py
      reporting.py
    metrics/
      __init__.py
      base.py
      basic.py
      correlation.py
      registry.py
      routing.py
      spectral.py
      vq.py
    plots/
      __init__.py
      base.py
      heatmap.py
      profile.py
      report.py
      series.py
    __init__.py
    activation.py
    analysis.py
    collector.py
    config.py
    connector.py
    formatting.py
    intervention.py
    monitor.py
    port.py
    probe.py
    snapshot.py
    store.py
  __init__.py
  expr_tensor.py
  optimize.py
  process.py
  profile_sequence.py
  recorder.py
  sample.py
  sequential.py
  shuffle.py
  sort.py
  split_shuffle.py
  task_chain.py
  time_bucket.py
  time.py
  visualize.py
__init__.py
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="core/train/__init__.py">
"""Foundational training contracts, state and reusable controllers.

Trainer implementations live in :mod:`rosen.core.trainers`. This package
must not import or re-export that higher-level package.
"""

from rosen.core.train.context import (
    EvaluationFeedback,
    TrainContext,
    TrainerComponents,
    TrainerDataLoaders,
)
from rosen.core.train.early_stopper import EarlyStopper
from rosen.core.train.ema import EMAConfig, EMAController, ema_decay_tag
from rosen.core.train.hooks import (
    CustomTrainStepProtocol,
    ObservationSourceProtocol,
    TrainingCycleProtocol,
    TrainingPhaseProtocol,
    TrainerStateProtocol,
)
from rosen.core.train.loss import LossPort, LossTerm
from rosen.core.train.two_stage import (
    StageReport,
    StageSnapshot,
    TwoStagePhase,
    TwoStageTrainingProtocol,
)

__all__ = [
    "CustomTrainStepProtocol",
    "EMAConfig",
    "EMAController",
    "EarlyStopper",
    "EvaluationFeedback",
    "LossPort",
    "LossTerm",
    "ObservationSourceProtocol",
    "StageReport",
    "StageSnapshot",
    "TwoStagePhase",
    "TrainContext",
    "TrainingCycleProtocol",
    "TrainingPhaseProtocol",
    "TrainerComponents",
    "TrainerDataLoaders",
    "TrainerStateProtocol",
    "TwoStageTrainingProtocol",
    "ema_decay_tag",
]
</file>

<file path="core/train/context.py">
from dataclasses import dataclass
from typing import Any, Optional, Iterable, Mapping
import torch

from rosen.core.interaction import Interaction
from rosen.core.train.early_stopper import EarlyStopper


@dataclass(frozen=True)
class EvaluationFeedback:
    """Typed evaluation targets available to model training-cycle methods."""

    metrics_by_target: Mapping[str, Optional[dict]]
    primary_target: str
    primary_metrics: Optional[dict]


@dataclass
class TrainContext:
    # 1. 训练进度统计（通常是只读的）
    epoch: int
    global_step: int
    batch_idx: int

    # 2. 核心操纵杆
    optimizer: torch.optim.Optimizer

    # 3. 实验与全局组件
    manager: Any  # 你的 SchemaManager
    cfg: Any  # 全局配置参数
    timer: Any  # 允许模型给特殊的内部步骤计时
    recorder: Any = None
    stage: str = "train"
    phase: str = "default"
    evaluation_feedback: Optional[EvaluationFeedback] = None


@dataclass
class TrainerDataLoaders:
    train: Iterable[Interaction]
    valid: Optional[Iterable[Interaction]]
    test: Optional[Iterable[Interaction]] = None  # 扩展性好，支持可选参数


@dataclass
class TrainerComponents:
    evaluator_manager: Any = None

    # Keep core/train independent from the concrete Observatory package.
    recorder: Any = None
    timer: Any = None
    early_stopper: Optional[EarlyStopper] = None
    health_monitor: Any = None

    def __post_init__(self) -> None:
        if self.early_stopper is not None and not isinstance(self.early_stopper, EarlyStopper):
            raise TypeError(
                "TrainerComponents.early_stopper must be an EarlyStopper or None, "
                f"got {type(self.early_stopper).__name__}."
            )
</file>

<file path="core/train/early_stopper.py">
"""Metric selection and patience state for early stopping."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopper:
    patience: int = 5
    min_delta: float = 0.0
    preferred_evaluators: tuple[str, ...] = ("overall", "all")
    preferred_metrics: tuple[str, ...] = (
        "auc",
        "gauc",
        "ndcg@10",
        "ndcg@20",
        "ndcg",
        "hr@10",
        "hr@20",
        "hr",
        "recall@10",
        "recall@20",
        "recall",
        "logloss",
        "loss",
    )

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        """Explicitly restart this object's patience window."""
        self.best_metric_value: Optional[float] = None
        self.best_metric_name: Optional[str] = None
        self.best_evaluator_name: Optional[str] = None
        self.best_epoch: Optional[int] = None
        self.bad_epoch_count: int = 0
        self.last_metric_value: Optional[float] = None
        self.last_metric_name: Optional[str] = None
        self.last_evaluator_name: Optional[str] = None
        self.last_status_message: Optional[str] = None

    def _ordered_evaluator_names(self, summary_dict: dict[str, dict[str, float]]) -> list[str]:
        names = list(summary_dict.keys())
        ordered = [name for name in self.preferred_evaluators if name in summary_dict]
        ordered.extend(name for name in names if name not in ordered)
        return ordered

    def _pick_metric(self, summary_dict: dict[str, dict[str, float]]):
        if not summary_dict:
            return None

        for evaluator_name in self._ordered_evaluator_names(summary_dict):
            metrics_dict = summary_dict[evaluator_name]
            if not metrics_dict:
                continue
            lowered = {metric_name.lower(): metric_name for metric_name in metrics_dict}
            for preferred_metric in self.preferred_metrics:
                metric_name = lowered.get(preferred_metric.lower())
                if metric_name is not None:
                    return evaluator_name, metric_name, float(metrics_dict[metric_name])
            metric_name, metric_value = next(iter(metrics_dict.items()))
            return evaluator_name, metric_name, float(metric_value)
        return None

    @staticmethod
    def _metric_mode(metric_name: str) -> str:
        metric_name = metric_name.lower()
        if metric_name in {"loss", "logloss", "mae", "mse", "rmse"}:
            return "min"
        return "max"

    @staticmethod
    def _format_metric_value(metric_value: Optional[float]) -> str:
        if metric_value is None:
            return "None"
        return f"{metric_value:.6f}"

    def step(
        self,
        summary_dict: Optional[dict[str, dict[str, float]]],
        epoch: Optional[int] = None,
    ) -> tuple[bool, bool]:
        if summary_dict is None:
            return False, False

        early_metric = self._pick_metric(summary_dict)
        if early_metric is None:
            return False, False

        evaluator_name, metric_name, metric_value = early_metric
        mode = self._metric_mode(metric_name)
        self.last_evaluator_name = evaluator_name
        self.last_metric_name = metric_name
        self.last_metric_value = metric_value

        if self.best_metric_value is None:
            improved = True
        elif mode == "max":
            improved = metric_value > self.best_metric_value + self.min_delta
        else:
            improved = metric_value < self.best_metric_value - self.min_delta

        if improved:
            self.best_metric_value = metric_value
            self.best_metric_name = metric_name
            self.best_evaluator_name = evaluator_name
            self.best_epoch = epoch
            self.bad_epoch_count = 0
            self.last_status_message = (
                f"[EarlyStop] best {evaluator_name}/{metric_name}="
                f"{self._format_metric_value(metric_value)} at epoch {epoch}, patience 0/{self.patience}"
            )
            return True, False

        self.bad_epoch_count += 1
        self.last_status_message = (
            f"[EarlyStop] wait {self.bad_epoch_count}/{self.patience}, "
            f"current {evaluator_name}/{metric_name}={self._format_metric_value(metric_value)}, "
            f"best {self.best_evaluator_name}/{self.best_metric_name}="
            f"{self._format_metric_value(self.best_metric_value)} at epoch {self.best_epoch}"
        )
        return False, self.bad_epoch_count >= self.patience

    def status_text(self) -> Optional[str]:
        return self.last_status_message
</file>

<file path="core/train/ema.py">
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
</file>

<file path="core/train/hooks.py">
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from rosen.core.train.context import TrainContext


@runtime_checkable
class CustomTrainStepProtocol(Protocol):
    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        ...


@runtime_checkable
class TrainingCycleProtocol(Protocol):
    """Model behavior at the fixed epoch/evaluation boundaries."""

    def on_train_epoch_start(self, ctx: TrainContext) -> Optional[tuple[str, ...]]:
        ...

    def on_train_epoch_end(self, ctx: TrainContext) -> Optional[tuple[str, ...]]:
        ...

    def on_eval_epoch_end(
            self,
            metrics: Optional[dict[str, Any]],
            ctx: TrainContext,
    ) -> Optional[tuple[str, ...]]:
        ...


@runtime_checkable
class TrainerStateProtocol(Protocol):
    """Checkpoint state owned by model-side training behavior."""

    def trainer_state_dict(self) -> Mapping[str, Any]:
        ...

    def load_trainer_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


@runtime_checkable
class TrainingPhaseProtocol(Protocol):
    """Model exposes a named phase only when its training is phase-aware."""

    @property
    def training_phase(self) -> str:
        ...


@runtime_checkable
class ObservationSourceProtocol(Protocol):
    def should_collect_observations(self, ctx: TrainContext, stage: str) -> bool:
        ...

    def collect_observations(self, ctx: TrainContext, stage: str) -> None:
        ...
</file>

<file path="core/train/loss.py">
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
</file>

<file path="core/train/two_stage.py">
"""Opt-in contract for two independent training stages.

Usage
-----
Only a stage-one model that genuinely needs two-stage training implements
``TwoStageTrainingProtocol``::

    class SearchModel(BaseModel):
        def should_finish_stage_one(self, report: StageReport) -> bool:
            return report.epoch >= self.search_epochs

        def initialize_stage_two(self, snapshot: StageSnapshot) -> BaseModel:
            model = RetrainModel(...)
            model.load_state_dict(snapshot.model_state)
            model.freeze_search_only_parameters()
            return model

The experiment supplies a factory that creates *fresh* Trainer state for each
model, then delegates orchestration to ``TwoStageRunner``::

    def trainer_factory(model, stage: TwoStagePhase):
        return EMATrainer(
            model=model,
            optimizer=build_optimizer(model),
            manager=manager,
            loaders=build_loaders(stage),
            components=build_fresh_components(),
            cfg=build_fresh_run_config(stage),
            ema_config=ema_config,
        )

    result = TwoStageRunner(trainer_factory).run(search_model)

The same factory is called with ``TwoStagePhase.STAGE_ONE`` and ``STAGE_TWO``;
the enum is for experiment output isolation and is never interpreted by the
model. ``initialize_stage_two`` must return a new model object, and the factory
must create fresh Trainer-owned state on every call. This keeps optimizer
parameter references, EMA teachers, EarlyStopper state, counters, evaluators,
and monitors out of the stage boundary. Ordinary and stage-two-only models do
not implement this protocol. If checkpoints are enabled, the two configs must
use different checkpoint directories so their fresh epoch/step counters
cannot produce the same filename. Reusing loaders is safe only when they are
re-iterable; one-shot iterators must also be rebuilt by the factory.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, TYPE_CHECKING, runtime_checkable

import torch

if TYPE_CHECKING:
    from rosen.models.base import BaseModel


class TwoStagePhase(str, Enum):
    """Typed factory input; models never need to inspect stage-name strings."""

    STAGE_ONE = "stage_one"
    STAGE_TWO = "stage_two"


@dataclass(frozen=True)
class StageReport:
    """Evaluation and progress visible to the stage-one stop condition."""

    epoch: int
    global_step: int
    metrics_by_target: Mapping[str, Optional[dict]]
    primary_target: str
    primary_metrics: Optional[dict]

    def metrics_for(self, target: str) -> Optional[dict]:
        target = str(target)
        if target == "primary":
            return self.primary_metrics
        return self.metrics_by_target.get(target)


@dataclass(frozen=True)
class StageSnapshot:
    """Detached stage-one model state passed to stage-two initialization."""

    model_state: Mapping[str, Any]
    report: StageReport

    @classmethod
    def capture(
            cls,
            model: torch.nn.Module,
            report: StageReport,
    ) -> "StageSnapshot":
        state: dict[str, Any] = {}
        for name, value in model.state_dict().items():
            if isinstance(value, torch.Tensor):
                state[name] = value.detach().cpu().clone()
            else:
                state[name] = copy.deepcopy(value)
        return cls(model_state=state, report=report)


@runtime_checkable
class TwoStageTrainingProtocol(Protocol):
    """Model-owned termination and stage-two construction rules."""

    def should_finish_stage_one(self, report: StageReport) -> bool:
        ...

    def initialize_stage_two(self, snapshot: StageSnapshot) -> "BaseModel":
        ...


__all__ = [
    "StageReport",
    "StageSnapshot",
    "TwoStagePhase",
    "TwoStageTrainingProtocol",
]
</file>

<file path="core/trainers/audit/__init__.py">
"""Heavyweight trainer variants for observation and intervention audits."""

from rosen.core.trainers.audit.cut import CutAuditTrainer, ObservationTrainer

__all__ = [
    "CutAuditTrainer",
    "ObservationTrainer",
]
</file>

<file path="core/trainers/audit/cut.py">
from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch

from rosen.core.train.context import TrainContext
from rosen.core.train.hooks import ObservationSourceProtocol
from rosen.core.trainers.with_life import BaseTrainerWithLife
from rosen.utils.observatory.audit.cut import (
    accumulate_watch_stats,
    finalize_watch_stats,
    metric_changes,
    watch_changes,
)


class ObservationTrainer(BaseTrainerWithLife):
    """Base training flow with optional ObservationSource collection."""

    def _run_with_optional_observation_collection(self, stage: str, ctx: TrainContext, fn, model=None):
        observed_model = self.model if model is None else model
        if not isinstance(observed_model, ObservationSourceProtocol):
            return fn()

        should_collect = observed_model.should_collect_observations(ctx, stage=stage)
        with observed_model.obp.collecting(enabled=should_collect, clear=True):
            output = fn()

        if should_collect:
            observed_model.collect_observations(ctx, stage=stage)
        return output

    def calculate_train_loss(self, batch, ctx: TrainContext):
        return self._run_with_optional_observation_collection(
            "train",
            ctx,
            lambda: self.model.calculate_loss(batch),
        )


class CutAuditTrainer(ObservationTrainer):
    """Observation-aware trainer with cut/watch evaluation audits."""

    @torch.no_grad()
    def _evaluate_model(
            self,
            model,
            *,
            active_cut: Optional[str] = None,
            collect_watch_stats: bool = False,
    ):
        if not self.valid_loader or not self.evaluator:
            return None

        model.eval()
        watch_storage = {}
        port = model.obp
        for batch_idx, batch_interaction in enumerate(self.valid_loader):
            batch_interaction = batch_interaction.to(self.cfg.device)
            uids = batch_interaction[self.manager.uid_field]
            labels = batch_interaction[self.manager.label_field]
            eval_ctx = self._context(batch_idx=batch_idx, stage="eval")
            cut_ctx = port.cutting(active_cut)
            watch_ctx = (
                port.watching(enabled=True, clear=True)
                if collect_watch_stats
                else nullcontext()
            )
            with cut_ctx, watch_ctx:
                scores = self._run_with_optional_observation_collection(
                    "eval",
                    eval_ctx,
                    lambda: model.predict(batch_interaction),
                    model=model,
                )
            if collect_watch_stats:
                accumulate_watch_stats(watch_storage, port.watch_snapshot(clear=True))
            self.evaluator.collect(uids, labels, batch_interaction, batch_preds_1d=scores)

        metrics_result = self.evaluator.summary(self.epoch)
        self.evaluator.clear()
        if collect_watch_stats:
            return metrics_result, finalize_watch_stats(watch_storage)
        return metrics_result

    def _evaluate_cut_audit(self, model, baseline_metrics: Optional[dict], baseline_watch: dict) -> list[dict]:
        cut_names = list(model.obp.cut_names())
        if not cut_names:
            return []

        rows = []
        for cut_name in cut_names:
            metrics, watch_stats = self._evaluate_model(
                model,
                active_cut=cut_name,
                collect_watch_stats=True,
            )
            row = {
                "cut": cut_name,
                "metrics": metrics,
                "metric_changes": metric_changes(baseline_metrics, metrics),
                "watch_stats": watch_stats,
                "watch_changes": watch_changes(baseline_watch, watch_stats),
            }
            rows.append(row)
            print(
                f"[CutAudit] cut={cut_name} "
                f"metric_changes={row['metric_changes']} "
                f"watch_changes={row['watch_changes']}",
                flush=True,
            )
        return rows

    @torch.no_grad()
    def evaluate_epoch(self):
        result = self._evaluate_model(
            self.model,
            collect_watch_stats=True,
        )
        if result is None:
            return None
        metrics, watch_stats = result
        self._notify_evaluation_end(metrics)
        self.last_cut_audit_rows = self._evaluate_cut_audit(self.model, metrics, watch_stats)
        return metrics
</file>

<file path="core/trainers/__init__.py">
"""Trainer implementations built from :mod:`rosen.core.train` components."""

from rosen.core.trainers.base import BaseTrainer
from rosen.core.trainers.ema import EMATrainer
from rosen.core.trainers.two_stage import TwoStageRunResult, TwoStageRunner
from rosen.core.trainers.with_life import BaseTrainerWithLife

__all__ = [
    "BaseTrainer",
    "BaseTrainerWithLife",
    "EMATrainer",
    "TwoStageRunResult",
    "TwoStageRunner",
]
</file>

<file path="core/trainers/base.py">
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional, Union

import torch

from rosen.core.interaction import Interaction
from rosen.core.train.context import (
    EvaluationFeedback,
    TrainContext,
    TrainerComponents,
    TrainerDataLoaders,
)
from rosen.core.train.hooks import (
    CustomTrainStepProtocol,
    TrainerStateProtocol,
)
from rosen.emb import SchemaManager
from rosen.experiment.param import ConfigBase
from rosen.models.base import BaseModel


class BaseTrainer:
    """Minimal trainer for ordinary train/eval/checkpoint flow.

    This class deliberately does not interpret model training-cycle methods or
    health monitoring. Those fixed boundaries live in ``BaseTrainerWithLife``;
    experiment audits live under :mod:`rosen.core.trainers.audit`.
    """

    def __init__(
            self,
            model: BaseModel,
            optimizer: torch.optim.Optimizer,
            manager: SchemaManager,
            loaders: TrainerDataLoaders,
            components: TrainerComponents,
            cfg: ConfigBase,
    ):
        if components.health_monitor is not None:
            raise ValueError(
                "BaseTrainer does not run training-cycle health monitoring; "
                "use BaseTrainerWithLife."
            )
        self.model = model.to(cfg.device)
        self.optimizer = optimizer
        self.manager = manager
        self.train_loader = loaders.train
        self.valid_loader = loaders.valid
        self.test_loader = loaders.test

        self.evaluator = components.evaluator_manager
        self.timer = components.timer
        self.recorder = components.recorder
        self.early_stopper = components.early_stopper
        self.cfg = cfg

        self.epoch = 0
        self.global_step = 0
        self.last_evaluation_feedback: EvaluationFeedback | None = None

    def _timed(self, name: str):
        return self.timer(name) if self.timer is not None else nullcontext()

    def _context(
            self,
            *,
            batch_idx: int,
            stage: str = "train",
            evaluation_feedback: EvaluationFeedback | None = None,
    ) -> TrainContext:
        return TrainContext(
            epoch=self.epoch,
            global_step=self.global_step,
            batch_idx=int(batch_idx),
            optimizer=self.optimizer,
            manager=self.manager,
            cfg=self.cfg,
            timer=self.timer,
            recorder=self.recorder,
            stage=str(stage),
            phase="default",
            evaluation_feedback=evaluation_feedback,
        )

    def predict_step(self, batch_interaction: Interaction):
        return self.model.predict(batch_interaction)

    def _checkpoint_payload(self, metrics: Optional[dict] = None) -> dict:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_trainer_state_dict": (
                self.model.trainer_state_dict()
                if isinstance(self.model, TrainerStateProtocol) else None
            ),
            "metrics": metrics,
            "experiment_name": self.cfg.experiment_name,
            "dataset_name": self.cfg.dataset_name,
        }

    def save_checkpoint(self, tag: str = "", metrics: Optional[dict] = None) -> Union[Path, None]:
        ckpt_dir = getattr(self.cfg, "ckpt_dir", "")
        if not ckpt_dir:
            return None

        save_dir = Path(ckpt_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        safe_tag = f"_{tag}" if tag else ""
        filename = f"epoch_{self.epoch:04d}_step_{self.global_step:08d}{safe_tag}.pt"
        save_path = save_dir / filename

        torch.save(self._checkpoint_payload(metrics), save_path)
        return save_path

    def load_checkpoint(self, path: Union[str, Path], *, map_location=None) -> dict:
        checkpoint = torch.load(path, map_location=map_location or self.cfg.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        model_trainer_state = checkpoint.get("model_trainer_state_dict")
        if isinstance(self.model, TrainerStateProtocol) and model_trainer_state is not None:
            self.model.load_trainer_state_dict(model_trainer_state)
        return checkpoint

    def calculate_train_loss(self, batch, ctx: TrainContext):
        del ctx
        return self.model.calculate_loss(batch)

    def _default_train_step(self, batch, ctx: TrainContext):
        raw_loss = self.calculate_train_loss(batch, ctx)
        loss = self.model.lsp.settle(raw_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        return loss

    def train_step(self, batch, ctx: TrainContext) -> torch.Tensor:
        """Run one train step inside the model's implicit ``main`` Loss scope."""
        with self.model.lsp.scope():
            if not isinstance(self.model, CustomTrainStepProtocol):
                return self._default_train_step(batch, ctx)

            loss = self.model.custom_train_step(batch, ctx)
            if not isinstance(loss, torch.Tensor):
                raise TypeError(
                    "custom_train_step() must settle and return a scalar Tensor, "
                    f"got {type(loss).__name__}."
                )
            if loss.numel() != 1:
                raise ValueError(
                    "custom_train_step() must return a scalar Tensor, "
                    f"got shape {tuple(loss.shape)}."
                )
            if self.model.lsp.has_terms:
                raise RuntimeError(
                    "custom_train_step() returned with unsettled LossPort terms."
                )
            if not self.model.lsp.has_settlements:
                raise RuntimeError(
                    "custom_train_step() must call model.lsp.settle() for each "
                    "optimization objective before returning."
                )
            return loss

    def should_stop_train_epoch(self, ctx: TrainContext, loss) -> bool:
        """Extension point for trainers that intentionally cap an epoch."""
        del loss
        limit = int(getattr(self.cfg, "max_train_steps", 0) or 0)
        return limit > 0 and int(ctx.global_step) + 1 >= limit

    def _has_reached_train_step_limit(self) -> bool:
        limit = int(getattr(self.cfg, "max_train_steps", 0) or 0)
        return limit > 0 and self.global_step >= limit

    def _run_train_batches(self) -> int:
        log_interval = max(1, int(getattr(self.cfg, "train_log_interval", 100)))
        last_batch_idx = -1
        for batch_idx, batch_interaction in enumerate(self.train_loader):
            batch_interaction = batch_interaction.to(self.cfg.device)
            self.optimizer.zero_grad(set_to_none=True)

            ctx = self._context(batch_idx=batch_idx, stage="train")
            last_batch_idx = batch_idx
            current_step = self.global_step + 1
            self.model.lsp.start_step(
                step=current_step,
                epoch=ctx.epoch,
                batch_idx=ctx.batch_idx,
                stage=ctx.stage,
            )
            loss = self.train_step(batch_interaction, ctx)

            loss_value = float(loss.detach().float().cpu().item())
            if current_step % log_interval == 0:
                print(f"step:{current_step} loss:{loss_value}")

            self.global_step += 1
            if self.should_stop_train_epoch(ctx, loss_value):
                break
        return last_batch_idx

    def train_epoch(self):
        with self._timed("train_epoch"):
            self.model.train()
            self._run_train_batches()
            self.epoch += 1

    @torch.no_grad()
    def _evaluate_loader(self, model, loader):
        if loader is None or self.evaluator is None:
            return None

        model.eval()
        for batch_idx, batch_interaction in enumerate(loader):
            batch_interaction = batch_interaction.to(self.cfg.device)
            uids = batch_interaction[self.manager.uid_field]
            labels = batch_interaction[self.manager.label_field]
            if model is self.model:
                scores = self.predict_step(batch_interaction)
            else:
                scores = model.predict(batch_interaction)
            self.evaluator.collect(uids, labels, batch_interaction, batch_preds_1d=scores)

        metrics_result = self.evaluator.summary(self.epoch)
        self.evaluator.clear()
        return metrics_result

    def _evaluate_model(self, model):
        return self._evaluate_loader(model, self.valid_loader)

    @torch.no_grad()
    def evaluate_epoch(self):
        with self._timed("valid_epoch"):
            return self._evaluate_model(self.model)

    @torch.no_grad()
    def evaluate_test(self):
        with self._timed("test_epoch"):
            return self._evaluate_loader(self.model, self.test_loader)

    def run(
            self,
            *,
            stop_condition: Callable[["BaseTrainer", EvaluationFeedback], bool] | None = None,
    ):
        last_metrics = None
        for _ in range(self.cfg.max_epochs):
            if self._has_reached_train_step_limit():
                break
            self.train_epoch()
            self.last_evaluation_feedback = None
            metrics = self.evaluate_epoch()
            last_metrics = metrics
            print(metrics)
            feedback = self.last_evaluation_feedback or EvaluationFeedback(
                metrics_by_target={"student": metrics},
                primary_target="student",
                primary_metrics=metrics,
            )
            should_stop = bool(
                stop_condition is not None and stop_condition(self, feedback)
            )
            if self.early_stopper:
                is_best, early_stop = self.early_stopper.step(metrics, epoch=self.epoch)
                status_text = self.early_stopper.status_text()
                if status_text:
                    print(status_text)
                if is_best:
                    save_path = self.save_checkpoint(tag="best", metrics=metrics)
                    if save_path is not None:
                        print(f"[Checkpoint] saved: {save_path}")
                if early_stop:
                    print(f"[EarlyStop] stop at epoch {self.epoch}")
                    should_stop = True
            if should_stop:
                break
        return last_metrics
</file>

<file path="core/trainers/ema.py">
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch

from rosen.core.train.context import (
    EvaluationFeedback,
    TrainContext,
    TrainerComponents,
    TrainerDataLoaders,
)
from rosen.core.train.ema import EMAConfig, EMAController, _parse_ema_decays, ema_decay_tag
from rosen.core.trainers.with_life import BaseTrainerWithLife
from rosen.emb import SchemaManager
from rosen.experiment.param import ConfigBase
from rosen.models.base import BaseModel


class EMATrainer(BaseTrainerWithLife):
    """BaseTrainer variant with model-agnostic EMA teacher checkpoints."""

    def __init__(
            self,
            model: BaseModel,
            optimizer: torch.optim.Optimizer,
            manager: SchemaManager,
            loaders: TrainerDataLoaders,
            components: TrainerComponents,
            cfg: ConfigBase,
            *,
            ema_config: EMAConfig | None = None,
    ):
        super().__init__(model, optimizer, manager, loaders, components, cfg)

        self.ema_config = ema_config or EMAConfig()
        resolved_decays = _parse_ema_decays(self.ema_config.decays)
        self.ema = None
        if resolved_decays:
            self.ema = EMAController(
                self.model,
                decays=resolved_decays,
                start_step=self.ema_config.start_step,
                update_every=self.ema_config.update_every,
                include_buffers=self.ema_config.include_buffers,
                backend=self.ema_config.backend,
            )
        self.last_eval_metrics_by_checkpoint: Dict[str, Optional[dict]] = {}

    @staticmethod
    def _update_step(ctx: TrainContext) -> int:
        return int(ctx.global_step) + (1 if ctx.stage == "train" else 0)

    def _update_ema(
            self,
            ctx: TrainContext,
            *,
            state_names: tuple[str, ...] | None = None,
    ) -> None:
        if self.ema is not None:
            self.ema.update(
                self.model,
                step=self._update_step(ctx),
                state_names=state_names,
            )

    def train_step(self, batch, ctx: TrainContext):
        loss = super().train_step(batch, ctx)
        self._update_ema(ctx)
        return loss

    def _after_cycle_model_update(
            self,
            ctx: TrainContext,
            state_names: tuple[str, ...],
    ) -> None:
        self._update_ema(ctx, state_names=state_names)

    def _default_primary_checkpoint(self) -> str:
        if self.ema_config.primary_checkpoint:
            return str(self.ema_config.primary_checkpoint)
        if self.ema_config.eval_ema and self.ema is not None and self.ema.decays:
            return f"ema{ema_decay_tag(self.ema.decays[-1])}"
        return "student"

    @torch.no_grad()
    def evaluate_epoch(self):
        if not self.valid_loader or not self.evaluator:
            return None

        metrics_by_checkpoint: Dict[str, Optional[dict]] = {}
        if self.ema_config.eval_student:
            metrics = self._evaluate_model(self.model)
            metrics_by_checkpoint["student"] = metrics
            print(f"[EMA Eval] checkpoint=student metrics={metrics}")

        if self.ema_config.eval_ema and self.ema is not None:
            for checkpoint, _decay, teacher in self.ema.checkpoint_items():
                metrics = self._evaluate_model(teacher)
                metrics_by_checkpoint[checkpoint] = metrics
                print(f"[EMA Eval] checkpoint={checkpoint} metrics={metrics}")

        self.last_eval_metrics_by_checkpoint = metrics_by_checkpoint
        primary = self._default_primary_checkpoint()
        if primary not in metrics_by_checkpoint:
            primary = next(iter(metrics_by_checkpoint), "")
        primary_metrics = metrics_by_checkpoint.get(primary)
        self._notify_evaluation_end(
            primary_metrics,
            feedback=EvaluationFeedback(
                metrics_by_target=metrics_by_checkpoint,
                primary_target=primary,
                primary_metrics=primary_metrics,
            ),
        )
        return primary_metrics

    def _checkpoint_payload(self, metrics: Optional[dict] = None) -> dict:
        payload = super()._checkpoint_payload(metrics)
        payload.update({
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "metrics_by_checkpoint": self.last_eval_metrics_by_checkpoint,
        })
        return payload

    def load_checkpoint(self, path: Union[str, Path], *, map_location=None) -> dict:
        checkpoint = super().load_checkpoint(path, map_location=map_location)
        ema_state = checkpoint.get("ema_state_dict")
        if self.ema is not None and ema_state is not None:
            self.ema.load_state_dict(ema_state)
        self.last_eval_metrics_by_checkpoint = dict(checkpoint.get("metrics_by_checkpoint") or {})
        return checkpoint
</file>

<file path="core/trainers/two_stage.py">
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from rosen.core.train.context import EvaluationFeedback
from rosen.core.train.two_stage import (
    StageReport,
    StageSnapshot,
    TwoStagePhase,
    TwoStageTrainingProtocol,
)
from rosen.core.trainers.base import BaseTrainer
from rosen.models.base import BaseModel


TrainerFactory = Callable[[BaseModel, TwoStagePhase], BaseTrainer]


@dataclass(frozen=True)
class TwoStageRunResult:
    """Both independent training sessions and their transition snapshot."""

    stage_one_trainer: BaseTrainer
    stage_two_trainer: BaseTrainer
    stage_one_report: StageReport
    stage_snapshot: StageSnapshot
    stage_two_model: BaseModel
    stage_one_metrics: Optional[dict]
    stage_two_metrics: Optional[dict]


class TwoStageRunner:
    """Run two fresh Trainer instances around a model-owned transition."""

    def __init__(self, trainer_factory: TrainerFactory):
        if not callable(trainer_factory):
            raise TypeError("trainer_factory must be callable.")
        self.trainer_factory = trainer_factory

    @staticmethod
    def _report(trainer: BaseTrainer, feedback: EvaluationFeedback) -> StageReport:
        return StageReport(
            epoch=int(trainer.epoch),
            global_step=int(trainer.global_step),
            metrics_by_target=dict(feedback.metrics_by_target),
            primary_target=str(feedback.primary_target),
            primary_metrics=feedback.primary_metrics,
        )

    def _build_trainer(
            self,
            model: BaseModel,
            stage: TwoStagePhase,
    ) -> BaseTrainer:
        trainer = self.trainer_factory(model, stage)
        if not isinstance(trainer, BaseTrainer):
            raise TypeError(
                "trainer_factory must return a BaseTrainer, "
                f"got {type(trainer).__name__}."
            )
        if trainer.model is not model:
            raise ValueError("trainer_factory must attach the exact model object it receives.")
        return trainer

    @staticmethod
    def _validate_fresh_runtime(
            stage_one_trainer: BaseTrainer,
            stage_two_trainer: BaseTrainer,
    ) -> None:
        if stage_two_trainer is stage_one_trainer:
            raise ValueError("trainer_factory must create a fresh Trainer for stage two.")
        if stage_two_trainer.optimizer is stage_one_trainer.optimizer:
            raise ValueError("trainer_factory must create a fresh optimizer for stage two.")

        stage_one_checkpoint_dir = getattr(stage_one_trainer.cfg, "ckpt_dir", "")
        stage_two_checkpoint_dir = getattr(stage_two_trainer.cfg, "ckpt_dir", "")
        if stage_one_checkpoint_dir and stage_two_checkpoint_dir:
            stage_one_checkpoint_dir = Path(stage_one_checkpoint_dir).expanduser().resolve()
            stage_two_checkpoint_dir = Path(stage_two_checkpoint_dir).expanduser().resolve()
            if stage_one_checkpoint_dir == stage_two_checkpoint_dir:
                raise ValueError(
                    "two-stage Trainers need different checkpoint directories because "
                    "their epoch and step counters restart."
                )

        stage_two_parameter_ids = {
            id(parameter) for parameter in stage_two_trainer.model.parameters()
        }
        optimizer_parameter_ids = {
            id(parameter)
            for group in stage_two_trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        unknown_parameter_ids = optimizer_parameter_ids.difference(stage_two_parameter_ids)
        if unknown_parameter_ids:
            raise ValueError("stage-two optimizer contains parameters from another model.")

        for attribute in ("train_loader", "valid_loader"):
            stage_one_loader = getattr(stage_one_trainer, attribute)
            stage_two_loader = getattr(stage_two_trainer, attribute)
            if stage_one_loader is None or stage_two_loader is not stage_one_loader:
                continue
            if iter(stage_two_loader) is stage_two_loader:
                raise ValueError(
                    f"trainer_factory must rebuild one-shot {attribute} for stage two."
                )

        for attribute in (
            "early_stopper",
            "health_monitor",
            "evaluator",
            "timer",
            "recorder",
            "ema",
        ):
            stage_one_value = getattr(stage_one_trainer, attribute, None)
            stage_two_value = getattr(stage_two_trainer, attribute, None)
            if stage_one_value is not None and stage_two_value is stage_one_value:
                raise ValueError(
                    f"trainer_factory must create a fresh {attribute} for stage two."
                )

    def run(self, stage_one_model: BaseModel) -> TwoStageRunResult:
        if not isinstance(stage_one_model, TwoStageTrainingProtocol):
            raise TypeError(
                "stage_one_model must implement TwoStageTrainingProtocol, "
                f"got {type(stage_one_model).__name__}."
            )

        stage_one_trainer = self._build_trainer(
            stage_one_model,
            TwoStagePhase.STAGE_ONE,
        )
        latest_report: StageReport | None = None

        def should_stop(trainer: BaseTrainer, feedback: EvaluationFeedback) -> bool:
            nonlocal latest_report
            latest_report = self._report(trainer, feedback)
            return bool(stage_one_model.should_finish_stage_one(latest_report))

        stage_one_metrics = stage_one_trainer.run(stop_condition=should_stop)
        if latest_report is None:
            feedback = stage_one_trainer.last_evaluation_feedback or EvaluationFeedback(
                metrics_by_target={"student": stage_one_metrics},
                primary_target="student",
                primary_metrics=stage_one_metrics,
            )
            latest_report = self._report(stage_one_trainer, feedback)

        snapshot = StageSnapshot.capture(stage_one_model, latest_report)
        stage_two_model = stage_one_model.initialize_stage_two(snapshot)
        if not isinstance(stage_two_model, BaseModel):
            raise TypeError(
                "initialize_stage_two() must return a BaseModel, "
                f"got {type(stage_two_model).__name__}."
            )
        if stage_two_model is stage_one_model:
            raise ValueError("initialize_stage_two() must return a new model object.")

        stage_two_trainer = self._build_trainer(
            stage_two_model,
            TwoStagePhase.STAGE_TWO,
        )
        self._validate_fresh_runtime(stage_one_trainer, stage_two_trainer)
        stage_two_metrics = stage_two_trainer.run()

        return TwoStageRunResult(
            stage_one_trainer=stage_one_trainer,
            stage_two_trainer=stage_two_trainer,
            stage_one_report=latest_report,
            stage_snapshot=snapshot,
            stage_two_model=stage_two_model,
            stage_one_metrics=stage_one_metrics,
            stage_two_metrics=stage_two_metrics,
        )


__all__ = ["TrainerFactory", "TwoStageRunResult", "TwoStageRunner"]
</file>

<file path="core/trainers/with_life.py">
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

from rosen.core.train.context import EvaluationFeedback, TrainContext
from rosen.core.train.hooks import TrainingCycleProtocol, TrainingPhaseProtocol
from rosen.core.trainers.base import BaseTrainer


class BaseTrainerWithLife(BaseTrainer):
    """BaseTrainer plus fixed model training-cycle and health boundaries."""

    def __init__(self, model, optimizer, manager, loaders, components, cfg):
        health_monitor = components.health_monitor
        super().__init__(
            model,
            optimizer,
            manager,
            loaders,
            replace(components, health_monitor=None),
            cfg,
        )
        self.health_monitor = health_monitor

    def _phase(self) -> str:
        if isinstance(self.model, TrainingPhaseProtocol):
            return str(self.model.training_phase)
        return "default"

    def _context(
            self,
            *,
            batch_idx: int,
            stage: str = "train",
            evaluation_feedback: EvaluationFeedback | None = None,
    ) -> TrainContext:
        ctx = super()._context(
            batch_idx=batch_idx,
            stage=stage,
            evaluation_feedback=evaluation_feedback,
        )
        ctx.phase = self._phase()
        return ctx

    @staticmethod
    def _updated_state_names(value) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, tuple) or any(not isinstance(name, str) for name in value):
            raise TypeError(
                "TrainingCycleProtocol methods must return tuple[str, ...] or None."
            )
        return value

    def _after_cycle_model_update(
            self,
            ctx: TrainContext,
            state_names: tuple[str, ...],
    ) -> None:
        del ctx, state_names

    def _call_cycle_method(self, method, ctx: TrainContext, *args) -> tuple[str, ...]:
        self.model.lsp.start_step(
            step=ctx.global_step,
            epoch=ctx.epoch,
            batch_idx=ctx.batch_idx,
            stage=ctx.stage,
        )
        capture = (
            self.health_monitor.capture_scope(self.model, ctx, force=True)
            if self.health_monitor is not None else nullcontext()
        )
        with capture:
            state_names = self._updated_state_names(method(*args, ctx))
        if state_names:
            self._after_cycle_model_update(ctx, state_names)
        if self.health_monitor is not None:
            self.health_monitor.record_training_cycle(
                self.model,
                ctx,
                state_names=state_names,
                phase=self._phase(),
            )
        return state_names

    def train_step(self, batch, ctx: TrainContext):
        capture = (
            self.health_monitor.capture_scope(self.model, ctx)
            if self.health_monitor is not None else nullcontext()
        )
        with capture:
            loss = super().train_step(batch, ctx)
        if self.health_monitor is not None:
            self.health_monitor.record_train_step(self.model, ctx)
        return loss

    def _run_train_batches(self) -> int:
        cycle_model = (
            self.model if isinstance(self.model, TrainingCycleProtocol) else None
        )
        if cycle_model is not None:
            start_ctx = self._context(batch_idx=-1, stage="epoch_start")
            self._call_cycle_method(cycle_model.on_train_epoch_start, start_ctx)

        last_batch_idx = super()._run_train_batches()

        if cycle_model is not None:
            end_ctx = self._context(batch_idx=last_batch_idx, stage="epoch_end")
            self._call_cycle_method(cycle_model.on_train_epoch_end, end_ctx)
        return last_batch_idx

    def _notify_evaluation_end(
            self,
            metrics,
            *,
            feedback: EvaluationFeedback | None = None,
    ) -> None:
        feedback = feedback or EvaluationFeedback(
            metrics_by_target={"student": metrics},
            primary_target="student",
            primary_metrics=metrics,
        )
        self.last_evaluation_feedback = feedback
        if not isinstance(self.model, TrainingCycleProtocol):
            return
        eval_ctx = self._context(
            batch_idx=-1,
            stage="eval_end",
            evaluation_feedback=feedback,
        )
        self._call_cycle_method(self.model.on_eval_epoch_end, eval_ctx, metrics)

    def evaluate_epoch(self):
        metrics = super().evaluate_epoch()
        if self.valid_loader is not None and self.evaluator is not None:
            self._notify_evaluation_end(metrics)
        return metrics

    def _checkpoint_payload(self, metrics=None) -> dict:
        payload = super()._checkpoint_payload(metrics)
        payload["health_monitor_state_dict"] = (
            self.health_monitor.state_dict() if self.health_monitor is not None else None
        )
        return payload

    def load_checkpoint(self, path, *, map_location=None) -> dict:
        checkpoint = super().load_checkpoint(path, map_location=map_location)
        health_state = checkpoint.get("health_monitor_state_dict")
        if self.health_monitor is not None and health_state is not None:
            self.health_monitor.load_state_dict(health_state)
        return checkpoint

    def run(self, *, stop_condition=None):
        metrics = super().run(stop_condition=stop_condition)
        if self.health_monitor is not None:
            end_ctx = self._context(batch_idx=-1, stage="train_end")
            self.health_monitor.flush(self.model, end_ctx)
        return metrics


__all__ = ["BaseTrainerWithLife"]
</file>

<file path="core/__init__.py">

</file>

<file path="core/enum_type.py">
# -*- coding: utf-8 -*-
# @Time   : 2020/8/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.utils.enum_type
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``GENERAL``: General Recommendation
    - ``SEQUENTIAL``: Sequential Recommendation
    - ``CONTEXT``: Context-aware Recommendation
    - ``KNOWLEDGE``: Knowledge-based Recommendation
    """

    GENERAL = 1
    SEQUENTIAL = 2
    CONTEXT = 3
    KNOWLEDGE = 4
    TRADITIONAL = 5
    DECISIONTREE = 6


class KGDataLoaderState(Enum):
    """States for Knowledge-based DataLoader.

    - ``RSKG``: Return both knowledge graph information and user-item interaction information.
    - ``RS``: Only return the user-item interaction.
    - ``KG``: Only return the triplets with negative examples in a knowledge graph.
    """

    RSKG = 1
    RS = 2
    KG = 3


class EvaluatorType(Enum):
    """Type for evaluation metrics.

    - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
    - ``VALUE``: Value-based metrics like AUC, etc.
    """

    RANKING = 1
    VALUE = 2


class InputType(Enum):
    """Type of Models' input.

    - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
    - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
    """

    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3


class FeatureType(Enum):
    """Type of features.

    - ``TOKEN``: Token features like user_id and item_id.
    - ``FLOAT``: Float features like rating and timestamp.
    - ``TOKEN_SEQ``: Token sequence features like review.
    - ``FLOAT_SEQ``: Float sequence features like pretrained vector.
    """

    TOKEN = "token"
    FLOAT = "float"
    TOKEN_SEQ = "token_seq"
    FLOAT_SEQ = "float_seq"


class FeatureSource(Enum):
    """Source of features.

    - ``INTERACTION``: Features from ``.inter`` (other than ``user_id`` and ``item_id``).
    - ``USER``: Features from ``.user`` (other than ``user_id``).
    - ``ITEM``: Features from ``.item`` (other than ``item_id``).
    - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``.
    - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``.
    - ``KG``: Features from ``.kg``.
    - ``NET``: Features from ``.net``.
    """
    UNKNOWN = "unknown"
    INTERACTION = "inter"
    USER = "user"
    ITEM = "item"
    USER_ID = "user_id"
    ITEM_ID = "item_id"
    SEQ = "seq"
    SEQ_GROUP = "seq_group"
    KG = "kg"
    NET = "net"
</file>

<file path="core/interaction.py">
# @Time   : 2020/7/10
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time    : 2022/7/8, 2020/9/15, 2020/9/16, 2020/8/12
# @Author  : Zhen Tian, Yupeng Hou, Yushuo Chen, Xingyu Pan
# @email   : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, panxy@ruc.edu.cn

"""
recbole.data.interaction
############################
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils

# 他只是对齐......
def _convert_to_tensor(data):
    """
    增强版转换函数：完美兼容 uint32 并自动对齐 PyTorch 常用类型。
    """
    # 1. 预处理：如果是含有 uint32 的 NumPy 数组，强制转为 int64 (Long)
    # 因为 PyTorch 不支持 uint32，且 ID 类数据转 int64 最保险（防止溢出）
    if isinstance(data, np.ndarray) and data.dtype == np.uint32:
        data = data.astype(np.int64)
    elif isinstance(data, pd.Series) and data.dtype == "uint32":
        data = data.astype(np.int64)

    elem = data[0]

    # 情况 A：单值/数值数组处理
    if isinstance(elem, (float, int, np.floating, np.integer)):
        new_data = torch.as_tensor(data)

    # 情况 B：序列处理 (例如 tags, history_videos)
    elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
        # 对每一个子序列进行递归或类型防御
        seq_data = []
        for d in data:
            # 防御子序列中的 uint32
            if isinstance(d, np.ndarray) and d.dtype == np.uint32:
                d = d.astype(np.int64)
            seq_data.append(torch.as_tensor(d))

        new_data = rnn_utils.pad_sequence(seq_data, batch_first=True) # 这里对序列做了pad保证同长度！

    else:
        raise ValueError(f"[{type(elem)}] is not supported!")

    # 3. 类型后置对齐
    # float64 转 float32 是推荐系统的基操，节省一半显存
    if new_data.dtype == torch.float64:
        new_data = new_data.float()
    # uint8 虽然 PyTorch 支持，但在做 Embedding 索引时通常需要 Long
    elif new_data.dtype == torch.uint8:
        new_data = new_data.long()

    return new_data


class Interaction(object):
    """The basic class representing a batch of interaction records.

    Note:
        While training, there is no strict rules for data in one Interaction object.

        While testing, it should be guaranteed that all interaction records of one single
        user will not appear in different Interaction object, and records of the same user
        should be continuous. Meanwhile, the positive cases of one user always need to occur
        **earlier** than this user's negative cases.

        A correct example:
            =======     =======     =======
            user_id     item_id     label
            =======     =======     =======
            1           2           1
            1           6           1
            1           3           1
            1           1           0
            2           3           1
            ...         ...         ...
            =======     =======     =======

        Some wrong examples for Interaction objects used in testing:

        1.
            =======     =======     =======     ============
            user_id     item_id     label
            =======     =======     =======     ============
            1           2           1
            1           6           0           # positive cases of one user always need to

                                                occur earlier than this user's negative cases
            1           3           1
            1           1           0
            2           3           1
            ...         ...         ...
            =======     =======     =======     ============

        2.
            =======     =======     =======     ========
            user_id     item_id     label
            =======     =======     =======     ========
            1           2           1
            1           6           1
            1           3           1
            2           3           1           # records of the same user should be continuous.
            1           1           0
            ...         ...         ...
            =======     =======     =======     ========

    Attributes:
        interaction (dict or pandas.DataFrame): keys are meaningful str (also can be called field name),
            and values are Torch Tensor of numpy Array with shape (batch_size, \\*).
    """

    def __init__(self, interaction):
        self.interaction = dict()
        if isinstance(interaction, dict):
            for key, value in interaction.items():
                if isinstance(value, (list, np.ndarray)):
                    self.interaction[key] = _convert_to_tensor(value)
                elif isinstance(value, torch.Tensor):
                    self.interaction[key] = value
                else:
                    raise ValueError(
                        f"The type of {key}[{type(value)}] is not supported!"
                    )
        elif isinstance(interaction, pd.DataFrame):
            for key in interaction:
                value = interaction[key].values
                self.interaction[key] = _convert_to_tensor(value)
        else:
            raise ValueError(
                f"[{type(interaction)}] is not supported for initialize `Interaction`!"
            )
        self.length = -1
        for k in self.interaction:
            self.length = max(self.length, self.interaction[k].unsqueeze(-1).shape[0])

    def __iter__(self):
        return self.interaction.__iter__()

    def __getattr__(self, item):
        if "interaction" not in self.__dict__:
            raise AttributeError(f"'Interaction' object has no attribute 'interaction'")
        if item in self.interaction:
            return self.interaction[item]
        raise AttributeError(f"'Interaction' object has no attribute '{item}'")

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        if isinstance(index, (np.ndarray, torch.Tensor)):
            index = index.tolist()

        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k][index]
        return Interaction(ret)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError(f"{type(key)} object does not support item assigment")
        self.interaction[key] = value

    def __delitem__(self, key):
        if key not in self.interaction:
            raise KeyError(f"{type(key)} object does not in this interaction")
        del self.interaction[key]

    def __contains__(self, item):
        return item in self.interaction

    def __len__(self):
        return self.length

    def __str__(self):
        info = [f"The batch_size of interaction: {self.length}"]
        for k in self.interaction:
            inter = self.interaction[k]
            temp_str = f"    {k}, {inter.shape}, {inter.device.type}, {inter.dtype}"
            info.append(temp_str)
        info.append("\n")
        return "\n".join(info)

    def __repr__(self):
        return self.__str__()

    @property
    def columns(self):
        """
        Returns:
            list of str: The columns of interaction.
        """
        return list(self.interaction.keys())

    def to(self, device, selected_field=None):
        """Transfer Tensors in this Interaction object to the specified device.

        Args:
            device (torch.device), str: target device.
            selected_field (str or iterable object, optional): if specified, only Tensors
            with keys in selected_field will be sent to device.

        Returns:
            Interaction: a coped Interaction object with Tensors which are sent to
            the specified device.
        """
        ret = {}
        if isinstance(selected_field, str):
            selected_field = [selected_field]

        if selected_field is not None:
            selected_field = set(selected_field)
            for k in self.interaction:
                if k in selected_field:
                    ret[k] = self.interaction[k].to(device)
                else:
                    ret[k] = self.interaction[k]
        else:
            for k in self.interaction:
                ret[k] = self.interaction[k].to(device)
        return Interaction(ret)

    def cpu(self):
        """Transfer Tensors in this Interaction object to cpu.

        Returns:
            Interaction: a coped Interaction object with Tensors which are sent to cpu.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].cpu()
        return Interaction(ret)

    def numpy(self):
        """Transfer Tensors to numpy arrays.

        Returns:
            dict: keys the same as Interaction object, are values are corresponding numpy
            arrays transformed from Tensor.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].numpy()
        return ret

    def repeat(self, sizes):
        """Repeats each tensor along the batch dim.

        Args:
            sizes (int): repeat times.

        Example:
            >>> a = Interaction({'k': torch.zeros(4)})
            >>> a.repeat(3)
            The batch_size of interaction: 12
                k, torch.Size([12]), cpu

            >>> a = Interaction({'k': torch.zeros(4, 7)})
            >>> a.repeat(3)
            The batch_size of interaction: 12
                k, torch.Size([12, 7]), cpu

        Returns:
            a copyed Interaction object with repeated Tensors.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat(
                [sizes] + [1] * (len(self.interaction[k].shape) - 1)
            )
        return Interaction(ret)

    def repeat_interleave(self, repeats, dim=0):
        """Similar to repeat_interleave of PyTorch.

        Details can be found in:

            https://pytorch.org/docs/stable/tensors.html?highlight=repeat#torch.Tensor.repeat_interleave

        Note:
            ``torch.repeat_interleave()`` is supported in PyTorch >= 1.2.0.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat_interleave(repeats, dim=dim)
        return Interaction(ret)

    def update(self, new_inter):
        """Similar to ``dict.update()``

        Args:
            new_inter (Interaction): current interaction will be updated by new_inter.
        """
        for k in new_inter.interaction:
            self.interaction[k] = new_inter.interaction[k]

    def drop(self, column):
        """Drop column in interaction.

        Args:
            column (str): the column to be dropped.
        """
        if column not in self.interaction:
            raise ValueError(f"Column [{column}] is not in [{self}].")
        del self.interaction[column]

    def _reindex(self, index):
        """Reset the index of interaction inplace.

        Args:
            index: the new index of current interaction.
        """
        for k in self.interaction:
            self.interaction[k] = self.interaction[k][index]

    def shuffle(self):
        """Shuffle current interaction inplace."""
        index = torch.randperm(self.length)
        self._reindex(index)

    def sort(self, by, ascending=True):
        """Sort the current interaction inplace.

        Args:
            by (str or list of str): Field that as the key in the sorting process.
            ascending (bool or list of bool, optional): Results are ascending if ``True``, otherwise descending.
                Defaults to ``True``
        """
        if isinstance(by, str):
            if by not in self.interaction:
                raise ValueError(f"[{by}] is not exist in interaction [{self}].")
            by = [by]
        elif isinstance(by, (list, tuple)):
            for b in by:
                if b not in self.interaction:
                    raise ValueError(f"[{b}] is not exist in interaction [{self}].")
        else:
            raise TypeError(f"Wrong type of by [{by}].")

        if isinstance(ascending, bool):
            ascending = [ascending]
        elif isinstance(ascending, (list, tuple)):
            for a in ascending:
                if not isinstance(a, bool):
                    raise TypeError(f"Wrong type of ascending [{ascending}].")
        else:
            raise TypeError(f"Wrong type of ascending [{ascending}].")

        if len(by) != len(ascending):
            if len(ascending) == 1:
                ascending = ascending * len(by)
            else:
                raise ValueError(
                    f"by [{by}] and ascending [{ascending}] should have same length."
                )

        for b, a in zip(by[::-1], ascending[::-1]):
            if len(self.interaction[b].shape) == 1:
                key = self.interaction[b]
            else:
                key = self.interaction[b][..., 0]
            index = np.argsort(key, kind="stable")
            if not a:
                index = torch.tensor(np.array(index)[::-1])
            self._reindex(index)

    def add_prefix(self, prefix):
        """Add prefix to current interaction's columns.

        Args:
            prefix (str): The prefix to be added.
        """
        self.interaction = {
            prefix + key: value for key, value in self.interaction.items()
        }


def cat_interactions(interactions):
    """Concatenate list of interactions to single interaction.

    Args:
        interactions (list of :class:`Interaction`): List of interactions to be concatenated.

    Returns:
        :class:`Interaction`: Concatenated interaction.
    """
    if not isinstance(interactions, (list, tuple)):
        raise TypeError(f"Interactions [{interactions}] should be list or tuple.")
    if len(interactions) == 0:
        raise ValueError(
            f"Interactions [{interactions}] should have some interactions."
        )

    columns_set = set(interactions[0].columns)
    for inter in interactions:
        if columns_set != set(inter.columns):
            raise ValueError(
                f"Interactions [{interactions}] should have some interactions."
            )

    new_inter = {
        col: torch.cat([inter[col] for inter in interactions]) for col in columns_set
    }
    return Interaction(new_inter)
</file>

<file path="data/__init__.py">
"""Data utilities for rosen."""

from .dataset import (
    DataScanner,
    DataTransformer,
    ParquetStreamDataset,
    PipelineStreamDataset,
    ProfileParquetStreamDataset,
    ProfileSequenceStore,
    RawParquetStreamDataset,
    ShuffleBuffer,
)
from .padding import FeatureContext, TensorFormatter
from .profile import ProfileDatasetBundle, ProfileDatasetLayout

__all__ = [
    "DataScanner",
    "DataTransformer",
    "FeatureContext",
    "ParquetStreamDataset",
    "ProfileDatasetBundle",
    "ProfileDatasetLayout",
    "PipelineStreamDataset",
    "ProfileParquetStreamDataset",
    "ProfileSequenceStore",
    "RawParquetStreamDataset",
    "ShuffleBuffer",
    "TensorFormatter",
]
</file>

<file path="data/dataset.py">
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from rosen.emb import SchemaManager
from .padding import ColumnFormatter, FeatureContext, TensorFormatter


ScannerSource = Union[str, Path, Sequence[Union[str, Path]], pl.LazyFrame]


def _resolve_parquet_paths(source: Union[str, Path, Sequence[Union[str, Path]]]) -> list[str]:
    if isinstance(source, (list, tuple)):
        paths = [str(Path(path)) for path in source]
    else:
        path_obj = Path(source)
        if path_obj.is_dir():
            paths = [str(path) for path in sorted(path_obj.glob("*.parquet"))]
        else:
            paths = [str(path_obj)]

    if not paths:
        raise FileNotFoundError(f"No parquet files found from input: {source}")
    return paths


def _parquet_top_level_columns(path: Union[str, Path]) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _fill_null_scalar(col_type: pa.DataType):
    if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
        return pa.scalar("", type=col_type)
    if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
        return pa.scalar([], type=col_type)
    if pa.types.is_boolean(col_type):
        return pa.scalar(False, type=col_type)
    return pa.scalar(0, type=col_type)


def arrow_table_to_batch_dict(
    table: pa.Table,
    selected_col_names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    selected_set = set(selected_col_names) if selected_col_names is not None else None
    batch_dict: Dict[str, np.ndarray] = {}
    for col_name, col_data in zip(table.column_names, table.columns):
        if selected_set is not None and col_name not in selected_set:
            continue

        clean_col = pc.fill_null(col_data, _fill_null_scalar(col_data.type))
        batch_dict[col_name] = clean_col.to_numpy(zero_copy_only=False)
    return batch_dict


def _concat_tables(tables: Sequence[pa.Table]) -> Optional[pa.Table]:
    non_empty_tables = [table for table in tables if table is not None and table.num_rows > 0]
    if not non_empty_tables:
        return None
    if len(non_empty_tables) == 1:
        return non_empty_tables[0]
    return pa.concat_tables(non_empty_tables)


class DataScanner:
    def __init__(
        self,
        source: ScannerSource,
        read_cols: Optional[Iterable[str]] = None,
        filter_expr: Optional[pl.Expr] = None,
    ):
        self.source = source
        self.read_cols = list(dict.fromkeys(read_cols or [])) or None
        self.filter_expr = filter_expr
        self.parquet_paths = None if isinstance(source, pl.LazyFrame) else _resolve_parquet_paths(source)
        self.lazy_frame = self._build_lazy_frame(source)

    def _build_lazy_frame(self, source: ScannerSource) -> pl.LazyFrame:
        if isinstance(source, pl.LazyFrame):
            lf = source
        else:
            lf = pl.scan_parquet(self.parquet_paths)

        if self.filter_expr is not None:
            lf = lf.filter(self.filter_expr)
        if self.read_cols is not None:
            lf = lf.select(self.read_cols)
        return lf

    def _apply_worker_shard(self, lf: pl.LazyFrame, worker_id: int, num_workers: int) -> pl.LazyFrame:
        if num_workers <= 1:
            return lf
        worker_col = "__scanner_worker_row_id"
        return (
            lf.with_row_index(worker_col)
            .filter((pl.col(worker_col) % num_workers) == worker_id)
            .drop(worker_col)
        )

    def _iter_lazy_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
        lf = self._apply_worker_shard(self.lazy_frame, worker_id=worker_id, num_workers=num_workers)
        for batch_df in lf.collect_batches(
            chunk_size=batch_size,
            maintain_order=True,
            engine="streaming",
        ):
            if batch_df.height == 0:
                continue
            yield batch_df.to_arrow()

    def _iter_parquet_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
        row_group_offset = 0
        for parquet_file in self.parquet_paths or []:
            parquet_file_reader = pq.ParquetFile(parquet_file)
            row_group_count = parquet_file_reader.metadata.num_row_groups
            if num_workers > 1:
                row_groups = [
                    row_group_idx
                    for row_group_idx in range(row_group_count)
                    if (row_group_offset + row_group_idx) % num_workers == worker_id
                ]
                if not row_groups:
                    row_group_offset += row_group_count
                    continue
            else:
                row_groups = None

            for record_batch in parquet_file_reader.iter_batches(
                batch_size=batch_size,
                row_groups=row_groups,
                columns=self.read_cols,
            ):
                yield pa.Table.from_batches([record_batch])
            row_group_offset += row_group_count

    def iter_batches(
        self,
        batch_size: int,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> Iterator[pa.Table]:
        if self.parquet_paths is not None and self.filter_expr is None:
            yield from self._iter_parquet_batches(batch_size, worker_id, num_workers)
            return

        yield from self._iter_lazy_batches(batch_size, worker_id, num_workers)


class DataTransformer:
    def __init__(
        self,
        manager: SchemaManager,
        preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
        filter_expr: Optional[pl.Expr] = None,
        output_col_names: Optional[Iterable[str]] = None,
    ):
        self.manager = manager
        self.preprocess_exprs = list(preprocess_exprs or [])
        self.filter_expr = filter_expr
        self.output_col_names = list(dict.fromkeys(output_col_names or manager.fields()))

    def __call__(self, raw_table: pa.Table) -> Optional[pa.Table]:
        if raw_table.num_rows == 0:
            return None

        raw_df = pl.from_arrow(raw_table)
        if self.preprocess_exprs:
            raw_df = raw_df.with_columns(self.preprocess_exprs)
        if self.filter_expr is not None:
            raw_df = raw_df.filter(self.filter_expr)
            if raw_df.height == 0:
                return None

        transformed_df = (
            self.manager.transform(raw_df.lazy())
            .select(self.output_col_names)
            .collect(engine="streaming")
        )
        if transformed_df.height == 0:
            return None
        return transformed_df.to_arrow()


class ShuffleBuffer:
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = True,
        selected_col_names: Optional[Sequence[str]] = None,
    ):
        self.capacity = max(batch_size, int(capacity))
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.selected_col_names = list(selected_col_names) if selected_col_names is not None else None
        self.reset()

    def reset(self):
        self.buffer: list[pa.Table] = []
        self.current_rows = 0

    def push(self, table: Optional[pa.Table]):
        if table is None or table.num_rows == 0:
            return
        self.buffer.append(table)
        self.current_rows += table.num_rows

    def _drain(self, force: bool) -> Iterator[Dict[str, np.ndarray]]:
        table = _concat_tables(self.buffer)
        if table is None or table.num_rows == 0:
            self.reset()
            return

        total_rows = table.num_rows
        batch_dict = arrow_table_to_batch_dict(table, selected_col_names=self.selected_col_names)
        indices = np.random.permutation(total_rows) if self.shuffle else np.arange(total_rows)
        num_complete_batches = total_rows // self.batch_size

        for batch_idx in range(num_complete_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]
            yield {col: arr[batch_indices] for col, arr in batch_dict.items()}

        residual_start = num_complete_batches * self.batch_size
        residual_indices = indices[residual_start:]
        residual_count = len(residual_indices)

        self.reset()
        if residual_count == 0:
            return

        if force:
            if not self.drop_last:
                yield {col: arr[residual_indices] for col, arr in batch_dict.items()}
            return

        residual_table = table.take(pa.array(residual_indices, type=pa.int64()))
        self.buffer = [residual_table]
        self.current_rows = residual_table.num_rows

    def yield_batches(self) -> Iterator[Dict[str, np.ndarray]]:
        if self.current_rows < self.capacity:
            return
        yield from self._drain(force=False)

    def flush(self) -> Iterator[Dict[str, np.ndarray]]:
        yield from self._drain(force=True)


class PipelineStreamDataset(IterableDataset):
    def __init__(
        self,
        scanner: DataScanner,
        formatter: TensorFormatter,
        transformer: Optional[DataTransformer] = None,
        buffer: Optional[ShuffleBuffer] = None,
        scan_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.scanner = scanner
        self.transformer = transformer
        self.buffer = buffer
        self.formatter = formatter
        self.scan_batch_size = scan_batch_size or (buffer.batch_size if buffer is not None else 4096)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if self.buffer is not None:
            self.buffer.reset()

        for raw_table in self.scanner.iter_batches(
            batch_size=self.scan_batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            table = self.transformer(raw_table) if self.transformer else raw_table
            if table is None or table.num_rows == 0:
                continue

            if self.buffer is None:
                yield self.formatter.format(
                    arrow_table_to_batch_dict(table, selected_col_names=self.formatter.context.output_col_names)
                )
                continue

            self.buffer.push(table)
            for batch_dict in self.buffer.yield_batches():
                yield self.formatter.format(batch_dict)

        if self.buffer is not None:
            for batch_dict in self.buffer.flush():
                yield self.formatter.format(batch_dict)


class ParquetStreamDataset(PipelineStreamDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ):
        context = FeatureContext.from_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        scanner = DataScanner(parquet_path, read_cols=context.read_col_names)
        formatter = TensorFormatter(context)
        buffer = ShuffleBuffer(
            capacity=shuffle_buffer_size if shuffle else batch_size,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=context.output_col_names,
        )
        super().__init__(
            scanner=scanner,
            formatter=formatter,
            transformer=None,
            buffer=buffer,
            scan_batch_size=batch_size * 4 if shuffle else batch_size,
        )

        self.parquet_path = parquet_path
        self.parquet_paths = list(scanner.parquet_paths or [])
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trigger_size = buffer.capacity
        self.context = context
        self.output_col_names = list(context.output_col_names)
        self.valid_col_names = list(context.output_col_names)
        self.read_col_names = list(context.read_col_names)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.col_formatters = formatter.col_formatters


def _empty_object_sequences(batch_size: int) -> np.ndarray:
    values = np.empty(batch_size, dtype=object)
    for idx in range(batch_size):
        values[idx] = []
    return values


def _default_missing_column(name: str, formatter: ColumnFormatter, batch_size: int) -> np.ndarray:
    formatter_name = formatter.__class__.__name__
    if formatter_name.startswith("Padded"):
        return _empty_object_sequences(batch_size)
    if formatter_name in {"DenseFormatter"}:
        return np.zeros(batch_size, dtype=np.float32)
    if formatter_name in {"VectorDenseFormatter"}:
        dim = int(getattr(formatter, "dim", 0))
        return np.zeros((batch_size, dim), dtype=np.float32)
    return np.zeros(batch_size, dtype=np.int64)


class ProfileSequenceStore:
    HISTORY_TIME_FILTERS = {
        "cursor_order",
        "strict_time",
        "drop_same_ts",
        "same_ts_only",
    }

    def __init__(
        self,
        manager: SchemaManager,
        profile_root: Union[str, Path],
        group_dirs: Optional[Mapping[str, Union[str, Path]]] = None,
    ):
        self.manager = manager
        self.profile_root = Path(profile_root)
        self.group_dirs = {str(k): Path(v) for k, v in dict(group_dirs or {}).items()}
        self.groups = self._compile_groups()
        self.tables = {
            group_name: self._load_group(group_name, spec)
            for group_name, spec in self.groups.items()
        }

    def _compile_groups(self) -> Dict[str, dict]:
        groups: Dict[str, dict] = {}
        for setting in self.manager.settings:
            if not setting.is_sequence_setting:
                continue
            group_name = setting.group_name
            group = setting.group
            spec = groups.setdefault(
                group_name,
                {
                    "owner_col": getattr(group, "owner_field_name", None) or self.manager.uid_field,
                    "seq_len_col": setting.seq_len_field_name,
                    "max_len": int(setting.max_len),
                    "fields": [],
                    "time_field": setting.time_field_name,
                    "anchor_col": getattr(group, "anchor_field_name", None),
                    "history_time_filter": getattr(
                        group,
                        "history_time_filter",
                        "cursor_order",
                    ),
                    "truncate_mode": setting.truncate_mode,
                },
            )
            shared_values = {
                "owner_col": getattr(group, "owner_field_name", None) or self.manager.uid_field,
                "seq_len_col": setting.seq_len_field_name,
                "max_len": int(setting.max_len),
                "time_field": setting.time_field_name,
                "anchor_col": getattr(group, "anchor_field_name", None),
                "history_time_filter": getattr(group, "history_time_filter", "cursor_order"),
                "truncate_mode": setting.truncate_mode,
            }
            for key, value in shared_values.items():
                if spec[key] != value:
                    raise ValueError(
                        f"Profile sequence group '{group_name}' has inconsistent "
                        f"{key}: {spec[key]!r} != {value!r}."
                    )
            if setting.field_name not in spec["fields"]:
                spec["fields"].append(setting.field_name)
            if setting.time_field_name and setting.time_field_name not in spec["fields"]:
                spec["fields"].append(setting.time_field_name)
            if spec["history_time_filter"] not in self.HISTORY_TIME_FILTERS:
                raise ValueError(
                    f"Unknown history_time_filter {spec['history_time_filter']!r} "
                    f"for profile sequence group '{group_name}'."
                )
        return groups

    def _resolve_group_dir(self, group_name: str) -> Path:
        if group_name in self.group_dirs:
            return self.group_dirs[group_name]
        return self.profile_root / group_name

    @staticmethod
    def _profile_value_to_array(value) -> np.ndarray:
        if value is None:
            return np.asarray([], dtype=np.int64)
        try:
            return np.asarray(value, dtype=np.int64)
        except (TypeError, ValueError):
            return np.asarray(value, dtype=object)

    def _load_group(self, group_name: str, spec: dict) -> Dict[int, Dict[str, np.ndarray]]:
        group_dir = self._resolve_group_dir(group_name)
        if group_dir.is_dir():
            files = sorted(group_dir.glob("*.parquet"))
        else:
            files = [group_dir] if group_dir.exists() else []
        if not files:
            raise FileNotFoundError(f"No profile parquet files found for group '{group_name}' under {group_dir}")

        read_cols = [spec["owner_col"], *spec["fields"]]
        df = pl.read_parquet([str(path) for path in files], columns=read_cols)
        table: Dict[int, Dict[str, np.ndarray]] = {}
        for row in df.iter_rows(named=True):
            uid = int(row[spec["owner_col"]])
            table[uid] = {}
            for field in spec["fields"]:
                value = row[field]
                table[uid][field] = self._profile_value_to_array(value)
        return table

    def inject(self, batch_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.groups:
            return batch_dict
        batch_size = len(next(iter(batch_dict.values())))

        for group_name, spec in self.groups.items():
            owner_col = spec["owner_col"]
            seq_len_col = spec["seq_len_col"]
            if owner_col not in batch_dict or seq_len_col not in batch_dict:
                raise KeyError(
                    f"Profile group '{group_name}' requires columns "
                    f"'{owner_col}' and '{seq_len_col}' in main batch."
                )

            users = np.asarray(batch_dict[owner_col]).reshape(-1)
            cursors = np.asarray(batch_dict[seq_len_col]).reshape(-1)
            table = self.tables[group_name]
            max_len = int(spec["max_len"])
            history_time_filter = spec["history_time_filter"]
            time_field = spec["time_field"]
            anchor_col = spec["anchor_col"]
            anchors = None
            if history_time_filter != "cursor_order":
                if not time_field or not anchor_col:
                    raise ValueError(
                        f"Profile group '{group_name}' requires time_field_name and "
                        f"anchor_field_name for history_time_filter={history_time_filter!r}."
                    )
                if anchor_col not in batch_dict:
                    raise KeyError(
                        f"Profile group '{group_name}' requires anchor column "
                        f"'{anchor_col}' in main batch."
                    )
                anchors = np.asarray(batch_dict[anchor_col]).reshape(-1)

            sliced_fields = {
                field: np.empty(batch_size, dtype=object)
                for field in spec["fields"]
            }
            actual_lens = np.zeros(batch_size, dtype=np.int64)

            for row_idx, (uid, cursor) in enumerate(zip(users, cursors)):
                profile = table.get(int(uid))
                cursor_int = max(0, int(cursor))
                base_field = spec["fields"][0]
                base_seq = profile.get(base_field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
                end = min(cursor_int, len(base_seq))
                selected_indices = np.arange(end, dtype=np.int64)

                if history_time_filter != "cursor_order":
                    history_time = profile.get(time_field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
                    if len(history_time) < end:
                        raise ValueError(
                            f"Profile group '{group_name}' has {len(history_time)} "
                            f"timestamps for cursor {end} (user={int(uid)})."
                        )
                    candidate_time = history_time[:end]
                    anchor = anchors[row_idx]
                    if history_time_filter in {"strict_time", "drop_same_ts"}:
                        selected_indices = selected_indices[candidate_time < anchor]
                    else:
                        selected_indices = selected_indices[candidate_time == anchor]

                if spec["truncate_mode"] == "head":
                    selected_indices = selected_indices[:max_len]
                elif spec["truncate_mode"] == "tail":
                    selected_indices = selected_indices[-max_len:]
                else:
                    raise ValueError(
                        f"Unknown truncate_mode {spec['truncate_mode']!r} "
                        f"for profile sequence group '{group_name}'."
                    )
                actual_lens[row_idx] = len(selected_indices)

                for field in spec["fields"]:
                    full_seq = profile.get(field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
                    if len(full_seq) < end:
                        raise ValueError(
                            f"Profile group '{group_name}' field '{field}' has "
                            f"length {len(full_seq)} for cursor {end} (user={int(uid)})."
                        )
                    sliced_fields[field][row_idx] = full_seq[selected_indices]

            batch_dict.update(sliced_fields)
            batch_dict[seq_len_col] = actual_lens

        return batch_dict


class ProfileParquetStreamDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        profile_root: Union[str, Path],
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
        group_dirs: Optional[Mapping[str, Union[str, Path]]] = None,
        profile_store: Optional[ProfileSequenceStore] = None,
    ):
        super().__init__()
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.context = FeatureContext.from_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        self.formatter = TensorFormatter(self.context)
        self.profile_store = (
            profile_store
            if profile_store is not None
            else ProfileSequenceStore(
                manager=manager,
                profile_root=profile_root,
                group_dirs=group_dirs,
            )
        )

        parquet_paths = _resolve_parquet_paths(parquet_path)
        available_cols = _parquet_top_level_columns(parquet_paths[0])
        read_cols = [col for col in self.context.output_col_names if col in available_cols]
        for spec in self.profile_store.groups.values():
            required_cols = [spec["owner_col"], spec["seq_len_col"]]
            if spec["history_time_filter"] != "cursor_order":
                required_cols.append(spec["anchor_col"])
            for required_col in required_cols:
                if required_col in available_cols and required_col not in read_cols:
                    read_cols.append(required_col)

        self.scanner = DataScanner(parquet_path, read_cols=read_cols)
        self.buffer = ShuffleBuffer(
            capacity=shuffle_buffer_size if shuffle else batch_size,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=None,
        )
        self.scan_batch_size = batch_size * 4 if shuffle else batch_size
        self.parquet_path = parquet_path
        self.parquet_paths = parquet_paths
        self.profile_root = Path(profile_root)
        self.output_col_names = list(self.context.output_col_names)
        self.valid_col_names = list(self.context.output_col_names)
        self.read_col_names = list(read_cols)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.trigger_size = self.buffer.capacity
        self.col_formatters = self.formatter.col_formatters

    def _complete_batch(self, batch_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not batch_dict:
            return batch_dict
        batch_size = len(next(iter(batch_dict.values())))
        self.profile_store.inject(batch_dict)
        for col in self.context.output_col_names:
            if col in batch_dict:
                continue
            formatter = self.formatter.col_formatters.get(col, self.formatter.fallback_formatter)
            batch_dict[col] = _default_missing_column(col, formatter, batch_size)
        return {col: batch_dict[col] for col in self.context.output_col_names if col in batch_dict}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        self.buffer.reset()

        for raw_table in self.scanner.iter_batches(
            batch_size=self.scan_batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            self.buffer.push(raw_table)
            for batch_dict in self.buffer.yield_batches():
                yield self.formatter.format(self._complete_batch(batch_dict))

        for batch_dict in self.buffer.flush():
            yield self.formatter.format(self._complete_batch(batch_dict))


class RawParquetStreamDataset(PipelineStreamDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
        transform_buffer_size: Optional[int] = None,
        raw_filter_expr: Optional[pl.Expr] = None,
        raw_preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
    ):
        preprocess_exprs = list(raw_preprocess_exprs or [])
        scanner_filter_expr = raw_filter_expr if not preprocess_exprs else None
        transformer_filter_expr = None if scanner_filter_expr is not None else raw_filter_expr
        context = FeatureContext.from_raw_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        scanner = DataScanner(
            parquet_path,
            read_cols=context.read_col_names,
            filter_expr=scanner_filter_expr,
        )
        formatter = TensorFormatter(context)
        transformer = DataTransformer(
            manager=manager,
            preprocess_exprs=preprocess_exprs,
            filter_expr=transformer_filter_expr,
            output_col_names=context.output_col_names,
        )
        capacity = transform_buffer_size
        if capacity is None:
            capped_rows = batch_size * (8 if shuffle else 1)
            capacity = max(batch_size, min(shuffle_buffer_size, capped_rows))
        buffer = ShuffleBuffer(
            capacity=capacity,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=context.output_col_names,
        )
        super().__init__(
            scanner=scanner,
            formatter=formatter,
            transformer=transformer,
            buffer=buffer,
            scan_batch_size=batch_size * 2 if shuffle else batch_size,
        )

        self.parquet_path = parquet_path
        self.parquet_paths = list(scanner.parquet_paths or [])
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trigger_size = buffer.capacity
        self.context = context
        self.output_col_names = list(context.output_col_names)
        self.valid_col_names = list(context.output_col_names)
        self.read_col_names = list(context.read_col_names)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.raw_filter_expr = raw_filter_expr
        self.raw_preprocess_exprs = preprocess_exprs
        self.transform_buffer_size = buffer.capacity
        self.col_formatters = formatter.col_formatters
</file>

<file path="data/padding.py">
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Literal, Optional

import numpy as np
import torch

from rosen.core.interaction import Interaction

if TYPE_CHECKING:
    from rosen.emb import SchemaManager


PaddingSide = Literal["left", "right"]


def _is_missing_sequence(value) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _pad_list(
    sequences,
    max_len: int,
    padding_side: PaddingSide = "right",
    pad_val=0,
    dtype=np.int64,
) -> np.ndarray:
    padded = np.full((len(sequences), max_len), pad_val, dtype=dtype)
    for i, seq in enumerate(sequences):
        if _is_missing_sequence(seq) or len(seq) == 0:
            continue

        seq_list = list(seq)
        valid_len = min(len(seq_list), max_len)
        trunc_seq = seq_list[-valid_len:]
        if padding_side == "right":
            padded[i, :valid_len] = trunc_seq
        elif padding_side == "left":
            padded[i, max_len - valid_len:] = trunc_seq
        else:
            raise ValueError(f"Unknown padding_side: {padding_side}")

    return padded


def _pad_nested_list(
    sequences,
    max_seq_len: int,
    max_tag_len: int,
    padding_side: PaddingSide = "right",
    pad_val: int = 0,
) -> np.ndarray:
    padded = np.full((len(sequences), max_seq_len, max_tag_len), pad_val, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if _is_missing_sequence(seq) or len(seq) == 0:
            continue

        seq_list = list(seq)
        valid_seq_len = min(len(seq_list), max_seq_len)
        trunc_seq = seq_list[-valid_seq_len:]
        start_idx = 0 if padding_side == "right" else max_seq_len - valid_seq_len

        for j in range(valid_seq_len):
            tags = trunc_seq[j]
            if _is_missing_sequence(tags) or len(tags) == 0:
                continue

            tags_list = list(tags)
            valid_tag_len = min(len(tags_list), max_tag_len)
            trunc_tags = tags_list[:valid_tag_len]
            padded[i, start_idx + j, :valid_tag_len] = trunc_tags

    return padded


class ColumnFormatter(ABC):
    @abstractmethod
    def format(self, data) -> torch.Tensor:
        pass


@dataclass(frozen=True)
class DenseFormatter(ColumnFormatter):
    def format(self, data) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)


@dataclass(frozen=True)
class VectorDenseFormatter(ColumnFormatter):
    dim: int
    zero_fill: bool = True

    def format(self, data) -> torch.Tensor:
        arr = np.zeros((len(data), self.dim), dtype=np.float32)
        for i, row in enumerate(data):
            if _is_missing_sequence(row) or len(row) == 0:
                if not self.zero_fill:
                    raise ValueError(
                        f"Expected non-empty vector dim={self.dim}, but got an empty value at row {i}."
                    )
                continue

            row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
            use_len = min(row_arr.shape[0], self.dim)
            arr[i, :use_len] = row_arr[:use_len]

        return torch.tensor(arr, dtype=torch.float32)


@dataclass(frozen=True)
class IntFormatter(ColumnFormatter):
    def format(self, data) -> torch.Tensor:
        if data.dtype == np.uint32:
            data = data.astype(np.int64)
        return torch.tensor(data, dtype=torch.long)


@dataclass(frozen=True)
class FallbackFormatter(ColumnFormatter):
    def format(self, data) -> torch.Tensor:
        if data.dtype in (np.float32, np.float64):
            return torch.tensor(data, dtype=torch.float32)
        if data.dtype == np.uint32:
            return torch.tensor(data.astype(np.int64), dtype=torch.long)
        return torch.tensor(data)


@dataclass(frozen=True)
class PaddedIntSequenceFormatter(ColumnFormatter):
    max_len: int
    padding_side: PaddingSide = "right"
    pad_val: int = 0

    def format(self, data) -> torch.Tensor:
        arr = _pad_list(
            data,
            max_len=self.max_len,
            padding_side=self.padding_side,
            pad_val=self.pad_val,
            dtype=np.int64,
        )
        return torch.tensor(arr, dtype=torch.long)


@dataclass(frozen=True)
class PaddedFloatSequenceFormatter(ColumnFormatter):
    max_len: int
    padding_side: PaddingSide = "right"
    pad_val: float = 0.0

    def format(self, data) -> torch.Tensor:
        arr = _pad_list(
            data,
            max_len=self.max_len,
            padding_side=self.padding_side,
            pad_val=self.pad_val,
            dtype=np.float32,
        )
        return torch.tensor(arr, dtype=torch.float32)


@dataclass(frozen=True)
class PaddedNestedSequenceFormatter(ColumnFormatter):
    max_seq_len: int
    max_tag_len: int
    padding_side: PaddingSide = "right"
    pad_val: int = 0

    def format(self, data) -> torch.Tensor:
        arr = _pad_nested_list(
            data,
            max_seq_len=self.max_seq_len,
            max_tag_len=self.max_tag_len,
            padding_side=self.padding_side,
            pad_val=self.pad_val,
        )
        return torch.tensor(arr, dtype=torch.long)


DEFAULT_FALLBACK_FORMATTER = FallbackFormatter()


def _dedupe_names(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(name for name in names if name))


def _resolve_raw_read_col_names(
    manager: "SchemaManager",
    extra_col_names: Iterable[str],
    extra_col_formatters: Dict[str, ColumnFormatter],
) -> list[str]:
    raw_col_names: list[str] = []
    for setting in manager.settings:
        raw_col_names.extend(setting.get_raw_field_names())

    raw_col_names.extend(manager.label_fields)
    raw_col_names.extend(manager.domain_fields)
    if manager.time_field:
        raw_col_names.append(manager.time_field)

    raw_col_names.extend(extra_col_names)
    raw_col_names.extend(extra_col_formatters.keys())
    return _dedupe_names(raw_col_names)


@dataclass(frozen=True)
class FeatureContext:
    manager: "SchemaManager"
    read_col_names: tuple[str, ...]
    output_col_names: tuple[str, ...]
    extra_col_formatters: Dict[str, ColumnFormatter] = field(default_factory=dict)

    @classmethod
    def from_manager(
        cls,
        manager: "SchemaManager",
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ) -> "FeatureContext":
        formatter_map = dict(extra_col_formatters or {})
        extra_names = list(extra_col_names or [])
        output_cols = _dedupe_names([*manager.fields(), *extra_names, *formatter_map.keys()])
        return cls(
            manager=manager,
            read_col_names=tuple(output_cols),
            output_col_names=tuple(output_cols),
            extra_col_formatters=formatter_map,
        )

    @classmethod
    def from_raw_manager(
        cls,
        manager: "SchemaManager",
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ) -> "FeatureContext":
        formatter_map = dict(extra_col_formatters or {})
        extra_names = list(extra_col_names or [])
        output_cols = _dedupe_names([*manager.fields(), *extra_names, *formatter_map.keys()])
        read_cols = _resolve_raw_read_col_names(manager, extra_names, formatter_map)
        return cls(
            manager=manager,
            read_col_names=tuple(read_cols),
            output_col_names=tuple(output_cols),
            extra_col_formatters=formatter_map,
        )


@dataclass
class TensorFormatter:
    context: FeatureContext
    fallback_formatter: ColumnFormatter = DEFAULT_FALLBACK_FORMATTER
    col_formatters: Dict[str, ColumnFormatter] = field(init=False)

    def __post_init__(self):
        self.col_formatters = self._compile_formatters()

    def _compile_formatters(self) -> Dict[str, ColumnFormatter]:
        formatters: Dict[str, ColumnFormatter] = {}
        for setting in self.context.manager.settings:
            formatters.update(setting.get_formatters())

        for ctx_col in (
            *self.context.manager.label_fields,
            *self.context.manager.domain_fields,
            self.context.manager.time_field,
        ):
            if ctx_col and ctx_col not in formatters:
                formatters[ctx_col] = self.fallback_formatter

        formatters.update(self.context.extra_col_formatters)
        return formatters

    def format_tensors(self, batch_dict) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        for col, data in batch_dict.items():
            formatter = self.col_formatters.get(col, self.fallback_formatter)
            try:
                tensor_dict[col] = formatter.format(data)
            except Exception as exc:
                raise RuntimeError(
                    f"列 [{col}] 在转换为 Tensor 时发生错误，匹配到的 formatter 为: {formatter}。错误信息: {exc}"
                ) from exc
        return tensor_dict

    def format(self, batch_dict) -> Interaction:
        return Interaction(self.format_tensors(batch_dict))
</file>

<file path="data/profile.py">
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
</file>

<file path="datasets/__init__.py">
"""Dataset definitions and readers."""
</file>

<file path="datasets/aliccp.py">
from rosen.datasets.base import DatasetBase
import polars as pl

sparse_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '216', '508', '509', '702', '853', '301', '109_14', '110_14', '127_14', '150_14']
dense_columns = ['109_14', '110_14', '127_14', '150_14', '508', '509', '702', '853']
multi_sparse_columns = ['210']

common_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '109_14', '110_14', '127_14', '150_14']
skeleton_columns = ['205', '206', '207', '210', '216', '301', '508', '509', '702', '853']

common_sparse_columns = [col for col in sparse_columns if col in common_columns]
common_dense_columns = [col for col in dense_columns if col in common_columns]

skeleton_sparse_columns = [col for col in sparse_columns if col in skeleton_columns]
skeleton_dense_columns = [col for col in dense_columns if col in skeleton_columns]
skeleton_multi_sparse_columns = ['210']

rename_dict = {
    "click": "click",
    "purchase": "purchase",
    # === User Features (用户特征) ===
    "101": "user_id",
    "D109_14": "user_hist_category_count",
    "D110_14": "user_hist_shop_count",
    "D127_14": "user_hist_brand_count",
    "D150_14": "user_hist_intention_count",
    "121": "user_profile_id",           # 用户画像分类id
    "122": "user_profile_group_id",
    "124": "user_gender_id",
    "125": "user_age_id",
    "126": "user_consumption_level_1",
    "127": "user_consumption_level_2",
    "128": "user_is_working",             # 职业：是否工作
    "129": "user_geography_info",

    # === Item Features (商品/物品特征) ===
    "205": "item_id",
    "206": "item_category_id",
    "207": "item_shop_id",
    "210": "item_intention_node_id",
    "216": "item_brand_id",

    # === Combination Features (交叉/组合特征) ===
    # 命名逻辑: combo_特征A_特征B
    "D508": "combo_hist_category_cross_item_category",   # 109_14 & 206
    "D509": "combo_hist_shop_cross_item_shop",           # 110_14 & 207
    "D702": "combo_hist_brand_cross_item_brand",         # 127_14 & 216
    "D853": "combo_hist_intention_cross_item_intention", # 150_14 & 210

    # === Context Features (上下文特征) ===
    "301": "domain_id"
}

class _ALIDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "Ali-CCP"
    TRAIN_DIR = BASE_DIR / "sample_train"
    TEST_DIR = BASE_DIR / "sample_test"

    TRAIN_USER_CTX_FEATURES = TRAIN_DIR / "common_features_train.csv"
    TRAIN_ITEM_INTERACTIONS = TRAIN_DIR / "sample_skeleton_train.csv"

    TEST_USER_CTX_FEATURES = TEST_DIR / "common_features_test.csv"
    TEST_ITEM_INTERACTIONS = TEST_DIR / "sample_skeleton_test.csv"

    TRAIN_WHOLE_INTERACTIONS = TRAIN_DIR / "train_whole_inter.parquet"
    TEST_WHOLE_INTERACTIONS = TEST_DIR / "test_whole_inter.parquet"

    @staticmethod
    def _parse_common_features(file_path) -> pl.LazyFrame:
        """
        使用 Polars LazyFrame 处理 Ali-CCP 的公共特征文件。
        全程无 Python 级别的 for 循环遍历，底层使用 Rust 多线程和 SIMD 正则匹配。
        """
        extract_exprs = []
        for col in common_sparse_columns:
            # 正则含义: 匹配开头或\x01，紧跟 field_id 和 \x02，捕获中间的特征值直到遇见 \x03
            pattern = rf"(?:^|\x01){col}\x02([^\x03]+)\x03"
            extract_exprs.append(
                pl.col("raw_features")
                .str.extract(pattern, 1)  # 1 表示提取第一个括号内的捕获组
                .fill_null("0")  # 你原代码中的 feat_dict.get(k, '0') 逻辑
                .alias(col)
            )

        for col in common_dense_columns:
            # 正则含义: 匹配 field_id 和 \x02，跳过特征值和 \x03，捕获后面的权重值直到遇见 \x01 或结尾
            pattern = rf"(?:^|\x01){col}\x02[^\x03]+\x03([^\x01]+)"
            extract_exprs.append(
                pl.col("raw_features")
                .str.extract(pattern, 1)
                .fill_null("0")
                .cast(pl.Float32)  # Dense 特征直接转换为浮点数，省去了后续的转换
                .alias(f"D{col}")  # 加上 D 前缀
            )
        lazy_df = (
            pl.scan_csv(
                file_path,
                has_header=False,
                new_columns=["common_id", "feature_num", "raw_features"]
            )
            .with_columns(extract_exprs)  # 一次性并发执行所有正则提取
            .drop(["feature_num", "raw_features"])  # 丢弃无用的原始列
        )
        return lazy_df

    @staticmethod
    def _parse_skeleton(file_path) -> pl.LazyFrame:
        """
        使用 Polars LazyFrame 解析样本骨架表，支持多值特征提取为 List
        """
        extract_exprs = []
        # 1. 提取单值稀疏特征 (Single Sparse) -> str
        for col in skeleton_sparse_columns:
            pattern = rf"(?:^|\x01){col}\x02([^\x03]+)\x03"
            extract_exprs.append(
                pl.col("raw_item_features")
                .str.extract(pattern, 1)
                .fill_null("0")  # 如果没找到该特征，填充默认值 "0"
                .alias(col)
            )
        # 2. 提取连续特征 (Dense) -> float32
        for col in skeleton_dense_columns:
            pattern = rf"(?:^|\x01){col}\x02[^\x03]+\x03([^\x01]+)"
            extract_exprs.append(
                pl.col("raw_item_features")
                .str.extract(pattern, 1)
                .fill_null("0")
                .cast(pl.Float32)  # 在内存分配前直接转换为小内存的 float32
                .alias(f"{col}")  # 加上 D 前缀
            )
        # 3. 提取多值稀疏特征 (Multi Sparse) -> List[str]
        for col in skeleton_multi_sparse_columns:
            extract_exprs.append(
                pl.col("raw_item_features")
                .str.split("\x01")  # 先炸开成数组
                .list.eval(
                    pl.element()
                    .filter(pl.element().str.starts_with(f"{col}\x02"))  # 只过滤出包含该 Field 的项
                    .str.extract(r"\x02([^\x03]+)", 1)  # 提取特征 ID
                )
                .alias(col)  # 这里的最终产物是一个真正的 List!
            )
        # 4. 构建 Lazy 计算图
        lf = (
            pl.scan_csv(
                file_path,
                has_header=False,
                new_columns=["sample_id", "click", "purchase", "common_id", "feature_num", "raw_item_features"],
                truncate_ragged_lines=True
            )
            .filter(
                ~((pl.col("click") == 0) & (pl.col("purchase") == 1))
            )
            .with_columns(extract_exprs)
            .drop(["sample_id", "feature_num", "raw_item_features"]) # 没必要的col
        )
        return lf

    @classmethod
    def _merge_table(cls, mode="train") -> pl.LazyFrame:
        """内部方法：负责 Join 宽表"""
        assert mode in ["train", "test"]
        if mode == "train":
            common_path = cls.TRAIN_USER_CTX_FEATURES
            skeleton_path = cls.TRAIN_ITEM_INTERACTIONS
        else:
            common_path = cls.TEST_USER_CTX_FEATURES
            skeleton_path = cls.TEST_ITEM_INTERACTIONS

        lf_common = cls._parse_common_features(common_path)
        lf_skeleton = cls._parse_skeleton(skeleton_path)

        # Join 并丢弃关联主键
        return lf_skeleton.join(
            lf_common, on="common_id", how="left"
        ).drop("common_id")

    @property
    def TRAIN_INTER_LF(self)-> pl.LazyFrame:
        if not self.TRAIN_WHOLE_INTERACTIONS.exists():
            self._merge_table("train").sink_parquet(self.TRAIN_WHOLE_INTERACTIONS)
        return pl.scan_parquet(self.TRAIN_WHOLE_INTERACTIONS).rename(rename_dict)
    @property
    def TEST_INTER_LF(self)-> pl.LazyFrame:
        if not self.TEST_WHOLE_INTERACTIONS.exists():
            self._merge_table("test").sink_parquet(self.TEST_WHOLE_INTERACTIONS)
        return pl.scan_parquet(self.TEST_WHOLE_INTERACTIONS).rename(rename_dict)

AliCCPDataset = _ALIDS()

# ================= 测试运行 =================
if __name__ == "__main__":
    lf_train = AliCCPDataset.TEST_INTER_LF

    print("准备打印前 5 行数据：")
    df_preview = lf_train.head(5).collect()
    print(df_preview)
    print(df_preview.columns)
    from rosen.utils import plot_power2_sparsity, plot_sparsity_ecdf, plot_bias_distributions
    plot_power2_sparsity(lf_train,"user_id", "item_id")
    plot_sparsity_ecdf(lf_train, "user_id", "item_id")
    plot_bias_distributions(lf_train, "user_id", "item_id", "click")
</file>

<file path="datasets/amazon.py">
import polars as pl

from rosen.datasets.base import DatasetBase


class _AmazonDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "Amazon"

    ALL_BEAUTY = BASE_DIR / "All_Beauty.csv"
    BOOKS = BASE_DIR / "Books.csv"
    CDS_AND_VINYL = BASE_DIR / "CDs_and_Vinyl.csv"
    CELL_PHONES_AND_ACCESSORIES = BASE_DIR / "Cell_Phones_and_Accessories.csv"
    CLOTHING_SHOES_AND_JEWELRY = BASE_DIR / "Clothing_Shoes_and_Jewelry.csv"
    GROCERY_AND_GOURMET_FOOD = BASE_DIR / "Grocery_and_Gourmet_Food.csv"
    HEALTH_AND_HOUSEHOLD = BASE_DIR / "Health_and_Household.csv"
    MOVIES_AND_TV = BASE_DIR / "Movies_and_TV.csv"
    MUSICAL_INSTRUMENTS = BASE_DIR / "Musical_Instruments.csv"
    PET_SUPPLIES = BASE_DIR / "Pet_Supplies.csv"
    SPORTS_AND_OUTDOORS = BASE_DIR / "Sports_and_Outdoors.csv"
    TOOLS_AND_HOME_IMPROVEMENT = BASE_DIR / "Tools_and_Home_Improvement.csv"
    TOYS_AND_GAMES = BASE_DIR / "Toys_and_Games.csv"
    VIDEO_GAMES = BASE_DIR / "Video_Games.csv"

    BEAUTY = ALL_BEAUTY
    CLOTHING = CLOTHING_SHOES_AND_JEWELRY
    HEALTH = HEALTH_AND_HOUSEHOLD

    AMAZON_DATASETS = {
        "All_Beauty": ALL_BEAUTY,
        "Beauty": BEAUTY,
        "Books": BOOKS,
        "CDs_and_Vinyl": CDS_AND_VINYL,
        "Cell_Phones_and_Accessories": CELL_PHONES_AND_ACCESSORIES,
        "Clothing_Shoes_and_Jewelry": CLOTHING_SHOES_AND_JEWELRY,
        "Clothing": CLOTHING,
        "Grocery_and_Gourmet_Food": GROCERY_AND_GOURMET_FOOD,
        "Health_and_Household": HEALTH_AND_HOUSEHOLD,
        "Health": HEALTH,
        "Movies_and_TV": MOVIES_AND_TV,
        "Musical_Instruments": MUSICAL_INSTRUMENTS,
        "Pet_Supplies": PET_SUPPLIES,
        "Sports_and_Outdoors": SPORTS_AND_OUTDOORS,
        "Tools_and_Home_Improvement": TOOLS_AND_HOME_IMPROVEMENT,
        "Toys_and_Games": TOYS_AND_GAMES,
        "Video_Games": VIDEO_GAMES,
    }

    ALL_BEAUTY_LF = pl.scan_csv(ALL_BEAUTY)
    BOOKS_LF = pl.scan_csv(BOOKS)
    CDS_AND_VINYL_LF = pl.scan_csv(CDS_AND_VINYL)
    CELL_PHONES_AND_ACCESSORIES_LF = pl.scan_csv(CELL_PHONES_AND_ACCESSORIES)
    CLOTHING_SHOES_AND_JEWELRY_LF = pl.scan_csv(CLOTHING_SHOES_AND_JEWELRY)
    GROCERY_AND_GOURMET_FOOD_LF = pl.scan_csv(GROCERY_AND_GOURMET_FOOD)
    HEALTH_AND_HOUSEHOLD_LF = pl.scan_csv(HEALTH_AND_HOUSEHOLD)
    MOVIES_AND_TV_LF = pl.scan_csv(MOVIES_AND_TV)
    MUSICAL_INSTRUMENTS_LF = pl.scan_csv(MUSICAL_INSTRUMENTS)
    PET_SUPPLIES_LF = pl.scan_csv(PET_SUPPLIES)
    SPORTS_AND_OUTDOORS_LF = pl.scan_csv(SPORTS_AND_OUTDOORS)
    TOOLS_AND_HOME_IMPROVEMENT_LF = pl.scan_csv(TOOLS_AND_HOME_IMPROVEMENT)
    TOYS_AND_GAMES_LF = pl.scan_csv(TOYS_AND_GAMES)
    VIDEO_GAMES_LF = pl.scan_csv(VIDEO_GAMES)

    BEAUTY_LF = ALL_BEAUTY_LF
    CLOTHING_LF = CLOTHING_SHOES_AND_JEWELRY_LF
    HEALTH_LF = HEALTH_AND_HOUSEHOLD_LF

    AMAZON_DATASET_LFS = {
        "All_Beauty": ALL_BEAUTY_LF,
        "Beauty": BEAUTY_LF,
        "Books": BOOKS_LF,
        "CDs_and_Vinyl": CDS_AND_VINYL_LF,
        "Cell_Phones_and_Accessories": CELL_PHONES_AND_ACCESSORIES_LF,
        "Clothing_Shoes_and_Jewelry": CLOTHING_SHOES_AND_JEWELRY_LF,
        "Clothing": CLOTHING_LF,
        "Grocery_and_Gourmet_Food": GROCERY_AND_GOURMET_FOOD_LF,
        "Health_and_Household": HEALTH_AND_HOUSEHOLD_LF,
        "Health": HEALTH_LF,
        "Movies_and_TV": MOVIES_AND_TV_LF,
        "Musical_Instruments": MUSICAL_INSTRUMENTS_LF,
        "Pet_Supplies": PET_SUPPLIES_LF,
        "Sports_and_Outdoors": SPORTS_AND_OUTDOORS_LF,
        "Tools_and_Home_Improvement": TOOLS_AND_HOME_IMPROVEMENT_LF,
        "Toys_and_Games": TOYS_AND_GAMES_LF,
        "Video_Games": VIDEO_GAMES_LF,
    }

    @classmethod
    def resolve_name(cls, name: str) -> str:
        normalized = str(name).strip()
        aliases = {
            "beauty": "Beauty",
            "all_beauty": "All_Beauty",
            "clothing": "Clothing",
            "clothing_shoes_and_jewelry": "Clothing_Shoes_and_Jewelry",
            "health": "Health",
            "health_and_household": "Health_and_Household",
        }
        return aliases.get(normalized.lower(), normalized)

    @classmethod
    def resolve_path(cls, name: str):
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON_DATASETS[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon dataset: {name!r}") from exc

    @classmethod
    def resolve_lf(cls, name: str):
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON_DATASET_LFS[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon dataset: {name!r}") from exc


AmazonDataset = _AmazonDS()


if __name__ == "__main__":
    books_lf = AmazonDataset.BOOKS_LF

    print("Books path:")
    print(AmazonDataset.BOOKS)

    print("\nBooks schema:")
    print(books_lf.collect_schema())

    print("\nBooks row count:")
    print(books_lf.select(pl.len().alias("rows")).collect())

    print("\nBooks head:")
    print(books_lf.head(5).collect())
</file>

<file path="datasets/amazon2014.py">
from __future__ import annotations

import ast
import gzip
import math
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from rosen.datasets.base import DatasetBase


class _Amazon2014DS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "Amazon2014"
    EXTRACTED_DIR = BASE_DIR / "extracted"
    METADATA_DIR = BASE_DIR / "metadata"

    BEAUTY_GZ = BASE_DIR / "reviews_Beauty_5.json.gz"
    CLOTHING_GZ = BASE_DIR / "reviews_Clothing_Shoes_and_Jewelry_5.json.gz"
    HEALTH_GZ = BASE_DIR / "reviews_Health_and_Personal_Care_5.json.gz"

    BEAUTY = EXTRACTED_DIR / "reviews_Beauty_5.json"
    CLOTHING = EXTRACTED_DIR / "reviews_Clothing_Shoes_and_Jewelry_5.json"
    HEALTH = EXTRACTED_DIR / "reviews_Health_and_Personal_Care_5.json"

    BEAUTY_METADATA = METADATA_DIR / "meta_Beauty.json"
    CLOTHING_METADATA = METADATA_DIR / "meta_Clothing_Shoes_and_Jewelry.json"
    HEALTH_METADATA = METADATA_DIR / "meta_Health_and_Personal_Care.json"

    AMAZON2014_DATASETS = {
        "Beauty": BEAUTY,
        "Clothing": CLOTHING,
        "Clothing_Shoes_and_Jewelry": CLOTHING,
        "Health": HEALTH,
        "Health_and_Personal_Care": HEALTH,
    }

    AMAZON2014_COMPRESSED_DATASETS = {
        "Beauty": BEAUTY_GZ,
        "Clothing": CLOTHING_GZ,
        "Clothing_Shoes_and_Jewelry": CLOTHING_GZ,
        "Health": HEALTH_GZ,
        "Health_and_Personal_Care": HEALTH_GZ,
    }

    AMAZON2014_METADATA = {
        "Beauty": BEAUTY_METADATA,
        "Clothing": CLOTHING_METADATA,
        "Clothing_Shoes_and_Jewelry": CLOTHING_METADATA,
        "Health": HEALTH_METADATA,
        "Health_and_Personal_Care": HEALTH_METADATA,
    }

    _metadata_cache: dict[str, pl.DataFrame] = {}

    @classmethod
    def resolve_name(cls, name: str) -> str:
        normalized = str(name).strip()
        aliases = {
            "beauty": "Beauty",
            "clothing": "Clothing",
            "clothing_shoes_and_jewelry": "Clothing_Shoes_and_Jewelry",
            "health": "Health",
            "health_and_personal_care": "Health_and_Personal_Care",
        }
        return aliases.get(normalized.lower(), normalized)

    @classmethod
    def resolve_names(cls, names: Sequence[str] | str | None = None) -> tuple[str, ...]:
        if names is None:
            return ("Beauty", "Clothing", "Health")
        if isinstance(names, str):
            return (cls.resolve_name(names),)
        return tuple(cls.resolve_name(name) for name in names)

    @classmethod
    def resolve_path(cls, name: str) -> Path:
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON2014_DATASETS[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon2014 dataset: {name!r}") from exc

    @classmethod
    def resolve_compressed_path(cls, name: str) -> Path:
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON2014_COMPRESSED_DATASETS[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon2014 dataset: {name!r}") from exc

    @classmethod
    def resolve_metadata_path(cls, name: str) -> Path:
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON2014_METADATA[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon2014 metadata dataset: {name!r}") from exc

    @staticmethod
    def _parse_price(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            price = float(value)
            return price if math.isfinite(price) else None
        text = str(value).strip()
        if not text:
            return None
        match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if match is None:
            return None
        try:
            price = float(match.group(0).replace(",", ""))
        except ValueError:
            return None
        return price if math.isfinite(price) else None

    @staticmethod
    def _normalize_categories(value: Any) -> list[list[str]]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [[stripped]] if stripped else []
        if not isinstance(value, (list, tuple)):
            return []

        categories: list[list[str]] = []
        for path in value:
            if isinstance(path, str):
                stripped = path.strip()
                if stripped:
                    categories.append([stripped])
                continue
            if not isinstance(path, (list, tuple)):
                continue
            cleaned = [str(part).strip() for part in path if part is not None and str(part).strip()]
            if cleaned:
                categories.append(cleaned)
        return categories

    @classmethod
    def _load_item_metadata(cls, name: str) -> pl.DataFrame:
        resolved = cls.resolve_name(name)
        cached = cls._metadata_cache.get(resolved)
        if cached is not None:
            return cached

        path = cls.resolve_metadata_path(resolved)
        if not path.exists():
            raise FileNotFoundError(path)

        asins: list[str] = []
        brands: list[str | None] = []
        prices: list[float | None] = []
        categories: list[list[list[str]]] = []
        seen: set[str] = set()
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = ast.literal_eval(text)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(f"Failed to parse Amazon2014 metadata line {line_no} in {path}") from exc
                if not isinstance(record, dict):
                    continue
                asin = record.get("asin")
                if asin is None:
                    continue
                asin = str(asin)
                if not asin or asin in seen:
                    continue
                seen.add(asin)
                asins.append(asin)
                brand = record.get("brand")
                brand = str(brand).strip() if brand is not None else None
                brands.append(brand or None)
                prices.append(cls._parse_price(record.get("price")))
                categories.append(cls._normalize_categories(record.get("categories")))

        df = pl.DataFrame(
            {
                "parent_asin": asins,
                "brand": brands,
                "price": prices,
                "categories": categories,
            },
            schema={
                "parent_asin": pl.Utf8,
                "brand": pl.Utf8,
                "price": pl.Float32,
                "categories": pl.List(pl.List(pl.Utf8)),
            },
        )
        cls._metadata_cache[resolved] = df
        return df

    @classmethod
    def scan_item_metadata(cls, names: Sequence[str] | str | None = None) -> pl.LazyFrame:
        """Return item-side metadata keyed by parent_asin.

        Brand, price, and categories are exposed. Callers that do not need metadata
        should keep include_metadata=False to avoid parsing the metadata files.
        """
        resolved_names = cls.resolve_names(names)
        if not resolved_names:
            raise ValueError("names must be non-empty.")

        frames = [cls._load_item_metadata(name) for name in resolved_names]
        df = pl.concat(frames, how="vertical")
        if len(frames) > 1:
            df = df.unique(subset=["parent_asin"], keep="first", maintain_order=True)
        return df.lazy()

    @classmethod
    def ensure_extracted(cls, names: Sequence[str] | str | None = None, redo: bool = False) -> list[Path]:
        resolved_names = cls.resolve_names(names)
        cls.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for name in resolved_names:
            src = cls.resolve_compressed_path(name)
            dst = cls.resolve_path(name)
            if dst.exists() and not redo:
                output_paths.append(dst)
                continue
            if not src.exists():
                raise FileNotFoundError(src)
            with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
            output_paths.append(dst)
        return output_paths

    @classmethod
    def scan_domain(cls, name: str, domain_id: int = 0, include_metadata: bool = False) -> pl.LazyFrame:
        resolved = cls.resolve_name(name)
        path = cls.resolve_path(resolved)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. Run Amazon2014Dataset.ensure_extracted() first.")
        lf = (
            pl.scan_ndjson(path)
            .select(
                pl.col("reviewerID").cast(pl.Utf8).alias("user_id"),
                pl.col("asin").cast(pl.Utf8).alias("parent_asin"),
                pl.col("overall").cast(pl.Float32).alias("rating"),
                pl.col("unixReviewTime").cast(pl.Int64).alias("timestamp"),
            )
            .with_columns(
                pl.lit(int(domain_id)).cast(pl.Int32).alias("domain_id"),
                pl.lit(str(resolved)).alias("domain_name"),
            )
        )
        if include_metadata:
            lf = lf.join(cls.scan_item_metadata(resolved), on="parent_asin", how="left")
        return lf

    @classmethod
    def scan_reviews(cls, names: Sequence[str] | str | None = None, include_metadata: bool = False) -> pl.LazyFrame:
        resolved_names = cls.resolve_names(names)
        frames = [
            cls.scan_domain(name, domain_id=domain_id, include_metadata=include_metadata)
            for domain_id, name in enumerate(resolved_names)
        ]
        if not frames:
            raise ValueError("names must be non-empty.")
        return pl.concat(frames, how="vertical")


Amazon2014Dataset = _Amazon2014DS()
</file>

<file path="datasets/base.py">
import os
from pathlib import Path


class DatasetBase:
    """Base class for adapters backed by an external source-data root.

    The root may contain original files or lossless, experiment-agnostic canonical
    forms such as typed Parquet. It is read-only during experiments. Schemas,
    filtered splits, caches, and other experiment-derived assets belong under the
    repository ``workspace`` instead.
    """

    SYSTEM_DATA_DIR = Path(
        os.path.expandvars(
            os.environ.get(
                "ROSEN_DATASOURCE_ROOT",
                os.environ.get("ROSEN_DATA_DIR", "/home/lx/datasource"),
            )
        )
    ).expanduser()

    def __init__(self):
        pass
</file>

<file path="datasets/douban.py">
from typing import List

from rosen.datasets.base import DatasetBase
from pathlib import Path
import polars as pl
"""
注意，这个数据集来自recbole-CDR
实际上USER文件完全相同
"""

def _drop_recbole_type(lf: pl.LazyFrame)->pl.LazyFrame:
    lf = lf.rename(lambda x: x.split(":")[0])
    return lf

class _DBDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "Douban"
    BOOK_DIR = BASE_DIR / "DoubanBook"
    MOVIE_DIR = BASE_DIR / "DoubanMovie"
    MUSIC_DIR = BASE_DIR / "DoubanMusic"

    BOOK_INTER = BOOK_DIR / "DoubanBook.inter"
    BOOK_USER = BOOK_DIR / "DoubanBook.user"
    MOVIE_INTER = MOVIE_DIR / "DoubanMovie.inter"
    MOVIE_USER = MOVIE_DIR / "DoubanMovie.user"
    MUSIC_INTER = MUSIC_DIR / "DoubanMusic.inter"
    MUSIC_USER = MUSIC_DIR / "DoubanMusic.user"

    INTERS = {
        "book": BOOK_INTER,
        "movie": MOVIE_INTER,
        "music": MUSIC_INTER
    }
    USERS = {
        "book": BOOK_USER,
        "movie": MOVIE_USER,
        "music": MUSIC_USER
    }
    USER = BOOK_USER

    @property
    def MERGED_INTERS_LF(self)->pl.LazyFrame:
        lfs = []
        for name, inter in self.INTERS.items():
            lf = pl.scan_csv(inter, separator="\t")
            lf = lf.with_columns(
                pl.lit(name).alias("domain")
            )
            lfs.append(lf)
        lf_combined = pl.concat(lfs, how="vertical")
        return _drop_recbole_type(lf_combined)

    @property
    def ALL_INTERS_LF(self)-> List[pl.LazyFrame]:
        lfs = []
        for name, inter in self.INTERS.items():
            lf = pl.scan_csv(inter, separator="\t")
            lf = lf.with_columns(
                pl.lit(name).alias("domain")
            )
            lf = _drop_recbole_type(lf)
            lfs.append(lf)
        return lfs

    @property
    def USER_LF(self)->pl.LazyFrame:
        return _drop_recbole_type(pl.scan_csv(self.USER, separator="\t"))

DoubanDataset = _DBDS()
if __name__ == '__main__':
    import polars as pl
    from pprint import pprint
    from rosen.datasets.overview import get_general_info, get_head_info, get_group_stats
    df = DoubanDataset.MERGED_INTERS_LF.collect()
    ret = get_general_info(df)
    pprint(ret)

    ret = DoubanDataset.USER_LF.collect()
    pprint(get_general_info(ret))

    pprint(get_group_stats(df, "domain"))
</file>

<file path="datasets/kuairand.py">
import polars as pl
from rosen.datasets.base import DatasetBase
import pprint

class KuaiRandDataset(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "KuaiRand"
    DATA_KUAI_1K = BASE_DIR / "data" / "KuaiRand-1K" / "data"
    STD_LOG_FORMER_DATA = DATA_KUAI_1K / "log_standard_4_08_to_4_21_1k.csv"
    STD_LOG_FORMER_DATA_P1 = STD_LOG_FORMER_DATA
    STD_LOG_FORMER_DATA_P2 = DATA_KUAI_1K / "log_standard_4_22_to_5_08_1k.csv"
    RAND_LOG_FORMER_DATA = DATA_KUAI_1K / "log_random_4_22_to_5_08_1k.csv"
    USER_FEATURES = DATA_KUAI_1K / "user_features_1k.csv"
    VIDEO_FEATURES = DATA_KUAI_1K / "video_features_basic_1k.csv"

# ==========================================
# 执行与测试示例
# ==========================================
if __name__ == "__main__":
    # 为了防止路径不存在报错，这里用 try-except 包裹文件读取逻辑
    # 请确保你的实际路径中存在这些 CSV 文件
    from rosen.datasets.overview import get_general_info, get_head_info, get_group_stats
    try:
        # 读取三个文件 (这里假设存在且格式标准)
        df_log = pl.read_csv(KuaiRandDataset.STD_LOG_FORMER_DATA_P2)
        df_user = pl.read_csv(KuaiRandDataset.USER_FEATURES)
        df_video = pl.read_csv(KuaiRandDataset.VIDEO_FEATURES)

        datasets = {
            "Log Standard": df_log,
            "User Features": df_user,
            "Video Features": df_video
        }

        # 1. 测试通用信息打印
        print("=== 一、通用信息统计 ===")
        for name, df in datasets.items():
            info = get_general_info(df)
            print(f"\n[{name}] 数据集统计:")
            # 使用 pprint 打印字典，结构清晰且不会省略数据
            pprint.pprint(info, depth=4, compact=False)

        # 2. 测试分组统计函数 (以 df_log 中的某个假设列为例，例如 'is_click')
        print("\n=== 二、分组统计 ===")
        # 这里假设 df_log 有一个 'is_click' 列，你可以替换为你实际想看的列名
        test_col = "tab"
        group_stats = get_group_stats(df_log, test_col)
        print(f"对 '{test_col}' 的分组统计结果:")
        pprint.pprint(group_stats)

        # 3. 测试 Head(3) 视图
        print("\n=== 三、Head(3) 完整数据及类型视图 ===")
        head_view = get_head_info(df_user)  # 以 User Features 为例
        print("[User Features] 前三行完整数据:")
        pprint.pprint(head_view)

    except FileNotFoundError as e:
        print(f"文件未找到，请检查路径: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
</file>

<file path="datasets/kuaisar.py">
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
</file>

<file path="datasets/movielens.py">
from functools import cached_property
from pathlib import Path

import pandas as pd
import polars as pl
from rosen.datasets.base import DatasetBase

class _MLDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR /"MovieLens"
    ML_1M_DIR = BASE_DIR / "ml-1m"
    USER_FEATURES = ML_1M_DIR / "users.dat"
    MOVIES_FEATURES = ML_1M_DIR / "movies.dat"
    RATINGS_DATA = ML_1M_DIR / "ratings.dat"

    @cached_property  # 改用 cached_property
    def ITEM_FEATURES_DF(self):
        return pd.read_csv(
            self.MOVIES_FEATURES,
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )

    @cached_property
    def USER_FEATURES_DF(self):
        return pd.read_csv(
            self.USER_FEATURES,
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zip-code']
        )

    @cached_property
    def INTERACTION_DF(self):
        return pd.read_csv(
            self.RATINGS_DATA,
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

MovieLensDataset = _MLDS()
# 使用方法：
if __name__ == "__main__":
    from rosen.datasets.overview import get_general_info, get_head_info, get_group_stats
    import pprint
    user_df = pl.from_pandas(MovieLensDataset.USER_FEATURES_DF)
    item_df = pl.from_pandas(MovieLensDataset.ITEM_FEATURES_DF)
    inter_df = pl.from_pandas(MovieLensDataset.INTERACTION_DF)
    info = get_general_info(user_df)
    pprint.pprint(info, depth=4, compact=False)
    info = get_general_info(item_df)
    pprint.pprint(info, depth=4, compact=False)
    info = get_general_info(inter_df)
    pprint.pprint(info, depth=4, compact=False)
</file>

<file path="datasets/overview.py">
# ==========================================
# 一、 通用信息打印函数
# ==========================================
import polars as pl


def get_general_info(df: pl.DataFrame) -> dict:
    """
    返回包含行数、列名、类型、不重复数量、空值数量以及数值类型五数概括的字典。
    """
    info = {
        "rows": df.height,
        "cols": {}
    }

    for col_name in df.columns:
        series = df[col_name]
        dtype = series.dtype

        # 获取基础统计信息
        n_unique = series.n_unique()
        n_nulls = series.null_count()  # 新增：计算空值数量

        col_info = {
            "type": str(dtype),
            "n_unique": n_unique,
            "null_count": n_nulls  # 新增：存入字典
        }

        # 判断是否为数值类型，如果是则计算五维数据 (Min, Q1, Median, Q3, Max)
        if dtype.is_numeric():
            # 剔除空值以进行准确计算
            s_clean = series.drop_nulls()
            if len(s_clean) > 0:
                col_info["5_number_summary"] = {
                    "min": s_clean.min(),
                    "q1": s_clean.quantile(0.25),
                    "median": s_clean.median(),
                    "q3": s_clean.quantile(0.75),
                    "max": s_clean.max()
                }
            else:
                col_info["5_number_summary"] = "All Nulls"

        info["cols"][col_name] = col_info

    return info


# ==========================================
# 二、 分组统计函数
# ==========================================
def get_group_stats(df: pl.DataFrame, col_name: str) -> dict:
    """
    传入列名，返回该列各组的名称及组内数量。如果列不存在则报错返回。
    """
    if col_name not in df.columns:
        return {"error": f"列名 '{col_name}' 不存在于数据集中。"}

    # 使用 value_counts() 获取分组及数量
    vc = df[col_name].value_counts()

    # vc 通常包含两列：[col_name, "count"]
    count_col_name = vc.columns[1]

    # 转换为字典 {组名: 数量}
    keys = vc[col_name].to_list()
    values = vc[count_col_name].to_list()

    return dict(zip(keys, values))


# ==========================================
# 三、 Head(3) 完整视图函数
# ==========================================
def get_head_info(df: pl.DataFrame) -> dict:
    """
    以字典方式返回前3行完整数据（防截断遮罩），并同时附带字段的Type信息。
    """
    head_df = df.head(3)
    # as_series=False 会将数据直接转换为 Python 原生的 list，不会被终端打印遮罩
    data_dict = head_df.to_dict(as_series=False)

    result = {}
    for col_name in head_df.columns:
        result[col_name] = {
            "type": str(head_df[col_name].dtype),
            "data": data_dict[col_name]
        }

    return result
</file>

<file path="datasets/taac2026.py">
from rosen.datasets.base import DatasetBase
import polars as pl
"""
### User int Features (46 columns)

- `user_int_feats_{1,3,4,48-59,82,86,92-109}`: Scalar `int64`, total 35 columns.
- `user_int_feats_{15, 60, 62-66, 80, 89-91}`: Array `list<int64>`, total 11 columns.


### User Dense Features (10 columns)

- `user_dense_feats_{61-66, 87, 89-91}`: Array `list<float>`, total 10 columns.


### Item int Features (14 columns)

- `item_int_feats_{5-10, 12-13, 16, 81, 83-85}`: Scalar `int64`, total 13 columns.
- `item_int_feats_{11}`: Array `list<int64>`, total 1 column.


### Domain Sequence Features (45 columns)

`list<int64>` sequences from 4 behavioral domains:

- `domain_a_seq_{38-46}`: 9 columns
- `domain_b_seq_{67-79, 88}`: 14 columns
- `domain_c_seq_{27-37, 47}`: 12 columns
- `domain_d_seq_{17-26}`: 10 columns
"""

"""
统计发现：
1. user_int_feats_{1,49,50,58,92-109} 在sample1000中数量小于10
2. user_int_feats_{55,59,82,93} 出现数量小于50
3. user_int_feats_51 最小值 40，最大值 150，但 n_unique 只有 6，且 Q1、Median、Q3 都是 56。说明绝大部分用户的值都是 56，只有极少数离群点。建议作为 Sparse 对待
4. null占比极高，大于80%的：user_int_feats_101, user_int_feats_100, item_int_feats_83

3. item_int_feats_{9,13,81,83} 出现次数小于等于25
4. label_type{1,2}
"""

class _TAACDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "TAAC2026"
    WHOLE = BASE_DIR / "demo_1000.parquet"

    # ==========================================
    # 1. 核心标识与目标 (Base & Target)
    # ==========================================
    # user_id 绝对不能进模型；label_type 需要减 1 变 0/1；time 用于算时间差
    col_user_id = "user_id"
    col_item_id = "item_id"
    col_label = "label_type"
    cols_time = ["label_time", "timestamp"]

    # ==========================================
    # 2. 物品侧静态特征 (Item Features)
    # ==========================================
    # 常规离散标量 (适合标准 Embedding)
    item_sparse_cols = [
        f"item_int_feats_{i}" for i in [5, 6, 7, 8, 9, 10, 12, 13, 16, 81, 83, 84, 85]
    ]
    # 多值离散特征 (大概率是Tags，适合 Mean/Sum Pooling)
    item_varlen_sparse_cols = ["item_int_feats_11"]

    # ==========================================
    # 3. 用户侧上下文与画像 (User Features)
    # ==========================================
    # 3.1 单值离散特征 (适合标准 Embedding)
    user_sparse_cols = [
        f"user_int_feats_{i}" for i in [1, 3, 4, *range(48, 60), 82, 86, *range(92, 110)]
    ]

    # 3.2 多值离散特征 (近期Tag/行为集合，适合 Mean Pooling)
    user_varlen_sparse_cols = [
        f"user_int_feats_{i}" for i in [15, 60, 62, 63, 64, 65, 66, 80, 89, 90, 91]
    ]

    # 3.3 预训练稠密向量 (高阶画像，适合直接过 MLP 或 Dense 层降维)
    # 注意：这里修正了你的笔误，它们在日志中是 user_dense_feats
    user_dense_emb_cols = [
        f"user_dense_feats_{i}" for i in [61, 87, 89, 90, 91]
    ]

    # 3.4 历史累积统计值 (绝对数值极大，送入模型前必须做 Log(x+1) 平滑处理)
    user_dense_stat_cols = [
        f"user_dense_feats_{i}" for i in [62, 63, 64, 65, 66]
    ]

    # ==========================================
    # 4. 终身行为序列跨域特征 (Cross-Domain Sequences)
    # ==========================================
    # 使用结构化字典管理，便于给 DIN / SIM 等 Attention 模块传参
    seq_domains = {
        "A": {
            "time": "domain_a_seq_39",
            "id": "domain_a_seq_38",
            "attrs": [f"domain_a_seq_{i}" for i in range(40, 47)]
        },
        "B": {
            "time": "domain_b_seq_67",
            "id": "domain_b_seq_69",
            "attrs": [f"domain_b_seq_{i}" for i in [68, *range(70, 80), 88]]
        },
        "C": {
            "time": "domain_c_seq_27",
            "id": "domain_c_seq_47",
            "attrs": [f"domain_c_seq_{i}" for i in range(28, 38)]
        },
        "D": {
            # D域无明显独立时间戳（或隐含对齐），长达3800，必须做 SIM 截断检索
            "time": "domain_d_seq_26",
            "id": "domain_d_seq_23",
            "attrs": [f"domain_d_seq_{i}" for i in range(17, 27) if i not in (23, 26)]
        }
    }
    high_null_cols = ["user_int_feats_101", "user_int_feats_100", "item_int_feats_83"]
    low_cardinality_cols = [
        f"user_int_feats_{i}" for i in [1, 49, 50, 58, 55, 59, 82, 93, *range(92, 110)]
    ]

    @property
    def WHOLE_LF(self):
        whole_lf = pl.scan_parquet(self.WHOLE)
        whole_lf = whole_lf.with_columns(pl.col("label_type") - 1)
        return whole_lf

TAAC2026Dataset = _TAACDS()

if __name__ == '__main__':
    import polars as pl
    from rosen.datasets.overview import get_general_info
    lf = pl.scan_parquet(TAAC2026Dataset.WHOLE).collect()
    info = get_general_info(lf)
    from pprint import pprint
    pprint(info)
</file>

<file path="emb/schema/__init__.py">
from .base import EmbSetting, EmbType, ProfileSeqGroupConfig, SeqGroupConfig
from .categorical import BaseCategoricalSetting, MultiSparseSetting, QuantileEmbSetting, SparseEmbSetting
from .numerical import BaseNumericalSetting, MinMaxDenseSetting, VectorDenseSetting
from .sequence import SequenceSetting
from .utils import NULL_FALLBACK, clear_seq_expr, explode_expr, seq_length_expr

__all__ = [
    "NULL_FALLBACK",
    "EmbSetting",
    "EmbType",
    "SeqGroupConfig",
    "ProfileSeqGroupConfig",
    "BaseCategoricalSetting",
    "BaseNumericalSetting",
    "SparseEmbSetting",
    "MultiSparseSetting",
    "SequenceSetting",
    "QuantileEmbSetting",
    "MinMaxDenseSetting",
    "VectorDenseSetting",
    "clear_seq_expr",
    "explode_expr",
    "seq_length_expr",
]
</file>

<file path="emb/schema/base.py">
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import polars as pl
import torch
from torch import nn

from rosen.core.enum_type import FeatureSource

if TYPE_CHECKING:
    from rosen.data.padding import ColumnFormatter


class EmbType(Enum):
    UNKNOWN = "none"
    SPARSE = "sparse"
    MULTI_SPARSE = "multi_sparse"
    QUANTILE = "quantile"
    SPARSE_SEQ = "sparse_seq"
    SPARSE_SET = "sparse_set"
    DENSE = "dense"
    VECTOR_DENSE = "vector_dense"
    DENSE_SEQ = "dense_seq"
    SEQUENCE = "sequence"


@dataclass
class SeqGroupConfig:
    group_name: str
    seq_len_field_name: str
    max_len: int
    padding_side: Literal["left", "right"] = "right"
    time_field_name: Optional[str] = None


@dataclass
class ProfileSeqGroupConfig(SeqGroupConfig):
    owner_field_name: str = "user_id"
    id_field_name: Optional[str] = None
    anchor_field_name: Optional[str] = None
    profile_name: Optional[str] = None
    history_time_filter: Literal[
        "cursor_order",
        "strict_time",
        "drop_same_ts",
        "same_ts_only",
    ] = "cursor_order"
    backend: Literal["profile"] = "profile"


class EmbSetting(ABC):
    emb_type = EmbType.UNKNOWN

    def __init__(
        self,
        field_name: str,
        embedding_dim: int,
        source: FeatureSource = FeatureSource.UNKNOWN,
        padding_zero: bool = True,
        use_oov: bool = True,
    ):
        self.field_name = field_name
        self.embedding_dim = embedding_dim
        self.source = source
        self.is_fitted = False

        self.padding_zero = padding_zero
        self.use_oov = use_oov
        self.vocab: Dict[str, int] = {}
        self.oov_idx: int = -1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return self.vocab_size + int(self.padding_zero) + int(self.use_oov)

    @property
    def serialized_type_name(self) -> str:
        return self.emb_type.name

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name}

    @property
    def requires_embedding_module(self) -> bool:
        return self.num_embeddings > 0

    @property
    def embedding_field_name(self) -> str:
        return self.field_name

    @property
    def is_sequence_setting(self) -> bool:
        return False

    def get_output_field_names(self) -> List[str]:
        return [self.field_name]

    def get_raw_field_names(self) -> List[str]:
        return [self.field_name]

    @abstractmethod
    def get_fit_exprs(self) -> List[pl.Expr]:
        pass

    @abstractmethod
    def parse_fit_result(self, result_df: pl.DataFrame):
        pass

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        return self.parse_fit_result(result_df)

    @abstractmethod
    def get_transform_expr(self):
        pass

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element fitting.")

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element transform.")

    @abstractmethod
    def get_formatters(self) -> Dict[str, "ColumnFormatter"]:
        pass

    def get_sequence_formatters(
        self,
        max_len: int,
        padding_side: Literal["left", "right"],
    ) -> Dict[str, "ColumnFormatter"]:
        from rosen.data.padding import (
            DenseFormatter,
            IntFormatter,
            PaddedFloatSequenceFormatter,
            PaddedIntSequenceFormatter,
        )

        base_formatters = self.get_formatters()
        fmt = base_formatters[self.field_name]
        if isinstance(fmt, DenseFormatter):
            return {self.field_name: PaddedFloatSequenceFormatter(max_len=max_len, padding_side=padding_side)}
        if isinstance(fmt, IntFormatter):
            return {self.field_name: PaddedIntSequenceFormatter(max_len=max_len, padding_side=padding_side)}
        raise NotImplementedError(
            f"Formatter {fmt.__class__.__name__} cannot be promoted to a sequence automatically."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.serialized_type_name,
            "field_name": self.field_name,
            "embedding_size": self.embedding_dim,
            "num_embeddings": self.num_embeddings,
            "vocab_size": self.vocab_size,
            "feature_source": self.source.name,
            "is_fitted": self.is_fitted,
            "padding_zero": self.padding_zero,
            "use_oov": self.use_oov,
            "vocab": self.vocab,
            "oov_idx": self.oov_idx,
        }

    def load_state(self, state_dict: Dict[str, Any]):
        self.is_fitted = state_dict.get("is_fitted", True)
        self.vocab = state_dict.get("vocab", {})
        self.use_oov = state_dict.get("use_oov", self.use_oov)
        self.padding_zero = state_dict.get("padding_zero", self.padding_zero)
        self.oov_idx = state_dict.get("oov_idx", self.oov_idx)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbSetting":
        pass

    @abstractmethod
    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        pass
</file>

<file path="emb/schema/categorical.py">
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import torch
from torch import nn

from rosen.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType
from .utils import NULL_FALLBACK, map_list_to_indices, mean_pooling


class BaseCategoricalSetting(EmbSetting):
    """离散特征基类：统一管理词表与 OOV 逻辑。"""
    NULL_TOKEN = "__NULL__"

    def __init__(
        self,
        field_name: str,
        embedding_dim: int,
        source: FeatureSource,
        padding_zero: bool = True,
        use_null: bool = True,
        use_oov: bool = True,
        min_freq: int = 1,
    ):
        super().__init__(field_name, embedding_dim, source, padding_zero, use_oov)
        if use_null and not padding_zero:
            raise ValueError("use_null=True requires padding_zero=True for sparse categorical features.")
        self.min_freq = min_freq
        self.use_null = use_null
        self.vocab = {}
        self.null_idx = -1
        self.oov_idx = -1

    def _build_vocab_indices(self, valid_vals: List[str]):
        start_idx = 1 if self.padding_zero else 0
        self.vocab = {str(val): idx + start_idx for idx, val in enumerate(valid_vals)}
        next_idx = len(self.vocab) + start_idx
        if self.use_null:
            self.null_idx = next_idx
            next_idx += 1
        else:
            self.null_idx = 0 if self.padding_zero else -1
        if self.use_oov:
            self.oov_idx = next_idx
        else:
            self.oov_idx = 0 if self.padding_zero else -1
        self.is_fitted = True

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return (
            self.vocab_size
            + int(self.padding_zero)
            + int(self.use_null)
            + int(self.use_oov)
        )


class SparseEmbSetting(BaseCategoricalSetting):
    emb_type = EmbType.SPARSE

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        embedding_dim: int = 16,
        padding_zero: bool = True,
        use_null: bool = True,
        fill_null: str = "null",
        min_freq: int = 1,
        use_oov: bool = True,
    ):
        super().__init__(
            field_name,
            embedding_dim,
            source,
            padding_zero,
            use_null=use_null,
            use_oov=use_oov,
            min_freq=min_freq,
        )
        if fill_null not in {"null", "oov", "zero", None}:
            raise ValueError("fill_null must be one of {'null', 'oov', 'zero', None}")
        self.fill_null = fill_null

    def get_fit_exprs(self) -> List[pl.Expr]:
        return [
            pl.col(self.field_name)
            .cast(pl.Utf8)
            .drop_nulls()
            .value_counts()
            .implode()
            .alias(self.field_name)
        ]

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        exploded_raw = seq_expr.explode().cast(pl.Utf8)
        exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
        return [exploded.value_counts().implode().alias(alias)]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        rows = result_df.get_column(alias).to_list()[0]
        valid_vals = sorted(
            [
                row[alias]
                for row in rows
                if row[alias] is not None and row.get("count", row.get("counts", 0)) >= self.min_freq
            ]
        )
        self._build_vocab_indices(valid_vals)

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        replace_kwargs = {"default": pl.lit(self.oov_idx, dtype=pl.UInt32)} if self.oov_idx >= 0 else {}
        mapped_expr = expr.cast(pl.Utf8).replace_strict(self.vocab, **replace_kwargs)
        if not self.use_null:
            return mapped_expr.cast(pl.UInt32)

        if self.fill_null == "zero":
            null_expr = pl.lit(0, dtype=pl.UInt32)
        elif self.fill_null == "oov":
            null_expr = pl.lit(self.oov_idx, dtype=pl.UInt32)
        else:
            null_expr = pl.lit(self.null_idx, dtype=pl.UInt32)
        return pl.when(expr.is_null()).then(null_expr).otherwise(mapped_expr).cast(pl.UInt32)

    def get_transform_expr(self) -> pl.Expr:
        return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)

    def get_formatters(self):
        from rosen.data.padding import IntFormatter

        return {self.field_name: IntFormatter()}

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_freq"] = self.min_freq
        data["use_null"] = self.use_null
        data["null_idx"] = self.null_idx
        data["fill_null"] = self.fill_null
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_freq = state_dict.get("min_freq", 1)
        self.use_null = state_dict.get("use_null", self.use_null)
        self.null_idx = state_dict.get("null_idx", self.null_idx)
        self.fill_null = state_dict.get("fill_null", self.fill_null)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SparseEmbSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data["embedding_size"],
            padding_zero=data.get("padding_zero", True),
            use_null=data.get("use_null", True),
            fill_null=data.get("fill_null", "null"),
            min_freq=data.get("min_freq", 1),
            use_oov=data.get("use_oov", True),
        )
        obj.load_state(data)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return emb_modules[self.embedding_field_name](interaction[self.field_name])


class QuantileEmbSetting(EmbSetting):
    emb_type = EmbType.QUANTILE

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        bucket_count: int = 10,
        embedding_dim: int = 16,
        boundaries: Optional[List[float]] = None,
    ):
        super().__init__(field_name, embedding_dim, source, padding_zero=True, use_oov=False)
        self.bucket_count = bucket_count
        self.boundaries = boundaries if boundaries is not None else []
        if self.boundaries:
            self.is_fitted = True

    @property
    def vocab_size(self) -> int:
        return len(self.boundaries) + 1 if self.boundaries else 0

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return self.vocab_size + 1

    def get_fit_exprs(self) -> List[pl.Expr]:
        q_list = np.linspace(0, 1, self.bucket_count + 1)[1:-1]
        return [
            pl.col(self.field_name).drop_nulls().cast(pl.Float64).quantile(q).alias(f"{self.field_name}_q_{idx}")
            for idx, q in enumerate(q_list)
        ]

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        q_list = np.linspace(0, 1, self.bucket_count + 1)[1:-1]
        exploded = seq_expr.explode().drop_nulls().cast(pl.Float64)
        return [
            exploded.quantile(q).alias(f"{alias}_q_{idx}")
            for idx, q in enumerate(q_list)
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        boundaries = []
        for idx in range(self.bucket_count - 1):
            value = result_df.get_column(f"{alias}_q_{idx}")[0]
            if value is not None and not np.isnan(value):
                boundaries.append(value)
        self.boundaries = sorted(set(boundaries))
        self.is_fitted = True

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        if not self.is_fitted or not self.boundaries:
            return pl.lit(0, dtype=pl.UInt32)

        bucket_count = len(self.boundaries) + 1
        labels = [str(idx) for idx in range(1, bucket_count + 1)]
        return (
            expr
            .cut(breaks=self.boundaries, labels=labels)
            .cast(pl.String)
            .cast(pl.UInt32)
            .fill_null(0)
        )

    def get_transform_expr(self) -> pl.Expr:
        return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)

    def get_formatters(self):
        from rosen.data.padding import IntFormatter

        return {self.field_name: IntFormatter()}

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["bucket_count"] = self.bucket_count
        data["boundaries"] = self.boundaries
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.bucket_count = state_dict.get("bucket_count", self.bucket_count)
        self.boundaries = state_dict.get("boundaries", [])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantileEmbSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            bucket_count=data.get("bucket_count", 10),
            embedding_dim=data["embedding_size"],
            boundaries=data.get("boundaries", []),
        )
        obj.is_fitted = data.get("is_fitted", False)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return emb_modules[self.embedding_field_name](interaction[self.field_name])


class MultiSparseSetting(BaseCategoricalSetting):
    emb_type = EmbType.MULTI_SPARSE

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        embedding_dim: int = 16,
        max_tag_len: int = 5,
        is_string_format: bool = False,
        separator: str = ",",
        padding_zero: bool = True,
        min_freq: int = 1,
        use_oov: bool = True,
        agg: str = "sum",
    ):
        super().__init__(
            field_name,
            embedding_dim,
            source,
            padding_zero,
            use_null=False,
            use_oov=use_oov,
            min_freq=min_freq,
        )
        self.max_tag_len = max_tag_len
        self.is_string_format = is_string_format
        self.separator = separator
        self.agg = agg

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name, EmbType.SPARSE_SET.name}

    @property
    def fill_empty_sequence_with_fallback(self) -> bool:
        return False

    def _clean_list_expr(self, expr: pl.Expr, fill_empty_with_fallback: bool = False) -> pl.Expr:
        if self.is_string_format:
            cleaned = (
                expr.cast(pl.Utf8)
                .fill_null("")
                .str.split(self.separator)
                .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
            )
        else:
            cleaned = (
                expr.cast(pl.List(pl.Utf8))
                .fill_null([])
                .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
            )

        if not fill_empty_with_fallback:
            return cleaned

        return (
            pl.when(cleaned.list.len() > 0)
            .then(cleaned)
            .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
        )

    def _truncate_tag_expr(self, expr: pl.Expr) -> pl.Expr:
        return expr.list.head(self.max_tag_len)

    def get_fit_exprs(self) -> List[pl.Expr]:
        exploded_raw = self._truncate_tag_expr(self._clean_list_expr(pl.col(self.field_name))).explode().cast(pl.Utf8)
        exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
        return [exploded.value_counts().implode().alias(self.field_name)]

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        nested_expr = seq_expr.list.eval(
            self._truncate_tag_expr(self._clean_list_expr(pl.element(), fill_empty_with_fallback=False))
        )
        exploded_raw = nested_expr.explode().explode().cast(pl.Utf8)
        exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
        return [exploded.value_counts().implode().alias(alias)]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        rows = result_df.get_column(alias).to_list()[0]
        valid_vals = sorted(
            [
                row[alias]
                for row in rows
                if row[alias] is not None and row.get("count", row.get("counts", 0)) >= self.min_freq
            ]
        )
        self._build_vocab_indices(valid_vals)

    def get_transform_expr(self) -> pl.Expr:
        expr = self._truncate_tag_expr(
            self._clean_list_expr(pl.col(self.field_name), fill_empty_with_fallback=False)
        )
        return map_list_to_indices(expr, self.vocab, self.oov_idx).alias(self.field_name)

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        cleaned = self._truncate_tag_expr(
            self._clean_list_expr(expr, fill_empty_with_fallback=False)
        )
        return map_list_to_indices(cleaned, self.vocab, self.oov_idx)

    def get_formatters(self):
        from rosen.data.padding import PaddedIntSequenceFormatter

        return {self.field_name: PaddedIntSequenceFormatter(max_len=self.max_tag_len)}

    def get_sequence_formatters(self, max_len: int, padding_side: str):
        from rosen.data.padding import PaddedNestedSequenceFormatter

        return {
            self.field_name: PaddedNestedSequenceFormatter(
                max_seq_len=max_len,
                max_tag_len=self.max_tag_len,
                padding_side=padding_side,
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_freq"] = self.min_freq
        data["max_tag_len"] = self.max_tag_len
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        data["agg"] = self.agg
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_freq = state_dict.get("min_freq", 1)
        self.max_tag_len = state_dict.get("max_tag_len", state_dict.get("max_len", self.max_tag_len))
        self.is_string_format = state_dict.get("is_string_format", False)
        self.separator = state_dict.get("separator", ",")
        self.agg = state_dict.get("agg", "sum")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiSparseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data["embedding_size"],
            max_tag_len=data.get("max_tag_len", data.get("max_len", 5)),
            is_string_format=data.get("is_string_format", False),
            separator=data.get("separator", ","),
            padding_zero=data.get("padding_zero", True),
            min_freq=data.get("min_freq", 1),
            use_oov=data.get("use_oov", True),
            agg=data.get("agg", "sum"),
        )
        obj.load_state(data)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        idx_tensor = interaction[self.field_name]
        emb = emb_modules[self.embedding_field_name](idx_tensor)
        emb = torch.sum(emb, dim=-2)
        if self.agg == "mean":
            emb = mean_pooling(emb, idx_tensor, self.padding_zero)
        return emb


if __name__ == "__main__":
    source = FeatureSource.USER

    sparse_null = SparseEmbSetting(
        "demo",
        source=source,
        padding_zero=True,
        use_null=True,
        fill_null="null",
        use_oov=True,
    )
    sparse_null._build_vocab_indices(["a", "b"])
    expr = sparse_null.get_transform_expr()
    df = pl.DataFrame({"demo": [None, "a", "zzz"]}).lazy().with_columns(expr).collect()
    assert df["demo"].to_list() == [sparse_null.null_idx, 1, sparse_null.oov_idx]
    assert sparse_null.num_embeddings == 5

    sparse_zero = SparseEmbSetting(
        "demo",
        source=source,
        padding_zero=True,
        use_null=True,
        fill_null="zero",
        use_oov=True,
    )
    sparse_zero._build_vocab_indices(["a", "b"])
    expr = sparse_zero.get_transform_expr()
    df = pl.DataFrame({"demo": [None, "a", "zzz"]}).lazy().with_columns(expr).collect()
    assert df["demo"].to_list() == [0, 1, sparse_zero.oov_idx]
    assert sparse_zero.num_embeddings == 5

    sparse_oov = SparseEmbSetting(
        "demo",
        source=source,
        padding_zero=True,
        use_null=True,
        fill_null="oov",
        use_oov=True,
    )
    sparse_oov._build_vocab_indices(["a", "b"])
    expr = sparse_oov.get_transform_expr()
    df = pl.DataFrame({"demo": [None, "a", "zzz"]}).lazy().with_columns(expr).collect()
    assert df["demo"].to_list() == [sparse_oov.oov_idx, 1, sparse_oov.oov_idx]
    assert sparse_oov.num_embeddings == 5

    sparse_strict = SparseEmbSetting(
        "demo",
        source=source,
        padding_zero=False,
        use_null=False,
        use_oov=False,
    )
    sparse_strict._build_vocab_indices(["1", "2"])
    expr = sparse_strict.get_transform_expr()
    df = pl.DataFrame({"demo": ["1", "2"]}).lazy().with_columns(expr).collect()
    assert df["demo"].to_list() == [0, 1]

    multi = MultiSparseSetting(
        "tags",
        source=source,
        padding_zero=True,
        use_oov=True,
        is_string_format=True,
        separator=",",
    )
    multi._build_vocab_indices(["a", "b"])
    expr = multi.get_transform_expr()
    df = pl.DataFrame({"tags": [None, "", "a,b", "zzz"]}).lazy().with_columns(expr).collect()
    assert df["tags"].to_list() == [[], [], [1, 2], [multi.oov_idx]]

    print("categorical.py self-check passed")
</file>

<file path="emb/schema/numerical.py">
from typing import Any, Dict, List, Optional

import polars as pl
import torch
from torch import nn

from rosen.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType


class BaseNumericalSetting(EmbSetting):
    """连续数值特征基类：显式声明无词表。"""

    def __init__(self, field_name: str, source: FeatureSource, embedding_dim: int = 1):
        super().__init__(
            field_name=field_name,
            embedding_dim=embedding_dim,
            source=source,
            padding_zero=False,
            use_oov=False,
        )
        self.is_fitted = False

    @property
    def vocab_size(self) -> int:
        return 0

    @property
    def num_embeddings(self) -> int:
        return -1

    @property
    def requires_embedding_module(self) -> bool:
        return False

    def get_formatters(self):
        from rosen.data.padding import DenseFormatter

        return {self.field_name: DenseFormatter()}


class MinMaxDenseSetting(BaseNumericalSetting):
    emb_type = EmbType.DENSE

    def __init__(self, field_name: str, source: FeatureSource, min_val: float = None, max_val: float = None):
        super().__init__(field_name=field_name, source=source, embedding_dim=1)
        self.min_val = min_val
        self.max_val = max_val
        self.is_fitted = min_val is not None and max_val is not None

    def get_fit_exprs(self) -> List[pl.Expr]:
        return [
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().min().alias(f"{self.field_name}_min"),
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().max().alias(f"{self.field_name}_max"),
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        self.min_val = result_df.get_column(f"{alias}_min").to_list()[0]
        self.max_val = result_df.get_column(f"{alias}_max").to_list()[0]
        if self.min_val is None or self.max_val is None:
            self.min_val = 0.0
            self.max_val = 1.0
        elif self.min_val == self.max_val:
            self.max_val = self.min_val + 1e-6
        self.is_fitted = True

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        exploded = seq_expr.explode().cast(pl.Float64)
        return [
            exploded.min().alias(f"{alias}_min"),
            exploded.max().alias(f"{alias}_max"),
        ]

    def get_element_transform_expr(self, expr: pl.Expr = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        range_val = self.max_val - self.min_val
        return ((expr.cast(pl.Float32, strict=False) - self.min_val) / range_val).fill_null(0.0).cast(pl.Float32)

    def get_transform_expr(self) -> pl.Expr:
        return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_val"] = self.min_val
        data["max_val"] = self.max_val
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_val = state_dict.get("min_val")
        self.max_val = state_dict.get("max_val")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinMaxDenseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            min_val=data.get("min_val"),
            max_val=data.get("max_val"),
        )
        obj.is_fitted = data.get("is_fitted", False)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return interaction[self.field_name].unsqueeze(-1)


class VectorDenseSetting(BaseNumericalSetting):
    emb_type = EmbType.VECTOR_DENSE

    def __init__(self, field_name: str, source: 'FeatureSource', embedding_dim: Optional[int] = None,
                 zero_fill: bool = True):
        super().__init__(field_name=field_name, source=source, embedding_dim=embedding_dim or 0)
        self.zero_fill = zero_fill
        self.is_fitted = embedding_dim is not None

    def get_fit_exprs(self) -> List[pl.Expr]:
        if self.is_fitted:
            return []

        len_expr = pl.col(self.field_name).drop_nulls().list.len()
        return [
            len_expr.value_counts().implode().alias(f"{self.field_name}_len_counts"),
            pl.col(self.field_name).is_null().sum().alias(f"{self.field_name}_null_count"),
            len_expr.eq(0).sum().alias(f"{self.field_name}_empty_count"),
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        if self.is_fitted:
            return

        len_counts_key = f"{self.field_name}_len_counts"
        len_rows = result_df.get_column(len_counts_key).to_list()[0]
        null_count = int(result_df.get_column(f"{self.field_name}_null_count")[0] or 0)
        empty_count = int(result_df.get_column(f"{self.field_name}_empty_count")[0] or 0)

        length_distribution: Dict[int, int] = {}
        for row in len_rows or []:
            count = row.get("count", row.get("counts", 0))
            length = None
            for key, value in row.items():
                if key not in ("count", "counts"):
                    length = value
                    break
            if length is None:
                continue
            length_distribution[int(length)] = int(count)

        non_empty_distribution = {
            int(length): int(count)
            for length, count in length_distribution.items()
            if int(length) > 0 and int(count) > 0
        }

        if not non_empty_distribution:
            raise ValueError(
                f"Feature '{self.field_name}' has no non-empty vectors in training data. "
                f"null_rows={null_count}, empty_rows={empty_count}. "
                "Please set `embedding_dim` explicitly if this feature should be kept."
            )

        if len(non_empty_distribution) != 1:
            dist_lines = [
                f"  {length} -> {count} rows"
                for length, count in sorted(non_empty_distribution.items())
            ]
            distribution_text = "\n".join(dist_lines)
            raise ValueError(
                f"Feature '{self.field_name}' has inconsistent vector lengths on training data.\n"
                f"Observed non-empty length distribution:\n{distribution_text}\n"
                f"null_rows={null_count}, empty_rows={empty_count}\n"
                "Please set `embedding_dim` explicitly for this feature."
            )

        self.embedding_dim = next(iter(non_empty_distribution))
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        if not self.is_fitted or not self.embedding_dim:
            raise RuntimeError(f"Feature '{self.field_name}' is not fitted yet. embedding_dim is unknown.")

        # Keep the raw float list column. Fixed-width padding/truncation is
        # handled in the dataset formatter so checkpoints preserve the original
        # vector information instead of silently zeroing mismatched rows.
        return pl.col(self.field_name).cast(pl.List(pl.Float32)).alias(self.field_name)

    def get_formatters(self):
        from rosen.data.padding import VectorDenseFormatter

        return {
            self.field_name: VectorDenseFormatter(
                dim=int(self.embedding_dim),
                zero_fill=self.zero_fill,
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["embedding_dim"] = self.embedding_dim
        data["zero_fill"] = self.zero_fill
        return data

    @property
    def compatible_type_names(self) -> set[str]:
        # Backward compatibility: older checkpoints stored vector dense as DENSE.
        return {self.serialized_type_name, EmbType.DENSE.name}

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.embedding_dim = state_dict.get("embedding_dim", self.embedding_dim)
        self.zero_fill = state_dict.get("zero_fill", self.zero_fill)
        self.is_fitted = self.embedding_dim is not None and self.embedding_dim > 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDenseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data.get("embedding_dim", data.get("embedding_size")),
            zero_fill=data.get("zero_fill", True),
        )
        obj.is_fitted = data.get("is_fitted", obj.embedding_dim is not None)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return interaction[self.field_name].to(torch.float32)
</file>

<file path="emb/schema/sequence.py">
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import polars as pl
import torch
from torch import nn

from rosen.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType, SeqGroupConfig
from .utils import NULL_FALLBACK


def _dedupe_names(names: List[Optional[str]]) -> List[str]:
    return list(dict.fromkeys(name for name in names if name))


def _clean_outer_sequence_expr(
    expr: pl.Expr,
    is_string_format: bool,
    separator: str,
    fill_empty_with_fallback: bool = False,
) -> pl.Expr:
    if is_string_format:
        cleaned = (
            expr.cast(pl.Utf8)
            .fill_null("")
            .str.split(separator)
            .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
        )
    elif fill_empty_with_fallback:
        cleaned = (
            expr.cast(pl.List(pl.Utf8))
            .fill_null([])
            .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
        )
    else:
        cleaned = expr.fill_null([]).list.eval(pl.element().filter(pl.element().is_not_null()))

    if not fill_empty_with_fallback:
        return cleaned

    return (
        pl.when(cleaned.list.len() > 0)
        .then(cleaned)
        .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
    )


def _clean_time_sequence_expr(expr: pl.Expr, is_string_format: bool, separator: str) -> pl.Expr:
    if is_string_format:
        return (
            expr.cast(pl.Utf8)
            .fill_null("")
            .str.split(separator)
            .list.eval(
                pl.when(pl.element().str.strip_chars() != "")
                .then(pl.element().str.strip_chars().cast(pl.Int64, strict=False))
                .otherwise(None)
            )
            .list.drop_nulls()
        )
    return (
        expr.cast(pl.List(pl.Int64))
        .fill_null([])
        .list.eval(pl.element().cast(pl.Int64, strict=False).filter(pl.element().is_not_null()))
    )


class SequenceSetting(EmbSetting):
    emb_type = EmbType.SEQUENCE

    def __init__(
        self,
        field_name: str,
        element_setting: EmbSetting,
        group: SeqGroupConfig,
        truncate_mode: str = "tail",
        is_string_format: bool = False,
        separator: str = ",",
    ):
        super().__init__(
            field_name=field_name,
            embedding_dim=element_setting.embedding_dim,
            source=FeatureSource.SEQ,
            padding_zero=element_setting.padding_zero,
            use_oov=element_setting.use_oov,
        )
        self.element_setting = element_setting
        self.group = group
        self.max_len = group.max_len
        self.group_name = group.group_name
        self.seq_len_field_name = group.seq_len_field_name
        self.padding_side = group.padding_side
        self.time_field_name = group.time_field_name
        self.truncate_mode = truncate_mode
        self.is_string_format = is_string_format
        self.separator = separator
        self.is_fitted = element_setting.is_fitted

    @property
    def compatible_type_names(self) -> set[str]:
        legacy = {self.serialized_type_name, EmbType.SPARSE_SEQ.name, EmbType.DENSE_SEQ.name, "SHARE_SEQ"}
        return legacy

    @property
    def is_sequence_setting(self) -> bool:
        return True

    @property
    def embedding_field_name(self) -> str:
        return self.element_setting.embedding_field_name

    @property
    def vocab_size(self) -> int:
        return self.element_setting.vocab_size

    @property
    def num_embeddings(self) -> int:
        return self.element_setting.num_embeddings

    @property
    def requires_embedding_module(self) -> bool:
        return self.element_setting.requires_embedding_module

    def get_output_field_names(self) -> List[str]:
        return _dedupe_names([self.field_name, self.seq_len_field_name, self.time_field_name])

    def get_raw_field_names(self) -> List[str]:
        return _dedupe_names([self.field_name, self.time_field_name])

    def get_clean_expr(self, fill_empty_with_fallback: bool = False, field_name: Optional[str] = None) -> pl.Expr:
        seq_field = field_name or self.field_name
        return _clean_outer_sequence_expr(
            pl.col(seq_field),
            is_string_format=self.is_string_format,
            separator=self.separator,
            fill_empty_with_fallback=fill_empty_with_fallback,
        )

    def truncate_seq_expr(self, expr: pl.Expr) -> pl.Expr:
        if self.truncate_mode == "head":
            return expr.list.head(self.max_len)
        if self.truncate_mode == "tail":
            return expr.list.tail(self.max_len)
        raise ValueError(f"Unknown truncate_mode: {self.truncate_mode}")

    def get_truncated_len_expr(self, raw_len_expr: pl.Expr) -> pl.Expr:
        return (
            pl.min_horizontal(
                raw_len_expr.cast(pl.UInt32),
                pl.lit(self.max_len, dtype=pl.UInt32),
            )
            .alias(self.seq_len_field_name)
        )

    def get_seq_len_expr(self, field_name: Optional[str] = None) -> pl.Expr:
        raw_len = self.get_clean_expr(fill_empty_with_fallback=False, field_name=field_name).list.len()
        return self.get_truncated_len_expr(raw_len)

    def get_fit_seq_expr(self) -> pl.Expr:
        return self.truncate_seq_expr(
            self.get_clean_expr(fill_empty_with_fallback=False)
        )

    def get_fit_exprs(self) -> List[pl.Expr]:
        if self.element_setting.is_fitted:
            self.is_fitted = True
            return []
        return self.element_setting.get_element_fit_exprs(self.get_fit_seq_expr(), alias=self.field_name)

    def parse_fit_result(self, result_df: pl.DataFrame):
        if self.element_setting.is_fitted:
            self.is_fitted = True
            return
        self.element_setting.parse_element_fit_result(result_df, alias=self.field_name)
        self.is_fitted = self.element_setting.is_fitted

    def get_time_transform_expr(self) -> Optional[pl.Expr]:
        if not self.time_field_name:
            return None
        expr = _clean_time_sequence_expr(
            pl.col(self.time_field_name),
            is_string_format=self.is_string_format,
            separator=self.separator,
        )
        return self.truncate_seq_expr(expr).alias(self.time_field_name)

    def get_transform_expr(self) -> List[pl.Expr]:
        expr = self.truncate_seq_expr(
            self.get_clean_expr(fill_empty_with_fallback=False)
        )
        mapped_expr = expr.list.eval(self.element_setting.get_element_transform_expr()).alias(self.field_name)
        exprs: List[pl.Expr] = [mapped_expr, self.get_seq_len_expr()]
        time_expr = self.get_time_transform_expr()
        if time_expr is not None:
            exprs.append(time_expr)
        return exprs

    def get_formatters(self) -> Dict[str, "ColumnFormatter"]:
        from rosen.data.padding import IntFormatter, PaddedIntSequenceFormatter

        element_formatter = next(
            iter(
                self.element_setting.get_sequence_formatters(
                    max_len=self.max_len,
                    padding_side=self.padding_side,
                ).values()
            )
        )
        formatters = {
            self.field_name: element_formatter,
            self.seq_len_field_name: IntFormatter(),
        }
        if self.time_field_name:
            formatters[self.time_field_name] = PaddedIntSequenceFormatter(
                max_len=self.max_len,
                padding_side=self.padding_side,
            )
        return formatters

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["group_name"] = self.group_name
        data["seq_len_field_name"] = self.seq_len_field_name
        data["max_len"] = self.max_len
        data["padding_side"] = self.padding_side
        data["time_field_name"] = self.time_field_name
        if is_dataclass(self.group):
            data["group_config"] = asdict(self.group)
            data["group_config"]["type"] = self.group.__class__.__name__
        data["truncate_mode"] = self.truncate_mode
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        data["element_setting_field_name"] = self.element_setting.field_name
        data["element_setting_type"] = self.element_setting.serialized_type_name
        data["element_setting_state"] = self.element_setting.to_dict()
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        self.group_name = state_dict.get("group_name", self.group_name)
        self.seq_len_field_name = state_dict.get("seq_len_field_name", self.seq_len_field_name)
        self.max_len = state_dict.get("max_len", self.max_len)
        self.padding_side = state_dict.get("padding_side", self.padding_side)
        self.time_field_name = state_dict.get("time_field_name", self.time_field_name)
        self.truncate_mode = state_dict.get("truncate_mode", self.truncate_mode)
        self.is_string_format = state_dict.get("is_string_format", self.is_string_format)
        self.separator = state_dict.get("separator", self.separator)
        element_state = state_dict.get("element_setting_state")
        if element_state:
            self.element_setting.load_state(element_state)
        self.is_fitted = self.element_setting.is_fitted

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SequenceSetting":
        raise NotImplementedError("SequenceSetting 需在 Python 代码中显式注入 element_setting 依赖。")

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        mock_interaction = {
            self.field_name: interaction[self.field_name],
            self.element_setting.field_name: interaction[self.field_name],
        }
        return self.element_setting.compute_tensor(mock_interaction, emb_modules)
</file>

<file path="emb/schema/utils.py">
from typing import Dict

import polars as pl
import torch


# Historical compatibility placeholder.
# We no longer use this in the main feature pipeline:
# - sparse nulls are mapped directly to a reserved null index
# - sequence empties are kept empty and padded by the dataset formatter
NULL_FALLBACK = "NULL_FALLBACK"


def explode_expr(field_name: str, is_string_format: bool = True, separator: str = ",") -> pl.Expr:
    expr = pl.col(field_name).drop_nulls()
    if is_string_format:
        exploded = (
            expr.cast(pl.Utf8)
            .str.split(separator)
            .explode()
            .cast(pl.Utf8)
            .str.strip_chars()
        )
    else:
        exploded = expr.explode().cast(pl.Utf8)
    valid_mask = exploded.is_not_null() & (exploded != "")
    return exploded.filter(valid_mask)


def clear_seq_expr(
    field_name: str,
    is_string_format: bool,
    separator: str,
    fill_empty_with_fallback: bool = False,
) -> pl.Expr:
    if is_string_format:
        expr = (
            pl.col(field_name)
            .cast(pl.Utf8)
            .fill_null("")
            .str.split(separator)
            .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
        )
    else:
        expr = (
            pl.col(field_name)
            .cast(pl.List(pl.Utf8))
            .fill_null([])
            .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
        )

    if not fill_empty_with_fallback:
        return expr

    return (
        pl.when(expr.list.len() > 0)
        .then(expr)
        .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
    )


def seq_length_expr(
    field_name: str,
    is_string_format: bool,
    separator: str,
    alias: str,
) -> pl.Expr:
    clean_expr = clear_seq_expr(
        field_name=field_name,
        is_string_format=is_string_format,
        separator=separator,
        fill_empty_with_fallback=False,
    )
    return clean_expr.list.len().cast(pl.UInt32).alias(alias)


def map_list_to_indices(expr: pl.Expr, vocab: Dict[str, int], oov_idx: int) -> pl.Expr:
    keys = pl.Series(list(vocab.keys()), dtype=pl.Utf8)
    vals = pl.Series(list(vocab.values()), dtype=pl.UInt32)
    default_expr = pl.lit(oov_idx, dtype=pl.UInt32) if oov_idx >= 0 else pl.lit(None, dtype=pl.UInt32)
    return expr.list.eval(
        pl.element()
        .replace_strict(old=keys, new=vals, default=default_expr)
        .cast(pl.UInt32)
    )


def mean_pooling(emb: torch.Tensor, idx_tensor: torch.Tensor, padding_zero: bool) -> torch.Tensor:
    if padding_zero:
        mask_sum = (idx_tensor != 0).sum(dim=-1, keepdim=True).clamp(min=1)
    else:
        mask_shape = idx_tensor.shape[:-1] + (1,)
        mask_sum = torch.full(mask_shape, idx_tensor.shape[-1], dtype=emb.dtype, device=idx_tensor.device)
        return emb / mask_sum
    return emb / mask_sum.to(dtype=emb.dtype)
</file>

<file path="emb/__init__.py">
from .manager import SchemaManager
</file>

<file path="emb/emblayer.py">
from typing import Iterable, List, Dict, Union, Literal, TYPE_CHECKING, Optional, Any

import numpy as np
import pandas as pd
import torch
from rosen.emb.schema import (
    EmbSetting,
    MultiSparseSetting,
    SequenceSetting,
)
from rosen.core.enum_type import FeatureSource
from torch import nn

from rosen.core.interaction import Interaction

if TYPE_CHECKING:
    from rosen.emb.manager import SchemaManager


# 将settings中的col分成用户侧和物品侧，BoleEmbLayer会把宽表Interaction 变成对应的Dict Embedding


SPLIT_METHODS = Literal["source", "name", "none", None]


class RecEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, init_method='normal', init_std=1e-4):
        super(RecEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.init_method = init_method
        self.init_std = init_std
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_method == 'normal':
            nn.init.normal_(self.embedding.weight, mean=0.0, std=self.init_std)
        elif self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight.data)
        else:
            pass
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def reinitialize(self, init_method: Optional[str] = None, init_std: Optional[float] = None):
        if init_method is not None:
            self.init_method = init_method
        if init_std is not None:
            self.init_std = init_std
        self.reset_parameters()

    def forward(self, x):
        return self.embedding(x)


class BoleEmbLayer(nn.Module):
    def __init__(self, emb_settings: Iterable[EmbSetting]):
        super(BoleEmbLayer, self).__init__()
        self.settings: List[EmbSetting] = list(emb_settings)
        self.emb_modules = nn.ModuleDict()
        self.source2settings: Dict[FeatureSource, List[EmbSetting]] = {}

        for setting in self.settings:
            if self.source2settings.get(setting.source) is None:
                self.source2settings[setting.source] = []
            self.source2settings[setting.source].append(setting)

            if setting.requires_embedding_module and setting.embedding_field_name not in self.emb_modules:
                self.emb_modules[setting.embedding_field_name] = RecEmbedding(
                    num_embeddings=setting.num_embeddings,
                    embedding_dim=setting.embedding_dim,
                    padding_idx={True: 0, False: None}[setting.padding_zero]
                )

    def forward(self, interaction, split_by: SPLIT_METHODS="none") -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor]:
        split_by = split_by.lower() if isinstance(split_by, str) else None
        if split_by not in ("source", "name", "none", None):
            raise ValueError

        output_embs = {}
        for source, settings in self.source2settings.items():
            if not settings:
                continue
            emb_list = []
            for setting in settings:
                emb = setting.compute_tensor(interaction, self.emb_modules)
                emb_list.append([setting.field_name, emb])

            if emb_list:
                if split_by == "source":
                    output_embs[source] = torch.cat([emb for name, emb in emb_list], dim=-1)
                elif split_by == "name":
                    for name, emb in emb_list:
                        output_embs[name] = emb
                else:
                    output_embs["all"] = output_embs.get("all", [])
                    output_embs["all"].extend([emb for name, emb in emb_list])

        if split_by is None or split_by == "none":
            if output_embs.get("all", None) is None:
                return None
            else:
                return torch.cat(output_embs.get("all"), dim=-1)
        else:
            return output_embs


class SideEmb(nn.Module):
    def __init__(self, all_settings: Iterable[EmbSetting],
                 target_sources: Union[FeatureSource, Iterable[FeatureSource]]=None):
        super(SideEmb, self).__init__()

        # 处理单 Source 和多 Source 的兼容

        if target_sources is not None:
            if not isinstance(target_sources, Iterable):
                target_sources = [target_sources]
            # 核心逻辑：过滤出属于自己这一侧的 settings
            side_settings = [s for s in all_settings if s.source in target_sources]
        else:
            side_settings = all_settings

        self.target_sources = target_sources  # 转为 tuple 方便后续固定顺序遍历
        self.embedding = BoleEmbLayer(side_settings)
        self.embedding_size = sum([s.embedding_dim for s in side_settings])

    def forward(self, interaction, split_by: SPLIT_METHODS="none") -> Union[torch.Tensor, Dict[FeatureSource, torch.Tensor]]:
        # 直接把大宽表吐出的 interaction 传给底层
        emb_dict = self.embedding(interaction, split_by)
        return emb_dict


# 具体化实现，极其干净优雅
class UserSideEmb(SideEmb):
    VALID_SOURCE = (FeatureSource.USER_ID, FeatureSource.USER)

    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings, self.VALID_SOURCE)


class ItemSideEmb(SideEmb):
    VALID_SOURCE = (FeatureSource.ITEM_ID, FeatureSource.ITEM)

    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings, self.VALID_SOURCE)


class InterSideEmb(SideEmb):
    VALID_SOURCE = (FeatureSource.INTERACTION,)

    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings, self.VALID_SOURCE)

# 无任何约束
class CustomSideEmb(SideEmb):
    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings)

"""已废弃"""
class ProfileEncoder(nn.Module):
    """
    全自动静态画像编码器：告别硬编码，完全数据驱动！
    """

    def __init__(self, settings: List['EmbSetting'], profile_path: str,
                 id_source: FeatureSource, feature_source: FeatureSource):
        super().__init__()

        # 1. 解析图纸：找出谁是主键 (ID)，谁是属性 (Features)
        self.id_setting = next((s for s in settings if s.source == id_source), None)
        if not self.id_setting:
            raise ValueError(f"在 settings 中未找到主键源: {id_source}")

        self.feature_settings = [s for s in settings if s.source == feature_source]
        self.feature_names = [s.field_name for s in self.feature_settings]

        # 2. 初始化底层 Embedding 层 (直接复用你写好的 SideEmb)
        self.side_emb = SideEmb(settings, target_sources=[id_source, feature_source])

        # 3. 读取静态宽表
        df = pd.read_parquet(profile_path)

        # 4. 构建地基 (最大 ID 决定了查找表的行数)
        id_col = self.id_setting.field_name
        num_entities = df[id_col].max() + 1
        entity_ids = torch.tensor(df[id_col].astype(np.int64).values, dtype=torch.long)

        # 5. 全自动装配流水线
        for setting in self.feature_settings:
            field = setting.field_name

            # --- 分支 A：处理变长序列特征 (必须 Padding) ---
            if isinstance(setting, MultiSparseSetting):
                max_len = getattr(setting, 'max_tag_len', 50)
                padded_matrix = self._pad_sequences(df[field], max_len)

                # 初始化空矩阵 [num_entities, max_len]
                buffer_tensor = torch.zeros((num_entities, max_len), dtype=torch.long)
                vals = torch.tensor(padded_matrix, dtype=torch.long)

            # --- 分支 B：处理单值特征 ---
            else:
                # 初始化一维向量 [num_entities]
                buffer_tensor = torch.zeros(num_entities, dtype=torch.long)
                clean_series = df[field].fillna(0).astype(np.int64)
                vals = torch.tensor(clean_series.values, dtype=torch.long)

            # 散列赋值：自动对齐坑位 (完美解决残缺错位问题)
            buffer_tensor[entity_ids] = vals

            # 动态注册为模型的 Buffer (名字比如: buf_tag, buf_author_id)
            self.register_buffer(f"buf_{field}", buffer_tensor)

    def _pad_sequences(self, series: pd.Series, max_len: int, pad_val: int = 0) -> np.ndarray:
        """高性能 Numpy 序列补齐工具，自带数据清洗"""
        padded = np.full((len(series), max_len), pad_val, dtype=np.int64)
        for i, seq in enumerate(series):
            # 防御由于 left join 产生的 NaN/None
            if seq is None or (isinstance(seq, float) and np.isnan(seq)) or len(seq) == 0:
                continue

            valid_len = min(len(seq), max_len)
            # 【核心修复】不仅截断，还要强制清洗内部元素的类型为 int64
            padded[i, :valid_len] = np.array(seq[:valid_len], dtype=np.int64)

        return padded

    def forward(self, query_ids: torch.Tensor, split_by: SPLIT_METHODS= "none"):
        """
        前向传播：根据传入的 query_ids (比如一串 item_id)，
        瞬间从缓存中拉出所有特征，组装成 interaction 字典，喂给 SideEmb。
        """
        # 1. 构建虚拟 Interaction 字典
        interaction = {self.id_setting.field_name: query_ids}
        # 2. 动态查表
        for field in self.feature_names:
            # 取出注册好的 Buffer 矩阵
            buffer_tensor = getattr(self, f"buf_{field}")
            interaction[field] = buffer_tensor[query_ids]

        # 3. 完美接入你的 SideEmb
        return self.side_emb(interaction, split_by=split_by)

class UserProfileEncoder(ProfileEncoder):
    ID_SOURCE = FeatureSource.USER_ID
    FEATURE_SOURCE = FeatureSource.USER
    def __init__(self, settings: List['EmbSetting'], profile_path: str,):
        super().__init__(settings, profile_path,
                         id_source=self.ID_SOURCE, feature_source=self.FEATURE_SOURCE)

class ItemProfileEncoder(ProfileEncoder):
    ID_SOURCE = FeatureSource.ITEM_ID
    FEATURE_SOURCE = FeatureSource.ITEM
    def __init__(self, settings: List['EmbSetting'], profile_path: str,):
        super().__init__(settings, profile_path,
                         id_source=self.ID_SOURCE, feature_source=self.FEATURE_SOURCE)


class SeqEmbedder(nn.Module):
    def __init__(self, seq_field_name, settings: List[EmbSetting], profile_encoder: ProfileEncoder):
        super().__init__()
        self.settings = settings
        filtered_list = [setting for setting in settings if setting.field_name == seq_field_name]
        if len(filtered_list) != 1:
            raise ValueError
        self.id_setting = filtered_list[0]
        if not isinstance(self.id_setting, SequenceSetting):
            raise ValueError
        self.id_setting: SequenceSetting = self.id_setting
        self.seq_field_name = seq_field_name
        self.seq_len_field_name = self.id_setting.seq_len_field_name

        self.profile_encoder = profile_encoder

    def forward(self, interaction: Interaction, split_by: SPLIT_METHODS="none"):
        inter_dict = interaction.interaction
        seq = inter_dict[self.seq_field_name]
        seq_len = inter_dict[self.seq_len_field_name]
        emb_seq = self.profile_encoder.forward(seq, split_by)
        return emb_seq, seq_len


class EmbView:
    """
    轻量级特征视图：
    本身没有任何参数，只负责将请求转发给底层的 OmniEmbLayer，并自动绑定 target_sources。
    """
    def __init__(self,
                 omni_layer: 'OmniEmbLayer',
                 target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
                 include_fields: Iterable[str] = None,
                 exclude_fields: Iterable[str] = None):
        super().__init__()
        # 引用同一个巨无霸底座
        self.omni = omni_layer
        self.target_sources = target_sources
        self.include_fields = tuple(include_fields) if include_fields is not None else None
        self.exclude_fields = tuple(exclude_fields) if exclude_fields is not None else None

    def forward(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
        return self.omni(
            interaction,
            split_by=split_by,
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    def __call__(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[
        Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
        return self.omni(
            interaction,
            split_by=split_by,
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    @property
    def embedding_dim(self) -> int:
        return self.omni.get_output_dim(
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    @property
    def selected_settings(self) -> tuple[EmbSetting, ...]:
        return tuple(
            self.omni.get_settings(
                target_sources=self.target_sources,
                include_fields=self.include_fields,
                exclude_fields=self.exclude_fields,
            )
        )

    @property
    def field_segments(self) -> tuple[tuple[str, int, int], ...]:
        cursor = 0
        segments = []
        for setting in self.selected_settings:
            start = cursor
            cursor += int(setting.embedding_dim)
            segments.append((setting.field_name, start, cursor))
        return tuple(segments)

    @property
    def field_slices(self) -> Dict[str, slice]:
        return {
            field_name: slice(start, stop)
            for field_name, start, stop in self.field_segments
        }


class SeqGroupView(EmbView):
    """
    专为序列组设计的强化版视图：
    不仅能获取序列特征，还能自动获取严格对齐的 Target 特征和序列长度。
    """

    def __init__(self, omni_layer, group_name: str, include_fields: Iterable[str]):
        super().__init__(omni_layer, include_fields=include_fields)
        self.group_name = group_name

        # 预加载对齐的 element 设置与 seq_len 字段
        self.target_settings = []
        self.seq_len_field = None
        self.time_field = None
        self._target_forward_valid = True

        for field in self.include_fields:
            setting = next((s for s in self.omni.settings if s.field_name == field), None)
            if setting is None:
                continue

            if hasattr(setting, 'element_setting'):
                self.target_settings.append(setting.element_setting)
            else:
                self._target_forward_valid = False

            if self.seq_len_field is None and hasattr(setting, 'seq_len_field_name'):
                self.seq_len_field = setting.seq_len_field_name
            if hasattr(setting, 'time_field_name'):
                setting_time_field = setting.time_field_name
                if setting_time_field is None:
                    continue
                if self.time_field is None:
                    self.time_field = setting_time_field
                elif self.time_field != setting_time_field:
                    raise ValueError(
                        f"Seq group '{self.group_name}' has inconsistent time_field_name: "
                        f"{self.time_field} vs {setting_time_field}"
                    )

    def forward_target(self, interaction, split_by: str = "none"):
        # 实际上所有 setting 都应暴露 element_setting，forward_target 才不会有歧义
        if split_by == "source":
            raise ValueError("Target sequence view strictly supports split_by='none' or 'name'.")

        computed_results = []
        for target_s in self.target_settings:
            emb = target_s.compute_tensor(interaction, self.omni.emb_modules)
            computed_results.append((target_s, emb))
        return self.omni.format_output(computed_results, split_by)

    def fetch_all(self, interaction, include_time: bool = False):
        """默认返回 (seq_emb, target_emb, seq_len)，需要时间时返回四元组。"""
        seq_emb = self.forward(interaction, split_by="none")
        tar_emb = (
            self.forward_target(interaction, split_by="none")
            if self._target_forward_valid
            else ValueError("存在部分 setting 没有 element_setting，函数中断")
        )
        seq_len = interaction[self.seq_len_field] if self.seq_len_field else None
        hist_time = interaction[self.time_field] if self.time_field else None
        if include_time:
            return seq_emb, tar_emb, seq_len, hist_time
        return seq_emb, tar_emb, seq_len

class OmniEmbLayer(nn.Module):
    """
    巨无霸统一 Embedding 层：
    内部纳管所有 settings。forward 时可通过 target_sources 动态指定范围。
    彻底取代 SideEmb、UserSideEmb、ItemSideEmb 等一众子类。
    """
    def __init__(self, emb_settings: Iterable[EmbSetting] = None, manager: 'SchemaManager' = None, init_method='normal', init_std=1e-3):
        super(OmniEmbLayer, self).__init__()
        if manager is not None:
            emb_settings = manager.settings
        if emb_settings is None:
            raise ValueError("OmniEmbLayer requires manager or emb_settings.")

        self.manager = manager
        self.settings: List[EmbSetting] = list(emb_settings)
        self.emb_modules = nn.ModuleDict()
        self.source2settings: Dict[FeatureSource, List[EmbSetting]] = {}

        self.group2fields = {}
        self._init_embedding(init_method=init_method, init_std=init_std)

        self.seq_groups: Dict[str, SeqGroupView] = {}
        for g_name, fields in self.group2fields.items():
            self.seq_groups[g_name] = SeqGroupView(self, group_name=g_name, include_fields=fields)

        self.whole = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
                                                   FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                   FeatureSource.INTERACTION))
        self.user_all = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER))
        self.item_all = EmbView(self, target_sources=(FeatureSource.ITEM_ID, FeatureSource.ITEM))
        self.inter = EmbView(self, target_sources=(FeatureSource.INTERACTION,))

        self.user_id = EmbView(self, target_sources=(FeatureSource.USER_ID,))
        self.item_id = EmbView(self, target_sources=(FeatureSource.ITEM_ID,))


        domain_fields = tuple(manager.domain_fields) if manager is not None else tuple()
        domain_field = manager.domain_field if manager is not None else None
        self.domain = EmbView(self, include_fields=domain_fields)
        self.domain_id = EmbView(
            self,
            include_fields=((domain_field,) if domain_field is not None else tuple()),
        )
        self.whole_without_domain = EmbView(
            self,
            target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
                            FeatureSource.ITEM_ID, FeatureSource.ITEM,
                            FeatureSource.INTERACTION),
            exclude_fields=domain_fields,
        )
        self.inter_without_domain = EmbView(
            self,
            target_sources=(FeatureSource.INTERACTION,),
            exclude_fields=domain_fields,
        )

    def _init_embedding(self, init_method='normal', init_std=1e-3):
        for setting in self.settings:
            # A. 维护 Source 映射
            self.source2settings.setdefault(setting.source, []).append(setting)

            # B. 确定哪些特征需要物理权重 (nn.Embedding)
            if setting.requires_embedding_module and setting.embedding_field_name not in self.emb_modules:
                self.emb_modules[setting.embedding_field_name] = RecEmbedding(
                    num_embeddings=setting.num_embeddings,
                    embedding_dim=setting.embedding_dim,
                    padding_idx={True: 0, False: None}[setting.padding_zero],
                    init_method=init_method, init_std=init_std
                )

            # C. 自动收集带有组名标签的序列特征
            if hasattr(setting, 'group_name') and setting.group_name:
                self.group2fields.setdefault(setting.group_name, []).append(setting.field_name)

    def reinitialize_large_vocab_embeddings(
            self,
            vocab_size_threshold: int,
            *,
            init_method: Optional[str] = None,
            init_std: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reinitialize embedding modules whose linked setting vocab size exceeds the threshold.

        Notes:
        - This method is opt-in only. Nothing in the default training flow calls it.
        - When multiple settings share one embedding table, the table is reinitialized at most once.
        - The threshold compares `setting.vocab_size`, not `num_embeddings`, so padding/OOV slots
          do not affect the trigger.
        """
        vocab_size_threshold = int(vocab_size_threshold)
        embedding_meta: Dict[str, Dict[str, Any]] = {}

        for setting in self.settings:
            if not setting.requires_embedding_module:
                continue

            emb_field_name = setting.embedding_field_name
            meta = embedding_meta.setdefault(
                emb_field_name,
                {
                    "embedding_field_name": emb_field_name,
                    "field_names": [],
                    "vocab_size": 0,
                    "num_embeddings": int(setting.num_embeddings),
                },
            )
            if setting.field_name not in meta["field_names"]:
                meta["field_names"].append(setting.field_name)
            meta["vocab_size"] = max(int(meta["vocab_size"]), int(setting.vocab_size))
            meta["num_embeddings"] = max(int(meta["num_embeddings"]), int(setting.num_embeddings))

        reinitialized = []
        for emb_field_name, meta in embedding_meta.items():
            if meta["vocab_size"] <= vocab_size_threshold:
                continue

            if emb_field_name not in self.emb_modules:
                continue
            emb_module = self.emb_modules[emb_field_name]

            emb_module.reinitialize(init_method=init_method, init_std=init_std)
            reinitialized.append(meta)

        return sorted(reinitialized, key=lambda item: item["vocab_size"], reverse=True)


    def forward(
            self,
            interaction,
            split_by: SPLIT_METHODS = "none",
            target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
            include_fields: Iterable[str] = None,
            exclude_fields: Iterable[str] = None,
    ) -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:

        split_by = split_by.lower() if isinstance(split_by, str) else None
        if split_by not in ("source", "name", "none", None):
            raise ValueError(f"Invalid split_by: {split_by}")

        # 1. 路由过滤：找出本次 forward 需要处理的所有 Setting
        valid_settings = self._filter_settings(target_sources, include_fields, exclude_fields, split_by)
        computed = [(s, s.compute_tensor(interaction, self.emb_modules)) for s in valid_settings]
        return self.format_output(computed, split_by)


    def _filter_settings(self, target_sources, include_fields, exclude_fields, split_by) -> List[EmbSetting]:
        """职责一：专门负责解析 target_sources 和 include/exclude_fields，返回合法的 Setting 列表"""
        target_sources = set(target_sources) if target_sources else None
        include_fields = set(include_fields) if include_fields else None
        exclude_fields = set(exclude_fields) if exclude_fields else None
        valid_settings = []
        for source, settings in self.source2settings.items():
            if target_sources and source not in target_sources:
                continue
            for setting in settings:
                if include_fields and setting.field_name not in include_fields:
                    continue
                if exclude_fields and setting.field_name in exclude_fields:
                    continue

                if (split_by == "none" or split_by is None) and setting.is_sequence_setting:
                    if not (include_fields and setting.field_name in include_fields):
                        continue

                valid_settings.append(setting)
        return valid_settings

    def format_output(self, computed_results: List[tuple], split_by: str):
        """职责三：根据 split_by 策略打包最终输出的 Tensor 或 Dict"""
        if not computed_results:
            return None

        if split_by == "name":
            return {setting.field_name: emb for setting, emb in computed_results}

        if split_by == "source":
            output_dict = {}
            for setting, emb in computed_results:
                output_dict.setdefault(setting.source, []).append(emb)
            return {source: torch.cat(embs, dim=-1) for source, embs in output_dict.items()}

        all_embs = [emb for _, emb in computed_results]
        return torch.cat(all_embs, dim=-1)

    def get_output_dim(self,
                       target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
                       include_fields: Iterable[str] = None,
                       exclude_fields: Iterable[str] = None) -> int:
        """
        获取指定 target_sources 下拼接后 Embedding 的总维度大小。
        这是为了弥补移除 SideEmb 后，下游网络无法直接获取 self.embedding_size 的问题。
        """
        valid_settings = self.get_settings(
            target_sources=target_sources,
            include_fields=include_fields,
            exclude_fields=exclude_fields,
        )
        return sum(setting.embedding_dim for setting in valid_settings)

    def get_settings(
            self,
            target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
            include_fields: Iterable[str] = None,
            exclude_fields: Iterable[str] = None,
    ) -> tuple[EmbSetting, ...]:
        return tuple(
            self._filter_settings(
                target_sources=target_sources,
                include_fields=include_fields,
                exclude_fields=exclude_fields,
                split_by="none",
            )
        )
</file>

<file path="emb/manager.py">
# ==========================================
# 2. 调度层 (Manager Layer) - 计算图统筹
# ==========================================
import gc
import json
import shutil
import ctypes
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union
import polars as pl

from rosen.emb.split import SplitConfig, SplitContext, create_split_strategy
from rosen.emb.schema import EmbSetting, EmbType, MultiSparseSetting, SparseEmbSetting
from rosen.core.enum_type import FeatureSource

from rosen.utils.sort import sort_parquet_inplace
from rosen.utils.shuffle import write_shuffled_parquet


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def collect_lazy(lf: pl.LazyFrame, streaming: bool = False) -> pl.DataFrame:
    if not streaming:
        return lf.collect()
    try:
        return lf.collect(streaming=True)
    except TypeError:
        return lf.collect(engine="streaming")
    except ValueError:
        return lf.collect()


def trim_process_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


VOCAB_DIR_NAME = "vocab"
VOCAB_VALUE_COL = "value"
VOCAB_ID_COL = "id"


class SchemaManager:
    ITEM_PROFILE_NAME = "item_profile.parquet"
    USER_PROFILE_NAME = "user_profile.parquet"
    SCHEMA_META_NAME = "feature_meta.json"
    WHOLE_DATA_NAME = "whole_dataframe.parquet"
    def __init__(self, settings_list: List[EmbSetting], schema_dir: str,
                 time_field=None,
                 label_fields: Union[str, Iterable[str]]=None,
                 domain_fields: Union[str, Iterable[str]]=None
                 ):
        self.settings = settings_list

        ## path & dir
        self.schema_dir = Path(schema_dir)
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self.meta_filepath = self.schema_dir / self.SCHEMA_META_NAME

        ## field_name
        self.uid_field = next(
            (s.field_name for s in self.settings if s.source == FeatureSource.USER_ID),
            None
        )
        self.iid_field =next(
            (s.field_name for s in self.settings if s.source == FeatureSource.ITEM_ID),
            None
        )
        self.time_field = time_field
        self._label_fields = ensure_list(label_fields)
        self._domain_fields = ensure_list(domain_fields)

    @property
    def label_field(self):
        return next(iter(self._label_fields), None)
    @property
    def label_fields(self) -> tuple:
        return tuple(self._label_fields)
    @property
    def domain_field(self):
        return next(iter(self._domain_fields), None)
    @property
    def domain_fields(self) -> tuple:
        return tuple(self._domain_fields)


    def _parse_output_dir(self, output_dir=None):
        if output_dir is None:
            output_dir = self.schema_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        return output_dir


    def prepare_data(self, lazy_df: pl.LazyFrame, output_dir: Union[Path, str]=None, redo=False):
        """
        不建议使用，全自动化流程：自动判断输入文件类型，自动构建计算图
        """
        output_parquet = self._parse_output_dir(output_dir) / self.WHOLE_DATA_NAME
        if self.meta_filepath.exists() and output_parquet.exists() and not redo:
            print("[Info] 检测到已存在元数据和目标文件，跳过处理直接加载 Schema。")
            self.load_schema()
            return pl.scan_parquet(output_parquet)

        print("[*] 正在执行计算图截断与中间态落盘，请稍候...")
        lazy_df = self._make_checkpoint(lazy_df)

        print("[*] 阶段一：构建 Fit 计算图...")
        fit_exprs = []
        for setting in self.settings:
            if not setting.is_fitted:
                fit_exprs.extend(setting.get_fit_exprs())

        if fit_exprs:
            print(f"[*] 启动单次全表扫描获取统计量，包含 {len(fit_exprs)} 个计算节点...")
            fit_result = lazy_df.select(fit_exprs).collect()
            for setting in self.settings:
                if not setting.is_fitted:
                    setting.parse_fit_result(fit_result)
            print("[+] 统计量计算与映射表构建完成。")
        else:
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
        print("[*] 阶段二：构建 Transform 计算图...")
        transform_exprs = self._build_transform_exprs()
        print(f"[*] 启动流式转换计算，引擎正在将数据写入 {output_parquet} ...")
        lazy_df.with_columns(transform_exprs) \
            .sink_parquet(output_parquet)
        self.save_schema()
        print("[+] 预处理全流程结束！")
        return pl.scan_parquet(output_parquet)


    def fit(self, train_raw_lf: pl.LazyFrame, low_memory: bool = False):
        """
        阶段一：仅使用训练集拟合统计量和词表。

        :param train_raw_lf: Polars LazyFrame 格式的训练数据
        :param low_memory: 是否开启低内存模式。
                           - False (默认，推荐): 单次全表扫描，所有特征并行计算。速度极快，但内存开销较大。
                           - True (内存受限时): 逐个特征扫描。内存占用极低，但 I/O 放大，耗时成倍增加。
        """
        if self.meta_filepath.exists():
            self.load_schema()
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
            return
        unfitted_settings = [s for s in self.settings if not s.is_fitted]
        if not unfitted_settings:
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
            self.save_schema()
            return

        if low_memory:
            print(f"[*] 启动极低内存 Fit 模式，将对 {len(unfitted_settings)} 个特征进行逐列扫描 (耗时较长)...")
            for i, setting in enumerate(unfitted_settings):
                fit_exprs = setting.get_fit_exprs()
                if not fit_exprs:
                    continue
                print(f"    -> 正在拟合 [{i + 1}/{len(unfitted_settings)}]: {setting.field_name}")
                col_fit_result = collect_lazy(train_raw_lf.select(fit_exprs), streaming=True)
                setting.parse_fit_result(col_fit_result)
                del col_fit_result # 显式释放哈希表内存
                trim_process_memory()
            print("[+] 逐列统计量与词表构建完成。")
        else:
            print(f"[*] 启动高性能并行 Fit 模式，单次扫描全表计算 {len(unfitted_settings)} 个统计量...")
            fit_exprs = [expr for s in unfitted_settings for expr in s.get_fit_exprs()]
            if fit_exprs:
                # 即使是并行，也建议开启 streaming=True，让 Polars 底层自己尽力优化和 Spilling
                fit_result = collect_lazy(train_raw_lf.select(fit_exprs), streaming=True)
                for setting in unfitted_settings:
                    setting.parse_fit_result(fit_result)
        self.save_schema()

    def _checkpoint_path(self, file_name: Union[str, Path]) -> Path:
        path = Path(file_name)
        if not path.is_absolute():
            path = self.schema_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def make_checkpoint(self, lf, file_name="custom_checkpoint.parquet", redo=True):
        path = self._checkpoint_path(file_name)
        if not path.exists() or redo:
            self._make_checkpoint(lf, file_name)
            return pl.scan_parquet(path)
        else:
            return pl.scan_parquet(path)

    def make_sorted_checkpoint(
        self,
        lf,
        file_name="custom_sorted_checkpoint.parquet",
        sort_by=None,
        descending=False,
        redo=True,
        memory_limit: str = "84GB",
    ):
        if sort_by is None:
            raise ValueError("sort_by is required for make_sorted_checkpoint.")
        path = self._checkpoint_path(file_name)
        if not path.exists() or redo:
            self._make_checkpoint(lf, file_name)
            sort_parquet_inplace(
                path,
                sort_by,
                descending=descending,
                memory_limit=memory_limit,
                temp_dir=path.parent / f"_{path.stem}_sort_tmp",
            )
        return pl.scan_parquet(path)

    def make_shuffled_checkpoint(
        self,
        lf,
        file_name="custom_shuffled_checkpoint.parquet",
        seed: int = 2026,
        bucket_count: int = 128,
        batch_size: int = 65536,
        redo: bool = True,
        exact: bool = False,
        mode: Literal["target_sort", "bucket_chunk"] = "target_sort",
        compression: str = "zstd",
    ):
        path = self._checkpoint_path(file_name)
        if path.exists() and not redo:
            return pl.scan_parquet(path)

        source_path = path.parent / f"_{path.stem}_unshuffled.parquet"
        if source_path.exists():
            source_path.unlink()
        try:
            lf.sink_parquet(source_path, compression=compression)
            write_shuffled_parquet(
                [source_path],
                path,
                seed=seed,
                bucket_count=bucket_count,
                batch_size=batch_size,
                compression=compression,
                redo=True,
                exact=exact,
                mode=mode,
                temp_dir=path.parent / f"_{path.stem}_shuffle_tmp",
            )
        finally:
            if source_path.exists():
                source_path.unlink()
        return pl.scan_parquet(path)

    def transform(self, raw_lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        阶段二：根据已固化的 Schema 进行流式转换
        """
        transform_exprs = self._build_transform_exprs()
        return raw_lf.with_columns(transform_exprs)

    def transform_csv_partitioned(
        self,
        input_paths: Union[str, Path, Iterable[Union[str, Path]]],
        output_dir: Union[str, Path],
        uid_field: Optional[str] = None,
        n_parts: int = 100,
        batch_size: int = 50000,
        batches_per_read: int = 1,
        buffer_flush_rows: int = 500000,
        read_csv_kwargs: Optional[Mapping[str, Any]] = None,
        redo: bool = True,
    ) -> List[Path]:
        """
        分批读取 CSV，边 transform 边按原始 uid hash 分区落盘。

        输出结构:
            output_dir/p000/c0000.parquet
            output_dir/p000/c0001.parquet
            output_dir/p001/c0000.parquet
            ...
        """
        uid_field = uid_field or self.uid_field
        if uid_field is None:
            raise ValueError("uid_field is required when manager.uid_field is not set.")
        if n_parts <= 0:
            raise ValueError("n_parts must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batches_per_read <= 0:
            raise ValueError("batches_per_read must be positive.")
        if buffer_flush_rows <= 0:
            raise ValueError("buffer_flush_rows must be positive.")

        output_dir = Path(output_dir)
        if output_dir.exists() and redo:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        part_dirs = [output_dir / f"p{pid:03d}" for pid in range(n_parts)]
        for part_dir in part_dirs:
            part_dir.mkdir(parents=True, exist_ok=True)

        existing = [path for part_dir in part_dirs for path in part_dir.glob("*.parquet")]
        if existing and not redo:
            return part_dirs

        transform_exprs = self._build_transform_exprs()
        part_col = "__bb_part"
        buffers = {pid: [] for pid in range(n_parts)}
        buffer_rows = {pid: 0 for pid in range(n_parts)}
        file_counts = {pid: 0 for pid in range(n_parts)}

        def flush(pid: int):
            if not buffers[pid]:
                return
            out = pl.concat(buffers[pid], how="vertical")
            path = part_dirs[pid] / f"c{file_counts[pid]:04d}.parquet"
            out.write_parquet(path)
            file_counts[pid] += 1
            buffers[pid] = []
            buffer_rows[pid] = 0

        csv_kwargs = dict(read_csv_kwargs or {})
        csv_kwargs["batch_size"] = batch_size
        total_rows = 0

        for input_path in ensure_list(input_paths):
            input_path = Path(input_path)
            print(f"[*] transform csv partitioned: {input_path}")
            reader = pl.read_csv_batched(input_path, **csv_kwargs)

            while True:
                batches = reader.next_batches(batches_per_read)
                if batches is None:
                    break

                for raw_df in batches:
                    if raw_df.height == 0:
                        continue
                    if uid_field not in raw_df.columns:
                        raise ValueError(f"uid_field '{uid_field}' not found in {input_path}")
                    if part_col in raw_df.columns:
                        raise ValueError(f"reserved column '{part_col}' already exists in {input_path}")

                    df = (
                        raw_df.lazy()
                        .with_columns(
                            (pl.col(uid_field).hash(seed=0) % n_parts)
                            .cast(pl.UInt32)
                            .alias(part_col)
                        )
                        .with_columns(transform_exprs)
                        .pipe(collect_lazy, streaming=True)
                    )

                    for key, part_df in df.partition_by(
                        part_col,
                        maintain_order=False,
                        include_key=True,
                        as_dict=True,
                    ).items():
                        pid = int(key[0])
                        part_df = part_df.drop(part_col)
                        buffers[pid].append(part_df)
                        buffer_rows[pid] += part_df.height
                        if buffer_rows[pid] >= buffer_flush_rows:
                            flush(pid)

                    total_rows += df.height
                    del raw_df, df

            print(f"    -> done, cumulative rows={total_rows:,}")

        for pid in range(n_parts):
            flush(pid)

        print(f"[+] partitioned transform done: rows={total_rows:,}, output={output_dir}")
        return part_dirs

    def extract_sequence_partitions(
        self,
        flat_part_dirs: Iterable[Union[str, Path]],
        output_dir: Union[str, Path],
        max_seq_len: int,
        user_col: str,
        time_col: str,
        feature_mapping: Mapping[str, str],
        seq_len_col: str = "seq_len",
        label_col: Optional[str] = None,
        positive_label: Any = 1,
        select_fields: Optional[Iterable[str]] = None,
        redo: bool = True,
    ) -> List[Path]:
        from rosen.utils.sequential import extract_history_sequences

        output_dir = Path(output_dir)
        if output_dir.exists() and redo:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: List[Path] = []
        fields = list(select_fields) if select_fields is not None else None

        for idx, part in enumerate(flat_part_dirs):
            part = Path(part)
            out_path = output_dir / f"{part.name}.parquet"
            outputs.append(out_path)
            if out_path.exists() and not redo:
                continue

            if part.is_dir():
                files = sorted(part.glob("*.parquet"))
            else:
                files = [part] if part.exists() else []

            if not files:
                continue

            seq_lf = extract_history_sequences(
                pl.scan_parquet([str(path) for path in files]),
                max_seq_len=max_seq_len,
                user_col=user_col,
                time_col=time_col,
                feature_mapping=dict(feature_mapping),
                seq_len_col=seq_len_col,
                label_col=label_col,
                positive_label=positive_label,
            )
            if fields is not None:
                seq_lf = seq_lf.select(fields)
            seq_lf.sink_parquet(out_path)
            print(f"    -> seq part {idx + 1}: {out_path}")

        return outputs

    def _build_transform_exprs(self) -> List[pl.Expr]:
        transform_exprs: List[pl.Expr] = []
        seen_output_names = set()
        for setting in self.settings:
            if setting.emb_type == EmbType.UNKNOWN:
                continue
            expr = setting.get_transform_expr()
            expr_list = expr if isinstance(expr, list) else [expr]
            for item in expr_list:
                try:
                    output_name = item.meta.output_name()
                except Exception:
                    output_name = None
                if output_name is not None and output_name in seen_output_names:
                    continue
                if output_name is not None:
                    seen_output_names.add(output_name)
                transform_exprs.append(item)
        return transform_exprs

    def _make_checkpoint(self, lazy_df, file_name="temp_checkpoint.parquet"):
        temp_checkpoint_path = self._checkpoint_path(file_name)
        lazy_df.sink_parquet(temp_checkpoint_path)
        lazy_df = pl.scan_parquet(temp_checkpoint_path)
        return lazy_df

    # NOTE 不推荐使用
    def generate_profiles(self, lazy_df: pl.LazyFrame, output_dir: str=None, redo=False):
        """
        利用 FeatureSource 统筹提取并保存静态画像表 (Profile)
        """
        item_profile_path = self._parse_output_dir(output_dir) / self.ITEM_PROFILE_NAME
        user_profile_path = self._parse_output_dir(output_dir) / self.USER_PROFILE_NAME

        print("[*] 阶段三：开始提取静态特征查找表 (Profiles)...")
        # 1. Item Profile
        item_features = [setting.field_name for setting in self.settings
                         if setting.source == FeatureSource.ITEM]
        if self.iid_field and (not item_profile_path.exists() or redo):
            # 确保主键在提取列表中
            if self.iid_field not in item_features:
                item_features.insert(0, self.iid_field)

            print(f"[*] 正在提取 Item 画像，包含特征: {item_features}")
            (
                lazy_df
                .select(item_features)
                .drop_nulls(subset=[self.iid_field])  # 剔除由于 left join 产生的空物品
                .unique(subset=[self.iid_field])  # 按物品 ID 去重
                .sort(self.iid_field)  # 【极其关键】按 ID 从小到大排序！
                .sink_parquet(item_profile_path)
            )
        # 2. 提取 User Profile (同理)
        user_features = [
            setting.field_name
            for setting in self.settings
            if setting.source == FeatureSource.USER
        ]

        if self.uid_field and (not user_profile_path.exists() or redo):
            if self.uid_field not in user_features:
                user_features.insert(0, self.uid_field)

            print(f"[*] 正在提取 User 画像，包含特征: {user_features}")
            (
                lazy_df
                .select(user_features)
                .drop_nulls(subset=[self.uid_field])
                .unique(subset=[self.uid_field])
                .sort(self.uid_field)
                .sink_parquet(user_profile_path)
            )
        print("[+] 静态查找表提取完成！")

    def split_dataset(
        self,
        lf: pl.LazyFrame,
        strategy: Literal['loo', 'time', 'sequential_ratio', 'random_ratio'] = "random_ratio",
        output_dir: str = None,
        redo: bool = False,
        config: Optional[SplitConfig] = None,
        **kwargs,
    ):
        output_dir = self._parse_output_dir(output_dir)
        split_ctx = SplitContext(
            uid_field=self.uid_field,
            iid_field=self.iid_field,
            time_field=self.time_field,
            checkpoint_fn=self._make_checkpoint  # 把方法当做变量传递进去 (Callback)
        )
        splitter = create_split_strategy(strategy=strategy, context=split_ctx, config=config, **kwargs)
        return splitter.split(lf, output_dir, redo)

    def save_as_dataset(self, train_lf, valid_lf=None, test_lf=None, output_dir: str = None, redo: bool = False):
        """
        如果预设的split_dataset不和你的想法，那么把三个lf传过来，这边直接保存。利好Kuairand等预设test的数据集
        """
        output_dir = self._parse_output_dir(output_dir)
        train_path = output_dir / "train.parquet"
        valid_path = output_dir / "valid.parquet"
        test_path = output_dir / "test.parquet"
        split_paths = str(train_path), str(valid_path), str(test_path)
        if not redo and (train_path.exists() or valid_path.exists() or test_path.exists()):
            return split_paths
        gc.collect()
        for lf, save_path, name in zip([train_lf, valid_lf, test_lf], split_paths, ["Train", "Valid", "Test"]):
            if lf is not None:
                print(f"    -> 正在写入 {name} 集合到 {save_path}")
                lf.sink_parquet(str(save_path),)
        print("[+] 所有数据落盘完成！")
        return split_paths

    def save_schema(self):
        vocab_dir = self.meta_filepath.parent / VOCAB_DIR_NAME
        vocab_dir.mkdir(parents=True, exist_ok=True)
        meta_data = []
        for setting in self.settings:
            data = setting.to_dict()
            for state in (data, data.get("element_setting_state")):
                if not isinstance(state, dict) or not isinstance(state.get("vocab"), dict) or not state["vocab"]:
                    continue
                ref = f"{VOCAB_DIR_NAME}/{state['field_name']}.parquet"
                vocab = state["vocab"]
                rows = sorted(vocab.items(), key=lambda item: int(item[1]))
                pl.DataFrame(
                    {
                        VOCAB_VALUE_COL: [str(key) for key, _ in rows],
                        VOCAB_ID_COL: [int(value) for _, value in rows],
                    }
                ).write_parquet(self.meta_filepath.parent / ref)
                state["vocab"] = ref
            meta_data.append(data)
        with open(self.meta_filepath, 'w') as f:
            json.dump(meta_data, f, indent=4)

    def load_schema(self):
        if not self.meta_filepath.exists():
            print(f"[Warning] 找不到元数据文件 {self.meta_filepath}，无法加载 schema。")
            return

        with open(self.meta_filepath, 'r', encoding='utf-8') as f:
            meta_data_list = json.load(f)

        for data in meta_data_list:
            for state in (data, data.get("element_setting_state")):
                if not isinstance(state, dict) or not isinstance(state.get("vocab"), str):
                    continue
                vocab_df = pl.read_parquet(
                    self.meta_filepath.parent / state["vocab"],
                    columns=[VOCAB_VALUE_COL, VOCAB_ID_COL],
                )
                state["vocab"] = {
                    str(row[VOCAB_VALUE_COL]): int(row[VOCAB_ID_COL])
                    for row in vocab_df.iter_rows(named=True)
                }

        meta_dict = {item["field_name"]: item for item in meta_data_list}
        for setting in self.settings:
            if setting.field_name in meta_dict:
                saved_state = meta_dict[setting.field_name]
                saved_type = saved_state.get("type")
                if saved_type not in setting.compatible_type_names:
                    raise ValueError(
                        f"特征 {setting.field_name} 的类型不匹配: "
                        f"期望 {sorted(setting.compatible_type_names)}, JSON 中为 {saved_type}"
                    )

                setting.load_state(saved_state)
                print(f"[-] 成功恢复特征状态: {setting.field_name}")
            else:
                print(f"[Warning] 特征 {setting.field_name} 在 JSON 元数据中未找到，将保持默认/未拟合状态。")

    def source2emb_dim(self, *sources: FeatureSource):
        emb_dim = 0
        for setting in self.settings:
            if setting.source in sources:
                emb_dim += setting.embedding_dim
        return emb_dim

    def get_setting(self, field_name):
        for setting in self.settings:
            if setting.field_name == field_name:
                return setting
        return None

    def fields(self):
        fields = []
        for setting in self.settings:
            for field_name in setting.get_output_field_names():
                if field_name not in fields:
                    fields.append(field_name)

        for ctx_field in (self.time_field, *self.label_fields, *self.domain_fields):
            if ctx_field is not None and ctx_field not in fields:
                fields.append(ctx_field)
        return fields


# ==========================================
# 3. 使用示例 (假设使用 KuaiRand)
# ==========================================
if __name__ == "__main__":
    # 假设特征注册
    user_id_setting = SparseEmbSetting("user_id", FeatureSource.USER)

    # 高级序列特征映射：无论原数据是 "1,5,6" 还是 ["1", "5", "6"]，瞬间处理完毕
    tags = MultiSparseSetting(
        field_name="tag",
        source=FeatureSource.ITEM,
        is_string_format=True,
        separator=","
    )

    manager = SchemaManager([user_id_setting, tags])

    from rosen.datasets.kuairand import KuaiRand
    item_lf = pl.scan_csv(KuaiRand.VIDEO_FEATURES)
    inter_lf = pl.scan_csv(KuaiRand.STD_LOG_FORMER_DATA)
    user_lf = pl.scan_csv(KuaiRand.USER_FEATURES)

    whole_lf = inter_lf.join(item_lf, on="video_id", how="left")
    whole_lf = whole_lf.join(user_lf, on="user_id", how="left")
    # 只需要喂入路径，不用 pd.read_csv。即使 kuairand_log.csv 有 50GB 也能在普通笔记本上跑完
    # manager.prepare_data("kuairand_log.csv", "kuairand_processed.parquet")
    manager.prepare_data(whole_lf, "kuairand_processed.parquet", redo=True)
    print("架构编译成功，可供调用。")

    import pandas as pd
    df = pd.read_parquet("kuairand_processed.parquet")
    print(df.head(3).to_string())
</file>

<file path="emb/split.py">
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union

import polars as pl


@dataclass(frozen=True)
class SplitContext:
    uid_field: Optional[str]
    iid_field: Optional[str]
    time_field: Optional[str]
    checkpoint_fn: Callable[[pl.LazyFrame], pl.LazyFrame]


@dataclass(frozen=True)
class LooConfig:
    k_core: int = 3


@dataclass(frozen=True)
class SequentialRatioConfig:
    train_ratio: float = 0.8
    valid_ratio: float = 0.1


@dataclass(frozen=True)
class TimeSplitConfig:
    valid_start: Any
    test_start: Any


@dataclass(frozen=True)
class RandomRatioConfig:
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    group_by: Optional[str] = None


SplitConfig = Union[LooConfig, SequentialRatioConfig, TimeSplitConfig, RandomRatioConfig]
ConfigT = TypeVar("ConfigT")


class BaseSplitStrategy(Generic[ConfigT], ABC):
    config_type: Type[ConfigT]

    def __init__(self, context: SplitContext, config: ConfigT):
        self.ctx = context
        self.config = config

    @abstractmethod
    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        pass


class LooSplitStrategy(BaseSplitStrategy[LooConfig]):
    config_type = LooConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool):
        print(f"[*] 正在执行 LOO 切分，K-core={self.config.k_core}...")
        if self.ctx.time_field is None:
            raise RuntimeError("执行 loo 切分必须在 SplitContext 中指定 time_field！")

        processed_lf = lf.with_columns(
            [
                pl.count(self.ctx.iid_field).over(self.ctx.uid_field).alias("user_seq_len"),
                pl.col(self.ctx.time_field)
                .rank(descending=True, method="ordinal")
                .over(self.ctx.uid_field)
                .alias("reverse_rank"),
            ]
        ).filter(pl.col("user_seq_len") >= self.config.k_core)

        checkpoint_lf = self.ctx.checkpoint_fn(processed_lf)
        drop_cols = ["user_seq_len", "reverse_rank"]
        train_lf = checkpoint_lf.filter(pl.col("reverse_rank") >= 3).select(pl.all().exclude(drop_cols))
        valid_lf = checkpoint_lf.filter(pl.col("reverse_rank") == 2).select(pl.all().exclude(drop_cols))
        test_lf = checkpoint_lf.filter(pl.col("reverse_rank") == 1).select(pl.all().exclude(drop_cols))
        return train_lf, valid_lf, test_lf


class SequentialRatioStrategy(BaseSplitStrategy[SequentialRatioConfig]):
    config_type = SequentialRatioConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        if self.ctx.time_field is None:
            raise RuntimeError("执行 sequential_ratio 切分必须在 SplitContext 中指定 time_field！")

        train_ratio = self.config.train_ratio
        valid_ratio = self.config.valid_ratio
        test_ratio = 1 - train_ratio - valid_ratio
        print(
            f"[*] 正在执行顺序时序比例 (Sequential Ratio) 切分，比例: "
            f"{train_ratio}:{valid_ratio}:{test_ratio:.2f}..."
        )

        time_col = self.ctx.time_field
        time_count_lf = (
            lf.group_by(time_col)
            .len()
            .sort(time_col)
            .with_columns(pl.col("len").cum_sum().alias("cum_rows"))
        )
        total_rows = time_count_lf.select(pl.col("len").sum().alias("total_rows")).collect(engine="streaming").item()
        train_cut = int(total_rows * train_ratio)
        valid_cut = train_cut + int(total_rows * valid_ratio)

        boundary_df = (
            time_count_lf.select(
                [
                    pl.col(time_col).filter(pl.col("cum_rows") >= train_cut).first().alias("valid_start"),
                    pl.col(time_col).filter(pl.col("cum_rows") >= valid_cut).first().alias("test_start"),
                ]
            ).collect(engine="streaming")
        )
        valid_start = boundary_df["valid_start"][0]
        test_start = boundary_df["test_start"][0]

        train_lf = lf.filter(pl.col(time_col) < valid_start)
        valid_lf = lf.filter((pl.col(time_col) >= valid_start) & (pl.col(time_col) < test_start))
        test_lf = lf.filter(pl.col(time_col) >= test_start)
        return train_lf, valid_lf, test_lf


class TimeSplitStrategy(BaseSplitStrategy[TimeSplitConfig]):
    config_type = TimeSplitConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        if self.ctx.time_field is None:
            raise RuntimeError("执行 time 切分必须在 SplitContext 中指定 time_field！")

        print(
            f"[*] 正在执行绝对时间阈值 (Time) 切分，"
            f"train_end={self.config.valid_start}, valid_end={self.config.test_start}..."
        )
        time_expr = pl.col(self.ctx.time_field)
        train_lf = lf.filter(time_expr < self.config.valid_start)
        valid_lf = lf.filter((time_expr >= self.config.valid_start) & (time_expr < self.config.test_start))
        test_lf = lf.filter(time_expr >= self.config.test_start)
        return train_lf, valid_lf, test_lf


class RandomRatioStrategy(BaseSplitStrategy[RandomRatioConfig]):
    config_type = RandomRatioConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        train_ratio = self.config.train_ratio
        valid_ratio = self.config.valid_ratio
        group_by = self.config.group_by

        group_msg = f"按 '{group_by}' 分组" if group_by else "全局"
        print(
            f"[*] 正在执行{group_msg}随机比例 (Random Ratio) 切分，"
            f"比例: {train_ratio}:{valid_ratio}:{1 - train_ratio - valid_ratio:.2f}..."
        )

        len_expr = pl.len().over(group_by) if group_by else pl.len()
        row_idx_expr = pl.int_range(0, pl.len()).shuffle(seed=42)
        if group_by:
            row_idx_expr = row_idx_expr.over(group_by)

        train_idx_expr = (len_expr * train_ratio).cast(pl.Int64)
        valid_idx_expr = train_idx_expr + (len_expr * valid_ratio).cast(pl.Int64)
        processed_lf = lf.with_columns(
            row_idx_expr.alias("row_idx"),
            train_idx_expr.alias("train_idx"),
            valid_idx_expr.alias("valid_idx"),
        )

        checkpoint_lf = self.ctx.checkpoint_fn(processed_lf)
        temp_cols = ["row_idx", "train_idx", "valid_idx"]
        train_lf = checkpoint_lf.filter(pl.col("row_idx") < pl.col("train_idx")).drop(temp_cols)
        valid_lf = checkpoint_lf.filter(
            (pl.col("row_idx") >= pl.col("train_idx")) & (pl.col("row_idx") < pl.col("valid_idx"))
        ).drop(temp_cols)
        test_lf = checkpoint_lf.filter(pl.col("row_idx") >= pl.col("valid_idx")).drop(temp_cols)
        return train_lf, valid_lf, test_lf


SPLIT_STRATEGIES: Dict[str, Type[BaseSplitStrategy]] = {
    "loo": LooSplitStrategy,
    "time": TimeSplitStrategy,
    "sequential_ratio": SequentialRatioStrategy,
    "random_ratio": RandomRatioStrategy,
}

SPLIT_CONFIG_TYPES: Dict[str, Type[Any]] = {
    "loo": LooConfig,
    "time": TimeSplitConfig,
    "sequential_ratio": SequentialRatioConfig,
    "random_ratio": RandomRatioConfig,
}


def build_split_config(strategy: str, **kwargs) -> SplitConfig:
    config_cls = SPLIT_CONFIG_TYPES.get(strategy)
    if config_cls is None:
        raise ValueError(f"未知的切分策略: {strategy}")

    allowed_fields = {item.name for item in fields(config_cls)}
    unknown_fields = sorted(set(kwargs) - allowed_fields)
    if unknown_fields:
        raise TypeError(
            f"切分策略 '{strategy}' 不接受参数: {unknown_fields}。"
            f"允许参数为: {sorted(allowed_fields)}"
        )

    return config_cls(**kwargs)


def create_split_strategy(
    strategy: str,
    context: SplitContext,
    config: Optional[SplitConfig] = None,
    **kwargs,
) -> BaseSplitStrategy:
    strategy_cls = SPLIT_STRATEGIES.get(strategy)
    if strategy_cls is None:
        raise ValueError(f"未知的切分策略: {strategy}")

    if config is None:
        config = build_split_config(strategy, **kwargs)
    elif kwargs:
        raise TypeError("请在 split_dataset 中二选一：传入显式 config，或传入兼容 kwargs。")

    if not isinstance(config, strategy_cls.config_type):
        raise TypeError(
            f"切分策略 '{strategy}' 期望配置类型为 {strategy_cls.config_type.__name__}，"
            f"实际得到 {type(config).__name__}"
        )

    return strategy_cls(context=context, config=config)
</file>

<file path="evaluate/__init__.py">
"""Evaluation metrics and evaluators."""
</file>

<file path="evaluate/evaluator.py">
import json
from typing import Iterable, Union, Literal

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import torch


class BaseEvaluator:
    VALID_METRICS = ["auc", "logloss", "gauc", "hr", "ndcg", "recall"]
    def __init__(self, metrics):
        # 统一转小写并过滤非法指标
        self.metrics = [m.lower() for m in metrics]
        for m in self.metrics:
            base_m = m.split('@')[0] if '@' in m else m
            assert base_m in self.VALID_METRICS, f"不支持的指标: {m}"
        self.clear()

    def collect(self, batch_users, batch_targets, batch_preds):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


import numpy as np
from rosen.evaluate.metrics import METRIC_REGISTRY

class PointWiseEvaluator(BaseEvaluator):
    def __init__(self, metrics):
        super().__init__(metrics)

    def collect(self, batch_users, batch_targets, batch_preds):
        # 粗暴地把当前 Batch 的数据怼进大列表
        self.all_users.extend(batch_users.tolist())
        self.all_targets.extend(batch_targets.tolist())
        self.all_preds.extend(batch_preds.tolist())

    def summary(self):
        results = {}
        if not self.all_targets:
            return results

        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)

        # 遍历指标，通过注册表调用静态计算方法
        for m in self.metrics:
            calc_class = METRIC_REGISTRY[m]
            results[m] = calc_class.calculate(y_true=y_true, y_pred=y_pred, users=self.all_users)

        return results

    def clear(self):
        self.all_users = []
        self.all_targets = []
        self.all_preds = []


class TopKEvaluator(BaseEvaluator):
    def __init__(self, metrics, history_dict):
        self.history_dict = history_dict
        super().__init__(metrics)

        # 解析 K 值 (例如将 'hit@5', 'ndcg@10' 解析出 k_list = [5, 10])
        self.k_list = sorted(list(set([int(m.split('@')[1]) for m in self.metrics])))
        self.max_k = max(self.k_list) if self.k_list else 0

    def collect(self, batch_users, batch_targets, batch_scores):
        device = batch_scores.device
        batch_size = batch_users.size(0)
        batch_indices = torch.arange(batch_size, device=device)
        target_original_scores = batch_scores[batch_indices, batch_targets].clone()
        # 1. 瞬间 Mask 历史物品
        rows, cols = [], []
        for i, uid in enumerate(batch_users.tolist()):
            hist = self.history_dict.get(uid, [])
            rows.extend([i] * len(hist))
            cols.extend(hist)
        if rows:
            batch_scores[rows, cols] = -float('inf')
        batch_scores[batch_indices, batch_targets] = target_original_scores # 将目标物品的分数还原
        # 2. 取 Top-Max_K 并生成命中矩阵 [Batch_size, max_k]
        _, topk_indices = torch.topk(batch_scores, self.max_k, dim=-1)
        hits = (topk_indices == batch_targets.unsqueeze(1)).float()

        # 3. 现结现清：调用指标类算总和并累加
        for k in self.k_list:
            hits_k = hits[:, :k]
            for m in self.metrics:
                metric_name, metric_k = m.split('@')
                if int(metric_k) == k:
                    calc_class = METRIC_REGISTRY[metric_name]
                    # 调用静态方法获取当前 Batch 的总得分
                    batch_sum = calc_class.calculate(hits_k=hits_k, k=k)
                    self.metric_sums[m] += batch_sum

        self.total_users += batch_size

    def summary(self):
        results = {}
        if self.total_users == 0:
            return results
        for m in self.metrics:
            results[m] = self.metric_sums[m] / self.total_users
        return results

    def clear(self):
        self.metric_sums = {m: 0.0 for m in self.metrics}
        self.total_users = 0

import polars as pl
ValidMetrics = Literal["auc", "logloss", "gauc", "HR@*", "NDCG@*"]
class Evaluator:
    def __init__(self, *metrics: ValidMetrics, history_dict=None):
        """
        metrics: list of str, 例如 ['auc', 'logloss', 'hit@10', 'ndcg@20']
        history_dict: dict, TopK 评估必须传入的静态历史黑名单
        """
        self.metrics = [m.lower() for m in metrics]
        # 1. 自动解析和分流指标
        self.point_metrics = [m for m in self.metrics if '@' not in m]
        self.topk_metrics = [m for m in self.metrics if '@' in m]

        # 2. 实例化底层 Evaluator
        self.point_evaluator = None
        self.topk_evaluator = None

        if self.point_metrics:
            self.point_evaluator = PointWiseEvaluator(self.point_metrics)

        if self.topk_metrics:
            assert history_dict is not None, "错误：计算 TopK 指标必须传入 history_dict 以进行历史剔除！"
            self.topk_evaluator = TopKEvaluator(self.topk_metrics, history_dict)

    def collect(self, batch_users, batch_targets, batch_preds_1d=None, batch_scores_2d=None):
        if self.point_evaluator:
            self.collect_pointwise(batch_users, batch_targets, batch_preds_1d)
        if self.topk_evaluator:
            self.collect_topk(batch_users, batch_targets, batch_scores_2d)

    def collect_pointwise(self, batch_users, batch_targets, batch_preds_1d):
        """
        收集 CTR 等 Point-wise 任务的预测结果。
        - batch_preds_1d: shape [Batch_Size]，通常为 0~1 的概率值。
        """
        if self.point_evaluator is None:
            raise RuntimeError("未初始化 PointWise 评估器。请检查是否传入了 auc, logloss 等指标。")
        self.point_evaluator.collect(batch_users, batch_targets, batch_preds_1d)

    def collect_topk(self, batch_users, batch_targets, batch_scores_2d):
        """
        收集序列推荐/召回等 Top-K 任务的全排序得分。
        - batch_scores_2d: shape [Batch_Size, Num_Items]，通常为 Raw Logits。
        """
        if self.topk_evaluator is None:
            raise RuntimeError("未初始化 TopK 评估器。请检查是否传入了 hit@k, ndcg@k 等指标。")
        self.topk_evaluator.collect(batch_users, batch_targets, batch_scores_2d)

    def summary(self):
        """
        Epoch 结束时调用，合并所有底层 Evaluator 的成绩单并保存
        epoch/step: 可选，用于记录当前是第几轮或第几步
        """
        final_results = {}

        if self.point_evaluator:
            final_results.update(self.point_evaluator.summary())

        if self.topk_evaluator:
            final_results.update(self.topk_evaluator.summary())

        return final_results

    def clear(self):
        """
        清空缓存和累加器，准备下一个 Epoch
        """
        if self.point_evaluator:
            self.point_evaluator.clear()

        if self.topk_evaluator:
            self.topk_evaluator.clear()

class EvaluatorDecorator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
    def collect_pointwise(self, batch_users, batch_targets, batch_preds_1d):
        self.evaluator.collect_pointwise(batch_users, batch_targets, batch_preds_1d)
    def collect_topk(self, batch_users, batch_targets, batch_scores_2d):
        self.evaluator.collect_topk(batch_users, batch_targets, batch_scores_2d)
    def collect(self, batch_users, batch_targets, batch_preds_1d=None, batch_scores_2d=None):
        self.evaluator.collect(batch_users, batch_targets, batch_preds_1d, batch_scores_2d)
    def summary(self, epoch=None, step=None):
        return self.evaluator.summary()
    def clear(self):
        self.evaluator.clear()

import os
from datetime import datetime
class LogDecorator(EvaluatorDecorator):
    def __init__(self, evaluator, save_path=None, title=None):
        """
        evaluator: 原始的 Evaluator 实例
        save_path: 日志保存路径
        """
        super().__init__(evaluator)
        self.save_path = save_path
        self.title = title
        self._is_first_summary = True

        if self.save_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)

    def summary(self, epoch=None, step=None):
        """
        拦截原有的 summary 方法，在拿到结果后增加纯文本写入的逻辑
        同时扩展了接受 epoch 和 step 的能力
        """
        # 1. 调用原 Evaluator 的 summary 获取干净的指标字典
        final_results = self.evaluator.summary()

        # 2. 拦截并执行纯文本日志写入逻辑
        if self.save_path:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.save_path, 'a', encoding='utf-8') as f:
                # 首次写入打分割线
                if self._is_first_summary:
                    title = self.title or ""
                    f.write(f"\n========== {title}实验进程开始 | 首次评估时间: {current_time} ==========\n")
                    self._is_first_summary = False

                # 拼接易读字符串
                log_parts = [f"[{current_time}]"]
                if epoch is not None:
                    log_parts.append(f"Epoch: {epoch:02d}")
                if step is not None:
                    log_parts.append(f"Step: {step}")

                # 遍历原始指标写入
                for metric_name, value in final_results.items():
                    if isinstance(value, float):
                        log_parts.append(f"{metric_name}: {value:.4f}")
                    else:
                        log_parts.append(f"{metric_name}: {value}")

                f.write(" | ".join(log_parts) + '\n')

        return final_results

if __name__ == '__main__':
    ...
</file>

<file path="evaluate/manager.py">
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import polars as pl


class EvaluatorFilter(ABC):
    @abstractmethod
    def __call__(self, inter):
        raise NotImplementedError


class AllowAllFilter(EvaluatorFilter):
    def __call__(self, inter):
        return True


class DomainFilter(EvaluatorFilter):
    def __init__(self, field_name: str, domain_id: int):
        self.field_name = field_name
        self.domain_id = domain_id

    def __call__(self, inter):
        return inter[self.field_name] == self.domain_id

class EvaluatorManager:
    def __init__(self, log_path=None, title="untitled"):
        self.registry = {}
        self.log_path = log_path
        self.title = title
        self._is_first_summary = True

        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def register(self, name, evaluator, filter_fn=None):
        """
        注册评估器。
        filter_fn: 接收 batch_inter，返回布尔掩码
        """
        if filter_fn is None:
            filter_fn = AllowAllFilter()

        self.registry[name] = {
            "evaluator": evaluator,
            "filter_fn": filter_fn,
        }

    def collect(self, batch_users, batch_targets, batch_inter, batch_preds_1d=None, batch_scores_2d=None, ):
        """
        前4个参数完全对齐你的基础 Evaluator。
        唯一多出的 batch_inter 仅作为计算 mask 的上下文，不参与任何解包。
        """
        for name, config in self.registry.items():
            evaluator = config["evaluator"]
            filter_fn = config["filter_fn"]

            # 1. 仅把 inter 交给 filter_fn 算掩码，绝不从 inter 里取真实数据
            mask = filter_fn(batch_inter)

            # 2. 全量不过滤的情况
            if isinstance(mask, bool) and mask is True:
                evaluator.collect(batch_users, batch_targets, batch_preds_1d, batch_scores_2d)
                continue

            # 提前拦截空掩码
            if isinstance(mask, bool) and mask is False:
                continue
            if hasattr(mask, 'any') and not mask.any():
                continue

            # 3. 对传入的张量进行原生的布尔切片
            e_users = batch_users[mask]
            e_targets = batch_targets[mask]
            e_preds_1d = batch_preds_1d[mask] if batch_preds_1d is not None else None
            e_scores_2d = batch_scores_2d[mask] if batch_scores_2d is not None else None

            # 4. 完美对齐调用底层的 Evaluator
            evaluator.collect(e_users, e_targets, e_preds_1d, e_scores_2d)

    def _safe_summary(self, evaluator, epoch=None, step=None):
        try:
            return evaluator.summary(epoch=epoch, step=step)
        except TypeError:
            return evaluator.summary()

    def summary(self, epoch=None, step=None):
        """
        返回每个 evaluator 的完整指标：
        {
            "overall": {"auc": 0.73, "logloss": 0.41},
            "domain_a": {"gauc": 0.69},
        }
        """
        summary_by_name = {}
        log_metrics = {}
        for name, config in self.registry.items():
            res = self._safe_summary(config["evaluator"], epoch=epoch, step=step)
            summary_by_name[name] = dict(res)
            for metric_name, metric_value in res.items():
                log_metrics[f"{name}.{metric_name}"] = metric_value

        if self.log_path:
            self._write_polars_log(log_metrics, epoch, step)

        return summary_by_name

    def _write_polars_log(self, metrics_dict, epoch, step):
        """内部方法：专门负责将指标转成 Polars 字符串并追加到文件"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将字典转为纵向结构，方便阅读
        df = pl.DataFrame({
            "Metric": list(metrics_dict.keys()),
            "Value": list(metrics_dict.values())
        })

        # 临时配置 Polars 渲染参数：
        # tbl_rows=100：防止指标太多中间被折叠成 "..."
        # tbl_hide_dataframe_shape=True：隐藏底部的 (shape: 10, 2) 这种无关信息
        with pl.Config(tbl_rows=100, tbl_hide_dataframe_shape=True):
            table_str = str(df)

        # 写入文件
        with open(self.log_path, 'a', encoding='utf-8') as f:
            # 首次记录打印实验 Header
            if self._is_first_summary:
                f.write(f"\n========== 🚀 [{self.title}] 开始记录 | {current_time} ==========\n")
                self._is_first_summary = False

            # 打印当前 Step/Epoch 的信息头
            header_parts = [f"\n[{current_time}]"]
            if epoch is not None: header_parts.append(f"Epoch: {epoch:02d}")
            if step is not None: header_parts.append(f"Step: {step}")

            f.write(" | ".join(header_parts) + "\n")

            # 写入 Polars 格式化好的表格
            f.write(table_str + "\n")

    def clear(self):
        for config in self.registry.values():
            config["evaluator"].clear()
</file>

<file path="evaluate/metrics.py">
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

# ==================== 指标基类 ====================
class BaseMetric:
    @classmethod
    def calculate(cls, *args, **kwargs):
        raise NotImplementedError

# ==================== PointWise 指标簇 ====================
class AUC(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.0

class LogLoss(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        return log_loss(y_true, y_pred)

class GAUC(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, users, **kwargs):
        user_data = {}
        for u, t, p in zip(users, y_true, y_pred):
            user_data.setdefault(u, {'t': [], 'p': []})
            user_data[u]['t'].append(t)
            user_data[u]['p'].append(p)

        total_weight, weighted_auc = 0.0, 0.0
        for data in user_data.values():
            t_arr, p_arr = np.array(data['t']), np.array(data['p'])
            weight = len(t_arr)
            if len(set(t_arr)) > 1:
                try:
                    weighted_auc += roc_auc_score(t_arr, p_arr) * weight
                    total_weight += weight
                except ValueError:
                    continue
        return weighted_auc / total_weight if total_weight > 0 else 0.0

# ==================== TopK 指标簇 ====================
class HR(BaseMetric):
    @classmethod
    def calculate(cls, hits_k, **kwargs):
        # hits_k shape: [Batch_size, K]
        return hits_k.sum().item()

class NDCG(BaseMetric):
    @classmethod
    def calculate(cls, hits_k, k, **kwargs):
        # 针对 Leave-One-Out 的极速 NDCG 计算
        device = hits_k.device
        ranks = torch.arange(1, k + 1, device=device).float()
        dcg_weights = 1.0 / torch.log2(ranks + 1)
        return (hits_k * dcg_weights).sum().item()

# 注册表：将字符串映射到对应的计算类
METRIC_REGISTRY = {
    "auc": AUC, "logloss": LogLoss, "gauc": GAUC,
    "hr": HR, "ndcg": NDCG
}
</file>

<file path="experiment/__init__.py">
from pathlib import Path
from .config import ROOT_DIR, WORKSPACE

def ignore_future_warning():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

def change_root_workdir():
    import os
    print("切换到: ", ROOT_DIR)
    os.chdir(ROOT_DIR)

def set_all():
    ignore_future_warning()
    change_root_workdir()

if __name__ == '__main__':
    print(preset_workdir())
</file>

<file path="experiment/config.py">
from pathlib import Path

ROOT_DIR = Path(__file__).parents[3]
WORKSPACE = ROOT_DIR / "workspace"
</file>

<file path="experiment/engine.py">
import itertools
import csv
import json
import subprocess
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
"""
原理：只需要指定目标python文件路径就行，只要你使用了ParamManager这套，命令行传参都是第一优先级
"""

class GridSearchEngine:
    def __init__(self, script_path: str,):
        """
        :param script_path: 运行python的cwd下面找script_path相对路径
        """
        self.script_path = script_path
        self.base_python = sys.executable

    def _default_run_name(self) -> str:
        script_name = os.path.splitext(os.path.basename(self.script_path))[0]
        return script_name or "grid_search"

    def _make_run_dir(self, log_dir: str, run_name: str = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_label = run_name or self._default_run_name()
        run_dir = os.path.join(log_dir, f"{timestamp}_{run_label}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _log(self, engine_log_path: str, message: str):
        print(message)
        with open(engine_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

    def make_grid(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """将参数空间字典展开为笛卡尔积列表"""
        keys = param_space.keys()
        values = param_space.values()
        combinations = itertools.product(*values)
        return [dict(zip(keys, combo)) for combo in combinations]

    def build_experiment_name(self, params: Dict[str, Any]) -> str:
        """根据当前参数组合动态生成实验名"""
        if "_name" in params:
            name = str(params["_name"])
            if "seed" in params:
                name = f"{name}_seed{params['seed']}"
            return name

        parts = []
        for k, v in params.items():
            if k == "experiment_name" or k.startswith("_"):
                continue
            # 简化浮点数等长格式
            val_str = str(v).replace(".", "_") if isinstance(v, float) else str(v)
            parts.append(f"{k}{val_str}")
        if parts:
            return "-".join(parts)
        if "experiment_name" in params:
            return str(params["experiment_name"])
        return "baseline"

    def worker(self, params: Dict[str, Any], gpu_id: int, log_dir: str):
        """单个实验的执行工作流"""
        # 1. 构建实验名并注入参数
        run_params = dict(params)
        exp_name = self.build_experiment_name(run_params)
        if "experiment_name" not in run_params:
            run_params["experiment_name"] = exp_name
        cmd = [self.base_python, self.script_path]
        for k, v in run_params.items():
            if k.startswith("_"):
                continue
            cmd.extend([f"--{k}", str(v)])
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{exp_name}.log")
        start_time = time.time()
        return_code = None
        error_message = ""

        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                process = subprocess.run(
                    cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True
                )
            return_code = int(process.returncode)
            status = "success" if process.returncode == 0 else "failed"
        except Exception as e:
            status = "error"
            error_message = str(e)

        end_time = time.time()
        return {
            "exp_name": exp_name,
            "status": status,
            "gpu_id": gpu_id,
            "log_path": log_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "return_code": return_code,
            "params": run_params,
            "error_message": error_message,
        }

    def run(
            self,
            param_space: Dict[str, List[Any]],
            available_gpus: List[int],
            log_dir: str = "./logs",
            run_name: str = None,
    ):
        tasks = self.make_grid(param_space)
        run_dir = self._make_run_dir(log_dir=log_dir, run_name=run_name)
        experiment_log_dir = os.path.join(run_dir, "experiments")
        os.makedirs(experiment_log_dir, exist_ok=True)
        engine_log_path = os.path.join(run_dir, "engine.log")
        summary_path = os.path.join(run_dir, "summary.csv")

        self._log(engine_log_path, f"📁 RunDir: {run_dir}")
        self._log(engine_log_path, f"📊 发现 {len(tasks)} 个实验组合，可用 GPU: {available_gpus}")
        self._log(engine_log_path, f"🧩 ParamSpace: {json.dumps(param_space, ensure_ascii=False, sort_keys=True)}")
        max_workers = len(available_gpus) if available_gpus else 1

        with open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(
                summary_file,
                fieldnames=[
                    "exp_name",
                    "status",
                    "gpu_id",
                    "return_code",
                    "start_time",
                    "end_time",
                    "duration",
                    "log_path",
                    "params",
                    "error_message",
                ],
            )
            writer.writeheader()

            future_to_meta = {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i, task_params in enumerate(tasks):
                    # 轮询分配 GPU
                    gpu_id = available_gpus[i % len(available_gpus)] if available_gpus else None
                    exp_name = self.build_experiment_name(task_params)
                    log_path = os.path.join(experiment_log_dir, f"{exp_name}.log")
                    self._log(
                        engine_log_path,
                        f"🚀 [Start] GPU:{gpu_id} | Exp:{exp_name} | Log:{log_path}",
                    )
                    future = executor.submit(self.worker, task_params, gpu_id, experiment_log_dir)
                    future_to_meta[future] = {"exp_name": exp_name, "gpu_id": gpu_id, "log_path": log_path}

                for future in as_completed(future_to_meta):
                    meta = future_to_meta[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {
                            "exp_name": meta["exp_name"],
                            "status": "error",
                            "gpu_id": meta["gpu_id"],
                            "return_code": None,
                            "start_time": None,
                            "end_time": None,
                            "duration": None,
                            "log_path": meta["log_path"],
                            "params": {},
                            "error_message": str(e),
                        }

                    writer.writerow(
                        {
                            "exp_name": result["exp_name"],
                            "status": result["status"],
                            "gpu_id": result["gpu_id"],
                            "return_code": result["return_code"],
                            "start_time": result["start_time"],
                            "end_time": result["end_time"],
                            "duration": result["duration"],
                            "log_path": result["log_path"],
                            "params": json.dumps(result["params"], ensure_ascii=False, sort_keys=True),
                            "error_message": result["error_message"],
                        }
                    )
                    summary_file.flush()

                    status_icon = {"success": "✅ Success", "failed": "❌ Failed", "error": "💥 Error"}.get(
                        result["status"], result["status"]
                    )
                    duration = result["duration"]
                    duration_text = f"{duration:.1f}s" if duration is not None else "unknown"
                    self._log(
                        engine_log_path,
                        f"{status_icon} [{duration_text}] | Exp:{result['exp_name']} | Log:{result['log_path']}",
                    )

        self._log(engine_log_path, f"🧾 Summary: {summary_path}")


if __name__ == '__main__':
    # 定义搜索空间 (列表中包含你需要尝试的各种值)
    search_space = {
        "model": ["star", "ple"],
        "seed": [2024, 2025, 2026],
        "device": ["cuda"]
    }

    # 假设你有一台服务器，有两张卡可以跑实验
    gpus_to_use = [0]
    engine = GridSearchEngine(script_path="run.py")
    engine.run(param_space=search_space, available_gpus=gpus_to_use)
</file>

<file path="experiment/param.py">
import argparse
import ast
import dataclasses
from dataclasses import dataclass, fields, field, asdict
from typing import get_args, get_origin, Literal, get_type_hints
from  .config import WORKSPACE, ROOT_DIR
@dataclass
class ConfigBase:
    experiment_name: str = "untitled" # 提示这次实验的名字
    dataset_name: str = "unknown" # 影响workdir的设置
    seed: int = 2026
    device: str = "cpu"
    max_epochs: int = 100
    ckpt_dir: str = ""

    extras: dict = field(default_factory=dict)

    @property
    def preset_workspace(self):
        return WORKSPACE

    @property
    def preset_dataset_dir(self):
        return self.preset_workspace / self.dataset_name

    @property
    def preset_output_dir(self):
        return self.preset_workspace / self.dataset_name / "experiments" / self.experiment_name

    def __getattr__(self, name):
        if name in self.extras:
            return self.extras[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self):
        config_dict = asdict(self)
        lines = [f"\n🚀 [Active Config State]: <{self.__class__.__name__}>"]
        lines.append("-" * 50)
        max_key_len = max([len(str(k)) for k in config_dict.keys()] + [10])

        for key, value in config_dict.items():
            if isinstance(value, type):
                display_val = f"<class '{value.__name__}'>"
            elif callable(value):
                display_val = f"<callable '{getattr(value, '__name__', str(value))}'>"
            elif isinstance(value, dict) and key == "extras":
                if not value:
                    display_val = "{}"
                else:
                    display_val = "{" + ", ".join(f"'{k}': {repr(v)}" for k, v in value.items()) + "}"
            else:
                display_val = repr(value)  # 使用 repr 可以让字符串带上引号，更严谨
            lines.append(f"  {key:<{max_key_len}} = {display_val}")
        lines.append("-" * 50)
        return "\n".join(lines)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        annotations = getattr(cls, '__annotations__', {})
        missing_annotations = []
        for key, value in cls.__dict__.items():
            if key.startswith('_') or callable(value) or isinstance(value, (property, classmethod, staticmethod)):
                continue
            if key not in annotations:
                missing_annotations.append(key)
        if missing_annotations:
            missing_str = ", ".join(f"'{f}'" for f in missing_annotations)
            raise TypeError(
                f"🚨 拦截: 数据类 '{cls.__name__}' 存在未加类型注解的属性！\n"
                f"缺失注解的字段共 {len(missing_annotations)} 个: [{missing_str}]\n"
                f"👉 修复建议: 请修改为 '字段名: 类型 = ...'，如无特定类型请使用 'Any'。"
            )

def seed_everything(seed=42):
    import os
    import numpy as np
    import random
    import torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
"""
调用build方法可以从任何地方获取参数
"""


class ParamManager:
    def __init__(self, config_class):
        self.config_class = config_class
        self.resolved_type_hints = get_type_hints(config_class)

        # 1. 基础类型转换注册表：消除对 bool 等基本类型的 if-else
        self.type_parsers = {
            bool: lambda x: str(x).lower() in ('true', '1', 'yes', 'y', 't'),
            list: lambda x: x.split(','),
        }

        # 2. 实例工厂注册表：{ 字段名: { 字符串key: 实例化方法/类 } }
        self.instance_registry = {}

    def register(self, field_name: str, mapping_dict: dict):
        """
        优雅的字符串到实例注册方法
        :param field_name: dataclass 中的字段名
        :param mapping_dict: 字符串参数到类/工厂函数的映射字典
        """
        self.instance_registry[field_name] = mapping_dict
        return self

    def _get_field_type(self, field):
        return self.resolved_type_hints.get(field.name, field.type)

    def _get_parser_action(self, field):
        """决定如何将命令行的字符串解析为目标对象"""
        field_type = self._get_field_type(field)

        # 优先级1：如果字段被注册了实例映射器 (例如 optimizer 字段)
        if field.name in self.instance_registry:
            mapping = self.instance_registry[field.name]
            def instance_factory(x):
                if x not in mapping:
                    raise ValueError(f"Invalid option '{x}' for {field.name}. Available: {list(mapping.keys())}")
                return mapping[x]
            return instance_factory
        if get_origin(field_type) is Literal:
            literal_values = get_args(field_type)
            if not literal_values:
                raise ValueError(f"Literal field '{field.name}' has no candidate values")
            return type(literal_values[0])
        if get_origin(field_type) is list:
            return self.type_parsers[list]
        if field_type in self.type_parsers:
            return self.type_parsers[field_type]
        return field_type

    @staticmethod
    def _parse_extra_value(raw_value: str):
        lowered = raw_value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        try:
            return ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            return raw_value

    def _parse_unknown_cli_args(self, unknown_args):
        extras = {}
        index = 0
        while index < len(unknown_args):
            token = unknown_args[index]
            if not token.startswith("--"):
                raise ValueError(
                    f"Unexpected CLI token '{token}'. Extra parameters must use '--key value' or '--key=value'."
                )

            if "=" in token:
                key, raw_value = token[2:].split("=", 1)
                extras[key.replace("-", "_")] = self._parse_extra_value(raw_value)
                index += 1
                continue

            key = token[2:].replace("-", "_")
            if not key:
                raise ValueError("Encountered an empty extra CLI parameter name.")

            if index + 1 < len(unknown_args) and not unknown_args[index + 1].startswith("--"):
                extras[key] = self._parse_extra_value(unknown_args[index + 1])
                index += 2
            else:
                extras[key] = True
                index += 1
        return extras

    def build(self, **code_kwargs):
        # 1. 【修复点】安全提取基类默认字段（同时兼容 default 和 default_factory）
        config_dict = {}
        for f in fields(self.config_class):
            if f.default is not dataclasses.MISSING:
                config_dict[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                config_dict[f.name] = f.default_factory()  # 这里会安全地生成 {}

        converters = {f.name: self._get_parser_action(f) for f in fields(self.config_class)}

        extras_dict = {}

        # 2. 处理代码直接传入的参数 (code_kwargs)
        for k, v in code_kwargs.items():
            if k in self.instance_registry and isinstance(v, str):
                mapping = self.instance_registry[k]
                if v not in mapping:
                    raise ValueError(f"Invalid option '{v}' for '{k}'. Available: {list(mapping.keys())}")
                processed_v = mapping[v]
            else:
                processed_v = v

            # 分发结果
            if k in config_dict:
                config_dict[k] = processed_v
            elif k != "extras":  # 防止覆盖本身的 extras 字段
                extras_dict[k] = processed_v

        # 3. 处理命令行参数
        parser = argparse.ArgumentParser()
        for f in fields(self.config_class):
            parser.add_argument(f"--{f.name}", type=converters[f.name], default=argparse.SUPPRESS)

        known_field_names = {f.name for f in fields(self.config_class)}
        for reg_k, mapping in self.instance_registry.items():
            if reg_k not in known_field_names:
                def dynamic_factory(x, m=mapping, key=reg_k):
                    if x not in m: raise ValueError(f"Invalid '{key}': {x}")
                    return m[x]

                parser.add_argument(f"--{reg_k}", type=dynamic_factory, default=argparse.SUPPRESS)

        cli_args, unknown_args = parser.parse_known_args()

        # 4. 合并命令行参数解析结果
        for k, v in vars(cli_args).items():
            if k in config_dict:
                config_dict[k] = v
            elif k != "extras":
                extras_dict[k] = v

        extras_dict.update(self._parse_unknown_cli_args(unknown_args))

        # 5. 【现在的 config_dict["extras"] 绝对是一个字典了】
        if "extras" in config_dict:
            config_dict["extras"].update(extras_dict)

        # 6. 实例化
        cfg = self.config_class(**config_dict)
        if hasattr(cfg, "seed"):
            seed_everything(cfg.seed)
        return cfg
    def __str__(self):
        lines = [f"ParamManager managing <{self.config_class.__name__}>"]
        lines.append("-" * 60)
        lines.append(f"{'Parameter':<18} | {'Type':<12} | {'Default'}")
        lines.append("-" * 60)
        for f in fields(self.config_class):
            field_type = self._get_field_type(f)
            type_name = getattr(field_type, '__name__', str(field_type))
            if f.default is dataclasses.MISSING and f.default_factory is not dataclasses.MISSING:
                default_val = "<factory>"
            elif f.default is dataclasses.MISSING:
                default_val = "<required>"
            else:
                default_val = repr(f.default)
            lines.append(f"{f.name:<18} | {type_name:<12} | {default_val}")
        lines.append("-" * 60)
        return "\n".join(lines)

if __name__ == '__main__':
    from rosen.models.msr.sharedbottom import SharedBottomLessModel
    manager = ParamManager(ConfigBase) # 必须是出现在ConfigBase通过doc声明的方式才能被探测到
    manager.register("model", {
        "sharedbottom": SharedBottomLessModel
    })
    cfg = manager.build(model="sharedbottom")
    print(manager)
    print(cfg)
</file>

<file path="experiment/tracker.py">
import os
from pathlib import Path

import torch
import numpy as np


class TrainingTracker:
    def __init__(self, workdir):
        self.save_dir = Path(workdir) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.current_epoch = 0

        self.metrics_history = {}  # 例如: {'loss': [], 'auc': []}
        self.best_metric_value = -float('inf')  # 假设指标越大越好，如果是 loss 则改为 float('inf')

        self.saved_vectors = {}

    def step(self):
        """推进全局步数"""
        self.global_step += 1

    def log_metrics(self, metrics_dict):
        """
        通用指标记录接口
        tracker.log_metrics({'train_loss': 0.5, 'lr': 0.001})
        """
        for k, v in metrics_dict.items():
            if k not in self.metrics_history:
                self.metrics_history[k] = []
            self.metrics_history[k].append((self.global_step, v))

    def save_checkpoint(self, model, optimizer=None, is_best=False, metric_val=None):
        """
        保存断点（包含模型、优化器、步数）
        """
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metric_val is not None:
            checkpoint['metric'] = metric_val

        # 1. 保存常规 checkpoint (可以每 N 步覆盖保存一次，或者带上 step 后缀)
        save_path = os.path.join(self.save_dir, f"ckpt_step_{self.global_step}.pth")
        torch.save(checkpoint, save_path)

        # 2. 如果是当前最佳模型，单独存一份
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"🌟 [Step {self.global_step}] 发现并保存了新的最佳模型！")

    def load_checkpoint(self, model, optimizer=None, ckpt_path=None):
        """
        断点续训/推理加载接口
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

        print(f"正在加载 Checkpoint: {ckpt_path}")
        # 注意：如果是多卡训练后在单卡加载，可能需要 map_location='cpu'
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

        # 恢复模型权重
        model.load_state_dict(checkpoint['model_state_dict'])

        # 恢复优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复全局状态
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)

        print(f"✅ 加载成功！恢复至 Epoch {self.current_epoch}, Global Step {self.global_step}")
        return model, optimizer

    # --- 向量存取相关的接口保持不变 ---
    def save_vector(self, name, vector_tensor):
        self.saved_vectors[name] = vector_tensor.detach().cpu().numpy()

    def export_vectors(self):
        if not self.saved_vectors:
            return
        file_path = os.path.join(self.save_dir, f"vectors_{self.global_step}.npz")
        np.savez(file_path, **self.saved_vectors)
        self.saved_vectors.clear()
</file>

<file path="utils/observatory/audit/__init__.py">
"""Post-training audits built on ObservationPort traces and interventions.

Import concrete workflows, metrics and reporters from their defining modules so
lightweight trainer integrations do not eagerly load the full audit stack.
"""
</file>

<file path="utils/observatory/audit/cut.py">
from __future__ import annotations

from typing import Optional

import torch


def flatten_metric_dict(metrics: Optional[dict]) -> dict[str, float]:
    if not metrics:
        return {}
    flat = {}
    for evaluator_name, values in metrics.items():
        if not isinstance(values, dict):
            continue
        for metric_name, metric_value in values.items():
            try:
                flat[f"{evaluator_name}.{metric_name}"] = float(metric_value)
            except (TypeError, ValueError):
                continue
    return flat


def metric_change_name(metric_name: str) -> str:
    lowered = metric_name.lower()
    if lowered.endswith(".loss") or lowered.endswith(".logloss") or lowered.endswith(".mse"):
        return f"{metric_name}_delta"
    return f"{metric_name}_drop"


def metric_changes(baseline: Optional[dict], candidate: Optional[dict]) -> dict[str, float]:
    base = flatten_metric_dict(baseline)
    current = flatten_metric_dict(candidate)
    changes = {}
    for name, base_value in base.items():
        if name not in current:
            continue
        current_value = current[name]
        change_name = metric_change_name(name)
        if change_name.endswith("_delta"):
            changes[change_name] = current_value - base_value
        else:
            changes[change_name] = base_value - current_value
    return changes


def accumulate_watch_stats(storage: dict, watch_trace: dict) -> None:
    for name, value in watch_trace.items():
        if not isinstance(value, torch.Tensor):
            continue
        flat = value.detach().float().reshape(value.shape[0], -1).cpu()
        row = storage.setdefault(name, {
            "batches": 0,
            "rows": 0,
            "flat_dim": int(flat.size(1)),
            "mean_sum": 0.0,
            "abs_mean_sum": 0.0,
            "l2_mean_sum": 0.0,
        })
        rows = int(flat.size(0))
        row["batches"] += 1
        row["rows"] += rows
        row["mean_sum"] += float(flat.mean().item()) * rows
        row["abs_mean_sum"] += float(flat.abs().mean().item()) * rows
        row["l2_mean_sum"] += float(flat.norm(dim=1).mean().item()) * rows


def finalize_watch_stats(storage: dict) -> dict[str, dict[str, float]]:
    result = {}
    for name, row in storage.items():
        rows = max(1, int(row["rows"]))
        result[name] = {
            "batches": int(row["batches"]),
            "rows": int(row["rows"]),
            "flat_dim": int(row["flat_dim"]),
            "mean": float(row["mean_sum"]) / rows,
            "abs_mean": float(row["abs_mean_sum"]) / rows,
            "l2_mean": float(row["l2_mean_sum"]) / rows,
        }
    return result


def watch_changes(baseline: dict, candidate: dict) -> dict[str, dict[str, float]]:
    result = {}
    for name, base in baseline.items():
        current = candidate.get(name)
        if current is None:
            continue
        result[name] = {
            "mean_delta": float(current["mean"]) - float(base["mean"]),
            "abs_mean_delta": float(current["abs_mean"]) - float(base["abs_mean"]),
            "l2_mean_delta": float(current["l2_mean"]) - float(base["l2_mean"]),
        }
    return result
</file>

<file path="utils/observatory/audit/metrics.py">
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import polars as pl
import torch
from sklearn.metrics import log_loss, roc_auc_score


def safe_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return math.nan
    return float(roc_auc_score(labels, preds))


def safe_logloss(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0:
        return math.nan
    return float(log_loss(labels, np.clip(preds, 1e-7, 1.0 - 1e-7), labels=[0, 1]))


def metric_row(
        *,
        model_name: str,
        split: str,
        scope: str,
        labels: np.ndarray,
        preds: np.ndarray,
        trained_batches: int,
) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split,
        "scope": scope,
        "trained_batches": trained_batches,
        "rows": int(labels.size),
        "positives": int(labels.sum()) if labels.size else 0,
        "ctr": float(labels.mean()) if labels.size else math.nan,
        "pred_mean": float(preds.mean()) if preds.size else math.nan,
        "auc": safe_auc(labels, preds),
        "logloss": safe_logloss(labels, preds),
    }


def gradient_vector(grad: torch.Tensor) -> torch.Tensor:
    grad = grad.detach().float()
    if grad.ndim == 0:
        return grad.reshape(1)
    if grad.ndim == 1:
        return grad.reshape(-1, 1).mean(dim=0)
    return grad.reshape(grad.size(0), -1).mean(dim=0)


def compare_gradients(
        left_grads: Mapping[str, torch.Tensor],
        right_grads: Mapping[str, torch.Tensor],
) -> list[dict[str, object]]:
    rows = []
    hidden_taps = {"shared_z", "head_input", "logits"}
    for name in sorted(set(left_grads) & set(right_grads)):
        left = gradient_vector(left_grads[name])
        right = gradient_vector(right_grads[name])
        if left.numel() != right.numel():
            continue
        dot = float(torch.dot(left, right).detach().cpu().item())
        norm_left = float(torch.linalg.vector_norm(left).detach().cpu().item())
        norm_right = float(torch.linalg.vector_norm(right).detach().cpu().item())
        cosine = dot / (norm_left * norm_right) if norm_left > 0.0 and norm_right > 0.0 else math.nan
        rows.append({
            "tap": name,
            "tap_type": "hidden" if name in hidden_taps else "field",
            "vector_dim": int(left.numel()),
            "dot": dot,
            "norm_left": norm_left,
            "norm_right": norm_right,
            "cosine": cosine,
            "left_shape": "x".join(str(dim) for dim in left_grads[name].shape),
            "right_shape": "x".join(str(dim) for dim in right_grads[name].shape),
        })
    return rows


def summarize_gradients(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pl.DataFrame(rows)
    summary = (
        df.group_by(["model", "split", "mode", "tap", "tap_type"])
        .agg(
            pl.len().alias("batches"),
            pl.max("vector_dim").alias("vector_dim"),
            pl.mean("rows_left").alias("rows_left_mean"),
            pl.mean("rows_right").alias("rows_right_mean"),
            pl.mean("ctr_left").alias("ctr_left_mean"),
            pl.mean("ctr_right").alias("ctr_right_mean"),
            pl.mean("loss_left").alias("loss_left_mean"),
            pl.mean("loss_right").alias("loss_right_mean"),
            pl.sum("dot").alias("dot_sum"),
            (pl.col("norm_left") ** 2).sum().alias("norm_left_sq_sum"),
            (pl.col("norm_right") ** 2).sum().alias("norm_right_sq_sum"),
            pl.mean("cosine").alias("cosine_mean"),
            pl.quantile("cosine", 0.10).alias("cosine_p10"),
            pl.quantile("cosine", 0.50).alias("cosine_p50"),
            (pl.col("cosine") < 0.0).mean().alias("negative_batch_ratio"),
        )
        .with_columns(
            (
                pl.col("dot_sum")
                / (pl.col("norm_left_sq_sum").sqrt() * pl.col("norm_right_sq_sum").sqrt())
            ).alias("global_cosine")
        )
        .sort(["model", "split", "mode", "tap_type", "global_cosine"])
    )
    return summary.to_dicts()
</file>

<file path="utils/observatory/audit/observation.py">
from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager
from rosen.models.base import BaseModel
from rosen.utils.observatory.audit.metrics import (
    compare_gradients,
    metric_row,
    safe_auc,
    safe_logloss,
    summarize_gradients,
)
from rosen.utils.observatory.audit.reporting import (
    format_value as _fmt,
    rank_value as _rank_value,
    write_csv,
)


def _as_sequence(value: Optional[Sequence | str]) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


class ObservationAuditWorkflow:
    """Post-training ObservationPort sensitivity and gradient-conflict audit."""

    def __init__(
            self,
            model: BaseModel,
            manager: SchemaManager,
            *,
            result_loader: Iterable[Interaction],
            gradient_loader: Iterable[Interaction],
            device: str | torch.device,
            trained_batches: int = 0,
            seed: int = 2026,
            train_limit: str = "external",
            result_split: str = "test",
            gradient_split: str = "train",
            output_dir: Optional[Path | str] = None,
            run_name: str = "obp_audit",
            model_name: str = "",
            domains: Optional[Sequence[int] | str] = None,
            fields: Optional[Sequence[str] | str] = None,
            gradient_modes: Optional[Sequence[str] | str] = None,
            audit_batches: int = 12,
            gradient_balanced_max_per_label: int = 512,
            topk: int = 10,
            write_outputs: bool = True,
    ):
        self.model = model.to(device)
        self.manager = manager
        self.result_loader = result_loader
        self.gradient_loader = gradient_loader
        self.device = torch.device(device)
        self.trained_batches = int(trained_batches)
        self.seed = int(seed)
        self.train_limit = str(train_limit)
        self.result_split = result_split
        self.gradient_split = gradient_split
        self.output_dir = Path(output_dir) if output_dir else None
        self.run_name = run_name
        self.model_name = model_name or type(model).__name__
        self.domains = [int(domain) for domain in _as_sequence(domains)]
        self.fields = [str(field) for field in _as_sequence(fields)]
        self.gradient_modes = [str(mode) for mode in (_as_sequence(gradient_modes) or ["natural", "label_balanced"])]
        self.audit_batches = int(audit_batches)
        self.gradient_balanced_max_per_label = int(gradient_balanced_max_per_label)
        self.topk = int(topk)
        self.write_outputs = bool(write_outputs)
        self.audit_outputs: dict[str, object] = {}
        self.audit_output_paths: dict[str, Path] = {}

    @torch.no_grad()
    def predict_arrays(
            self,
            loader: Iterable[Interaction],
            *,
            split: str,
            cut_name: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        if loader is None:
            raise ValueError("ObservationAuditWorkflow requires a result loader.")

        labels_chunks: list[np.ndarray] = []
        domain_chunks: list[np.ndarray] = []
        logit_chunks: list[np.ndarray] = []
        prob_chunks: list[np.ndarray] = []
        port = getattr(self.model, "obp", None)
        self.model.eval()
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(self.device)
            labels = batch[self.manager.label_field].float().view(-1)
            domain_ids = batch[self.manager.domain_field].long().view(-1)
            cut_ctx = port.cutting(cut_name) if port is not None and hasattr(port, "cutting") else nullcontext()
            with cut_ctx:
                logits = self.model.predict(batch).float().view(-1)
            probs = torch.sigmoid(logits)
            labels_chunks.append(labels.detach().cpu().numpy().astype(np.int8))
            domain_chunks.append(domain_ids.detach().cpu().numpy().astype(np.int16))
            logit_chunks.append(logits.detach().cpu().numpy().astype(np.float32))
            prob_chunks.append(probs.detach().cpu().numpy().astype(np.float32))
            if batch_idx == 1 or batch_idx % 20 == 0:
                tag = "baseline" if cut_name is None else f"cut={cut_name}"
                print(f"[AuditPredict] split={split} {tag} batches={batch_idx}", flush=True)

        return {
            "labels": np.concatenate(labels_chunks),
            "domains": np.concatenate(domain_chunks),
            "logits": np.concatenate(logit_chunks),
            "probs": np.concatenate(prob_chunks),
        }

    def metric_rows_from_arrays(
            self,
            arrays: Mapping[str, np.ndarray],
            *,
            split: str,
    ) -> list[dict[str, object]]:
        labels = arrays["labels"]
        probs = arrays["probs"]
        domain_values = arrays["domains"]
        rows = [metric_row(
            model_name=self.model_name,
            split=split,
            scope="overall",
            labels=labels,
            preds=probs,
            trained_batches=self.trained_batches,
        )]
        for domain in self.domains or sorted(int(item) for item in np.unique(domain_values)):
            mask = domain_values == int(domain)
            rows.append(metric_row(
                model_name=self.model_name,
                split=split,
                scope=f"d{int(domain)}",
                labels=labels[mask],
                preds=probs[mask],
                trained_batches=self.trained_batches,
            ))
        return rows

    def feature_sensitivity(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        metric_rows: list[dict[str, object]] = []
        sensitivity_rows: list[dict[str, object]] = []
        baseline = self.predict_arrays(self.result_loader, split=self.result_split)
        split_metric_rows = self.metric_rows_from_arrays(baseline, split=self.result_split)
        metric_rows.extend(split_metric_rows)
        base_by_scope = {str(row["scope"]): row for row in split_metric_rows}

        port = getattr(self.model, "obp", None)
        cut_names = tuple(self.fields) if self.fields else tuple(port.cut_names() if port is not None else ())
        for field in cut_names:
            cut_arrays = self.predict_arrays(self.result_loader, split=self.result_split, cut_name=field)
            scopes = [("overall", np.ones(baseline["labels"].shape, dtype=bool))]
            scopes.extend((f"d{int(domain)}", baseline["domains"] == int(domain)) for domain in self.domains)
            for scope, mask in scopes:
                labels = baseline["labels"][mask]
                cut_probs = cut_arrays["probs"][mask]
                base_probs = baseline["probs"][mask]
                delta_logits = baseline["logits"][mask] - cut_arrays["logits"][mask]
                delta_probs = base_probs - cut_probs
                base = base_by_scope[scope]
                cut_auc = safe_auc(labels, cut_probs)
                cut_logloss = safe_logloss(labels, cut_probs)
                base_auc = float(base["auc"])
                base_logloss = float(base["logloss"])
                sensitivity_rows.append({
                    "model": self.model_name,
                    "split": self.result_split,
                    "field": field,
                    "scope": scope,
                    "trained_batches": self.trained_batches,
                    "rows": int(labels.size),
                    "baseline_auc": base_auc,
                    "cut_auc": cut_auc,
                    "auc_drop": base_auc - cut_auc if not math.isnan(cut_auc) else math.nan,
                    "baseline_logloss": base_logloss,
                    "cut_logloss": cut_logloss,
                    "logloss_delta": cut_logloss - base_logloss if not math.isnan(cut_logloss) else math.nan,
                    "mean_abs_delta_logit": float(np.abs(delta_logits).mean()) if delta_logits.size else math.nan,
                    "mean_signed_delta_logit": float(delta_logits.mean()) if delta_logits.size else math.nan,
                    "mean_abs_delta_prob": float(np.abs(delta_probs).mean()) if delta_probs.size else math.nan,
                    "mean_signed_delta_prob": float(delta_probs.mean()) if delta_probs.size else math.nan,
                })
        return metric_rows, sensitivity_rows

    @staticmethod
    def sample_domain_mask(
            *,
            domain_ids: torch.Tensor,
            labels: torch.Tensor,
            domain: int,
            mode: str,
            max_per_label: int,
            generator: torch.Generator,
    ) -> torch.Tensor:
        domain_mask = domain_ids.long().view(-1) == int(domain)
        if mode == "natural":
            return domain_mask
        if mode != "label_balanced":
            raise ValueError(f"Unknown gradient mode: {mode!r}.")
        positives = torch.nonzero(domain_mask & (labels.view(-1) > 0.5), as_tuple=False).view(-1)
        negatives = torch.nonzero(domain_mask & (labels.view(-1) <= 0.5), as_tuple=False).view(-1)
        count = min(int(positives.numel()), int(negatives.numel()))
        if max_per_label > 0:
            count = min(count, int(max_per_label))
        balanced = torch.zeros_like(domain_mask)
        if count <= 0:
            return balanced
        pos_perm = torch.randperm(positives.numel(), device=positives.device, generator=generator)[:count]
        neg_perm = torch.randperm(negatives.numel(), device=negatives.device, generator=generator)[:count]
        balanced[positives[pos_perm]] = True
        balanced[negatives[neg_perm]] = True
        return balanced

    @staticmethod
    def slice_interaction(batch: Interaction, mask: torch.Tensor) -> Interaction:
        indices = torch.nonzero(mask, as_tuple=False).view(-1)
        return batch[indices]

    def zero_grad(self) -> None:
        for param in self.model.parameters():
            param.grad = None

    def gradient_conflict(self) -> list[dict[str, object]]:
        if len(self.domains) != 2:
            raise ValueError("ObservationAuditWorkflow gradient conflict expects exactly two domains.")
        left_domain, right_domain = int(self.domains[0]), int(self.domains[1])
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + self.trained_batches)
        rows: list[dict[str, object]] = []
        port = getattr(self.model, "obp", None)
        if port is None or not hasattr(port, "tapping_gradients"):
            return rows

        self.model.eval()
        for batch_idx, batch in enumerate(self.gradient_loader, start=1):
            if self.audit_batches > 0 and batch_idx > self.audit_batches:
                break
            batch = batch.to(self.device)
            labels = batch[self.manager.label_field].float().view(-1)
            domain_ids = batch[self.manager.domain_field].long().view(-1)
            for mode in self.gradient_modes:
                left_mask = self.sample_domain_mask(
                    domain_ids=domain_ids,
                    labels=labels,
                    domain=left_domain,
                    mode=mode,
                    max_per_label=self.gradient_balanced_max_per_label,
                    generator=generator,
                )
                right_mask = self.sample_domain_mask(
                    domain_ids=domain_ids,
                    labels=labels,
                    domain=right_domain,
                    mode=mode,
                    max_per_label=self.gradient_balanced_max_per_label,
                    generator=generator,
                )
                rows_left = int(left_mask.sum().detach().cpu().item())
                rows_right = int(right_mask.sum().detach().cpu().item())
                if rows_left <= 0 or rows_right <= 0:
                    continue

                self.zero_grad()
                with port.tapping_gradients(enabled=True, clear=True):
                    loss_left = self.model.calculate_loss(self.slice_interaction(batch, left_mask))
                    loss_left.backward()
                    left_grads = port.gradient_snapshot(clear=True)

                self.zero_grad()
                with port.tapping_gradients(enabled=True, clear=True):
                    loss_right = self.model.calculate_loss(self.slice_interaction(batch, right_mask))
                    loss_right.backward()
                    right_grads = port.gradient_snapshot(clear=True)

                for row in compare_gradients(left_grads, right_grads):
                    row.update({
                        "model": self.model_name,
                        "split": self.gradient_split,
                        "mode": mode,
                        "batch_idx": batch_idx,
                        "left_domain": left_domain,
                        "right_domain": right_domain,
                        "rows_left": rows_left,
                        "rows_right": rows_right,
                        "ctr_left": float(labels[left_mask].mean().detach().cpu().item()),
                        "ctr_right": float(labels[right_mask].mean().detach().cpu().item()),
                        "loss_left": float(loss_left.detach().cpu().item()),
                        "loss_right": float(loss_right.detach().cpu().item()),
                    })
                    rows.append(row)
                self.zero_grad()
            if batch_idx == 1 or batch_idx % 5 == 0:
                print(f"[AuditGrad] split={self.gradient_split} batches={batch_idx}", flush=True)
        self.zero_grad()
        return rows

    def _output_paths(self) -> dict[str, Path]:
        if self.output_dir is None:
            return {}
        return {
            "metrics": self.output_dir / f"{self.run_name}.metrics.csv",
            "sensitivity": self.output_dir / f"{self.run_name}.sensitivity.csv",
            "gradient_batches": self.output_dir / f"{self.run_name}.gradient_batches.csv",
            "gradient_summary": self.output_dir / f"{self.run_name}.gradient_summary.csv",
            "summary": self.output_dir / f"{self.run_name}.md",
        }

    def make_report(
            self,
            *,
            output_paths: Mapping[str, Path],
            metrics: Sequence[Mapping[str, object]],
            sensitivity: Sequence[Mapping[str, object]],
            gradient_summary: Sequence[Mapping[str, object]],
    ) -> str:
        lines = [
            "# ObservationPort Audit",
            "",
            f"- model: `{self.model_name}`",
            f"- domains: {self.domains}",
            f"- train_limit: `{self.train_limit}`",
            f"- trained_batches: {self.trained_batches}",
            f"- result_split: `{self.result_split}`",
            f"- gradient_split: `{self.gradient_split}`",
            "",
            "## Output Files",
            "",
        ]
        for label, path in output_paths.items():
            lines.append(f"- {label}: `{path}`")

        lines.extend(["", "## Baseline Metrics", ""])
        for row in metrics:
            if row["scope"] != "overall":
                continue
            lines.append(
                f"- {row['split']}: auc={_fmt(row['auc'])} "
                f"logloss={_fmt(row['logloss'])} ctr={_fmt(row['ctr'])} rows={row['rows']}"
            )

        lines.extend(["", f"## Top Feature Sensitivity ({self.result_split})", ""])
        top_sensitivity = [
            row for row in sensitivity
            if row["scope"] == "overall" and row["split"] == self.result_split
        ]
        for row in sorted(top_sensitivity, key=lambda item: _rank_value(item.get("auc_drop"), -math.inf), reverse=True)[: self.topk]:
            lines.append(
                f"- {row['field']}: auc_drop={_fmt(row['auc_drop'])} "
                f"logloss_delta={_fmt(row['logloss_delta'])} "
                f"abs_logit={_fmt(row['mean_abs_delta_logit'])}"
            )

        summary_mode = self.gradient_modes[-1] if self.gradient_modes else ""
        if any(row.get("mode") == summary_mode for row in gradient_summary):
            top_grad = [
                row for row in gradient_summary
                if row["split"] == self.gradient_split and row["mode"] == summary_mode
            ]
        else:
            top_grad = [
                row for row in gradient_summary
                if row["split"] == self.gradient_split
            ]
        lines.extend(["", f"## Most Conflicting Gradient Taps ({self.gradient_split}/{summary_mode})", ""])
        for row in sorted(top_grad, key=lambda item: _rank_value(item.get("global_cosine"), math.inf))[: self.topk]:
            lines.append(
                f"- {row['tap']}: global_cos={_fmt(row['global_cosine'])} "
                f"mean_cos={_fmt(row['cosine_mean'])} "
                f"neg_batch={_fmt(row['negative_batch_ratio'])} "
                f"type={row['tap_type']}"
            )
        return "\n".join(lines) + "\n"

    def run_observation_audit(self) -> dict[str, object]:
        metrics, sensitivity = self.feature_sensitivity()
        gradient_batches = self.gradient_conflict()
        gradient_summary = summarize_gradients(gradient_batches)

        output_paths = self._output_paths()
        if self.write_outputs and output_paths:
            write_csv(output_paths["metrics"], metrics)
            write_csv(output_paths["sensitivity"], sensitivity)
            write_csv(output_paths["gradient_batches"], gradient_batches)
            write_csv(output_paths["gradient_summary"], gradient_summary)
            report = self.make_report(
                output_paths=output_paths,
                metrics=metrics,
                sensitivity=sensitivity,
                gradient_summary=gradient_summary,
            )
            output_paths["summary"].write_text(report, encoding="utf-8")
            print(report, flush=True)

        self.audit_outputs = {
            "metrics": metrics,
            "sensitivity": sensitivity,
            "gradient_batches": gradient_batches,
            "gradient_summary": gradient_summary,
        }
        self.audit_output_paths = output_paths
        return self.audit_outputs

    def run(self):
        return self.run_observation_audit()
</file>

<file path="utils/observatory/audit/reporting.py">
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping, Sequence


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def format_value(value: object, digits: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def rank_value(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(numeric) else numeric
</file>

<file path="utils/observatory/metrics/__init__.py">
from rosen.utils.observatory.metrics.base import MetricInput, MetricResult, MetricSpec
from rosen.utils.observatory.metrics.basic import (
    basic_stats,
    dead_dim_ratio,
    feature_mean,
    feature_var,
    flat_dim,
    near_zero_ratio,
    num_samples,
    per_dim_mean,
    per_dim_var,
)
from rosen.utils.observatory.metrics.correlation import (
    correlation_summary,
    linear_cka,
    max_abs_corr,
    mean_abs_corr,
    principal_angles,
    sample_cosine_stats,
    subspace_overlap,
)
from rosen.utils.observatory.metrics.registry import DEFAULT_TENSOR_METRICS, METRIC_GROUPS, ROUTING_METRICS, VQ_METRICS
from rosen.utils.observatory.metrics.routing import (
    expert_dead_ratio,
    expert_usage,
    gate_entropy,
    gate_mean,
    gate_var,
    pre_post_gate_var,
    weighted_feature_var,
)
from rosen.utils.observatory.metrics.spectral import (
    SpectrumStats,
    compute_spectrum,
    effective_rank,
    participation_ratio,
    singular_values,
    spectral_summary,
    stable_rank,
    topk_energy_ratio,
)
from rosen.utils.observatory.metrics.vq import (
    code_entropy,
    code_usage_hist,
    code_usage_js_divergence,
    codebook_min_separation,
    codebook_pairwise_cos,
    nearest_second_margin,
    per_domain_code_usage,
    quantized_cos,
    residual_norm,
    used_code_ratio,
    vq_summary,
)

__all__ = [
    "MetricInput",
    "MetricResult",
    "MetricSpec",
    "SpectrumStats",
    "basic_stats",
    "correlation_summary",
    "compute_spectrum",
    "dead_dim_ratio",
    "DEFAULT_TENSOR_METRICS",
    "effective_rank",
    "expert_dead_ratio",
    "expert_usage",
    "feature_mean",
    "feature_var",
    "flat_dim",
    "gate_entropy",
    "gate_mean",
    "gate_var",
    "linear_cka",
    "max_abs_corr",
    "mean_abs_corr",
    "METRIC_GROUPS",
    "near_zero_ratio",
    "num_samples",
    "participation_ratio",
    "per_dim_mean",
    "per_dim_var",
    "pre_post_gate_var",
    "principal_angles",
    "ROUTING_METRICS",
    "sample_cosine_stats",
    "singular_values",
    "spectral_summary",
    "stable_rank",
    "subspace_overlap",
    "topk_energy_ratio",
    "VQ_METRICS",
    "weighted_feature_var",
    "code_entropy",
    "code_usage_hist",
    "code_usage_js_divergence",
    "codebook_min_separation",
    "codebook_pairwise_cos",
    "nearest_second_margin",
    "per_domain_code_usage",
    "quantized_cos",
    "residual_norm",
    "used_code_ratio",
    "vq_summary",
]
</file>

<file path="utils/observatory/metrics/base.py">
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch


@dataclass
class MetricInput:
    name: str
    tensor: torch.Tensor
    step: Optional[int] = None
    domain_ids: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    kind: str
    axis_names: Tuple[str, ...]
    values: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSpec:
    name: str
    fn: Callable[..., MetricResult]
    kind: str
    axis_names: Tuple[str, ...]
    default_enabled: bool = True
    default_plot: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
</file>

<file path="utils/observatory/metrics/basic.py">
from typing import Dict

import torch

from rosen.utils.observatory.analysis import flatten_tensor


def _flat(x: torch.Tensor) -> torch.Tensor:
    return flatten_tensor(x).float()


def feature_mean(x: torch.Tensor) -> float:
    return float(_flat(x).mean().item())


def feature_var(x: torch.Tensor) -> float:
    return float(_flat(x).var(unbiased=False).item())


def num_samples(x: torch.Tensor) -> int:
    return int(_flat(x).size(0))


def flat_dim(x: torch.Tensor) -> int:
    return int(_flat(x).size(1))


def per_dim_mean(x: torch.Tensor) -> torch.Tensor:
    return _flat(x).mean(dim=0)


def per_dim_var(x: torch.Tensor) -> torch.Tensor:
    return _flat(x).var(dim=0, unbiased=False)


def dead_dim_ratio(x: torch.Tensor, threshold: float = 1e-6) -> float:
    dim_var = per_dim_var(x)
    return float((dim_var < threshold).float().mean().item())


def near_zero_ratio(x: torch.Tensor, threshold: float = 1e-4) -> float:
    return float((_flat(x).abs() < threshold).float().mean().item())


def basic_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "feature_mean": feature_mean(x),
        "feature_var": feature_var(x),
        "num_samples": float(num_samples(x)),
        "flat_dim": float(flat_dim(x)),
        "dead_dim_ratio": dead_dim_ratio(x),
        "near_zero_ratio": near_zero_ratio(x),
    }
</file>

<file path="utils/observatory/metrics/correlation.py">
from typing import Dict, Tuple

import torch

from rosen.utils.observatory.analysis import flatten_tensor


def _flat(x: torch.Tensor) -> torch.Tensor:
    return flatten_tensor(x).float()


def mean_abs_corr(x: torch.Tensor, eps: float = 1e-12) -> float:
    flat_x = _flat(x)
    if flat_x.size(0) < 2 or flat_x.size(1) < 2:
        return 0.0
    centered = flat_x - flat_x.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    valid_mask = std > eps
    if int(valid_mask.sum().item()) < 2:
        return 0.0
    normalized = centered[:, valid_mask] / std[valid_mask].clamp_min(eps)
    corr = normalized.t().mm(normalized) / max(1, normalized.size(0))
    eye = torch.eye(corr.size(0), dtype=torch.bool)
    off_diag = corr.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0
    return float(off_diag.abs().mean().item())


def max_abs_corr(x: torch.Tensor, eps: float = 1e-12) -> float:
    flat_x = _flat(x)
    if flat_x.size(0) < 2 or flat_x.size(1) < 2:
        return 0.0
    centered = flat_x - flat_x.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    valid_mask = std > eps
    if int(valid_mask.sum().item()) < 2:
        return 0.0
    normalized = centered[:, valid_mask] / std[valid_mask].clamp_min(eps)
    corr = normalized.t().mm(normalized) / max(1, normalized.size(0))
    eye = torch.eye(corr.size(0), dtype=torch.bool)
    off_diag = corr.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0
    return float(off_diag.abs().max().item())


def sample_cosine_stats(x: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float]:
    flat_x = _flat(x)
    if flat_x.size(0) < 2 or flat_x.size(1) == 0:
        return 0.0, 0.0
    normalized = torch.nn.functional.normalize(flat_x, p=2, dim=1, eps=eps)
    sim = normalized.mm(normalized.t())
    eye = torch.eye(sim.size(0), dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0, 0.0
    return float(off_diag.mean().item()), float(off_diag.abs().mean().item())


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = _flat(x) - _flat(x).mean(dim=0, keepdim=True)
    y = _flat(y) - _flat(y).mean(dim=0, keepdim=True)
    if x.size(0) < 2 or y.size(0) < 2 or x.size(1) == 0 or y.size(1) == 0:
        return 0.0
    cross = x.t().mm(y)
    xx = x.t().mm(x)
    yy = y.t().mm(y)
    numerator = cross.pow(2).sum()
    denominator = torch.sqrt(xx.pow(2).sum().clamp_min(eps) * yy.pow(2).sum().clamp_min(eps))
    return float((numerator / denominator.clamp_min(eps)).item())


def principal_angles(x: torch.Tensor, y: torch.Tensor, rank: int = 8) -> Tuple[float, float]:
    x = _flat(x) - _flat(x).mean(dim=0, keepdim=True)
    y = _flat(y) - _flat(y).mean(dim=0, keepdim=True)
    usable_rank = min(rank, x.size(0), x.size(1), y.size(0), y.size(1))
    if usable_rank <= 0:
        return 0.0, 0.0
    ux = torch.linalg.svd(x, full_matrices=False).U[:, :usable_rank]
    uy = torch.linalg.svd(y, full_matrices=False).U[:, :usable_rank]
    sigma = torch.linalg.svdvals(ux.t().mm(uy)).clamp(0.0, 1.0)
    return float(sigma.mean().item()), float(sigma.min().item())


def subspace_overlap(x: torch.Tensor, y: torch.Tensor, rank: int = 8) -> Tuple[float, float]:
    return principal_angles(x, y, rank=rank)


def correlation_summary(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12, rank: int = 8) -> Dict[str, float]:
    angle_mean, angle_min = principal_angles(x, y, rank=rank)
    return {
        "mean_abs_corr": mean_abs_corr(x, eps=eps),
        "max_abs_corr": max_abs_corr(x, eps=eps),
        "sample_cos_mean": sample_cosine_stats(x, eps=eps)[0],
        "sample_cos_abs_mean": sample_cosine_stats(x, eps=eps)[1],
        "linear_cka": linear_cka(x, y, eps=eps),
        "subspace_mean_cos": angle_mean,
        "subspace_min_cos": angle_min,
    }
</file>

<file path="utils/observatory/metrics/registry.py">
from typing import Dict, Tuple


DEFAULT_TENSOR_METRICS: Tuple[str, ...] = (
    "feature_mean",
    "feature_var",
    "effective_rank",
    "participation_ratio",
    "stable_rank",
    "dead_dim_ratio",
    "mean_abs_corr",
    "max_abs_corr",
)

ROUTING_METRICS: Tuple[str, ...] = (
    "gate_mean",
    "gate_var",
    "gate_entropy",
    "expert_usage",
    "expert_dead_ratio",
)

VQ_METRICS: Tuple[str, ...] = (
    "used_code_ratio",
    "code_entropy",
    "quantized_cos",
    "residual_norm",
)

METRIC_GROUPS: Dict[str, Tuple[str, ...]] = {
    "default_tensor": DEFAULT_TENSOR_METRICS,
    "routing": ROUTING_METRICS,
    "vq": VQ_METRICS,
}
</file>

<file path="utils/observatory/metrics/routing.py">
from typing import Dict

import torch

from rosen.utils.observatory.analysis import flatten_tensor


def gate_mean(gate_weights: torch.Tensor) -> torch.Tensor:
    return flatten_tensor(gate_weights).float().mean(dim=0)


def gate_var(gate_weights: torch.Tensor) -> torch.Tensor:
    return flatten_tensor(gate_weights).float().var(dim=0, unbiased=False)


def gate_entropy(gate_weights: torch.Tensor, eps: float = 1e-12) -> float:
    flat = flatten_tensor(gate_weights).float()
    probs = flat.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return float(entropy.item())


def expert_usage(gate_weights: torch.Tensor) -> torch.Tensor:
    flat = flatten_tensor(gate_weights).float()
    return flat.mean(dim=0)


def expert_dead_ratio(gate_weights: torch.Tensor, threshold: float = 1e-3) -> float:
    usage = expert_usage(gate_weights)
    return float((usage < threshold).float().mean().item())


def weighted_feature_var(raw_feature: torch.Tensor, gate_weights: torch.Tensor) -> float:
    raw = flatten_tensor(raw_feature).float()
    gate = flatten_tensor(gate_weights).float()
    weighted = raw * gate.mean(dim=-1, keepdim=True)
    return float(weighted.var(unbiased=False).item())


def pre_post_gate_var(raw_feature: torch.Tensor, gated_feature: torch.Tensor) -> Dict[str, float]:
    raw = flatten_tensor(raw_feature).float()
    gated = flatten_tensor(gated_feature).float()
    return {
        "raw_var": float(raw.var(unbiased=False).item()),
        "gated_var": float(gated.var(unbiased=False).item()),
        "ratio": float((gated.var(unbiased=False) / raw.var(unbiased=False).clamp_min(1e-12)).item()),
    }
</file>

<file path="utils/observatory/metrics/spectral.py">
from dataclasses import dataclass
from typing import Dict

import torch

from rosen.utils.observatory.analysis import flatten_tensor


@dataclass
class SpectrumStats:
    singular_values: torch.Tensor
    energy: torch.Tensor
    total_energy: torch.Tensor
    effective_rank: float
    participation_ratio: float
    stable_rank: float
    spectral_entropy: float


def compute_spectrum(x: torch.Tensor, eps: float = 1e-12) -> SpectrumStats:
    flat_x = flatten_tensor(x).float()
    centered = flat_x - flat_x.mean(dim=0, keepdim=True)
    if centered.numel() == 0 or centered.size(1) == 0 or centered.size(0) < 2:
        zeros = torch.zeros(1)
        return SpectrumStats(
            singular_values=zeros,
            energy=zeros,
            total_energy=zeros,
            effective_rank=0.0,
            participation_ratio=0.0,
            stable_rank=0.0,
            spectral_entropy=0.0,
        )

    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.pow(2)
    total_energy = energy.sum().clamp_min(eps)
    probs = energy / total_energy
    spectral_entropy = float((-(probs * probs.clamp_min(eps).log()).sum()).item())
    effective_rank = float(torch.exp(torch.tensor(spectral_entropy)).item())
    participation_ratio = float((total_energy.pow(2) / energy.pow(2).sum().clamp_min(eps)).item())
    stable_rank = float((total_energy / energy.max().clamp_min(eps)).item())
    return SpectrumStats(
        singular_values=singular_values,
        energy=energy,
        total_energy=total_energy,
        effective_rank=effective_rank,
        participation_ratio=participation_ratio,
        stable_rank=stable_rank,
        spectral_entropy=spectral_entropy,
    )


def singular_values(x: torch.Tensor) -> torch.Tensor:
    return compute_spectrum(x).singular_values


def effective_rank(x: torch.Tensor) -> float:
    return compute_spectrum(x).effective_rank


def participation_ratio(x: torch.Tensor) -> float:
    return compute_spectrum(x).participation_ratio


def stable_rank(x: torch.Tensor) -> float:
    return compute_spectrum(x).stable_rank


def topk_energy_ratio(x: torch.Tensor, k: int = 1, eps: float = 1e-12) -> float:
    stats = compute_spectrum(x, eps=eps)
    if stats.energy.numel() == 0:
        return 0.0
    k = max(1, min(int(k), int(stats.energy.numel())))
    return float((stats.energy[:k].sum() / stats.total_energy.clamp_min(eps)).item())


def spectral_summary(x: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    stats = compute_spectrum(x, eps=eps)
    return {
        "effective_rank": stats.effective_rank,
        "participation_ratio": stats.participation_ratio,
        "stable_rank": stats.stable_rank,
        "spectral_entropy": stats.spectral_entropy,
        "top1_energy_ratio": topk_energy_ratio(x, k=1, eps=eps),
        "top2_energy_ratio": topk_energy_ratio(x, k=2, eps=eps),
    }
</file>

<file path="utils/observatory/metrics/vq.py">
from typing import Dict

import torch

from rosen.utils.observatory.analysis import flatten_tensor


def code_usage_hist(code_indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    flat = code_indices.view(-1).long()
    return torch.bincount(flat, minlength=int(codebook_size)).float()


def used_code_ratio(code_indices: torch.Tensor, codebook_size: int) -> float:
    hist = code_usage_hist(code_indices, codebook_size)
    return float((hist > 0).float().mean().item())


def code_entropy(code_indices: torch.Tensor, codebook_size: int, eps: float = 1e-12) -> float:
    hist = code_usage_hist(code_indices, codebook_size)
    probs = hist / hist.sum().clamp_min(eps)
    probs = probs[probs > 0]
    if probs.numel() == 0:
        return 0.0
    return float((-(probs * probs.log()).sum()).item())


def per_domain_code_usage(code_indices: torch.Tensor, domain_ids: torch.Tensor, codebook_size: int) -> torch.Tensor:
    code_indices = code_indices.view(-1).long()
    domain_ids = domain_ids.view(-1).long()
    num_domains = int(domain_ids.max().item()) + 1 if domain_ids.numel() > 0 else 0
    usage = torch.zeros(num_domains, int(codebook_size), dtype=torch.float32, device=code_indices.device)
    for domain_idx in range(num_domains):
        mask = domain_ids == domain_idx
        if mask.any():
            usage[domain_idx] = code_usage_hist(code_indices[mask], codebook_size)
    return usage


def code_usage_js_divergence(code_indices: torch.Tensor, domain_ids: torch.Tensor, codebook_size: int, eps: float = 1e-12) -> float:
    usage = per_domain_code_usage(code_indices, domain_ids, codebook_size)
    if usage.size(0) < 2:
        return 0.0
    probs = usage / usage.sum(dim=-1, keepdim=True).clamp_min(eps)
    mean_prob = probs.mean(dim=0).clamp_min(eps)
    js = 0.0
    for i in range(probs.size(0)):
        p = probs[i].clamp_min(eps)
        m = mean_prob
        js += (p * (p / m).log()).sum()
    return float((js / probs.size(0)).item())


def quantized_cos(z: torch.Tensor, quantized: torch.Tensor, eps: float = 1e-12) -> float:
    z = flatten_tensor(z).float()
    quantized = flatten_tensor(quantized).float()
    z_n = torch.nn.functional.normalize(z, dim=-1, eps=eps)
    q_n = torch.nn.functional.normalize(quantized, dim=-1, eps=eps)
    return float((z_n * q_n).sum(dim=-1).mean().item())


def residual_norm(z: torch.Tensor, quantized: torch.Tensor) -> float:
    z = flatten_tensor(z).float()
    quantized = flatten_tensor(quantized).float()
    return float((z - quantized).norm(dim=-1).mean().item())


def codebook_pairwise_cos(codebook: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    codebook = flatten_tensor(codebook).float()
    codebook = torch.nn.functional.normalize(codebook, dim=-1, eps=eps)
    return codebook.mm(codebook.t())


def codebook_min_separation(codebook: torch.Tensor, eps: float = 1e-12) -> float:
    sim = codebook_pairwise_cos(codebook, eps=eps)
    eye = torch.eye(sim.size(0), dtype=torch.bool)
    off_diag = sim.masked_fill(eye, -1.0)
    if off_diag.numel() == 0:
        return 0.0
    return float((1.0 - off_diag.max().item()))


def nearest_second_margin(similarity: torch.Tensor) -> float:
    if similarity.ndim != 2 or similarity.size(1) < 2:
        return 0.0
    top2 = torch.topk(similarity, k=2, dim=1).values
    return float((top2[:, 0] - top2[:, 1]).mean().item())


def vq_summary(code_indices: torch.Tensor, codebook_size: int, z: torch.Tensor = None, quantized: torch.Tensor = None) -> Dict[str, float]:
    summary = {
        "used_code_ratio": used_code_ratio(code_indices, codebook_size),
        "code_entropy": code_entropy(code_indices, codebook_size),
    }
    if z is not None and quantized is not None:
        summary["quantized_cos"] = quantized_cos(z, quantized)
        summary["residual_norm"] = residual_norm(z, quantized)
    return summary
</file>

<file path="utils/observatory/plots/__init__.py">
from rosen.utils.observatory.plots.base import PlotSeries, PlotSpec
from rosen.utils.observatory.plots.heatmap import (
    plot_domain_code_usage,
    plot_heatmap,
    plot_similarity_matrix,
    plot_step_dim_heatmap,
)
from rosen.utils.observatory.plots.profile import plot_dim_profile, plot_ranked_profile, plot_topk_bar
from rosen.utils.observatory.plots.report import build_multi_series_report, build_tensor_report, build_vq_report
from rosen.utils.observatory.plots.series import plot_multi_series, plot_scalar_series

__all__ = [
    "build_multi_series_report",
    "build_tensor_report",
    "build_vq_report",
    "PlotSeries",
    "PlotSpec",
    "plot_dim_profile",
    "plot_domain_code_usage",
    "plot_heatmap",
    "plot_multi_series",
    "plot_ranked_profile",
    "plot_scalar_series",
    "plot_similarity_matrix",
    "plot_step_dim_heatmap",
    "plot_topk_bar",
]
</file>

<file path="utils/observatory/plots/base.py">
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


PathLike = Union[str, Path]


@dataclass
class PlotSeries:
    name: str
    x: Sequence[float]
    y: Sequence[float]
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class PlotSpec:
    title: str
    xlabel: str
    ylabel: str
    save_path: Optional[PathLike] = None
    figsize: tuple = (8, 5)
    legend: bool = True


def ensure_parent_dir(save_path: Optional[PathLike]) -> None:
    if not save_path:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def finalize_plot(spec: PlotSpec) -> None:
    plt.title(spec.title)
    plt.xlabel(spec.xlabel)
    plt.ylabel(spec.ylabel)
    plt.tight_layout()
    if spec.legend:
        plt.legend()
    if spec.save_path:
        ensure_parent_dir(spec.save_path)
        plt.savefig(spec.save_path, dpi=150, bbox_inches="tight")
        plt.close()


def maybe_numpy(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)
</file>

<file path="utils/observatory/plots/heatmap.py">
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from rosen.utils.observatory.plots.base import PlotSpec, finalize_plot


def plot_heatmap(
        values: Sequence[Sequence[float]],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path=None,
        xticklabels=None,
        yticklabels=None,
        cmap: str = "viridis",
        colorbar_label: str = None,
):
    plt.figure(figsize=(8, 6))
    matrix = np.asarray(values, dtype=float)
    image = plt.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    if xticklabels is not None:
        plt.xticks(np.arange(len(xticklabels)), xticklabels, rotation=45, ha="right")
    if yticklabels is not None:
        plt.yticks(np.arange(len(yticklabels)), yticklabels)
    cbar = plt.colorbar(image)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)
    finalize_plot(PlotSpec(title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path, legend=False))


def plot_domain_code_usage(values, title: str, save_path=None):
    values = np.asarray(values, dtype=float)
    xticklabels = [str(i) for i in range(values.shape[1])]
    yticklabels = [str(i) for i in range(values.shape[0])]
    plot_heatmap(
        values=values,
        title=title,
        xlabel="code_idx",
        ylabel="domain_idx",
        save_path=save_path,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        colorbar_label="usage",
    )


def plot_step_dim_heatmap(values, title: str, save_path=None):
    values = np.asarray(values, dtype=float)
    xticklabels = [str(i) for i in range(values.shape[1])]
    yticklabels = [str(i) for i in range(values.shape[0])]
    plot_heatmap(
        values=values,
        title=title,
        xlabel="dim_idx",
        ylabel="step_idx",
        save_path=save_path,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        colorbar_label="value",
    )


def plot_similarity_matrix(values, title: str, save_path=None, ticklabels=None):
    values = np.asarray(values, dtype=float)
    plot_heatmap(
        values=values,
        title=title,
        xlabel="tensor_b",
        ylabel="tensor_a",
        save_path=save_path,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cmap="coolwarm",
        colorbar_label="similarity",
    )
</file>

<file path="utils/observatory/plots/profile.py">
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from rosen.utils.observatory.plots.base import PlotSpec, finalize_plot, maybe_numpy


def plot_dim_profile(
        dim_idx: Sequence[float],
        values: Sequence[float],
        title: str,
        ylabel: str,
        save_path=None,
        kind: str = "line",
):
    plt.figure(figsize=(9, 5))
    x = maybe_numpy(dim_idx)
    y = maybe_numpy(values)
    if kind == "bar":
        plt.bar(x, y)
    else:
        plt.plot(x, y, marker="o", linewidth=1.6)
    finalize_plot(PlotSpec(title=title, xlabel="dim_idx", ylabel=ylabel, save_path=save_path, legend=False))


def plot_topk_bar(
        dim_idx: Sequence[float],
        values: Sequence[float],
        title: str,
        ylabel: str,
        save_path=None,
):
    plot_dim_profile(dim_idx=dim_idx, values=values, title=title, ylabel=ylabel, save_path=save_path, kind="bar")


def plot_ranked_profile(
        values: Sequence[float],
        title: str,
        ylabel: str,
        save_path=None,
        descending: bool = True,
):
    values = maybe_numpy(values)
    ranked = np.sort(values)
    if descending:
        ranked = ranked[::-1]
    dim_idx = np.arange(ranked.size)
    plot_dim_profile(dim_idx=dim_idx, values=ranked, title=title, ylabel=ylabel, save_path=save_path, kind="line")
</file>

<file path="utils/observatory/plots/report.py">
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from rosen.utils.observatory.plots.heatmap import plot_domain_code_usage
from rosen.utils.observatory.plots.profile import plot_dim_profile
from rosen.utils.observatory.plots.series import plot_multi_series, plot_scalar_series
from rosen.utils.observatory.plots.base import PlotSeries


def build_tensor_report(
        output_dir,
        step_values: Dict[str, Sequence[float]],
        step_axis: Sequence[float],
        dim_profile: Dict[str, Sequence[float]] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, values in step_values.items():
        plot_scalar_series(
            steps=step_axis,
            values=values,
            title=metric_name,
            ylabel=metric_name,
            save_path=output_dir / ("%s_series.png" % metric_name),
        )

    if dim_profile:
        for metric_name, values in dim_profile.items():
            plot_dim_profile(
                dim_idx=np.arange(len(values)),
                values=values,
                title=metric_name,
                ylabel=metric_name,
                save_path=output_dir / ("%s_profile.png" % metric_name),
            )


def build_vq_report(
        output_dir,
        entropy_series: Sequence[float],
        step_axis: Sequence[float],
        domain_code_usage,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scalar_series(
        steps=step_axis,
        values=entropy_series,
        title="code_entropy",
        ylabel="code_entropy",
        save_path=output_dir / "code_entropy_series.png",
    )
    plot_domain_code_usage(
        values=domain_code_usage,
        title="domain_code_usage",
        save_path=output_dir / "domain_code_usage.png",
    )


def build_multi_series_report(output_dir, series_map: Dict[str, Iterable[PlotSeries]], xlabel: str, ylabel: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for report_name, series_list in series_map.items():
        plot_multi_series(
            series_list=series_list,
            title=report_name,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=output_dir / ("%s.png" % report_name),
        )
</file>

<file path="utils/observatory/plots/series.py">
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

from rosen.utils.observatory.plots.base import PlotSeries, PlotSpec, finalize_plot, maybe_numpy


def plot_scalar_series(
        steps: Sequence[float],
        values: Sequence[float],
        title: str,
        ylabel: str,
        save_path=None,
        label: str = None,
):
    plt.figure(figsize=(8, 5))
    plt.plot(maybe_numpy(steps), maybe_numpy(values), marker="o", linewidth=1.8, label=label or ylabel)
    finalize_plot(PlotSpec(title=title, xlabel="step", ylabel=ylabel, save_path=save_path, legend=label is not None))


def plot_multi_series(
        series_list: Iterable[PlotSeries],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path=None,
):
    plt.figure(figsize=(8, 5))
    for series in series_list:
        plt.plot(maybe_numpy(series.x), maybe_numpy(series.y), marker="o", linewidth=1.6, label=series.name)
    finalize_plot(PlotSpec(title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path, legend=True))
</file>

<file path="utils/observatory/__init__.py">
from rosen.utils.observatory.activation import IndividualReLURecorder
from rosen.utils.observatory.collector import TensorObservatory
from rosen.utils.observatory.config import (
    RelationOptions,
    TensorDisplayConfig,
    TensorMonitorOptions,
    TensorSketchConfig,
)
from rosen.utils.observatory.connector import record_observation_port
from rosen.utils.observatory.monitor import HealthMonitor, HealthMonitorConfig
from rosen.utils.observatory.probe import ModuleProbeSpec
from rosen.utils.observatory.snapshot import ModelSnapshot, SignalSnapshot
from rosen.utils.observatory.store import SnapshotStore
from rosen.utils.observatory.intervention import FieldZeroingResult, field_zeroing_intervention
from rosen.utils.observatory.port import AxisSpec, CapturedObservation, ObservationPort, ObservationSpec
from rosen.utils.observatory.metrics import (
    DEFAULT_TENSOR_METRICS,
    METRIC_GROUPS,
    ROUTING_METRICS,
    VQ_METRICS,
)
from rosen.utils.observatory.plots import (
    PlotSeries,
    PlotSpec,
    build_multi_series_report,
    build_tensor_report,
    build_vq_report,
    plot_dim_profile,
    plot_domain_code_usage,
    plot_heatmap,
    plot_multi_series,
    plot_ranked_profile,
    plot_scalar_series,
    plot_similarity_matrix,
    plot_step_dim_heatmap,
    plot_topk_bar,
)

__all__ = [
    "build_multi_series_report",
    "build_tensor_report",
    "build_vq_report",
    "AxisSpec",
    "CapturedObservation",
    "DEFAULT_TENSOR_METRICS",
    "FieldZeroingResult",
    "HealthMonitor",
    "HealthMonitorConfig",
    "IndividualReLURecorder",
    "METRIC_GROUPS",
    "PlotSeries",
    "PlotSpec",
    "RelationOptions",
    "ROUTING_METRICS",
    "TensorDisplayConfig",
    "TensorMonitorOptions",
    "TensorObservatory",
    "TensorSketchConfig",
    "ObservationPort",
    "ObservationSpec",
    "ModelSnapshot",
    "ModuleProbeSpec",
    "SignalSnapshot",
    "SnapshotStore",
    "VQ_METRICS",
    "plot_dim_profile",
    "plot_domain_code_usage",
    "plot_heatmap",
    "plot_multi_series",
    "plot_ranked_profile",
    "plot_scalar_series",
    "plot_similarity_matrix",
    "plot_step_dim_heatmap",
    "plot_topk_bar",
    "field_zeroing_intervention",
    "record_observation_port",
]
</file>

<file path="utils/observatory/activation.py">
from collections import deque

import torch.nn as nn


class IndividualReLURecorder:
    def __init__(self, model, window_size=50):
        self.window_size = window_size
        self.zero_ratios = {}
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        registered_ids = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                mod_id = id(module)
                if mod_id in registered_ids:
                    print(f"[提示] 层 '{name}' 复用了已挂载的 ReLU 实例，跳过重复注册。")
                    continue
                self.zero_ratios[name] = deque(maxlen=self.window_size)
                hook_handle = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook_handle)
                registered_ids.add(mod_id)

    def _make_hook(self, layer_name):
        def hook(module, input_args, output):
            del module, input_args
            zero_ratio = (output.detach() == 0).float().mean().item()
            self.zero_ratios[layer_name].append(zero_ratio)
        return hook

    def get_layer_stats(self):
        stats = {}
        for name, ratios in self.zero_ratios.items():
            stats[name] = sum(ratios) / len(ratios) if ratios else 0.0
        return stats

    def get_latest_batch_stats(self):
        return {name: ratios[-1] for name, ratios in self.zero_ratios.items() if ratios}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
</file>

<file path="utils/observatory/analysis.py">
import math
from typing import Dict, Tuple

import torch


def flatten_tensor(tensor_data: torch.Tensor) -> torch.Tensor:
    tensor_cpu = tensor_data.detach().float().cpu()
    if tensor_cpu.ndim == 0:
        return tensor_cpu.view(1, 1)
    if tensor_cpu.ndim == 1:
        return tensor_cpu.view(-1, 1)
    if tensor_cpu.numel() == 0:
        return tensor_cpu.new_empty((tensor_cpu.shape[0], math.prod(tensor_cpu.shape[1:])))
    return tensor_cpu.reshape(tensor_cpu.shape[0], -1)


def uniform_sample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
    if x.size(0) <= max_rows:
        return x
    row_index = torch.linspace(
        0,
        x.size(0) - 1,
        steps=max_rows,
        device=x.device,
    ).round().long()
    return x.index_select(0, row_index)


def select_top_variance_dims(x: torch.Tensor, max_dims: int) -> torch.Tensor:
    if x.size(1) <= max_dims:
        return x
    dim_var = x.var(dim=0, unbiased=False)
    topk = min(max_dims, x.size(1))
    dim_index = torch.topk(dim_var, k=topk).indices.sort().values
    return x.index_select(1, dim_index)


def build_sketch(x: torch.Tensor, max_rows: int, max_dims: int) -> torch.Tensor:
    sketch = uniform_sample_rows(x, max_rows)
    sketch = select_top_variance_dims(sketch, max_dims)
    return sketch.contiguous()


def safe_corrcoef(x: torch.Tensor, eps: float) -> Tuple[float, float]:
    if x.size(0) < 2 or x.size(1) < 2:
        return 0.0, 0.0
    centered = x - x.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    valid_mask = std > eps
    if int(valid_mask.sum().item()) < 2:
        return 0.0, 0.0
    normalized = centered[:, valid_mask] / std[valid_mask].clamp_min(eps)
    corr = normalized.t().mm(normalized) / max(1, normalized.size(0))
    eye = torch.eye(corr.size(0), dtype=torch.bool)
    off_diag = corr.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0, 0.0
    return float(off_diag.abs().mean().item()), float(off_diag.abs().max().item())


def sample_cosine_stats(x: torch.Tensor, eps: float) -> Tuple[float, float]:
    if x.size(0) < 2 or x.size(1) == 0:
        return 0.0, 0.0
    normalized = torch.nn.functional.normalize(x, p=2, dim=1, eps=eps)
    sim = normalized.mm(normalized.t())
    eye = torch.eye(sim.size(0), dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0, 0.0
    return float(off_diag.mean().item()), float(off_diag.abs().mean().item())


def spectral_stats(x: torch.Tensor, eps: float) -> Dict[str, float]:
    if x.numel() == 0 or x.size(1) == 0:
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "stable_rank": 0.0,
            "top1_energy_ratio": 0.0,
            "top2_energy_ratio": 0.0,
            "dead_dim_ratio": 1.0,
            "mean_dim_var": 0.0,
            "max_dim_var": 0.0,
            "mean_abs_corr": 0.0,
            "max_abs_corr": 0.0,
            "sample_cosine_mean": 0.0,
            "sample_cosine_abs_mean": 0.0,
        }

    centered = x - x.mean(dim=0, keepdim=True)
    dim_var = centered.var(dim=0, unbiased=False)
    dead_dim_ratio = float((dim_var < 1e-6).float().mean().item())
    mean_dim_var = float(dim_var.mean().item())
    max_dim_var = float(dim_var.max().item()) if dim_var.numel() > 0 else 0.0
    mean_abs_corr, max_abs_corr = safe_corrcoef(centered, eps=eps)
    cosine_mean, cosine_abs_mean = sample_cosine_stats(centered, eps=eps)

    if centered.size(0) < 2:
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "stable_rank": 0.0,
            "top1_energy_ratio": 0.0,
            "top2_energy_ratio": 0.0,
            "dead_dim_ratio": dead_dim_ratio,
            "mean_dim_var": mean_dim_var,
            "max_dim_var": max_dim_var,
            "mean_abs_corr": mean_abs_corr,
            "max_abs_corr": max_abs_corr,
            "sample_cosine_mean": cosine_mean,
            "sample_cosine_abs_mean": cosine_abs_mean,
        }

    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.pow(2)
    total_energy = energy.sum().clamp_min(eps)
    probs = energy / total_energy
    spectral_entropy = -(probs * probs.clamp_min(eps).log()).sum()
    effective_rank = torch.exp(spectral_entropy)
    participation_ratio = total_energy.pow(2) / energy.pow(2).sum().clamp_min(eps)
    stable_rank = total_energy / energy.max().clamp_min(eps)
    top1_ratio = energy[0] / total_energy
    top2_ratio = energy[:2].sum() / total_energy

    return {
        "effective_rank": float(effective_rank.item()),
        "participation_ratio": float(participation_ratio.item()),
        "stable_rank": float(stable_rank.item()),
        "top1_energy_ratio": float(top1_ratio.item()),
        "top2_energy_ratio": float(top2_ratio.item()),
        "dead_dim_ratio": dead_dim_ratio,
        "mean_dim_var": mean_dim_var,
        "max_dim_var": max_dim_var,
        "mean_abs_corr": mean_abs_corr,
        "max_abs_corr": max_abs_corr,
        "sample_cosine_mean": cosine_mean,
        "sample_cosine_abs_mean": cosine_abs_mean,
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    if x.size(0) < 2 or y.size(0) < 2 or x.size(1) == 0 or y.size(1) == 0:
        return 0.0
    cross = x.t().mm(y)
    xx = x.t().mm(x)
    yy = y.t().mm(y)
    numerator = cross.pow(2).sum()
    denominator = torch.sqrt(xx.pow(2).sum().clamp_min(eps) * yy.pow(2).sum().clamp_min(eps))
    return float((numerator / denominator.clamp_min(eps)).item())


def subspace_overlap(x: torch.Tensor, y: torch.Tensor, rank: int) -> Tuple[float, float]:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    usable_rank = min(rank, x.size(0), x.size(1), y.size(0), y.size(1))
    if usable_rank <= 0:
        return 0.0, 0.0
    ux = torch.linalg.svd(x, full_matrices=False).U[:, :usable_rank]
    uy = torch.linalg.svd(y, full_matrices=False).U[:, :usable_rank]
    sigma = torch.linalg.svdvals(ux.t().mm(uy)).clamp(0.0, 1.0)
    return float(sigma.mean().item()), float(sigma.min().item())
</file>

<file path="utils/observatory/collector.py">
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from rosen.utils.observatory.analysis import (
    build_sketch,
    flatten_tensor,
    linear_cka,
    spectral_stats,
    subspace_overlap,
)
from rosen.utils.observatory.config import RelationOptions, TensorDisplayConfig, TensorMonitorOptions
from rosen.utils.observatory.formatting import render_relation_report, render_tensor_report


class TensorObservatory:
    def __init__(
            self,
            window_size: int = 50,
            default_options: Optional[TensorMonitorOptions] = None,
            relation_options: Optional[RelationOptions] = None,
    ):
        self.window_size = int(window_size)
        self.default_options = default_options or TensorMonitorOptions()
        self.relation_options = relation_options or RelationOptions()
        self.features: Dict[str, Deque[Dict[str, object]]] = {}
        self.options_by_name: Dict[str, TensorMonitorOptions] = {}
        self._record_counters: Dict[str, int] = {}

    def register(self, name: str, options: Optional[TensorMonitorOptions] = None) -> None:
        if name not in self.features:
            self.features[name] = deque(maxlen=self.window_size)
        if options is not None:
            self.options_by_name[name] = options

    def configure_relations(self, relation_options: RelationOptions) -> None:
        self.relation_options = relation_options

    def _resolve_options(self, name: str, options: Optional[TensorMonitorOptions]) -> TensorMonitorOptions:
        if options is not None:
            self.options_by_name[name] = options
            return options
        if name in self.options_by_name:
            return self.options_by_name[name]
        return self.default_options

    def _resolve_step(self, name: str, step: Optional[int]) -> int:
        if step is not None:
            resolved = int(step)
        else:
            resolved = int(self._record_counters.get(name, 0))
        self._record_counters[name] = resolved + 1
        return resolved

    def record(self, name, tensor_data, options: Optional[TensorMonitorOptions] = None, step: Optional[int] = None):
        resolved = self._resolve_options(name, options)
        self.register(name)
        resolved_step = self._resolve_step(name, step)

        flat_tensor = flatten_tensor(tensor_data)
        sketch = build_sketch(
            flat_tensor,
            max_rows=resolved.sketch.max_samples,
            max_dims=resolved.sketch.max_dims,
        )

        batch_mean = flat_tensor.mean(dim=0).numpy()
        batch_var = flat_tensor.var(dim=0, unbiased=False).numpy()
        feature_mean = float(flat_tensor.mean().item())
        feature_var = float(flat_tensor.var(unbiased=False).item())

        entry = {
            "step": resolved_step,
            "batch_mean": batch_mean,
            "batch_var": batch_var,
            "feature_mean": feature_mean,
            "feature_var": feature_var,
            "num_samples": int(flat_tensor.size(0)),
            "flat_dim": int(flat_tensor.size(1)),
            "sketch": sketch,
        }
        metrics = set(resolved.metrics)
        if any(metric in metrics for metric in ("spectral", "correlation", "cosine")):
            entry.update(spectral_stats(sketch, eps=resolved.sketch.eps))
        self.features[name].append(entry)

    def _aggregate_window(self, name: str) -> Optional[Dict[str, object]]:
        data_queue = self.features.get(name)
        if not data_queue:
            return None

        window_means = np.stack([item["batch_mean"] for item in data_queue], axis=0)
        window_vars = np.stack([item["batch_var"] for item in data_queue], axis=0)

        sample_mean = np.mean(window_means, axis=0)
        sample_var = np.mean(window_vars, axis=0)
        train_var = np.var(window_means, axis=0)

        scalar_keys = [
            "feature_mean",
            "feature_var",
            "num_samples",
            "flat_dim",
            "effective_rank",
            "participation_ratio",
            "stable_rank",
            "top1_energy_ratio",
            "top2_energy_ratio",
            "dead_dim_ratio",
            "mean_dim_var",
            "max_dim_var",
            "mean_abs_corr",
            "max_abs_corr",
            "sample_cosine_mean",
            "sample_cosine_abs_mean",
        ]
        aggregated = {
            "batch_mean": sample_mean,
            "batch_var": sample_var,
            "train_var": train_var,
        }
        for key in scalar_keys:
            values = [item[key] for item in data_queue if key in item]
            if values:
                aggregated[key] = float(np.mean(values))
        return aggregated

    def get_window_stats(
            self,
            names: Optional[Iterable[str]] = None,
            include_relations: Optional[bool] = None,
            relation_limit: Optional[int] = None,
            relation_names: Optional[Iterable[str]] = None,
    ) -> str:
        if names is None:
            names = list(self.features.keys())
        else:
            names = list(names)

        reports: List[str] = []
        for name in names:
            aggregated = self._aggregate_window(name)
            if aggregated is None:
                continue
            options = self.options_by_name.get(name, self.default_options)
            reports.append(render_tensor_report(name, aggregated, options.display))

        if include_relations is None:
            include_relations = self.relation_options.enabled
        if include_relations:
            relation_text = self.get_relation_stats(
                names=relation_names if relation_names is not None else names,
                max_pairs=relation_limit or self.relation_options.max_pairs,
            )
            if relation_text:
                reports.append(relation_text)
        return "\n".join(reports)

    def _pairwise_relations(self, names: Iterable[str], max_pairs: int) -> List[Dict[str, object]]:
        valid_names = [name for name in names if name in self.features and len(self.features[name]) > 0]
        rows = []
        for i in range(len(valid_names)):
            for j in range(i + 1, len(valid_names)):
                x = self.features[valid_names[i]][-1]["sketch"]
                y = self.features[valid_names[j]][-1]["sketch"]
                common_rows = min(x.size(0), y.size(0))
                if common_rows < 2:
                    continue
                x_aligned = x[:common_rows]
                y_aligned = y[:common_rows]
                eps = self.options_by_name.get(valid_names[i], self.default_options).sketch.eps
                cka = linear_cka(x_aligned, y_aligned, eps=eps)
                overlap_mean, overlap_min = subspace_overlap(x_aligned, y_aligned, rank=self.relation_options.rank)
                rows.append({
                    "tensor_a": valid_names[i],
                    "tensor_b": valid_names[j],
                    "n": int(common_rows),
                    "linear_cka": cka,
                    "subspace_mean_cos": overlap_mean,
                    "subspace_min_cos": overlap_min,
                })
        rows.sort(key=lambda row: row["linear_cka"], reverse=True)
        return rows[:max_pairs]

    def get_relation_stats(self, names: Optional[Iterable[str]] = None, max_pairs: Optional[int] = None) -> str:
        if names is None:
            if self.relation_options.names is not None:
                names = self.relation_options.names
            else:
                names = sorted(self.features.keys())
        else:
            names = list(names)
        rows = self._pairwise_relations(names=names, max_pairs=max_pairs or self.relation_options.max_pairs)
        return render_relation_report(rows)

    def get_scalar_history(self, name: str, key: str) -> Tuple[List[int], List[float]]:
        data_queue = self.features.get(name)
        if not data_queue:
            return [], []
        steps = [int(item.get("step", idx)) for idx, item in enumerate(data_queue)]
        values = [float(item[key]) for item in data_queue if key in item]
        if len(values) != len(steps):
            aligned_steps = []
            aligned_values = []
            for idx, item in enumerate(data_queue):
                if key not in item:
                    continue
                aligned_steps.append(int(item.get("step", idx)))
                aligned_values.append(float(item[key]))
            return aligned_steps, aligned_values
        return steps, values

    def get_vector_history(self, name: str, key: str) -> Tuple[List[int], List[np.ndarray]]:
        data_queue = self.features.get(name)
        if not data_queue:
            return [], []
        steps = []
        values = []
        for idx, item in enumerate(data_queue):
            if key not in item:
                continue
            steps.append(int(item.get("step", idx)))
            values.append(np.asarray(item[key]))
        return steps, values

    def get_step_dim_matrix(self, name: str, key: str) -> Tuple[List[int], np.ndarray]:
        steps, vectors = self.get_vector_history(name, key)
        if not vectors:
            return steps, np.zeros((0, 0), dtype=float)
        return steps, np.stack(vectors, axis=0)


def _build_demo_options() -> Dict[str, TensorMonitorOptions]:
    return {
        "shared_hidden": TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=10,
                topk_display_dims=6,
                rank_by="variance",
            )
        ),
        "specific_hidden": TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=10,
                topk_display_dims=6,
                rank_by="variance",
            )
        ),
        "gate_weights": TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=8,
                topk_display_dims=4,
                rank_by="mean_abs",
            )
        ),
    }


def _demo_record_batches(observatory: TensorObservatory) -> None:
    torch.manual_seed(2026)

    for step in range(4):
        batch_size = 128
        latent = torch.randn(batch_size, 12)
        shared_hidden = torch.cat(
            [
                1.4 * latent[:, :4] + 0.10 * torch.randn(batch_size, 4),
                0.25 * torch.randn(batch_size, 8),
            ],
            dim=1,
        )
        specific_hidden = torch.cat(
            [
                0.35 * latent[:, :4] + 0.25 * torch.randn(batch_size, 4),
                1.1 * torch.randn(batch_size, 8),
            ],
            dim=1,
        )
        gate_logits = torch.stack(
            [
                1.2 + 0.15 * latent[:, 0],
                0.8 + 0.20 * latent[:, 1],
                0.4 + 0.35 * latent[:, 2] + 0.05 * step,
            ],
            dim=-1,
        )
        gate_weights = torch.softmax(gate_logits, dim=-1)

        observatory.record("shared_hidden", shared_hidden)
        observatory.record("specific_hidden", specific_hidden)
        observatory.record("gate_weights", gate_weights)


def main() -> None:
    observatory = TensorObservatory(window_size=4)
    options_by_name = _build_demo_options()
    for name, options in options_by_name.items():
        observatory.register(name, options=options)

    observatory.configure_relations(
        RelationOptions(
            enabled=True,
            rank=4,
            max_pairs=6,
            names=("shared_hidden", "specific_hidden", "gate_weights"),
        )
    )
    _demo_record_batches(observatory)
    print(observatory.get_window_stats())


if __name__ == "__main__":
    main()
</file>

<file path="utils/observatory/config.py">
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TensorSketchConfig:
    max_samples: int = 256
    max_dims: int = 256
    eps: float = 1e-12


@dataclass
class TensorDisplayConfig:
    show_global_summary: bool = True
    show_per_dim: bool = True
    max_display_dims: int = 16
    topk_display_dims: int = 8
    rank_by: str = "variance"


@dataclass
class TensorMonitorOptions:
    metrics: Tuple[str, ...] = ("basic", "spectral", "correlation", "cosine")
    display: TensorDisplayConfig = field(default_factory=TensorDisplayConfig)
    sketch: TensorSketchConfig = field(default_factory=TensorSketchConfig)


@dataclass
class RelationOptions:
    enabled: bool = False
    rank: int = 8
    max_pairs: int = 12
    names: Optional[Tuple[str, ...]] = None
</file>

<file path="utils/observatory/connector.py">
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from rosen.utils.observatory.config import TensorMonitorOptions
from rosen.utils.observatory.port import ObservationPort


def record_observation_port(
        port: ObservationPort,
        recorder: Any,
        *,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        clear: bool = True,
) -> Dict[str, Any]:
    """
    Drain an ObservationPort into a TensorObservatory-like recorder.

    Models own the forward trace. The port owns static observation specs. The
    recorder owns metric computation. This connector is the small boundary
    between those pieces.
    """
    trace = port.snapshot(clear=clear)
    if recorder is None or not hasattr(recorder, "record"):
        return trace

    namespace = port.namespace if prefix is None else str(prefix)
    for name, value in trace.items():
        spec = port.spec(name)
        if spec is None or not spec.metrics:
            continue
        if not isinstance(value, torch.Tensor):
            continue

        record_name = f"{namespace}.{name}" if namespace else str(name)
        options = TensorMonitorOptions(metrics=tuple(spec.metrics))
        recorder.record(record_name, value, options=options, step=step)

    return trace
</file>

<file path="utils/observatory/formatting.py">
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from rosen.utils.observatory.config import TensorDisplayConfig


SCALAR_LABELS = {
    "feature_mean": "feat_mean",
    "feature_var": "feat_var",
    "num_samples": "n",
    "flat_dim": "dim",
    "effective_rank": "eff_rank",
    "participation_ratio": "part_ratio",
    "stable_rank": "stable_rank",
    "top1_energy_ratio": "top1_energy",
    "top2_energy_ratio": "top2_energy",
    "dead_dim_ratio": "dead_dim",
    "mean_dim_var": "mean_dim_var",
    "max_dim_var": "max_dim_var",
    "mean_abs_corr": "mean_abs_corr",
    "max_abs_corr": "max_abs_corr",
    "sample_cosine_mean": "cos_mean",
    "sample_cosine_abs_mean": "abs_cos_mean",
}


def _render_df(df: pl.DataFrame) -> str:
    with pl.Config(
        tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=2000,
        fmt_str_lengths=2000,
    ):
        return str(df)


def _global_summary_frame(window_stats: Dict[str, object]) -> pl.DataFrame:
    row = {}
    for key, label in SCALAR_LABELS.items():
        if key in window_stats:
            row[label] = [window_stats[key]]
    return pl.DataFrame(row)


def _top_dimension_frame(window_stats: Dict[str, object], display: TensorDisplayConfig) -> Optional[pl.DataFrame]:
    mean_vec = np.asarray(window_stats["batch_mean"])
    var_vec = np.asarray(window_stats["batch_var"])
    train_var_vec = np.asarray(window_stats["train_var"])
    if mean_vec.ndim == 0:
        mean_vec = mean_vec.reshape(1)
        var_vec = var_vec.reshape(1)
        train_var_vec = train_var_vec.reshape(1)

    if mean_vec.size == 0:
        return None

    if not display.show_per_dim:
        return None

    if display.rank_by == "mean_abs":
        rank_score = np.abs(mean_vec)
    elif display.rank_by == "train_var":
        rank_score = train_var_vec
    else:
        rank_score = var_vec

    if mean_vec.size <= display.max_display_dims:
        selected = np.arange(mean_vec.size)
    else:
        topk = min(display.topk_display_dims, mean_vec.size)
        selected = np.argsort(-rank_score)[:topk]
        selected = np.sort(selected)

    frame = pl.DataFrame({
        "dim_idx": selected.astype(int),
        "batch_mean": mean_vec[selected],
        "batch_var": var_vec[selected],
        "train_var": train_var_vec[selected],
    })
    return frame


def render_tensor_report(name: str, window_stats: Dict[str, object], display: TensorDisplayConfig) -> str:
    sections: List[str] = [f"{name:=^20}"]
    if display.show_global_summary:
        sections.append(_render_df(_global_summary_frame(window_stats)))

    dim_frame = _top_dimension_frame(window_stats, display)
    if dim_frame is not None:
        if len(dim_frame) < int(window_stats["flat_dim"]):
            sections.append("[Top-Dim]")
        else:
            sections.append("[Per-Dim]")
        sections.append(_render_df(dim_frame))
    return " \n".join(sections)


def render_relation_report(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return ""
    return f"{'relation_stats':=^20} \n{_render_df(pl.DataFrame(rows))}"
</file>

<file path="utils/observatory/intervention.py">
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
</file>

<file path="utils/observatory/monitor.py">
from __future__ import annotations

import csv
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from rosen.utils.observatory.analysis import build_sketch, uniform_sample_rows
from rosen.utils.observatory.metrics.spectral import compute_spectrum
from rosen.utils.observatory.port import CapturedObservation, ObservationPort
from rosen.utils.observatory.probe import ModuleProbeRegistry, ModuleProbeSpec
from rosen.utils.observatory.snapshot import ModelSnapshot, SignalSnapshot
from rosen.utils.observatory.store import SnapshotStore


@dataclass(frozen=True)
class HealthMonitorConfig:
    snapshot_every: int = 100
    summary_capacity: int = 1000
    max_sketch_rows: int = 256
    max_sketch_dims: int = 256
    max_metric_vectors: int = 4096
    max_vector_pairs: int = 2048
    capture_first_step: bool = False
    output_jsonl: str = ""
    print_snapshots: bool = False


class HealthMonitor:
    """Trainer-owned, model-agnostic health snapshot coordinator."""

    def __init__(self, config: HealthMonitorConfig | None = None):
        self.config = config or HealthMonitorConfig()
        self.store = SnapshotStore(self.config.summary_capacity)
        self.probes = ModuleProbeRegistry()
        self._snapshot_sequence = 0
        self._attached = False
        self._jsonl_path = Path(self.config.output_jsonl) if self.config.output_jsonl else None

    def observe_module(
        self,
        module_path: str,
        *,
        prefix: str | None = None,
        include_root: bool = True,
        leaf_policy: str = "representation",
        role: str = "hidden",
        metrics: Sequence[str] = ("basic", "rank"),
        domain_routed: bool = False,
    ) -> None:
        self.probes.add(
            ModuleProbeSpec(
                module_path=str(module_path),
                prefix=str(prefix or module_path or "model"),
                include_root=bool(include_root),
                leaf_policy=str(leaf_policy),
                role=str(role),
                metrics=tuple(str(metric) for metric in metrics),
                domain_routed=bool(domain_routed),
            )
        )

    def ensure_attached(self, model) -> None:
        if self._attached:
            return
        port = getattr(model, "obp", None)
        if not isinstance(port, ObservationPort):
            raise TypeError("HealthMonitor requires model.obp to be an ObservationPort.")
        self.probes.attach(model, port)
        self._attached = True

    def should_capture(self, ctx, *, force: bool = False) -> bool:
        if force:
            return True
        step = int(ctx.global_step) + 1
        if self.config.capture_first_step and step == 1:
            return True
        return step % max(1, int(self.config.snapshot_every)) == 0

    @contextmanager
    def capture_scope(self, model, ctx, *, force: bool = False) -> Iterator[None]:
        self.ensure_attached(model)
        port = model.obp
        with port.collecting(enabled=self.should_capture(ctx, force=force), clear=True):
            yield

    @staticmethod
    def _loss_signals(summary: Mapping[str, Any]) -> dict[str, SignalSnapshot]:
        settlements = summary.get("settlements") or {}
        signals: dict[str, SignalSnapshot] = {}
        for settlement_scope, settlement in settlements.items():
            prefix = (
                "loss"
                if settlement_scope == "main"
                else f"loss.{settlement_scope}"
            )
            total_value = float(settlement["total"])
            total_name = f"{prefix}.total"
            signals[total_name] = SignalSnapshot(
                name=total_name,
                kind="loss",
                role="loss",
                scope="point",
                metrics={
                    "raw": total_value,
                    "weight": 1.0,
                    "weighted": total_value,
                    "finite": math.isfinite(total_value),
                },
                metadata={"settlement_scope": str(settlement_scope)},
            )
            for term_name, term in settlement["terms"].items():
                if term_name == "total":
                    continue
                raw = float(term["raw"])
                weight = float(term["weight"])
                weighted = float(term["weighted"])
                signal_name = f"{prefix}.{term_name}"
                signals[signal_name] = SignalSnapshot(
                    name=signal_name,
                    kind="loss",
                    role="loss",
                    scope="point",
                    metrics={
                        "raw": raw,
                        "weight": weight,
                        "weighted": weighted,
                        "finite": math.isfinite(raw) and math.isfinite(weighted),
                    },
                    metadata={"settlement_scope": str(settlement_scope)},
                )
        return signals

    @staticmethod
    def _vector_matrix(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.reshape(-1, 1)
        vector_dim = int(tensor.shape[-1])
        vector_count = math.prod(int(size) for size in tensor.shape[:-1])
        return tensor.reshape(vector_count, vector_dim)

    @staticmethod
    def _batch_matrix(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.reshape(-1, 1)
        batch_size = int(tensor.shape[0])
        flat_dim = math.prod(int(size) for size in tensor.shape[1:])
        return tensor.reshape(batch_size, flat_dim)

    def _vector_collection_metrics(
        self,
        safe_vectors: torch.Tensor,
        vector_energy: torch.Tensor,
    ) -> dict[str, Any]:
        vector_count, vector_dim = int(safe_vectors.size(0)), int(safe_vectors.size(1))
        empty = {
            "coordinate_participation_ratio": 0.0,
            "pairwise_abs_cosine_mean": 0.0,
            "pairwise_abs_cosine_p95": 0.0,
            "pairwise_sample_count": 0,
            "population_effective_rank": 0.0,
            "population_effective_rank_ratio": 0.0,
        }
        if vector_count == 0 or vector_dim == 0:
            return empty

        squared = safe_vectors.square()
        fourth_energy = squared.square().sum(dim=-1)
        participation = torch.where(
            vector_energy > 1e-12,
            vector_energy.square() / fourth_energy.clamp_min(1e-12) / float(vector_dim),
            torch.zeros_like(vector_energy),
        )

        active = safe_vectors[vector_energy > 1e-12]
        active_count = int(active.size(0))
        pair_mean = 0.0
        pair_p95 = 0.0
        pair_count = 0
        if active_count >= 2 and int(self.config.max_vector_pairs) > 0:
            total_pairs = active_count * (active_count - 1) // 2
            pair_count = min(int(self.config.max_vector_pairs), total_pairs)
            if pair_count == total_pairs:
                pair_indices = torch.triu_indices(active_count, active_count, offset=1)
                left, right = pair_indices[0], pair_indices[1]
            else:
                generator = torch.Generator(device="cpu").manual_seed(0)
                left = torch.randint(active_count, (pair_count,), generator=generator)
                offsets = torch.randint(1, active_count, (pair_count,), generator=generator)
                right = torch.remainder(left + offsets, active_count)
            unit = torch.nn.functional.normalize(active, p=2, dim=-1, eps=1e-12)
            pair_cosine = (unit[left] * unit[right]).sum(dim=-1).abs()
            pair_mean = float(pair_cosine.mean().item())
            pair_p95 = float(torch.quantile(pair_cosine, 0.95).item())

        population_rank = 0.0
        population_rank_ratio = 0.0
        if vector_count >= 2:
            sketch = build_sketch(
                safe_vectors,
                max_rows=max(2, int(self.config.max_sketch_rows)),
                max_dims=max(1, int(self.config.max_sketch_dims)),
            )
            spectrum = compute_spectrum(sketch)
            if float(spectrum.energy.sum().item()) > 1e-12:
                population_rank = spectrum.effective_rank
                observable_rank = max(
                    1,
                    min(int(sketch.size(0)) - 1, int(sketch.size(1))),
                )
                population_rank_ratio = population_rank / observable_rank

        return {
            "coordinate_participation_ratio": float(participation.mean().item()),
            "pairwise_abs_cosine_mean": pair_mean,
            "pairwise_abs_cosine_p95": pair_p95,
            "pairwise_sample_count": pair_count,
            "population_effective_rank": population_rank,
            "population_effective_rank_ratio": population_rank_ratio,
        }

    def _tensor_metrics(self, tensor: torch.Tensor, requested: Sequence[str]) -> dict[str, Any]:
        tensor_float = tensor.detach().float()
        batch_matrix = self._batch_matrix(tensor_float)
        full_vectors = self._vector_matrix(tensor_float)
        vector_count, vector_dim = int(full_vectors.size(0)), int(full_vectors.size(1))
        sampled_vectors = uniform_sample_rows(
            full_vectors,
            max(1, int(self.config.max_metric_vectors)),
        ).cpu()
        metric_vector_count = int(sampled_vectors.size(0))
        if sampled_vectors.numel() == 0:
            finite_vectors = torch.ones_like(sampled_vectors, dtype=torch.bool)
            safe_vectors = sampled_vectors
            vector_energy = sampled_vectors.new_empty((metric_vector_count,))
            l2_metrics = {
                "vector_count": vector_count,
                "vector_dim": vector_dim,
                "metric_vector_count": metric_vector_count,
                "l2_norm_mean": 0.0,
                "l2_norm_std": 0.0,
                "l2_norm_min": 0.0,
                "l2_norm_max": 0.0,
            }
        else:
            finite_vectors = torch.isfinite(sampled_vectors)
            safe_vectors = torch.where(
                finite_vectors,
                sampled_vectors,
                torch.zeros_like(sampled_vectors),
            )
            vector_norms = torch.linalg.vector_norm(safe_vectors, ord=2, dim=-1)
            vector_energy = vector_norms.square()
            l2_metrics = {
                "vector_count": vector_count,
                "vector_dim": vector_dim,
                "metric_vector_count": metric_vector_count,
                "l2_norm_mean": float(vector_norms.mean().item()),
                "l2_norm_std": float(vector_norms.std(unbiased=False).item()),
                "l2_norm_min": float(vector_norms.min().item()),
                "l2_norm_max": float(vector_norms.max().item()),
            }
        requested_set = {str(metric).lower() for metric in requested}
        sample_count, flat_dim = int(batch_matrix.size(0)), int(batch_matrix.size(1))
        if safe_vectors.numel() == 0:
            metrics = {
                "sample_count": sample_count,
                "flat_dim": flat_dim,
                "finite_ratio": 1.0,
                **l2_metrics,
            }
            if "vector_collection" in requested_set:
                metrics.update(self._vector_collection_metrics(safe_vectors, vector_energy))
            return metrics
        finite = finite_vectors.reshape(-1)
        safe = safe_vectors.reshape(-1)
        metrics: dict[str, Any] = {
            "sample_count": sample_count,
            "flat_dim": flat_dim,
            "mean": float(safe.mean().item()),
            "std": float(safe.std(unbiased=False).item()),
            "min": float(safe.min().item()),
            "max": float(safe.max().item()),
            "abs_max": float(safe.abs().max().item()),
            "finite_ratio": float(finite.float().mean().item()),
            "zero_ratio": float((safe.abs() < 1e-8).float().mean().item()),
            **l2_metrics,
        }
        if "vector_collection" in requested_set:
            metrics.update(self._vector_collection_metrics(safe_vectors, vector_energy))
        elif "rank" in requested_set or "spectral" in requested_set:
            sampled_batch = uniform_sample_rows(
                batch_matrix,
                max(2, int(self.config.max_sketch_rows)),
            ).cpu()
            sampled_batch = torch.where(
                torch.isfinite(sampled_batch),
                sampled_batch,
                torch.zeros_like(sampled_batch),
            )
            sketch = build_sketch(
                sampled_batch,
                max_rows=max(2, int(self.config.max_sketch_rows)),
                max_dims=max(1, int(self.config.max_sketch_dims)),
            )
            spectrum = compute_spectrum(sketch)
            denominator = max(1, min(int(sketch.size(0)), int(sketch.size(1))))
            metrics.update({
                "effective_rank": spectrum.effective_rank,
                "rank_ratio": spectrum.effective_rank / denominator,
                "participation_ratio": spectrum.participation_ratio,
                "stable_rank": spectrum.stable_rank,
                "top1_energy_ratio": (
                    float((spectrum.energy[:1].sum() / spectrum.total_energy).item())
                    if spectrum.energy.numel() and float(spectrum.total_energy.item()) > 0.0 else 0.0
                ),
            })
        return metrics

    def _trace_signals(self, port: ObservationPort) -> dict[str, SignalSnapshot]:
        trace = port.summary(clear=True)
        signals: dict[str, SignalSnapshot] = {}
        for name, captured in trace.items():
            if not isinstance(captured, CapturedObservation):
                continue
            value = captured.value
            spec = port.spec(name)
            role = spec.role if spec is not None else "tensor"
            requested = spec.metrics if spec is not None else ("basic", "rank")
            metadata = dict(spec.metadata) if spec is not None else {}
            coordinates = dict(metadata.pop("coordinates", {}))
            if isinstance(value, torch.Tensor):
                metadata["shape"] = list(value.shape)
                signals[captured.full_name] = SignalSnapshot(
                    name=captured.full_name,
                    kind="tensor",
                    role=role,
                    scope="point",
                    metrics=self._tensor_metrics(value, requested),
                    coordinates=coordinates,
                    metadata=metadata,
                )
            else:
                signals[captured.full_name] = SignalSnapshot(
                    name=captured.full_name,
                    kind="state",
                    role=role,
                    scope="point",
                    metrics={"value": value},
                    coordinates=coordinates,
                    metadata=metadata,
                )
        return signals

    def _commit(
        self,
        *,
        ctx,
        stage: str,
        phase: str,
        target: str,
        signals: Mapping[str, SignalSnapshot],
        model_mode: str,
        interval_start: int | None = None,
        interval_end: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelSnapshot:
        self._snapshot_sequence += 1
        step = int(ctx.global_step) + (1 if stage == "train" else 0)
        snapshot = ModelSnapshot(
            snapshot_id=f"{stage}:{phase}:{step}:{self._snapshot_sequence}",
            step=step,
            epoch=int(ctx.epoch),
            batch_idx=int(ctx.batch_idx),
            stage=str(stage),
            phase=str(phase),
            target=str(target),
            model_mode=str(model_mode),
            interval_start=interval_start,
            interval_end=interval_end,
            signals=dict(signals),
            metadata=dict(metadata or {}),
        )
        self.store.commit(snapshot)
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot.to_dict(), ensure_ascii=False, default=str) + "\n")
        if self.config.print_snapshots:
            print(
                f"[HealthSnapshot] step={snapshot.step} stage={snapshot.stage} "
                f"phase={snapshot.phase} signals={len(snapshot.signals)}",
                flush=True,
            )
        return snapshot

    def record_train_step(self, model, ctx) -> ModelSnapshot | None:
        if not self.should_capture(ctx):
            model.obp.clear()
            return None
        step = int(ctx.global_step) + 1
        signals = self._trace_signals(model.obp)
        signals.update(self._loss_signals(model.lsp.summary(clear=True)))
        snapshot = self._commit(
            ctx=ctx,
            stage="train",
            phase=getattr(ctx, "phase", "default"),
            target="student",
            model_mode="train",
            signals=signals,
            interval_start=step,
            interval_end=step,
            metadata={},
        )
        return snapshot

    def record_training_cycle(
        self,
        model,
        ctx,
        *,
        state_names: tuple[str, ...] = (),
        phase: str | None = None,
    ) -> ModelSnapshot | None:
        metadata = {
            "model_updated": bool(state_names),
            "updated_state_names": list(state_names),
        }
        return self._record_probe(
            model,
            ctx,
            phase=phase,
            metadata=metadata,
            allow_empty=bool(state_names),
        )

    def record_probe(
        self,
        model,
        ctx,
        *,
        target: str = "student",
        phase: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelSnapshot | None:
        """Commit one explicitly captured model probe outside periodic train steps."""

        return self._record_probe(
            model,
            ctx,
            target=target,
            phase=phase,
            metadata=metadata,
            allow_empty=False,
        )

    def _record_probe(
        self,
        model,
        ctx,
        *,
        target: str = "student",
        phase: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        allow_empty: bool,
    ) -> ModelSnapshot | None:
        trace = self._trace_signals(model.obp)
        trace.update(self._loss_signals(model.lsp.summary(clear=True)))
        if not trace and not allow_empty:
            return None
        return self._commit(
            ctx=ctx,
            stage=str(ctx.stage),
            phase=str(phase or getattr(ctx, "phase", "default")),
            target=str(target),
            model_mode="train" if model.training else "eval",
            signals=trace,
            metadata=metadata,
        )

    def flush(self, model, ctx, *, stage: str = "train_end") -> ModelSnapshot | None:
        signals = self._loss_signals(model.lsp.summary(clear=True))
        if not signals:
            return None
        step = int(ctx.global_step)
        snapshot = self._commit(
            ctx=ctx,
            stage=stage,
            phase=getattr(ctx, "phase", "default"),
            target="student",
            model_mode="eval" if not model.training else "train",
            signals=signals,
            interval_start=step,
            interval_end=step,
        )
        return snapshot

    def export_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "snapshot_id", "step", "epoch", "batch_idx", "stage", "phase", "target",
            "signal", "kind", "role", "scope", "metric", "value", "metadata",
        ]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for snapshot in self.store.list():
                for signal_name, signal in snapshot.signals.items():
                    for metric, value in signal.metrics.items():
                        writer.writerow({
                            "snapshot_id": snapshot.snapshot_id,
                            "step": snapshot.step,
                            "epoch": snapshot.epoch,
                            "batch_idx": snapshot.batch_idx,
                            "stage": snapshot.stage,
                            "phase": snapshot.phase,
                            "target": snapshot.target,
                            "signal": signal_name,
                            "kind": signal.kind,
                            "role": signal.role,
                            "scope": signal.scope,
                            "metric": metric,
                            "value": value,
                            "metadata": json.dumps(signal.metadata, ensure_ascii=False, default=str),
                        })
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "snapshot_sequence": self._snapshot_sequence,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._snapshot_sequence = int(state.get("snapshot_sequence", self._snapshot_sequence))

    def close(self) -> None:
        self.probes.close()
        self._attached = False


__all__ = ["HealthMonitor", "HealthMonitorConfig"]
</file>

<file path="utils/observatory/port.py">
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
</file>

<file path="utils/observatory/probe.py">
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
</file>

<file path="utils/observatory/snapshot.py">
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
</file>

<file path="utils/observatory/store.py">
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
</file>

<file path="utils/__init__.py">
from .time_bucket import (
    AdjacentTimeEmbedding,
    RelativeTimeEmbedding,
    TIME_BUCKET_BOUNDARIES,
    bucketize_adjacent_time,
    bucketize_relative_time,
    build_padding_mask,
    get_num_time_buckets,
)
from .observatory import (
    AxisSpec,
    CapturedObservation,
    DEFAULT_TENSOR_METRICS,
    FieldZeroingResult,
    HealthMonitor,
    HealthMonitorConfig,
    IndividualReLURecorder,
    METRIC_GROUPS,
    ObservationPort,
    ObservationSpec,
    ModelSnapshot,
    ModuleProbeSpec,
    SignalSnapshot,
    SnapshotStore,
    PlotSeries,
    PlotSpec,
    RelationOptions,
    ROUTING_METRICS,
    TensorDisplayConfig,
    TensorMonitorOptions,
    TensorObservatory,
    TensorSketchConfig,
    VQ_METRICS,
    build_multi_series_report,
    build_tensor_report,
    build_vq_report,
    plot_dim_profile,
    plot_domain_code_usage,
    plot_heatmap,
    plot_multi_series,
    plot_ranked_profile,
    plot_scalar_series,
    plot_similarity_matrix,
    plot_step_dim_heatmap,
    plot_topk_bar,
    field_zeroing_intervention,
    record_observation_port,
)

__all__ = [
    "AdjacentTimeEmbedding",
    "RelativeTimeEmbedding",
    "TIME_BUCKET_BOUNDARIES",
    "bucketize_adjacent_time",
    "bucketize_relative_time",
    "build_padding_mask",
    "get_num_time_buckets",
    "build_multi_series_report",
    "build_tensor_report",
    "build_vq_report",
    "AxisSpec",
    "CapturedObservation",
    "DEFAULT_TENSOR_METRICS",
    "FieldZeroingResult",
    "HealthMonitor",
    "HealthMonitorConfig",
    "IndividualReLURecorder",
    "METRIC_GROUPS",
    "ObservationPort",
    "ObservationSpec",
    "ModelSnapshot",
    "ModuleProbeSpec",
    "SignalSnapshot",
    "SnapshotStore",
    "PlotSeries",
    "PlotSpec",
    "RelationOptions",
    "ROUTING_METRICS",
    "TensorDisplayConfig",
    "TensorMonitorOptions",
    "TensorObservatory",
    "TensorSketchConfig",
    "VQ_METRICS",
    "plot_dim_profile",
    "plot_domain_code_usage",
    "plot_heatmap",
    "plot_multi_series",
    "plot_ranked_profile",
    "plot_scalar_series",
    "plot_similarity_matrix",
    "plot_step_dim_heatmap",
    "plot_topk_bar",
    "field_zeroing_intervention",
    "record_observation_port",
]
</file>

<file path="utils/expr_tensor.py">
from __future__ import annotations

from typing import Any

import torch


class ExprTensor:
    """Tensor wrapper that records a symbolic expression while running real ops."""

    def __init__(self, data: torch.Tensor, expr: str):
        self.data = data
        self.expr = expr

    @staticmethod
    def _unwrap(x: Any):
        if isinstance(x, ExprTensor):
            return x.data
        if isinstance(x, (list, tuple)):
            return type(x)(ExprTensor._unwrap(i) for i in x)
        return x

    @staticmethod
    def _get_expr(a: Any) -> str:
        if isinstance(a, ExprTensor):
            return a.expr
        if isinstance(a, torch.nn.Parameter):
            return "W"
        if isinstance(a, torch.Tensor):
            return "W" if a.requires_grad else "C"
        if isinstance(a, (list, tuple)):
            return "[" + ", ".join(ExprTensor._get_expr(i) for i in a) + "]"
        if isinstance(a, slice):
            start = ExprTensor._get_expr(a.start) if a.start is not None else ""
            stop = ExprTensor._get_expr(a.stop) if a.stop is not None else ""
            step = f":{ExprTensor._get_expr(a.step)}" if a.step is not None else ""
            return f"{start}:{stop}{step}"
        if isinstance(a, str):
            return f"'{a}'"
        return str(a)

    def __add__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data + o_data, f"({self.expr} + {o_expr})")

    def __sub__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data - o_data, f"({self.expr} - {o_expr})")

    def __mul__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data * o_data, f"({self.expr} * {o_expr})")

    def __truediv__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data / o_data, f"({self.expr} / {o_expr})")

    def __pow__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data**o_data, f"({self.expr} ^ {o_expr})")

    def __matmul__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data @ o_data, f"({self.expr} @ {o_expr})")

    def __neg__(self):
        return ExprTensor(-self.data, f"(-{self.expr})")

    def __radd__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) + self.data, f"({o_expr} + {self.expr})")

    def __rsub__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) - self.data, f"({o_expr} - {self.expr})")

    def __rmul__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) * self.data, f"({o_expr} * {self.expr})")

    def __rtruediv__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) / self.data, f"({o_expr} / {self.expr})")

    def __rmatmul__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) @ self.data, f"({o_expr} @ {self.expr})")

    def __getitem__(self, key):
        real_key = self._unwrap(key)
        out_data = self.data[real_key]

        if isinstance(key, tuple):
            key_str = ", ".join(self._get_expr(k) for k in key)
        else:
            key_str = self._get_expr(key)

        return ExprTensor(out_data, f"{self.expr}[{key_str}]")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        real_args = tuple(cls._unwrap(a) for a in args)
        real_kwargs = {k: cls._unwrap(v) for k, v in kwargs.items()}
        out_data = func(*real_args, **real_kwargs)

        args_str = [cls._get_expr(a) for a in args]
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)
        if "Tensor." in func_name:
            func_name = func_name.split(".")[-1]

        if func_name == "linear":
            new_expr = f"Linear({args_str[0]})"
        elif func_name == "batch_norm":
            new_expr = f"BN({args_str[0]})"
        else:
            new_expr = f"{func_name}({', '.join(args_str)})"

        if isinstance(out_data, torch.Tensor):
            return ExprTensor(out_data, new_expr)
        if isinstance(out_data, tuple):
            return tuple(
                ExprTensor(d, f"{new_expr}_{i}") if isinstance(d, torch.Tensor) else d
                for i, d in enumerate(out_data)
            )
        return out_data

    def __getattr__(self, name):
        if name in ["shape", "device", "dtype", "ndim"]:
            return getattr(self.data, name)

        attr = getattr(self.data, name)
        if not callable(attr):
            return attr

        def method_wrapper(*args, **kwargs):
            real_args = tuple(self._unwrap(a) for a in args)
            real_kwargs = {k: self._unwrap(v) for k, v in kwargs.items()}
            out_data = attr(*real_args, **real_kwargs)

            args_str = [self._get_expr(a) for a in args]
            kwargs_str = [f"{k}={self._get_expr(v)}" for k, v in kwargs.items()]
            all_args = ", ".join(args_str + kwargs_str)
            new_expr = f"{self.expr}.{name}({all_args})"

            if isinstance(out_data, torch.Tensor):
                return ExprTensor(out_data, new_expr)
            if isinstance(out_data, tuple):
                return tuple(
                    ExprTensor(d, f"{new_expr}_{i}") if isinstance(d, torch.Tensor) else d
                    for i, d in enumerate(out_data)
                )
            return out_data

        return method_wrapper

    def dim(self):
        return self.data.dim()

    def size(self, *args):
        return self.data.size(*args)

    def __repr__(self):
        return f"ExprTensor(expr='{self.expr}')"


if __name__ == "__main__":
    from torch import nn

    model = nn.Sequential(
        nn.Linear(3, 2),
        nn.BatchNorm1d(2),
        nn.ReLU(),
    )
    x = ExprTensor(torch.randn(4, 3), "x")
    out = model(x)
    assert out.expr == "relu(BN(Linear(x)))"
    assert out.data.shape == (4, 2)

    reshaped = (ExprTensor(torch.arange(6.0).reshape(2, 3), "x") + 1).reshape(3, 2)
    assert reshaped.expr == "(x + 1).reshape(3, 2)"

    weight = torch.nn.Parameter(torch.randn(3, 4))
    matmul = ExprTensor(torch.randn(2, 3), "x") @ weight
    assert matmul.expr == "(x @ W)"
    assert matmul.data.shape == (2, 4)

    print("ExprTensor self-check passed.")
</file>

<file path="utils/optimize.py">
import torch.nn as nn
from typing import Any, Dict, Iterable, List, Tuple, Union


DEFAULT_NO_DECAY_KEYWORDS = (
    "position_ids",
    # Free parameters with embedding-like semantics.  These are not always
    # owned by nn.Embedding, so module-type inspection alone is insufficient.
    "seed",
    "shift_begin",
    "shift_end",
    "boundary_flag",
    "_vectors",
)

def split_params_by_decay(
        named_params: Iterable[Tuple[str, nn.Parameter]],  # 修改这里
        weight_decay: float = 0.01,
        no_decay_keywords: Union[tuple, list] = ("embedding", "position_ids")
) -> List[Dict[str, Any]]:
    """
    基于张量维度和关键字自动分组优化器参数。
    """
    decay_params = []
    no_decay_params = []

    for name, param in named_params: # 修改这里
        if not param.requires_grad:
            continue

        if param.ndim < 2 or any(nd in name.lower() for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]


def split_model_params_by_decay(
    model: nn.Module,
    weight_decay: float = 1.0e-6,
    no_decay_keywords: Union[tuple, list] = DEFAULT_NO_DECAY_KEYWORDS,
) -> List[Dict[str, Any]]:
    """Build optimizer groups while never decaying embedding-like parameters.

    Matrix/tensor weights receive ``weight_decay``. Biases, normalization
    parameters, scalar/vector parameters, parameters owned directly by an
    ``nn.Embedding`` module, and explicitly named embedding-like free
    parameters receive zero decay. A Linear nested under a container whose
    name happens to contain ``embedding`` is still decayed.

    Inspecting module ownership matters: an embedding table may be named
    ``lookup.weight`` and therefore cannot be identified reliably from the
    parameter name alone.
    """
    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    lowered_keywords = tuple(str(keyword).lower() for keyword in no_decay_keywords)
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered_name = name.lower()
        no_decay = (
            parameter.ndim < 2
            or id(parameter) in embedding_parameter_ids
            or any(keyword in lowered_name for keyword in lowered_keywords)
        )
        (no_decay_params if no_decay else decay_params).append(parameter)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups

def create_complex_optimizer_groups(
        model: nn.Module,
        decay_dict: Dict[str, float],
        default_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    根据传入的 decay_dict 灵活配置不同网络层的前缀参数衰减。

    参数示例:
    decay_dict = {
        "embeddings": 0.0,       # 推荐系统 Embedding
        "moe_router": 0.05,      # 假设 MoE 门控网络需要更强的正则化
        "bias": 0.0,             # 偏置项
        "norm": 0.0              # 归一化层
    }
    """
    # 构建一个按 weight_decay 分组的字典
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 默认衰减值
        current_decay = default_decay

        # 遍历字典进行匹配（可以根据需求换成正则 re.search）
        for key, decay_value in decay_dict.items():
            if key in name.lower():
                current_decay = decay_value
                break  # 匹配到第一个优先级最高的就跳出

        # 归入对应的 decay 组
        if current_decay not in groups:
            groups[current_decay] = []
        groups[current_decay].append(param)

    # 转换为 Optimizer 需要的格式
    return [
        {"params": params, "weight_decay": wd}
        for wd, params in groups.items()
    ]
</file>

<file path="utils/process.py">
from typing import Iterable

import psutil
ABOVE_NORMAL_PRIORITY_CLASS = psutil.ABOVE_NORMAL_PRIORITY_CLASS
NORMAL_PRIORITY_CLASS = psutil.NORMAL_PRIORITY_CLASS
HIGH_PRIORITY_CLASS = psutil.HIGH_PRIORITY_CLASS
LARGE_CORE_CPUS = tuple([i for i in range(16)])
SMALL_CORE_CPUS = tuple([i for i in range(16, 24)])
def set_priority(ps_priority=psutil.HIGH_PRIORITY_CLASS):
    import os
    # 获取当前进程
    current_process = psutil.Process(os.getpid())
    # 设置优先级（Windows 可用值：psutil.HIGH_PRIORITY_CLASS）
    current_process.nice(ps_priority)  # 等同于 "high priority"
    # 查看当前优先级
    print(current_process.nice())

def get_cpu_load_rank():
    per_cpu_percent = psutil.cpu_percent(interval=10, percpu=True)
    load_rank = sorted(enumerate(per_cpu_percent), key=lambda x: -x[-1])
    return load_rank

def set_affinity(pid, cpus):
    process = psutil.Process(pid)
    process.cpu_affinity(cpus)

def get_affinity(pid):
    process = psutil.Process(pid)
    affinity = process.cpu_affinity()
    print(f"进程可以运行的CPU核心: {affinity}")

import psutil
def get_idle_cpus(nums, groups=1, exclude_cpus: Iterable=None):
    if exclude_cpus is None:
        exclude_cpus = tuple()
    logical_cpu_count = psutil.cpu_count(logical=True)
    if nums * groups > logical_cpu_count:
        raise ValueError
    load_rank = get_cpu_load_rank()
    print("cpu负载", load_rank)
    idle_cpus = [x[0] for x in load_rank if x[0] not in exclude_cpus]
    idle_cpus = idle_cpus[-nums*groups: ]
    return [idle_cpus[i:i + nums] for i in range(0, len(idle_cpus), nums)]
</file>

<file path="utils/profile_sequence.py">
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import polars as pl
import pyarrow.parquet as pq


@dataclass
class ProfileSequenceOutputs:
    main_paths: list[Path]
    profile_paths: list[Path]
    manifest_path: Path


def _as_paths(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(path) for path in paths]


def _part_files(part: str | Path) -> list[Path]:
    part = Path(part)
    if part.is_dir():
        return sorted(part.glob("*.parquet"))
    return [part] if part.exists() else []


def _ordered_unique(fields: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(fields))


def _parquet_columns(files: Sequence[Path]) -> set[str]:
    if not files:
        return set()
    return set(pq.ParquetFile(files[0]).schema_arrow.names)


def _existing_input_paths(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(path) for path in paths if Path(path).exists()]


def _filter_existing_fields(paths: Sequence[Path], fields: Optional[Iterable[str]]) -> Optional[list[str]]:
    if fields is None:
        return None
    cols = _parquet_columns(paths)
    return [field for field in fields if field in cols]


def _main_select_fields(
    files: Sequence[Path],
    select_fields: Optional[Iterable[str]],
    seq_len_col: str,
) -> list[str]:
    source_cols = _parquet_columns(files)
    if select_fields is None:
        fields = [name for name in source_cols if not name.startswith("__bb_")]
    else:
        fields = [name for name in select_fields if name in source_cols or name == seq_len_col]
    if seq_len_col not in fields:
        fields.append(seq_len_col)
    return _ordered_unique(fields)


def build_profile_sequence_partitions(
    flat_part_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    group_name: str,
    user_col: str,
    time_col: str,
    feature_mapping: Mapping[str, str],
    seq_len_col: str,
    label_col: Optional[str] = None,
    positive_label: Any = 1,
    select_fields: Optional[Iterable[str]] = None,
    redo: bool = True,
    compression: str = "zstd",
) -> ProfileSequenceOutputs:
    """Build lightweight main parts plus row-per-user sequence profiles.

    Each input partition must contain a disjoint user set. The main output keeps
    current-sample fields and a cursor (`seq_len_col`). The profile output stores
    full positive histories for each user with aligned list columns.
    """

    output_dir = Path(output_dir)
    main_dir = output_dir / "main_parts"
    profile_dir = output_dir / "profiles" / group_name
    manifest_path = output_dir / "profile_manifest.json"

    if output_dir.exists() and redo:
        shutil.rmtree(output_dir)
    main_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    main_paths: list[Path] = []
    profile_paths: list[Path] = []
    part_dirs = _as_paths(flat_part_dirs)

    for idx, part in enumerate(part_dirs):
        files = _part_files(part)
        if not files:
            continue

        main_path = main_dir / f"{part.name}.parquet"
        profile_path = profile_dir / f"{part.name}.parquet"
        main_paths.append(main_path)
        profile_paths.append(profile_path)

        if main_path.exists() and profile_path.exists() and not redo:
            continue

        cols = _parquet_columns(files)
        missing = [col for col in [user_col, time_col, *(feature_mapping.keys())] if col not in cols]
        if label_col is not None and label_col not in cols:
            missing.append(label_col)
        if missing:
            raise ValueError(f"Missing columns in {part}: {sorted(set(missing))}")

        is_positive = (
            (pl.col(label_col) == positive_label).cast(pl.UInt32)
            if label_col is not None
            else pl.lit(1, dtype=pl.UInt32)
        )

        ordered_lf = (
            pl.scan_parquet([str(path) for path in files])
            .with_row_index("__bb_row_nr")
            .sort([user_col, time_col, "__bb_row_nr"])
            .with_columns(is_positive.alias("__bb_is_positive"))
            .with_columns(
                (
                    pl.col("__bb_is_positive").cum_sum().over(user_col)
                    - pl.col("__bb_is_positive")
                )
                .cast(pl.UInt32)
                .alias(seq_len_col)
            )
        )

        profile_exprs = [pl.col(src_col) for src_col in feature_mapping.keys()]
        (
            ordered_lf
            .filter(pl.col("__bb_is_positive") == 1)
            .group_by(user_col, maintain_order=True)
            .agg(profile_exprs)
            .rename(dict(feature_mapping))
            .sink_parquet(profile_path, compression=compression)
        )

        main_fields = _main_select_fields(files, select_fields, seq_len_col)
        (
            ordered_lf
            .select(main_fields)
            .sink_parquet(main_path, compression=compression)
        )
        print(f"    -> profile seq part {idx + 1}/{len(part_dirs)}: {main_path}")

    manifest = {
        "group_name": group_name,
        "user_col": user_col,
        "time_col": time_col,
        "seq_len_col": seq_len_col,
        "label_col": label_col,
        "positive_label": positive_label,
        "feature_mapping": dict(feature_mapping),
        "main_paths": [str(path) for path in main_paths],
        "profile_paths": [str(path) for path in profile_paths],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return ProfileSequenceOutputs(
        main_paths=main_paths,
        profile_paths=profile_paths,
        manifest_path=manifest_path,
    )


def write_filtered_parquet(
    input_paths: Iterable[str | Path],
    out_path: str | Path,
    *,
    fields: Iterable[str],
    filter_expr: Optional[pl.Expr] = None,
    compression: str = "zstd",
    redo: bool = True,
) -> Path:
    out_path = Path(out_path)
    if out_path.exists() and not redo:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = _existing_input_paths(input_paths)
    selected_fields = _filter_existing_fields(paths, fields) or []
    lf = pl.scan_parquet([str(path) for path in paths])
    if filter_expr is not None:
        lf = lf.filter(filter_expr)
    lf.select(selected_fields).sink_parquet(out_path, compression=compression)
    return out_path
</file>

<file path="utils/recorder.py">
"""
Legacy compatibility layer for training-time tensor monitoring.

The real implementation now lives in `rosen.utils.observatory`.
Models can keep importing `ExplicitFeatureRecorder` from this module,
while new development should prefer the observatory package directly.
"""

from rosen.utils.observatory import (
    IndividualReLURecorder,
    RelationOptions,
    TensorDisplayConfig,
    TensorMonitorOptions,
    TensorObservatory,
    TensorSketchConfig,
)


class ExplicitFeatureRecorder(TensorObservatory):
    pass


__all__ = [
    "ExplicitFeatureRecorder",
    "IndividualReLURecorder",
    "RelationOptions",
    "TensorDisplayConfig",
    "TensorMonitorOptions",
    "TensorObservatory",
    "TensorSketchConfig",
]
</file>

<file path="utils/sample.py">
from collections import Counter
from typing import Literal, Dict, Tuple, Union

import torch
import numpy as np


class AbstractSampler(object):
    """:class:`AbstractSampler` is a abstract class, all sampler should inherit from it. This sampler supports returning
    a certain number of random value_ids according to the input key_id, and it also supports to prohibit
    certain key-value pairs by setting used_ids.

    Args:
        distribution (str): The string of distribution, which is used for subclass.

    Attributes:
        used_ids (numpy.ndarray): The result of :meth:`get_used_ids`.
    """

    def __init__(self, distribution, alpha):
        self.distribution = ""
        self.alpha = alpha
        self.set_distribution(distribution)
        self.used_ids = self.get_used_ids()

    def set_distribution(self, distribution):
        """Set the distribution of sampler.

        Args:
            distribution (str): Distribution of the negative items.
        """
        self.distribution = distribution
        if distribution == "popularity":
            self._build_alias_table()

    def _uni_sampling(self, sample_num):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_uni_sampling] should be implemented")

    def _get_candidates_list(self):
        """Get sample candidates list for _pop_sampling()

        Returns:
            candidates_list (list): a list of candidates id.
        """
        raise NotImplementedError("Method [_get_candidates_list] should be implemented")

    def _build_alias_table(self):
        """Build alias table for popularity_biased sampling."""
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))
        self.alias = self.prob.copy()
        large_q = []
        small_q = []
        for i in self.prob:
            self.alias[i] = -1
            self.prob[i] = self.prob[i] / len(candidates_list)
            self.prob[i] = pow(self.prob[i], self.alpha)
        normalize_count = sum(self.prob.values())
        for i in self.prob:
            self.prob[i] = self.prob[i] / normalize_count * len(self.prob)
            if self.prob[i] > 1:
                large_q.append(i)
            elif self.prob[i] < 1:
                small_q.append(i)
        while len(large_q) != 0 and len(small_q) != 0:
            l = large_q.pop(0)
            s = small_q.pop(0)
            self.alias[s] = l
            self.prob[l] = self.prob[l] - (1 - self.prob[s])
            if self.prob[l] < 1:
                small_q.append(l)
            elif self.prob[l] > 1:
                large_q.append(l)

    def _pop_sampling(self, sample_num):
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """

        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)
        random_prob_list = np.random.random(sample_num)
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])

        return np.array(final_random_list)

    def sampling(self, sample_num):
        """Sampling [sample_num] item_ids.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        if self.distribution == "uniform":
            return self._uni_sampling(sample_num)
        elif self.distribution == "popularity":
            return self._pop_sampling(sample_num)
        else:
            raise NotImplementedError(
                f"The sampling distribution [{self.distribution}] is not implemented."
            )

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.
        """
        raise NotImplementedError("Method [get_used_ids] should be implemented")

    def sample_by_key_ids(self, key_ids, num):
        """Sampling by key_ids.

        Args:
            key_ids (numpy.ndarray or list): Input key_ids.
            num (int): Number of sampled value_ids for each key_id.

        Returns:
            torch.tensor: Sampled value_ids.
            value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
            is sampled for key_ids[0];
            value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
            value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
        """
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list))
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list))
                check_list = np.array(
                    [
                        i
                        for i, used, v in zip(
                            check_list,
                            self.used_ids[key_ids[check_list]],
                            value_ids[check_list],
                        )
                        if v in used
                    ]
                )
        return torch.tensor(value_ids, dtype=torch.long)

import polars as pl


class PolarsUISampler(AbstractSampler):
    def __init__(
            self,
            whole_lf: pl.LazyFrame,
            user_col: str,
            item_col: str,
            distribution="uniform",
            alpha=1.0
    ):
        # ==========================================
        # 核心优化 1：只执行一次 collect() 进行物化！
        # 将这两列拉入内存，变成 eager DataFrame，后续全部零拷贝复用
        # ==========================================
        self.ui_df = (
            whole_lf.select(
                pl.col(user_col).alias("user_id"),
                pl.col(item_col).alias("item_id")
            )
            .collect()  # <--- 注意：这里变成了 collect()
        )

        # ==========================================
        # 核心优化 2：所有的操作全部基于内存中的 self.ui_df，快如闪电
        # ==========================================

        # 1. 构建候选池：因为数据已经在内存里，.unique().to_numpy() 是毫秒级操作
        self.item_pool = (
            self.ui_df.select("item_id")
            .unique()
            .drop_nulls()
            .filter(pl.col("item_id") > 0)  # 剔除 Padding
            .to_series()
            .to_numpy()
        )

        self.pool_size = len(self.item_pool)
        if self.pool_size == 0:
            raise ValueError("Item pool is empty! Please check your dataset.")

        # 调用父类初始化
        super().__init__(distribution=distribution, alpha=alpha)

    def _uni_sampling(self, sample_num):
        """非连续集合的均匀分布采样"""
        return np.random.choice(self.item_pool, size=sample_num, replace=True)

    def _get_candidates_list(self):
        """获取流行度候选列表"""
        # 直接复用内存里的列，无需 collect
        return (
            self.ui_df.select("item_id")
            .filter(pl.col("item_id") > 0)
            .to_series()
            .to_list()
        )

    def get_used_ids(self):
        """
        获取用户历史交互集合：
        通过 .lazy() 重新进入惰性模式以利用 Polars 的多线程引擎加速 Join
        """
        # 1. 极速获取 max_user_id
        max_u_res = self.ui_df.select(pl.col("user_id").max()).item()
        if max_u_res is None:
            return np.empty(0, dtype=object)

        max_u = int(max_u_res)
        all_users = pl.LazyFrame({"user_id": np.arange(max_u + 1)})

        # 2. 从 eager 重新转为 lazy 执行图，利用向量化引擎极速 join 和 agg
        # 这里的 join 是完全在内存中发生的，不涉及磁盘 IO
        final_df = (
            all_users
            .join(
                self.ui_df.lazy().group_by("user_id").agg(pl.col("item_id").unique().alias("items")),
                on="user_id",
                how="left"
            )
            .with_columns(pl.col("items").fill_null([]))
            .sort("user_id")
            .collect()
        )

        used_ids = np.empty(max_u + 1, dtype=object)
        used_ids[:] = [set(lst) for lst in final_df["items"].to_list()]
        return used_ids
    def sample_by_key_ids(self, key_ids, num, format: Literal['flat', 'listwise']="flat"):
        """
        Args:
            format (str): 输出格式。
                          'flat' 保持一维张量交错输出 [n0_1, n1_1, n0_2, n1_2]；
                          'listwise' 输出形状为 (batch_size, num) 的矩阵。
        """
        # 1. 调取原生的扁平化结果
        raw_values = super().sample_by_key_ids(key_ids, num)

        # 2. 根据要求进行后处理
        if format == "flat":
            return raw_values
        elif format == "listwise":
            batch_size = len(key_ids)
            return raw_values.view(num, batch_size).t().contiguous()
        else:
            raise ValueError(f"不支持的输出格式: {format}")


def extract_history_dict(
        *lfs: pl.LazyFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        merge: bool = False
) -> Union[Dict[int, list], Tuple[Dict[int, list], ...]]:
    """
    极速从任意数量的 Polars LazyFrame 中提取 User-History 字典。

    Args:
        *lfs: 任意数量的 pl.LazyFrame (例如 train_lf, valid_lf, test_lf)
        user_col: 用户列的列名
        item_col: 物品列的列名
        merge: 是否将所有输入的 LazyFrame 合并成一个全局静态大字典。
               True -> 返回单个 Dict
               False -> 返回 Tuple[Dict, ...], 与输入的 lf 数量一一对应
    Returns:
        history_dict 或者 (history_dict1, history_dict2, ...)
    """
    # 核心聚合算子：在 Lazy 模式下定义图计算逻辑
    def _build_dict_from_lf(lf: pl.LazyFrame) -> Dict:
        lf_ui = lf.select([user_col, item_col])
        df_agg = lf_ui.group_by(user_col, maintain_order=False).agg(
            pl.col(item_col)
        ).collect()  # 这里才真正触发计算，返回 DataFrame
        return dict(zip(df_agg[user_col].to_list(), df_agg[item_col].to_list()))

    # ================= 分支 1：合并计算 =================
    if merge:
        if not lfs:
            return {}
        selected_lfs = [lf.select([user_col, item_col]) for lf in lfs]
        merged_lf = pl.concat(selected_lfs)
        return _build_dict_from_lf(merged_lf)
    else:
        # 直接利用列表推导式，依次处理返回 Tuple
        return tuple(_build_dict_from_lf(lf) for lf in lfs)
import time

# 测试代码
if __name__ == "__main__":
    # 1. 模拟推荐系统交互日志
    num_items = 20  # 物品池大小：0 ~ 19

    # 模拟交互序列：
    # User 0: 看了 1, 2
    # User 1: 看了 5, 5, 6 (重复点击)
    # User 2: 完全没有记录 (测试索引断层)
    # User 3: 看了 10
    # User 4: 看了 15, 16, 17
    mock_uids = [0, 0, 1, 1, 1, 3, 4, 4, 4]
    mock_iids = [1, 2, 5, 5, 6, 10, 15, 16, 17]

    # 构建 LazyFrame
    user_lf = pl.LazyFrame({"user_id": mock_uids})
    item_lf = pl.LazyFrame({"item_id": mock_iids})
    whole_lf = pl.concat([user_lf, item_lf], how="horizontal")

    print("🚀 初始化 PolarsUISampler...")
    start_time = time.time()

    # 初始化会自动触发 get_used_ids 的底层 Polars 计算图
    sampler = PolarsUISampler(
        whole_lf=whole_lf,
        user_col="user_id",
        item_col="item_id",
        distribution="uniform"
    )
    print(f"✅ 初始化完成，耗时: {time.time() - start_time:.4f} 秒\n")

    # 测试 1: 检查 used_ids 结构是否完美对齐
    print("--- 测试 1: 检查底层解析的 used_ids ---")
    for uid, items in enumerate(sampler.used_ids):
        print(f"User {uid:2d} 历史交互: {items}")

    print("\n--- 测试 2: 触发负采样 ---")
    # 假设构建 Batch，要为 user_id=0 和 user_id=1 各采样 4 个负样本
    target_users = [0, 1]
    num_neg_per_user = 4

    print(f"目标请求: 为用户 {target_users} 各采样 {num_neg_per_user} 个样本")

    # 调用继承自父类的方法，它内部会循环检查并避开 used_ids
    sampled_negatives = sampler.sample_by_key_ids(key_ids=target_users, num=num_neg_per_user)

    # 父类返回的 tensor 是交错平铺的：
    # [u0_n1, u1_n1, u0_n2, u1_n2, ...]
    neg_array = sampled_negatives.numpy()

    print(f"\n原始输出 Tensor 形状: {sampled_negatives.shape}")
    print(f"原始输出 Tensor 数据: {neg_array}")

    # 还原一下视角，看看每个用户到底分到了什么：
    u0_samples = neg_array[0::2]  # 从索引 0 开始，步长 2
    u1_samples = neg_array[1::2]  # 从索引 1 开始，步长 2

    print(f"\n检查 User 0 采到的样本: {u0_samples} (绝对不会包含 {sampler.used_ids[0]})")
    print(f"检查 User 1 采到的样本: {u1_samples} (绝对不会包含 {sampler.used_ids[1]})")
</file>

<file path="utils/sequential.py">
from typing import Union, Tuple, Dict

import polars as pl
from typing import Dict, Optional, Any


def extract_history_sequences(
        lf: pl.LazyFrame,
        max_seq_len: int,
        user_col: str,
        time_col: str,
        feature_mapping: Dict[str, str],
        seq_len_col: str = "seq_len",
        label_col: Optional[str] = None,
        positive_label: Any = 1
) -> pl.LazyFrame:
    """
    严格提取用户历史正向行为序列。避免负样本稀释，且绝对保证无数据泄漏（防穿越）。
    """
    if feature_mapping is None:
        feature_mapping = {"item_id": "item_seq"}

    # 1. 确保全局严格排序，并分配全局连续的物理行号作为“逻辑时钟”
    lf = lf.sort([user_col, time_col])
    lf = lf.with_row_index("_global_row_idx")

    # 2. 剥离出纯正样本的 DataFrame (继承了主表的 _global_row_idx)
    if label_col is not None:
        pos_lf = lf.filter(pl.col(label_col) == positive_label)
    else:
        pos_lf = lf

    # 在纯正样本中，为每个用户打上局部的连续序号 (0, 1, 2...)
    pos_lf = pos_lf.with_columns(
        pl.int_range(0, pl.len(), dtype=pl.UInt32).over(user_col).alias("_pos_idx")
    )

    # 3. 在纯正样本上进行严格的 N 长度开窗
    # 因为表中全都是正样本，所以开窗 max_seq_len 就能拿满 N 个正交互
    agg_exprs = [
        pl.col(src).alias(tgt_seq_name)
        for src, tgt_seq_name in feature_mapping.items()
    ]

    pos_rolling = pos_lf.rolling(
        index_column="_pos_idx",
        group_by=user_col,
        period=f"{max_seq_len}i"
    ).agg(agg_exprs)

    # 将生成的纯正序列和原有的“逻辑时钟”拼起来，且重命名避免冲突
    # pos_state 代表：在这个 _state_row_idx 时刻【之后】，用户的正向历史状态更新为了什么
    pos_state = pos_lf.select([user_col, "_global_row_idx", "_pos_idx"]).join(
        pos_rolling, on=[user_col, "_pos_idx"]
    ).rename({"_global_row_idx": "_state_row_idx"}).drop("_pos_idx")

    # 【关键修复】: 将 state_row_idx 转为有符号整数并严格排序
    pos_state = pos_state.with_columns(
        pl.col("_state_row_idx").cast(pl.Int64)
    ).sort("_state_row_idx")

    # 4. ASOF JOIN 拼回主表 (核心防泄漏机制)
    # 对于主表的每一行，我们要找的是严格发生在它【之前】的最新正向状态
    # 【关键修复】: 左表的 join_idx = 当前行号 - 1，必须先 cast 为 Int64 避免无符号 0 - 1 下溢
    lf = lf.with_columns(
        (pl.col("_global_row_idx").cast(pl.Int64) - 1).alias("_join_idx")
    )

    lf = lf.join_asof(
        pos_state,
        left_on="_join_idx",
        right_on="_state_row_idx",
        by=user_col,
        strategy="backward"  # 寻找 <= _join_idx 的最新状态
    )

    # 5. 清理与填补
    first_seq_col = list(feature_mapping.values())[0]
    drop_cols = ["_global_row_idx", "_join_idx", "_state_row_idx"]

    # 将匹配不到历史状态的 Null 填充为空列表 []
    fill_exprs = [
        pl.col(tgt).fill_null([]) for tgt in feature_mapping.values()
    ]

    lf = (
        lf.with_columns(fill_exprs)
        .with_columns(
            pl.col(first_seq_col).list.len().fill_null(0).cast(pl.UInt32).alias(seq_len_col)
        )
        .drop(drop_cols)
    )

    return lf

def extract_seq_len(
        lf: pl.LazyFrame,
        seq_col: str,
        seq_len_col: str = "seq_len"
) -> pl.LazyFrame:
    return lf.with_columns(
        pl.col(seq_col)
        .list.len()
        .fill_null(0)
        .cast(pl.UInt32)
        .alias(seq_len_col)
    )

def extract_history_items(
        lf: pl.LazyFrame,
        max_seq_len: int,
        user_col: str,
        time_col: str,
        item_col: str,
        seq_col: str = "items_seq",
        seq_len_col: str = "seq_len",
        label_col: Optional[str] = None,
        positive_label: Any = 1
) -> pl.LazyFrame:
    return extract_history_sequences(
        lf, max_seq_len, user_col, time_col,
        {item_col: seq_col}, seq_len_col,
        label_col, positive_label
    )




if __name__ == "__main__":
    def main():
        # 1. 构造模拟数据：两个用户，交易时间乱序（测试排序逻辑）
        data = {
            "user_id": [1, 2, 1, 1, 2, 1, 2],
            "timestamp": [100, 102, 101, 103, 200, 202, 201],
            "video_id": ["v_A", "v_C", "v_B", "v_D", "v_E", "v_G", "v_F"],
            "label": [0, 0, 1, 1, 1, 1, 1]
        }

        lf = pl.LazyFrame(data)

        # 2. 设置最大序列长度
        MAX_SEQ_LEN = 3

        print(f"--- 原始输入 (LazyFrame) ---")
        print(lf.collect())

        # 3. 调用你的提取函数
        # 逻辑预期：对于每个用户，当前行的 items_seq 应该是其【之前】的最多 3 个 video_id
        result_lf = extract_history_sequences(
            lf=lf,
            max_seq_len=MAX_SEQ_LEN,
            user_col="user_id",
            time_col="timestamp",
            label_col="label",
            feature_mapping={
                "video_id": "history_list"
            }
        )

        # 4. 执行并打印结果
        df_result = result_lf.collect()

        print(f"\n--- 处理后的结果 (max_seq_len={MAX_SEQ_LEN}) ---")
        # 按照用户和时间打印，方便观察序列增长
        print(df_result.sort(["user_id", "timestamp"]))

        # 5. 验证核心逻辑点
        print("\n--- 逻辑校验 ---")
        # 校验用户1的最后一行：时间103，前三行应为 [v_A, v_B, v_C]
        last_user1 = df_result.filter((pl.col("user_id") == 1) & (pl.col("timestamp") == 103))
        print(f"用户1(T=103) 的历史序列: {last_user1.get_column('history_list').to_list()}")

        # 校验用户2的第一行：历史序列应该是空的 []
        first_user2 = df_result.filter((pl.col("user_id") == 2) & (pl.col("timestamp") == 200))
        print(f"用户2(T=200) 的历史序列: {first_user2.get_column('history_list').to_list()}")
    main()
</file>

<file path="utils/shuffle.py">
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, Literal, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def _resolve_input_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidates = [Path(paths)]
    else:
        candidates = [Path(path) for path in paths]

    resolved: list[Path] = []
    for path in candidates:
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.parquet")))
        elif path.exists():
            resolved.append(path)

    if not resolved:
        raise FileNotFoundError("No input parquet files found for shuffle checkpoint.")
    return resolved


def _parquet_columns(paths: list[Path]) -> list[str]:
    return pq.ParquetFile(paths[0]).schema_arrow.names


def _filter_existing_fields(paths: list[Path], fields: Optional[Iterable[str]]) -> Optional[list[str]]:
    if fields is None:
        return None
    existing = set(_parquet_columns(paths))
    return [field for field in fields if field in existing]


def _empty_table(paths: list[Path], fields: Optional[list[str]]) -> pa.Table:
    schema = pq.ParquetFile(paths[0]).schema_arrow
    if fields is not None:
        schema = pa.schema([field for field in schema if field.name in fields])
    arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _append_parquet_batches(
    input_paths: list[Path],
    out_path: Path,
    *,
    batch_size: int,
    compression: str,
    rand_col: str,
    shuffle_rows: bool,
    seed: int,
) -> None:
    writer = None
    wrote_rows = False
    try:
        chunk_idx = 0
        for path in input_paths:
            parquet_file = pq.ParquetFile(path)
            for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
                df = pl.from_arrow(batch)
                if df.height == 0:
                    continue
                if shuffle_rows:
                    row_seed = (int(seed) + chunk_idx * 1000003 + batch_idx * 9176) & 0xFFFFFFFF
                    df = df.sample(fraction=1.0, shuffle=True, seed=row_seed)
                if rand_col in df.columns:
                    df = df.drop(rand_col)
                table = df.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
                writer.write_table(table)
                wrote_rows = True
            chunk_idx += 1
    finally:
        if writer is not None:
            writer.close()

    if not wrote_rows:
        table = _empty_table(input_paths, None)
        pq.write_table(table, out_path, compression=compression)


class _ParquetWriterGroup:
    def __init__(self, compression: str):
        self.compression = compression
        self._writers: dict[int, pq.ParquetWriter] = {}

    def write(self, key: int, path: Path, df: pl.DataFrame) -> int:
        if df.height == 0:
            return 0
        table = df.to_arrow()
        if key not in self._writers:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writers[key] = pq.ParquetWriter(str(path), table.schema, compression=self.compression)
        self._writers[key].write_table(table)
        return int(df.height)

    def close(self) -> None:
        first_error = None
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                first_error = first_error or exc
        self._writers.clear()
        if first_error is not None:
            raise first_error


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _hash_range_target_expr(rand_col: str, target_count: int) -> pl.Expr:
    if target_count == 1:
        return pl.lit(0, dtype=pl.UInt32)
    if _is_power_of_two(target_count):
        shift = 64 - (target_count.bit_length() - 1)
        step = 1 << shift
        return (pl.col(rand_col) // step).cast(pl.UInt32)
    return (
        ((pl.col(rand_col).cast(pl.Float64) / float(1 << 64)) * target_count)
        .floor()
        .clip(0, target_count - 1)
        .cast(pl.UInt32)
    )


def _write_target_sort_shuffled_parquet(
    input_paths: list[Path],
    out_path: Path,
    *,
    selected_fields: Optional[list[str]],
    filter_expr: Optional[pl.Expr],
    seed: int,
    target_count: int,
    batch_size: int,
    compression: str,
    temp_dir: str | Path | None,
) -> None:
    """External random-key shuffle.

    Rows are first range-partitioned by the high bits of a stable row hash, then
    each target is sorted by the full hash. Concatenating targets in target id
    order is equivalent to sorting the full dataset by the random key, without
    materializing the full table in memory.
    """

    rand_col = "__bb_shuffle_id"
    target_col = "__bb_shuffle_target"
    tmp_cols = ["__bb_file_idx", "__bb_row_idx", rand_col, target_col]
    existing_cols = set(_parquet_columns(input_paths))
    conflict_cols = sorted(existing_cols.intersection(tmp_cols))
    if conflict_cols:
        raise ValueError(f"Input parquet contains reserved shuffle columns: {conflict_cols}")

    tmp_dir = Path(temp_dir) if temp_dir is not None else out_path.parent / f"_{out_path.stem}_shuffle_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    staging_root = tmp_dir / "stage"
    final_root = tmp_dir / "final"
    staging_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    staging_paths = [
        staging_root / f"part-{target_id:05d}.parquet"
        for target_id in range(target_count)
    ]
    read_columns = None if filter_expr is not None else selected_fields
    writers = _ParquetWriterGroup(compression=compression)

    try:
        for file_idx, path in enumerate(input_paths):
            row_offset = 0
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
                df = pl.from_arrow(batch)
                if filter_expr is not None:
                    df = df.filter(filter_expr)
                    if df.height == 0:
                        row_offset += batch.num_rows
                        continue
                    if selected_fields is not None:
                        df = df.select(selected_fields)
                elif selected_fields is not None:
                    df = df.select(selected_fields)

                if df.height > 0:
                    df = (
                        df
                        .with_row_index("__bb_row_idx", offset=row_offset)
                        .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
                        .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias(rand_col))
                        .with_columns(_hash_range_target_expr(rand_col, target_count).alias(target_col))
                    )

                    for key, target_df in df.partition_by(
                        target_col,
                        maintain_order=False,
                        include_key=True,
                        as_dict=True,
                    ).items():
                        target_id = int(key[0] if isinstance(key, tuple) else key)
                        writers.write(target_id, staging_paths[target_id], target_df)

                row_offset += batch.num_rows
            print(f"    -> shuffle targetize {file_idx + 1}/{len(input_paths)}")
    finally:
        writers.close()

    try:
        target_paths: list[Path] = []
        for target_id, staging_path in enumerate(staging_paths):
            if not staging_path.exists():
                continue
            final_path = final_root / f"part-{target_id:05d}.parquet"
            (
                pl.scan_parquet(staging_path)
                .sort(rand_col)
                .drop(tmp_cols)
                .sink_parquet(final_path, compression=compression)
            )
            target_paths.append(final_path)

        if out_path.exists():
            out_path.unlink()
        if target_paths:
            _append_parquet_batches(
                target_paths,
                out_path,
                batch_size=batch_size,
                compression=compression,
                rand_col=rand_col,
                shuffle_rows=False,
                seed=seed,
            )
        else:
            pq.write_table(_empty_table(input_paths, selected_fields), out_path, compression=compression)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def write_shuffled_parquet(
    input_paths: str | Path | Iterable[str | Path],
    out_path: str | Path,
    *,
    fields: Optional[Iterable[str]] = None,
    filter_expr: Optional[pl.Expr] = None,
    seed: int = 2026,
    bucket_count: int = 64,
    batch_size: int = 65536,
    compression: str = "zstd",
    redo: bool = True,
    exact: bool = False,
    mode: Literal["target_sort", "bucket_chunk"] = "target_sort",
    temp_dir: str | Path | None = None,
) -> Path:
    """Write a physically shuffled parquet checkpoint.

    The default target_sort mode performs an external random-key sort by
    range-partitioning rows into power-of-two targets, sorting each target by
    the stable row hash, and concatenating targets in key-range order.
    Set mode="bucket_chunk" to use the older low-memory chunk shuffle.
    Set exact=True only when a single in-memory/logical full sort is required.
    """

    paths = _resolve_input_paths(input_paths)
    out_path = Path(out_path)
    if out_path.exists() and not redo:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected_fields = _filter_existing_fields(paths, fields)
    rand_col = "__bb_rand"

    if exact:
        lf = pl.scan_parquet([str(path) for path in paths])
        if filter_expr is not None:
            lf = lf.filter(filter_expr)
        if selected_fields is not None:
            lf = lf.select(selected_fields)
        (
            lf
            .with_row_index("__bb_row_idx")
            .with_columns(pl.struct(["__bb_row_idx"]).hash(seed=seed).alias(rand_col))
            .sort(rand_col)
            .drop(["__bb_row_idx", rand_col])
            .sink_parquet(out_path, compression=compression)
        )
        return out_path

    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if mode not in {"target_sort", "bucket_chunk"}:
        raise ValueError("mode must be either 'target_sort' or 'bucket_chunk'.")

    if mode == "target_sort":
        _write_target_sort_shuffled_parquet(
            paths,
            out_path,
            selected_fields=selected_fields,
            filter_expr=filter_expr,
            seed=seed,
            target_count=bucket_count,
            batch_size=batch_size,
            compression=compression,
            temp_dir=temp_dir,
        )
        return out_path

    # TODO: This bucket/chunk path is not equivalent to a true global shuffle.
    # If upstream files are sorted by user/time and downstream training reads
    # output shards sequentially, large local order blocks can remain. Add a
    # round-robin/random shard reader or make exact hash-sort the default for
    # no-sequence datasets where the memory cost is acceptable.
    tmp_dir = Path(temp_dir) if temp_dir is not None else out_path.parent / f"_{out_path.stem}_shuffle_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    bucket_root = tmp_dir / "buckets"
    bucket_root.mkdir(parents=True, exist_ok=True)
    for bucket_id in range(bucket_count):
        (bucket_root / f"b{bucket_id:03d}").mkdir(parents=True, exist_ok=True)

    file_counts = {bucket_id: 0 for bucket_id in range(bucket_count)}
    read_columns = None if filter_expr is not None else selected_fields

    try:
        for file_idx, path in enumerate(paths):
            row_offset = 0
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
                df = pl.from_arrow(batch)
                if filter_expr is not None:
                    df = df.filter(filter_expr)
                    if df.height == 0:
                        row_offset += batch.num_rows
                        continue
                    if selected_fields is not None:
                        df = df.select(selected_fields)

                df = (
                    df
                    .with_row_index("__bb_row_idx", offset=row_offset)
                    .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
                    .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias(rand_col))
                    .with_columns((pl.col(rand_col) % bucket_count).cast(pl.UInt32).alias("__bb_bucket"))
                )

                for key, bucket_df in df.partition_by(
                    "__bb_bucket",
                    maintain_order=False,
                    include_key=True,
                    as_dict=True,
                ).items():
                    bucket_id = int(key[0] if isinstance(key, tuple) else key)
                    chunk_path = (
                        bucket_root
                        / f"b{bucket_id:03d}"
                        / f"c{file_counts[bucket_id]:06d}.parquet"
                    )
                    (
                        bucket_df
                        .drop(["__bb_bucket", "__bb_file_idx", "__bb_row_idx"])
                        .write_parquet(chunk_path, compression=compression)
                    )
                    file_counts[bucket_id] += 1

                row_offset += batch.num_rows
            print(f"    -> shuffle bucketize {file_idx + 1}/{len(paths)}")

        rng = random.Random(seed)
        bucket_order = list(range(bucket_count))
        rng.shuffle(bucket_order)

        shuffled_chunks: list[Path] = []
        for bucket_id in bucket_order:
            bucket_files = sorted((bucket_root / f"b{bucket_id:03d}").glob("*.parquet"))
            rng.shuffle(bucket_files)
            shuffled_chunks.extend(bucket_files)

        if out_path.exists():
            out_path.unlink()
        if shuffled_chunks:
            _append_parquet_batches(
                shuffled_chunks,
                out_path,
                batch_size=batch_size,
                compression=compression,
                rand_col=rand_col,
                shuffle_rows=True,
                seed=seed,
            )
        else:
            pq.write_table(_empty_table(paths, selected_fields), out_path, compression=compression)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_path


def shuffle_parquet_files(*args, **kwargs) -> Path:
    return write_shuffled_parquet(*args, **kwargs)
</file>

<file path="utils/sort.py">
import os
import shutil

import duckdb
from typing import Union, List
from pathlib import Path


def duckdb_sort_parquet(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        sort_by: Union[str, List[str]],
        descending: Union[bool, List[bool]] = False,
        memory_limit: str = "84GB",  # 物理防爆盾：强制最高内存限制
        temp_dir: str = "./workspace/tmp"  # 磁盘防爆盾：防止把系统 C 盘写满
):
    input_path = str(input_path)
    output_path = str(output_path)

    # 确保安全的临时目录存在
    os.makedirs(temp_dir, exist_ok=True)

    # 统一转换 sort_by 和 descending 为列表，对齐 Polars 体验
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    if isinstance(descending, bool):
        descending = [descending] * len(sort_by)
    elif len(descending) != len(sort_by):
        raise ValueError("descending 的长度必须与 sort_by 一致")

    # 构建 SQL 的 ORDER BY 语句
    order_clauses = []
    for col, desc in zip(sort_by, descending):
        direction = "DESC" if desc else "ASC"
        order_clauses.append(f'"{col}" {direction}')
    order_by_str = ", ".join(order_clauses)

    print(f"\n[DuckDB] 🚄 启动核外大表物理排序...")
    print(f"    -> 输入: {input_path}")
    print(f"    -> 输出: {output_path}")
    print(f"    -> 规则: ORDER BY {order_by_str}")
    print(f"    -> 资源限制: 内存不超过 {memory_limit}, 临时缓冲: {temp_dir}")

    # 建立 DuckDB 连接并注入资源防爆指令
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")

    # 执行查询并流式落盘，开启高压缩比
    query = f"""
        COPY (
            SELECT * FROM read_parquet('{input_path}')
            ORDER BY {order_by_str}
        ) TO '{output_path}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD');
    """
    con.execute(query)
    con.close()

def sort_parquet_inplace(
        file_path: Union[str, Path],
        sort_by: Union[str, List[str]],
        descending: Union[bool, List[bool]] = False,
        memory_limit: str = "64GB",
        temp_dir = "./tmp"
):
    """
    使用 DuckDB 对 Parquet 文件进行极低内存的“原地排序”。
    机制：将排序结果输出到 ./tmp 目录下，完成后直接覆盖原文件，绝对安全。
    """
    file_path = Path(file_path).absolute()
    if not file_path.exists():
        raise FileNotFoundError(f"找不到目标文件: {file_path}")

    # 创建安全的临时目录
    temp_dir_path = Path(temp_dir).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    temp_output_path = temp_dir_path / f"sorted_{file_path.name}"

    # 统一转换 sort_by 和 descending 为列表
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    if isinstance(descending, bool):
        descending = [descending] * len(sort_by)
    elif len(descending) != len(sort_by):
        raise ValueError("descending 的长度必须与 sort_by 一致")

    # 构建 SQL 的 ORDER BY 语句
    order_clauses = [
        f'"{col}" {"DESC" if desc else "ASC"}'
        for col, desc in zip(sort_by, descending)
    ]
    order_by_str = ", ".join(order_clauses)

    print(f"\n[DuckDB] 🚄 启动大表原地物理排序 (In-place Sort)...")
    print(f"    -> 目标文件: {file_path}")
    print(f"    -> 排序规则: ORDER BY {order_by_str}")
    print(f"    -> 资源限制: 内存 <= {memory_limit}, 临时引擎目录: {temp_dir_path}")

    try:
        # 建立 DuckDB 连接并注入资源限制
        con = duckdb.connect()
        # 注意：DuckDB 接收路径最好用 POSIX 格式 (特别是 Windows 下)，避免转义坑
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        con.execute(f"PRAGMA temp_directory='{temp_dir_path.as_posix()}'")

        # 执行查询并流式落盘到 temp_output_path
        query = f"""
            COPY (
                SELECT * FROM read_parquet('{file_path.as_posix()}')
                ORDER BY {order_by_str}
            ) TO '{temp_output_path.as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD');
        """
        con.execute(query)
        con.close()

        # ✨ 核心：安全地替换原文件
        # 使用 shutil.move 可以跨磁盘移动，并且覆盖原文件
        shutil.move(str(temp_output_path), str(file_path))
        print(f"[DuckDB] ✅ 排序完成！已成功覆写原文件: {file_path.name}\n")

    except Exception as e:
        print(f"\n[DuckDB] ❌ 排序过程中发生错误: {e}")
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise e
</file>

<file path="utils/split_shuffle.py">
from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class SplitBucketedOutputs:
    train_dir: Path
    valid_path: Path
    test_path: Path
    train_paths: list[Path]
    row_counts: dict[str, int]
    manifest_path: Path


def _resolve_parquet_paths(paths: Iterable[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.parquet")))
        elif path.exists():
            resolved.append(path)
    if not resolved:
        raise FileNotFoundError("No input parquet files found.")
    return resolved


def _parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _select_existing_fields(paths: list[Path], fields: Optional[Iterable[str]]) -> Optional[list[str]]:
    if fields is None:
        return None
    existing = set(_parquet_columns(paths[0]))
    return [field for field in fields if field in existing]


def _count_parquet_rows(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        total += int(pq.ParquetFile(path).metadata.num_rows)
    return total


class _ParquetWriterGroup:
    def __init__(self, compression: str):
        self.compression = compression
        self._writers: dict[str, pq.ParquetWriter] = {}

    def write(self, key: str, path: Path, df: pl.DataFrame) -> int:
        if df.height == 0:
            return 0
        table = df.to_arrow()
        if key not in self._writers:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writers[key] = pq.ParquetWriter(
                str(path),
                table.schema,
                compression=self.compression,
            )
        self._writers[key].write_table(table)
        return int(df.height)

    def close(self) -> None:
        first_error = None
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                first_error = first_error or exc
        self._writers.clear()
        if first_error is not None:
            raise first_error


def _empty_table_from_schema(path: Path, fields: Optional[list[str]]) -> pa.Table:
    schema = pq.ParquetFile(path).schema_arrow
    if fields is not None:
        schema = pa.schema([field for field in schema if field.name in fields])
    arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_empty_if_missing(path: Path, source_path: Path, fields: Optional[list[str]], compression: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_empty_table_from_schema(source_path, fields), path, compression=compression)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _hash_range_target_expr(hash_col: str, target_count: int) -> pl.Expr:
    if target_count == 1:
        return pl.lit(0, dtype=pl.UInt32)
    if _is_power_of_two(target_count):
        shift = 64 - (target_count.bit_length() - 1)
        step = 1 << shift
        return (pl.col(hash_col) // step).cast(pl.UInt32)
    return (
        ((pl.col(hash_col).cast(pl.Float64) / float(1 << 64)) * target_count)
        .floor()
        .clip(0, target_count - 1)
        .cast(pl.UInt32)
    )


def write_split_bucketed_main(
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    fields: Optional[Iterable[str]] = None,
    train_filter: pl.Expr,
    valid_filter: pl.Expr,
    test_filter: pl.Expr,
    train_bucket_count: int = 64,
    seed: int = 2026,
    batch_size: int = 262_144,
    compression: str = "zstd",
    redo: bool = True,
    train_shuffle_mode: Literal["target_sort", "append_bucket"] = "target_sort",
    shuffle_train_bucket_order: bool = False,
) -> SplitBucketedOutputs:
    """Split main parquet parts and write train rows into bucketed parquet files.

    The input is scanned once. Train rows are distributed by row-level hash into
    `train.shuffle/part-*.parquet`; valid/test are appended to single parquet
    files. target_sort range-partitions train rows by hash high bits and sorts
    each target by the full hash, which removes source-part append-order leakage.
    ``shuffle_train_bucket_order`` additionally randomizes the order in which
    the sorted targets are assigned to the output part files.
    """

    paths = _resolve_parquet_paths(input_paths)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train.shuffle"
    valid_path = output_dir / "valid.parquet"
    test_path = output_dir / "test.parquet"
    manifest_path = output_dir / "split_manifest.json"

    if train_bucket_count <= 0:
        raise ValueError("train_bucket_count must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if train_shuffle_mode not in {"target_sort", "append_bucket"}:
        raise ValueError("train_shuffle_mode must be either 'target_sort' or 'append_bucket'.")
    if shuffle_train_bucket_order and train_shuffle_mode != "target_sort":
        raise ValueError("shuffle_train_bucket_order requires train_shuffle_mode='target_sort'.")

    if not redo and train_dir.exists() and valid_path.exists() and test_path.exists():
        train_paths = sorted(train_dir.glob("*.parquet"))
        return SplitBucketedOutputs(
            train_dir=train_dir,
            valid_path=valid_path,
            test_path=test_path,
            train_paths=train_paths,
            row_counts={
                "train": _count_parquet_rows(train_paths),
                "valid": _count_parquet_rows([valid_path]),
                "test": _count_parquet_rows([test_path]),
            },
            manifest_path=manifest_path,
        )

    if output_dir.exists() and redo:
        if train_dir.exists():
            shutil.rmtree(train_dir)
        for path in (valid_path, test_path, manifest_path, output_dir / "_train_shuffle_tmp"):
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    selected_fields = _select_existing_fields(paths, fields)
    if selected_fields is None:
        read_columns = None
    else:
        # The split filters are expected to use columns already in selected_fields
        # for the current preprocessing scripts.
        read_columns = selected_fields

    bucket_paths = [
        train_dir / f"part-{bucket_id:03d}.parquet"
        for bucket_id in range(train_bucket_count)
    ]
    for path in bucket_paths:
        if path.exists():
            path.unlink()

    writers = _ParquetWriterGroup(compression=compression)
    row_counts = {"train": 0, "valid": 0, "test": 0}
    bucket_row_counts = {bucket_id: 0 for bucket_id in range(train_bucket_count)}
    tmp_cols = ["__bb_file_idx", "__bb_row_idx", "__bb_hash", "__bb_bucket"]
    staging_root = output_dir / "_train_shuffle_tmp"
    staging_paths = [
        staging_root / f"part-{bucket_id:05d}.parquet"
        for bucket_id in range(train_bucket_count)
    ]
    bucket_expr = (
        _hash_range_target_expr("__bb_hash", train_bucket_count)
        if train_shuffle_mode == "target_sort"
        else (pl.col("__bb_hash") % train_bucket_count).cast(pl.UInt32)
    )
    if train_shuffle_mode == "target_sort":
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

    try:
        for file_idx, path in enumerate(paths):
            row_offset = 0
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
                df = pl.from_arrow(batch)
                if selected_fields is not None:
                    df = df.select(selected_fields)

                train_df = df.filter(train_filter)
                valid_df = df.filter(valid_filter)
                test_df = df.filter(test_filter)

                if train_df.height > 0:
                    train_df = (
                        train_df
                        .with_row_index("__bb_row_idx", offset=row_offset)
                        .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
                        .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias("__bb_hash"))
                        .with_columns(bucket_expr.alias("__bb_bucket"))
                    )

                    for key, bucket_df in train_df.partition_by(
                        "__bb_bucket",
                        maintain_order=False,
                        include_key=True,
                        as_dict=True,
                    ).items():
                        bucket_id = int(key[0] if isinstance(key, tuple) else key)
                        if train_shuffle_mode == "target_sort":
                            written = writers.write(
                                f"stage-{bucket_id}",
                                staging_paths[bucket_id],
                                bucket_df,
                            )
                        else:
                            clean_df = bucket_df.drop(tmp_cols)
                            written = writers.write(
                                f"train-{bucket_id}",
                                bucket_paths[bucket_id],
                                clean_df,
                            )
                        row_counts["train"] += written
                        bucket_row_counts[bucket_id] += written

                row_counts["valid"] += writers.write("valid", valid_path, valid_df)
                row_counts["test"] += writers.write("test", test_path, test_df)
                row_offset += batch.num_rows

            print(f"    -> split bucketize {file_idx + 1}/{len(paths)}: {path}")
    finally:
        writers.close()

    output_target_order = list(range(train_bucket_count))
    if shuffle_train_bucket_order:
        random.Random(seed).shuffle(output_target_order)

    if train_shuffle_mode == "target_sort":
        for output_id, target_id in enumerate(output_target_order):
            staging_path = staging_paths[target_id]
            bucket_path = bucket_paths[output_id]
            if not staging_path.exists():
                continue
            (
                pl.scan_parquet(staging_path)
                .sort("__bb_hash")
                .drop(tmp_cols)
                .sink_parquet(bucket_path, compression=compression)
            )
        shutil.rmtree(staging_root, ignore_errors=True)

    _write_empty_if_missing(valid_path, paths[0], selected_fields, compression)
    _write_empty_if_missing(test_path, paths[0], selected_fields, compression)

    train_paths = [path for path in bucket_paths if path.exists()]
    manifest = {
        "train_dir": str(train_dir),
        "valid_path": str(valid_path),
        "test_path": str(test_path),
        "train_paths": [str(path) for path in train_paths],
        "row_counts": row_counts,
        "bucket_row_counts": {
            str(output_id): int(bucket_row_counts[target_id])
            for output_id, target_id in enumerate(output_target_order)
            if bucket_row_counts[target_id] > 0
        },
        "target_bucket_row_counts": {
            str(bucket_id): int(count)
            for bucket_id, count in bucket_row_counts.items()
            if count > 0
        },
        "train_bucket_count": int(train_bucket_count),
        "train_shuffle_mode": train_shuffle_mode,
        "train_shuffle_partition": "hash_range_high_bits" if train_shuffle_mode == "target_sort" else "append_bucket",
        "shuffle_train_bucket_order": bool(shuffle_train_bucket_order),
        "output_target_bucket_order": output_target_order,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "input_paths": [str(path) for path in paths],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return SplitBucketedOutputs(
        train_dir=train_dir,
        valid_path=valid_path,
        test_path=test_path,
        train_paths=train_paths,
        row_counts=row_counts,
        manifest_path=manifest_path,
    )
</file>

<file path="utils/task_chain.py">
import os
import psutil
# 指定一个本地锁文件用于传递 PID
from pathlib import Path
LOCK_FILE = Path(__file__).parent / "task_chain.lock"
"""
潜在的小隐患：PID 重用（极低概率）
虽然代码很稳健，但在 Windows 上存在一个理论上的极端情况：PID 复用。

Windows 的进程 ID 是循环使用的。如果：

前者（PID: 1001）意外崩溃。

在后者启动前的千万分之一秒内，系统刚好把 PID 1001 分配给了一个全新的、无关的长期运行程序（比如你的浏览器）。

后者读取到了 1001，发现这个进程竟然还在跑，它就会一直等下去。
"""
def auto_queue():
    my_pid = os.getpid()
    wait_pid = None

    # 1. 尝试读取上一个任务的 PID
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                pid_str = f.read().strip()
                if pid_str.isdigit():
                    wait_pid = int(pid_str)
        except Exception:
            pass

    # 2. 立刻把自己的 PID 写进去，让后面追加的任务等自己
    with open(LOCK_FILE, "w") as f:
        f.write(str(my_pid))

    # 3. 如果发现有前置任务，进入阻塞等待
    if wait_pid and wait_pid != my_pid:
        try:
            p = psutil.Process(wait_pid)
            if p.is_running():
                print(f"[排队系统] 已挂起。正在等待前置任务 (PID: {wait_pid}) 完成...")
                p.wait()  # 进程在这里完全停住，不消耗 CPU
        except psutil.NoSuchProcess:
            # 说明上一个任务刚好在这极短的时间内跑完，或者 PID 已经失效
            pass

    print(f"\n[排队系统] 前置任务已清空，当前任务 (PID: {my_pid}) 开始执行！")


if __name__ == "__main__":
    # 在所有业务逻辑开始前调用
    auto_queue()

    # ==========================================
    # 下面写你经常修改的业务代码
    # ==========================================
    import time

    print("代码版本 A 正在运行...")
    time.sleep(10)  # 模拟长耗时任务
    print("代码版本 A 运行结束。")
</file>

<file path="utils/time_bucket.py">
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn


# Matches the official kddsample baseline: 64 bucket boundaries in seconds,
# bucket id 0 reserved for padding, 1..64 for valid relative-time buckets.
# [-, 1), [1, 5)
TIME_BUCKET_BOUNDARIES = (
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
)


# Long-sequence ablation: preserve every baseline boundary through 24 hours,
# split 24h--72h into 6-hour intervals, and merge the sparsely used tail.  It
# deliberately has the same length as TIME_BUCKET_BOUNDARIES so model and
# checkpoint tensor shapes remain unchanged.
LONG_SEQUENCE_TIME_BUCKET_BOUNDARIES = (
    *TIME_BUCKET_BOUNDARIES[:47],
    108000, 129600, 151200, 172800,
    194400, 216000, 237600, 259200,
    345600, 432000, 604800, 1209600,
    2592000, 5184000, 15552000, 31536000,
)


def get_num_time_buckets(boundaries: Optional[Iterable[int]] = None) -> int:
    effective_boundaries = tuple(boundaries) if boundaries is not None else TIME_BUCKET_BOUNDARIES
    if not effective_boundaries:
        raise ValueError("Time bucket boundaries must be a non-empty sequence.")
    # 0 is reserved for padding, 1..N for valid buckets.
    return len(effective_boundaries) + 1


def build_padding_mask(seq_lens: torch.Tensor, max_len: int) -> torch.Tensor:
    if seq_lens.ndim != 1:
        raise ValueError(f"Expected seq_lens to have shape [B], got {tuple(seq_lens.shape)}")
    idx = torch.arange(max_len, device=seq_lens.device).unsqueeze(0)
    return idx >= seq_lens.unsqueeze(1)


def bucketize_relative_time(
    curr_ts: torch.Tensor,
    hist_ts: torch.Tensor,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    boundaries: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """
    Convert absolute timestamps into official-style relative-time bucket ids.

    Args:
        curr_ts: shape [B] or [B, 1], current interaction timestamps.
        hist_ts: shape [B, L], historical timestamps aligned to a sequence.
        seq_lens: optional shape [B], used to mask padded sequence positions.
        padding_mask: optional shape [B, L], True means padding.
        boundaries: optional iterable of ascending second-level boundaries.

    Returns:
        LongTensor with shape [B, L], where 0 is padding and 1..N are buckets.
    """
    if hist_ts.ndim != 2:
        raise ValueError(f"Expected hist_ts to have shape [B, L], got {tuple(hist_ts.shape)}")

    if curr_ts.ndim == 2 and curr_ts.shape[1] == 1:
        curr_ts = curr_ts.squeeze(1)
    if curr_ts.ndim != 1:
        raise ValueError(f"Expected curr_ts to have shape [B], got {tuple(curr_ts.shape)}")
    if curr_ts.shape[0] != hist_ts.shape[0]:
        raise ValueError(
            f"Batch size mismatch between curr_ts {tuple(curr_ts.shape)} and hist_ts {tuple(hist_ts.shape)}"
        )

    if padding_mask is None and seq_lens is not None:
        padding_mask = build_padding_mask(seq_lens, hist_ts.shape[1])
    elif padding_mask is not None and padding_mask.shape != hist_ts.shape:
        raise ValueError(
            f"Expected padding_mask to have shape {tuple(hist_ts.shape)}, got {tuple(padding_mask.shape)}"
        )

    if boundaries is None:
        boundaries = TIME_BUCKET_BOUNDARIES
    boundary_tensor = torch.as_tensor(
        tuple(boundaries),
        device=hist_ts.device,
        dtype=hist_ts.dtype,
    )
    if boundary_tensor.ndim != 1 or boundary_tensor.numel() == 0:
        raise ValueError("Time bucket boundaries must be a non-empty 1D sequence.")

    curr_ts = curr_ts.to(hist_ts.device, dtype=hist_ts.dtype)
    time_diff = torch.clamp(curr_ts.unsqueeze(1) - hist_ts, min=0)

    bucket_ids = torch.bucketize(time_diff, boundary_tensor, right=False)
    max_bucket_idx = boundary_tensor.numel() - 1
    bucket_ids = torch.clamp(bucket_ids, min=0, max=max_bucket_idx) + 1
    bucket_ids = bucket_ids.to(torch.long)

    effective_padding = hist_ts.eq(0)
    if padding_mask is not None:
        effective_padding = effective_padding | padding_mask.to(device=hist_ts.device)
    bucket_ids[effective_padding] = 0
    return bucket_ids


def bucketize_adjacent_time(
    hist_ts: torch.Tensor,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    boundaries: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Bucketize the gap from each valid history token to its predecessor.

    Bucket 0 is used for padding and for the first visible token because that
    token has no predecessor in the supplied sequence window. Valid positions
    from index 1 onward use buckets 1..N, including zero-second gaps.
    """
    if hist_ts.ndim != 2:
        raise ValueError(
            f"Expected hist_ts to have shape [B, L], got {tuple(hist_ts.shape)}"
        )
    if padding_mask is None and seq_lens is not None:
        padding_mask = build_padding_mask(seq_lens, hist_ts.shape[1])
    elif padding_mask is not None and padding_mask.shape != hist_ts.shape:
        raise ValueError(
            f"Expected padding_mask to have shape {tuple(hist_ts.shape)}, "
            f"got {tuple(padding_mask.shape)}"
        )

    effective_boundaries = tuple(boundaries or TIME_BUCKET_BOUNDARIES)
    boundary_tensor = torch.as_tensor(
        effective_boundaries,
        device=hist_ts.device,
        dtype=hist_ts.dtype,
    )
    if boundary_tensor.ndim != 1 or boundary_tensor.numel() == 0:
        raise ValueError("Time bucket boundaries must be a non-empty 1D sequence.")

    bucket_ids = torch.zeros_like(hist_ts, dtype=torch.long)
    if hist_ts.shape[1] > 1:
        gaps = torch.clamp(hist_ts[:, 1:] - hist_ts[:, :-1], min=0)
        gap_buckets = torch.bucketize(gaps, boundary_tensor, right=False)
        gap_buckets = torch.clamp(
            gap_buckets,
            min=0,
            max=boundary_tensor.numel() - 1,
        ).to(torch.long) + 1
        bucket_ids[:, 1:] = gap_buckets

    effective_padding = hist_ts.eq(0)
    if padding_mask is not None:
        effective_padding = effective_padding | padding_mask.to(device=hist_ts.device)
    bucket_ids[effective_padding] = 0
    bucket_ids[:, 0] = 0
    return bucket_ids


class RelativeTimeEmbedding(nn.Module):
    """
    Dynamic relative-time encoder built on top of bucketized time deltas.
    use seconds

    Usage:
        time_emb = RelativeTimeEmbedding(embedding_dim=64)
        rel_time = time_emb(curr_ts, hist_ts, seq_lens=seq_len)   # [B, L, 64]
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        boundaries: Optional[Iterable[int]] = None,
        padding_idx: int = 0,
        init_std: float = 1e-4,
    ):
        super().__init__()
        self.boundaries = tuple(boundaries) if boundaries is not None else TIME_BUCKET_BOUNDARIES
        self.num_buckets = get_num_time_buckets(self.boundaries)
        self.padding_idx = padding_idx
        self.init_std = init_std
        self.embedding = nn.Embedding(
            num_embeddings=self.num_buckets,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.reset_parameters()

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight.data, mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def bucketize(
        self,
        curr_ts: torch.Tensor,
        hist_ts: torch.Tensor,
        *,
        seq_lens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return bucketize_relative_time(
            curr_ts=curr_ts,
            hist_ts=hist_ts,
            seq_lens=seq_lens,
            padding_mask=padding_mask,
            boundaries=self.boundaries,
        )

    def forward(
        self,
        curr_ts: torch.Tensor,
        hist_ts: torch.Tensor,
        *,
        seq_lens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_bucket_ids: bool = False,
    ):
        bucket_ids = self.bucketize(
            curr_ts=curr_ts,
            hist_ts=hist_ts,
            seq_lens=seq_lens,
            padding_mask=padding_mask,
        )
        time_emb = self.embedding(bucket_ids)
        if return_bucket_ids:
            return time_emb, bucket_ids
        return time_emb


class AdjacentTimeEmbedding(nn.Module):
    """Embedding of within-window gaps between adjacent history tokens."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        boundaries: Optional[Iterable[int]] = None,
        padding_idx: int = 0,
        init_std: float = 1e-4,
    ):
        super().__init__()
        self.boundaries = tuple(boundaries or TIME_BUCKET_BOUNDARIES)
        self.num_buckets = get_num_time_buckets(self.boundaries)
        self.padding_idx = padding_idx
        self.init_std = init_std
        self.embedding = nn.Embedding(
            num_embeddings=self.num_buckets,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight.data, mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(
        self,
        hist_ts: torch.Tensor,
        *,
        seq_lens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_bucket_ids: bool = False,
    ):
        bucket_ids = bucketize_adjacent_time(
            hist_ts,
            seq_lens=seq_lens,
            padding_mask=padding_mask,
            boundaries=self.boundaries,
        )
        time_emb = self.embedding(bucket_ids)
        if return_bucket_ids:
            return time_emb, bucket_ids
        return time_emb
</file>

<file path="utils/time.py">
import ctypes
import math
import os
import time
import warnings
from enum import Enum
from typing import Union

LOGGER = print

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 记录函数开始执行的时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.perf_counter()  # 记录函数结束执行的时间
        if LOGGER is not None:
            print(f"【函数计时】:{func.__name__} 使用 {end_time - start_time:.4f} s")
        return result

    return wrapper


import functools
import contextlib  # <--- 新增导入
from collections import defaultdict

class NamedTimer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NamedTimer, cls).__new__(cls)
            cls._instance._records = defaultdict(float)
            cls._instance._counts = defaultdict(int)
            cls._instance._start_times = {}
        return cls._instance

    def start_record(self, name):
        self._start_times[name] = time.perf_counter()

    def stop_record(self, name):
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times.pop(name)
            self._records[name] += elapsed
            self._counts[name] += 1
            return elapsed
        return 0

    def loop_flag(self, name):
        self.stop_record(name)
        self.start_record(name)

    def collect(self, name=None):
        def decorator(func):
            record_name = name if name else func.__name__
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_record(record_name)
                result = func(*args, **kwargs)
                self.stop_record(record_name)
                return result
            return wrapper
        return decorator

    # --- 修复点：使用 contextmanager 替代原来的 __call__, __enter__, __exit__ ---
    @contextlib.contextmanager
    def __call__(self, name):
        self.start_record(name)
        try:
            yield self  # 允许 with ntr("name") as t: 的语法
        finally:
            self.stop_record(name)

    def report(self):
        print(f"\n{'[ Timer Report ]':^40}")
        print(f"{'Name':<20} | {'Total':>8} | {'Avg':>8} | {'Calls':>5}")
        print("-" * 45)
        for name in self._records:
            total = self._records[name]
            count = self._counts[name]
            avg = total / count if count > 0 else 0
            print(f"{name:<20} | {total:>7.4f}s | {avg:>7.4f}s | {count:>5}")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class CudaNamedTimer(NamedTimer):
        def start_record(self, name):
            # 只有在 GPU 可用时才同步，避免在只有 CPU 的 torch 环境下报错
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            super().start_record(name)

        def stop_record(self, name):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return super().stop_record(name)
else:
    # 如果没有 torch，CudaNamedTimer 就退化为普通的 NamedTimer
    # 这样业务代码里写 CudaNamedTimer() 也不会崩，只是没有同步功能
    class CudaNamedTimer(NamedTimer):
        pass
</file>

<file path="utils/visualize.py">
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union


def plot_bias_distributions(
        lf: pl.LazyFrame,
        uid_col: str = "user_id",
        iid_col: str = "item_id",
        label_col: str = "label",
        save_path: Union[str, Path] = "bias_distribution.png",
        min_item_interactions: int = 10
):
    """
    计算并绘制用户偏置（User Bias）和物品偏置（Item Bias）的分布直方图。

    参数:
        lf: 包含交互数据的 Polars LazyFrame。
        uid_col: 用户 ID 列名。
        iid_col: 物品 ID 列名。
        label_col: 标签列名（应为 0 或 1 的二分类目标）。
        save_path: 图片保存路径。
        min_item_interactions: 物品的最少交互次数阈值，用于在绘图时过滤长尾冷门物品以避免视觉噪音。
    """
    print(f"开始计算 {uid_col} 和 {iid_col} 的偏置分布...")

    # 1. 计算 User Bias: 每个用户的正样本率
    user_bias_df = lf.group_by(uid_col).agg(
        pl.len().alias("total_inter"),
        pl.col(label_col).sum().alias("pos_inter")
    ).with_columns(
        (pl.col("pos_inter") / pl.col("total_inter")).alias("user_bias_ratio")
    ).collect()

    # 2. 计算 Item Bias: 每个物品的正样本率
    item_bias_df = lf.group_by(iid_col).agg(
        pl.len().alias("total_inter"),
        pl.col(label_col).sum().alias("pos_inter")
    ).with_columns(
        (pl.col("pos_inter") / pl.col("total_inter")).alias("item_bias_ratio")
    ).collect()

    # 打印基础统计信息
    print(f"\n--- User Bias ({uid_col}) 统计 ---")
    print(user_bias_df.select("user_bias_ratio").describe())

    print(f"\n--- Item Bias ({iid_col}) 统计 ---")
    print(item_bias_df.select("item_bias_ratio").describe())

    # 3. 绘制分布直方图
    plt.figure(figsize=(12, 5))

    # User Bias 子图
    plt.subplot(1, 2, 1)
    plt.hist(user_bias_df["user_bias_ratio"].to_numpy(), bins=50, color='skyblue', edgecolor='black')
    plt.title(f'User Bias Distribution\n(Ratio of {label_col}=1 per user)')
    plt.xlabel('Positive Ratio')
    plt.ylabel('Number of Users')

    # Item Bias 子图
    plt.subplot(1, 2, 2)
    item_bias_filtered = item_bias_df.filter(pl.col("total_inter") > min_item_interactions)
    plt.hist(item_bias_filtered["item_bias_ratio"].to_numpy(), bins=50, color='salmon', edgecolor='black')
    plt.title(f'Item Bias Distribution\n(Ratio of {label_col}=1 per item)')
    plt.xlabel('Positive Ratio')
    plt.ylabel(f'Number of Items (Interactions > {min_item_interactions})')

    plt.tight_layout()

    # 确保保存目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # 释放内存

    print(f"偏置分布直方图已保存至: {save_path}")



def plot_sparsity_distributions(
        lf: pl.LazyFrame,
        uid_col: str = "user_id",
        iid_col: str = "item_id",
        save_path: Union[str, Path] = "sparsity_distribution.png",
        log_y: bool = True
):
    """
    计算并绘制用户和物品的交互次数分布，以评估数据的稀疏程度和长尾效应。

    参数:
        lf: 包含交互数据的 Polars LazyFrame。
        uid_col: 用户 ID 列名。
        iid_col: 物品 ID 列名。
        save_path: 图片保存路径。
        log_y: 是否对 Y 轴（频数）使用对数刻度。强烈建议在推荐系统数据中开启，以应对长尾分布。
    """
    print(f"开始计算 {uid_col} 和 {iid_col} 的交互稀疏度分布...")

    # 1. 计算每个用户的交互次数 (User Activity)
    user_inter_df = lf.group_by(uid_col).agg(
        pl.len().alias("interaction_count")
    ).collect()

    # 2. 计算每个物品的交互次数 (Item Popularity)
    item_inter_df = lf.group_by(iid_col).agg(
        pl.len().alias("interaction_count")
    ).collect()

    # 打印基础统计信息（分位数对于观察稀疏度非常有用）
    print(f"\n--- User Interactions ({uid_col}) 统计 ---")
    print(user_inter_df.select("interaction_count").describe())

    print(f"\n--- Item Interactions ({iid_col}) 统计 ---")
    print(item_inter_df.select("interaction_count").describe())

    # 3. 绘制分布直方图
    plt.figure(figsize=(12, 5))

    # User 交互次数分布子图
    plt.subplot(1, 2, 1)
    plt.hist(user_inter_df["interaction_count"].to_numpy(), bins=50, color='mediumpurple', edgecolor='black', log=log_y)
    plt.title('User Interaction Distribution\n(Activity per user)')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users' + (' (Log Scale)' if log_y else ''))

    # Item 交互次数分布子图
    plt.subplot(1, 2, 2)
    plt.hist(item_inter_df["interaction_count"].to_numpy(), bins=50, color='lightgreen', edgecolor='black', log=log_y)
    plt.title('Item Interaction Distribution\n(Popularity per item)')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Items' + (' (Log Scale)' if log_y else ''))

    plt.tight_layout()

    # 确保保存目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # 释放内存

    print(f"稀疏度分布直方图已保存至: {save_path}")


def plot_sparsity_ecdf(
        lf: pl.LazyFrame,
        uid_col: str = "user_id",
        iid_col: str = "item_id",
        save_path: Union[str, Path] = "sparsity_ecdf.png"
):
    """
    计算并绘制用户和物品交互次数的经验累积分布函数 (ECDF) 图。
    该图表专门用于直观读取 50%、75%、90% 等分位数指标，非常适合极端长尾的数据集。

    参数:
        lf: 包含交互数据的 Polars LazyFrame。
        uid_col: 用户 ID 列名。
        iid_col: 物品 ID 列名。
        save_path: 图片保存路径。
    """
    print(f"开始计算 {uid_col} 和 {iid_col} 的累积分布(ECDF)...")

    # 1. 统计每个用户/物品的交互次数并转为 NumPy 数组
    user_counts = lf.group_by(uid_col).agg(pl.len().alias("count")).collect()["count"].to_numpy()
    item_counts = lf.group_by(iid_col).agg(pl.len().alias("count")).collect()["count"].to_numpy()

    # 2. 绘图设置
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def plot_ecdf(ax, data, title):
        # 排序数据以计算 ECDF
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)

        # 提取关键分位数的精确值
        p50 = np.percentile(x, 50)
        p75 = np.percentile(x, 75)
        p90 = np.percentile(x, 90)

        # 绘制 ECDF 阶梯曲线
        ax.step(x, y, where='post', color='dodgerblue', linewidth=2)

        # 绘制水平参考线 (Y轴累积比例)
        ax.axhline(0.50, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.75, color='gray', linestyle='--', alpha=0.5)

        # 绘制垂直参考线及图例 (X轴具体的交互次数)
        ax.axvline(p50, color='red', linestyle=':', linewidth=2, label=f'50% (Median) = {p50:.0f}')
        ax.axvline(p75, color='green', linestyle=':', linewidth=2, label=f'75% = {p75:.0f}')
        ax.axvline(p90, color='darkorange', linestyle=':', linewidth=2, label=f'90% = {p90:.0f}')

        # 核心设置：X轴使用对数刻度以展开高度集中的长尾数据
        ax.set_xscale('log')

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Number of Interactions (Log Scale)', fontsize=10)
        ax.set_ylabel('Cumulative Proportion', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, which="both") # 开启网格线以便更精准对齐

    # 分别绘制用户和物品的 ECDF
    plot_ecdf(axes[0], user_counts, f'User ECDF ({uid_col})')
    plot_ecdf(axes[1], item_counts, f'Item ECDF ({iid_col})')

    plt.tight_layout()

    # 确保保存目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"ECDF累积分布图已保存至: {save_path}")


from matplotlib.ticker import PercentFormatter
def plot_power2_sparsity(
        lf: pl.LazyFrame,
        uid_col: str = "user_id",
        iid_col: str = "item_id",
        save_path: Union[str, Path] = "sparsity_power2.png"
):
    """
    计算并绘制按 2^x 区间分桶的稀疏度分布图。
    横坐标为交互次数的 2^x 对数区间，纵坐标为落入该区间的用户/物品百分比。
    """
    print(f"开始计算 {uid_col} 和 {iid_col} 的 2^x 区间百分比分布...")

    # 1. 统计交互次数
    user_counts = lf.group_by(uid_col).agg(pl.len().alias("count")).collect()["count"].to_numpy()
    item_counts = lf.group_by(iid_col).agg(pl.len().alias("count")).collect()["count"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    def plot_subplot(ax, data, title, color):
        # 交互次数转为 log2 以实现 2^x 分桶
        log2_data = np.log2(data)

        # 确定最大需要的指数（确保最大值能被包含在最后一个桶的左边界）
        max_power = int(np.floor(np.log2(data.max()))) + 1
        bins = np.arange(0, max_power + 1)

        # 计算权重以显示百分比（每个样本的权重为 1/总数 * 100）
        weights = np.ones_like(log2_data) / len(log2_data) * 100

        # 绘制直方图，rwidth 留出柱子间距让区间感更强
        counts, edges, patches = ax.hist(
            log2_data,
            bins=bins,
            weights=weights,
            color=color,
            edgecolor='black',
            rwidth=0.85
        )

        # 设置 X 轴的刻度与标签（显示为 2^0, 2^1, 2^2...）
        ax.set_xticks(bins)
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in bins])

        # 设置 Y 轴为百分比格式
        ax.yaxis.set_major_formatter(PercentFormatter())

        # 在柱子上标注具体的百分比数值（过滤掉太小的，避免遮挡）
        for i in range(len(counts)):
            if counts[i] > 0.5:  # 大于 0.5% 才标注数值，保持图面整洁
                # 横坐标取柱子中心，纵坐标取柱子高度加一点点
                ax.text(edges[i] + 0.5, counts[i] + 0.5, f'{counts[i]:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(f'Interaction Count Range $[2^x, 2^{{x+1}})$', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)

        # 添加水平参考网格
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # 移除顶部和右侧的边框线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 2. 分别绘制 User 和 Item
    plot_subplot(axes[0], user_counts, f'User Distribution ({uid_col})', 'mediumpurple')
    plot_subplot(axes[1], item_counts, f'Item Distribution ({iid_col})', 'lightgreen')

    plt.tight_layout()

    # 3. 保存图片
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"2^x分布图已保存至: {save_path}")
</file>

<file path="__init__.py">
"""rosen 顶层包。"""
</file>

</files>
