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
