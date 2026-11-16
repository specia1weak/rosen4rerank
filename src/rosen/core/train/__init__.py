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
