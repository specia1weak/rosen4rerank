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
