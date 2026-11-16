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
