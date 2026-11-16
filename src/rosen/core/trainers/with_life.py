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
