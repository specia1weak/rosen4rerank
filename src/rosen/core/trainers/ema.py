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
