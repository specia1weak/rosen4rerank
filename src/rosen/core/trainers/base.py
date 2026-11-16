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
