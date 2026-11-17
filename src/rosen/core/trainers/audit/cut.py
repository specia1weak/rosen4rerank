from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch

from rosen.core.train.context import TrainContext
from rosen.core.prediction import as_prediction
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
            self.evaluator.collect(batch_interaction, as_prediction(scores, target=model.LABEL))

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
