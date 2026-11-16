from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from rosen.core.train.two_stage import StageReport, StageSnapshot


MetricPath = Union[str, Sequence[str], Tuple[str, ...]]


class TwoStageRetrainMixin:
    """Reusable search/retrain implementation of TwoStageTrainingProtocol.

    The mixin never mutates a running Trainer into another phase. It tracks
    stage-one best state, decides when stage one is complete, and constructs a
    new stage-two model. The child model supplies only its constructor and
    model-specific retrain initialization rules.
    """

    def _init_two_stage_retrain(
            self,
            *,
            search_epochs: int,
            metric_path: MetricPath = ("overall", "auc"),
            restore_best_before_retrain: bool = True,
            feedback_target: str = "student",
    ) -> None:
        self.search_epochs = max(1, int(search_epochs))
        self.retrain_metric_path = metric_path
        self.restore_best_before_retrain = bool(restore_best_before_retrain)
        self.two_stage_feedback_target = str(feedback_target)

        self.in_retrain_phase = False
        self._search_best_metric = float("-inf")
        self._search_best_state: Optional[Dict[str, Any]] = None
        self._retrain_start_epoch: Optional[int] = None

    @property
    def training_phase(self) -> str:
        return "retrain" if self.in_retrain_phase else "search"

    def _build_stage_two_model(self):
        """Return a new, uninitialized model of the desired stage-two class."""

        raise NotImplementedError(
            "Two-stage models must explicitly construct their stage-two model."
        )

    def _prepare_retrain_phase(self) -> None:
        """Apply model-specific reset/freeze rules to the new stage-two model."""

        raise NotImplementedError

    def _restore_retrain_runtime(self) -> None:
        """Restore flags after checkpoint load without reinitializing weights."""

        raise NotImplementedError(
            "Two-stage models must restore retrain runtime without rerunning "
            "destructive stage initialization."
        )

    def _extract_two_stage_metric(self, metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        if not metrics:
            return None

        path = self.retrain_metric_path
        if isinstance(path, str):
            value = metrics.get(path)
            return float(value) if isinstance(value, (int, float)) else None

        value: Any = metrics
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return float(value) if isinstance(value, (int, float)) else None

    def _feedback_metrics(self, report: StageReport) -> tuple[Optional[dict], str]:
        target = self.two_stage_feedback_target
        metrics = report.metrics_for(target)
        if metrics is None:
            target = report.primary_target
            metrics = report.primary_metrics
        return metrics, str(target)

    def _update_search_best_state(
            self,
            metrics: Optional[Dict[str, Any]],
            report: StageReport,
    ) -> None:
        metric_value = self._extract_two_stage_metric(metrics)
        if metric_value is None or metric_value <= self._search_best_metric:
            return
        self._search_best_metric = metric_value
        self._search_best_state = dict(StageSnapshot.capture(self, report).model_state)

    def should_finish_stage_one(self, report: StageReport) -> bool:
        metrics, _ = self._feedback_metrics(report)
        self._update_search_best_state(metrics, report)
        finished = int(report.epoch) >= self.search_epochs
        return finished

    def initialize_stage_two(self, snapshot: StageSnapshot):
        stage_two_model = self._build_stage_two_model()
        if stage_two_model is self:
            raise ValueError("_build_stage_two_model() must return a new model object.")

        source_state = snapshot.model_state
        if self.restore_best_before_retrain and self._search_best_state is not None:
            source_state = self._search_best_state
        stage_two_model.load_state_dict(source_state)

        stage_two_model.in_retrain_phase = True
        stage_two_model._search_best_metric = self._search_best_metric
        # The selected state has already been loaded. Keeping another full
        # copy would duplicate the model in every stage-two checkpoint.
        stage_two_model._search_best_state = None
        stage_two_model._retrain_start_epoch = 0
        stage_two_model.two_stage_feedback_target = self.two_stage_feedback_target
        for name in ("current_temperature", "_search_step"):
            if hasattr(self, name) and hasattr(stage_two_model, name):
                setattr(stage_two_model, name, getattr(self, name))
        if hasattr(self, "_initial_branch_state") and hasattr(stage_two_model, "_initial_branch_state"):
            stage_two_model._initial_branch_state = copy.deepcopy(self._initial_branch_state)

        stage_two_model._prepare_retrain_phase()
        return stage_two_model

    def two_stage_status(self) -> Dict[str, Any]:
        return {
            "in_retrain_phase": bool(self.in_retrain_phase),
            "search_best_metric": (
                0.0 if self._search_best_metric == float("-inf")
                else float(self._search_best_metric)
            ),
            "search_epochs": int(self.search_epochs),
            "retrain_start_epoch": self._retrain_start_epoch,
        }

    def trainer_state_dict(self) -> Dict[str, Any]:
        state = {
            **self.two_stage_status(),
            "search_best_state": self._search_best_state,
            "feedback_target": self.two_stage_feedback_target,
        }
        for name in ("current_temperature", "_search_step"):
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def load_trainer_state_dict(self, state: Dict[str, Any]) -> None:
        self.in_retrain_phase = bool(state.get("in_retrain_phase", self.in_retrain_phase))
        self._search_best_metric = float(state.get("search_best_metric", self._search_best_metric))
        self._search_best_state = state.get("search_best_state", self._search_best_state)
        self._retrain_start_epoch = state.get("retrain_start_epoch", self._retrain_start_epoch)
        self.two_stage_feedback_target = str(
            state.get("feedback_target", self.two_stage_feedback_target)
        )
        for name in ("current_temperature", "_search_step"):
            if name in state and hasattr(self, name):
                setattr(self, name, state[name])
        if self.in_retrain_phase:
            self._restore_retrain_runtime()

__all__ = ["TwoStageRetrainMixin"]
