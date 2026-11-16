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
