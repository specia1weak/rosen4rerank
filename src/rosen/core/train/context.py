from dataclasses import dataclass
from typing import Any, Optional, Iterable, Mapping
import torch

from rosen.core.interaction import Interaction
from rosen.core.train.early_stopper import EarlyStopper


@dataclass(frozen=True)
class EvaluationFeedback:
    """Typed evaluation targets available to model training-cycle methods."""

    metrics_by_target: Mapping[str, Optional[dict]]
    primary_target: str
    primary_metrics: Optional[dict]


@dataclass
class TrainContext:
    # 1. 训练进度统计（通常是只读的）
    epoch: int
    global_step: int
    batch_idx: int

    # 2. 核心操纵杆
    optimizer: torch.optim.Optimizer

    # 3. 实验与全局组件
    manager: Any  # 你的 SchemaManager
    cfg: Any  # 全局配置参数
    timer: Any  # 允许模型给特殊的内部步骤计时
    recorder: Any = None
    stage: str = "train"
    phase: str = "default"
    evaluation_feedback: Optional[EvaluationFeedback] = None


@dataclass
class TrainerDataLoaders:
    train: Iterable[Interaction]
    valid: Optional[Iterable[Interaction]]
    test: Optional[Iterable[Interaction]] = None  # 扩展性好，支持可选参数


@dataclass
class TrainerComponents:
    evaluator_manager: Any = None

    # Keep core/train independent from the concrete Observatory package.
    recorder: Any = None
    timer: Any = None
    early_stopper: Optional[EarlyStopper] = None
    health_monitor: Any = None

    def __post_init__(self) -> None:
        if self.early_stopper is not None and not isinstance(self.early_stopper, EarlyStopper):
            raise TypeError(
                "TrainerComponents.early_stopper must be an EarlyStopper or None, "
                f"got {type(self.early_stopper).__name__}."
            )
