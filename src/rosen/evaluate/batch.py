"""One resolved, row-filtered task view for every evaluator."""
from dataclasses import dataclass
from typing import Optional

import torch

from rosen.core.interaction import Interaction
from rosen.core.prediction import Prediction


@dataclass(frozen=True)
class EvalBatch:
    pred: Prediction
    inter: Optional[Interaction] = None
    scores: Optional[torch.Tensor] = None
    order: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    users: Optional[torch.Tensor] = None
