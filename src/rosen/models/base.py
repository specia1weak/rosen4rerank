from typing import Optional

import numpy as np
import torch
from torch import nn

from rosen.core.interaction import Interaction
from rosen.core.prediction import Prediction
from rosen.core.train.loss import LossPort
from rosen.emb import SchemaManager
from rosen.emb.emblayer import OmniEmbLayer
from rosen.utils.observatory.port import ObservationPort


class BaseModel(nn.Module):
    def __init__(self, manager: SchemaManager):
        super(BaseModel, self).__init__()
        self.manager: SchemaManager = manager
        self.LABEL: Optional[str] = manager.label_field
        self.LABELS: tuple[str, ...] = manager.label_fields
        self.DOMAIN: Optional[str] = manager.domain_field
        self.omni_embedding: OmniEmbLayer = OmniEmbLayer(manager=manager)
        self.obp: ObservationPort = ObservationPort(namespace=type(self).__name__)
        self.lsp: LossPort = LossPort()

    def get_label_mask(
        self,
        interaction: Interaction,
        label_field: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the declared supervision mask, or ones matching the label."""
        target_label = label_field or self.LABEL
        if target_label is None:
            raise ValueError("get_label_mask requires a label field.")
        get_spec = getattr(self.manager, "get_label_spec", lambda name: None)
        spec = get_spec(target_label)
        if spec is not None and spec.mask_field and spec.mask_field in interaction:
            return interaction[spec.mask_field].to(dtype=dtype)
        return torch.ones_like(interaction[target_label], dtype=dtype)

    def apply_xavier_initialization(self) -> None:
        """Optionally reset linear layers to Xavier normal with zero bias."""
        def _reset(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

        self.apply(_reset)

    def calculate_loss(self, interaction: Interaction):
        raise NotImplementedError

    def predict(self, interaction: Interaction) -> torch.Tensor | Prediction:
        raise NotImplementedError

    def full_sort_predict(self, interaction: Interaction):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + f": {params}"
        )
