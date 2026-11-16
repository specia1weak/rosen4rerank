import numpy as np
from torch import nn

from rosen.core.interaction import Interaction
from rosen.core.train.loss import LossPort
from rosen.emb import SchemaManager
from rosen.emb.emblayer import OmniEmbLayer
from rosen.utils.observatory.port import ObservationPort


class BaseModel(nn.Module):
    def __init__(self, manager: SchemaManager):
        super(BaseModel, self).__init__()
        self.manager: SchemaManager = manager
        self.omni_embedding: OmniEmbLayer = OmniEmbLayer(manager=manager)
        self.obp: ObservationPort = ObservationPort(namespace=type(self).__name__)
        self.lsp: LossPort = LossPort()

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

    def predict(self, interaction: Interaction):
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
