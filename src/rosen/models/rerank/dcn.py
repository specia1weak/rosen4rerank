"""DCN with vector cross recurrence and the reference's concatenated cross states."""
import torch
from torch import nn

from .slate import SlateRerankModel
from .components.generators import mlp


class CrossNetwork(nn.Module):
    def __init__(self, width, num_layers):
        super().__init__()
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("cross_layers must be a positive integer.")
        self.weights = nn.ModuleList(nn.Linear(width, 1, bias=False) for _ in range(num_layers))
        self.biases = nn.ParameterList(nn.Parameter(torch.zeros(width)) for _ in range(num_layers))
        for layer in self.weights:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        state, states = x, [x]
        for weight, bias in zip(self.weights, self.biases):
            state = x * weight(state) + bias + state
            states.append(state)
        return torch.cat(states, -1)


class DCNRerankModel(SlateRerankModel):
    def __init__(self, manager, *, cross_layers=3, mlp_a=(60, 32, 20), mlp_b=(64, 32), **kwargs):
        super().__init__(manager, **kwargs)
        self.cross = CrossNetwork(self.item_dim, cross_layers)
        self.encoder, width = mlp(self.item_dim * (cross_layers + 2), mlp_a, nn.SiLU)
        self.head, width = mlp(width + 1, mlp_b, nn.SiLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode_items(self, cate, dense, mask):
        x = cate.flatten(-2)
        # The source concatenates input_embedding and y_cross, which includes x0.
        return self.encoder(torch.cat((x, self.cross(x)), -1))

    def _score_encoded(self, items, dense, mask, context):
        return self.output(self.head(torch.cat((items, dense), -1))).squeeze(-1)
