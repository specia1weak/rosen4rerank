import torch
from torch import nn

from rosen.utils.optimize import split_model_params_by_decay


class _OptimizerGroupingFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Deliberately avoid "embedding" in the module name. Module ownership
        # must still keep this table out of weight decay.
        self.lookup = nn.Embedding(11, 4)
        self.embedding_adapter = nn.Linear(4, 4, bias=False)
        self.projection = nn.Linear(4, 3)
        self.norm = nn.LayerNorm(4)
        self.query_seed = nn.Parameter(torch.randn(5, 4))
        self.domain_vectors = nn.Parameter(torch.randn(5, 3, 4))
        self.frozen_matrix = nn.Parameter(torch.randn(4, 4), requires_grad=False)


def test_model_decay_groups_exclude_embeddings_norm_bias_and_free_vectors() -> None:
    model = _OptimizerGroupingFixture()
    groups = split_model_params_by_decay(model, weight_decay=1.0e-6)
    decay_ids = {
        id(parameter)
        for group in groups
        if group["weight_decay"] == 1.0e-6
        for parameter in group["params"]
    }
    no_decay_ids = {
        id(parameter)
        for group in groups
        if group["weight_decay"] == 0.0
        for parameter in group["params"]
    }

    assert decay_ids == {
        id(model.projection.weight),
        id(model.embedding_adapter.weight),
    }
    assert no_decay_ids == {
        id(model.lookup.weight),
        id(model.projection.bias),
        id(model.norm.weight),
        id(model.norm.bias),
        id(model.query_seed),
        id(model.domain_vectors),
    }
    assert id(model.frozen_matrix) not in decay_ids | no_decay_ids
    assert decay_ids.isdisjoint(no_decay_ids)
