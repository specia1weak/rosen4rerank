"""PIER diversity variant: append the farthest non-retrieved slate before OCPM."""
import torch

from .pier import PIERRerankModel
from .base import select_by_order
from .components.diversity import candidate_statistics


class PIERSubmodularRerankModel(PIERRerankModel):
    def _top_orders(self, cate, distance):
        ids = self.retriever.top_k(distance, self.top_k)
        permutations = self.retriever.full_permutations
        top = permutations[ids]
        if self.top_k == len(permutations):
            return top
        cate = cate.detach()
        center = select_by_order(cate, top).mean(1)
        distances = []
        for chunk in permutations.split(self.candidate_chunk_size):
            values = select_by_order(cate, chunk)
            distances.append((values - center[:, None]).square().flatten(2).sum(-1))
        distances = torch.cat(distances, 1).scatter(1, ids, -torch.inf)
        additional = permutations[distances.argmax(-1)][:, None]
        return torch.cat((top, additional), 1)

    def predict(self, interaction):
        pred = super().predict(interaction)
        cate, _ = self._slate_inputs(interaction)
        orders = pred.candidates.orders
        stats = candidate_statistics(select_by_order(cate, orders), orders)
        return pred.with_extra(stats)
