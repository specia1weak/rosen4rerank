"""MultiGE metric variants: supervise the best generated candidate and report diversity."""
import torch

from .mge import MultiGERerankModel
from .presets import resolve_rerank_config
from .base import select_by_order, masked_bce_with_logits
from .components.diversity import candidate_statistics


class MultiGEMetricRerankModel(MultiGERerankModel):
    preset_name = "mge_metric"

    def __init__(self, manager, **kwargs):
        super().__init__(manager, **resolve_rerank_config(self.preset_name, **kwargs))

    def loss_components(self, interaction):
        if self.LABEL is None:
            raise ValueError("Training a reranker requires a label field.")
        inputs = self._inputs(interaction)
        cate, history, dense, pages = inputs
        labels, slot_mask = interaction[self.LABEL], self._mask(interaction, cate)
        _, train_aux = self._generate(*inputs, labels=labels, label_mask=slot_mask, teacher_forcing=True)
        terms = self._generator_loss_components(labels, slot_mask, train_aux)
        # Teacher-forced loss and greedy legal selection are distinct passes.
        # Selection builds no unused generator graph; only the chosen judge loss
        # updates current/history embeddings and OCPM.
        with torch.no_grad():
            orders, aux = self._generate(*inputs, teacher_forcing=False)
        logits = self.candidate_logits(
            cate, history, dense, pages, orders,
            candidate_dense=aux["candidate_dense"] if self.candidate_dense_mode == "generator" else None)
        mask = select_by_order(self._mask(interaction, cate), orders)
        best = (logits.sigmoid() * mask).mean(-1).argmax(-1)
        rows = torch.arange(cate.shape[0], device=cate.device)
        labels = select_by_order(interaction[self.LABEL], orders)
        terms["evaluator"] = masked_bce_with_logits(logits[rows, best], labels[rows, best],
                                                    mask[rows, best], self.loss_reduction)
        if self.use_contrastive_loss:
            terms["contrastive"] = self._contrastive_loss(inputs, slot_mask, train_aux)
        return terms

    def predict(self, interaction):
        pred = super().predict(interaction)
        cate, _, _, _ = self._inputs(interaction)
        orders = pred.candidates.orders
        stats = candidate_statistics(select_by_order(cate, orders), orders)
        return pred.with_extra(stats)


class SubmodularMultiGEMetricRerankModel(MultiGEMetricRerankModel):
    preset_name = "mge_cosine_metric"
