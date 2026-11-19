"""PRS: supervise the first exposure-weighted beam candidate with aligned labels."""
from .pier import PIERRerankModel
from .base import select_by_order, masked_bce_with_logits


class PRSRerankModel(PIERRerankModel):
    def _supervised_loss(self, interaction, inputs):
        cate, history, dense, pages = inputs
        permutations = self.retriever.full_permutations
        ids = self.beam_retriever(select_by_order(dense, permutations).squeeze(-1), 1)
        order = permutations[ids]
        logits = self.candidate_logits(cate, history, dense, pages, order)[:, 0]
        # The TF source feeds sigmoid probabilities as logits and leaves labels
        # in the input order. Use raw logits and transport labels/masks together.
        labels = select_by_order(interaction[self.LABEL], order)[:, 0]
        mask = select_by_order(self._mask(interaction, dense), order)[:, 0]
        return masked_bce_with_logits(logits, labels, mask, self.loss_reduction)
