"""PIER: PPML/SimHash retrieval followed by supervised OCPM candidate selection."""
import torch

from rosen.core.prediction import CandidatePool, Prediction
from .slate import OCPMRerankModel
from .base import select_by_order, masked_bce_with_logits
from .components.attention import OAMBlock, IntraPOISelfAttention, PageTargetAttention
from .components.generators import ListwiseGenerator
from .components.retrieval import PPMLSimHashRetriever, BeamSearchRetriever


class PIERRerankModel(OCPMRerankModel):
    def __init__(self, manager, *, top_k=20, mlp_a=(60, 32, 20), mlp_b=(128, 32),
                 attention_scale=8., use_contrastive_loss=True, contrastive_weight=.01,
                 eval_all_perms=True, candidate_chunk_size=32, position_encoding=None,
                 hash_vector=None, distance_weight=None, expose_rate=None, retrieval_seed=2026, **kwargs):
        super().__init__(manager, candidate_chunk_size=candidate_chunk_size, **kwargs)
        self.require_history()
        if not isinstance(top_k, int) or top_k <= 0 or candidate_chunk_size <= 0:
            raise ValueError("top_k and candidate_chunk_size must be positive.")
        self.retriever = PPMLSimHashRetriever(
            self.poi_num, self.page_num, self.emb_dim, position_encoding=position_encoding,
            hash_vector=hash_vector, distance_weight=distance_weight, seed=retrieval_seed)
        self.beam_retriever = BeamSearchRetriever(self.poi_num, expose_rate)
        self.top_k = min(top_k, len(self.retriever.full_permutations))
        rng = torch.Generator().manual_seed(retrieval_seed + 1)
        self.register_buffer("random_candidate_ids", torch.randperm(
            len(self.retriever.full_permutations), generator=rng)[:self.top_k])
        self.candidate_chunk_size, self.eval_all_perms = candidate_chunk_size, eval_all_perms
        self.use_contrastive_loss, self.contrastive_weight = use_contrastive_loss, contrastive_weight
        self.oam = OAMBlock(self.poi_num, self.num_attrs, self.emb_dim, attention_scale)
        self.intra_poi = IntraPOISelfAttention(self.num_attrs, self.emb_dim, attention_scale)
        self.target_attention = PageTargetAttention()
        self.evaluator = ListwiseGenerator(self.num_attrs, self.emb_dim, mlp_a, mlp_b)

    def _top_orders(self, cate, distance):
        return self.retriever.full_permutations[self.retriever.top_k(distance, self.top_k)]

    def candidate_orders(self, cate, history, dense, pages):
        permutations = self.retriever.full_permutations
        distance = self.retriever(cate.detach(), history.detach(), pages)
        result = {
            "topk": self._top_orders(cate, distance),
            "lastk": permutations[self.retriever.last_k(distance, self.top_k)],
            "randk": permutations[self.random_candidate_ids],
            "beamk": permutations[self.beam_retriever(select_by_order(dense, permutations).squeeze(-1), self.top_k)],
        }
        if self.eval_all_perms:
            result["all"] = permutations
        return result

    def _supervised_loss(self, interaction, inputs):
        logits = self._evaluator_logits(*inputs)
        return masked_bce_with_logits(logits, interaction[self.LABEL], self._mask(interaction, logits), self.loss_reduction)

    def loss_components(self, interaction):
        if self.LABEL is None:
            raise ValueError("Supervised training requires a label field.")
        inputs = self._inputs(interaction)
        terms = {"supervised": self._supervised_loss(interaction, inputs)}
        if self.use_contrastive_loss:
            cate, history, dense, pages = inputs
            mask = self._mask(interaction, dense)
            permutations = self.retriever.full_permutations
            distance = self.retriever(cate.detach(), history.detach(), pages)
            scores = []
            for orders in (self._top_orders(cate, distance),
                           permutations[self.retriever.last_k(distance, self.top_k)]):
                logits = self.candidate_logits(cate.detach(), history, dense.detach(), pages, orders)
                scores.append((logits.sigmoid() * select_by_order(mask, orders)).mean())
            terms["contrastive"] = -self.contrastive_weight * (scores[0] - scores[1])
        return terms

    def predict(self, interaction):
        cate, history, dense, pages = self._inputs(interaction)
        mask = self._mask(interaction, dense)
        original = self._evaluator_logits(cate, history, dense, pages)
        orders = self.candidate_orders(cate, history, dense, pages)
        scores = {name: self.candidate_logits(cate, history, dense, pages, value).sigmoid()
                  * select_by_order(mask, value) for name, value in orders.items()}
        pool = CandidatePool(scores=scores["topk"], orders=orders["topk"])
        chosen_order, chosen_scores = pool.select(pool.scores.mean(-1).argmax(-1))
        extra = {name: value for name, value in scores.items() if name != "topk"}
        extra.update({f"{name}_order": value[None].expand(cate.shape[0], -1, -1)
                      if value.ndim == 2 else value for name, value in orders.items() if name != "topk"})
        return Prediction(scores=original.sigmoid(), order=chosen_order, chosen_scores=chosen_scores,
                          candidates=pool, target=self.LABEL, extra=extra)
