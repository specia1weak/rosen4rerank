"""Multi-generator reranking with a shared supervised OCPM evaluator."""

from typing import Literal, Optional, Sequence

import torch

from rosen.core.prediction import CandidatePool, Prediction
from .base import select_by_order, masked_bce_with_logits
from .slate import OCPMRerankModel
from .components.attention import OAMBlock, IntraPOISelfAttention, PageTargetAttention
from .components.retrieval import PPMLSimHashRetriever, BeamSearchRetriever
from .components.scoring import EncodedSlate, score_candidate_probs
from .components.generators import (PointwiseGenerator, ListwiseGenerator, AutoregressiveGenerator,
                                    PointerHead, cosine_coverage, lc_exact_score, lc_soft_score)


class MultiGERerankModel(OCPMRerankModel):
    def __init__(
        self, manager, *, slate_dense_field: str,
        slate_cate_fields: Optional[Sequence[str]] = None, slate_cate_group: Optional[str] = None,
        hist_cate_fields: Optional[Sequence[str]] = None, hist_cate_group: Optional[str] = None,
        page_mask_field: Optional[str] = None, label_field: Optional[str] = None,
        poi_num: int = 5, page_num: int = 5, top_k: int = 20,
        submodularity_mode: Literal["none", "cosine", "lc", "lc_soft"] = "lc",
        mlp_a: Sequence[int] = (60, 32, 20), mlp_b: Sequence[int] = (128, 32),
        hidden_dim: int = 64, attention_scale: float = 8.0,
        share_pointer_head: bool = True, lc_weight: float = 1.0, lc_temperature: float = 1.0,
        loss_reduction: Literal["mean", "valid_mean", "nonzero_mean"] = "mean",
        use_contrastive_loss: bool = False, contrastive_weight: float = 1.0,
        eval_all_perms: bool = True, eval_diagnostic_pools: bool = True, candidate_chunk_size: int = 32,
        candidate_dense_mode: Literal["prior", "generator"] = "prior",
        position_encoding=None, hash_vector=None, distance_weight=None, expose_rate=None,
        retrieval_seed: int = 2026,
    ):
        super().__init__(manager, slate_dense_field=slate_dense_field,
                         slate_cate_fields=slate_cate_fields, slate_cate_group=slate_cate_group,
                         hist_cate_fields=hist_cate_fields, hist_cate_group=hist_cate_group,
                         page_mask_field=page_mask_field, label_field=label_field,
                         poi_num=poi_num, page_num=page_num, loss_reduction=loss_reduction,
                         candidate_chunk_size=candidate_chunk_size)
        self.require_history()
        if submodularity_mode not in ("none", "cosine", "lc", "lc_soft"):
            raise ValueError("submodularity_mode must be none, cosine, lc or lc_soft.")
        if not isinstance(page_num, int) or page_num <= 0 or hidden_dim <= 0 or candidate_chunk_size <= 0:
            raise ValueError("Page count, hidden width and candidate chunk size must be positive.")
        if top_k <= 0 or loss_reduction not in ("mean", "valid_mean", "nonzero_mean") or lc_temperature <= 0:
            raise ValueError("Invalid top_k, loss_reduction or lc_temperature.")
        if candidate_dense_mode not in ("prior", "generator"):
            raise ValueError("candidate_dense_mode must be prior or generator.")
        self.submodularity_mode = submodularity_mode
        self.lc_weight, self.lc_temperature = lc_weight, lc_temperature
        self.use_contrastive_loss, self.contrastive_weight = use_contrastive_loss, contrastive_weight
        self.eval_all_perms, self.candidate_chunk_size = eval_all_perms, candidate_chunk_size
        self.eval_diagnostic_pools = eval_diagnostic_pools
        self.candidate_dense_mode = candidate_dense_mode

        f, d = self.num_attrs, self.emb_dim
        self.retriever = PPMLSimHashRetriever(poi_num, page_num, d, position_encoding=position_encoding,
                                              hash_vector=hash_vector, distance_weight=distance_weight,
                                              seed=retrieval_seed)
        self.beam_retriever = BeamSearchRetriever(poi_num, expose_rate)
        self.top_k = min(top_k, self.retriever.full_permutations.shape[0])
        rng = torch.Generator().manual_seed(retrieval_seed + 1)
        self.register_buffer("random_candidate_ids", torch.randperm(
            self.retriever.full_permutations.shape[0], generator=rng)[:self.top_k])

        # Shared modules are registered exactly once. Every caller receives them
        # explicitly; deepcopy/EMA/state_dict preserve one unambiguous owner.
        self.oam = OAMBlock(poi_num, f, d, attention_scale)
        self.intra_poi = IntraPOISelfAttention(f, d, attention_scale)
        self.target_attention = PageTargetAttention()
        self.evaluator = ListwiseGenerator(f, d, mlp_a, mlp_b)
        self.pointwise_generator = PointwiseGenerator()
        self.listwise_generator = ListwiseGenerator(f, d, mlp_a, mlp_b)
        self.generative_generator = AutoregressiveGenerator(f, d, mlp_a, hidden_dim)
        self.pointer_head = PointerHead(hidden_dim)
        # Optional modules come after all common initialization (paired variants).
        self.submodularity_generator = (AutoregressiveGenerator(f, d, mlp_a, hidden_dim)
                                        if submodularity_mode != "none" else None)
        self.submodularity_pointer_head = (PointerHead(hidden_dim)
                                           if self.submodularity_generator is not None and not share_pointer_head else None)

    def _generate(self, cate, history, dense, page_mask, *, labels=None, label_mask=None, teacher_forcing=False):
        cate, history, dense, page_mask = (value.detach() for value in (cate, history, dense, page_mask))
        distance = self.retriever(cate, history, page_mask)
        # Pointwise logits, labels and proposed item indices share input-slot coordinates.
        point_logits, point_indices = self.pointwise_generator(dense)
        common = (self.oam, self.intra_poi, self.target_attention)
        list_logits, list_indices = self.listwise_generator(cate, self.oam(history)[0], dense, page_mask, *common)
        generative = self.generative_generator(cate, history, dense, page_mask, *common, self.pointer_head,
                                               labels=labels, label_mask=label_mask, teacher_forcing=teacher_forcing)
        indices = torch.cat((point_indices, list_indices, generative.indices), 1)
        # Score-based heads output input-slot scores; OCPM candidate dense features
        # must follow each proposed output order. AR maxima already follow decode steps.
        candidate_dense = [select_by_order(point_logits.sigmoid(), point_indices)[:, 0],
                           select_by_order(list_logits.sigmoid(), list_indices)[:, 0], generative.max_scores]
        aux = {"pointwise_logits": point_logits, "listwise_logits": list_logits,
               "generative": generative, "distance": distance}
        if self.submodularity_generator is not None:
            coverage = (cosine_coverage(cate, select_by_order(cate, indices))
                        if self.submodularity_mode == "cosine" else None)
            head = self.submodularity_pointer_head if self.submodularity_pointer_head is not None else self.pointer_head
            sub = self.submodularity_generator(cate, history, dense, page_mask, *common, head,
                                               labels=labels, label_mask=label_mask, coverage=coverage,
                                               teacher_forcing=teacher_forcing)
            if self.submodularity_mode == "lc_soft":
                aux["coverage"] = lc_soft_score(sub.probabilities, indices, self.lc_temperature).mean()
            indices = torch.cat((indices, sub.indices), 1)
            candidate_dense.append(sub.max_scores)
            if self.submodularity_mode == "lc":
                aux["coverage"] = lc_exact_score(indices).mean()
            aux["submodularity"] = sub
        aux["candidate_dense"] = torch.stack(candidate_dense, 1).unsqueeze(-1).detach()
        return indices, aux

    def _score_candidates(self, cate, history, dense, page_mask, mask, permutations, *, candidate_dense=None):
        encoded = EncodedSlate(cate, dense, mask, self._history_context(history, page_mask))
        return score_candidate_probs(
            encoded, permutations, scorer=self._score_encoded,
            chunk_size=self.candidate_chunk_size, candidate_dense=candidate_dense)

    def _generator_loss_components(self, labels, mask, aux):
        self.loss_diagnostics = {}
        terms = {
            "gen.pointwise": masked_bce_with_logits(aux["pointwise_logits"], labels, mask, self.loss_reduction),
            "gen.listwise": masked_bce_with_logits(aux["listwise_logits"], labels, mask, self.loss_reduction),
            "gen.generative": aux["generative"].loss,
        }
        if self.submodularity_generator is not None:
            terms["gen.submodularity"] = aux["submodularity"].loss
            if self.submodularity_mode == "lc_soft":
                terms["gen.coverage"] = -self.lc_weight * aux["coverage"]
            elif self.submodularity_mode == "lc":
                # This discrete statistic is diagnostic, not an optimization term.
                self.loss_diagnostics["lc_exact"] = aux["coverage"].detach()
        return terms

    def _contrastive_loss(self, inputs, mask, aux):
        cate, history, dense, page_mask = inputs
        perms = self.retriever.full_permutations
        best = perms[self.retriever.top_k(aux["distance"], self.top_k)]
        worst = perms[self.retriever.last_k(aux["distance"], self.top_k)]
        best_scores = self._score_candidates(cate.detach(), history, dense.detach(), page_mask, mask, best)
        worst_scores = self._score_candidates(cate.detach(), history, dense.detach(), page_mask, mask, worst)
        return -self.contrastive_weight * (best_scores.mean() - worst_scores.mean())

    def loss_components(self, interaction):
        if self.LABEL is None:
            raise ValueError("Training a reranker requires a label field.")
        inputs = self._inputs(interaction)
        cate, history, dense, page_mask = inputs
        labels, mask = interaction[self.LABEL], self._mask(interaction, cate)
        _, aux = self._generate(*inputs, labels=labels, label_mask=mask, teacher_forcing=True)
        terms = self._generator_loss_components(labels, mask, aux)
        evaluator_logits = self._evaluator_logits(*inputs)
        terms["evaluator"] = masked_bce_with_logits(evaluator_logits, labels, mask, self.loss_reduction)
        if self.use_contrastive_loss:
            terms["contrastive"] = self._contrastive_loss(inputs, mask, aux)
        return terms

    def predict(self, interaction):
        cate, history, dense, page_mask = self._inputs(interaction)
        mask = self._mask(interaction, cate)
        generated, aux = self._generate(cate, history, dense, page_mask, teacher_forcing=False)
        original = self._evaluator_logits(cate, history, dense, page_mask)
        candidates = {"ctr_topk": generated}
        if self.eval_diagnostic_pools:
            permutations = self.retriever.full_permutations
            all_priors = select_by_order(dense, permutations).squeeze(-1)
            candidates.update({
                "ctr_lastk": permutations[self.retriever.last_k(aux["distance"], self.top_k)],
                "ctr_randk": permutations[self.random_candidate_ids],
                "ctr_beamk": permutations[self.beam_retriever(all_priors, self.top_k)],
            })
            if self.eval_all_perms:
                candidates["ctr_all"] = permutations
        scores = {name: self._score_candidates(
                      cate, history, dense, page_mask, mask, indices,
                      candidate_dense=(aux["candidate_dense"]
                                       if name == "ctr_topk" and self.candidate_dense_mode == "generator" else None))
                  for name, indices in candidates.items()}
        pool = CandidatePool(scores=scores["ctr_topk"], orders=generated)
        chosen_order, chosen_scores = pool.select(pool.scores.mean(-1).argmax(-1))
        extra = {name.removeprefix("ctr_"): value for name, value in scores.items() if name != "ctr_topk"}
        extra.update({f"{name.removeprefix('ctr_')}_order": indices.expand(cate.shape[0], -1, -1)
                      if indices.ndim == 2 else indices for name, indices in candidates.items() if name != "ctr_topk"})
        return Prediction(scores=original.sigmoid(), order=chosen_order, chosen_scores=chosen_scores,
                          candidates=pool, target=self.LABEL, extra=extra)
