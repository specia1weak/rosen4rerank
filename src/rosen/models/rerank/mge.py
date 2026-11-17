"""Multi-generator reranking with a shared supervised OCPM evaluator."""

from typing import Literal, Optional, Sequence

import torch

from rosen.core.prediction import Prediction
from rosen.emb.emblayer import EmbView
from .base import RerankModel, apply_permutation, fold_history_pages, masked_bce_with_logits
from .components.attention import OAMBlock, IntraPOISelfAttention, PageTargetAttention
from .components.retrieval import PPMLSimHashRetriever, BeamSearchRetriever
from .components.generators import (PointwiseGenerator, ListwiseGenerator, AutoregressiveGenerator,
                                    PointerHead, cosine_coverage, lc_exact_score, lc_soft_score)


class MultiGERerankModel(RerankModel):
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
        loss_reduction: Literal["mean", "valid_mean"] = "mean",
        use_contrastive_loss: bool = False, contrastive_weight: float = 1.0,
        eval_all_perms: bool = True, candidate_chunk_size: int = 32,
        position_encoding=None, hash_vector=None, distance_weight=None, expose_rate=None,
        retrieval_seed: int = 2026,
    ):
        super().__init__(manager)
        if submodularity_mode not in ("none", "cosine", "lc", "lc_soft"):
            raise ValueError("submodularity_mode must be none, cosine, lc or lc_soft.")
        if not isinstance(page_num, int) or page_num <= 0 or hidden_dim <= 0 or candidate_chunk_size <= 0:
            raise ValueError("Page count, hidden width and candidate chunk size must be positive.")
        if top_k <= 0 or loss_reduction not in ("mean", "valid_mean") or lc_temperature <= 0:
            raise ValueError("Invalid top_k, loss_reduction or lc_temperature.")
        self.poi_num, self.page_num = poi_num, page_num
        self.submodularity_mode = submodularity_mode
        self.slate_dense_field, self.page_mask_field = slate_dense_field, page_mask_field
        self.LABEL = label_field or self.LABEL
        self.loss_reduction = loss_reduction
        self.lc_weight, self.lc_temperature = lc_weight, lc_temperature
        self.use_contrastive_loss, self.contrastive_weight = use_contrastive_loss, contrastive_weight
        self.eval_all_perms, self.candidate_chunk_size = eval_all_perms, candidate_chunk_size

        self.slate_view = self._view(slate_cate_fields, slate_cate_group, poi_num)
        self.history_view = self._view(hist_cate_fields, hist_cate_group, page_num * poi_num)
        if self.slate_view.num_fields != self.history_view.num_fields or self.slate_view.field_dim != self.history_view.field_dim:
            raise ValueError("Slate/history views must have equal field counts and embedding dimensions.")
        if tuple(s.embedding_field_name for s in self.slate_view.selected_settings) != tuple(
            s.embedding_field_name for s in self.history_view.selected_settings
        ):
            raise ValueError("Slate/history fields must be aligned and share the same atomic embedding settings.")
        f, d = self.slate_view.num_fields, self.slate_view.field_dim
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

    def _view(self, fields, group, length):
        if (fields is None) == (group is None):
            raise ValueError("Specify exactly one of cate_fields or cate_group for each view.")
        view = self.omni_embedding.seq_groups[group] if group is not None else EmbView(
            self.omni_embedding, include_fields=fields,
        )
        selected = view.selected_settings
        if not selected or (fields is not None and set(fields) != {s.field_name for s in selected}):
            raise ValueError("Categorical views must select existing, non-empty sequence fields.")
        if any(not s.is_sequence_setting or s.max_len != length for s in selected):
            raise ValueError(f"Each categorical field must be a sequence with max_len={length}.")
        return view

    def _inputs(self, interaction):
        cate = self.slate_view(interaction, split_by="stack")
        history = fold_history_pages(self.history_view(interaction, split_by="stack"), self.page_num, self.poi_num)
        dense = interaction[self.slate_dense_field]
        if dense.ndim == 2:
            dense = dense.unsqueeze(-1)
        if dense.shape != cate.shape[:2] + (1,):
            raise ValueError("Prior scores must have shape [B,K] or [B,K,1].")
        page_mask = (interaction[self.page_mask_field] if self.page_mask_field is not None
                     else cate.new_ones(cate.shape[0], self.page_num))
        if page_mask.shape != (cate.shape[0], self.page_num):
            raise ValueError("page_mask must have shape [B,P].")
        return cate, history, dense.to(cate.dtype), page_mask

    def _mask(self, interaction, cate):
        spec = self.manager.get_label_spec(self.LABEL) if self.LABEL else None
        if spec is not None and spec.mask_field and spec.mask_field in interaction:
            mask = interaction[spec.mask_field].to(cate.dtype)
        else:
            mask = cate.new_ones(cate.shape[:2])
        if mask.shape != cate.shape[:2]:
            raise ValueError("Supervision mask must have shape [B,K].")
        return mask

    def _generate(self, cate, history, dense, page_mask, *, labels=None, label_mask=None, teacher_forcing=False):
        cate, history, dense, page_mask = (value.detach() for value in (cate, history, dense, page_mask))
        distance = self.retriever(cate, history, page_mask)
        best = self.retriever.full_permutations[self.retriever.top_k(distance, 1)]
        top_prior = apply_permutation(dense, best)[:, 0]
        point_logits, point_indices = self.pointwise_generator(top_prior)
        common = (self.oam, self.intra_poi, self.target_attention)
        list_logits, list_indices = self.listwise_generator(cate, history, dense, page_mask, *common)
        generative = self.generative_generator(cate, history, dense, page_mask, *common, self.pointer_head,
                                               labels=labels, label_mask=label_mask, teacher_forcing=teacher_forcing)
        indices = torch.cat((point_indices, list_indices, generative.indices), 1)
        aux = {"pointwise_logits": point_logits, "listwise_logits": list_logits,
               "generative": generative, "distance": distance}
        if self.submodularity_generator is not None:
            coverage = (cosine_coverage(cate, apply_permutation(cate, indices))
                        if self.submodularity_mode == "cosine" else None)
            head = self.submodularity_pointer_head if self.submodularity_pointer_head is not None else self.pointer_head
            sub = self.submodularity_generator(cate, history, dense, page_mask, *common, head,
                                               labels=labels, label_mask=label_mask, coverage=coverage,
                                               teacher_forcing=teacher_forcing)
            if self.submodularity_mode == "lc_soft":
                aux["coverage"] = lc_soft_score(sub.probabilities, indices, self.lc_temperature).mean()
            indices = torch.cat((indices, sub.indices), 1)
            if self.submodularity_mode == "lc":
                aux["coverage"] = lc_exact_score(indices).mean()
            aux["submodularity"] = sub
        return indices, aux

    def _score_candidates(self, cate, history, dense, page_mask, mask, permutations):
        """Chunk the permutation axis; preserve sample-major history alignment."""
        if permutations.ndim == 2:
            permutations = permutations[None].expand(cate.shape[0], -1, -1)
        outputs = []
        for chunk in permutations.split(self.candidate_chunk_size, dim=1):
            b, m, k = chunk.shape
            candidates = apply_permutation(cate, chunk).flatten(0, 1)
            priors = apply_permutation(dense, chunk).flatten(0, 1)
            history_expanded = history[:, None].expand(-1, m, -1, -1, -1, -1).reshape(b * m, *history.shape[1:])
            masks_expanded = page_mask[:, None].expand(-1, m, -1).reshape(b * m, -1)
            logits, _ = self.evaluator(candidates, history_expanded, priors, masks_expanded,
                                        self.oam, self.intra_poi, self.target_attention)
            outputs.append(logits.sigmoid().reshape(b, m, k) * apply_permutation(mask, chunk))
        return torch.cat(outputs, dim=1)

    def loss_components(self, interaction):
        if self.LABEL is None:
            raise ValueError("Training a reranker requires a label field.")
        cate, history, dense, page_mask = self._inputs(interaction)
        labels, mask = interaction[self.LABEL], self._mask(interaction, cate)
        _, aux = self._generate(cate, history, dense, page_mask, labels=labels,
                                label_mask=mask, teacher_forcing=True)
        evaluator_logits, _ = self.evaluator(cate, history, dense, page_mask,
                                              self.oam, self.intra_poi, self.target_attention)
        terms = {
            "gen.pointwise": masked_bce_with_logits(aux["pointwise_logits"], labels, mask, self.loss_reduction),
            "gen.listwise": masked_bce_with_logits(aux["listwise_logits"], labels, mask, self.loss_reduction),
            "gen.generative": aux["generative"].loss,
            "evaluator": masked_bce_with_logits(evaluator_logits, labels, mask, self.loss_reduction),
        }
        if self.submodularity_generator is not None:
            sub_loss = aux["submodularity"].loss
            if self.submodularity_mode in ("lc", "lc_soft"):
                sub_loss = sub_loss - self.lc_weight * aux["coverage"]
            terms["gen.submodularity"] = sub_loss
        if self.use_contrastive_loss:
            perms = self.retriever.full_permutations
            best = perms[self.retriever.top_k(aux["distance"], self.top_k)]
            worst = perms[self.retriever.last_k(aux["distance"], self.top_k)]
            best_scores = self._score_candidates(cate.detach(), history, dense.detach(), page_mask, mask, best)
            worst_scores = self._score_candidates(cate.detach(), history, dense.detach(), page_mask, mask, worst)
            terms["contrastive"] = -self.contrastive_weight * (best_scores.mean() - worst_scores.mean())
        return terms

    def calculate_loss(self, interaction):
        terms = self.loss_components(interaction)
        if self.lsp.active:
            for name, value in terms.items():
                self.lsp.add(name, value)
        return sum(terms.values())

    def predict(self, interaction):
        cate, history, dense, page_mask = self._inputs(interaction)
        mask = self._mask(interaction, cate)
        generated, aux = self._generate(cate, history, dense, page_mask, teacher_forcing=False)
        original, _ = self.evaluator(cate, history, dense, page_mask,
                                     self.oam, self.intra_poi, self.target_attention)
        permutations = self.retriever.full_permutations
        all_priors = apply_permutation(dense, permutations).squeeze(-1)
        candidates = {
            "ctr_topk": generated,
            "ctr_lastk": permutations[self.retriever.last_k(aux["distance"], self.top_k)],
            "ctr_randk": permutations[self.random_candidate_ids],
            "ctr_beamk": permutations[self.beam_retriever(all_priors, self.top_k)],
        }
        if self.eval_all_perms:
            candidates["ctr_all"] = permutations
        scores = {name: self._score_candidates(cate, history, dense, page_mask, mask, indices)
                  for name, indices in candidates.items()}
        best = scores["ctr_topk"].mean(-1).argmax(-1)
        chosen = generated[torch.arange(cate.shape[0], device=cate.device), best][:, None]
        probabilities = original.sigmoid()
        extra = {name.removeprefix("ctr_"): value for name, value in scores.items()}
        extra.update({f"{name.removeprefix('ctr_')}_order": indices.expand(cate.shape[0], -1, -1)
                      if indices.ndim == 2 else indices for name, indices in candidates.items()})
        return Prediction(scores=probabilities, order=chosen.squeeze(1), target=self.LABEL, extra=extra)
