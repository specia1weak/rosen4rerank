"""Shared schema/input and supervised output contract for independent rerankers."""
from typing import Optional, Sequence

import torch

from rosen.core.prediction import Prediction
from rosen.emb.emblayer import EmbView
from .base import RerankModel, fold_history_pages, masked_bce_with_logits
from .components.scoring import EncodedSlate, OCPMContext, score_candidate_logits, score_candidate_probs


class SlateRerankModel(RerankModel):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if {'logits', 'score_permutations'} & cls.__dict__.keys():
            raise TypeError("Slate models implement _encode_items/_encode_context/_score_encoded; logits and score_permutations are fixed entry points.")

    def __init__(self, manager, *, slate_dense_field: str,
                 slate_cate_fields: Optional[Sequence[str]] = None, slate_cate_group: Optional[str] = None,
                 hist_cate_fields: Optional[Sequence[str]] = None, hist_cate_group: Optional[str] = None,
                 page_mask_field: Optional[str] = None, label_field: Optional[str] = None,
                 poi_num: int = 5, page_num: int = 5, loss_reduction: str = "nonzero_mean",
                 candidate_chunk_size: int = 32):
        super().__init__(manager)
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (poi_num, page_num)):
            raise ValueError("poi_num and page_num must be positive integers.")
        if loss_reduction not in ("mean", "valid_mean", "nonzero_mean"):
            raise ValueError("Invalid loss_reduction.")
        if isinstance(candidate_chunk_size, bool) or not isinstance(candidate_chunk_size, int) or candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be a positive integer.")
        self.candidate_chunk_size = candidate_chunk_size
        self.poi_num, self.page_num = poi_num, page_num
        self.slate_dense_field, self.page_mask_field = slate_dense_field, page_mask_field
        self.LABEL = label_field or self.LABEL
        self.loss_reduction = loss_reduction
        self.slate_view = self._view(slate_cate_fields, slate_cate_group, poi_num)
        self.history_view = (self._view(hist_cate_fields, hist_cate_group, page_num * poi_num)
                             if hist_cate_fields is not None or hist_cate_group is not None else None)
        self.num_attrs, self.emb_dim = self.slate_view.num_fields, self.slate_view.field_dim
        self.item_dim = self.num_attrs * self.emb_dim
        if self.history_view is not None:
            current = tuple(s.embedding_field_name for s in self.slate_view.selected_settings)
            history = tuple(s.embedding_field_name for s in self.history_view.selected_settings)
            if current != history or self.history_view.field_dim != self.emb_dim:
                raise ValueError("Current/history attributes must align and share atomic embeddings.")

    def _view(self, fields, group, length):
        if (fields is None) == (group is None):
            raise ValueError("Specify exactly one of cate_fields or cate_group per view.")
        view = self.omni_embedding.seq_groups[group] if group is not None else EmbView(
            self.omni_embedding, include_fields=fields)
        selected = view.selected_settings
        if not selected or (fields is not None and set(fields) != {s.field_name for s in selected}):
            raise ValueError("Select existing non-empty sequence fields.")
        if any(not s.is_sequence_setting or s.max_len != length for s in selected):
            raise ValueError(f"Categorical fields must be sequences of max_len={length}.")
        return view

    def require_history(self):
        if self.history_view is None:
            raise ValueError(f"{type(self).__name__} requires hist_cate_fields or hist_cate_group.")

    def _slate_inputs(self, interaction):
        cate = self.slate_view(interaction, split_by="stack")
        dense = interaction[self.slate_dense_field]
        if dense.ndim == 2:
            dense = dense.unsqueeze(-1)
        if dense.shape != cate.shape[:2] + (1,):
            raise ValueError("Prior scores must be [B,K] or [B,K,1].")
        return cate, dense.to(cate.dtype)

    def _inputs(self, interaction):
        cate, dense = self._slate_inputs(interaction)
        history, pages = self._history_inputs(interaction, cate)
        return cate, history, dense, pages

    def _history_inputs(self, interaction, reference):
        self.require_history()
        history = fold_history_pages(self.history_view(interaction, split_by="stack"), self.page_num, self.poi_num)
        pages = (interaction[self.page_mask_field] if self.page_mask_field else
                 reference.new_ones(reference.shape[0], self.page_num))
        if pages.shape != (reference.shape[0], self.page_num):
            raise ValueError("page_mask must be [B,P].")
        return history, pages

    def _mask(self, interaction, reference):
        spec = self.manager.get_label_spec(self.LABEL) if self.LABEL else None
        mask = (interaction[spec.mask_field] if spec and spec.mask_field and spec.mask_field in interaction
                else reference.new_ones(reference.shape[:2]))
        if mask.shape != reference.shape[:2]:
            raise ValueError("Slot mask must be [B,K].")
        return mask.to(reference.dtype)

    def _encode_items(self, cate, dense, mask):
        return cate

    def _encode_context(self, interaction, cate, dense):
        return None

    def encode_slate(self, interaction) -> EncodedSlate:
        """Lookup items once and encode history once, before any candidate expansion."""
        cate, dense = self._slate_inputs(interaction)
        mask = self._mask(interaction, dense)
        return EncodedSlate(self._encode_items(cate, dense, mask), dense, mask,
                            self._encode_context(interaction, cate, dense))

    def _score_encoded(self, items, dense, mask, context):
        """Return logits [B,M,N] using explicit candidate slots and fixed context."""
        raise NotImplementedError

    def logits(self, interaction):
        encoded = self.encode_slate(interaction)
        return self._score_encoded(encoded.items[:, None], encoded.dense[:, None],
                                   encoded.mask[:, None], encoded.context)[:, 0]

    def _validate_scored_orders(self, interaction, orders):
        if (not isinstance(orders, torch.Tensor) or orders.dtype != torch.long or orders.ndim != 3
                or orders.shape[0] != len(interaction) or not orders.shape[1]
                or not 1 <= orders.shape[2] <= self.poi_num):
            raise ValueError("score_permutations requires LongTensor[B,M,K], M > 0 and 1 <= K <= N.")
        sorted_orders = orders.sort(-1).values
        if ((sorted_orders[..., 0] < 0).any() or (sorted_orders[..., -1] >= self.poi_num).any()
                or (sorted_orders[..., 1:] <= sorted_orders[..., :-1]).any()):
            raise ValueError("Each scored order must select unique indices in [0,N).")
        if self.requires_full_slate and orders.shape[2] != self.poi_num:
            raise ValueError(f"{type(self).__name__} requires full slate width K == {self.poi_num}")
        if orders.device != interaction[self.slate_dense_field].device:
            raise ValueError("Scored orders must be on the same device as slate features.")

    def loss_components(self, interaction):
        if self.LABEL is None:
            raise ValueError("Supervised training requires a label field.")
        logits = self.logits(interaction)
        return {"supervised": masked_bce_with_logits(
            logits, interaction[self.LABEL], self._mask(interaction, logits), self.loss_reduction)}

    def score_permutations(self, interaction, orders):
        """One encoded, chunked path for all supported full and partial slates."""
        self._validate_scored_orders(interaction, orders)
        return score_candidate_probs(self.encode_slate(interaction), orders,
                                     scorer=self._score_encoded, chunk_size=self.candidate_chunk_size)

    def calculate_loss(self, interaction):
        terms = self.loss_components(interaction)
        if self.lsp.active:
            for name, value in terms.items():
                self.lsp.add(name, value)
        return sum(terms.values())

    def predict(self, interaction):
        logits = self.logits(interaction)
        valid = self._mask(interaction, logits) > 0
        order = logits.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        return Prediction(scores=logits.sigmoid(), order=order, target=self.LABEL)


class OCPMRerankModel(SlateRerankModel):
    """Shared fixed-width judge with OAM history encoded before candidate chunks."""
    requires_full_slate = True
    outputs_candidate_pool = True

    def _history_context(self, history, pages):
        return OCPMContext(self.oam(history)[0], pages)

    def _encode_context(self, interaction, cate, dense):
        return self._history_context(*self._history_inputs(interaction, cate))

    def _score_encoded(self, items, dense, mask, context: OCPMContext):
        return self.evaluator(items, context.history_pages, dense, context.page_mask,
                              self.oam, self.intra_poi, self.target_attention)[0]

    def _evaluator_logits(self, cate, history, dense, pages):
        return self.evaluator(cate, self.oam(history)[0], dense, pages,
                              self.oam, self.intra_poi, self.target_attention)[0]

    def candidate_logits(self, cate, history, dense, pages, orders, *, candidate_dense=None):
        encoded = EncodedSlate(cate, dense, dense.new_ones(cate.shape[:2]), self._history_context(history, pages))
        return score_candidate_logits(encoded, orders, scorer=self._score_encoded,
                                       chunk_size=self.candidate_chunk_size, candidate_dense=candidate_dense)
