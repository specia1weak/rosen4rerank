"""Resolve raw sources and select output slots before metric collection."""
from collections.abc import Callable, Mapping
from typing import Literal

import torch

from .batch import EvalBatch
from .scorer import BaseScorer, PredictionScorer, EvalContext, LabelScorer, normalize_orders, gather_by_order


def _cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return tensor.detach().cpu() if tensor is not None else None


class BatchResolver:
    """Build one tensor view after filtering.

    - "ranking":    Keep [B, N] candidate tensors and pass `order [B, K]` to EvalBatch.
    - "score_only": Keep [B, N] candidate tensors and ignore `Prediction.order`.
    - "slate":      Apply `order [B, K]` or `[B, M, K]` inside scorers/masks to produce aligned [B, K] slate tensors.
    """

    def __init__(
        self,
        *,
        pred_scorer: BaseScorer | None = None,
        label_scorer: BaseScorer | None = None,
        order_field: str = "order",
        orders: torch.Tensor | None = None,
        ref_scorer: BaseScorer | None = None,
        view: Literal["ranking", "score_only", "slate"] = "ranking",
        mask_field: str | None = None,
    ):
        if view not in ("ranking", "score_only", "slate"):
            raise ValueError("view must be ranking, score_only or slate")
        self.pred_scorer = pred_scorer
        self.label_scorer = label_scorer if label_scorer is not None else LabelScorer()
        if not isinstance(order_field, str) or not order_field.strip():
            raise ValueError("order_field must be a non-empty string")
        self.order_field = order_field
        if orders is not None:
            if (not isinstance(orders, torch.Tensor) or orders.dtype != torch.long
                    or orders.ndim not in (2, 3) or (orders.ndim == 3 and orders.shape[0] != 1)):
                raise ValueError("Fixed shared orders must be [M,K] or [1,M,K] LongTensors")
            if view != "slate":
                raise ValueError("Fixed shared orders require view='slate'")
            # Explicitly shared, including M == B; never row-filter these constants.
            orders = orders[None] if orders.ndim == 2 else orders
            orders = orders.detach().clone()
        self.orders = orders
        self.ref_scorer = ref_scorer
        self.view = view
        self.mask_field = mask_field

    @classmethod
    def for_candidates(cls, *, label_scorer: BaseScorer | None = None,
                       ref_scorer: BaseScorer | None = None, mask_field: str | None = None) -> "BatchResolver":
        """Evaluate the typed primary candidate pool in slate coordinates."""
        return cls(pred_scorer=PredictionScorer("candidates.scores"), label_scorer=label_scorer,
                   ref_scorer=ref_scorer, order_field="candidates.orders", view="slate", mask_field=mask_field)

    def _active_scorers(self, pred):
        prediction = self.pred_scorer
        if prediction is None:
            if (self.view == "slate" and self._uses_prediction_order(pred)
                    and (pred.chosen_scores is not None or pred.candidates is not None)):
                prediction = PredictionScorer("chosen_scores")
            elif pred.scores is not None:
                prediction = PredictionScorer()
        return self.label_scorer, prediction, self.ref_scorer

    @property
    def uses_schema_labels(self) -> bool:
        return any(scorer is not None and scorer.uses_schema_labels
                   for scorer in (self.label_scorer, self.pred_scorer, self.ref_scorer))

    def _uses_prediction_order(self, pred) -> bool:
        return (self.orders is None and self.view != "score_only"
                and self.order_field == "order" and pred.order is not None)

    def uses_prediction_target(self, pred) -> bool:
        return (any(scorer is not None and scorer.uses_prediction_target(pred)
                    for scorer in self._active_scorers(pred))
                or (self._uses_prediction_order(pred) and isinstance(pred.order, torch.Tensor))
                or (self.view == "slate" and self.orders is None
                    and self.order_field == "candidates.orders" and pred.candidates is not None)
                or pred.full_scores is not None)

    def needs_target(self, pred) -> bool:
        if any(scorer is not None and scorer.needs_target(pred)
               for scorer in self._active_scorers(pred)):
            return True
        if self._uses_prediction_order(pred) and isinstance(pred.order, Mapping):
            return True
        return pred.target is not None and self.uses_prediction_target(pred)

    def _extract_order(self, context):
        if self.view == "score_only":
            return None
        if self.orders is not None:
            return self.orders
        if self.order_field in ("order", "candidates.orders"):
            order = context.prediction_slot(self.order_field)
            if self.order_field == "candidates.orders" and order is None:
                raise ValueError("Candidate-pool evaluation requires Prediction.candidates.")
            return order
        return context.pred.extra.get(self.order_field.removeprefix("extra."))

    def _infer_num_candidates(self, context: EvalContext, mask: torch.Tensor | None, scorers) -> int:
        """Infer original candidate pool width N before applying order [B, K]."""
        if mask is not None and mask.ndim == 2:
            return mask.shape[1]
        for scorer in scorers:
            if scorer is not None:
                n = scorer.num_candidates(context)
                if n is not None:
                    return n
        raise ValueError("Ordered views require original candidate width from mask, sources or scorer.num_candidates")

    @torch.no_grad()
    def resolve(self, context: EvalContext, *, mask=None, users=None,
                to_cpu: Callable[[torch.Tensor | None], torch.Tensor | None] = _cpu) -> EvalBatch:
        # Step 1: Resolve active scorers
        label_scorer, pred_scorer, ref_scorer = self._active_scorers(context.pred)

        # Step 2: Determine how `Prediction.order` is used under the current view
        num_candidates = None
        order = self._extract_order(context)
        if order is not None:
            num_candidates = self._infer_num_candidates(context, mask, (label_scorer, pred_scorer, ref_scorer))
            for scorer in (label_scorer, pred_scorer, ref_scorer):
                width = scorer.num_candidates(context) if scorer is not None else None
                if width is not None and width != num_candidates:
                    raise ValueError("Candidate sources must align to the same original width N; declare selected alignment explicitly")
            normalized, single = normalize_orders(order, context.pred.batch_size, num_candidates)
            if self.view == "ranking":
                if not single:
                    raise ValueError("Ranking view requires Prediction.order [B,K], not a candidate set")
            elif not single:
                order = normalized

        # Only "slate" view projects tensors into [B, K] output slots during scoring
        slate_order = order if self.view == "slate" else None

        # Step 3: Execute scorers and verify 2D slot alignment
        labels = label_scorer(context, order=slate_order)
        scores = pred_scorer(context, order=slate_order) if pred_scorer is not None else None
        reference = ref_scorer(context, order=slate_order) if ref_scorer is not None else None

        slot_shapes = {t.shape for t in (labels, scores, reference) if isinstance(t, torch.Tensor) and t.ndim == 2}
        if len(slot_shapes) > 1:
            raise ValueError("Resolved slot scores, labels and references must align")

        if self.view == "ranking" and labels.ndim == 2:
            for tensor in (scores, reference):
                if tensor is not None and tensor.shape != labels.shape:
                    raise ValueError("Ranking scores, labels and references must align to [B,N]")
        if self.view == "slate" and slate_order is not None:
            for tensor in (labels, scores, reference):
                if tensor is not None and tensor.ndim != 1 and tensor.shape != slate_order.shape:
                    raise ValueError("Slate tensors must align with their selected order")

        # Step 4: Project auxiliary tensors in "slate" view and offload to CPU
        if slate_order is not None:
            mask = gather_by_order(mask, slate_order) if mask is not None else None
            users = gather_by_order(users, slate_order) if users is not None and users.ndim == 2 else users

        return EvalBatch(
            labels=to_cpu(labels),
            scores=to_cpu(scores),
            ref_scores=to_cpu(reference),
            mask=to_cpu(mask),
            users=to_cpu(users),
            full_scores=to_cpu(context.pred.full_scores),
            order=to_cpu(order) if self.view == "ranking" else None,
        )
