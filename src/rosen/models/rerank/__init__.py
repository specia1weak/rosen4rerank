"""List-wise reranking models; independent from the multi-scenario registry."""

from .base import (RerankModel, fold_history_pages, build_full_permutations,
                   apply_permutation, masked_bce_with_logits)
from .mge import MultiGERerankModel


RERANK_MODEL_REGISTRY = {"mge": MultiGERerankModel}


def build_rerank_model(schema_manager, model_cls="mge", **model_kwargs):
    if isinstance(model_cls, str):
        if model_cls.lower() not in RERANK_MODEL_REGISTRY:
            raise ValueError(f"Unknown rerank model '{model_cls}'. Available: {sorted(RERANK_MODEL_REGISTRY)}")
        model_cls = RERANK_MODEL_REGISTRY[model_cls.lower()]
    if not isinstance(model_cls, type) or not issubclass(model_cls, RerankModel):
        raise TypeError("model_cls must be a RerankModel subclass or registered name.")
    return model_cls.from_manager(schema_manager, **model_kwargs)


__all__ = ["RerankModel", "MultiGERerankModel", "RERANK_MODEL_REGISTRY", "build_rerank_model",
           "fold_history_pages", "build_full_permutations", "apply_permutation", "masked_bce_with_logits"]
