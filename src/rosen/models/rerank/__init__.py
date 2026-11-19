"""List-wise reranking models; independent from the multi-scenario registry."""

from .base import (RerankModel, fold_history_pages, build_full_permutations,
                   select_by_order, masked_bce_with_logits)
from .mge import MultiGERerankModel
from .slate import OCPMRerankModel, SlateRerankModel
from .dnn import DNNRerankModel
from .dcn import DCNRerankModel
from .prm import PRMRerankModel
from .extr import EXTRRerankModel
from .kuaishou import KuaiShouRerankModel
from .pier import PIERRerankModel
from .prs import PRSRerankModel
from .nar import NARRerankModel
from .seq2slate import Seq2SlateRerankModel
from .pier_submodular import PIERSubmodularRerankModel
from .mge_metric import MultiGEMetricRerankModel, SubmodularMultiGEMetricRerankModel


RERANK_MODEL_REGISTRY = {
    "dnn": DNNRerankModel, "dcn": DCNRerankModel, "prm": PRMRerankModel,
    "extr": EXTRRerankModel, "kuaishou": KuaiShouRerankModel,
    "pier": PIERRerankModel, "prs": PRSRerankModel,
    "nar": NARRerankModel, "seq2slate": Seq2SlateRerankModel,
    "pier_submodular": PIERSubmodularRerankModel,
    "mge_metric": MultiGEMetricRerankModel, "mge_cosine_metric": SubmodularMultiGEMetricRerankModel,
    "mge3": MultiGERerankModel, "mge_cosine": MultiGERerankModel,
    "mge": MultiGERerankModel, "lc_mge": MultiGERerankModel,
}

from .presets import resolve_rerank_config


def build_rerank_model(schema_manager, model_cls="mge", **model_kwargs):
    if isinstance(model_cls, str):
        name = model_cls.lower()
        if name not in RERANK_MODEL_REGISTRY:
            raise ValueError(f"Unknown rerank model '{model_cls}'. Available: {sorted(RERANK_MODEL_REGISTRY)}")
        model_kwargs = resolve_rerank_config(name, **model_kwargs)
        model_cls = RERANK_MODEL_REGISTRY[name]
    if not isinstance(model_cls, type) or not issubclass(model_cls, RerankModel):
        raise TypeError("model_cls must be a RerankModel subclass or registered name.")
    return model_cls.from_manager(schema_manager, **model_kwargs)


__all__ = ["RerankModel", "SlateRerankModel", "OCPMRerankModel", "MultiGERerankModel", "DNNRerankModel", "DCNRerankModel",
           "PRMRerankModel", "EXTRRerankModel", "KuaiShouRerankModel", "PIERRerankModel", "PRSRerankModel",
           "NARRerankModel", "Seq2SlateRerankModel", "RERANK_MODEL_REGISTRY", "build_rerank_model", "resolve_rerank_config",
           "PIERSubmodularRerankModel", "MultiGEMetricRerankModel", "SubmodularMultiGEMetricRerankModel",
           "select_by_order", "fold_history_pages", "build_full_permutations", "masked_bce_with_logits"]
