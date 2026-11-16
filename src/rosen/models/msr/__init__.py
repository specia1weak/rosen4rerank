from typing import Dict, Type, Union

from rosen.emb import SchemaManager
from rosen.models.msr.all_expert_mixer import AllDomainExpertMixerModel
from rosen.models.msr.automtl import AutoMTLModel
from rosen.models.msr.base import MSRModel
from rosen.models.msr.crocodile import CrocodileModel
from rosen.models.msr.din import ADSDINModel, DINModel
from rosen.models.msr.hamur import HAMURModel
from rosen.models.msr.hierrec import HierRec
from rosen.models.msr.hypergate import HyperGateModel
from rosen.models.msr.m2m import M2MModel
from rosen.models.msr.m3oe import M3oEModel, M3oEVersion1Model, M3oEVersion2Model
from rosen.models.msr.mmoe import GradientBalancedMMoEModel, MMoEModel
from rosen.models.msr.multiemb import MultiEmbModel
from rosen.models.msr.pareto import ParetoModel
from rosen.models.msr.pepnet import EPNetModel, PEPNetModel, PPNetModel
from rosen.models.msr.ple import PLEModel, PLEVersion1Model
from rosen.models.msr.representation_contexture import ContexturePrototypeSharedBottomModel
from rosen.models.msr.riple import RIPLEModel
from rosen.models.msr.satrans import SATransModel
from rosen.models.msr.sdsp import SDSPMMoEModel, SDSPPLEModel
from rosen.models.msr.sharedbottom import (
    SharedBottomCodebookModel,
    SharedBottomLessModel,
    SharedBottomModel,
    SharedBottomPlusModel,
)
from rosen.models.msr.ssim import SSIMModel
from rosen.models.msr.star import AdditiveSTARModel, STARModel


MODEL_REGISTRY: Dict[str, Type[MSRModel]] = {
    "sharedbottom": SharedBottomModel,
    "sharedbottom_codebook": SharedBottomCodebookModel,
    "sharedbottom_less": SharedBottomLessModel,
    "sharedbottom_plus": SharedBottomPlusModel,
    "din": DINModel,
    "ads_din": ADSDINModel,
    "all_expert_mixer": AllDomainExpertMixerModel,
    "contexture_prototype_sharedbottom": ContexturePrototypeSharedBottomModel,
    "mmoe": MMoEModel,
    "mmoe_grad_balance": GradientBalancedMMoEModel,
    "ple": PLEModel,
    "ple_v1": PLEVersion1Model,
    "star": STARModel,
    "star_add": AdditiveSTARModel,
    "m3oe": M3oEModel,
    "m3oe_v1": M3oEVersion1Model,
    "m3oe_v2": M3oEVersion2Model,
    "m2m": M2MModel,
    "ppnet": PPNetModel,
    "epnet": EPNetModel,
    "pepnet": PEPNetModel,
    "crocodile": CrocodileModel,
    "pareto": ParetoModel,
    "hierrec": HierRec,
    "automtl": AutoMTLModel,
    "riple": RIPLEModel,
    "satrans": SATransModel,
    "hamur": HAMURModel,
    "hypergate": HyperGateModel,
    "sdsp_ple": SDSPPLEModel,
    "sdsp_mmoe": SDSPMMoEModel,
    "ssim": SSIMModel,
    "multiemb": MultiEmbModel,
}

def update_register(**kwargs) -> Dict[str, Type[MSRModel]]:
    for name, model_cls in kwargs.items():
        if not isinstance(name, str):
            raise TypeError(f"Model registry key must be str, got {type(name)}")
        if not isinstance(model_cls, type) or not issubclass(model_cls, MSRModel):
            raise TypeError(
                f"Model registry value for '{name}' must be an MSRModel subclass, got {model_cls!r}"
            )
        MODEL_REGISTRY[name.lower()] = model_cls
    return MODEL_REGISTRY


def build_model(
        schema_manager: SchemaManager,
        num_domains: int,
        model_cls: Union[str, Type[MSRModel]],
        **model_kwargs,
):

    if isinstance(model_cls, str):
        model_name = model_cls
        model_cls = MODEL_REGISTRY.get(model_name.lower())
    else:
        model_name = model_cls.__name__
    if model_cls is None:
        raise ValueError(f"Unknown model_name={model_name}. Available: {sorted(MODEL_REGISTRY.keys())}")
    return model_cls.from_manager(schema_manager, num_domains, **dict(model_kwargs or {}))


__all__ = [
    "MODEL_REGISTRY",
    "update_register",
    "build_model",
    "HierRec",
    "SharedBottomModel",
    "SharedBottomCodebookModel",
    "SharedBottomLessModel",
    "SharedBottomPlusModel",
    "DINModel",
    "ADSDINModel",
    "AllDomainExpertMixerModel",
    "ContexturePrototypeSharedBottomModel",
    "MMoEModel",
    "GradientBalancedMMoEModel",
    "PLEModel",
    "PLEVersion1Model",
    "STARModel",
    "AdditiveSTARModel",
    "M3oEModel",
    "M3oEVersion1Model",
    "M3oEVersion2Model",
    "RIPLEModel",
    "SATransModel",
    "SDSPMMoEModel",
    "SDSPPLEModel",
    "SSIMModel",
    "MultiEmbModel",
    "M2MModel",
    "PPNetModel",
    "EPNetModel",
    "PEPNetModel",
    "CrocodileModel",
    "ParetoModel",
    "AutoMTLModel",
    "HAMURModel",
    "HyperGateModel",
]
