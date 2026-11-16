import torch.nn as nn
from typing import Any, Dict, Iterable, List, Tuple, Union


DEFAULT_NO_DECAY_KEYWORDS = (
    "position_ids",
    # Free parameters with embedding-like semantics.  These are not always
    # owned by nn.Embedding, so module-type inspection alone is insufficient.
    "seed",
    "shift_begin",
    "shift_end",
    "boundary_flag",
    "_vectors",
)

def split_params_by_decay(
        named_params: Iterable[Tuple[str, nn.Parameter]],  # 修改这里
        weight_decay: float = 0.01,
        no_decay_keywords: Union[tuple, list] = ("embedding", "position_ids")
) -> List[Dict[str, Any]]:
    """
    基于张量维度和关键字自动分组优化器参数。
    """
    decay_params = []
    no_decay_params = []

    for name, param in named_params: # 修改这里
        if not param.requires_grad:
            continue

        if param.ndim < 2 or any(nd in name.lower() for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]


def split_model_params_by_decay(
    model: nn.Module,
    weight_decay: float = 1.0e-6,
    no_decay_keywords: Union[tuple, list] = DEFAULT_NO_DECAY_KEYWORDS,
) -> List[Dict[str, Any]]:
    """Build optimizer groups while never decaying embedding-like parameters.

    Matrix/tensor weights receive ``weight_decay``. Biases, normalization
    parameters, scalar/vector parameters, parameters owned directly by an
    ``nn.Embedding`` module, and explicitly named embedding-like free
    parameters receive zero decay. A Linear nested under a container whose
    name happens to contain ``embedding`` is still decayed.

    Inspecting module ownership matters: an embedding table may be named
    ``lookup.weight`` and therefore cannot be identified reliably from the
    parameter name alone.
    """
    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    lowered_keywords = tuple(str(keyword).lower() for keyword in no_decay_keywords)
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered_name = name.lower()
        no_decay = (
            parameter.ndim < 2
            or id(parameter) in embedding_parameter_ids
            or any(keyword in lowered_name for keyword in lowered_keywords)
        )
        (no_decay_params if no_decay else decay_params).append(parameter)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups

def create_complex_optimizer_groups(
        model: nn.Module,
        decay_dict: Dict[str, float],
        default_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    根据传入的 decay_dict 灵活配置不同网络层的前缀参数衰减。

    参数示例:
    decay_dict = {
        "embeddings": 0.0,       # 推荐系统 Embedding
        "moe_router": 0.05,      # 假设 MoE 门控网络需要更强的正则化
        "bias": 0.0,             # 偏置项
        "norm": 0.0              # 归一化层
    }
    """
    # 构建一个按 weight_decay 分组的字典
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 默认衰减值
        current_decay = default_decay

        # 遍历字典进行匹配（可以根据需求换成正则 re.search）
        for key, decay_value in decay_dict.items():
            if key in name.lower():
                current_decay = decay_value
                break  # 匹配到第一个优先级最高的就跳出

        # 归入对应的 decay 组
        if current_decay not in groups:
            groups[current_decay] = []
        groups[current_decay].append(param)

    # 转换为 Optimizer 需要的格式
    return [
        {"params": params, "weight_decay": wd}
        for wd, params in groups.items()
    ]
