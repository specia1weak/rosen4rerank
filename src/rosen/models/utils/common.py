from collections.abc import Iterable
from torch import nn
from rosen.models.utils.general import MLP

def default_dims(input_dim):
    return [input_dim, input_dim // 2, 128]

def to_dims(value, default):
    if value is None:
        value = default

    if isinstance(value, int):
        dims = (int(value),)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        dims = tuple(int(v) for v in value)
    else:
        raise TypeError(f"Unsupported dims type: {type(value)}")

    if not dims:
        raise ValueError("dims can not be empty")
    return dims

def default_backbone_dims(input_dim: int, depth: int = 2) -> tuple[int, ...]:
    input_dim = max(1, int(input_dim))
    if depth <= 1:
        return (input_dim,)
    hidden = max(1, input_dim)
    return tuple(hidden for _ in range(depth))


def build_mlp(
        input_dim: int,
        hidden_dims,
        *,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
):
    dims = to_dims(hidden_dims, default_backbone_dims(input_dim))
    return MLP(input_dim, *dims, dropout_rate=dropout_rate, activation=activation, batch_norm=batch_norm)


def build_gate(input_dim: int,
               hidden_dims,
               out_dim: int = 1,
               dropout_rate: float = 0.0,
               activation: str = "relu",
               batch_norm: bool = False,):
    """
    构建专家网络。
    :param output_dim: 如果有输出要求，默认emb // size 改成 传参
    :param in_dim: 输入特征维度
    :param depth_multiplier: 深度倍数。
                             1 对应单层 PLE 的深度 [D, D/2]
                             2 对应两层 PLE 的总深度 [D, D, D/2]
    """
    # 初始化输入维度
    dims = to_dims(hidden_dims, default_backbone_dims(input_dim))
    out_dim = to_dims(out_dim, 1)
    out_nums = out_dim[0]
    dims += out_dim
    mlp = MLP(input_dim, *dims, dropout_rate=dropout_rate, activation=activation, batch_norm=batch_norm)
    if out_nums > 1 :
        return nn.Sequential(mlp, nn.Softmax(dim=-1))
    else:
        return nn.Sequential(mlp, nn.Sigmoid())

if __name__ == '__main__':
    gate = build_gate(128, (256, 64), 4)
    print(gate)
    import torch
    x = torch.randn(10, 128)
    x = gate(x)
    print(x.shape)