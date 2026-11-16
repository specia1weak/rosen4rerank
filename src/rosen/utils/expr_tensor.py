from __future__ import annotations

from typing import Any

import torch


class ExprTensor:
    """Tensor wrapper that records a symbolic expression while running real ops."""

    def __init__(self, data: torch.Tensor, expr: str):
        self.data = data
        self.expr = expr

    @staticmethod
    def _unwrap(x: Any):
        if isinstance(x, ExprTensor):
            return x.data
        if isinstance(x, (list, tuple)):
            return type(x)(ExprTensor._unwrap(i) for i in x)
        return x

    @staticmethod
    def _get_expr(a: Any) -> str:
        if isinstance(a, ExprTensor):
            return a.expr
        if isinstance(a, torch.nn.Parameter):
            return "W"
        if isinstance(a, torch.Tensor):
            return "W" if a.requires_grad else "C"
        if isinstance(a, (list, tuple)):
            return "[" + ", ".join(ExprTensor._get_expr(i) for i in a) + "]"
        if isinstance(a, slice):
            start = ExprTensor._get_expr(a.start) if a.start is not None else ""
            stop = ExprTensor._get_expr(a.stop) if a.stop is not None else ""
            step = f":{ExprTensor._get_expr(a.step)}" if a.step is not None else ""
            return f"{start}:{stop}{step}"
        if isinstance(a, str):
            return f"'{a}'"
        return str(a)

    def __add__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data + o_data, f"({self.expr} + {o_expr})")

    def __sub__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data - o_data, f"({self.expr} - {o_expr})")

    def __mul__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data * o_data, f"({self.expr} * {o_expr})")

    def __truediv__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data / o_data, f"({self.expr} / {o_expr})")

    def __pow__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data**o_data, f"({self.expr} ^ {o_expr})")

    def __matmul__(self, other):
        o_expr = self._get_expr(other)
        o_data = other.data if isinstance(other, ExprTensor) else other
        return ExprTensor(self.data @ o_data, f"({self.expr} @ {o_expr})")

    def __neg__(self):
        return ExprTensor(-self.data, f"(-{self.expr})")

    def __radd__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) + self.data, f"({o_expr} + {self.expr})")

    def __rsub__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) - self.data, f"({o_expr} - {self.expr})")

    def __rmul__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) * self.data, f"({o_expr} * {self.expr})")

    def __rtruediv__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) / self.data, f"({o_expr} / {self.expr})")

    def __rmatmul__(self, other):
        o_expr = self._get_expr(other)
        return ExprTensor(self._unwrap(other) @ self.data, f"({o_expr} @ {self.expr})")

    def __getitem__(self, key):
        real_key = self._unwrap(key)
        out_data = self.data[real_key]

        if isinstance(key, tuple):
            key_str = ", ".join(self._get_expr(k) for k in key)
        else:
            key_str = self._get_expr(key)

        return ExprTensor(out_data, f"{self.expr}[{key_str}]")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        real_args = tuple(cls._unwrap(a) for a in args)
        real_kwargs = {k: cls._unwrap(v) for k, v in kwargs.items()}
        out_data = func(*real_args, **real_kwargs)

        args_str = [cls._get_expr(a) for a in args]
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)
        if "Tensor." in func_name:
            func_name = func_name.split(".")[-1]

        if func_name == "linear":
            new_expr = f"Linear({args_str[0]})"
        elif func_name == "batch_norm":
            new_expr = f"BN({args_str[0]})"
        else:
            new_expr = f"{func_name}({', '.join(args_str)})"

        if isinstance(out_data, torch.Tensor):
            return ExprTensor(out_data, new_expr)
        if isinstance(out_data, tuple):
            return tuple(
                ExprTensor(d, f"{new_expr}_{i}") if isinstance(d, torch.Tensor) else d
                for i, d in enumerate(out_data)
            )
        return out_data

    def __getattr__(self, name):
        if name in ["shape", "device", "dtype", "ndim"]:
            return getattr(self.data, name)

        attr = getattr(self.data, name)
        if not callable(attr):
            return attr

        def method_wrapper(*args, **kwargs):
            real_args = tuple(self._unwrap(a) for a in args)
            real_kwargs = {k: self._unwrap(v) for k, v in kwargs.items()}
            out_data = attr(*real_args, **real_kwargs)

            args_str = [self._get_expr(a) for a in args]
            kwargs_str = [f"{k}={self._get_expr(v)}" for k, v in kwargs.items()]
            all_args = ", ".join(args_str + kwargs_str)
            new_expr = f"{self.expr}.{name}({all_args})"

            if isinstance(out_data, torch.Tensor):
                return ExprTensor(out_data, new_expr)
            if isinstance(out_data, tuple):
                return tuple(
                    ExprTensor(d, f"{new_expr}_{i}") if isinstance(d, torch.Tensor) else d
                    for i, d in enumerate(out_data)
                )
            return out_data

        return method_wrapper

    def dim(self):
        return self.data.dim()

    def size(self, *args):
        return self.data.size(*args)

    def __repr__(self):
        return f"ExprTensor(expr='{self.expr}')"


if __name__ == "__main__":
    from torch import nn

    model = nn.Sequential(
        nn.Linear(3, 2),
        nn.BatchNorm1d(2),
        nn.ReLU(),
    )
    x = ExprTensor(torch.randn(4, 3), "x")
    out = model(x)
    assert out.expr == "relu(BN(Linear(x)))"
    assert out.data.shape == (4, 2)

    reshaped = (ExprTensor(torch.arange(6.0).reshape(2, 3), "x") + 1).reshape(3, 2)
    assert reshaped.expr == "(x + 1).reshape(3, 2)"

    weight = torch.nn.Parameter(torch.randn(3, 4))
    matmul = ExprTensor(torch.randn(2, 3), "x") @ weight
    assert matmul.expr == "(x @ W)"
    assert matmul.data.shape == (2, 4)

    print("ExprTensor self-check passed.")
