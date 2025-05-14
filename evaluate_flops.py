import torch
import torch.nn as nn
import numpy as np
from torch.fx import symbolic_trace, GraphModule
from torch.fx.passes.shape_prop import ShapeProp
from typing import Callable, Dict

# ─── Helpers ────────────────────────────────────────────────────────────────


def _numel_from_meta(node):
    return int(np.prod(node.meta["tensor_meta"].shape))


def _numel_from_meta_simple(shape: tuple) -> int:
    return int(np.prod(shape))


# ─── Zero‑cost ops ─────────────────────────────────────────────────────────
_ZERO_COST = {
    torch.ones_like,
    torch.zeros_like,
    torch.tensor,
    torch.flatten,    # view only
}

# ─── Recognize both Python and ATen sum/prod ────────────────────────────────
_SUM_FNS = {torch.ops.aten.sum, torch.sum}
_PROD_FNS = {torch.ops.aten.prod, torch.prod}

# ─── ATen mapping: specific operators → flop functions ──────────────────────
ATEN_FLOP_MAP: Dict[Callable, Callable] = {
    # element‑wise
    torch.ops.aten.add.Tensor: lambda n: _numel_from_meta(n),
    torch.ops.aten.sub.Tensor: lambda n: _numel_from_meta(n),
    torch.ops.aten.mul.Tensor: lambda n: _numel_from_meta(n),
    torch.ops.aten.div.Tensor: lambda n: _numel_from_meta(n),
    torch.ops.aten.neg: lambda n: _numel_from_meta(n),
    torch.ops.aten.exp: lambda n: _numel_from_meta(n),
    torch.ops.aten.log: lambda n: _numel_from_meta(n),
    torch.ops.aten.sqrt: lambda n: _numel_from_meta(n),
    torch.ops.aten.sigmoid: lambda n: 2 * _numel_from_meta(n),
    torch.ops.aten.tanh: lambda n: 2 * _numel_from_meta(n),
    torch.ops.aten.square: lambda n: _numel_from_meta(n),

    # constant factories
    torch.ops.aten.ones_like: lambda n: 0,
    torch.ops.aten.zeros_like: lambda n: 0,

    # matmul
    torch.ops.aten.matmul: lambda n: 2
    * n.args[0].shape[0]
    * n.args[0].shape[1]
    * n.args[1].shape[-1],

    # reductions via ATen
    torch.ops.aten.sum: lambda n: _numel_from_meta(n.args[0]),
    torch.ops.aten.prod: lambda n: _numel_from_meta(n.args[0]),

    torch.bitwise_left_shift: lambda n: _numel_from_meta(n),
    torch.ops.aten.bitwise_left_shift: lambda n: _numel_from_meta(n),
}

# ─── nn.Module mapping → flop functions ────────────────────────────────────
MODULE_FLOP_MAP: Dict[type, Callable] = {
    nn.Conv1d: lambda m, o: 2 * o[0] * o[1] * o[2] * m.in_channels * m.kernel_size[0],
    nn.Conv2d: lambda m, o: 2 * o[0] * o[1] * o[2] * o[3]
    * m.in_channels
                                     * m.kernel_size[0]
                                     * m.kernel_size[1],
    nn.Conv3d: lambda m, o: 2 * o[0] * o[1] * o[2] * o[3] * o[4]
    * m.in_channels
                                     * m.kernel_size[0]
                                     * m.kernel_size[1]
                                     * m.kernel_size[2],
    nn.Linear: lambda m, o: 2 * o[0] * m.in_features * m.out_features,
    nn.BatchNorm1d: lambda m, o: 2 * _numel_from_meta_simple(o),
    nn.BatchNorm2d: lambda m, o: 2 * _numel_from_meta_simple(o),
    nn.BatchNorm3d: lambda m, o: 2 * _numel_from_meta_simple(o),
}

# ─── The FX‑based FLOP counter ─────────────────────────────────────────────


def fx_count_flops(
    model: torch.nn.Module,
    inputs: tuple,
    custom_aten: Dict[Callable, Callable] = None,
    custom_modules: Dict[type, Callable] = None,
) -> int:
    # 1) Trace & shape‑propagate
    if isinstance(model, GraphModule):
        gm = model
    else:
        gm = symbolic_trace(model)
    ShapeProp(gm).propagate(*inputs)

    aten_map = {**ATEN_FLOP_MAP,    **(custom_aten or {})}
    mod_map = {**MODULE_FLOP_MAP, **(custom_modules or {})}

    total_flops = 0
    for node in gm.graph.nodes:
        # 0) skip zero‑cost
        if node.op == "call_function" and node.target in _ZERO_COST:
            continue

        # 1) fused‑function fast path
        if node.op == "call_function" and getattr(node.target, "is_fused_function", False):
            ne = _numel_from_meta(node)
            total_flops += node.target.flops_per_element * ne
            continue

        # 2) nn.Modules
        if node.op == "call_module":
            sub = gm.get_submodule(node.target)
            for T, fn in mod_map.items():
                if isinstance(sub, T):
                    total_flops += int(fn(sub, node.meta["tensor_meta"].shape))
                    break
            continue

        # 3) explicit ATen (elementwise, matmul, constant factories, aten.sum/prod)
        if node.op == "call_function" and node.target in aten_map:
            total_flops += int(aten_map[node.target](node))
            continue

        # 4) reductions via Python‐level sum/prod
        if node.op == "call_function" and node.target in _SUM_FNS:
            # cost = #elements of the input
            total_flops += int(np.prod(node.args[0].meta["tensor_meta"].shape))
            continue
        if node.op == "call_function" and node.target in _PROD_FNS:
            total_flops += int(np.prod(node.args[0].meta["tensor_meta"].shape))
            continue

        # 5/6) generic fallback: any other call_*(function|method) that has tensor_meta
        if (node.op in ("call_function", "call_method") and
                "tensor_meta" in node.meta):
            total_flops += _numel_from_meta(node)
            continue

    return total_flops
