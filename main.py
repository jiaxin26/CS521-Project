import torch
from bitshift_swap import swap_bitshift_reducesum
from recip_associative_swap import swap_recip_associative
from fuse_one_to_one_operations import fuse_elementwise_chains


def apply_all_rewrites(gm: torch.fx.GraphModule):
    # Commutative rules
    gm = swap_bitshift_reducesum(gm)

    # Associative rules
    gm = swap_recip_associative(gm)

    # Distributive rules

    return gm


def optimize_graph(gm: torch.fx.GraphModule):
    # First apply algebraic rewrites
    gm = apply_all_rewrites(gm)
    print("=== Graph after rewrite ===")
    print(gm.graph)

    # Then apply one-to-one fusion
    gm = fuse_elementwise_chains(gm)
    print("=== Graph after fusion ===")
    print(gm.graph)

    return gm
