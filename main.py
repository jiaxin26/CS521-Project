import torch
from bitshift_swap import swap_bitshift_reducesum
from recip_associative_swap import swap_recip_associative
from fuse_one_to_one_operations import fuse_elementwise_chains
from distributive1 import DistributiveRulePass
from distributive2 import DistributiveRule2Pass
from associative2 import SqrtAssociativePass
from commutative2 import CommutativePass

def apply_all_rewrites(gm: torch.fx.GraphModule):
    # Commutative rules
    gm = swap_bitshift_reducesum(gm)
    gm = CommutativePass()(gm)

    # Associative rules
    gm = swap_recip_associative(gm)
    gm = SqrtAssociativePass()(gm)

    # Distributive rules
    gm = DistributiveRulePass()(gm)
    gm = DistributiveRule2Pass()(gm)

    return gm


def optimize_graph(gm: torch.fx.GraphModule):
    gm = apply_all_rewrites(gm)
    gm = fuse_elementwise_chains(gm)
    return gm
