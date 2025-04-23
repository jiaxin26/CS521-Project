import torch
from torch.fx import symbolic_trace
from bitshift_swap import swap_bitshift_reducesum


class BitShiftSwap(torch.nn.Module):
    def forward(self, x):
        # Original subgraph: ReduceSum(BitShift(A))
        t = torch.bitwise_left_shift(x, 1)  # BitShift(x,1)
        out = t.sum()                       # ReduceSum(t)
        return out


gm = symbolic_trace(BitShiftSwap().eval())
print("=== FX before rewrite ===")
print(gm.graph)

gm = swap_bitshift_reducesum(gm)
print("=== FX after rewrite ===")
print(gm.graph)

compiled_gm = torch.compile(gm, backend="inductor")

x = torch.randint(0, 16, (4, 4), dtype=torch.int32)

out_uncompiled = gm(x)
out = compiled_gm(x)

print("input:\n", x)
print("Uncompiled output:", out_uncompiled)
print("Compiled output:", out)

assert torch.equal(out_uncompiled, out), "Outputs do not match!"
