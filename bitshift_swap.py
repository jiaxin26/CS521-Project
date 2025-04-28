import torch
from torch.fx import symbolic_trace


def swap_bitshift_reducesum(gm: torch.fx.GraphModule):
    """Rewrite Commutative: ReduceSum(BitShift(x)) → BitShift(ReduceSum(x)) bit shift."""
    bitshift_ops = [torch.bitwise_left_shift]
    # Only left shift is included as it distributes over sum (e.g., sum(x << k) == sum(x) << k).
    # Right shift does NOT distribute over addition and can lead to incorrect rewrites.
    modified = False

    for node in list(gm.graph.nodes):
        is_sum = (
            (node.op == "call_method" and node.target == "sum") or
            (node.op == "call_function" and node.target in (torch.sum, sum))
        )

        if not is_sum or len(node.args) == 0:
            continue

        input_node = node.args[0]
        if input_node.op == "call_function" and input_node.target in bitshift_ops:
            x, shift = input_node.args

            with gm.graph.inserting_before(node):
                if node.op == "call_method":
                    new_sum = gm.graph.call_method(
                        "sum", args=(x,), kwargs=node.kwargs)
                else:
                    new_sum = gm.graph.call_function(node.target, args=(
                        x,) + node.args[1:], kwargs=node.kwargs)

                new_bitshift = gm.graph.call_function(
                    input_node.target, args=(
                        new_sum, shift), kwargs=input_node.kwargs
                )

            node.replace_all_uses_with(new_bitshift)
            gm.graph.erase_node(node)
            gm.graph.erase_node(input_node)
            modified = True

    if modified:
        gm.recompile()
    return gm


def run_bitshift_swap():
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


if __name__ == "__main__":
    run_bitshift_swap()
