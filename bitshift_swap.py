import torch


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
