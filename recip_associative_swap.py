import torch
import operator
from torch.fx import GraphModule, symbolic_trace


def swap_recip_associative(gm: GraphModule):
    """Rewrite Recip(A)*Recip(A*B) → square(Recip(A)) * Recip(B)."""
    nodes_to_delete = set()
    modified = False

    for node in list(gm.graph.nodes):
        # 1) match outer multiplication
        if node.op != "call_function" or node.target not in (torch.mul, operator.mul):
            continue
        lhs, rhs = node.args
        if not (is_reciprocal(lhs) and is_reciprocal(rhs)):
            continue

        # 2) figure out which side is Recip(A) vs Recip(A*B)
        a_node = b_node = recip_a_node = mul_node = None
        arg1, arg2 = get_recip_arg(lhs), get_recip_arg(rhs)

        # case A*B on rhs
        if (isinstance(arg2, torch.fx.Node)
                and arg2.op == "call_function"
                and arg2.target in (torch.mul, operator.mul)
                and arg1 in arg2.args):
            recip_a_node = lhs
            mul_node = arg2
            a_node = arg1
            b_node = arg2.args[1] if arg2.args[0] is arg1 else arg2.args[0]

        # case A*B on lhs
        elif (isinstance(arg1, torch.fx.Node)
              and arg1.op == "call_function"
              and arg1.target in (torch.mul, operator.mul)
              and arg2 in arg1.args):
            recip_a_node = rhs
            mul_node = arg1
            a_node = arg2
            b_node = arg1.args[1] if arg1.args[0] is arg2 else arg1.args[0]
        else:
            continue  # no match

        # 3) insert the new ops before the old `node`
        with gm.graph.inserting_before(node):
            square_recip = gm.graph.call_function(
                torch.square, args=(recip_a_node,))
            recip_b = gm.graph.call_function(
                torch.reciprocal, args=(b_node,))
            new_mul = gm.graph.call_function(
                torch.mul, args=(square_recip, recip_b))

        # 4) redirect uses, mark old nodes for deletion
        node.replace_all_uses_with(new_mul)
        nodes_to_delete.update({node, mul_node, lhs, rhs})
        modified = True

    # 5) bulk‑erase dead nodes in reverse order
    if modified:
        for n in reversed(list(gm.graph.nodes)):
            if n in nodes_to_delete and not n.users:
                gm.graph.erase_node(n)
        gm.graph.lint()
        new_module = GraphModule(gm, gm.graph)
        new_module.recompile()
        return new_module

    return gm


def is_reciprocal(node):
    if node.op != "call_function":
        return False
    if node.target == torch.reciprocal:
        return True
    if node.target == operator.truediv:
        return is_constant_one(node.args[0])
    if node.target in (torch.pow, operator.pow):
        return len(node.args) >= 2 and float(node.args[1]) == -1.0
    return False


def get_recip_arg(node):
    if node.target == torch.reciprocal:
        return node.args[0]
    if node.target == operator.truediv:
        return node.args[1]
    if node.target in (torch.pow, operator.pow) and float(node.args[1]) == -1.0:
        return node.args[0]
    return None


def is_constant_one(node):
    return (node.op == "call_function"
            and node.target == torch.tensor
            and isinstance(node.args[0], (int, float))
            and float(node.args[0]) == 1.0)


def run_associative_swap():
    class RecipAssociativeModel(torch.nn.Module):
        def forward(self, A, B):
            # Implements: Recip(A) ⊙ Recip(A ⊙ B)
            recip_a = torch.reciprocal(A)  # 1/A
            a_mul_b = A * B                # A*B
            recip_a_mul_b = torch.reciprocal(a_mul_b)  # 1/(A*B)
            return recip_a * recip_a_mul_b  # (1/A) * (1/(A*B))

    gm = symbolic_trace(RecipAssociativeModel().eval())

    print("=== Original Graph ===")
    print(gm.graph)

    optimized = swap_recip_associative(gm)
    print("=== Rewritten Graph ===")
    print(optimized.graph)

    compiled_gm = torch.compile(optimized, backend='inductor')

    A = torch.tensor([2.0, 4.0, 5.0])
    B = torch.tensor([3.0, 5.0, 7.0])

    original_output = gm(A, B)
    optimized_output = optimized(A, B)
    compiled_output = compiled_gm(A, B)

    print("Original output:", original_output)
    print("Optimized output:", optimized_output)
    print("Compiled output:", compiled_output)

    assert torch.allclose(original_output, optimized_output,
                          rtol=1e-5), "Outputs do not match!"
    assert torch.allclose(original_output, compiled_output,
                          rtol=1e-5), "Outputs do not match!"


if __name__ == "__main__":
    run_associative_swap()
