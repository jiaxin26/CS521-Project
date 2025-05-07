import torch
import operator
from torch.fx import symbolic_trace


def swap_recip_associative(gm: torch.fx.GraphModule):
    """Rewrite Associative: Recip(A) * Recip(A * B) → Square(Recip(A)) * Recip(B)"""
    modified = False

    for node in list(gm.graph.nodes):
        # Match outer multiplication
        if node.op != "call_function" or node.target not in (torch.mul, operator.mul):
            continue

        lhs, rhs = node.args

        # Ensure both sides are reciprocal calls
        if not (is_reciprocal(lhs) and is_reciprocal(rhs)):
            continue

        arg1, arg2 = get_recip_arg(lhs), get_recip_arg(rhs)
        match_found = False

        # Case 1:  rhs = Recip(A*B), lhs = Recip(A)
        # Pattern: rhs = Recip(A*B) and lhs = Recip(A)
        if (isinstance(arg2, torch.fx.Node)
            and arg2.op == "call_function" and arg2.target in (torch.mul, operator.mul)
                and arg1 in arg2.args):
            a_node = arg1
            mul_node = arg2
            b_node = arg2.args[1] if arg2.args[0] is a_node else arg2.args[0]
            recip_a_node = lhs
            match_found = True
        # Symmetric: lhs = Recip(A*B) and rhs = Recip(A)
        elif (isinstance(arg1, torch.fx.Node)
              and arg1.op == "call_function" and arg1.target in (torch.mul, operator.mul)
              and arg2 in arg1.args):
            a_node = arg2
            mul_node = arg1
            b_node = arg1.args[1] if arg1.args[0] is a_node else arg1.args[0]
            recip_a_node = rhs
            match_found = True

        if not match_found:
            continue

        with gm.graph.inserting_before(node):
            square_recip = gm.graph.call_function(
                torch.square, args=(recip_a_node,))
            recip_b = gm.graph.call_function(torch.reciprocal, args=(b_node,))
            new_mul = gm.graph.call_function(
                torch.mul, args=(square_recip, recip_b))

        node.replace_all_uses_with(new_mul)

        # Clean up dead nodes if no other users
        for dead in (node, mul_node, rhs if recip_a_node is lhs else lhs):
            if len(dead.users) == 0:
                gm.graph.erase_node(dead)

        # Cleanup orphaned torch.tensor(1) nodes
        for n in list(gm.graph.nodes):
            if (
                n.op == "call_function"
                and n.target == torch.tensor
                and not n.users
            ):
                gm.graph.erase_node(n)

        modified = True

    if modified:
        gm.graph.lint()
        new_gm = torch.fx.GraphModule(gm, gm.graph)
        new_gm.recompile()
        return new_gm
    return gm


def is_reciprocal(node):
    """Check if a node represents a reciprocal operation."""
    if node.op != "call_function":
        return False
    if node.target == torch.reciprocal:
        return True
    if node.target == operator.truediv:
        return is_constant_one(node.args[0])
    if node.target in (torch.pow, operator.pow):
        return (
            len(node.args) >= 2 and
            isinstance(node.args[1], (int, float)) and
            float(node.args[1]) == -1.0
        )

    return False


def get_recip_arg(node):
    """Get the argument to a reciprocal operation."""
    if node.target == torch.reciprocal:
        return node.args[0]
    if node.target == operator.truediv:
        return node.args[1]
    if node.target in (torch.pow, operator.pow) and float(node.args[1]) == -1.0:
        return node.args[0]
    return None


def is_constant_one(node):
    """Check if node represents the constant 1.0."""
    return (
        node.op == "call_function"
        and node.target == torch.tensor
        and isinstance(node.args[0], (int, float))
        and float(node.args[0]) == 1.0
    )


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
