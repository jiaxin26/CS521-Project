import torch
import operator


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

        recip_1, recip_2 = lhs, rhs
        arg_1, arg_2 = get_recip_arg(lhs), get_recip_arg(rhs)

        # Try: arg_2 = mul(A, B) where A == arg_1
        match_found = False
        if (
            arg_2.op == "call_function"
            and arg_2.target in (torch.mul, operator.mul)
            and arg_1 in arg_2.args
        ):
            a_node = arg_1
            mul_node = arg_2
            b_node = arg_2.args[1] if arg_2.args[0] == a_node else arg_2.args[0]
            match_found = True
        # Try reversed pattern: arg_1 = mul(A, B) where A == arg_2
        elif (
            arg_1.op == "call_function"
            and arg_1.target in (torch.mul, operator.mul)
            and arg_2 in arg_1.args
        ):
            a_node = arg_2
            mul_node = arg_1
            b_node = arg_1.args[1] if arg_1.args[0] == a_node else arg_1.args[0]
            recip_1, recip_2 = rhs, lhs
            match_found = True

        if not match_found:
            continue

        with gm.graph.inserting_before(node):
            square_recip_a = gm.graph.call_function(torch.square, (recip_1,))
            recip_b = gm.graph.call_function(torch.reciprocal, (b_node,))
            new_mul = gm.graph.call_function(
                torch.mul, (square_recip_a, recip_b))

        node.replace_all_uses_with(new_mul)

        # Clean up dead nodes if no other users
        for dead_node in [node, recip_2, mul_node]:
            if len(dead_node.users) == 0:
                gm.graph.erase_node(dead_node)

        modified = True

    if modified:
        gm.recompile()
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
