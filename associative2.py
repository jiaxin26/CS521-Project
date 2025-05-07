import torch
from torch.fx import symbolic_trace, GraphModule
import operator


class SqrtAssociativePass:
    """Implements the rewrite pattern: (A⊙√B)⊙(√B⊙C) => A⊙B⊙C"""

    def __init__(self):
        self.mul_patterns = {operator.mul, torch.mul, "mul"}

    def __call__(self, module):
        traced = symbolic_trace(module)
        graph = traced.graph

        # Keep track of nodes to delete
        nodes_to_delete = set()

        # Find all multiplication nodes
        for node in graph.nodes:
            is_mul_fn = node.op == "call_function" and node.target in self.mul_patterns
            is_mul_mth = node.op == "call_method" and node.target in (
                "mul", "__mul__")
            if not ((is_mul_fn or is_mul_mth) and len(node.args) == 2):
                continue

            # Check if this is a multiplication of two terms
            if len(node.args) != 2:
                continue

            lhs, rhs = node.args[0], node.args[1]

            # Check if lhs is a multiplication with sqrt (function or method)
            if isinstance(lhs, torch.fx.Node):
                is_lhs_mul_fn = lhs.op == "call_function" and lhs.target in self.mul_patterns
                is_lhs_mul_mth = lhs.op == "call_method" and lhs.target in (
                    "mul", "__mul__")
                if not ((is_lhs_mul_fn or is_lhs_mul_mth) and len(lhs.args) == 2):
                    continue

                lhs_lhs, lhs_rhs = lhs.args[0], lhs.args[1]

                # Check if lhs_rhs is a sqrt (function or method)
                is_sqrt_fn = lhs_rhs.op == "call_function" and lhs_rhs.target == torch.sqrt
                is_sqrt_mth = lhs_rhs.op == "call_method" and lhs_rhs.target == "sqrt"
                if not (is_sqrt_fn or is_sqrt_mth):
                    continue

                sqrt_arg = lhs_rhs.args[0]

                # Check if rhs is a multiplication with sqrt (fn or method)
                is_rhs_mul_fn = isinstance(
                    rhs, torch.fx.Node) and rhs.op == "call_function" and rhs.target in self.mul_patterns
                is_rhs_mul_mth = isinstance(
                    rhs, torch.fx.Node) and rhs.op == "call_method" and rhs.target in ("mul", "__mul__")
                if not ((is_rhs_mul_fn or is_rhs_mul_mth) and len(rhs.args) == 2):
                    continue

                rhs_lhs, rhs_rhs = rhs.args[0], rhs.args[1]

                # Check if rhs_lhs is the same sqrt (function or method)
                is_sqrt2_fn = rhs_lhs.op == "call_function" and rhs_lhs.target == torch.sqrt
                is_sqrt2_mth = rhs_lhs.op == "call_method" and rhs_lhs.target == "sqrt"
                if not ((is_sqrt2_fn or is_sqrt2_mth) and rhs_lhs.args[0] is sqrt_arg):
                    continue

                # Create new multiplication chain: A⊙B⊙C
                with graph.inserting_before(node):
                    # First multiply A and B
                    ab = graph.call_function(
                        torch.mul, args=(lhs_lhs, sqrt_arg), kwargs=node.kwargs)
                    # Then multiply with C
                    new_node = graph.call_function(
                        torch.mul, args=(ab, rhs_rhs), kwargs=node.kwargs)

                    # Replace the original node with the new one
                    node.replace_all_uses_with(new_node)

                    # Add nodes to delete set
                    nodes_to_delete.update(
                        [node, lhs, rhs, lhs_rhs, rhs_lhs])

        # Delete nodes in reverse order to avoid dependency issues
        for node in reversed(list(graph.nodes)):
            if node in nodes_to_delete:
                try:
                    graph.erase_node(node)
                except Exception as e:
                    # Skip if node is already deleted
                    pass

        graph.lint()
        new_mod = GraphModule(traced, graph)
        new_mod.recompile()
        return new_mod


# Test code
def test_sqrt_associative():
    class TestModule(torch.nn.Module):
        def forward(self, a, b, c):
            # Implement the (A⊙√B)⊙(√B⊙C) pattern
            sqrt_b = torch.sqrt(b)
            return (a * sqrt_b) * (sqrt_b * c)

    # Create test inputs
    a = torch.randn(2, 3) + 1  # Add 1 to ensure positive values
    b = torch.randn(2, 3) + 1  # Add 1 to ensure positive values
    c = torch.randn(2, 3)

    # Original module
    original_module = TestModule()
    original_output = original_module(a, b, c)

    # Apply optimization
    try:
        optimized_module = SqrtAssociativePass()(TestModule())
        optimized_output = optimized_module(a, b, c)

        # Manually compute expected optimized result
        expected_output = a * b * c

        # Verify results
        print("Original output:", original_output)
        print("Optimized output:", optimized_output)
        print("Expected output:", expected_output)
        print("Optimized matches original:", torch.allclose(
            original_output, optimized_output))
        print("Optimized matches expected:", torch.allclose(
            optimized_output, expected_output))

        # Print optimized graph
        print("\nOptimized graph:")
        optimized_module.graph.print_tabular()
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Print original graph to assist debugging
        print("\nOriginal graph:")
        original_module = symbolic_trace(TestModule())
        original_module.graph.print_tabular()
        raise


if __name__ == "__main__":
    test_sqrt_associative()
