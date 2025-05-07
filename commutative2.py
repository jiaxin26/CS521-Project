import torch
from torch.fx import symbolic_trace, GraphModule
import operator


class CommutativePass:
    """Implements the rewrite pattern: ReduceProd(Exp(A)) => Exp(ReduceSum(A))"""

    def __init__(self):
        self.reduce_prod_patterns = {torch.prod, "prod"}
        self.exp_patterns = {torch.exp, "exp"}

    def __call__(self, module):
        traced = symbolic_trace(module)
        graph = traced.graph

        # Keep track of nodes to delete
        nodes_to_delete = set()

        # Find all prod nodes (function and method)
        for node in graph.nodes:
            is_prod = (
                (node.op == "call_function" and node.target in self.reduce_prod_patterns) or (
                    node.op == "call_method" and node.target == "prod")
            )
            if not is_prod or len(node.args) < 1:
                continue

            # Get the argument of reduce_prod
            arg = node.args[0]

            # Check if the argument is an exp operation
            if not (
                isinstance(arg, torch.fx.Node) and (
                    (arg.op == "call_function" and arg.target in self.exp_patterns)
                    or (arg.op == "call_method" and arg.target == "exp")
                )
            ):
                continue

            # Get the argument of exp
            exp_arg = arg.args[0]

            # Get any additional arguments (like dim) from the prod operation
            kwargs = node.kwargs if hasattr(node, 'kwargs') else {}

            # Create new operations: Exp(ReduceSum(A))
            with graph.inserting_before(node):
                # First compute ReduceSum(A) with the same dimension if specified
                reduce_sum = graph.call_function(
                    torch.sum, args=(exp_arg,), kwargs=kwargs)
                # Then compute Exp(ReduceSum(A))
                new_node = graph.call_function(
                    torch.exp, args=(reduce_sum,))

                # Replace the original node with the new one
                node.replace_all_uses_with(new_node)

                # Add nodes to delete set
                nodes_to_delete.update([node, arg])

        # Delete nodes in reverse order to avoid dependency issues
        for node in reversed(list(graph.nodes)):
            if node in nodes_to_delete:
                try:
                    graph.erase_node(node)
                except Exception as e:
                    # Skip if node is already deleted
                    pass

        traced.recompile()
        return traced


# Test code
def test_commutative():
    class TestModule(torch.nn.Module):
        def forward(self, x):
            # Implement the ReduceProd(Exp(A)) pattern
            exp_x = torch.exp(x)
            return torch.prod(exp_x)

    # Create test inputs
    x = torch.randn(2, 3)

    # Original module
    original_module = TestModule()
    original_output = original_module(x)

    # Apply optimization
    try:
        optimized_module = CommutativePass()(TestModule())
        optimized_output = optimized_module(x)

        # Manually compute expected optimized result
        expected_output = torch.exp(torch.sum(x))

        # Verify results
        print("Input x:", x)
        print("Original output:", original_output)
        print("Optimized output:", optimized_output)
        print("Expected output:", expected_output)
        print("Optimized matches original:", torch.allclose(
            original_output, expected_output))
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
    test_commutative()
