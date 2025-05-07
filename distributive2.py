import torch
from torch.fx import symbolic_trace, GraphModule
import operator
import pytest


class DistributiveRule2Pass:
    """Implement distributive rule 2: A + A ⊙ B → A ⊙ (B + 1)"""

    def __init__(self):
        # Possible representations for addition and multiplication
        self.add_patterns = {operator.add, torch.add, "add", "__add__"}
        self.mul_patterns = {operator.mul, torch.mul, "mul", "__mul__"}

    def _is_add_node(self, node):
        return (node.op == "call_function" and node.target in self.add_patterns) or \
               (node.op == "call_method" and node.target in self.add_patterns)

    def _is_mul_node(self, node):
        return (node.op == "call_function" and node.target in self.mul_patterns) or \
               (node.op == "call_method" and node.target in self.mul_patterns)

    def __call__(self, module):
        # Symbolically trace the module
        traced = symbolic_trace(module)
        graph = traced.graph

        # First collect all matching "A + A*B" nodes
        matches = []
        for node in graph.nodes:
            if not self._is_add_node(node) or len(node.args) != 2:
                continue

            lhs, rhs = node.args[0], node.args[1]

            # Check pattern: A + (A * B)
            if self._is_mul_node(rhs) and len(rhs.args) == 2:
                if lhs is rhs.args[0]:  # A + (A * B)
                    matches.append((node, lhs, rhs, rhs.args[1]))
                elif lhs is rhs.args[1]:  # A + (B * A)
                    matches.append((node, lhs, rhs, rhs.args[0]))
            
            # Check pattern: (A * B) + A
            if self._is_mul_node(lhs) and len(lhs.args) == 2:
                if rhs is lhs.args[0]:  # (A * B) + A
                    matches.append((node, rhs, lhs, lhs.args[1]))
                elif rhs is lhs.args[1]:  # (B * A) + A
                    matches.append((node, rhs, lhs, lhs.args[0]))

        # Apply each matched transformation, then erase the old nodes
        nodes_to_delete = set()
        for add_node, a_term, mul_node, b_term in matches:
            # A*(B+1)
            with graph.inserting_before(add_node):
                # First create a tensor of ones matching B's shape
                one_node = graph.call_function(torch.ones_like, args=(b_term,))
                
                # Then compute B + 1
                b_plus_one = graph.call_function(torch.add, args=(b_term, one_node))
                
                # Finally compute A * (B + 1)
                fused = graph.call_function(torch.mul, args=(a_term, b_plus_one))
                
                add_node.replace_all_uses_with(fused)
                nodes_to_delete.update({add_node, mul_node})

        # Erase dead nodes
        for node in reversed(list(graph.nodes)):
            if node in nodes_to_delete and not node.users:
                graph.erase_node(node)

        graph.lint()

        new_mod = GraphModule(traced, graph)
        new_mod.recompile()
        return new_mod


# Test code
class TestModule(torch.nn.Module):
    def __init__(self, variation):
        super().__init__()
        self.variation = variation

    def forward(self, a, b):
        # Implement the A + A*B pattern in different variations
        if self.variation == 0:
            return a + a * b  # A + (A * B)
        elif self.variation == 1:
            return a + b * a  # A + (B * A)
        elif self.variation == 2:
            return a * b + a  # (A * B) + A
        else:
            return b * a + a  # (B * A) + A


@pytest.mark.parametrize("variation", [0, 1, 2, 3])
def test_distributive_rule2(variation):
    # Create test inputs
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)

    # Original module
    model = TestModule(variation).eval()
    original_output = model(a, b)

    # Apply optimization
    try:
        optimized_module = DistributiveRule2Pass()(model)
        optimized_output = optimized_module(a, b)

        # Manually compute expected optimized result
        expected_output = a * (b + 1)

        # Verify results
        print(f"Testing variation {variation}")
        print("Original output:", original_output)
        print("Optimized output:", optimized_output)
        print("Expected output:", expected_output)
        print("Optimized matches original:", torch.allclose(original_output, optimized_output))
        print("Optimized matches expected:", torch.allclose(optimized_output, expected_output))

        # Print optimized graph
        print("\nOptimized graph:")
        optimized_module.graph.print_tabular()
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Print original graph to assist debugging
        print("\nOriginal graph:")
        traced = symbolic_trace(model)
        traced.graph.print_tabular()
        raise


if __name__ == "__main__":
    for v in range(4):
        test_distributive_rule2(v)
        print("\n" + "="*50 + "\n")
