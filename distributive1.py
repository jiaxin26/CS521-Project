import torch
from torch.fx import symbolic_trace, GraphModule
import operator
import pytest


class DistributiveRulePass:
    """Implement distributive rule: A ⊙ C + A ⊙ B → A ⊙ (B + C)"""

    def __init__(self):
        # Possible representations for addition and multiplication
        self.add_patterns = {operator.add, torch.add, "add", "__add__"}
        self.mul_patterns = {operator.mul, torch.mul, "mul", "__mul__"}

    def _is_add_node(self, node):
        if not isinstance(node, torch.fx.Node):
            return False
        return (node.op == "call_function" and node.target in self.add_patterns) or \
               (node.op == "call_method" and node.target in self.add_patterns)

    def _is_mul_node(self, node):
        if not isinstance(node, torch.fx.Node):
            return False
        return (node.op == "call_function" and node.target in self.mul_patterns) or \
            (node.op == "call_method" and node.target in self.mul_patterns)

    def __call__(self, module):
        # Symbolically trace the module
        traced = symbolic_trace(module)
        graph = traced.graph

        # First collect all matching “A*C + A*B” nodes
        matches = []
        for node in graph.nodes:
            if not self._is_add_node(node) or len(node.args) != 2:
                continue

            lhs, rhs = node.args[0], node.args[1]

            # Both sides must be multiplication nodes
            if not (self._is_mul_node(lhs) and self._is_mul_node(rhs)):
                continue
            if len(lhs.args) != 2 or len(rhs.args) != 2:
                continue

            # Find common factor in any operand position
            # A*C + A*B  or  C*A + B*A  or  A*C + B*A  or  C*A + A*B
            A, B, C = None, None, None
            for potential_A in lhs.args:
                if potential_A in rhs.args:
                    A = potential_A
                    # Get remaining terms from both sides
                    C = lhs.args[0] if lhs.args[1] is A else lhs.args[1]
                    B = rhs.args[0] if rhs.args[1] is A else rhs.args[1]
                    break
            if A is None:
                continue

            matches.append((node, A, B, C, lhs, rhs))

        # Apply each matched transformation, then erase the old nodes
        nodes_to_delete = set()
        for add_node, A, B, C, lhs_node, rhs_node in matches:
            # A*(B+C)
            with graph.inserting_before(add_node):
                sum_bc = graph.call_function(torch.add, args=(B, C))
                fused = graph.call_function(torch.mul, args=(A, sum_bc))
                add_node.replace_all_uses_with(fused)
                nodes_to_delete.update({add_node, lhs_node, rhs_node})

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

    def forward(self, a, b, c):
        # Implement the A ⊙ C + A ⊙ B pattern here
        if self.variation == 0:
            return (c * a) + (b * a)
        elif self.variation == 1:
            return (a * c) + (b * a)
        else:
            return (c * a) + (a * b)


@pytest.mark.parametrize("variation", [0, 1, 2])
def test_distributive_rule(variation):
    # Create test inputs
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    c = torch.randn(2, 3)

    # Original module
    model = TestModule(variation).eval()
    original_output = model(a, b, c)

    # Apply optimization
    try:
        optimized_module = DistributiveRulePass()(model)
        optimized_output = optimized_module(a, b, c)

        # Verify results
        print("Original output:", original_output)
        print("Optimized output:", optimized_output)
        print("Optimized matches original:", torch.allclose(
            original_output, optimized_output))

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
    for v in [0, 1, 2]:
        test_distributive_rule(v)
