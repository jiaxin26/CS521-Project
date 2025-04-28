import torch
from torch.fx import symbolic_trace
import operator

class DistributiveRulePass:
    """Implement distributive rule: A ⊙ C + A ⊙ B → A ⊙ (B + C)"""
    
    def __init__(self):
        # Possible representations for addition and multiplication
        self.add_patterns = {operator.add, torch.add, "add"}
        self.mul_patterns = {operator.mul, torch.mul, "mul"}
    
    def __call__(self, module):
        # Symbolically trace the module
        traced = symbolic_trace(module)
        graph = traced.graph
        
        # First collect all matching nodes
        matches = []
        for node in graph.nodes:
            if not (any(node.target == pattern for pattern in self.add_patterns) and len(node.args) >= 2):
                continue
                
            lhs, rhs = node.args[0], node.args[1]
            
            # Both sides must be multiplication nodes
            if (isinstance(lhs, torch.fx.Node) and 
                any(lhs.target == pattern for pattern in self.mul_patterns) and 
                isinstance(rhs, torch.fx.Node) and 
                any(rhs.target == pattern for pattern in self.mul_patterns) and
                len(lhs.args) >= 2 and len(rhs.args) >= 2):
                
                # Check if the first operand of both multiplications is the same (A)
                if str(lhs.args[0]) == str(rhs.args[0]):
                    # Pattern: (A * C) + (A * B)
                    matches.append((node, lhs.args[0], lhs.args[1], rhs.args[1]))
                # Check the other possible positions of A
                elif str(lhs.args[0]) == str(rhs.args[1]):
                    # Pattern: (A * C) + (B * A)
                    matches.append((node, lhs.args[0], lhs.args[1], rhs.args[0]))
                elif str(lhs.args[1]) == str(rhs.args[0]):
                    # Pattern: (C * A) + (A * B)
                    matches.append((node, lhs.args[1], lhs.args[0], rhs.args[1]))
                elif str(lhs.args[1]) == str(rhs.args[1]):
                    # Pattern: (C * A) + (B * A)
                    matches.append((node, lhs.args[1], lhs.args[0], rhs.args[0]))
        
        # Create a new module to avoid modifying the original graph
        new_module = torch.fx.GraphModule(module, graph)
        
        # Apply the transformation to each matched node
        for add_node, a_term, b_term, c_term in matches:
            self._apply_transformation(graph, add_node, a_term, b_term, c_term)
        
        # Recompile the modified graph
        new_module.recompile()
        
        return new_module
    
    def _apply_transformation(self, graph, add_node, a_term, b_term, c_term):
        """Apply transformation: A ⊙ C + A ⊙ B → A ⊙ (B + C)"""
        with graph.inserting_before(add_node):
            # First compute B + C
            b_plus_c = graph.call_function(torch.add, args=(b_term, c_term))
            
            # Then compute A * (B + C)
            result = graph.call_function(torch.mul, args=(a_term, b_plus_c))
            
            # Replace the original add node with the new result
            add_node.replace_all_uses_with(result)

# Test code
def test_distributive_rule():
    # Create a simple module
    class TestModule(torch.nn.Module):
        def forward(self, a, b, c):
            # Implement the A ⊙ C + A ⊙ B pattern here
            return a * c + a * b
    
    # Create test inputs
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    c = torch.randn(2, 3)
    
    # Original module
    original_module = TestModule()
    original_output = original_module(a, b, c)
    
    # Apply optimization
    try:
        optimized_module = DistributiveRulePass()(TestModule())
        optimized_output = optimized_module(a, b, c)
        
        # Manually compute expected optimized result
        expected_output = a * (b + c)
        
        # Verify results
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
        original_module = symbolic_trace(TestModule())
        original_module.graph.print_tabular()
        raise

if __name__ == "__main__":
    test_distributive_rule()