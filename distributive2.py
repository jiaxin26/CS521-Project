import torch
from torch.fx import symbolic_trace
import operator

class DistributiveRule2Pass:
    """Implement distributive rule 2: A + A ⊙ B → A ⊙ (B + 1)"""
    
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
            
            # Check right side for multiplication: A + (A * B)
            if (isinstance(rhs, torch.fx.Node) and 
                any(rhs.target == pattern for pattern in self.mul_patterns) and 
                len(rhs.args) >= 2 and 
                str(rhs.args[0]) == str(lhs)):
                
                matches.append((node, lhs, rhs, rhs.args[1]))
            
            # Check left side for multiplication: (A * B) + A
            elif (isinstance(lhs, torch.fx.Node) and 
                  any(lhs.target == pattern for pattern in self.mul_patterns) and 
                  len(lhs.args) >= 2 and 
                  str(lhs.args[0]) == str(rhs)):
                
                matches.append((node, rhs, lhs, lhs.args[1]))
        
        # Create a new module to avoid modifying the original graph
        new_module = torch.fx.GraphModule(module, graph)
        
        # Apply the transformation to each matched node
        for add_node, a_term, mul_node, b_term in matches:
            self._apply_transformation(graph, add_node, a_term, b_term)
        
        # Recompile the modified graph
        new_module.recompile()
        
        return new_module
    
    def _apply_transformation(self, graph, add_node, a_term, b_term):
        """Apply transformation: A + A * B → A * (B + 1)"""
        with graph.inserting_before(add_node):
            # First create a tensor of ones matching B's shape
            one_node = graph.call_function(torch.ones_like, args=(b_term,))
            
            # Then compute B + 1
            b_plus_one = graph.call_function(torch.add, args=(b_term, one_node))
            
            # Finally compute A * (B + 1)
            result = graph.call_function(torch.mul, args=(a_term, b_plus_one))
            
            # Replace the original add node with the new result
            add_node.replace_all_uses_with(result)

# Test code
def test_distributive_rule2():
    # Create a simple module
    class TestModule(torch.nn.Module):
        def forward(self, x, y):
            # Implement the A + A * B pattern here
            return x + x * y
    
    # Create test inputs
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    
    # Original module
    original_module = TestModule()
    original_output = original_module(x, y)
    
    # Apply optimization
    try:
        optimized_module = DistributiveRule2Pass()(TestModule())
        optimized_output = optimized_module(x, y)
        
        # Manually compute expected optimized result
        expected_output = x * (y + 1)
        
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
    test_distributive_rule2()
