import torch
import pytest
from torch.fx import symbolic_trace
from associative2 import SqrtAssociativePass

def test_sqrt_associative_basic():
    """Test basic case: (a * sqrt(b)) * (sqrt(b) * c) => a * b * c"""
    class TestModule(torch.nn.Module):
        def forward(self, a, b, c):
            sqrt_b = torch.sqrt(b)
            return (a * sqrt_b) * (sqrt_b * c)
    
    # Create test inputs with positive values
    a = torch.randn(2, 3) + 2
    b = torch.abs(torch.randn(2, 3)) + 2  # Ensure positive values
    c = torch.randn(2, 3)
    
    # Print input values for debugging
    print("Input a:", a)
    print("Input b:", b)
    print("Input c:", c)
    
    # Original module
    original_module = TestModule()
    original_output = original_module(a, b, c)
    
    # Apply optimization
    optimized_module = SqrtAssociativePass()(TestModule())
    optimized_output = optimized_module(a, b, c)
    
    # Expected output
    expected_output = a * b * c
    
    # Print debugging information
    print("Original output:", original_output)
    print("Optimized output:", optimized_output)
    print("Expected output:", expected_output)
    
    # Trace the modules to get graphs
    traced_original = symbolic_trace(original_module)
    traced_optimized = symbolic_trace(optimized_module)
    print("Original graph:", traced_original.graph)
    print("Optimized graph:", traced_optimized.graph)
    
    # Verify results
    assert torch.allclose(original_output, optimized_output), "Optimized output doesn't match original"
    assert torch.allclose(optimized_output, expected_output), "Optimized output doesn't match expected"
    
    # Verify graph structure
    graph_str = str(traced_optimized.graph)
    assert "sqrt" not in graph_str, "sqrt operations should be eliminated"
    assert graph_str.count("mul") == 2 or graph_str.count("torch.mul") == 2, "Should have exactly 2 multiplications"


def test_sqrt_associative_different_order():
    """Test with different order: (sqrt(b) * a) * (c * sqrt(b)) => a * b * c"""
    class TestModule(torch.nn.Module):
        def forward(self, a, b, c):
            sqrt_b = torch.sqrt(b)
            return (sqrt_b * a) * (c * sqrt_b)
    
    a = torch.randn(2, 3) + 2
    b = torch.abs(torch.randn(2, 3)) + 2
    c = torch.randn(2, 3)
    
    original_module = TestModule()
    optimized_module = SqrtAssociativePass()(TestModule())
    
    assert torch.allclose(
        original_module(a, b, c),
        optimized_module(a, b, c)
    ), "Optimized output doesn't match original"


def test_sqrt_associative_no_match():
    """Test case where pattern doesn't match"""
    class TestModule(torch.nn.Module):
        def forward(self, a, b, c):
            sqrt_b = torch.sqrt(b)
            sqrt_c = torch.sqrt(c)
            return (a * sqrt_b) * (sqrt_c * c)  # Different sqrt arguments
    
    a = torch.randn(2, 3) + 2
    b = torch.abs(torch.randn(2, 3)) + 2
    c = torch.abs(torch.randn(2, 3)) + 2
    
    original_module = TestModule()
    optimized_module = SqrtAssociativePass()(TestModule())
    
    # Should remain unchanged
    assert torch.allclose(
        original_module(a, b, c),
        optimized_module(a, b, c)
    ), "Output should remain unchanged when pattern doesn't match"


def test_sqrt_associative_with_constants():
    """Test with constant values"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            sqrt_x = torch.sqrt(x)
            return (2.0 * sqrt_x) * (sqrt_x * 3.0)
    
    x = torch.abs(torch.randn(2, 3)) + 2
    
    original_module = TestModule()
    optimized_module = SqrtAssociativePass()(TestModule())
    
    assert torch.allclose(
        original_module(x),
        optimized_module(x)
    ), "Optimized output doesn't match original"


def test_sqrt_associative_complex():
    """Test with more complex expressions"""
    class TestModule(torch.nn.Module):
        def forward(self, a, b, c, d):
            sqrt_b = torch.sqrt(b)
            sqrt_d = torch.sqrt(d)
            # Multiple patterns in the same graph
            term1 = (a * sqrt_b) * (sqrt_b * c)
            term2 = (c * sqrt_d) * (sqrt_d * a)
            return term1 + term2
    
    a = torch.randn(2, 3) + 2
    b = torch.abs(torch.randn(2, 3)) + 2
    c = torch.randn(2, 3)
    d = torch.abs(torch.randn(2, 3)) + 2
    
    original_module = TestModule()
    optimized_module = SqrtAssociativePass()(TestModule())
    
    assert torch.allclose(
        original_module(a, b, c, d),
        optimized_module(a, b, c, d)
    ), "Optimized output doesn't match original"


def test_sqrt_associative_method_style():
    """Test with method-style operations"""
    class TestModule(torch.nn.Module):
        def forward(self, a, b, c):
            sqrt_b = b.sqrt()
            return (a.mul(sqrt_b)).mul(sqrt_b.mul(c))
    
    a = torch.randn(2, 3) + 2
    b = torch.abs(torch.randn(2, 3)) + 2
    c = torch.randn(2, 3)
    
    original_module = TestModule()
    optimized_module = SqrtAssociativePass()(TestModule())
    
    assert torch.allclose(
        original_module(a, b, c),
        optimized_module(a, b, c)
    ), "Optimized output doesn't match original"


def test_sqrt_associative_mixed_style():
    """Test with mixed function and method style operations"""
    class TestModule(torch.nn.Module):
        def forward(self, a, b, c):
            sqrt_b = torch.sqrt(b)
            return (a.mul(sqrt_b)) * (sqrt_b.mul(c))
    
    a = torch.randn(2, 3) + 2
    b = torch.abs(torch.randn(2, 3)) + 2
    c = torch.randn(2, 3)
    
    original_module = TestModule()
    optimized_module = SqrtAssociativePass()(TestModule())
    
    assert torch.allclose(
        original_module(a, b, c),
        optimized_module(a, b, c)
    ), "Optimized output doesn't match original"


@pytest.mark.parametrize("shape", [(1,), (2, 3), (2, 3, 4), (5, 5, 5, 5)])
def test_sqrt_associative_different_shapes(shape):
    """Test with different tensor shapes"""
    class TestModule(torch.nn.Module):
        def forward(self, a, b, c):
            sqrt_b = torch.sqrt(b)
            return (a * sqrt_b) * (sqrt_b * c)
    
    a = torch.randn(shape) + 2
    b = torch.abs(torch.randn(shape)) + 2
    c = torch.randn(shape)
    
    original_module = TestModule()
    optimized_module = SqrtAssociativePass()(TestModule())
    
    assert torch.allclose(
        original_module(a, b, c),
        optimized_module(a, b, c)
    ), f"Optimized output doesn't match original for shape {shape}"
