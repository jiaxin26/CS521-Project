import torch
import pytest
from torch.fx import symbolic_trace
from commutative2 import CommutativePass

def test_commutative_basic():
    """Test basic case: ReduceProd(Exp(A)) => Exp(ReduceSum(A))"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            exp_x = torch.exp(x)
            return torch.prod(exp_x)
    
    # Create test inputs
    x = torch.randn(2, 3)
    
    # Print input values for debugging
    print("Input x:", x)
    
    # Original module
    original_module = TestModule()
    original_output = original_module(x)
    
    # Apply optimization
    optimized_module = CommutativePass()(TestModule())
    optimized_output = optimized_module(x)
    
    # Expected output
    expected_output = torch.exp(torch.sum(x))
    
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
    assert "prod" not in graph_str, "prod operations should be eliminated"
    assert "exp" in graph_str, "exp operation should be present"
    assert "sum" in graph_str, "sum operation should be present"


def test_commutative_method_style():
    """Test with method-style operations"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x.exp().prod()
    
    x = torch.randn(2, 3)
    
    original_module = TestModule()
    optimized_module = CommutativePass()(TestModule())
    
    assert torch.allclose(
        original_module(x),
        optimized_module(x)
    ), "Optimized output doesn't match original"


def test_commutative_with_dim():
    """Test with dimension specification"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            exp_x = torch.exp(x)
            return torch.prod(exp_x, dim=1)
    
    x = torch.randn(2, 3)
    
    original_module = TestModule()
    optimized_module = CommutativePass()(TestModule())
    
    # Print debugging information
    print("Input x:", x)
    print("Original output:", original_module(x))
    print("Optimized output:", optimized_module(x))
    
    assert torch.allclose(
        original_module(x),
        optimized_module(x)
    ), "Optimized output doesn't match original"


def test_commutative_with_multiple_dims():
    """Test with multiple dimensions"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            exp_x = torch.exp(x)
            # First reduce along dim 0, then dim 1
            return torch.prod(torch.prod(exp_x, dim=0), dim=0)
    
    x = torch.randn(2, 3, 4)
    
    original_module = TestModule()
    optimized_module = CommutativePass()(TestModule())
    
    # Print debugging information
    print("Input shape:", x.shape)
    print("Original output shape:", original_module(x).shape)
    print("Optimized output shape:", optimized_module(x).shape)
    print("Original output:", original_module(x))
    print("Optimized output:", optimized_module(x))
    
    assert torch.allclose(
        original_module(x),
        optimized_module(x)
    ), "Optimized output doesn't match original"


def test_commutative_with_keepdim():
    """Test with keepdim parameter"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            exp_x = torch.exp(x)
            return torch.prod(exp_x, dim=1, keepdim=True)
    
    x = torch.randn(2, 3)
    
    original_module = TestModule()
    optimized_module = CommutativePass()(TestModule())
    
    assert torch.allclose(
        original_module(x),
        optimized_module(x)
    ), "Optimized output doesn't match original"


def test_commutative_with_all_params():
    """Test with all possible parameters"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            exp_x = torch.exp(x)
            # First reduce along dim 0, then dim 1, with keepdim=True
            return torch.prod(torch.prod(exp_x, dim=0, keepdim=True, dtype=torch.float64), 
                            dim=0, keepdim=True, dtype=torch.float64)
    
    x = torch.randn(2, 3, 4)
    
    original_module = TestModule()
    optimized_module = CommutativePass()(TestModule())
    
    # Print debugging information
    print("Input shape:", x.shape)
    print("Original output shape:", original_module(x).shape)
    print("Optimized output shape:", optimized_module(x).shape)
    print("Original output:", original_module(x))
    print("Optimized output:", optimized_module(x))
    
    assert torch.allclose(
        original_module(x),
        optimized_module(x)
    ), "Optimized output doesn't match original"


def test_commutative_no_match():
    """Test case where pattern doesn't match"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            # Just exp without prod
            return torch.exp(x)
    
    x = torch.randn(2, 3)
    
    original_module = TestModule()
    optimized_module = CommutativePass()(TestModule())
    
    # Should remain unchanged
    assert torch.allclose(
        original_module(x),
        optimized_module(x)
    ), "Output should remain unchanged when pattern doesn't match"


def test_commutative_complex():
    """Test with more complex expressions"""
    class TestModule(torch.nn.Module):
        def forward(self, x, y):
            # Multiple patterns in the same graph
            term1 = torch.prod(torch.exp(x))
            term2 = torch.prod(torch.exp(y))
            return term1 + term2
    
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    
    original_module = TestModule()
    optimized_module = CommutativePass()(TestModule())
    
    assert torch.allclose(
        original_module(x, y),
        optimized_module(x, y)
    ), "Optimized output doesn't match original"


@pytest.mark.parametrize("shape", [(1,), (2, 3), (2, 3, 4), (5, 5, 5, 5)])
def test_commutative_different_shapes(shape):
    """Test with different tensor shapes"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            exp_x = torch.exp(x)
            return torch.prod(exp_x)
    
    x = torch.randn(*shape)  # Unpack the shape tuple
    
    original_module = TestModule()
    optimized_module = CommutativePass()(TestModule())
    
    # Print debugging information
    print(f"\nTesting shape {shape}:")
    print("Input shape:", x.shape)
    print("Original output:", original_module(x))
    print("Optimized output:", optimized_module(x))
    
    assert torch.allclose(
        original_module(x),
        optimized_module(x)
    ), f"Optimized output doesn't match original for shape {shape}"


def test_commutative_numerical_stability():
    """Test numerical stability with various value ranges"""
    class TestModule(torch.nn.Module):
        def forward(self, x):
            exp_x = torch.exp(x)
            return torch.prod(exp_x)
    
    # Test case 1: Large positive values
    x1 = torch.tensor([10.0, 20.0, 30.0])
    original_module = TestModule()
    optimized_module = CommutativePass()(TestModule())
    
    # Both should give the same result, but optimized version should be more stable
    original_output1 = original_module(x1)
    optimized_output1 = optimized_module(x1)
    
    # Use relative tolerance for large values
    assert torch.allclose(original_output1, optimized_output1, rtol=1e-5), "Optimized output doesn't match original for large positive values"
    assert not torch.isnan(optimized_output1), "Optimized output should not be NaN for large positive values"
    assert not torch.isinf(optimized_output1), "Optimized output should not be Inf for large positive values"
    
    # Test case 2: Large negative values
    x2 = torch.tensor([-10.0, -20.0, -30.0])
    original_output2 = original_module(x2)
    optimized_output2 = optimized_module(x2)
    
    # Use relative tolerance for small values
    assert torch.allclose(original_output2, optimized_output2, rtol=1e-5), "Optimized output doesn't match original for large negative values"
    assert not torch.isnan(optimized_output2), "Optimized output should not be NaN for large negative values"
    assert not torch.isinf(optimized_output2), "Optimized output should not be Inf for large negative values"
    
    # Test case 3: Mixed positive and negative values
    x3 = torch.tensor([-5.0, 0.0, 5.0])
    original_output3 = original_module(x3)
    optimized_output3 = optimized_module(x3)
    
    # Use default tolerance for moderate values
    assert torch.allclose(original_output3, optimized_output3), "Optimized output doesn't match original for mixed values"
    assert not torch.isnan(optimized_output3), "Optimized output should not be NaN for mixed values"
    assert not torch.isinf(optimized_output3), "Optimized output should not be Inf for mixed values"
    
    # Test case 4: Values near zero
    x4 = torch.tensor([-0.1, 0.0, 0.1])
    original_output4 = original_module(x4)
    optimized_output4 = optimized_module(x4)
    
    # Use absolute tolerance for values near zero
    assert torch.allclose(original_output4, optimized_output4, atol=1e-7), "Optimized output doesn't match original for values near zero"
    assert not torch.isnan(optimized_output4), "Optimized output should not be NaN for values near zero"
    assert not torch.isinf(optimized_output4), "Optimized output should not be Inf for values near zero"
    
    # Print results for debugging
    print("\nNumerical Stability Test Results:")
    print("Large positive values:")
    print(f"Original: {original_output1}, Optimized: {optimized_output1}")
    print(f"Relative difference: {abs(original_output1 - optimized_output1) / abs(original_output1)}")
    
    print("\nLarge negative values:")
    print(f"Original: {original_output2}, Optimized: {optimized_output2}")
    print(f"Relative difference: {abs(original_output2 - optimized_output2) / abs(original_output2)}")
    
    print("\nMixed values:")
    print(f"Original: {original_output3}, Optimized: {optimized_output3}")
    print(f"Absolute difference: {abs(original_output3 - optimized_output3)}")
    
    print("\nValues near zero:")
    print(f"Original: {original_output4}, Optimized: {optimized_output4}")
    print(f"Absolute difference: {abs(original_output4 - optimized_output4)}") 
