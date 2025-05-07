import torch
import pytest
from torch.fx import symbolic_trace
from distributive2 import DistributiveRule2Pass


def make_model(expr):
    class PatternModule(torch.nn.Module):
        def forward(self, a, b):
            return expr(a, b)
    return PatternModule()


expressions = [
    lambda a, b: a + a * b,     # A + (A * B)
    lambda a, b: a + b * a,     # A + (B * A)
    lambda a, b: a * b + a,     # (A * B) + A
    lambda a, b: b * a + a,     # (B * A) + A
]


class NoPatternModule(torch.nn.Module):
    def forward(self, a, b):
        # No distributive pattern here: A + B
        return a + b


@pytest.mark.parametrize("expr", expressions)
def test_distributive_rule2_simple(expr):
    torch.manual_seed(0)
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)

    model = make_model(expr).eval()
    # Original vs optimized outputs
    original = model(a, b)
    optimized = DistributiveRule2Pass()(model)(a, b)
    expected = a * (b + torch.ones_like(b))

    # Verify functional correctness
    assert torch.allclose(optimized, original, rtol=1e-5, atol=1e-8)
    assert torch.allclose(optimized, expected, rtol=1e-5, atol=1e-8)


def test_distributive_rule2_no_match():
    torch.manual_seed(0)
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)

    # Trace original
    original_module = NoPatternModule().eval()
    traced = symbolic_trace(original_module)

    # Apply pass and re-trace
    optimized_module = DistributiveRule2Pass()(NoPatternModule())

    # Graph should remain unchanged
    assert str(traced.graph) == str(optimized_module.graph)

    # Outputs should match
    original = original_module(a, b)
    optimized = optimized_module(a, b)
    assert torch.allclose(optimized, original, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("shape", [(4, 5), (2, 3, 4), (1, 1)])
def test_distributive_rule2_different_shapes(shape):
    """Test the rule with tensors of different shapes"""
    torch.manual_seed(0)
    a = torch.randn(*shape)
    b = torch.randn(*shape)

    # Test all expression variations with this shape
    for expr in expressions:
        model = make_model(expr).eval()
        original = model(a, b)
        optimized = DistributiveRule2Pass()(model)(a, b)
        expected = a * (b + torch.ones_like(b))

        # Verify functional correctness
        assert torch.allclose(optimized, original, rtol=1e-5, atol=1e-8)
        assert torch.allclose(optimized, expected, rtol=1e-5, atol=1e-8)


def test_distributive_rule2_multiple_patterns():
    """Test with a model containing multiple patterns in sequence"""
    class MultiPatternModule(torch.nn.Module):
        def forward(self, x, y, z):
            # Two separate patterns: (x + x*y) + (z + z*y)
            return (x + x * y) + (z + z * y)

    torch.manual_seed(0)
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.randn(3, 3)

    original_module = MultiPatternModule().eval()
    original = original_module(x, y, z)
    
    optimized_module = DistributiveRule2Pass()(MultiPatternModule())
    optimized = optimized_module(x, y, z)
    
    # Expected result after optimization: x*(y+1) + z*(y+1)
    expected = x * (y + torch.ones_like(y)) + z * (y + torch.ones_like(y))
    
    assert torch.allclose(optimized, original, rtol=1e-5, atol=1e-8)
    assert torch.allclose(optimized, expected, rtol=1e-5, atol=1e-8)
