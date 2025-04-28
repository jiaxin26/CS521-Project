import torch
import pytest
from torch.fx import symbolic_trace
from distributive2 import DistributiveRule2Pass


class PatternModule(torch.nn.Module):
    def forward(self, x, y):
        # Pattern: A + A * B
        return x + x * y


class NoPatternModule(torch.nn.Module):
    def forward(self, x, y):
        # No pattern here: A + B
        return x + y


def test_distributive_rule2_simple():
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    y = torch.randn(4, 5)

    # Original vs optimized outputs
    original = PatternModule()(x, y)
    optimized_module = DistributiveRule2Pass()(PatternModule())
    optimized = optimized_module(x, y)
    expected = x * (y + torch.ones_like(y))

    # Verify functional correctness
    assert torch.allclose(optimized, original, rtol=1e-5, atol=1e-8)
    assert torch.allclose(optimized, expected, rtol=1e-5, atol=1e-8)


def test_distributive_rule2_no_match():
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    y = torch.randn(4, 5)

    # Trace original
    original_module = NoPatternModule()
    traced = symbolic_trace(original_module)

    # Apply pass and re-trace
    optimized_module = DistributiveRule2Pass()(NoPatternModule())

    # Graph should remain unchanged
    assert str(traced.graph) == str(optimized_module.graph)

    # Outputs should match
    original = original_module(x, y)
    optimized = optimized_module(x, y)
    assert torch.allclose(optimized, original, rtol=1e-5, atol=1e-8)
