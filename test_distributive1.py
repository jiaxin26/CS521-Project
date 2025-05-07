import torch
import pytest
from torch.fx import symbolic_trace
from distributive1 import DistributiveRulePass


def make_model(expr):
    class PatternModule(torch.nn.Module):
        def forward(self, a, b, c):
            return expr(a, b, c)
    return PatternModule()


expressions = [
    lambda a, b, c: a * c + a * b,
    lambda a, b, c: c * a + b * a,
    lambda a, b, c: a * c + b * a,
    lambda a, b, c: c * a + a * b,
]


class NoPatternModule(torch.nn.Module):
    def forward(self, a, b, c):
        # No distributive pattern here: A * B + C * C
        return a * b + c * c


@pytest.mark.parametrize("expr", expressions)
def test_distributive_rule_simple(expr):
    torch.manual_seed(0)
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    c = torch.randn(3, 3)

    model = make_model(expr).eval()
    # Original vs optimized outputs
    original = model(a, b, c)
    optimized = DistributiveRulePass()(model)(a, b, c)
    expected = a * (b + c)

    # Verify functional correctness
    assert torch.allclose(optimized, original, rtol=1e-5, atol=1e-8)
    assert torch.allclose(optimized, expected, rtol=1e-5, atol=1e-8)


def test_distributive_rule_no_match():
    torch.manual_seed(0)
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    c = torch.randn(3, 3)

    # Trace original
    original_module = NoPatternModule().eval()
    traced = symbolic_trace(original_module)

    # Apply pass and re-trace
    optimized_module = DistributiveRulePass()(NoPatternModule())

    # Graph should remain unchanged
    assert str(traced.graph) == str(optimized_module.graph)

    # Outputs should match
    original = original_module(a, b, c)
    optimized = optimized_module(a, b, c)
    assert torch.allclose(optimized, original, rtol=1e-5, atol=1e-8)
