import torch
import pytest
from torch.fx import symbolic_trace
from recip_associative_swap import swap_recip_associative


@pytest.mark.parametrize("shape", [
    (4,),
    (2, 3),
    (2, 3, 4),
])
def test_reciprocal_associative_rewrite(shape):
    class M(torch.nn.Module):
        def forward(self, A, B):
            recip_a = torch.reciprocal(A)
            a_mul_b = A * B
            recip_a_mul_b = torch.reciprocal(a_mul_b)
            return recip_a * recip_a_mul_b

    torch.manual_seed(0)
    A = torch.rand(shape) + 0.5  # Add 0.5 to ensure positive values
    B = torch.rand(shape) + 0.5

    # Original model
    model = M().eval()
    original_output = model(A, B)

    # Rewritten model
    traced = symbolic_trace(model)
    rewritten = swap_recip_associative(traced)
    rewritten_output = rewritten(A, B)

    # Verify results match
    torch.testing.assert_close(
        original_output, rewritten_output, rtol=1e-5, atol=1e-8)

    # Verify the graph was actually transformed
    graph_str = str(rewritten.graph)
    assert "square" in graph_str, "Expected torch.square in graph"
    assert "reciprocal" in graph_str, "Expected torch.reciprocal in graph"
    assert "mul" in graph_str, "Expected torch.mul in graph"


def test_no_rewrite_when_pattern_absent():
    class M(torch.nn.Module):
        def forward(self, A, B):
            return A + B  # No reciprocal or mul

    A, B = torch.rand(3), torch.rand(3)
    traced = symbolic_trace(M().eval())
    rewritten = swap_recip_associative(traced)

    # Graph should be unchanged
    assert str(traced.graph) == str(rewritten.graph)
    torch.testing.assert_close(
        traced(A, B), rewritten(A, B), rtol=1e-5, atol=1e-8)


def test_no_duplicate_mul_or_recip():
    class M(torch.nn.Module):
        def forward(self, A, B):
            recip_a = torch.reciprocal(A)
            recip_ab = torch.reciprocal(A * B)
            return recip_a * recip_ab

    traced = symbolic_trace(M())
    rewritten = swap_recip_associative(traced)

    # Should be exactly one square and one reciprocal
    squares = [n for n in rewritten.graph.nodes if n.target == torch.square]
    reciprocals = [
        n for n in rewritten.graph.nodes if n.target == torch.reciprocal]
    assert len(squares) == 1
    assert len(reciprocals) >= 1
