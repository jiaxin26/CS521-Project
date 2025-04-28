import torch
import pytest
from torch.fx import symbolic_trace
from main import optimize_graph


def _run_optimization_test(model, inputs, expect_optimization=True):
    model = model.eval()
    traced = symbolic_trace(model)
    ref_out = traced(*inputs)

    optimized = optimize_graph(traced)
    opt_out = optimized(*inputs)

    torch.testing.assert_close(ref_out, opt_out, rtol=1e-4, atol=1e-4)

    optimized_graph_str = str(optimized.graph)

    if expect_optimization:
        assert "call_function" in optimized_graph_str or "call_method" in optimized_graph_str
        return optimized, optimized_graph_str
    else:
        assert optimized_graph_str == str(traced.graph)
        return optimized, optimized_graph_str


class FusionAndAssociativeSwapModel(torch.nn.Module):
    def forward(self, x):
        # Fusion chain
        fused = torch.sigmoid(torch.relu(torch.add(x, 1)))

        # associative swap
        recip_a = torch.reciprocal(x)
        recip_ab = torch.reciprocal(x * recip_a)
        assoc = recip_a * recip_ab

        return fused + assoc


class NoRewriteModel(torch.nn.Module):
    def forward(self, x):
        return torch.matmul(x, x)  # Should not fuse or rewrite


def test_fusion_and_associative_swap():
    model = FusionAndAssociativeSwapModel()
    x = torch.randn(4, 4) + 1e-6  # avoid zero division
    optimized, optimized_graph_str = _run_optimization_test(
        model, (x,), expect_optimization=True)

    # Check for reciprocal associative rewrite signatures
    assert "square" in optimized_graph_str, "Reciprocal associative rewrite not applied"
    assert "reciprocal" in optimized_graph_str, "Reciprocal associative rewrite not applied"
    assert "mul" in optimized_graph_str, "Reciprocal associative rewrite not applied"

    # Check for fusion
    assert any(
        isinstance(node.target, type(lambda: 0)) and getattr(
            node.target, "is_fused_function", False)
        for node in optimized.graph.nodes
        if node.op == "call_function"
    ), "Fusion not applied"

    # Original operations should be gone
    assert "sigmoid" not in optimized_graph_str, "Original operations detected; not fused"


def test_no_rewrite():
    model = NoRewriteModel()
    x = torch.randn(4, 4)
    _run_optimization_test(model, (x,), expect_optimization=False)
