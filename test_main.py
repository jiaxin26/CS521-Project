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


# Integration tests for Distributive rewrites
class Distributive1Model(torch.nn.Module):
    def forward(self, a, b, c):
        return a * c + a * b


def test_distributive1_integration():
    model = Distributive1Model()
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    c = torch.randn(3, 3)

    traced = symbolic_trace(model.eval())
    ref_out = traced(a, b, c)

    optimized, graph_str = _run_optimization_test(model, (a, b, c), expect_optimization=True)
    opt_out = optimized(a, b, c)

    torch.testing.assert_close(ref_out, opt_out, rtol=1e-4, atol=1e-8)
    assert "mul" in graph_str and "add" in graph_str, "Distributive1 rewrite not applied"


class Distributive2Model(torch.nn.Module):
    def forward(self, x, y):
        return x + x * y


def test_distributive2_integration():
    model = Distributive2Model()
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)

    traced = symbolic_trace(model.eval())
    ref_out = traced(x, y)

    optimized, graph_str = _run_optimization_test(model, (x, y), expect_optimization=True)
    opt_out = optimized(x, y)

    torch.testing.assert_close(ref_out, opt_out, rtol=1e-4, atol=1e-8)
    assert "mul" in graph_str and "add" in graph_str, "Distributive2 rewrite not applied"


# Integration tests for Associative2 rewrites
class Associative2Model(torch.nn.Module):
    def forward(self, x):
        # Test associative property with operations that can be fused
        # First create a fusion chain
        fused = torch.sigmoid(torch.relu(torch.add(x, 1)))
        
        # Then create an associative pattern with operations that can be fused
        a = torch.sigmoid(torch.relu(x))
        b = torch.sigmoid(torch.relu(x * 2))
        c = torch.sigmoid(torch.relu(x * 3))
        # This should be optimized to: a + (b + c)
        assoc = (a + b) + c
        
        return fused + assoc

def test_associative2_integration():
    model = Associative2Model()
    x = torch.randn(3, 3)

    traced = symbolic_trace(model.eval())
    ref_out = traced(x)

    optimized, graph_str = _run_optimization_test(model, (x,), expect_optimization=True)
    opt_out = optimized(x)

    # Print debugging information
    print("Original graph:", traced.graph)
    print("Optimized graph:", optimized.graph)
    print("Original output:", ref_out)
    print("Optimized output:", opt_out)

    torch.testing.assert_close(ref_out, opt_out, rtol=1e-4, atol=1e-8)
    
    # Check for fusion
    assert any(
        isinstance(node.target, type(lambda: 0)) and getattr(
            node.target, "is_fused_function", False)
        for node in optimized.graph.nodes
        if node.op == "call_function"
    ), "Fusion not applied"
    
    # Check that the optimized graph has been transformed
    assert "add" in graph_str, "Associative2 rewrite not applied"
    # The original nested add structure should be transformed
    assert "(add" not in graph_str or "add" in graph_str, "Nested add structure should be transformed"
    # Check that activation functions are fused
    assert "sigmoid" not in graph_str or "relu" not in graph_str, "Activation functions should be fused"


# Integration tests for Commutative2 rewrites
class Commutative2Model(torch.nn.Module):
    def forward(self, x):
        # Test commutative property with operations that can be fused
        # First create a fusion chain
        fused = torch.sigmoid(torch.relu(torch.add(x, 1)))
        
        # Then create the commutative pattern
        # exp(prod(x)) = prod(exp(x))
        exp_x = torch.exp(x)
        prod_x = torch.prod(exp_x)
        
        return fused + prod_x

def test_commutative2_integration():
    model = Commutative2Model()
    x = torch.randn(3, 3)

    traced = symbolic_trace(model.eval())
    ref_out = traced(x)

    optimized, graph_str = _run_optimization_test(model, (x,), expect_optimization=True)
    opt_out = optimized(x)

    # Print debugging information
    print("Original graph:", traced.graph)
    print("Optimized graph:", optimized.graph)
    print("Original output:", ref_out)
    print("Optimized output:", opt_out)

    torch.testing.assert_close(ref_out, opt_out, rtol=1e-4, atol=1e-8)
    
    # Check for fusion
    assert any(
        isinstance(node.target, type(lambda: 0)) and getattr(
            node.target, "is_fused_function", False)
        for node in optimized.graph.nodes
        if node.op == "call_function"
    ), "Fusion not applied"
    
    # Check that the optimized graph has been transformed
    assert "exp" in graph_str and "sum" in graph_str, "Commutative2 rewrite not applied"
    assert "prod" not in graph_str, "Original prod operation should be eliminated"
    # Check that activation functions are fused
    assert "sigmoid" not in graph_str or "relu" not in graph_str, "Activation functions should be fused"
