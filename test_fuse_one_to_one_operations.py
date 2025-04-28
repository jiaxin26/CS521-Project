import torch
import pytest
from torch.fx import symbolic_trace
from fuse_one_to_one_operations import fuse_elementwise_chains


def _run_fusion_test(model, inputs, expect_fusion=True):
    model = model.eval()
    traced = symbolic_trace(model)
    ref_out = traced(*inputs)

    fused = fuse_elementwise_chains(traced)
    fused_out = fused(*inputs)

    torch.testing.assert_close(ref_out, fused_out, rtol=1e-4, atol=1e-4)

    fused_graph_str = str(fused.graph)

    if expect_fusion:
        assert "call_function" in fused_graph_str or "call_method" in fused_graph_str
        assert "fused_function" in fused_graph_str, "Fusion was not applied"
        return fused_graph_str
    else:
        assert fused_graph_str == str(traced.graph)
        assert "fused_function" not in fused_graph_str
        return fused_graph_str


########## Basic fusion tests ##########

class AddReluSigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(torch.relu(x + 1))


class ExpChainModel(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(torch.exp(torch.relu(x)))


def test_add_relu_sigmoid():
    class AddReluSigmoid(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(torch.relu(x + 1))
    model = AddReluSigmoid()
    x = torch.randn(4, 4)
    fused_graph_str = _run_fusion_test(model, (x,), expect_fusion=True)
    assert "sigmoid" not in fused_graph_str, "sigmoid should NOT be present in the graph"
    assert "relu" not in fused_graph_str, "relu should NOT be present in the graph"
    assert "add" not in fused_graph_str, "add should NOT be present in the graph"


def test_exp_chain():
    class ExpChainModel(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(torch.exp(torch.relu(x)))
    model = ExpChainModel()
    x = torch.randn(4, 4)
    fused_graph_str = _run_fusion_test(model, (x,), expect_fusion=True)
    assert "sigmoid" not in fused_graph_str, "sigmoid should NOT be present in the graph"
    assert "exp" not in fused_graph_str, "exp should NOT be present in the graph"
    assert "relu" not in fused_graph_str, "relu should NOT be present in the graph"


@pytest.mark.parametrize("shape", [(1,), (2, 3), (2, 3, 4), (5, 5, 5, 5)])
def test_different_tensor_shapes(shape):
    class SimpleChain(torch.nn.Module):
        def forward(self, x):
            return torch.log(torch.exp(x))

    model = SimpleChain()
    x = torch.randn(shape)
    _run_fusion_test(model, (x,))


def test_method_style_operations():
    class MethodStyleOps(torch.nn.Module):
        def forward(self, x):
            return x.relu().sigmoid().tanh()

    model = MethodStyleOps()
    x = torch.randn(4, 4)
    fused_graph_str = _run_fusion_test(model, (x,), expect_fusion=True)

    assert "sigmoid" not in fused_graph_str, "sigmoid should NOT be present in the graph"
    assert "tanh" not in fused_graph_str, "tanh should NOT be present in the graph"
    assert "relu" not in fused_graph_str, "relu should NOT be present in the graph"


def test_mixed_operation_styles():
    class MixedStyleOps(torch.nn.Module):
        def forward(self, x):
            return torch.tanh(torch.relu(x).sigmoid())

    model = MixedStyleOps()
    x = torch.randn(4, 4)
    _run_fusion_test(model, (x,))


@pytest.mark.parametrize("op_pair, expect_fusion", [
    ((torch.relu, torch.sigmoid), True),
    ((torch.tanh, torch.exp), True),
    ((torch.log, torch.abs), False),
    ((torch.neg, torch.sin), False),
])
def test_various_unary_operations(op_pair, expect_fusion):
    op1, op2 = op_pair

    class UnaryOps(torch.nn.Module):
        def forward(self, x):
            return op2(op1(x))

    model = UnaryOps()
    x = torch.rand(3, 3) + 1e-6 if op1 == torch.log else torch.randn(3, 3)
    _run_fusion_test(model, (x,), expect_fusion=expect_fusion)


def test_binary_operations_with_constants():
    class BinaryOpsWithConstants(torch.nn.Module):
        def forward(self, x):
            a = x + 1
            b = a * 2
            c = b - 3
            return c

    model = BinaryOpsWithConstants()
    x = torch.randn(3, 3)
    fused_graph_str = _run_fusion_test(model, (x,), expect_fusion=True)
    assert "add" not in fused_graph_str, "add should NOT be present in the graph"
    assert "mul" not in fused_graph_str, "mul should NOT be present in the graph"
    assert "sub" not in fused_graph_str, "sub should NOT be present in the graph"


def test_tensor_constants():
    class TensorConstants(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('const1', torch.tensor([1.0, 2.0, 3.0]))
            self.register_buffer('const2', torch.tensor([0.1, 0.2, 0.3]))

        def forward(self, x):
            return (x + self.const1) * self.const2

    model = TensorConstants()
    x = torch.randn(3)
    fused_graph_str = _run_fusion_test(model, (x,), expect_fusion=True)

    assert "add" not in fused_graph_str, "add should NOT be present after fusion"
    assert "mul" not in fused_graph_str, "mul should NOT be present after fusion"

########## Edge cases ##########


def test_chain_with_multiple_uses():
    """Chain where an intermediate result is used twice -> incomplete fusion."""
    class MultipleUses(torch.nn.Module):
        def forward(self, x):
            a = torch.relu(x)
            b = torch.sigmoid(a)  # Part of chain
            c = torch.tanh(b)     # Part of chain

            # use of b again breaks complete fusion
            return c + b

    model = MultipleUses().eval()
    x = torch.randn(3, 3)
    fused_graph_str = _run_fusion_test(
        model, (x,), expect_fusion=True)  # partial fusion
    assert "tanh" in fused_graph_str, "tanh should be present in the graph"
    assert "sigmoid" not in fused_graph_str, "sigmoid should NOT be present in the graph"
    assert "relu" not in fused_graph_str, "relu should NOT be present in the graph"


def test_multiple_fusion_chains():
    """Multiple independent chains in the same model."""
    class MultipleChains(torch.nn.Module):
        def forward(self, x, y):
            b1 = torch.sigmoid(torch.relu(x))
            b2 = torch.log(torch.exp(y))
            return b1 + b2

    model = MultipleChains().eval()
    traced = symbolic_trace(model)
    original_node_count = len(list(traced.graph.nodes))

    fused = fuse_elementwise_chains(traced)
    fused_node_count = len(list(fused.graph.nodes))

    x, y = torch.randn(3, 3), torch.randn(3, 3)

    ref_out = traced(x, y)
    fused_out = fused(x, y)

    torch.testing.assert_close(ref_out, fused_out, rtol=1e-4, atol=1e-4)

    assert fused_node_count < original_node_count
    assert "call_function" in str(fused.graph)
    assert "fused_function" in str(fused.graph)


def test_no_fusion_non_elementwise():
    class NonElementwiseModel(torch.nn.Module):
        def forward(self, x):
            return torch.matmul(x, x)

    model = NonElementwiseModel()
    x = torch.randn(4, 4)
    _run_fusion_test(model, (x,), expect_fusion=False)
