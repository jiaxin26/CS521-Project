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
    else:
        assert fused_graph_str == str(traced.graph)


########## Basic fusion tests ##########

class AddReluSigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(torch.relu(x + 1))


class ExpChainModel(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(torch.exp(torch.relu(x)))


@pytest.mark.parametrize("model_cls", [
    AddReluSigmoid,
    ExpChainModel,
])
def test_fuse_elementwise_chains(model_cls):
    model = model_cls()
    x = torch.randn(4, 4)

    _run_fusion_test(model, (x,))


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
    _run_fusion_test(model, (x,))


def test_mixed_operation_styles():
    class MixedStyleOps(torch.nn.Module):
        def forward(self, x):
            return torch.tanh(torch.relu(x).sigmoid())

    model = MixedStyleOps()
    x = torch.randn(4, 4)
    _run_fusion_test(model, (x,))


@pytest.mark.parametrize("op_pair", [
    (torch.relu, torch.sigmoid),
    (torch.tanh, torch.exp),
    (torch.log, torch.abs),
    (torch.neg, torch.sin),
])
def test_various_unary_operations(op_pair):
    op1, op2 = op_pair

    class UnaryOps(torch.nn.Module):
        def forward(self, x):
            return op2(op1(x))

    model = UnaryOps()
    x = torch.rand(3, 3) + 1e-6 if op1 == torch.log else torch.randn(3, 3)
    _run_fusion_test(model, (x,))


def test_binary_operations_with_constants():
    class BinaryOpsWithConstants(torch.nn.Module):
        def forward(self, x):
            a = x + 1
            b = a * 2
            c = b - 3
            return c

    model = BinaryOpsWithConstants()
    x = torch.randn(3, 3)
    _run_fusion_test(model, (x,))


def test_tensor_constants():
    const1 = torch.tensor([1.0, 2.0, 3.0])
    const2 = torch.tensor([0.1, 0.2, 0.3])

    class TensorConstants(torch.nn.Module):
        def forward(self, x):
            return (x + const1) * const2

    model = TensorConstants()
    x = torch.randn(3)
    _run_fusion_test(model, (x,))

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
    _run_fusion_test(model, (x,))


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


def test_no_fusion_non_elementwise():
    class NonElementwiseModel(torch.nn.Module):
        def forward(self, x):
            return torch.matmul(x, x)

    model = NonElementwiseModel()
    x = torch.randn(4, 4)
    _run_fusion_test(model, (x,), expect_fusion=False)
