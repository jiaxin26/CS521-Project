import torch
import pytest
from torch.fx import symbolic_trace
from bitshift_swap import swap_bitshift_reducesum


@pytest.mark.parametrize("shift_fn,shape,shift,sum_style", [
    (torch.bitwise_left_shift, (4, 4), 1, "method"),
    (torch.bitwise_left_shift, (2, 3), 5, "function"),
    (torch.bitwise_left_shift, (4, 4), 0, "method"),      # shift=0 edge case
    (torch.bitwise_left_shift, (0, 4), 2, "function"),     # empty input
    (torch.bitwise_left_shift, (100, 100), 1, "method"),   # large tensor
])
def test_swap_bitshift_sum(shift_fn, shape, shift, sum_style):
    class M(torch.nn.Module):
        def forward(self, x):
            shifted = shift_fn(x, shift)
            return shifted.sum() if sum_style == "method" else torch.sum(shifted)

    torch.manual_seed(0)
    x = torch.randint(0, 16, shape, dtype=torch.int32)

    model = M().eval()
    traced = symbolic_trace(model)
    fused = swap_bitshift_reducesum(traced)
    compiled_fused = torch.compile(fused, backend='inductor')

    # Check that values are equal
    out1 = model(x)
    out2 = fused(x)
    out3 = compiled_fused(x)

    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out1, out3)

    # Check that rewrite was applied
    assert "bitwise" in str(
        fused.graph), "BitShift should appear in rewritten graph"
    assert "sum" in str(fused.graph), "Sum should appear in rewritten graph"


def test_no_rewrite_when_no_bitshift():
    class NoRewrite(torch.nn.Module):
        def forward(self, x):
            return x.sum()

    x = torch.randint(0, 10, (4, 4), dtype=torch.int32)
    model = NoRewrite().eval()
    traced = symbolic_trace(model)
    fused = swap_bitshift_reducesum(traced)

    # Outputs match
    out1 = model(x)
    out2 = fused(x)
    torch.testing.assert_close(out1, out2)

    # Graph should be unchanged
    assert str(fused.graph) == str(
        traced.graph), "Graph should not be rewritten"
