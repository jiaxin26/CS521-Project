import torch
from torch.fx import symbolic_trace
from recip_associative_swap import swap_recip_associative


class RecipAssociativeModel(torch.nn.Module):
    def forward(self, A, B):
        # Implements: Recip(A) ⊙ Recip(A ⊙ B)
        recip_a = torch.reciprocal(A)  # 1/A
        a_mul_b = A * B                # A*B
        recip_a_mul_b = torch.reciprocal(a_mul_b)  # 1/(A*B)
        return recip_a * recip_a_mul_b  # (1/A) * (1/(A*B))


gm = symbolic_trace(RecipAssociativeModel().eval())

print("=== Original Graph ===")
print(gm.graph)

optimized = swap_recip_associative(gm)
print("=== Rewritten Graph ===")
print(optimized.graph)

compiled_gm = torch.compile(optimized, backend='inductor')

A = torch.tensor([2.0, 4.0, 5.0])
B = torch.tensor([3.0, 5.0, 7.0])

original_output = gm(A, B)
optimized_output = optimized(A, B)
compiled_output = compiled_gm(A, B)

print("Original output:", original_output)
print("Optimized output:", optimized_output)
print("Compiled output:", compiled_output)

assert torch.allclose(original_output, optimized_output,
                      rtol=1e-5), "Outputs do not match!"
assert torch.allclose(original_output, compiled_output,
                      rtol=1e-5), "Outputs do not match!"
