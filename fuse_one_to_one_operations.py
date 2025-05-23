import torch
import torch.nn as nn
from torch.fx import symbolic_trace, GraphModule
import operator

ELEMENTWISE_OPS = {
    torch.add, torch.sub, torch.mul, torch.div,
    torch.relu, torch.sigmoid, torch.tanh,
    torch.neg, torch.exp, torch.log,
    operator.add, operator.sub, operator.mul, operator.truediv
}

METHOD_OPS = {"relu", "sigmoid", "tanh", "add", "mul", "div", "sub"}


def is_elementwise(node):
    if getattr(node.target, "__name__", None) == "fused_function":
        return False
    return (node.op == "call_function" and node.target in ELEMENTWISE_OPS) or \
           (node.op == "call_method" and node.target in METHOD_OPS)


def find_elementwise_chains(graph):
    chains = []
    visited = set()

    for node in graph.nodes:
        if node in visited:
            continue
        if not is_elementwise(node):
            continue

        chain = [node]
        visited.add(node)
        current = node

        while True:
            users = list(current.users)
            if len(users) != 1:
                break

            next_node = users[0]
            if not is_elementwise(next_node) or next_node in visited:
                break

            only_depends_on_current = all(
                not (isinstance(arg, torch.fx.Node) and arg.op !=
                     "get_attr" and arg is not current)
                for arg in next_node.args
            )

            if not only_depends_on_current:
                break

            chain.append(next_node)
            visited.add(next_node)
            current = next_node

        if len(chain) >= 2:
            chains.append(chain)
    return chains


def generate_fused_fn(chain, gm):
    # Pre-capture tensor constants
    constants = {}

    # Check if this chain is safe to fuse
    for i, node in enumerate(chain):
        if node.op == "call_function" and len(node.args) >= 2:
            second_arg = node.args[1]
            if isinstance(second_arg, torch.fx.Node) and second_arg.op == "get_attr":
                constants[(i, 1)] = getattr(gm, second_arg.target)
            elif not isinstance(second_arg, torch.fx.Node):
                constants[(i, 1)] = second_arg
            else:
                return None  # No fusion if second arg is another variable node

    def fused_function(x):
        result = x
        for i, node in enumerate(chain):
            if node.op == "call_function":
                if len(node.args) == 1:  # unary op
                    result = node.target(result)
                elif len(node.args) >= 2:  # binary op
                    if (i, 1) in constants:
                        result = node.target(result, constants[(i, 1)])
                    else:
                        second_arg = node.args[1]
                        result = node.target(result, second_arg)

            elif node.op == "call_method":
                method = getattr(result, node.target)
                if len(node.args) == 0:
                    result = method()
                else:
                    method_args = node.args[1:]
                    result = method(*method_args)

        return result

    fused_function.__name__ = "fused_function"
    fused_function.flops_per_element = len(chain)
    fused_function.is_fused_function = True
    return fused_function


def fuse_elementwise_chains(gm: GraphModule):
    graph = gm.graph
    chains = find_elementwise_chains(graph)

    for chain in chains:
        # If the head node is no longer in the graph (erased by a prior fusion), skip
        if chain[0] not in graph.nodes:
            continue

        first = chain[0]
        last = chain[-1]
        input_val = first.args[0]

        fused_fn = generate_fused_fn(chain, gm)
        if fused_fn is None:
            continue

        with graph.inserting_before(first):
            fused = graph.call_function(fused_fn, args=(input_val,))

        last.replace_all_uses_with(fused)

        for node in reversed(chain):
            graph.erase_node(node)

    gm.recompile()
    return gm


class TestModel(nn.Module):
    def forward(self, x):
        return torch.sigmoid(torch.relu(torch.add(x, 1)))


if __name__ == "__main__":
    model = TestModel()
    traced = symbolic_trace(model)

    print("=== Before Fusion ===")
    print(traced.graph)

    optimized = fuse_elementwise_chains(traced)

    print("=== After Fusion ===")
    print(optimized.graph)

    x = torch.randn(3, 4)
    ref_out = model(x)
    opt_out = optimized(x)

    print("Ref Output:", ref_out)
    print("Fused Output:", opt_out)

    assert torch.allclose(ref_out, opt_out, rtol=1e-4), "Mismatch in outputs"

    compiled = torch.compile(optimized)
    compiled_out = compiled(x)
    print("Compiled Output:", compiled_out)
    assert torch.allclose(compiled_out, ref_out, rtol=1e-4)
