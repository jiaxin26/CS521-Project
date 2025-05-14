import time
from typing import Dict
import torch
import numpy as np
from torch.fx import symbolic_trace
from evaluate_flops import fx_count_flops
from torch import nn
import torch.fx as fx


def profile_model(
    model: nn.Module,
    dummy_input: torch.Tensor,  # Take pre-generated input tensor
    num_runs: int = 8,
    warmup: int = 3
) -> Dict[str, float]:
    device = dummy_input.device
    model = model.to(device).eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize() if device.type == "cuda" else None

    # Latency measurement
    times = []
    for _ in range(num_runs + 3):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.time()
            _ = model(dummy_input)
            times.append((time.time() - t0) * 1000)

    # Process times
    times = sorted(times)
    cut = int(len(times) * 0.1)
    avg_time = np.mean(times[cut:-cut])

    # Memory measurement
    peak_mem = 0.0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
            torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

    # FLOP counting
    total_flops = 0
    try:
        gm = symbolic_trace(model)
        fx.passes.shape_prop.ShapeProp(gm).propagate(dummy_input)
        total_flops = fx_count_flops(gm, (dummy_input,)) / 1e9
    except Exception as e:
        print(f"FLOP counting failed: {str(e)}")

    return {
        'avg_time_ms': avg_time,
        'peak_mem_mb': peak_mem,
        'total_flops_g': total_flops
    }
