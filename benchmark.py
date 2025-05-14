import torch
import pandas as pd
from torch.fx import symbolic_trace
from evaluation_models import (
    EnhancedResNet,
    EnhancedVGG,
    DeepElementwiseModel,
    BitShiftTestModel,
    RewriteTriggerResNet,
    RewriteTriggerResNetWrapped,
    ComprehensiveRewriteModel,
)
from bitshift_swap import swap_bitshift_reducesum
from recip_associative_swap import swap_recip_associative
from distributive1 import DistributiveRulePass
from distributive2 import DistributiveRule2Pass
from associative2 import SqrtAssociativePass
from commutative2 import CommutativePass
from evaluate_profile_model import profile_model
from main import optimize_graph


def evaluate_all_models_with_breakdown():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = []

    model_configs = [
        {
            "name": "RewriteResNet",
            "model": RewriteTriggerResNetWrapped(RewriteTriggerResNet()).to(device),
            "make_input": lambda: torch.cat([
                torch.randn(32, 3, 224, 224, device=device),
                torch.randint(0, 16, (32, 1, 1, 16384), device=device).float()
                    .expand(-1, 3, 224, -1)
            ], dim=3),
            "passes": [
                ("Commutative",        CommutativePass),
                ("SqrtAssociative",    SqrtAssociativePass),
                ("BitShift→Sum",       swap_bitshift_reducesum),
                ("Distributive",       DistributiveRulePass),
                ("Distributive2",      DistributiveRule2Pass),
                ("Recip‑Assoc",        swap_recip_associative),
            ]
        },
        {
            "name": "ComprehensiveRewrite",
            "model": ComprehensiveRewriteModel().to(device),
            "make_input": lambda: torch.randn(32, 3, 224, 224, device=device),
            "passes": [
                ("Commutative",        CommutativePass),
                ("SqrtAssociative",    SqrtAssociativePass),
                ("BitShift→Sum",       swap_bitshift_reducesum),
                ("Distributive",       DistributiveRulePass),
                ("Distributive2",      DistributiveRule2Pass),
                ("Recip‑Assoc",        swap_recip_associative),
            ]
        },
        # summary‑only models
        {
            "name": "EnhancedResNet",
            "model": EnhancedResNet().to(device),
            "make_input": lambda: torch.randn(32, 3, 224, 224, device=device),
            "passes": None
        },
        {
            "name": "EnhancedVGG",
            "model": EnhancedVGG().to(device),
            "make_input": lambda: torch.randn(32, 3, 224, 224, device=device),
            "passes": None
        },
        {
            "name": "DeepElementwise",
            "model": DeepElementwiseModel().to(device),
            "make_input": lambda: torch.randn(32, 3, 224, 224, device=device),
            "passes": None
        },
        {
            "name": "BitShiftTest",
            "model": BitShiftTestModel().to(device),
            "make_input": lambda: torch.randint(0, 16, (32, 3, 224, 224),
                                                device=device, dtype=torch.int32),
            "passes": None
        },
    ]

    for cfg in model_configs:
        name = cfg["name"]
        model = cfg["model"].eval()
        dummy = cfg["make_input"]()

        # 1) baseline
        gm0 = symbolic_trace(model)
        base_stat = profile_model(gm0, dummy)

        # 2) global optimize_graph
        gm1 = optimize_graph(symbolic_trace(model))
        opt_stat = profile_model(gm1, dummy)

        if cfg["passes"] is None:
            # just gather for final summary
            Δf = 100 * (base_stat["total_flops_g"] -
                        opt_stat["total_flops_g"]) / base_stat["total_flops_g"]
            Δt = 100 * (base_stat["avg_time_ms"] -
                        opt_stat["avg_time_ms"]) / base_stat["avg_time_ms"]
            Δm = 100 * (base_stat["peak_mem_mb"] -
                        opt_stat["peak_mem_mb"]) / base_stat["peak_mem_mb"]
            summary.append({
                "Model":        name,
                "Baseline FLOPs": base_stat["total_flops_g"],
                "Opt FLOPs":      opt_stat["total_flops_g"],
                "ΔFLOPs %":       f"{Δf:+.1f}%",
                "Baseline ms":    base_stat["avg_time_ms"],
                "Opt ms":         opt_stat["avg_time_ms"],
                "ΔLat %":         f"{Δt:+.1f}%",
                "Baseline MB":    base_stat["peak_mem_mb"],
                "Opt MB":         opt_stat["peak_mem_mb"],
                "ΔMem %":         f"{Δm:+.1f}%"
            })

        else:
            # detailed per‑pass breakdown
            print(f"\n--- {name} ---")
            print(f"Baseline  →  FLOPs {base_stat['total_flops_g']:.2f}G   "
                  f"Latency {base_stat['avg_time_ms']:.2f}ms   "
                  f"Mem {base_stat['peak_mem_mb']:.1f}MB\n")

            for pass_name, Pass in cfg["passes"]:
                gm_p = symbolic_trace(model)
                gm_opt = Pass()(gm_p) if isinstance(Pass, type) else Pass(gm_p)
                pstat = profile_model(gm_opt, dummy)

                Δf_p = 100 * \
                    (base_stat["total_flops_g"] - pstat["total_flops_g"]
                     ) / base_stat["total_flops_g"]
                Δt_p = 100 * \
                    (base_stat["avg_time_ms"] - pstat["avg_time_ms"]
                     ) / base_stat["avg_time_ms"]
                Δm_p = 100 * \
                    (base_stat["peak_mem_mb"] - pstat["peak_mem_mb"]
                     ) / base_stat["peak_mem_mb"]

                print(f"{pass_name:<18}"
                      f" FLOPs {pstat['total_flops_g']:.2f}G ({Δf_p:+.1f}%)  "
                      f"Time {pstat['avg_time_ms']:.2f}ms ({Δt_p:+.1f}%)  "
                      f"Mem {pstat['peak_mem_mb']:.1f}MB ({Δm_p:+.1f}%)")

    # 3) finally print the summary for the rest
    if summary:
        df = pd.DataFrame(summary).set_index("Model")
        print("\n=== Summary for the other models ===")
        display(df.round(2))


if __name__ == "__main__":
    evaluate_all_models_with_breakdown()
