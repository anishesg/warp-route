"""Python latency benchmark: fused vs naive MoE across Mixtral, DeepSeek-V2, DBRX configs."""

import sys
import torch

from warp_route.functional import fused_moe, naive_moe, _pack_expert_weights


def make_inputs(num_experts, hidden_dim, intermediate_dim, device, seed=0):
    torch.manual_seed(seed)
    hidden   = torch.randn(hidden_dim, device=device, dtype=torch.float32) * 0.1
    gate_w   = torch.randn(num_experts, hidden_dim, device=device, dtype=torch.float32) * 0.02
    gate_proj = [torch.randn(intermediate_dim, hidden_dim, device=device, dtype=torch.float32) * 0.01
                 for _ in range(num_experts)]
    up_proj   = [torch.randn(intermediate_dim, hidden_dim, device=device, dtype=torch.float32) * 0.01
                 for _ in range(num_experts)]
    down_proj = [torch.randn(hidden_dim, intermediate_dim, device=device, dtype=torch.float32) * 0.01
                 for _ in range(num_experts)]
    weight_buf = _pack_expert_weights(
        gate_proj, up_proj, down_proj,
        num_experts, hidden_dim, intermediate_dim
    )
    return hidden, gate_w, weight_buf


def bench_single(name, num_experts, hidden_dim, intermediate_dim, top_k,
                 warmup=20, iters=200):
    device = torch.device("cuda")
    hidden, gate_w, weight_buf = make_inputs(num_experts, hidden_dim, intermediate_dim, device)

    start_ev = torch.cuda.Event(enable_timing=True)
    stop_ev  = torch.cuda.Event(enable_timing=True)

    # Naive warmup + bench
    for _ in range(warmup):
        naive_moe(hidden, gate_w, weight_buf, num_experts, hidden_dim, intermediate_dim, top_k)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(iters):
        naive_moe(hidden, gate_w, weight_buf, num_experts, hidden_dim, intermediate_dim, top_k)
    stop_ev.record()
    torch.cuda.synchronize()
    naive_us = start_ev.elapsed_time(stop_ev) / iters * 1000.0

    # Fused warmup + bench
    for _ in range(warmup):
        fused_moe(hidden, gate_w, weight_buf, num_experts, hidden_dim, intermediate_dim, top_k)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(iters):
        fused_moe(hidden, gate_w, weight_buf, num_experts, hidden_dim, intermediate_dim, top_k)
    stop_ev.record()
    torch.cuda.synchronize()
    fused_us = start_ev.elapsed_time(stop_ev) / iters * 1000.0

    speedup = naive_us / fused_us if fused_us > 0 else float("inf")

    print(f"  {name:<20s}  experts={num_experts:<3d}  hidden={hidden_dim:<5d}  "
          f"inter={intermediate_dim:<6d}  top_k={top_k}")
    print(f"    naive:   {naive_us:8.1f} us")
    print(f"    fused:   {fused_us:8.1f} us")
    print(f"    speedup: {speedup:.2f}x\n")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        sys.exit(0)

    prop = torch.cuda.get_device_properties(0)
    print(f"Python latency benchmark\nGPU: {prop.name} (sm_{prop.major}{prop.minor})\n")

    configs = [
        # (name, num_experts, hidden_dim, intermediate_dim, top_k)
        ("mixtral-8x7b",   8,  4096, 14336, 2),
        ("mixtral-8x7b",   8,  4096, 14336, 1),
        ("deepseek-v2",   64,  7168, 18432, 2),
        ("deepseek-v2",   64,  7168, 18432, 1),
        ("dbrx",          16,  6144, 10752, 2),
        ("dbrx",          16,  6144, 10752, 1),
    ]

    print("=== Fused vs Naive latency (us per decode token) ===\n")
    for cfg in configs:
        name, ne, hd, inter, tk = cfg
        bench_single(name, ne, hd, inter, tk)


if __name__ == "__main__":
    main()
