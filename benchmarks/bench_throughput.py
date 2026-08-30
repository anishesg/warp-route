"""Throughput scaling benchmark: tokens/second vs batch size for batched fused MoE."""

import sys
import torch

from warp_route.functional import fused_moe, batched_fused_moe, _pack_expert_weights


def make_inputs(num_experts, hidden_dim, intermediate_dim, batch_size, device, seed=3):
    torch.manual_seed(seed)
    hidden_batch = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float32) * 0.1
    gate_w = torch.randn(num_experts, hidden_dim, device=device, dtype=torch.float32) * 0.02
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
    return hidden_batch, gate_w, weight_buf


def bench_batched(num_experts, hidden_dim, intermediate_dim, top_k, batch_size,
                  warmup=20, iters=200):
    device = torch.device("cuda")
    hidden_batch, gate_w, weight_buf = make_inputs(
        num_experts, hidden_dim, intermediate_dim, batch_size, device
    )

    start_ev = torch.cuda.Event(enable_timing=True)
    stop_ev  = torch.cuda.Event(enable_timing=True)

    # Batched kernel
    for _ in range(warmup):
        batched_fused_moe(hidden_batch, gate_w, weight_buf,
                          num_experts, hidden_dim, intermediate_dim, top_k)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(iters):
        batched_fused_moe(hidden_batch, gate_w, weight_buf,
                          num_experts, hidden_dim, intermediate_dim, top_k)
    stop_ev.record()
    torch.cuda.synchronize()
    batched_us = start_ev.elapsed_time(stop_ev) / iters * 1000.0

    # Loop of single-token calls
    single_hidden = hidden_batch[0].contiguous()
    for _ in range(warmup):
        for _ in range(batch_size):
            fused_moe(single_hidden, gate_w, weight_buf,
                      num_experts, hidden_dim, intermediate_dim, top_k)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(iters):
        for _ in range(batch_size):
            fused_moe(single_hidden, gate_w, weight_buf,
                      num_experts, hidden_dim, intermediate_dim, top_k)
    stop_ev.record()
    torch.cuda.synchronize()
    loop_us = start_ev.elapsed_time(stop_ev) / iters * 1000.0

    batched_tok_per_s  = batch_size / (batched_us * 1e-6)
    loop_tok_per_s     = batch_size / (loop_us   * 1e-6)
    per_tok_batched_us = batched_us / batch_size
    per_tok_loop_us    = loop_us    / batch_size
    speedup = loop_us / batched_us if batched_us > 0 else float("inf")

    return {
        "batch_size": batch_size,
        "batched_us": batched_us,
        "loop_us": loop_us,
        "batched_tok_per_s": batched_tok_per_s,
        "loop_tok_per_s": loop_tok_per_s,
        "per_tok_batched_us": per_tok_batched_us,
        "per_tok_loop_us": per_tok_loop_us,
        "speedup": speedup,
    }


def print_table(results):
    print(f"  {'batch':>5}  {'batched(us)':>11}  {'loop(us)':>9}  "
          f"{'batched tok/s':>13}  {'loop tok/s':>10}  "
          f"{'us/tok(bat)':>11}  {'us/tok(loop)':>12}  {'speedup':>7}")
    print("  " + "-" * 105)
    for r in results:
        print(f"  {r['batch_size']:>5}  {r['batched_us']:>11.1f}  {r['loop_us']:>9.1f}  "
              f"  {r['batched_tok_per_s']:>11.0f}  {r['loop_tok_per_s']:>10.0f}  "
              f"  {r['per_tok_batched_us']:>9.2f}  {r['per_tok_loop_us']:>10.2f}  {r['speedup']:>7.2f}x")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        sys.exit(0)

    prop = torch.cuda.get_device_properties(0)
    print(f"Throughput scaling benchmark\nGPU: {prop.name} (sm_{prop.major}{prop.minor})\n")

    configs = [
        ("mixtral-8x7b", 8, 4096, 14336, 2),
        ("dbrx",        16, 6144, 10752, 2),
    ]

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    for name, ne, hd, inter, tk in configs:
        print(f"=== {name}  experts={ne}  hidden={hd}  inter={inter}  top_k={tk} ===")
        results = []
        for bs in batch_sizes:
            r = bench_batched(ne, hd, inter, tk, bs)
            results.append(r)
        print_table(results)
        print()


if __name__ == "__main__":
    main()
