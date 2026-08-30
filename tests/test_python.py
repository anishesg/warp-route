"""Python correctness tests: fused_moe vs naive_moe vs a pure-torch reference."""

import math
import sys
import torch
import torch.nn.functional as F

from warp_route.functional import fused_moe, naive_moe, _pack_expert_weights


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def torch_moe_reference(
    hidden: torch.Tensor,
    gate_w: torch.Tensor,
    gate_proj: list,
    up_proj: list,
    down_proj: list,
    top_k: int,
) -> torch.Tensor:
    """Pure-torch MoE: manual softmax, topk, and SwiGLU FFN per selected expert."""
    logits = gate_w @ hidden                       # [num_experts]
    probs = torch.softmax(logits, dim=0)
    weights, indices = torch.topk(probs, top_k)

    out = torch.zeros_like(hidden)
    for ki in range(top_k):
        e = indices[ki].item()
        w = weights[ki].item()
        g = gate_proj[e] @ hidden                  # [intermediate]
        u = up_proj[e]   @ hidden                  # [intermediate]
        act = silu(g) * u
        expert_out = down_proj[e] @ act            # [hidden]
        out += w * expert_out
    return out


def make_expert_tensors(num_experts, hidden_dim, intermediate_dim, device, dtype, rng_seed):
    torch.manual_seed(rng_seed)
    gate_proj = [torch.randn(intermediate_dim, hidden_dim, device=device, dtype=dtype) * 0.01
                 for _ in range(num_experts)]
    up_proj   = [torch.randn(intermediate_dim, hidden_dim, device=device, dtype=dtype) * 0.01
                 for _ in range(num_experts)]
    down_proj = [torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype) * 0.01
                 for _ in range(num_experts)]
    return gate_proj, up_proj, down_proj


def run_test(num_experts, hidden_dim, intermediate_dim, top_k, seed=42):
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(seed)
    hidden = torch.randn(hidden_dim, device=device, dtype=dtype) * 0.1
    gate_w = torch.randn(num_experts, hidden_dim, device=device, dtype=dtype) * 0.02

    gate_proj, up_proj, down_proj = make_expert_tensors(
        num_experts, hidden_dim, intermediate_dim, device, dtype, seed + 1
    )

    weight_buf = _pack_expert_weights(
        gate_proj, up_proj, down_proj,
        num_experts, hidden_dim, intermediate_dim
    )

    # Reference: pure torch
    ref_out = torch_moe_reference(hidden, gate_w, gate_proj, up_proj, down_proj, top_k)

    # Naive kernel
    naive_out = naive_moe(hidden, gate_w, weight_buf, num_experts, hidden_dim, intermediate_dim, top_k)

    # Fused kernel (only supports top_k <= 2)
    if top_k <= 2:
        fused_out = fused_moe(hidden, gate_w, weight_buf, num_experts, hidden_dim, intermediate_dim, top_k)
    else:
        fused_out = None

    def cosine_sim(a, b):
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    def rel_mae(a, b):
        denom = a.abs().max().clamp(min=1e-9)
        return float((a - b).abs().mean().item() / denom)

    # naive vs torch reference
    cs_naive = cosine_sim(naive_out, ref_out)
    rmae_naive = rel_mae(naive_out, ref_out)

    ok_naive = cs_naive > 0.999 and rmae_naive < 0.01
    label_naive = "PASS" if ok_naive else "FAIL"

    print(f"  experts={num_experts:<3d} hidden={hidden_dim:<5d} inter={intermediate_dim:<6d} top_k={top_k}"
          f"  naive_vs_torch: cos={cs_naive:.6f} rel_mae={rmae_naive:.3e}  {label_naive}")

    ok_fused = True
    if fused_out is not None:
        cs_fused = cosine_sim(fused_out, ref_out)
        rmae_fused = rel_mae(fused_out, ref_out)
        ok_fused = cs_fused > 0.999 and rmae_fused < 0.01
        label_fused = "PASS" if ok_fused else "FAIL"
        print(f"  experts={num_experts:<3d} hidden={hidden_dim:<5d} inter={intermediate_dim:<6d} top_k={top_k}"
              f"  fused_vs_torch: cos={cs_fused:.6f} rel_mae={rmae_fused:.3e}  {label_fused}")

    return ok_naive and ok_fused


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    print("Python correctness tests: fused / naive vs torch reference\n")

    configs = [
        # (num_experts, hidden_dim, intermediate_dim, top_k)
        (8,  256,  512, 1),
        (8,  256,  512, 2),
        (8,  512, 1024, 1),
        (8,  512, 1024, 2),
        (16, 256,  512, 1),
        (16, 256,  512, 2),
        (8, 1024, 2048, 2),
    ]

    pass_count = 0
    total = len(configs)
    for cfg in configs:
        ok = run_test(*cfg)
        pass_count += int(ok)

    print(f"\n{pass_count}/{total} tests passed")
    sys.exit(0 if pass_count == total else 1)


if __name__ == "__main__":
    main()
