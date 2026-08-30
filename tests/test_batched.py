"""Batched MoE correctness: batched_fused_moe(B tokens) vs loop of single-token fused_moe."""

import sys
import torch

from warp_route.functional import fused_moe, batched_fused_moe, _pack_expert_weights


def make_inputs(num_experts, hidden_dim, intermediate_dim, batch_size, device, seed=7):
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


def run_correctness(num_experts, hidden_dim, intermediate_dim, top_k, batch_size, seed=42):
    device = torch.device("cuda")
    hidden_batch, gate_w, weight_buf = make_inputs(
        num_experts, hidden_dim, intermediate_dim, batch_size, device, seed
    )

    # Reference: loop of single-token fused_moe calls
    ref_outputs = []
    for i in range(batch_size):
        out = fused_moe(
            hidden_batch[i].contiguous(), gate_w, weight_buf,
            num_experts, hidden_dim, intermediate_dim, top_k
        )
        ref_outputs.append(out)
    ref = torch.stack(ref_outputs, dim=0)  # [B, hidden]

    # Batched kernel
    batched = batched_fused_moe(
        hidden_batch, gate_w, weight_buf,
        num_experts, hidden_dim, intermediate_dim, top_k
    )

    # Per-token cosine similarity and relative MAE
    all_pass = True
    for i in range(batch_size):
        a = ref[i]
        b = batched[i]
        cs = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        denom = a.abs().max().clamp(min=1e-9).item()
        rmae = float((a - b).abs().mean().item() / denom)
        ok = cs > 0.999 and rmae < 0.01
        if not ok:
            print(f"    token {i}: cos={cs:.6f}  rel_mae={rmae:.3e}  FAIL")
            all_pass = False

    label = "PASS" if all_pass else "FAIL"
    print(f"  experts={num_experts:<3d} hidden={hidden_dim:<4d} inter={intermediate_dim:<5d} "
          f"top_k={top_k} batch={batch_size:<3d}  {label}")
    return all_pass


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    print("Batched correctness tests: batched_fused_moe vs loop of fused_moe\n")

    # (num_experts, hidden_dim, intermediate_dim, top_k, batch_size)
    configs = []
    for batch_size in (1, 4, 16, 64):
        configs.append((8, 256, 512, 1, batch_size))
        configs.append((8, 256, 512, 2, batch_size))
        configs.append((16, 512, 1024, 2, batch_size))

    pass_count = 0
    total = len(configs)
    for cfg in configs:
        ok = run_correctness(*cfg)
        pass_count += int(ok)

    print(f"\n{pass_count}/{total} tests passed")
    sys.exit(0 if pass_count == total else 1)


if __name__ == "__main__":
    main()
