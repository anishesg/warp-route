# warp-route

Fused decode-time MoE dispatch: warp-cooperative gating, expert selection, and FFN in a single kernel launch.

## Problem

During autoregressive decode with Mixture-of-Experts models, each forward pass through an MoE layer involves a sequence of kernel launches:

1. Router matmul (hidden_dim x num_experts)
2. Softmax over expert logits
3. Top-K selection and weight computation
4. Token permutation (scatter to expert-ordered buffer)
5. N expert FFN launches (gate_proj, up_proj, silu, down_proj per expert)
6. Token unpermutation (gather back to sequence order)
7. Weighted summation of expert outputs

At decode time with batch_size=1, each of these launches incurs ~3-10us of kernel launch overhead, plus permutation steps move data through global memory purely for bookkeeping. With 32 MoE layers (Mixtral 8x7B), this overhead compounds to 1-2ms of pure launch and permutation cost on top of the actual FFN compute.

For Mixtral 8x7B (hidden=4096, intermediate=14336, num_experts=8, top_k=2), measured on A100 SXM:
- Naive multi-kernel: ~168us per MoE layer
- Launch and permutation overhead: ~35-60us (20-35% of total)
- Target: eliminate permutation entirely, reduce launches from 6+ to 1

For DeepSeek-MoE (hidden=2048, intermediate=1408, num_experts=64, top_k=6):
- Higher expert count magnifies per-expert launch overhead
- 64 small FFNs vs 8 larger ones: more launches, smaller compute per launch, worse utilization

## Approach

**Warp-cooperative expert routing**: each warp handles a single token. All 32 threads collaboratively compute gate scores using warp-level reductions (warp shuffle instructions), select top-K experts via `__ballot_sync` and register-level sorting, with no shared memory needed for the gating phase.

**In-register weight streaming**: after expert selection, the warp loads the selected expert's weight tiles from global memory into shared memory cooperatively. Each thread holds a portion of the accumulator in registers. This eliminates the permute/unpermute buffers entirely since no tokens are moved, only weights are accessed by index.

**SwiGLU fusion**: gate_proj and up_proj are loaded in interleaved tiles so the SiLU activation and element-wise multiply happen immediately on loaded data, cutting the working set in shared memory.

**Single output write**: the weighted combination of top-K expert outputs accumulates in registers across sequential expert passes, and the final result is written once to global memory.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requires CUDA 11.8+, sm_80 or newer (A100, A10, H100).

## Target configurations

| Model | num_experts | top_k | hidden_dim | intermediate_dim |
|-------|------------|-------|------------|-----------------|
| Mixtral 8x7B | 8 | 2 | 4096 | 14336 |
| DeepSeek-MoE | 64 | 6 | 2048 | 1408 |
| DBRX | 16 | 4 | 6144 | 10752 |

## Structure

```
src/
  expert_layout.cuh   weight tensor layout and device accessors
  gate.cuh / gate.cu  warp-cooperative gating and top-K selection
  tiled_matvec.cuh    tiled shared-memory matrix-vector product primitive
  naive_moe.cuh / .cu multi-kernel reference implementation
  fused_moe.cuh / .cu single-kernel fused dispatch
tests/
  test_correctness.cu fused vs naive comparison across configurations
benchmarks/
  bench_latency.cu    per-kernel timing breakdown and speedup ratios
```
