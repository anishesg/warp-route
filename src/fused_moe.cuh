#pragma once
#include <cuda_runtime.h>
#include "expert_layout.cuh"

// Fused MoE decode kernel: computes gating, expert FFN, and weighted output
// accumulation in a single kernel launch with no permutation buffers.
//
// top_k must be 1 or 2.
//
// top-1: single expert FFN computed in registers, output written once.
// top-2: both experts processed sequentially in the same warp; gate weights
//        are applied during accumulation so outputs are summed before
//        the final global memory write, avoiding two separate passes.
//
// For top_k > 2 use the naive reference.

struct FusedMoEParams {
    const float* gate_matrix;   // [num_experts x hidden_dim] router weights
    const float* weight_buf;    // packed expert weights (see expert_layout.cuh)
    ExpertLayoutConfig layout;
    int top_k;                  // 1 or 2
};

// Launch the fused MoE kernel.
// One warp per token; for batch_size=1 (decode), one warp total.
// hidden: [hidden_dim] in device memory
// output: [hidden_dim] in device memory
void fused_moe_forward(
    const FusedMoEParams& params,
    const float* d_hidden,
    float*       d_output,
    cudaStream_t stream = 0);
