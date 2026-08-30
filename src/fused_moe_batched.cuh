#pragma once
#include <cuda_runtime.h>
#include "expert_layout.cuh"

// Batched fused MoE decode kernel: processes B tokens in one kernel launch.
//
// Each warp-block handles one token independently:
//   - Computes router logits against router weight matrix (cached in shared memory)
//   - Applies softmax, selects top-k experts
//   - Runs SwiGLU FFN for each selected expert with warp-cooperative matvec
//   - Writes weighted-summed output to output[token_idx * hidden_dim]
//
// Grid:  (batch_size, 1)
// Block: 32 threads (one warp per token)
//
// top_k must be 1 or 2.

struct BatchedFusedMoEParams {
    const float* gate_matrix;   // [num_experts x hidden_dim] router weights
    const float* weight_buf;    // packed expert weights (see expert_layout.cuh)
    ExpertLayoutConfig layout;
    int top_k;                  // 1 or 2
    int batch_size;
};

// Launch the batched fused MoE kernel.
// d_hidden: [batch_size x hidden_dim] in device memory (row-major)
// d_output: [batch_size x hidden_dim] in device memory (row-major)
void batched_fused_moe_forward(
    const BatchedFusedMoEParams& params,
    const float* d_hidden,
    float*       d_output,
    cudaStream_t stream = 0);
