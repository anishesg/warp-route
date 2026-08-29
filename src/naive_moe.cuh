#pragma once
#include <cuda_runtime.h>
#include "expert_layout.cuh"

// Naive multi-kernel MoE forward pass for a single token at decode time.
//
// This is the correctness oracle. It performs:
//   1. Router matmul: hidden -> expert logits
//   2. Softmax + top-K selection (CPU-side via cudaMemcpy for simplicity)
//   3. For each selected expert:
//        a. gate_proj matvec: hidden -> gate_vec [intermediate_dim]
//        b. up_proj matvec:   hidden -> up_vec   [intermediate_dim]
//        c. element-wise SiLU(gate_vec) * up_vec -> act_vec
//        d. down_proj matvec: act_vec -> expert_out [hidden_dim]
//   4. Weighted sum of expert outputs -> output [hidden_dim]

struct NaiveMoEParams {
    const float* gate_matrix;   // [num_experts x hidden_dim] router weights
    const float* weight_buf;    // packed expert weights (see expert_layout.cuh)
    ExpertLayoutConfig layout;
    int top_k;
};

// Allocate and free device-side scratch buffers needed by the naive implementation.
struct NaiveMoEScratch {
    float* logits;          // [num_experts]
    float* expert_out;      // [hidden_dim] per-expert output before accumulation
    float* gate_vec;        // [intermediate_dim]
    float* up_vec;          // [intermediate_dim]
    float* act_vec;         // [intermediate_dim]  (= silu(gate) * up)
    float* output_accum;    // [hidden_dim]
};

NaiveMoEScratch naive_moe_alloc_scratch(const ExpertLayoutConfig& cfg);
void naive_moe_free_scratch(NaiveMoEScratch& s);

// Run the naive multi-kernel MoE forward pass.
// hidden: [hidden_dim], output: [hidden_dim]. Both in device memory.
void naive_moe_forward(
    const NaiveMoEParams& params,
    const float* d_hidden,
    float*       d_output,
    NaiveMoEScratch& scratch,
    cudaStream_t stream = 0);
