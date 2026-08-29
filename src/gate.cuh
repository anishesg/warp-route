#pragma once
#include <cuda_runtime.h>

// Maximum experts supported in device-side top-K selection.
#define GATE_MAX_EXPERTS 64
// Maximum top-K supported.
#define GATE_MAX_TOPK 6

struct TopKResult {
    int   expert_ids[GATE_MAX_TOPK];
    float weights[GATE_MAX_TOPK];  // softmax-normalized gate weights for selected experts
    int   k;
};

// Compute gating logits for a single token (hidden vector of length hidden_dim),
// apply softmax over all num_experts experts, and select the top-k experts using
// warp-level operations.
//
// gate_matrix: row-major [num_experts x hidden_dim] router weight matrix
// hidden:      input hidden state [hidden_dim]
// logits_buf:  shared scratch [num_experts] (must be allocated by caller)
// result:      output written by lane 0; other lanes hold partial data
//
// All threads in the warp must call this function together (implicit sync).
__device__ TopKResult warp_gate_topk(
    const float* __restrict__ gate_matrix,
    const float* __restrict__ hidden,
    float* __restrict__ logits_buf,   // shared memory scratch, size >= num_experts
    int hidden_dim,
    int num_experts,
    int top_k);

// Standalone logit computation: each thread in the warp computes a partial dot product
// for a single expert. Returns the full logit for expert `expert_id` computed
// cooperatively across the warp (result valid on all lanes).
__device__ float warp_dot_product(
    const float* __restrict__ row,    // gate_matrix row for one expert [hidden_dim]
    const float* __restrict__ hidden, // input hidden state [hidden_dim]
    int hidden_dim);
