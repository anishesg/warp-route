#include "gate.cuh"
#include <float.h>

// Warp-cooperative dot product: 32 threads each handle hidden_dim/32 elements,
// then reduce via shuffle.
__device__ float warp_dot_product(
    const float* __restrict__ row,
    const float* __restrict__ hidden,
    int hidden_dim)
{
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    for (int i = lane; i < hidden_dim; i += 32) {
        acc += row[i] * hidden[i];
    }
    // Warp reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc,  8);
    acc += __shfl_xor_sync(0xffffffff, acc,  4);
    acc += __shfl_xor_sync(0xffffffff, acc,  2);
    acc += __shfl_xor_sync(0xffffffff, acc,  1);
    return acc;  // valid on all lanes after reduction
}

// Compute logits for all experts, writing into logits_buf (shared memory).
// Each expert's dot product is computed by the full warp; result broadcast to all lanes.
static __device__ void compute_all_logits(
    const float* __restrict__ gate_matrix,
    const float* __restrict__ hidden,
    float* __restrict__ logits_buf,
    int hidden_dim,
    int num_experts)
{
    for (int e = 0; e < num_experts; ++e) {
        float logit = warp_dot_product(gate_matrix + (int64_t)e * hidden_dim, hidden, hidden_dim);
        // Lane 0 writes; all lanes produced the same value via shuffle reduction.
        if ((threadIdx.x & 31) == 0) {
            logits_buf[e] = logit;
        }
    }
    __syncwarp();
}

// In-place softmax over logits_buf[0..num_experts-1].
// Executed collaboratively: lane 0 does the scalar work, then broadcasts.
static __device__ void warp_softmax(float* __restrict__ logits_buf, int num_experts)
{
    // Find max for numerical stability.
    float vmax = -FLT_MAX;
    if ((threadIdx.x & 31) == 0) {
        for (int e = 0; e < num_experts; ++e) vmax = fmaxf(vmax, logits_buf[e]);
    }
    vmax = __shfl_sync(0xffffffff, vmax, 0);

    float sum = 0.0f;
    if ((threadIdx.x & 31) == 0) {
        for (int e = 0; e < num_experts; ++e) {
            float v = __expf(logits_buf[e] - vmax);
            logits_buf[e] = v;
            sum += v;
        }
        for (int e = 0; e < num_experts; ++e) logits_buf[e] /= sum;
    }
    __syncwarp();
}

// Select top-k entries from logits_buf by iterative argmax.
// Fills result->expert_ids and result->weights; result valid on all lanes.
static __device__ void select_topk(
    const float* __restrict__ logits_buf,
    int num_experts,
    int top_k,
    TopKResult* result)
{
    // Simple iterative argmax run by lane 0, broadcast to all.
    // For num_experts <= 64 and top_k <= 6 this is cheap.
    float used[GATE_MAX_EXPERTS] = {};  // marks already-selected experts

    if ((threadIdx.x & 31) == 0) {
        for (int ki = 0; ki < top_k; ++ki) {
            float best_val = -FLT_MAX;
            int   best_idx = 0;
            for (int e = 0; e < num_experts; ++e) {
                if (!used[e] && logits_buf[e] > best_val) {
                    best_val = logits_buf[e];
                    best_idx = e;
                }
            }
            result->expert_ids[ki] = best_idx;
            result->weights[ki]    = best_val;
            used[best_idx] = 1.0f;
        }
        result->k = top_k;
    }
    // Broadcast expert_ids and weights to all lanes.
    for (int ki = 0; ki < top_k; ++ki) {
        result->expert_ids[ki] = __shfl_sync(0xffffffff, result->expert_ids[ki], 0);
        result->weights[ki]    = __shfl_sync(0xffffffff, result->weights[ki],    0);
    }
    result->k = top_k;
}

__device__ TopKResult warp_gate_topk(
    const float* __restrict__ gate_matrix,
    const float* __restrict__ hidden,
    float* __restrict__ logits_buf,
    int hidden_dim,
    int num_experts,
    int top_k)
{
    compute_all_logits(gate_matrix, hidden, logits_buf, hidden_dim, num_experts);
    warp_softmax(logits_buf, num_experts);
    TopKResult result;
    select_topk(logits_buf, num_experts, top_k, &result);
    return result;
}
