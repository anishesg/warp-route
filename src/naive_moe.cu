#include "naive_moe.cuh"
#include "expert_layout.cuh"
#include "tiled_matvec.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Scratch allocation / deallocation
// ============================================================

NaiveMoEScratch naive_moe_alloc_scratch(const ExpertLayoutConfig& cfg) {
    NaiveMoEScratch s;
    CUDA_CHECK(cudaMalloc(&s.logits,       cfg.num_experts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.expert_out,   cfg.hidden_dim  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.gate_vec,     cfg.intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.up_vec,       cfg.intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.act_vec,      cfg.intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.output_accum, cfg.hidden_dim  * sizeof(float)));
    return s;
}

void naive_moe_free_scratch(NaiveMoEScratch& s) {
    cudaFree(s.logits);
    cudaFree(s.expert_out);
    cudaFree(s.gate_vec);
    cudaFree(s.up_vec);
    cudaFree(s.act_vec);
    cudaFree(s.output_accum);
}

// ============================================================
// Kernel: router matmul -- compute logits[e] = dot(gate_matrix[e], hidden)
// One block, 256 threads, each expert handled cooperatively.
// ============================================================

__global__ void k_router_matmul(
    const float* __restrict__ gate_matrix,
    const float* __restrict__ hidden,
    float*       __restrict__ logits,
    int hidden_dim,
    int num_experts)
{
    // Each warp of 32 threads handles one expert.
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;
    int warps   = blockDim.x / 32;

    for (int e = warp_id; e < num_experts; e += warps) {
        const float* row = gate_matrix + (int64_t)e * hidden_dim;
        float acc = 0.0f;
        for (int i = lane; i < hidden_dim; i += 32) acc += row[i] * hidden[i];
        acc += __shfl_xor_sync(0xffffffff, acc, 16);
        acc += __shfl_xor_sync(0xffffffff, acc,  8);
        acc += __shfl_xor_sync(0xffffffff, acc,  4);
        acc += __shfl_xor_sync(0xffffffff, acc,  2);
        acc += __shfl_xor_sync(0xffffffff, acc,  1);
        if (lane == 0) logits[e] = acc;
    }
}

// ============================================================
// Kernel: matvec for a single projection -- y = W * x
// 256 threads; warp per group of 32 output rows.
// ============================================================

__global__ void k_matvec(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int in_dim,
    int out_dim)
{
    extern __shared__ float smem[];
    // Each warp handles a contiguous group of output rows.
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x & 31;
    int warps   = blockDim.x / 32;
    float* xtile = smem + warp_id * 64;  // 64-element tile per warp

    for (int row_base = warp_id * 32; row_base < out_dim; row_base += warps * 32) {
        int row = row_base + lane;
        float acc = 0.0f;
        for (int col = 0; col < in_dim; col += 64) {
            int t = col + lane;
            xtile[lane]      = (t        < in_dim) ? x[t]       : 0.0f;
            xtile[lane + 32] = (t + 32   < in_dim) ? x[t + 32]  : 0.0f;
            __syncwarp();
            if (row < out_dim) {
                const float* w = W + (int64_t)row * in_dim + col;
                int len = min(64, in_dim - col);
                #pragma unroll 8
                for (int t2 = 0; t2 < len; ++t2) acc += w[t2] * xtile[t2];
            }
            __syncwarp();
        }
        if (row < out_dim) y[row] = acc;
    }
}

// ============================================================
// Kernel: element-wise act_vec[i] = silu(gate[i]) * up[i]
// ============================================================

__global__ void k_swiglu(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ act,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        act[i] = (g / (1.0f + __expf(-g))) * up[i];
    }
}

// ============================================================
// Kernel: weighted accumulate -- output += w * expert_out
// ============================================================

__global__ void k_weighted_accum(
    float*       __restrict__ output,
    const float* __restrict__ expert_out,
    float weight,
    int n,
    bool zero_first)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (zero_first ? 0.0f : output[i]) + weight * expert_out[i];
    }
}

// ============================================================
// CPU-side softmax + top-K (avoids device->host->device roundtrip overhead
// in the critical path; acceptable for correctness reference only).
// ============================================================

static void cpu_softmax_topk(
    const float* h_logits, int num_experts, int top_k,
    int* out_ids, float* out_weights)
{
    // softmax
    std::vector<float> probs(num_experts);
    float vmax = -FLT_MAX;
    for (int e = 0; e < num_experts; ++e) vmax = std::max(vmax, h_logits[e]);
    float sum = 0.0f;
    for (int e = 0; e < num_experts; ++e) { probs[e] = std::exp(h_logits[e] - vmax); sum += probs[e]; }
    for (int e = 0; e < num_experts; ++e) probs[e] /= sum;

    // top-k by iterative argmax
    std::vector<bool> used(num_experts, false);
    for (int ki = 0; ki < top_k; ++ki) {
        int best = 0; float bval = -FLT_MAX;
        for (int e = 0; e < num_experts; ++e) {
            if (!used[e] && probs[e] > bval) { bval = probs[e]; best = e; }
        }
        out_ids[ki]     = best;
        out_weights[ki] = bval;
        used[best]      = true;
    }
}

// ============================================================
// Public API
// ============================================================

void naive_moe_forward(
    const NaiveMoEParams& p,
    const float* d_hidden,
    float*       d_output,
    NaiveMoEScratch& scratch,
    cudaStream_t stream)
{
    const ExpertLayoutConfig& cfg = p.layout;
    ExpertOffsets off = compute_offsets(cfg);

    // 1. Router matmul
    k_router_matmul<<<1, 256, 0, stream>>>(
        p.gate_matrix, d_hidden, scratch.logits, cfg.hidden_dim, cfg.num_experts);

    // 2. Copy logits to host for softmax + top-K
    std::vector<float> h_logits(cfg.num_experts);
    CUDA_CHECK(cudaMemcpyAsync(h_logits.data(), scratch.logits,
        cfg.num_experts * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int>   sel_ids(p.top_k);
    std::vector<float> sel_weights(p.top_k);
    cpu_softmax_topk(h_logits.data(), cfg.num_experts, p.top_k,
                     sel_ids.data(), sel_weights.data());

    // 3. Per-expert FFN
    int threads = 256;
    int smem_per_warp = 64 * sizeof(float);
    int smem = (threads / 32) * smem_per_warp;

    for (int ki = 0; ki < p.top_k; ++ki) {
        int e = sel_ids[ki];
        float w = sel_weights[ki];

        const float* gp = gate_proj_ptr(p.weight_buf, off, e);
        const float* up = up_proj_ptr(p.weight_buf, off, e);
        const float* dp = down_proj_ptr(p.weight_buf, off, e);

        // gate_proj: hidden -> gate_vec
        k_matvec<<<1, threads, smem, stream>>>(
            gp, d_hidden, scratch.gate_vec, cfg.hidden_dim, cfg.intermediate_dim);

        // up_proj: hidden -> up_vec
        k_matvec<<<1, threads, smem, stream>>>(
            up, d_hidden, scratch.up_vec, cfg.hidden_dim, cfg.intermediate_dim);

        // SwiGLU activation
        int swiglu_blocks = (cfg.intermediate_dim + 255) / 256;
        k_swiglu<<<swiglu_blocks, 256, 0, stream>>>(
            scratch.gate_vec, scratch.up_vec, scratch.act_vec, cfg.intermediate_dim);

        // down_proj: act_vec -> expert_out
        k_matvec<<<1, threads, smem, stream>>>(
            dp, scratch.act_vec, scratch.expert_out, cfg.intermediate_dim, cfg.hidden_dim);

        // Weighted accumulate into output
        int accum_blocks = (cfg.hidden_dim + 255) / 256;
        k_weighted_accum<<<accum_blocks, 256, 0, stream>>>(
            d_output, scratch.expert_out, w, cfg.hidden_dim, ki == 0);
    }
}
