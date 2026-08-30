#include "fused_moe_batched.cuh"
#include "expert_layout.cuh"
#include "gate.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

static constexpr int B_TILE_IN  = 128;
static constexpr int B_TILE_OUT = 32;

__device__ __forceinline__ float silu_b(float x) {
    return x / (1.0f + __expf(-x));
}

// Warp-cooperative matvec row: dot(W[row,:], x[:]) using shared memory tile.
__device__ __forceinline__ float batched_matvec_row(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ x_tile,
    int in_dim,
    int row)
{
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    for (int col = 0; col < in_dim; col += B_TILE_IN) {
        #pragma unroll 4
        for (int t = lane; t < B_TILE_IN; t += 32)
            x_tile[t] = (col + t < in_dim) ? x[col + t] : 0.0f;
        __syncwarp();
        const float* w_row = W + (int64_t)row * in_dim + col;
        int len = min(B_TILE_IN, in_dim - col);
        #pragma unroll 8
        for (int t = 0; t < len; ++t) acc += w_row[t] * x_tile[t];
        __syncwarp();
    }
    return acc;
}

// ============================================================
// Batched fused MoE kernel.
//
// Grid:  (batch_size, 1)   -- one block per token
// Block: 32 threads        -- one warp per token
//
// Shared memory layout per block:
//   gate_smem[num_experts]         : router logits
//   xtile_smem[B_TILE_IN]          : input tile for matvec
//   gate_row_smem[B_TILE_OUT]      : gate_proj intermediate chunk
//   up_row_smem[B_TILE_OUT]        : up_proj intermediate chunk
//
// The router weight matrix is loaded element-by-element from global memory
// per block. For small hidden_dim or few experts this is the bottleneck;
// for large models the FFN computation dominates. Each block's smem is
// independent so all tokens run concurrently across SMs.
// ============================================================

__global__ void __launch_bounds__(32, 4) k_fused_moe_batched(
    const float* __restrict__ gate_matrix,   // [num_experts x hidden_dim]
    const float* __restrict__ weight_buf,    // packed expert weights
    const float* __restrict__ d_hidden,      // [batch_size x hidden_dim]
    float*       __restrict__ d_output,      // [batch_size x hidden_dim]
    ExpertOffsets off,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int top_k,
    int batch_size)
{
    const int token_idx = blockIdx.x;
    if (token_idx >= batch_size) return;

    extern __shared__ float smem[];

    const int lane = threadIdx.x;  // 0..31

    float* logits_smem = smem;
    float* xtile_smem  = smem + num_experts;
    float* gate_smem   = xtile_smem + B_TILE_IN;
    float* up_smem     = gate_smem  + B_TILE_OUT;

    const float* token_hidden = d_hidden  + (int64_t)token_idx * hidden_dim;
    float*       token_output = d_output  + (int64_t)token_idx * hidden_dim;

    // Step 1: Compute router logits for all experts.
    for (int e = 0; e < num_experts; ++e) {
        const float* row = gate_matrix + (int64_t)e * hidden_dim;
        float acc = 0.0f;
        for (int i = lane; i < hidden_dim; i += 32) acc += row[i] * token_hidden[i];
        acc += __shfl_xor_sync(0xffffffff, acc, 16);
        acc += __shfl_xor_sync(0xffffffff, acc,  8);
        acc += __shfl_xor_sync(0xffffffff, acc,  4);
        acc += __shfl_xor_sync(0xffffffff, acc,  2);
        acc += __shfl_xor_sync(0xffffffff, acc,  1);
        if (lane == 0) logits_smem[e] = acc;
    }
    __syncwarp();

    // Step 2: Softmax + top-K (lane 0, broadcast).
    float top_weights[2] = {0.0f, 0.0f};
    int   top_ids[2]     = {0, 0};

    if (lane == 0) {
        float vmax = -FLT_MAX;
        for (int e = 0; e < num_experts; ++e) vmax = fmaxf(vmax, logits_smem[e]);
        float sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            float v = __expf(logits_smem[e] - vmax);
            logits_smem[e] = v;
            sum += v;
        }
        for (int e = 0; e < num_experts; ++e) logits_smem[e] /= sum;

        float used[GATE_MAX_EXPERTS] = {};
        for (int ki = 0; ki < top_k; ++ki) {
            float bv = -FLT_MAX; int bi = 0;
            for (int e = 0; e < num_experts; ++e) {
                if (!used[e] && logits_smem[e] > bv) { bv = logits_smem[e]; bi = e; }
            }
            top_ids[ki]     = bi;
            top_weights[ki] = bv;
            used[bi]        = 1.0f;
        }
    }
    top_ids[0]     = __shfl_sync(0xffffffff, top_ids[0],     0);
    top_weights[0] = __shfl_sync(0xffffffff, top_weights[0], 0);
    if (top_k > 1) {
        top_ids[1]     = __shfl_sync(0xffffffff, top_ids[1],     0);
        top_weights[1] = __shfl_sync(0xffffffff, top_weights[1], 0);
    }

    // Step 3: SwiGLU FFN for each selected expert, accumulate into token_output.
    for (int ki = 0; ki < top_k; ++ki) {
        int   expert = top_ids[ki];
        float weight = top_weights[ki];

        const float* gp = gate_proj_ptr(weight_buf, off, expert);
        const float* up = up_proj_ptr  (weight_buf, off, expert);
        const float* dp = down_proj_ptr(weight_buf, off, expert);

        if (ki == 0) {
            for (int r = lane; r < hidden_dim; r += 32) token_output[r] = 0.0f;
        }
        __syncwarp();

        for (int ci = 0; ci < intermediate_dim; ci += B_TILE_OUT) {
            int irow = ci + lane;
            float gate_val = 0.0f, up_val = 0.0f;
            if (irow < intermediate_dim) {
                gate_val = batched_matvec_row(gp, token_hidden, xtile_smem, hidden_dim, irow);
                up_val   = batched_matvec_row(up, token_hidden, xtile_smem, hidden_dim, irow);
            }
            float act_val = silu_b(gate_val) * up_val;

            for (int out_base = 0; out_base < hidden_dim; out_base += 32) {
                int orow = out_base + lane;
                float contrib = 0.0f;
                #pragma unroll
                for (int src = 0; src < 32; ++src) {
                    int irow_src = ci + src;
                    float av = __shfl_sync(0xffffffff, act_val, src);
                    if (irow_src < intermediate_dim && orow < hidden_dim) {
                        contrib += dp[(int64_t)orow * intermediate_dim + irow_src] * av;
                    }
                }
                if (orow < hidden_dim) {
                    token_output[orow] += weight * contrib;
                }
            }
        }
    }
}

// ============================================================
// Public API
// ============================================================

void batched_fused_moe_forward(
    const BatchedFusedMoEParams& p,
    const float* d_hidden,
    float*       d_output,
    cudaStream_t stream)
{
    if (p.top_k < 1 || p.top_k > 2) {
        fprintf(stderr, "batched_fused_moe_forward: top_k must be 1 or 2, got %d\n", p.top_k);
        exit(1);
    }
    if (p.layout.num_experts > GATE_MAX_EXPERTS) {
        fprintf(stderr, "batched_fused_moe_forward: num_experts %d exceeds GATE_MAX_EXPERTS %d\n",
                p.layout.num_experts, GATE_MAX_EXPERTS);
        exit(1);
    }
    if (p.batch_size <= 0) return;

    const ExpertLayoutConfig& cfg = p.layout;
    ExpertOffsets off = compute_offsets(cfg);

    int smem_bytes = (cfg.num_experts + B_TILE_IN + 2 * B_TILE_OUT) * sizeof(float);

    k_fused_moe_batched<<<p.batch_size, 32, smem_bytes, stream>>>(
        p.gate_matrix, p.weight_buf,
        d_hidden, d_output,
        off,
        cfg.hidden_dim, cfg.intermediate_dim, cfg.num_experts,
        p.top_k, p.batch_size);
}
