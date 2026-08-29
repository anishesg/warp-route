#include "fused_moe.cuh"
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

// ============================================================
// Shared memory layout for one warp's execution:
//
//   [0 .. num_experts)              : gate logits (float)
//   [num_experts .. num_experts+TILE_IN) : input tile for matvec (float)
//
// TILE_IN is chosen to fill one cache line group without exceeding 48KB smem.
// ============================================================

static constexpr int TILE_IN  = 128;   // input elements loaded per tile
static constexpr int TILE_OUT = 32;    // output rows computed per pass (= warp width)

// SiLU: x * sigmoid(x)
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

// Warp-cooperative dot product of W[row, :] and x[:], tile-by-tile via shared memory.
// x_tile is shared memory scratch of size TILE_IN.
// Returns the dot product for `row`; only valid if row < out_dim.
__device__ __forceinline__ float matvec_row(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ x_tile,
    int in_dim,
    int row)
{
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    for (int col = 0; col < in_dim; col += TILE_IN) {
        // All 32 threads load TILE_IN elements of x cooperatively.
        #pragma unroll 4
        for (int t = lane; t < TILE_IN; t += 32) {
            x_tile[t] = (col + t < in_dim) ? x[col + t] : 0.0f;
        }
        __syncwarp();
        // Each thread computes its portion of the dot product for its row.
        const float* w_row = W + (int64_t)row * in_dim + col;
        int len = min(TILE_IN, in_dim - col);
        #pragma unroll 8
        for (int t = 0; t < len; ++t) acc += w_row[t] * x_tile[t];
        __syncwarp();
    }
    return acc;
}

// ============================================================
// Fused MoE kernel: one warp per token.
//
// Shared memory layout (all floats):
//   logits_smem[0..num_experts)   : router logits
//   xtile_smem[0..TILE_IN)        : input tile for matvec
//   gate_smem[0..out_rows)        : gate_proj output rows for this warp's pass
//   up_smem[0..out_rows)          : up_proj output rows for this warp's pass
//
// For intermediate_dim rows we process them in chunks of TILE_OUT=32.
// This keeps smem usage independent of intermediate_dim.
// ============================================================

__global__ void __launch_bounds__(32, 4) k_fused_moe(
    const float* __restrict__ gate_matrix,   // [num_experts x hidden_dim]
    const float* __restrict__ weight_buf,    // packed expert weights
    const float* __restrict__ d_hidden,      // [hidden_dim]
    float*       __restrict__ d_output,      // [hidden_dim]
    ExpertOffsets off,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int top_k)
{
    // This kernel is launched with exactly 32 threads per block (one warp).
    extern __shared__ float smem[];

    const int lane = threadIdx.x;  // 0..31

    float* logits_smem = smem;                              // [num_experts]
    float* xtile_smem  = smem + num_experts;                // [TILE_IN]
    float* gate_smem   = xtile_smem + TILE_IN;              // [TILE_OUT]
    float* up_smem     = gate_smem  + TILE_OUT;             // [TILE_OUT]

    // ----------------------------------------------------------------
    // Step 1: Compute router logits for all experts (warp-cooperative).
    // ----------------------------------------------------------------
    for (int e = 0; e < num_experts; ++e) {
        const float* row = gate_matrix + (int64_t)e * hidden_dim;
        float acc = 0.0f;
        for (int i = lane; i < hidden_dim; i += 32) acc += row[i] * d_hidden[i];
        // Warp reduction
        acc += __shfl_xor_sync(0xffffffff, acc, 16);
        acc += __shfl_xor_sync(0xffffffff, acc,  8);
        acc += __shfl_xor_sync(0xffffffff, acc,  4);
        acc += __shfl_xor_sync(0xffffffff, acc,  2);
        acc += __shfl_xor_sync(0xffffffff, acc,  1);
        if (lane == 0) logits_smem[e] = acc;
    }
    __syncwarp();

    // ----------------------------------------------------------------
    // Step 2: Softmax + top-K selection (lane 0, broadcast to warp).
    // ----------------------------------------------------------------
    float top_weights[2] = {0.0f, 0.0f};
    int   top_ids[2]     = {0, 0};

    if (lane == 0) {
        // Softmax
        float vmax = -FLT_MAX;
        for (int e = 0; e < num_experts; ++e) vmax = fmaxf(vmax, logits_smem[e]);
        float sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            float v = __expf(logits_smem[e] - vmax);
            logits_smem[e] = v;
            sum += v;
        }
        for (int e = 0; e < num_experts; ++e) logits_smem[e] /= sum;

        // Top-K via iterative argmax
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
    // Broadcast selection to all lanes.
    top_ids[0]     = __shfl_sync(0xffffffff, top_ids[0],     0);
    top_weights[0] = __shfl_sync(0xffffffff, top_weights[0], 0);
    if (top_k > 1) {
        top_ids[1]     = __shfl_sync(0xffffffff, top_ids[1],     0);
        top_weights[1] = __shfl_sync(0xffffffff, top_weights[1], 0);
    }

    // ----------------------------------------------------------------
    // Step 3: For each selected expert, compute SwiGLU FFN and accumulate
    // weighted result into output. Each lane owns output rows:
    //   lane, lane+32, lane+64, ... up to hidden_dim.
    //
    // intermediate buffer: we loop over intermediate_dim in chunks of TILE_OUT=32.
    // For each chunk of 32 intermediate rows, lane `l` holds act_buf[l].
    // ----------------------------------------------------------------

    // Output accumulator: lane owns hidden_dim/32 rows each.
    // We write output directly using a temporary float array in registers.
    // For hidden_dim up to 8192, each lane owns at most 256 rows, too large for registers.
    // Instead we do a two-pass approach: accumulate weighted partial sums directly into
    // global memory output buffer, zero-initializing on the first expert pass.

    for (int ki = 0; ki < top_k; ++ki) {
        int   expert = top_ids[ki];
        float weight = top_weights[ki];

        const float* gp = gate_proj_ptr(weight_buf, off, expert);
        const float* up = up_proj_ptr  (weight_buf, off, expert);
        const float* dp = down_proj_ptr(weight_buf, off, expert);

        // Compute act = silu(gate_proj * hidden) * (up_proj * hidden)
        // in chunks of TILE_OUT=32 intermediate rows.
        // Store act chunk in register `act_val` for lane `lane`.
        // Then accumulate: output += weight * (down_proj * act), where act is chunked.

        // We need to accumulate down_proj contributions across all intermediate chunks.
        // down_proj is [hidden_dim x intermediate_dim]. For each intermediate chunk ci,
        // down_proj[:, ci*TILE_OUT : (ci+1)*TILE_OUT] * act_chunk contributes to all
        // hidden_dim output rows.
        //
        // Strategy: accumulate output in global memory, writing zeroes first (ki==0).
        if (ki == 0 && lane < hidden_dim) {
            // Zero output on first expert pass.
            for (int r = lane; r < hidden_dim; r += 32) d_output[r] = 0.0f;
        }
        __syncwarp();

        for (int ci = 0; ci < intermediate_dim; ci += TILE_OUT) {
            int irow = ci + lane;  // this lane's intermediate row index

            // gate_proj row `irow` dot hidden
            float gate_val = 0.0f, up_val = 0.0f;
            if (irow < intermediate_dim) {
                gate_val = matvec_row(gp, d_hidden, xtile_smem, hidden_dim, irow);
                up_val   = matvec_row(up, d_hidden, xtile_smem, hidden_dim, irow);
            }
            float act_val = silu_f(gate_val) * up_val;

            // Now compute: for each output row r, output[r] += weight * act_val * down_proj[r][irow+lane]
            // But lane already indexes irow = ci + lane, so each lane has act_val for one intermediate row.
            // We need: output[r] += weight * sum_{j in chunk} down_proj[r][ci+j] * act_val_j
            //
            // Broadcast act_val from each lane to all lanes via shuffle, then each lane
            // computes its output rows.
            for (int out_base = 0; out_base < hidden_dim; out_base += 32) {
                int orow = out_base + lane;
                float contrib = 0.0f;
                // Each lane `src` has act_val for intermediate row ci+src.
                #pragma unroll
                for (int src = 0; src < 32; ++src) {
                    int irow_src = ci + src;
                    float av = __shfl_sync(0xffffffff, act_val, src);
                    if (irow_src < intermediate_dim && orow < hidden_dim) {
                        contrib += dp[(int64_t)orow * intermediate_dim + irow_src] * av;
                    }
                }
                if (orow < hidden_dim) {
                    d_output[orow] += weight * contrib;
                }
            }
        }
    }
}

// ============================================================
// Public API
// ============================================================

void fused_moe_forward(
    const FusedMoEParams& p,
    const float* d_hidden,
    float*       d_output,
    cudaStream_t stream)
{
    const ExpertLayoutConfig& cfg = p.layout;
    ExpertOffsets off = compute_offsets(cfg);

    // Shared memory: logits + xtile + gate_row + up_row
    int smem_bytes = (cfg.num_experts + TILE_IN + 2 * TILE_OUT) * sizeof(float);

    k_fused_moe<<<1, 32, smem_bytes, stream>>>(
        p.gate_matrix, p.weight_buf,
        d_hidden, d_output,
        off,
        cfg.hidden_dim, cfg.intermediate_dim, cfg.num_experts,
        p.top_k);
}
