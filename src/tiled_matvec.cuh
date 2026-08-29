#pragma once
#include <cuda_runtime.h>

// Tiled warp-cooperative matrix-vector product: y = W * x
//
// W is [out_dim x in_dim] row-major.
// x is [in_dim].
// y is [out_dim], accumulated into the caller's output array.
//
// TILE_IN controls how many input elements are loaded per tile (must divide 32 evenly
// or be handled with a tail). Tiles are processed iteratively so shared memory
// usage is fixed regardless of in_dim.
//
// All 32 threads in the warp participate. Thread `lane` accumulates partial dot
// products for output rows: lane t owns rows t, t+32, t+64, ... up to out_dim.
// The result is written directly to y[row] by the owning lane.
//
// Shared memory requirement: TILE_IN floats for the input tile.

template <int TILE_IN>
__device__ void tiled_matvec(
    const float* __restrict__ W,   // [out_dim x in_dim]
    const float* __restrict__ x,   // [in_dim]
    float*       __restrict__ y,   // [out_dim], written (not accumulated)
    float*       __restrict__ x_tile,  // shared memory scratch, size >= TILE_IN
    int in_dim,
    int out_dim)
{
    const int lane = threadIdx.x & 31;

    // Initialize output accumulators for this lane's rows.
    // Each lane owns rows: lane, lane+32, lane+64, ...
    float acc[1];  // We process one output row per tile pass per lane,
                   // but since out_dim can be large we loop over row groups.

    // Outer loop: groups of 32 output rows
    for (int row_base = 0; row_base < out_dim; row_base += 32) {
        int row = row_base + lane;
        float row_acc = 0.0f;

        // Inner loop: tiles over in_dim
        for (int col_base = 0; col_base < in_dim; col_base += TILE_IN) {
            // Cooperatively load a tile of x into shared memory.
            // Lane i loads x[col_base + i], lane i+32 loads col_base + i + 32 if TILE_IN > 32.
            #pragma unroll
            for (int t = lane; t < TILE_IN && (col_base + t) < in_dim; t += 32) {
                x_tile[t] = x[col_base + t];
            }
            __syncwarp();

            // Compute partial dot product for this row's tile.
            if (row < out_dim) {
                const float* w_row = W + (int64_t)row * in_dim + col_base;
                int tile_len = min(TILE_IN, in_dim - col_base);
                #pragma unroll 8
                for (int t = 0; t < tile_len; ++t) {
                    row_acc += w_row[t] * x_tile[t];
                }
            }
            __syncwarp();
        }

        if (row < out_dim) {
            y[row] = row_acc;
        }
    }
}

// Variant that accumulates into y rather than overwriting; useful for SwiGLU
// where gate and up projections must be computed before the multiply.
template <int TILE_IN>
__device__ void tiled_matvec_accumulate(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float*       __restrict__ y,    // y[row] += dot(W[row], x)
    float*       __restrict__ x_tile,
    int in_dim,
    int out_dim)
{
    const int lane = threadIdx.x & 31;

    for (int row_base = 0; row_base < out_dim; row_base += 32) {
        int row = row_base + lane;
        float row_acc = 0.0f;

        for (int col_base = 0; col_base < in_dim; col_base += TILE_IN) {
            #pragma unroll
            for (int t = lane; t < TILE_IN && (col_base + t) < in_dim; t += 32) {
                x_tile[t] = x[col_base + t];
            }
            __syncwarp();

            if (row < out_dim) {
                const float* w_row = W + (int64_t)row * in_dim + col_base;
                int tile_len = min(TILE_IN, in_dim - col_base);
                #pragma unroll 8
                for (int t = 0; t < tile_len; ++t) {
                    row_acc += w_row[t] * x_tile[t];
                }
            }
            __syncwarp();
        }

        if (row < out_dim) {
            y[row] += row_acc;
        }
    }
}

// SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
__device__ inline float silu(float x) {
    return x / (1.0f + __expf(-x));
}
