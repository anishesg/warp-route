#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Memory layout for expert FFN weights.
//
// Each expert's three projection matrices are stored in a contiguous flat buffer
// with the following layout:
//
//   [ gate_proj: intermediate x hidden ]
//   [ up_proj:   intermediate x hidden ]
//   [ down_proj: hidden x intermediate ]
//
// All matrices are row-major. gate_proj and up_proj each map hidden_dim -> intermediate_dim.
// down_proj maps intermediate_dim -> hidden_dim.
//
// The full weight buffer holds all num_experts experts packed contiguously, so
// expert i starts at byte offset i * expert_stride_bytes.

struct ExpertLayoutConfig {
    int num_experts;
    int hidden_dim;
    int intermediate_dim;
};

// Byte offsets within a single expert's weight block.
struct ExpertOffsets {
    int64_t gate_proj_offset;  // bytes from start of expert block
    int64_t up_proj_offset;
    int64_t down_proj_offset;
    int64_t expert_stride;     // total bytes per expert (= alignment-padded block size)
};

__host__ __device__ inline ExpertOffsets compute_offsets(const ExpertLayoutConfig& cfg) {
    ExpertOffsets off;
    // Size of each projection in elements.
    int64_t gateup_elems = (int64_t)cfg.intermediate_dim * cfg.hidden_dim;
    int64_t down_elems   = (int64_t)cfg.hidden_dim * cfg.intermediate_dim;

    off.gate_proj_offset = 0;
    off.up_proj_offset   = gateup_elems * sizeof(float);
    off.down_proj_offset = off.up_proj_offset + gateup_elems * sizeof(float);
    // Align expert stride to 128 bytes for cache-line alignment.
    int64_t raw_stride = off.down_proj_offset + down_elems * sizeof(float);
    off.expert_stride  = (raw_stride + 127) & ~int64_t(127);
    return off;
}

// Returns a pointer to expert e's gate_proj matrix (row-major, intermediate x hidden).
__device__ inline const float* gate_proj_ptr(const float* __restrict__ weight_buf,
                                              const ExpertOffsets& off,
                                              int e) {
    const char* base = reinterpret_cast<const char*>(weight_buf) + (int64_t)e * off.expert_stride;
    return reinterpret_cast<const float*>(base + off.gate_proj_offset);
}

// Returns a pointer to expert e's up_proj matrix (row-major, intermediate x hidden).
__device__ inline const float* up_proj_ptr(const float* __restrict__ weight_buf,
                                            const ExpertOffsets& off,
                                            int e) {
    const char* base = reinterpret_cast<const char*>(weight_buf) + (int64_t)e * off.expert_stride;
    return reinterpret_cast<const float*>(base + off.up_proj_offset);
}

// Returns a pointer to expert e's down_proj matrix (row-major, hidden x intermediate).
__device__ inline const float* down_proj_ptr(const float* __restrict__ weight_buf,
                                              const ExpertOffsets& off,
                                              int e) {
    const char* base = reinterpret_cast<const char*>(weight_buf) + (int64_t)e * off.expert_stride;
    return reinterpret_cast<const float*>(base + off.down_proj_offset);
}

// Host-side helper: returns total bytes required to store all expert weights.
inline int64_t total_weight_bytes(const ExpertLayoutConfig& cfg) {
    ExpertOffsets off = compute_offsets(cfg);
    return (int64_t)cfg.num_experts * off.expert_stride;
}
