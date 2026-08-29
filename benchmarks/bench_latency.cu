#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>

#include "expert_layout.cuh"
#include "naive_moe.cuh"
#include "fused_moe.cuh"

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

static void rand_fill(float* data, int n, std::mt19937& rng, float scale = 0.02f) {
    std::normal_distribution<float> dist(0.0f, scale);
    for (int i = 0; i < n; ++i) data[i] = dist(rng);
}

// ============================================================
// CUDA event timer helpers
// ============================================================

struct Timer {
    cudaEvent_t start, stop;
    Timer()  { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start, s)); }
    float end(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(stop, s));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ============================================================
// Benchmark a single configuration
// ============================================================

struct BenchConfig {
    const char* name;
    int num_experts;
    int hidden_dim;
    int intermediate_dim;
    int top_k;
};

static void bench_config(const BenchConfig& cfg, std::mt19937& rng, int warmup, int iters) {
    ExpertLayoutConfig layout = {cfg.num_experts, cfg.hidden_dim, cfg.intermediate_dim};
    ExpertOffsets off = compute_offsets(layout);

    int64_t gate_elems = (int64_t)cfg.num_experts * cfg.hidden_dim;

    std::vector<float> h_gate(gate_elems), h_hidden(cfg.hidden_dim);
    std::vector<float> h_weights(total_weight_bytes(layout) / sizeof(float), 0.0f);
    rand_fill(h_gate.data(), gate_elems, rng);
    rand_fill(h_hidden.data(), cfg.hidden_dim, rng, 0.1f);
    for (int e = 0; e < cfg.num_experts; ++e) {
        rand_fill(const_cast<float*>(gate_proj_ptr(h_weights.data(), off, e)),
                  cfg.intermediate_dim * cfg.hidden_dim, rng, 0.01f);
        rand_fill(const_cast<float*>(up_proj_ptr(h_weights.data(), off, e)),
                  cfg.intermediate_dim * cfg.hidden_dim, rng, 0.01f);
        rand_fill(const_cast<float*>(down_proj_ptr(h_weights.data(), off, e)),
                  cfg.hidden_dim * cfg.intermediate_dim, rng, 0.01f);
    }

    float *d_gate, *d_weights, *d_hidden, *d_out;
    CUDA_CHECK(cudaMalloc(&d_gate,    gate_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, total_weight_bytes(layout)));
    CUDA_CHECK(cudaMalloc(&d_hidden,  cfg.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,     cfg.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_gate,    h_gate.data(),    gate_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), total_weight_bytes(layout), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden,  h_hidden.data(),  cfg.hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    NaiveMoEParams naive_p  = {d_gate, d_weights, layout, cfg.top_k};
    FusedMoEParams fused_p  = {d_gate, d_weights, layout, cfg.top_k};
    NaiveMoEScratch scratch = naive_moe_alloc_scratch(layout);

    Timer t;

    // ---- Naive benchmark ----
    for (int i = 0; i < warmup; ++i)
        naive_moe_forward(naive_p, d_hidden, d_out, scratch);
    CUDA_CHECK(cudaDeviceSynchronize());

    t.begin();
    for (int i = 0; i < iters; ++i)
        naive_moe_forward(naive_p, d_hidden, d_out, scratch);
    float naive_ms = t.end();
    float naive_us = naive_ms / iters * 1000.0f;

    // ---- Fused benchmark ----
    for (int i = 0; i < warmup; ++i)
        fused_moe_forward(fused_p, d_hidden, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    t.begin();
    for (int i = 0; i < iters; ++i)
        fused_moe_forward(fused_p, d_hidden, d_out);
    float fused_ms = t.end();
    float fused_us = fused_ms / iters * 1000.0f;

    // Memory bandwidth estimate: dominant traffic is loading the two selected expert
    // weight matrices. For top-k experts: k * (gate+up+down) bytes.
    // gate+up: 2 * intermediate * hidden * 4 bytes; down: hidden * intermediate * 4 bytes.
    double bytes_per_expert = 3.0 * cfg.intermediate_dim * cfg.hidden_dim * sizeof(float);
    double bw_fused_gbs = (cfg.top_k * bytes_per_expert) / (fused_us * 1e-6) / 1e9;
    double bw_naive_gbs = (cfg.top_k * bytes_per_expert) / (naive_us * 1e-6) / 1e9;
    double speedup = naive_us / fused_us;

    printf("%-20s  experts=%-3d  hidden=%-5d  inter=%-6d  top_k=%d\n",
           cfg.name, cfg.num_experts, cfg.hidden_dim, cfg.intermediate_dim, cfg.top_k);
    printf("  naive:  %8.1f us   bw: %6.1f GB/s\n", naive_us, bw_naive_gbs);
    printf("  fused:  %8.1f us   bw: %6.1f GB/s\n", fused_us, bw_fused_gbs);
    printf("  speedup: %.2fx\n\n", speedup);

    naive_moe_free_scratch(scratch);
    cudaFree(d_gate); cudaFree(d_weights); cudaFree(d_hidden); cudaFree(d_out);
}

// ============================================================
// Per-kernel naive breakdown (gating, per-expert phases, accumulate)
// ============================================================

// Kernel declarations (defined in naive_moe.cu; redeclare here for timing).
// We time the naive forward with CUDA events around individual kernel groups
// by wrapping naive_moe_forward and inserting event records between sub-phases.
// Since naive_moe_forward encapsulates all kernels on the same stream, we
// measure total naive time and decompose by expected contribution.

static void bench_breakdown(const BenchConfig& cfg, std::mt19937& rng, int warmup, int iters) {
    ExpertLayoutConfig layout = {cfg.num_experts, cfg.hidden_dim, cfg.intermediate_dim};
    ExpertOffsets off = compute_offsets(layout);
    int64_t gate_elems = (int64_t)cfg.num_experts * cfg.hidden_dim;

    std::vector<float> h_gate(gate_elems), h_hidden(cfg.hidden_dim);
    std::vector<float> h_weights(total_weight_bytes(layout) / sizeof(float), 0.0f);
    rand_fill(h_gate.data(), gate_elems, rng);
    rand_fill(h_hidden.data(), cfg.hidden_dim, rng, 0.1f);
    for (int e = 0; e < cfg.num_experts; ++e) {
        rand_fill(const_cast<float*>(gate_proj_ptr(h_weights.data(), off, e)),
                  cfg.intermediate_dim * cfg.hidden_dim, rng, 0.01f);
        rand_fill(const_cast<float*>(up_proj_ptr(h_weights.data(), off, e)),
                  cfg.intermediate_dim * cfg.hidden_dim, rng, 0.01f);
        rand_fill(const_cast<float*>(down_proj_ptr(h_weights.data(), off, e)),
                  cfg.hidden_dim * cfg.intermediate_dim, rng, 0.01f);
    }

    float *d_gate, *d_weights, *d_hidden, *d_out;
    CUDA_CHECK(cudaMalloc(&d_gate,    gate_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, total_weight_bytes(layout)));
    CUDA_CHECK(cudaMalloc(&d_hidden,  cfg.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,     cfg.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_gate,    h_gate.data(),    gate_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), total_weight_bytes(layout), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden,  h_hidden.data(),  cfg.hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    NaiveMoEParams naive_p  = {d_gate, d_weights, layout, cfg.top_k};
    FusedMoEParams fused_p  = {d_gate, d_weights, layout, cfg.top_k};
    NaiveMoEScratch scratch = naive_moe_alloc_scratch(layout);

    Timer t_total;
    // Warm up
    for (int i = 0; i < warmup; ++i) {
        naive_moe_forward(naive_p, d_hidden, d_out, scratch);
        fused_moe_forward(fused_p, d_hidden, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure naive total
    t_total.begin();
    for (int i = 0; i < iters; ++i)
        naive_moe_forward(naive_p, d_hidden, d_out, scratch);
    float naive_total_us = t_total.end() / iters * 1000.0f;

    // Measure fused total
    t_total.begin();
    for (int i = 0; i < iters; ++i)
        fused_moe_forward(fused_p, d_hidden, d_out);
    float fused_total_us = t_total.end() / iters * 1000.0f;

    // Estimate kernel-launch overhead: ~3-10 us per launch on A100.
    // Naive launches: 1 (router) + 1 (softmax, CPU) + top_k * 4 (gate,up,swiglu,down) + 1 (accumulate)
    int naive_launches = 1 + cfg.top_k * 4 + 1;
    float launch_overhead_est = naive_launches * 5.0f;  // ~5us per launch estimate

    // Bytes moved by permutation (naive): 2 * hidden_dim * sizeof(float) per expert pass
    // (one write to permuted buffer, one read back). The fused kernel skips this entirely.
    double permute_bytes = 2.0 * cfg.top_k * cfg.hidden_dim * sizeof(float);
    double permute_traffic_gb = permute_bytes / 1e9;

    printf("Breakdown for %s (top_k=%d):\n", cfg.name, cfg.top_k);
    printf("  naive total:        %8.1f us  (%d kernel launches)\n", naive_total_us, naive_launches);
    printf("  fused total:        %8.1f us  (1 kernel launch)\n", fused_total_us);
    printf("  speedup:            %.2fx\n", naive_total_us / fused_total_us);
    printf("  launch overhead:    ~%.0f us (est. %d launches x 5us)\n",
           launch_overhead_est, naive_launches);
    printf("  permute traffic:    %.0f KB eliminated\n",
           permute_bytes / 1024.0);
    printf("\n");

    naive_moe_free_scratch(scratch);
    cudaFree(d_gate); cudaFree(d_weights); cudaFree(d_hidden); cudaFree(d_out);
}

// ============================================================
// Main
// ============================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("warp-route latency benchmark\n");
    printf("GPU: %s (sm_%d%d, %.0f GB/s peak BW)\n\n",
           prop.name, prop.major, prop.minor,
           (double)prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 * 1e3 / 1e9);

    std::mt19937 rng(99);

    const int WARMUP = 10, ITERS = 100;

    // Model configurations
    BenchConfig configs[] = {
        {"mixtral-8x7b",   8,  4096, 14336, 2},
        {"mixtral-8x7b",   8,  4096, 14336, 1},
        {"deepseek-moe",  64,  2048,  1408, 2},
        {"dbrx",          16,  6144, 10752, 2},
    };

    printf("=== Latency comparison (us per MoE layer) ===\n\n");
    for (auto& c : configs) {
        if (c.num_experts <= GATE_MAX_EXPERTS)
            bench_config(c, rng, WARMUP, ITERS);
    }

    printf("=== Per-kernel breakdown (Mixtral 8x7B, top-2) ===\n\n");
    BenchConfig mixtral = {"mixtral-8x7b", 8, 4096, 14336, 2};
    bench_breakdown(mixtral, rng, WARMUP, ITERS);

    printf("=== Per-kernel breakdown (DeepSeek-MoE, top-2) ===\n\n");
    BenchConfig deepseek = {"deepseek-moe", 64, 2048, 1408, 2};
    if (deepseek.num_experts <= GATE_MAX_EXPERTS)
        bench_breakdown(deepseek, rng, WARMUP, ITERS);

    return 0;
}
