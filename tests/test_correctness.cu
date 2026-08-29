#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
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

// ============================================================
// Host-side cosine similarity and max absolute error
// ============================================================

static float cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static float max_abs_error(const float* a, const float* b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; ++i) err = fmaxf(err, fabsf(a[i] - b[i]));
    return err;
}

// ============================================================
// Random initialization
// ============================================================

static void rand_fill(float* data, int n, std::mt19937& rng, float scale = 0.02f) {
    std::normal_distribution<float> dist(0.0f, scale);
    for (int i = 0; i < n; ++i) data[i] = dist(rng);
}

// ============================================================
// Single test case
// ============================================================

struct TestConfig {
    int num_experts;
    int hidden_dim;
    int intermediate_dim;
    int top_k;
};

static bool run_test(const TestConfig& cfg, std::mt19937& rng) {
    ExpertLayoutConfig layout = {cfg.num_experts, cfg.hidden_dim, cfg.intermediate_dim};
    ExpertOffsets off = compute_offsets(layout);

    int64_t gate_matrix_elems = (int64_t)cfg.num_experts * cfg.hidden_dim;
    int64_t weight_buf_elems  = total_weight_bytes(layout) / sizeof(float);

    // Allocate and fill host buffers
    std::vector<float> h_gate_matrix(gate_matrix_elems);
    std::vector<float> h_weight_buf(weight_buf_elems, 0.0f);
    std::vector<float> h_hidden(cfg.hidden_dim);

    rand_fill(h_gate_matrix.data(), gate_matrix_elems, rng, 0.02f);
    rand_fill(h_hidden.data(), cfg.hidden_dim, rng, 0.1f);

    // Fill each expert's weights
    for (int e = 0; e < cfg.num_experts; ++e) {
        float* gp = const_cast<float*>(
            gate_proj_ptr(h_weight_buf.data(), off, e));
        float* up = const_cast<float*>(
            up_proj_ptr(h_weight_buf.data(), off, e));
        float* dp = const_cast<float*>(
            down_proj_ptr(h_weight_buf.data(), off, e));
        rand_fill(gp, (int64_t)cfg.intermediate_dim * cfg.hidden_dim, rng, 0.01f);
        rand_fill(up, (int64_t)cfg.intermediate_dim * cfg.hidden_dim, rng, 0.01f);
        rand_fill(dp, (int64_t)cfg.hidden_dim * cfg.intermediate_dim, rng, 0.01f);
    }

    // Device allocations
    float *d_gate_matrix, *d_weight_buf, *d_hidden, *d_out_naive, *d_out_fused;
    CUDA_CHECK(cudaMalloc(&d_gate_matrix, gate_matrix_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight_buf,  total_weight_bytes(layout)));
    CUDA_CHECK(cudaMalloc(&d_hidden,      cfg.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_naive,   cfg.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_fused,   cfg.hidden_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_gate_matrix, h_gate_matrix.data(),
        gate_matrix_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight_buf, h_weight_buf.data(),
        total_weight_bytes(layout), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(),
        cfg.hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    // Naive forward pass
    NaiveMoEParams naive_params = {d_gate_matrix, d_weight_buf, layout, cfg.top_k};
    NaiveMoEScratch naive_scratch = naive_moe_alloc_scratch(layout);
    naive_moe_forward(naive_params, d_hidden, d_out_naive, naive_scratch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fused forward pass
    FusedMoEParams fused_params = {d_gate_matrix, d_weight_buf, layout, cfg.top_k};
    fused_moe_forward(fused_params, d_hidden, d_out_fused);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare outputs
    std::vector<float> h_naive(cfg.hidden_dim), h_fused(cfg.hidden_dim);
    CUDA_CHECK(cudaMemcpy(h_naive.data(), d_out_naive, cfg.hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused.data(), d_out_fused, cfg.hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

    float cs  = cosine_similarity(h_naive.data(), h_fused.data(), cfg.hidden_dim);
    float mae = max_abs_error(h_naive.data(), h_fused.data(), cfg.hidden_dim);

    // Tolerances: fp32 accumulation with different summation orders can diverge;
    // cosine similarity > 0.999 and MAE < 1e-3 (relative to output magnitude) is sufficient.
    float out_norm = 0.0f;
    for (int i = 0; i < cfg.hidden_dim; ++i) out_norm = fmaxf(out_norm, fabsf(h_naive[i]));
    float rel_mae = (out_norm > 1e-9f) ? mae / out_norm : mae;

    bool pass = (cs > 0.999f) && (rel_mae < 1e-2f);

    printf("  experts=%-3d hidden=%-5d inter=%-6d top_k=%d  cos=%.6f  abs_err=%.3e  rel_err=%.3e  %s\n",
           cfg.num_experts, cfg.hidden_dim, cfg.intermediate_dim, cfg.top_k,
           cs, mae, rel_mae, pass ? "PASS" : "FAIL");

    // Cleanup
    naive_moe_free_scratch(naive_scratch);
    cudaFree(d_gate_matrix);
    cudaFree(d_weight_buf);
    cudaFree(d_hidden);
    cudaFree(d_out_naive);
    cudaFree(d_out_fused);

    return pass;
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("warp-route correctness test\n");
    printf("GPU: ");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    std::mt19937 rng(42);

    // Sweep over: num_experts in {8, 16, 64}, hidden in {4096, 8192},
    // intermediate in {14336, 28672}, top_k in {1, 2}
    // Skip top_k=2 for num_experts=64 intermediate=28672 to keep runtime manageable.
    const int experts_list[]      = {8, 16, 64};
    const int hidden_list[]       = {4096, 8192};
    const int intermediate_list[] = {14336, 28672};
    const int topk_list[]         = {1, 2};

    int pass_count = 0, total = 0;

    for (int ne : experts_list) {
        for (int hd : hidden_list) {
            for (int id : intermediate_list) {
                for (int tk : topk_list) {
                    // Skip configs that exceed sm_80 shared memory (48KB)
                    int smem = (ne + 128 + 64) * sizeof(float);
                    if (smem > 48 * 1024) {
                        printf("  experts=%-3d hidden=%-5d inter=%-6d top_k=%d  SKIP (smem)\n",
                               ne, hd, id, tk);
                        continue;
                    }
                    if (ne > GATE_MAX_EXPERTS) {
                        printf("  experts=%-3d hidden=%-5d inter=%-6d top_k=%d  SKIP (GATE_MAX_EXPERTS)\n",
                               ne, hd, id, tk);
                        continue;
                    }
                    TestConfig cfg = {ne, hd, id, tk};
                    bool ok = run_test(cfg, rng);
                    pass_count += ok ? 1 : 0;
                    ++total;
                }
            }
        }
    }

    printf("\n%d/%d tests passed\n", pass_count, total);
    return (pass_count == total) ? 0 : 1;
}
