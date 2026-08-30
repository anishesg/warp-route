#include <torch/extension.h>
#include <cuda_runtime.h>

#include "expert_layout.cuh"
#include "naive_moe.cuh"
#include "fused_moe.cuh"

#define CHECK_CONTIGUOUS(t) TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")
#define CHECK_CUDA(t)       TORCH_CHECK((t).is_cuda(),       #t " must be a CUDA tensor")
#define CHECK_FLOAT(t)      TORCH_CHECK((t).scalar_type() == torch::kFloat32, #t " must be float32")

static void validate_1d(const torch::Tensor& t, const char* name, int expected) {
    CHECK_CUDA(t); CHECK_CONTIGUOUS(t); CHECK_FLOAT(t);
    TORCH_CHECK(t.dim() == 1 && t.size(0) == expected,
        name, " must have shape [", expected, "], got shape ", t.sizes());
}

static void validate_2d(const torch::Tensor& t, const char* name, int rows, int cols) {
    CHECK_CUDA(t); CHECK_CONTIGUOUS(t); CHECK_FLOAT(t);
    TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols,
        name, " must have shape [", rows, ", ", cols, "], got shape ", t.sizes());
}

// naive_moe_forward(hidden, gate_w, weight_buf, num_experts, hidden_dim, intermediate_dim, top_k)
torch::Tensor naive_moe_forward_torch(
    torch::Tensor hidden,
    torch::Tensor gate_w,
    torch::Tensor weight_buf,
    int num_experts,
    int hidden_dim,
    int intermediate_dim,
    int top_k)
{
    validate_1d(hidden, "hidden", hidden_dim);
    validate_2d(gate_w, "gate_w", num_experts, hidden_dim);
    CHECK_CUDA(weight_buf); CHECK_CONTIGUOUS(weight_buf); CHECK_FLOAT(weight_buf);

    ExpertLayoutConfig layout{num_experts, hidden_dim, intermediate_dim};
    int64_t expected_weight_bytes = total_weight_bytes(layout);
    TORCH_CHECK(weight_buf.numel() * (int64_t)sizeof(float) >= expected_weight_bytes,
        "weight_buf too small: need ", expected_weight_bytes, " bytes, got ",
        weight_buf.numel() * sizeof(float));

    auto output = torch::zeros({hidden_dim}, hidden.options());

    NaiveMoEParams params{
        gate_w.data_ptr<float>(),
        weight_buf.data_ptr<float>(),
        layout,
        top_k
    };
    NaiveMoEScratch scratch = naive_moe_alloc_scratch(layout);
    naive_moe_forward(params, hidden.data_ptr<float>(), output.data_ptr<float>(), scratch);
    cudaDeviceSynchronize();
    naive_moe_free_scratch(scratch);

    return output;
}

// fused_moe_forward(hidden, gate_w, weight_buf, num_experts, hidden_dim, intermediate_dim, top_k)
torch::Tensor fused_moe_forward_torch(
    torch::Tensor hidden,
    torch::Tensor gate_w,
    torch::Tensor weight_buf,
    int num_experts,
    int hidden_dim,
    int intermediate_dim,
    int top_k)
{
    validate_1d(hidden, "hidden", hidden_dim);
    validate_2d(gate_w, "gate_w", num_experts, hidden_dim);
    CHECK_CUDA(weight_buf); CHECK_CONTIGUOUS(weight_buf); CHECK_FLOAT(weight_buf);

    TORCH_CHECK(top_k >= 1 && top_k <= 2, "top_k must be 1 or 2 for fused kernel, got ", top_k);
    TORCH_CHECK(num_experts <= GATE_MAX_EXPERTS,
        "num_experts ", num_experts, " exceeds GATE_MAX_EXPERTS ", GATE_MAX_EXPERTS);

    ExpertLayoutConfig layout{num_experts, hidden_dim, intermediate_dim};
    int64_t expected_weight_bytes = total_weight_bytes(layout);
    TORCH_CHECK(weight_buf.numel() * (int64_t)sizeof(float) >= expected_weight_bytes,
        "weight_buf too small: need ", expected_weight_bytes, " bytes, got ",
        weight_buf.numel() * sizeof(float));

    auto output = torch::zeros({hidden_dim}, hidden.options());

    FusedMoEParams params{
        gate_w.data_ptr<float>(),
        weight_buf.data_ptr<float>(),
        layout,
        top_k
    };
    fused_moe_forward(params, hidden.data_ptr<float>(), output.data_ptr<float>());
    cudaDeviceSynchronize();

    return output;
}

// batched_fused_moe_forward(hidden_batch, gate_w, weight_buf, num_experts, hidden_dim, intermediate_dim, top_k)
// hidden_batch: [batch_size, hidden_dim], output: [batch_size, hidden_dim]
torch::Tensor batched_fused_moe_forward_torch(
    torch::Tensor hidden_batch,
    torch::Tensor gate_w,
    torch::Tensor weight_buf,
    int num_experts,
    int hidden_dim,
    int intermediate_dim,
    int top_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_moe_forward", &naive_moe_forward_torch,
          "Naive multi-kernel MoE forward (float32)");
    m.def("fused_moe_forward", &fused_moe_forward_torch,
          "Fused single-kernel MoE forward (float32)");
    m.def("batched_fused_moe_forward", &batched_fused_moe_forward_torch,
          "Batched fused MoE forward for B tokens (float32)");
}
