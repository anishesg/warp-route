from __future__ import annotations

import functools
from typing import List, Optional

import torch


@functools.lru_cache(maxsize=1)
def _load_extension():
    try:
        import warp_route._C as _C
        return _C
    except ImportError:
        raise ImportError(
            "warp_route._C extension not found. "
            "Build it with: pip install -e . (requires CUDA toolkit and PyTorch)"
        )


def _pack_expert_weights(
    gate_proj: List[torch.Tensor],
    up_proj: List[torch.Tensor],
    down_proj: List[torch.Tensor],
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> torch.Tensor:
    """Pack per-expert weight tensors into the flat buffer layout the CUDA kernels expect.

    Layout per expert (all row-major, float32):
      [gate_proj: intermediate x hidden]
      [up_proj:   intermediate x hidden]
      [down_proj: hidden x intermediate]
    Each expert block is padded to a 128-byte boundary.

    Returns a contiguous CUDA float32 tensor.
    """
    gate_up_elems = intermediate_dim * hidden_dim
    down_elems    = hidden_dim * intermediate_dim
    raw_stride    = (gate_up_elems * 2 + down_elems) * 4  # bytes, float32
    expert_stride = (raw_stride + 127) & ~127              # align to 128 bytes
    expert_stride_floats = expert_stride // 4

    device = gate_proj[0].device
    buf = torch.zeros(num_experts * expert_stride_floats, dtype=torch.float32, device=device)

    for e in range(num_experts):
        base = e * expert_stride_floats
        g = gate_proj[e].contiguous().view(-1)
        u = up_proj[e].contiguous().view(-1)
        d = down_proj[e].contiguous().view(-1)
        buf[base                         : base + gate_up_elems] = g
        buf[base + gate_up_elems         : base + gate_up_elems * 2] = u
        buf[base + gate_up_elems * 2     : base + gate_up_elems * 2 + down_elems] = d

    return buf.contiguous()


def fused_moe(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    weight_buf: torch.Tensor,
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
) -> torch.Tensor:
    """Single-token fused MoE forward using the fused CUDA kernel.

    Args:
        x:               Input hidden state, shape [hidden_dim], float32, CUDA.
        gate_w:          Router weight matrix, shape [num_experts, hidden_dim], float32, CUDA.
        weight_buf:      Packed expert weights from _pack_expert_weights(), float32, CUDA.
        num_experts:     Number of experts.
        hidden_dim:      Hidden dimension.
        intermediate_dim: Intermediate (up-projection) dimension.
        top_k:           Number of experts to select (1 or 2).

    Returns:
        Output tensor, shape [hidden_dim], float32, CUDA.
    """
    _C = _load_extension()
    return _C.fused_moe_forward(
        x, gate_w, weight_buf,
        num_experts, hidden_dim, intermediate_dim, top_k
    )


def naive_moe(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    weight_buf: torch.Tensor,
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
) -> torch.Tensor:
    """Single-token naive multi-kernel MoE forward (correctness reference).

    Args:
        x:               Input hidden state, shape [hidden_dim], float32, CUDA.
        gate_w:          Router weight matrix, shape [num_experts, hidden_dim], float32, CUDA.
        weight_buf:      Packed expert weights from _pack_expert_weights(), float32, CUDA.
        num_experts:     Number of experts.
        hidden_dim:      Hidden dimension.
        intermediate_dim: Intermediate (up-projection) dimension.
        top_k:           Number of experts to select.

    Returns:
        Output tensor, shape [hidden_dim], float32, CUDA.
    """
    _C = _load_extension()
    return _C.naive_moe_forward(
        x, gate_w, weight_buf,
        num_experts, hidden_dim, intermediate_dim, top_k
    )


def batched_fused_moe(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    weight_buf: torch.Tensor,
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
) -> torch.Tensor:
    """Batched fused MoE forward for multiple decode tokens in one kernel launch.

    Args:
        x:               Input batch, shape [batch_size, hidden_dim], float32, CUDA.
        gate_w:          Router weight matrix, shape [num_experts, hidden_dim], float32, CUDA.
        weight_buf:      Packed expert weights from _pack_expert_weights(), float32, CUDA.
        num_experts:     Number of experts.
        hidden_dim:      Hidden dimension.
        intermediate_dim: Intermediate (up-projection) dimension.
        top_k:           Number of experts to select per token (1 or 2).

    Returns:
        Output tensor, shape [batch_size, hidden_dim], float32, CUDA.
    """
    if x.dim() != 2:
        raise ValueError(f"batched_fused_moe expects 2D input [batch, hidden], got shape {x.shape}")
    batch_size, hd = x.shape
    if hd != hidden_dim:
        raise ValueError(f"Input hidden dim {hd} does not match hidden_dim={hidden_dim}")

    _C = _load_extension()
    return _C.batched_fused_moe_forward(
        x, gate_w, weight_buf,
        num_experts, hidden_dim, intermediate_dim, top_k
    )
