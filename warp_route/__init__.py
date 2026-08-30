from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .functional import fused_moe, naive_moe, batched_fused_moe, _pack_expert_weights


class MoELayer(nn.Module):
    """Single MoE layer with per-expert SwiGLU FFN blocks.

    Holds expert weights as separate tensors and dispatches to the fused
    CUDA kernel via the functional API.

    Args:
        num_experts:      Total number of experts.
        hidden_dim:       Hidden state dimension (input and output).
        intermediate_dim: Per-expert intermediate (up-projection) dimension.
        top_k:            Number of experts to activate per token (1 or 2).
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if top_k not in (1, 2):
            raise ValueError(f"top_k must be 1 or 2 for the fused kernel, got {top_k}")

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.top_k = top_k

        # Router: maps hidden -> expert logits
        self.router = nn.Parameter(torch.empty(num_experts, hidden_dim))

        # Per-expert projection weights stored as parameter lists so they
        # participate in state_dict / optimizer.
        self.gate_proj = nn.ParameterList(
            [nn.Parameter(torch.empty(intermediate_dim, hidden_dim)) for _ in range(num_experts)]
        )
        self.up_proj = nn.ParameterList(
            [nn.Parameter(torch.empty(intermediate_dim, hidden_dim)) for _ in range(num_experts)]
        )
        self.down_proj = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, intermediate_dim)) for _ in range(num_experts)]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.router, std=0.02)
        for e in range(self.num_experts):
            nn.init.normal_(self.gate_proj[e], std=0.02)
            nn.init.normal_(self.up_proj[e], std=0.02)
            nn.init.normal_(self.down_proj[e], std=0.02)

    def _pack_weights(self) -> torch.Tensor:
        """Pack per-expert weights into the flat layout expected by the CUDA kernel."""
        gate_list = [p.data for p in self.gate_proj]
        up_list   = [p.data for p in self.up_proj]
        down_list = [p.data for p in self.down_proj]
        return _pack_expert_weights(gate_list, up_list, down_list,
                                    self.num_experts, self.hidden_dim, self.intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run MoE forward pass.

        Args:
            x: Input tensor of shape [hidden_dim] (single decode token) or
               [batch_size, hidden_dim] for batched decode.

        Returns:
            Output tensor with the same shape as x.
        """
        weight_buf = self._pack_weights()
        router_w = self.router.data.contiguous()

        if x.dim() == 1:
            return fused_moe(
                x.contiguous(),
                router_w,
                weight_buf,
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
                self.top_k,
            )
        elif x.dim() == 2:
            return batched_fused_moe(
                x.contiguous(),
                router_w,
                weight_buf,
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
                self.top_k,
            )
        else:
            raise ValueError(f"Input must be 1D or 2D, got shape {x.shape}")
