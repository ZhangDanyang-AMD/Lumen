"""Megatron-compatible wrapper around apex FusedRMSNorm.

Supports the TransformerConfig constructor signature and
sequence_parallel attribute on weights.
"""

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init

from apex.normalization.fused_layer_norm import (
    fused_rms_norm_affine,
    fused_rms_norm,
    manual_rms_norm,
)

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import make_viewless_tensor


class MegatronFusedRMSNorm(torch.nn.Module):
    """RMSNorm using apex fused kernels, compatible with Megatron layer specs.

    Accepts the same (config, hidden_size, eps, ...) signature that Megatron
    layer-spec build_module() passes, and sets sequence_parallel on weights
    so gradient all-reduce works correctly with SP.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ):
        super().__init__()
        self.config = config
        self.zero_centered_gamma = config.layernorm_zero_centered_gamma

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps

        self.weight = Parameter(torch.empty(*self.hidden_size))
        self.reset_parameters()

        self.sequence_parallel = config.sequence_parallel
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self):
        if self.zero_centered_gamma:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        if not input.is_cuda:
            return manual_rms_norm(input, self.hidden_size, weight, self.eps)

        output = fused_rms_norm_affine(input, weight, self.hidden_size, self.eps, False)
        return make_viewless_tensor(
            inp=output, requires_grad=input.requires_grad, keep_graph=True
        )
