###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Lumen spec provider for Megatron-Core layer specs.

Every ``BackendSpecProvider`` method returns a Lumen module class.
Plug this into ``get_gpt_layer_with_transformer_engine_spec`` or compose
your own ``ModuleSpec`` tree to get Lumen-accelerated attention, norms,
and (optionally FP8) linear layers.

``grouped_mlp_modules`` must return an ``ExpertsBuilder`` (a callable /
``functools.partial``), matching Megatron-Core's current protocol. Returning
a ``(cls, submodules)`` tuple is rejected by ``MoELayer``.
"""

from functools import partial
from typing import Optional

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP

from lumen.modules.attention_megatron import LumenDotProductAttention
from lumen.modules.grouped_linear import (
    LumenColumnParallelGroupedLinear,
    LumenRowParallelGroupedLinear,
)
from lumen.modules.layernorm_linear import LumenLayerNormLinear
from lumen.modules.parallel_linear import LumenColumnParallelLinear, LumenRowParallelLinear


class _LumenNorm:
    """Factory that returns the correct Lumen norm wrapper class based on
    the ``normalization`` config attribute (``"RMSNorm"`` or ``"LayerNorm"``)."""

    def __new__(cls, config, hidden_size, eps=1e-6, **kwargs):
        norm_type = getattr(config, "normalization", "LayerNorm")
        sequence_parallel = bool(getattr(config, "sequence_parallel", False))
        if norm_type == "RMSNorm":
            from lumen.ops.normalization import LumenRMSNorm

            return LumenRMSNorm(hidden_size, eps=eps, sequence_parallel=sequence_parallel)
        else:
            from lumen.ops.normalization import LumenLayerNorm

            return LumenLayerNorm(hidden_size, eps=eps, sequence_parallel=sequence_parallel)


class LumenSpecProvider(BackendSpecProvider):
    """Provides Lumen module classes for Megatron-Core layer specs."""

    def linear(self):
        return LumenColumnParallelLinear

    def column_parallel_linear(self):
        return LumenColumnParallelLinear

    def row_parallel_linear(self):
        return LumenRowParallelLinear

    def fuse_layernorm_and_linear(self):
        return True

    def column_parallel_layer_norm_linear(self):
        return LumenLayerNormLinear

    def layer_norm(self, rms_norm=False, for_qk=False, has_residual=False, **kwargs):
        return _LumenNorm

    def core_attention(self):
        return LumenDotProductAttention

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool = False, **kwargs):
        """Return an ExpertsBuilder, not a (cls, submodules) tuple."""
        act = self.activation_func()
        if moe_use_grouped_gemm:
            try:
                from megatron.core.transformer.moe.experts import (
                    GroupedMLPSubmodules,
                    TEGroupedMLP,
                )

                return partial(
                    TEGroupedMLP,
                    submodules=GroupedMLPSubmodules(
                        linear_fc1=LumenColumnParallelGroupedLinear,
                        linear_fc2=LumenRowParallelGroupedLinear,
                        activation_func=act,
                    ),
                )
            except ImportError:
                pass

        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=LumenColumnParallelLinear,
                linear_fc2=LumenRowParallelLinear,
                activation_func=act,
            ),
        )

    def activation_func(self):
        return None
