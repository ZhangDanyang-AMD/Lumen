"""Lumen backend spec provider for DSV4 transformer blocks (MoE/MLP, no TE)."""

from megatron.core.transformer.moe.experts import TEGroupedMLP

from lumen.models.dsv4.megatron.layers import (
    LumenColumnParallelGroupedLinear,
    LumenColumnParallelLinear,
    LumenRowParallelGroupedLinear,
    LumenRowParallelLinear,
)
from lumen.models.spec_provider import LumenSpecProvider


class LumenDSV4SpecProvider(LumenSpecProvider):
    """Backend for DSV4 MoE/MLP/dense layers — Lumen linear/norm with tuned BF16 GEMM."""

    def column_parallel_linear(self):
        return LumenColumnParallelLinear

    def row_parallel_linear(self):
        return LumenRowParallelLinear

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool = False, **kwargs):
        from functools import partial

        act = self.activation_func()
        if moe_use_grouped_gemm:
            from megatron.core.transformer.moe.experts import GroupedMLPSubmodules

            return partial(
                TEGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=LumenColumnParallelGroupedLinear,
                    linear_fc2=LumenRowParallelGroupedLinear,
                    activation_func=act,
                ),
            )
        return super().grouped_mlp_modules(moe_use_grouped_gemm, **kwargs)
