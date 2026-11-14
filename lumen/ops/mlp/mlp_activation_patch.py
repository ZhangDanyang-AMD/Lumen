###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLP forwards that call the fused ``silu_and_mul`` kernel."""

from __future__ import annotations

from lumen.ops.silu_and_mul import silu_and_mul_fn


def _rank0_print(msg: str) -> None:
    from lumen.config import _rank0_print as _print

    _print(msg)


def patch_mlp_activation(model) -> None:
    import torch

    silu_fn = silu_and_mul_fn()

    hf_count = 0
    for _name, module in model.named_modules():
        if (hasattr(module, "gate_proj") and hasattr(module, "up_proj")
                and hasattr(module, "down_proj") and hasattr(module, "act_fn")):

            def _make_mlp_forward(m):
                def _forward(x):
                    gate = m.gate_proj(x)
                    up = m.up_proj(x)
                    gate_up = torch.cat([gate, up], dim=-1)
                    hidden = silu_fn.apply(gate_up)
                    return m.down_proj(hidden)
                return _forward

            module.forward = _make_mlp_forward(module)
            hf_count += 1

    if hf_count:
        _rank0_print(f"> Replaced {hf_count} HF MLP forward with ATOM fused silu_and_mul")

    meg_count = 0
    try:
        from megatron.core.transformer.mlp import MLP as _MegatronMLP
    except ImportError:
        _MegatronMLP = None

    if _MegatronMLP is not None:
        for _name, module in model.named_modules():
            if not isinstance(module, _MegatronMLP):
                continue
            if not getattr(module.config, "gated_linear_unit", False):
                continue

            def _make_megatron_mlp_forward(m):
                def _forward(hidden_states, per_token_scale=None):
                    intermediate, bias = m.linear_fc1(hidden_states)
                    if bias is not None:
                        intermediate = intermediate + bias
                    intermediate = silu_fn.apply(intermediate)
                    output, output_bias = m.linear_fc2(intermediate)
                    return output, output_bias
                return _forward

            module.forward = _make_megatron_mlp_forward(module)
            meg_count += 1

    if meg_count:
        _rank0_print(f"> Replaced {meg_count} Megatron MLP forward with ATOM fused silu_and_mul")
