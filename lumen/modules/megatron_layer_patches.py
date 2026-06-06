###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron-Core layer-level patches — norm, attention, MLP, FP8 linear.

These functions implement the actual module replacement logic.  They are
called from ``lumen.models.megatron`` builders and from
``LumenConfig.enable()`` but contain no model-building or CLI logic.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    logger.info(msg)


# ---------------------------------------------------------------------------
# Megatron-compatible norm wrappers
# ---------------------------------------------------------------------------


class MegatronCompatibleTLRMSNorm(torch.nn.Module):
    """Adapts :class:`LumenRMSNorm` to Megatron-Core's
    ``(config, hidden_size, eps=...)`` construction signature.
    """

    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        from lumen.ops.normalization import LumenRMSNorm

        self._norm = LumenRMSNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


class MegatronCompatibleTLLayerNorm(torch.nn.Module):
    """Adapts :class:`LumenLayerNorm` to Megatron-Core's
    ``(config, hidden_size, eps=...)`` construction signature.
    """

    def __init__(self, config, hidden_size, eps=1e-5, **kwargs):
        super().__init__()
        from lumen.ops.normalization import LumenLayerNorm

        self._norm = LumenLayerNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


class MegatronCompatibleTLNorm(torch.nn.Module):
    """Auto-detect norm type from Megatron config and dispatch."""

    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        norm_type = getattr(config, "normalization", "LayerNorm")
        if norm_type == "RMSNorm":
            from lumen.ops.normalization import LumenRMSNorm

            self._norm = LumenRMSNorm(hidden_size, eps=eps)
        else:
            from lumen.ops.normalization import LumenLayerNorm

            self._norm = LumenLayerNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


# ---------------------------------------------------------------------------
# Norm patching
# ---------------------------------------------------------------------------

_NORM_ATTRS = (
    "input_layernorm",
    "pre_mlp_layernorm",
    "pre_cross_attn_layernorm",
    "post_cross_attn_layernorm",
    "final_layernorm",
)


def patch_norms_in_spec(spec, norm_cls=None):
    """Replace all norm classes in a spec tree with Lumen norm modules.

    When *norm_cls* is ``None``, uses :class:`MegatronCompatibleTLNorm`
    which auto-detects RMSNorm vs LayerNorm from the Megatron config.

    IdentityOp placeholders are preserved and never replaced.
    """
    from megatron.core.transformer.identity_op import IdentityOp

    if norm_cls is None:
        norm_cls = MegatronCompatibleTLNorm

    for attr in _NORM_ATTRS:
        cur = getattr(spec, attr, None)
        if cur is not None and cur is not IdentityOp:
            setattr(spec, attr, norm_cls)

    if hasattr(spec, "submodules") and spec.submodules is not None:
        for attr in _NORM_ATTRS:
            cur = getattr(spec.submodules, attr, None)
            if cur is not None and cur is not IdentityOp:
                setattr(spec.submodules, attr, norm_cls)

    layer_specs = getattr(spec, "layer_specs", None)
    if layer_specs is None and hasattr(spec, "submodules"):
        layer_specs = getattr(spec.submodules, "layer_specs", None)
    if layer_specs:
        for layer_spec in layer_specs:
            patch_norms_in_spec(layer_spec, norm_cls)


def patch_rmsnorm(model, grad_quant_type=None):
    """Replace all Megatron-Core RMSNorm modules with :class:`LumenRMSNorm`."""
    from lumen.ops.normalization import LumenRMSNorm

    count = 0
    for _name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in (
                "RMSNorm",
                "MegatronRMSNorm",
                "TENorm",
                "_MegatronCompatibleTLRMSNorm",
                "MegatronCompatibleTLRMSNorm",
                "_MegatronCompatibleTLNorm",
                "MegatronCompatibleTLNorm",
            ):
                hidden_size = child.weight.shape[0]
                eps = getattr(child, "eps", getattr(child, "epsilon", 1e-6))
                replacement = LumenRMSNorm(
                    hidden_size,
                    eps=eps,
                    grad_quant_type=grad_quant_type,
                )
                replacement.weight.data.copy_(child.weight.data)
                setattr(module, attr_name, replacement)
                count += 1

    _rank0_print(f"> Replaced {count} RMSNorm modules with LumenRMSNorm")


def patch_layernorm(model, grad_quant_type=None):
    """Replace all Megatron-Core LayerNorm modules with :class:`LumenLayerNorm`."""
    from lumen.ops.normalization import LumenLayerNorm

    count = 0
    for _name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in (
                "LayerNorm",
                "FusedLayerNorm",
                "WrappedTorchNorm",
                "_MegatronCompatibleTLLayerNorm",
                "MegatronCompatibleTLLayerNorm",
                "_MegatronCompatibleTLNorm",
                "MegatronCompatibleTLNorm",
            ):
                hidden_size = child.weight.shape[0]
                eps = getattr(child, "eps", getattr(child, "epsilon", 1e-5))
                replacement = LumenLayerNorm(
                    hidden_size,
                    eps=eps,
                    grad_quant_type=grad_quant_type,
                )
                replacement.weight.data.copy_(child.weight.data)
                if hasattr(child, "bias") and child.bias is not None and replacement.bias is not None:
                    replacement.bias.data.copy_(child.bias.data)
                setattr(module, attr_name, replacement)
                count += 1

    _rank0_print(f"> Replaced {count} LayerNorm modules with LumenLayerNorm")


def patch_all_norms(model, normalization="RMSNorm", grad_quant_type=None):
    """Replace all norm modules with the appropriate Lumen implementation."""
    if normalization == "RMSNorm":
        patch_rmsnorm(model, grad_quant_type)
    else:
        patch_layernorm(model, grad_quant_type)


# ---------------------------------------------------------------------------
# Attention spec patching
# ---------------------------------------------------------------------------


def patch_core_attention(spec):
    """Recursively replace every ``core_attention`` in a spec tree
    with :class:`LumenDotProductAttention`."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    from lumen.modules.attention_megatron import LumenDotProductAttention

    if hasattr(spec, "submodules") and spec.submodules is not None:
        subs = spec.submodules
        if hasattr(subs, "self_attention") and subs.self_attention is not None:
            sa = subs.self_attention
            if hasattr(sa, "submodules") and sa.submodules is not None:
                sa_subs = sa.submodules
                if hasattr(sa_subs, "core_attention"):
                    sa_subs.core_attention = ModuleSpec(module=LumenDotProductAttention)
        if hasattr(subs, "layer_specs"):
            for layer_spec in subs.layer_specs:
                patch_core_attention(layer_spec)


def patch_mla_attention(spec):
    """Replace core_attention with :class:`LumenDotProductAttentionMLA`
    in MLA specs."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    from lumen.modules.attention_mla import LumenDotProductAttentionMLA

    if hasattr(spec, "submodules") and spec.submodules is not None:
        subs = spec.submodules
        if hasattr(subs, "self_attention") and subs.self_attention is not None:
            sa = subs.self_attention
            if hasattr(sa, "submodules") and sa.submodules is not None:
                sa_subs = sa.submodules
                if hasattr(sa_subs, "core_attention"):
                    sa_subs.core_attention = ModuleSpec(module=LumenDotProductAttentionMLA)
        if hasattr(subs, "layer_specs"):
            for layer_spec in subs.layer_specs:
                patch_mla_attention(layer_spec)


# ---------------------------------------------------------------------------
# MLP patching
# ---------------------------------------------------------------------------


def patch_fused_swiglu_mlp(model):
    """Patch Megatron MLP forward to use AITER fused SwiGLU when available.

    Only activates when: gated_linear_unit=True, no MLP bias, the AITER
    fused gated kernel is available, AND batch size M <= 64.
    """
    from lumen.ops.dispatch import _probe_aiter_fused_gated

    if not _probe_aiter_fused_gated():
        _rank0_print("WARNING: --lumen-fused-mlp requested but AITER fused gated kernel unavailable")
        return

    from megatron.core.transformer.mlp import MLP

    patched = 0
    for module in model.modules():
        if not isinstance(module, MLP):
            continue
        if not getattr(module.config, "gated_linear_unit", False):
            continue
        if getattr(module.config, "add_bias_linear", False):
            continue

        _orig_forward = module.forward

        def _make_fused_forward(mlp_module, orig_fwd):
            _w_down_cache = [None]

            def _fused_forward(hidden_states, per_token_scale=None):
                try:
                    from aiter.ops.triton.gemm.feed_forward import ff_a16w16_fused_gated

                    w_fc1 = mlp_module.linear_fc1.weight
                    w_fc2 = mlp_module.linear_fc2.weight

                    orig_shape = hidden_states.shape
                    x_2d = hidden_states.reshape(-1, orig_shape[-1]).contiguous()

                    M = x_2d.shape[0]
                    if M > 64:
                        return orig_fwd(hidden_states, per_token_scale=per_token_scale)

                    x_bf16 = x_2d.bfloat16() if x_2d.dtype != torch.bfloat16 else x_2d
                    w1_bf16 = w_fc1.bfloat16() if w_fc1.dtype != torch.bfloat16 else w_fc1

                    w2_data = w_fc2.data if not hasattr(w_fc2, "data") else w_fc2
                    w2_bf16 = w2_data.bfloat16() if w2_data.dtype != torch.bfloat16 else w2_data
                    if _w_down_cache[0] is None or _w_down_cache[0].data_ptr() != w2_bf16.data_ptr():
                        _w_down_cache[0] = w2_bf16.t().contiguous()
                    w_down = _w_down_cache[0]

                    out = ff_a16w16_fused_gated(
                        x_bf16,
                        w1_bf16,
                        w_down,
                        dtype=torch.bfloat16,
                        activation="silu",
                    )
                    out = out.reshape(orig_shape[:-1] + (out.shape[-1],))
                    return out, None
                except Exception:
                    return orig_fwd(hidden_states, per_token_scale=per_token_scale)

            return _fused_forward

        module.forward = _make_fused_forward(module, _orig_forward)
        patched += 1

    _rank0_print(f"Patched {patched} MLP modules with AITER fused SwiGLU forward")


# ---------------------------------------------------------------------------
# Cross-entropy patching
# ---------------------------------------------------------------------------

_cross_entropy_patched = False


def patch_cross_entropy():
    """Monkey-patch Megatron's cross-entropy to use Lumen's Triton kernel."""
    global _cross_entropy_patched
    if _cross_entropy_patched:
        return

    from lumen.modules.cross_entropy import lumen_parallel_cross_entropy

    try:
        import megatron.core.models.common.language_module.language_module as _lm_mod

        _lm_mod.te_parallel_cross_entropy = lumen_parallel_cross_entropy
    except (ImportError, AttributeError):
        pass

    try:
        import megatron.core.extensions.transformer_engine as _te_ext

        _te_ext.te_parallel_cross_entropy = lumen_parallel_cross_entropy
    except (ImportError, AttributeError):
        pass

    def _vocab_parallel_ce_adapter(logits, labels, label_smoothing=0.0):
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()
        return lumen_parallel_cross_entropy(logits, labels, tp_group)

    try:
        import megatron.core.tensor_parallel as _tp_mod
        import megatron.core.tensor_parallel.cross_entropy as _ce_mod

        _tp_mod.vocab_parallel_cross_entropy = _vocab_parallel_ce_adapter
        _ce_mod.vocab_parallel_cross_entropy = _vocab_parallel_ce_adapter
    except (ImportError, AttributeError):
        pass

    _cross_entropy_patched = True
    _rank0_print("> Patched cross-entropy with Lumen Triton kernel")


# ---------------------------------------------------------------------------
# FP8 parallel linear enablement
# ---------------------------------------------------------------------------


def enable_fp8_for_parallel_linear(
    model,
    scaling_manager=None,
    scaling_type="dynamic",
    fp8_dtype=None,
    block_size=None,
    fp8_mha=False,
    gradient_accumulation_fusion=False,
    delay_wgrad=False,
    quant_config=None,
):
    """Enable FP8 GEMM on all Lumen parallel linear modules in the model.

    When *fp8_mha* is True, a shared :class:`Blockwise2DScaleManager` is
    attached to each ``LumenDotProductAttention`` (or MLA variant).

    When *quant_config* is provided, per-module ScalingManagers are created
    with this config.
    """
    from lumen.modules.attention_megatron import LumenDotProductAttention
    from lumen.modules.attention_mla import LumenDotProductAttentionMLA
    from lumen.modules.grouped_linear import LumenGroupedLinear
    from lumen.modules.layernorm_linear import LumenLayerNormLinear
    from lumen.modules.parallel_linear import LumenColumnParallelLinear, LumenRowParallelLinear

    if fp8_dtype is None:
        from lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()

    count = 0
    for module in model.modules():
        if isinstance(
            module, (LumenColumnParallelLinear, LumenRowParallelLinear, LumenLayerNormLinear, LumenGroupedLinear)
        ):
            _mgr = scaling_manager
            if _mgr is None and quant_config is not None:
                from lumen.quantize import ScalingManager

                _mgr = ScalingManager(quant_config)
            module.enable_fp8(
                scaling_manager=_mgr,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
            )
            if hasattr(module, "gradient_accumulation_fusion"):
                module.gradient_accumulation_fusion = gradient_accumulation_fusion
            if hasattr(module, "delay_wgrad"):
                module.delay_wgrad = delay_wgrad
            count += 1

    if fp8_mha:
        from lumen.quantize.scaling_manager import Blockwise2DScaleManager

        attn_count = 0
        for module in model.modules():
            if isinstance(module, (LumenDotProductAttention, LumenDotProductAttentionMLA)):
                module.scale_manager = Blockwise2DScaleManager()
                attn_count += 1
        if attn_count > 0:
            _rank0_print(f"> Attached Blockwise2DScaleManager to {attn_count} attention modules for FP8 MHA")

    if count > 0:
        _rank0_print(f"> Enabled FP8 (scaling={scaling_type}) on {count} Lumen parallel linear modules")
