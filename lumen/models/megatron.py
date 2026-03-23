###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared Megatron-LM training components for Lumen models.

This module consolidates the building blocks that are common to all
Megatron-LM-based training scripts (LLaMA2 SFT, LLaMA 3.1 pretraining, etc.).

Model-specific code (batch construction, dataset providers, model-specific CLI
arguments) remains in the per-model subpackages.
"""

import logging
from functools import partial
from typing import Callable, Optional

import torch

# ---------------------------------------------------------------------------
# FusedLayerNorm patch (must run before any Megatron module imports)
# ---------------------------------------------------------------------------


def _install_fused_layer_norm_patch():
    """Patch FusedLayerNorm to support both RMSNorm and LayerNorm before
    any Megatron module imports it.  Must run before GPTModel/TransformerBlock import."""
    from megatron.core.fusions import fused_layer_norm as _fln_mod

    from lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

    class _FusedLayerNormCompat(torch.nn.Module):
        """Dispatches to Lumen's RMSNorm or LayerNorm based on config."""

        def __new__(cls, config, hidden_size, eps=1e-6, **kwargs):
            norm_type = getattr(config, "normalization", "LayerNorm")
            if norm_type == "RMSNorm":
                return object.__new__(cls)
            # Use Lumen LayerNorm for standard LayerNorm too
            return object.__new__(cls)

        def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
            super().__init__()
            norm_type = getattr(config, "normalization", "LayerNorm")
            if norm_type == "RMSNorm":
                self._norm = LumenRMSNorm(hidden_size, eps=eps)
            else:
                self._norm = LumenLayerNorm(hidden_size, eps=eps)
            self.weight = self._norm.weight

        def forward(self, x):
            return self._norm(x)

    _FusedLayerNormCompat.__name__ = "FusedLayerNorm"
    _fln_mod.FusedLayerNorm = _FusedLayerNormCompat


_install_fused_layer_norm_patch()

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from lumen.models.utils import safe_add_argument
from lumen.modules.attention_megatron import (
    LumenDotProductAttention,
)
from lumen.modules.attention_mla import LumenDotProductAttentionMLA

logger = logging.getLogger(__name__)


def _patch_moe_fused_router():
    """Monkey-patch Megatron-Core's moe_utils with Lumen fused router ops."""
    try:
        import megatron.core.transformer.moe.moe_utils as moe_utils

        from lumen.ops.moe.fused_router import (
            fused_compute_score_for_moe_aux_loss,
            fused_moe_aux_loss,
            fused_topk_with_score_function,
        )

        moe_utils.fused_topk_with_score_function = fused_topk_with_score_function
        moe_utils.fused_compute_score_for_moe_aux_loss = fused_compute_score_for_moe_aux_loss
        moe_utils.fused_moe_aux_loss = fused_moe_aux_loss

        try:
            import megatron.core.extensions.transformer_engine as te_ext

            te_ext.fused_topk_with_score_function = fused_topk_with_score_function
            te_ext.fused_compute_score_for_moe_aux_loss = fused_compute_score_for_moe_aux_loss
            te_ext.fused_moe_aux_loss = fused_moe_aux_loss
        except ImportError:
            pass

        logger.info("Patched Megatron-Core moe_utils with Lumen fused router ops")
    except ImportError:
        logger.debug("Megatron-Core moe_utils not found, skipping MoE router patch")


_patch_moe_fused_router()

stimer = StragglerDetector()


# ---------------------------------------------------------------------------
# Layer-spec patching helpers
# ---------------------------------------------------------------------------


def _patch_core_attention(spec):
    """Recursively walk a ModuleSpec tree and replace every ``core_attention``
    submodule with ``LumenDotProductAttention``."""
    from megatron.core.transformer.spec_utils import ModuleSpec

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
                _patch_core_attention(layer_spec)


class _MegatronCompatibleTLRMSNorm(torch.nn.Module):
    """Wrapper that adapts :class:`LumenRMSNorm` to the Megatron-Core
    norm construction signature ``(config, hidden_size, eps=...)``.

    Unlike ``WrappedTorchNorm``, this does **not** assert on
    ``persist_layer_norm`` or ``sequence_parallel`` — RMSNorm normalises
    over the hidden dimension only, so each position is independent and
    works correctly with SP-scattered inputs.
    """

    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        from lumen.ops.normalization import LumenRMSNorm

        self._norm = LumenRMSNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


class _MegatronCompatibleTLLayerNorm(torch.nn.Module):
    """Wrapper that adapts :class:`LumenLayerNorm` to the Megatron-Core
    norm construction signature ``(config, hidden_size, eps=...)``."""

    def __init__(self, config, hidden_size, eps=1e-5, **kwargs):
        super().__init__()
        from lumen.ops.normalization import LumenLayerNorm

        self._norm = LumenLayerNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


class _MegatronCompatibleTLNorm(torch.nn.Module):
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


_NORM_ATTRS = (
    "input_layernorm",
    "pre_mlp_layernorm",
    "pre_cross_attn_layernorm",
    "post_cross_attn_layernorm",
    "final_layernorm",
)


def _patch_norms_in_spec(spec, norm_cls=None):
    """Replace **all** norm classes in a spec tree with Lumen norm modules.

    When *norm_cls* is ``None``, uses :class:`_MegatronCompatibleTLNorm`
    which auto-detects RMSNorm vs LayerNorm from the Megatron config.

    Handles both:
    - Block-level specs (``TransformerBlockSubmodules`` with
      ``layer_specs`` and ``final_layernorm``)
    - Layer-level specs (``ModuleSpec`` with ``submodules``)
    """
    if norm_cls is None:
        norm_cls = _MegatronCompatibleTLNorm

    for attr in _NORM_ATTRS:
        if getattr(spec, attr, None) is not None:
            setattr(spec, attr, norm_cls)

    if hasattr(spec, "submodules") and spec.submodules is not None:
        for attr in _NORM_ATTRS:
            if getattr(spec.submodules, attr, None) is not None:
                setattr(spec.submodules, attr, norm_cls)

    layer_specs = getattr(spec, "layer_specs", None)
    if layer_specs is None and hasattr(spec, "submodules"):
        layer_specs = getattr(spec.submodules, "layer_specs", None)
    if layer_specs:
        for layer_spec in layer_specs:
            _patch_norms_in_spec(layer_spec, norm_cls)


def _patch_rmsnorm(model, grad_quant_type=None):
    """Replace all Megatron-Core RMSNorm modules with Lumen's
    Triton-accelerated :class:`LumenRMSNorm`."""
    from lumen.ops.normalization import LumenRMSNorm

    count = 0
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in (
                "RMSNorm",
                "MegatronRMSNorm",
                "TENorm",
                "_MegatronCompatibleTLRMSNorm",
                "_MegatronCompatibleTLNorm",
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

    print_rank_0(f"> Replaced {count} RMSNorm modules with LumenRMSNorm")


def _patch_layernorm(model, grad_quant_type=None):
    """Replace all Megatron-Core LayerNorm modules with Lumen's
    :class:`LumenLayerNorm`."""
    from lumen.ops.normalization import LumenLayerNorm

    count = 0
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in (
                "LayerNorm",
                "FusedLayerNorm",
                "WrappedTorchNorm",
                "_MegatronCompatibleTLLayerNorm",
                "_MegatronCompatibleTLNorm",
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

    print_rank_0(f"> Replaced {count} LayerNorm modules with LumenLayerNorm")


def _patch_all_norms(model, normalization="RMSNorm", grad_quant_type=None):
    """Replace all norm modules with the appropriate Lumen implementation."""
    if normalization == "RMSNorm":
        _patch_rmsnorm(model, grad_quant_type)
    else:
        _patch_layernorm(model, grad_quant_type)


# ---------------------------------------------------------------------------
# Override defaults for Lumen
# ---------------------------------------------------------------------------

_TE_FORCE_OVERRIDES = {
    "transformer_impl": "local",
    "fp8_param_gather": False,
    "keep_fp8_weight_transpose_cache": False,
    "deprecated_keep_fp8_weight_transpose_cache": False,
    "fp4": None,
    "fp4_param": False,
    "te_rng_tracker": False,
    "inference_rng_tracker": False,
}

_FP8_FORMAT_MAP = {"e4m3": "fp8_e4m3", "hybrid": "hybrid"}

_BACKEND_MAP = {
    "auto": ("aiter_csrc", "aiter_triton_fp8"),
    "triton": ("aiter_triton", "aiter_triton_fp8"),
    "csrc": ("aiter_csrc", "aiter_csrc_fp8"),
    "asm": ("aiter_csrc", "aiter_asm_fp8"),
}


def resolve_attn_backend(backend: str, fp8_attn: str) -> str:
    """Derive the concrete ``aiter_*`` backend string from user-facing flags.

    Args:
        backend: One of ``auto``, ``triton``, ``csrc``, ``asm``.
        fp8_attn: One of ``none``, ``dpa``, ``mha``.

    Returns:
        A concrete backend name like ``aiter_triton_fp8``.
    """
    bf16_be, fp8_be = _BACKEND_MAP.get(backend, ("aiter_csrc", "aiter_triton_fp8"))
    return fp8_be if fp8_attn in ("dpa", "mha") else bf16_be


def _override_te_args_for_lumen(args):
    """Configure Lumen FP8 settings from Megatron args.

    The ``--fp8-format`` value (``args.fp8``) is mapped to the Lumen
    :class:`QuantFormat` string and stored as ``args.lumen_fp8_format`` for
    :func:`apply_fp8_training`.  ``args.fp8`` is then set to ``None`` so
    that ``TransformerConfig`` uses Lumen's own FP8 code-paths.

    All other shared parameters (``fp8_margin``, ``fp8_recipe``,
    ``fp8_amax_history_len``, ``fp8_amax_compute_algo``, ``fp8_wgrad``,
    ``first_last_layers_bf16``, etc.) are kept as-is.
    """
    te_fp8 = getattr(args, "fp8", None)
    if te_fp8 is not None:
        args.lumen_fp8_format = _FP8_FORMAT_MAP.get(te_fp8, te_fp8)
    args.fp8 = None

    for attr, value in _TE_FORCE_OVERRIDES.items():
        setattr(args, attr, value)

    fp8_attn = getattr(args, "lumen_fp8_attn", "none")
    if getattr(args, "fp8_multi_head_attention", False):
        fp8_attn = "mha"
    elif getattr(args, "fp8_dot_product_attention", False) and fp8_attn == "none":
        fp8_attn = "dpa"
    args.lumen_fp8_attn = fp8_attn

    backend_base = getattr(args, "lumen_attn_backend", "auto")
    args.lumen_attn_backend = resolve_attn_backend(backend_base, fp8_attn)

    if getattr(args, "lumen_cross_entropy", False):
        _patch_cross_entropy()


_cross_entropy_patched = False


def _patch_cross_entropy():
    """Monkey-patch Megatron's cross-entropy so the GPTModel loss computation
    goes through Lumen's Triton kernel.

    ``LanguageModule.compute_language_model_loss`` dispatches through
    ``te_parallel_cross_entropy`` when ``cross_entropy_loss_fusion`` is
    enabled.  We replace that symbol with ``lumen_parallel_cross_entropy``
    (same signature) and also wrap it as ``vocab_parallel_cross_entropy``
    for the non-fusion fallback path.
    """
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
    print_rank_0("> Patched cross-entropy with Lumen Triton kernel")


# ---------------------------------------------------------------------------
# Custom GPT builder that injects Lumen attention
# ---------------------------------------------------------------------------


def lumen_gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, model_name="GPT"):
    """Build a GPTModel with Lumen attention replacing the default
    DotProductAttention in every layer.

    When ``--lumen-linear`` is set, uses the Lumen spec provider for all
    linear, norm, and attention modules.  Otherwise, uses the Megatron-Core
    local spec and patches attention/norms post-hoc.

    Args:
        model_name: Label used in the startup log message (e.g. ``"LLaMA 3.1"``).
    """
    if getattr(args, "lumen_linear", False):
        return lumen_gpt_builder_with_spec(
            args,
            pre_process,
            post_process,
            vp_stage=vp_stage,
            config=config,
            model_name=model_name,
        )

    print_rank_0(f"building {model_name} model with Lumen attention ...")

    _override_te_args_for_lumen(args)

    if config is None:
        args.apply_rope_fusion = False
        config = core_transformer_config_from_args(args)
        config.persist_layer_norm = False
        config.bias_swiglu_fusion = False

    transformer_layer_spec = get_gpt_layer_local_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        args.qk_layernorm,
        args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
        normalization=args.normalization,
        use_kitchen=config.use_kitchen,
    )

    _patch_core_attention(transformer_layer_spec)
    _patch_norms_in_spec(transformer_layer_spec)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
    )

    grad_quant_type = getattr(args, "grad_quant_type", None)
    normalization = getattr(args, "normalization", "RMSNorm")

    if getattr(args, "lumen_rmsnorm", False) or getattr(args, "lumen_norm", False):
        _patch_all_norms(model, normalization, grad_quant_type)

    return model


def lumen_gpt_builder_with_spec(args, pre_process, post_process, vp_stage=None, config=None, model_name="GPT"):
    """Build a GPTModel using the Lumen spec provider.

    Instead of patching individual modules post-hoc, this builder uses
    :class:`~lumen.models.spec_provider.LumenSpecProvider` to produce a
    layer spec where *all* linear, norm, and attention modules are Lumen
    classes from the start.  This is the recommended path for full Lumen
    integration including FP8 parallel linear layers.
    """
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    from lumen.models.spec_provider import LumenSpecProvider

    print_rank_0(f"building {model_name} model with Lumen spec provider ...")

    _override_te_args_for_lumen(args)

    if config is None:
        args.apply_rope_fusion = False
        config = core_transformer_config_from_args(args)
        config.persist_layer_norm = False
        config.bias_swiglu_fusion = False

    # Monkey-patch the TE spec provider lookup so get_gpt_layer_with_transformer_engine_spec
    # picks up Lumen modules without modifying Megatron source.
    import megatron.core.models.gpt.gpt_layer_specs as _gls

    _orig_te_spec = getattr(_gls, "TESpecProvider", None)
    _gls.TESpecProvider = LumenSpecProvider
    _gls.HAVE_TE = True

    try:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=getattr(args, "num_experts", None),
            moe_grouped_gemm=getattr(args, "moe_grouped_gemm", False),
            qk_layernorm=getattr(args, "qk_layernorm", False),
            multi_latent_attention=getattr(args, "multi_latent_attention", False),
            moe_use_legacy_grouped_gemm=getattr(args, "moe_use_legacy_grouped_gemm", False),
            use_kitchen=getattr(config, "use_kitchen", False),
        )
    finally:
        if _orig_te_spec is not None:
            _gls.TESpecProvider = _orig_te_spec

    # Patch MLA attention if needed
    if getattr(args, "multi_latent_attention", False):
        _patch_mla_attention(transformer_layer_spec)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
    )

    return model


def _patch_mla_attention(spec):
    """Replace core_attention with LumenDotProductAttentionMLA in MLA specs."""
    from megatron.core.transformer.spec_utils import ModuleSpec

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
                _patch_mla_attention(layer_spec)


def enable_fp8_for_parallel_linear(
    model,
    scaling_manager=None,
    scaling_type="dynamic",
    fp8_dtype=None,
    block_size=None,
    fp8_mha=False,
    gradient_accumulation_fusion=False,
    delay_wgrad=False,
):
    """Enable FP8 GEMM on all Lumen parallel linear modules in the model.

    When *fp8_mha* is True, a shared :class:`Blockwise2DScaleManager` is
    attached to each ``LumenDotProductAttention`` (or MLA variant) so that
    QKV projection, dot-product attention and output projection share the
    same FP8 scale context within a single MHA block.

    When *delay_wgrad* is True, backward passes compute only dgrad
    immediately and defer wgrad to a later ``backward_dw()`` call.

    When *gradient_accumulation_fusion* is True, weight gradients
    accumulate directly into ``param.main_grad``.
    """
    from lumen.modules.attention_megatron import LumenDotProductAttention
    from lumen.modules.attention_mla import LumenDotProductAttentionMLA
    from lumen.modules.grouped_linear import LumenGroupedLinear
    from lumen.modules.layernorm_linear import LumenLayerNormLinear
    from lumen.modules.parallel_linear import LumenColumnParallelLinear, LumenRowParallelLinear

    count = 0
    for module in model.modules():
        if isinstance(
            module, (LumenColumnParallelLinear, LumenRowParallelLinear, LumenLayerNormLinear, LumenGroupedLinear)
        ):
            module.enable_fp8(
                scaling_manager=scaling_manager,
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
            print_rank_0(f"> Attached Blockwise2DScaleManager to {attn_count} attention modules for FP8 MHA")

    if count > 0:
        print_rank_0(f"> Enabled FP8 (scaling={scaling_type}) on {count} Lumen parallel linear modules")


# ---------------------------------------------------------------------------
# LoRA (Parameter-Efficient Fine-Tuning)
# ---------------------------------------------------------------------------


def apply_lora(model: GPTModel, args) -> None:
    """Wrap linear layers with LoRA adapters for parameter-efficient fine-tuning."""
    from megatron.core.transformer.lora_adapter import LoraAdapter

    common = {
        "config": model.config,
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
    }

    if hasattr(model, "embedding") and model.embedding is not None:
        model.embedding.word_embeddings = LoraAdapter(model.embedding.word_embeddings, **common)

    if hasattr(model, "decoder") and model.decoder is not None:
        for layer in model.decoder.layers:
            layer.self_attention.linear_qkv = LoraAdapter(layer.self_attention.linear_qkv, **common)
            layer.self_attention.linear_proj = LoraAdapter(layer.self_attention.linear_proj, **common)
            if hasattr(layer, "mlp") and layer.mlp is not None:
                layer.mlp.linear_fc1 = LoraAdapter(layer.mlp.linear_fc1, **common)
                layer.mlp.linear_fc2 = LoraAdapter(layer.mlp.linear_fc2, **common)

    if hasattr(model, "output_layer") and model.output_layer is not None:
        model.output_layer = LoraAdapter(model.output_layer, **common)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_rank_0(
        f"> LoRA applied (rank={args.lora_rank}, alpha={args.lora_alpha}) — "
        f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )


# ---------------------------------------------------------------------------
# FP8 quantised training
# ---------------------------------------------------------------------------


def apply_fp8_training(model: GPTModel, args) -> None:
    """Enable FP8 quantised training via Lumen's non-invasive patching."""
    import lumen.quantize as quant
    from lumen.quantize import (
        AmaxAlgo,
        QuantConfig,
        QuantFormat,
        ScalingType,
    )

    fmt = getattr(args, "lumen_fp8_format", "fp8_e4m3")
    scaling = getattr(args, "linear_fp8_scaling", "delayed")
    block_size = getattr(args, "linear_fp8_block_size", 128)
    amax_algo = getattr(args, "linear_fp8_amax_algo", "max")
    reduce_amax = getattr(args, "linear_fp8_reduce_amax", False)
    history_len = getattr(args, "linear_fp8_amax_history", 16)
    margin = getattr(args, "linear_fp8_margin", 0)
    quant_act = getattr(args, "linear_fp8_activation", True)
    fp8_wgrad = getattr(args, "linear_fp8_wgrad", True)
    grad_quant_type = getattr(args, "grad_quant_type", None)
    first_last_bf16 = getattr(args, "first_last_layers_bf16", False)
    bf16_start = getattr(args, "num_layers_at_start_in_bf16", 1)
    bf16_end = getattr(args, "num_layers_at_end_in_bf16", 1)
    num_layers = getattr(args, "num_layers", 0)
    use_sdma = getattr(args, "use_sdma", False)
    fp8_attn = getattr(args, "lumen_fp8_attn", "none")
    fp8_dpa = fp8_attn in ("dpa", "mha")
    fp8_mha = fp8_attn == "mha"

    print_rank_0(
        f"> transformer_impl='Aiter Backend', fp8_format='{fmt}', \
        fp8_scaling='{scaling}', \
        fp8_block_size='{block_size}', \
        fp8_amax_algo='{amax_algo}', \
        fp8_reduce_amax='{reduce_amax}', \
        fp8_amax_history='{history_len}', \
        fp8_activation='{quant_act}', \
        fp8_wgrad='{fp8_wgrad}', \
        grad_quant='{grad_quant_type}', \
        first_last_bf16='{first_last_bf16}' (start={bf16_start}, end={bf16_end}), \
        use_sdma='{use_sdma}', \
        fp8_attn='{fp8_attn}'"
    )

    config = QuantConfig(
        format=QuantFormat(fmt),
        scaling=ScalingType(scaling),
        block_size=block_size,
        amax_algo=AmaxAlgo(amax_algo),
        margin=margin,
        reduce_amax=reduce_amax,
        history_len=history_len,
        quantize_activation=quant_act,
        fp8_wgrad=fp8_wgrad,
        quantize_grad=grad_quant_type,
        first_last_layers_bf16=first_last_bf16,
        num_layers_at_start_in_bf16=bf16_start,
        num_layers_at_end_in_bf16=bf16_end,
        num_layers=num_layers,
        use_sdma=use_sdma,
        fp8_dpa=fp8_dpa,
        fp8_mha=fp8_mha,
    )

    dp_group = None
    if reduce_amax:
        import torch.distributed as dist
        from megatron.core import parallel_state

        if dist.is_initialized():
            dp_group = parallel_state.get_data_parallel_group()

    quant.enable(model, config=config, dp_group=dp_group)
    print_rank_0(
        f"> FP8 training enabled (format={fmt}, scaling={scaling}, "
        f"block_size={block_size}, amax_algo={amax_algo}, "
        f"reduce_amax={reduce_amax}, history={history_len}, "
        f"activation={quant_act}, grad_quant={grad_quant_type})"
    )


def _find_scaling_manager(model):
    """Retrieve the ScalingManager from quant-patched modules."""
    for module in model.modules():
        sm = getattr(module, "_quant_manager", None)
        if sm is not None:
            return sm
    return None


def _enable_lumen_fp8_checkpoint(scaling_manager):
    """Monkey-patch tensor_parallel.checkpoint to preserve FP8 scaling state."""
    import megatron.core.tensor_parallel as tp_module
    import megatron.core.tensor_parallel.random as tp_random

    from lumen.utils.checkpoint import _FP8ScalingContext

    if hasattr(tp_module, "_lumen_fp8_checkpoint_patched"):
        return

    _original = tp_random.checkpoint

    def _patched(function, distribute_saved_activations, *args):
        ctx = _FP8ScalingContext()
        ctx.save(scaling_manager)
        orig_fn = function

        def wrapped(*a, **kw):
            ctx.restore(scaling_manager)
            return orig_fn(*a, **kw)

        return _original(wrapped, distribute_saved_activations, *args)

    tp_random.checkpoint = _patched
    tp_module.checkpoint = _patched
    tp_module._lumen_fp8_checkpoint_patched = True
    tp_module._lumen_fp8_checkpoint_original = _original
    print_rank_0("> FP8-aware activation checkpointing enabled (Lumen)")


def apply_lumen_pre_quant(model: GPTModel, args) -> None:
    """Phase 1: Set module attributes BEFORE quant.enable() captures them."""
    from lumen.modules.grouped_linear import LumenGroupedLinear
    from lumen.modules.layernorm_linear import LumenLayerNormLinear
    from lumen.modules.parallel_linear import LumenColumnParallelLinear, LumenRowParallelLinear

    delay_wgrad = getattr(args, "lumen_delay_wgrad", False)
    gaf = getattr(args, "lumen_gradient_accumulation_fusion", False)
    fp8_act_store = getattr(args, "lumen_fp8_activation_store", False)

    if not delay_wgrad and not gaf and not fp8_act_store:
        return

    lumen_linear_types = (
        LumenColumnParallelLinear,
        LumenRowParallelLinear,
        LumenLayerNormLinear,
        LumenGroupedLinear,
    )

    count = 0
    for module in model.modules():
        if isinstance(module, lumen_linear_types):
            if delay_wgrad and hasattr(module, "delay_wgrad"):
                module.delay_wgrad = delay_wgrad
            if gaf and hasattr(module, "gradient_accumulation_fusion"):
                module.gradient_accumulation_fusion = gaf
            if fp8_act_store:
                module.fp8_activation_store = True
            count += 1

    if count > 0:
        opts = []
        if delay_wgrad:
            opts.append("delay_wgrad")
        if gaf:
            opts.append("gradient_accumulation_fusion")
        if fp8_act_store:
            opts.append("fp8_activation_store")
        print_rank_0(f"> Lumen pre-quant optimizations ({', '.join(opts)}) applied to {count} modules")


apply_lumen_optimizations = apply_lumen_pre_quant


def apply_lumen_post_quant(model: GPTModel, args) -> None:
    """Phase 2: Features requiring ScalingManager (created by quant.enable)."""
    fp8_ckpt = getattr(args, "lumen_fp8_checkpoint", False)
    fp8_param = getattr(args, "lumen_fp8_param_gather", False)

    sm = _find_scaling_manager(model) if (fp8_ckpt or fp8_param) else None

    if fp8_ckpt:
        if sm is not None:
            _enable_lumen_fp8_checkpoint(sm)
        else:
            print_rank_0("> WARNING: --lumen-fp8-checkpoint requires FP8 quantization (--linear-fp8)")

    if fp8_param:
        if sm is not None:
            sm.enable_fp8_params(model)
            print_rank_0(f"> FP8 param gather enabled ({sm.num_fp8_params} params registered)")
        else:
            print_rank_0("> WARNING: --lumen-fp8-param-gather requires FP8 quantization (--linear-fp8)")


def get_cpu_offload_context(args):
    """Return CPU offload context manager (no-op if disabled)."""
    from lumen.utils.cpu_offload import lumen_cpu_offload_context

    enabled = getattr(args, "lumen_cpu_offload", False)
    return lumen_cpu_offload_context(enabled=enabled)


def register_fp8_param_optimizer_hook(model, optimizer):
    """Register optimizer post-step hook for FP8 param staleness marking.

    Must be called AFTER optimizer creation (outside model_provider).
    """
    sm = _find_scaling_manager(model)
    if sm is not None and sm.num_fp8_params > 0:
        sm.register_fp8_optimizer_hook(optimizer)
        print_rank_0("> FP8 param optimizer hook registered")


# ---------------------------------------------------------------------------
# Synthetic warmup + FP8 state reset
# ---------------------------------------------------------------------------

_warmup_step_counter = 0
_warmup_completed = False


def _get_synthetic_batch(args, *, zero_last_loss_mask=False):
    """Generate a synthetic batch for GPU kernel warmup.

    Args:
        zero_last_loss_mask: If ``True``, set ``loss_mask[:, -1] = 0``
            (used by SFT to match real-data masking behaviour).
    """
    seq_length = args.seq_length
    mbs = args.micro_batch_size

    tokens = torch.ones(mbs, seq_length, dtype=torch.long, device="cuda") * 3545
    tokens[:, -1] = 2
    labels = tokens.clone()
    loss_mask = torch.ones(mbs, seq_length, dtype=torch.float, device="cuda")
    if zero_last_loss_mask:
        loss_mask[:, -1] = 0
    attention_mask = torch.ones(mbs, 1, seq_length, seq_length, dtype=torch.bool, device="cuda")
    position_ids = torch.arange(seq_length, dtype=torch.long, device="cuda").unsqueeze(0).expand(mbs, -1)

    return tokens, labels, loss_mask, attention_mask, position_ids


def reset_fp8_state(model):
    """Reset FP8 scaling state in all Lumen quantised layers."""

    def _reset(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
        if hasattr(m, "_quant_manager"):
            m._quant_manager.reset()
        if hasattr(m, "_tl_scaling_manager"):
            m._tl_scaling_manager.reset()

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    unwrapped.apply(_reset)
    print_rank_0("> FP8 state reset after warmup")


# ---------------------------------------------------------------------------
# Loss function + early stopping
# ---------------------------------------------------------------------------

_val_loss_ema: Optional[float] = None
_early_stop_logged = False


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """LM loss with optional early stopping based on a validation-loss EMA."""
    global _val_loss_ema, _early_stop_logged

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    args = get_args()
    val_target = getattr(args, "val_loss_target", None)
    if val_target is not None and not _early_stop_logged:
        avg_loss = (loss.clone().detach() / max(num_tokens.item(), 1)).item()
        if _val_loss_ema is None:
            _val_loss_ema = avg_loss
        else:
            _val_loss_ema = 0.9 * _val_loss_ema + 0.1 * avg_loss
        if _val_loss_ema < val_target:
            print_rank_0(f"> [Early Stop] Loss EMA ({_val_loss_ema:.4f}) < " f"target ({val_target:.4f}). Stopping.")
            if hasattr(args, "iteration"):
                args.train_iters = args.iteration
            _early_stop_logged = True

    return loss, num_tokens, {"lm loss": reporting}


# ---------------------------------------------------------------------------
# Forward step factory
# ---------------------------------------------------------------------------


def make_forward_step(get_batch_fn: Callable, loss_fn: Callable = loss_func, zero_last_loss_mask: bool = False):
    """Return a ``forward_step`` function suitable for :func:`megatron.training.pretrain`.

    Args:
        get_batch_fn: Model-specific batch constructor
            ``(data_iterator, vp_stage) -> (tokens, labels, loss_mask, attention_mask, position_ids)``.
        loss_fn: Loss function (defaults to :func:`loss_func`).
        zero_last_loss_mask: Forwarded to :func:`_get_synthetic_batch`.
    """

    def forward_step(data_iterator, model: GPTModel):
        global _warmup_step_counter, _warmup_completed

        args = get_args()
        timers = get_timers()
        warmup_steps = getattr(args, "warmup_steps", 0)

        timers("batch-generator", log_level=2).start()
        with stimer(bdata=True):
            if warmup_steps > 0 and not _warmup_completed:
                _warmup_step_counter += 1
                if _warmup_step_counter <= warmup_steps:
                    tokens, labels, loss_mask, attention_mask, position_ids = _get_synthetic_batch(
                        args, zero_last_loss_mask=zero_last_loss_mask
                    )
                    if data_iterator is not None:
                        try:
                            next(data_iterator)
                        except StopIteration:
                            pass
                else:
                    if getattr(args, "linear_fp8", False):
                        reset_fp8_state(model)
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    _warmup_completed = True
                    print_rank_0(f"> Synthetic warmup complete ({warmup_steps} steps). " f"Resuming with real data.")
                    vp_stage = get_attr_wrapped_model(model, "vp_stage")
                    tokens, labels, loss_mask, attention_mask, position_ids = get_batch_fn(data_iterator, vp_stage)
            else:
                vp_stage = get_attr_wrapped_model(model, "vp_stage")
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch_fn(data_iterator, vp_stage)
        timers("batch-generator").stop()

        with stimer:
            with get_cpu_offload_context(args):
                output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)

        return output_tensor, partial(loss_fn, loss_mask, model=model)

    return forward_step


# ---------------------------------------------------------------------------
# Common CLI argument groups
# ---------------------------------------------------------------------------


def add_common_megatron_args(parser):
    """Register CLI argument groups shared by all Megatron model scripts.

    Registers: ``--backend``, lumen, mxfp8-block-config, lora,
    fp8-training, and warmup/early-stop groups.

    Uses :func:`safe_add_argument` so that model-specific scripts can
    pre-register any of these flags with different defaults **before**
    calling this function.
    """
    safe_add_argument(
        parser, "--backend", type=str, default="megatron", choices=["megatron", "fsdp"], help="Training backend."
    )

    lumen = parser.add_argument_group(title="Lumen")
    safe_add_argument(
        lumen,
        "--lumen-attn-backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "csrc", "asm"],
        help="Lumen attention kernel backend. 'auto' prefers csrc with triton fallback. "
        "'asm' uses ASM kernels with fallback chain: asm -> csrc -> triton.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-attn",
        type=str,
        default="none",
        choices=["none", "dpa", "mha"],
        help="FP8 attention scope: 'none' = BF16 attention, "
        "'dpa' = FP8 dot-product attention only, "
        "'mha' = FP8 for full Multi-Head Attention block "
        "(QKV projection + attention + output projection).",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-quant-type",
        type=str,
        default="blockwise",
        choices=["dynamic", "delayed", "blockwise", "blockwise2d", "per_token", "none", "mxfp8"],
        help="FP8 quantisation type for FP8 attention backends.",
    )
    safe_add_argument(
        lumen,
        "--lumen-rmsnorm",
        action="store_true",
        default=False,
        help="Replace RMSNorm with Lumen Triton-accelerated RMSNorm.",
    )
    safe_add_argument(
        lumen,
        "--lumen-norm",
        action="store_true",
        default=False,
        help="Replace all norm modules (RMSNorm and LayerNorm) with Lumen implementations.",
    )
    safe_add_argument(
        lumen,
        "--lumen-linear",
        action="store_true",
        default=False,
        help="Use Lumen parallel linear modules (LumenColumnParallelLinear, "
        "LumenRowParallelLinear, LumenLayerNormLinear) via the Lumen spec provider.",
    )
    safe_add_argument(
        lumen,
        "--lumen-cross-entropy",
        action="store_true",
        default=False,
        help="Compute loss using Lumen's Triton parallel cross-entropy kernel.",
    )
    safe_add_argument(
        lumen,
        "--lumen-cpu-offload",
        action="store_true",
        default=False,
        help="Offload activations to CPU pinned memory during forward, prefetch in backward.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-checkpoint",
        action="store_true",
        default=False,
        help="Use FP8-aware activation checkpointing that preserves scaling state.",
    )
    safe_add_argument(
        lumen,
        "--lumen-hip-graphs",
        action="store_true",
        default=False,
        help="Graph-capture training steps to reduce kernel launch overhead.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-activation-store",
        action="store_true",
        default=False,
        help="Store MLP activations in FP8 during forward for reduced memory in backward.",
    )
    safe_add_argument(
        lumen,
        "--lumen-cp-comm-type",
        type=str,
        default="a2a",
        choices=["a2a", "p2p"],
        help="Context parallelism communication type: 'a2a' (all-to-all) or 'p2p' (ring).",
    )
    safe_add_argument(
        lumen,
        "--lumen-delay-wgrad",
        action="store_true",
        default=False,
        help="Defer weight gradient computation to overlap with next layer comm.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-param-gather",
        action="store_true",
        default=False,
        help="Store and all-gather parameters in FP8 for reduced communication volume.",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap",
        action="store_true",
        default=False,
        help="Overlap TP communication with GEMM computation. "
        "Mode is set by --lumen-tp-comm-overlap-mode (default: none, which uses "
        "SDMA async overlap when --use-sdma is set). Use 'pipeline' for chunked "
        "NCCL fused pipelining (requires sequence_parallel, BF16/scaling_type=none).",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap-mode",
        type=str,
        default="none",
        choices=["none", "pipeline"],
        help="TP comm-GEMM overlap mode. 'none': legacy SDMA async overlap (requires "
        "--use-sdma). 'pipeline': chunked NCCL fused pipelining with user-buffer "
        "double-buffering (requires sequence_parallel, BF16).",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap-chunks",
        type=int,
        default=4,
        help="Number of pipeline chunks for 'pipeline' overlap mode. Sequence length "
        "must be divisible by this value. More chunks = finer overlap granularity "
        "but higher scheduling overhead.",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap-method",
        type=str,
        default="nccl",
        choices=["nccl"],
        help="Communication backend for 'pipeline' overlap mode. Currently only 'nccl' " "is supported.",
    )
    safe_add_argument(
        lumen,
        "--use-sdma",
        action="store_true",
        default=False,
        help="Use mori SDMA instead of torch.distributed for supported collectives "
        "(TP comm, amax all-reduce, CP all-to-all) when available.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fused-rope",
        action="store_true",
        default=False,
        help="Use AITER fused RoPE kernel for rotary positional embeddings.",
    )
    safe_add_argument(
        lumen,
        "--lumen-gradient-accumulation-fusion",
        action="store_true",
        default=False,
        help="Fuse weight gradient accumulation into GEMM backward (accumulate into main_grad).",
    )
    safe_add_argument(
        lumen,
        "--lumen-fused-mlp",
        action="store_true",
        default=False,
        help="Use fused MLP modules (LumenFusedMLP / LumenGatedMLP) for reduced kernel launch overhead.",
    )
    mxfp8 = parser.add_argument_group(title="mxfp8-block-config")
    safe_add_argument(mxfp8, "--mxfp8-block-m-fwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-fwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-m-dq-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-dq-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-m-dkv-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-dkv-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-quant-block-size", type=int, default=128)

    lora = parser.add_argument_group(title="lora")
    safe_add_argument(lora, "--lora-rank", type=int, default=0, help="LoRA rank. 0 = disabled.")
    safe_add_argument(lora, "--lora-alpha", type=float, default=32.0)
    safe_add_argument(lora, "--lora-dropout", type=float, default=0.1)
    safe_add_argument(
        lora,
        "--lora-a2a",
        action="store_true",
        default=False,
        help="Enable LoRA all-to-all communication optimisation.",
    )

    lfp8 = parser.add_argument_group(title="linear-fp8")
    safe_add_argument(
        lfp8,
        "--linear-fp8",
        action="store_true",
        default=False,
        help="Enable FP8 quantised training for Linear layers.",
    )
    safe_add_argument(
        lfp8,
        "--linear-fp8-scaling",
        type=str,
        default="delayed",
        choices=["dynamic", "delayed", "blockwise", "blockwise2d", "per_token", "none"],
    )
    safe_add_argument(lfp8, "--linear-fp8-block-size", type=int, default=128)
    safe_add_argument(lfp8, "--linear-fp8-amax-algo", type=str, default="max", choices=["max", "most_recent"])
    safe_add_argument(lfp8, "--linear-fp8-reduce-amax", action="store_true", default=False)
    safe_add_argument(lfp8, "--linear-fp8-amax-history", type=int, default=16)
    safe_add_argument(
        lfp8, "--linear-fp8-margin", type=int, default=0, help="Margin for FP8 scaling factor computation."
    )
    safe_add_argument(lfp8, "--linear-fp8-activation", action="store_true", default=True)
    safe_add_argument(lfp8, "--no-linear-fp8-activation", dest="linear_fp8_activation", action="store_false")
    safe_add_argument(lfp8, "--linear-fp8-wgrad", action="store_true", default=True)
    safe_add_argument(
        lfp8,
        "--no-linear-fp8-wgrad",
        dest="linear_fp8_wgrad",
        action="store_false",
        help="Execute weight gradient GEMM in higher precision (BF16) even for FP8 runs.",
    )
    safe_add_argument(
        lfp8,
        "--grad-quant-type",
        type=str,
        default=None,
        choices=["fp8", "mxfp8", "fp4"],
        help="Gradient quantization type (None=disabled). Applies to Linear, Attention, and RMSNorm.",
    )

    wes = parser.add_argument_group(title="warmup-early-stop")
    safe_add_argument(wes, "--warmup-steps", type=int, default=0)
    safe_add_argument(wes, "--val-loss-target", type=float, default=None)

    parser.set_defaults(**_TE_FORCE_OVERRIDES)

    return parser
