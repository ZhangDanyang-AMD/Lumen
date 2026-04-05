###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron-LM compatibility patches for Lumen.

All patches in this module are applied via import-time monkey-patching.
They fix incompatibilities between Lumen and Megatron-LM-AMD, including:

- RMSNorm / FusedLayerNorm dispatch
- LanguageModule checkpoint save with LoRA
- Memory-mapped checkpoint loading (mmap)
- LoRA base_layer key remapping during checkpoint load
- requires_grad fix for LoRA + activation checkpointing
- SwiGLU FP8 dtype on MI300X (e4m3fnuz) + chunked backward
- MLP-only recompute for non-recomputed layers

Call :func:`install_all` once at import time (before any Megatron model
classes are imported) to apply every patch.  Individual installers can
also be called selectively.
"""

import torch

# ── 1. FusedLayerNorm → Lumen RMSNorm / LayerNorm ──────────────────────────


def install_fused_layer_norm():
    """Replace Megatron's ``FusedLayerNorm`` with a Lumen-compatible wrapper.

    Must run **before** ``GPTModel`` or ``TransformerBlock`` is imported,
    because those modules capture ``FusedLayerNorm`` at class-definition time.
    """
    from megatron.core.fusions import fused_layer_norm as _fln_mod

    from lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

    class _FusedLayerNormCompat(torch.nn.Module):
        def __new__(cls, config, hidden_size, eps=1e-6, **kwargs):
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


# ── 2. LanguageModule checkpoint save guard ─────────────────────────────────


def install_language_module_checkpoint_guard():
    """Guard ``LanguageModule.sharded_state_dict`` against missing
    ``output_layer.weight`` when LoRA wraps that layer."""
    from megatron.core.models.common.language_module import language_module as _lm_mod

    def _patched_sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        assert not sharded_offsets, "Unexpected sharded offsets"
        from megatron.core.transformer.module import MegatronModule

        sharded_state_dict = MegatronModule.sharded_state_dict(self, prefix, sharded_offsets, metadata)
        first_stage_word_emb_key = f"{prefix}embedding.word_embeddings.weight"
        output_layer_weight_key = f"{prefix}output_layer.weight"
        output_layer_bias_key = f"{prefix}output_layer.bias"

        if self.share_embeddings_and_output_weights:
            self.tie_embeddings_and_output_weights_state_dict(
                sharded_state_dict, output_layer_weight_key, first_stage_word_emb_key
            )
        elif self.post_process:
            if output_layer_weight_key in sharded_state_dict:
                sharded_state_dict[output_layer_weight_key].allow_shape_mismatch = True

        if self.post_process and output_layer_bias_key in sharded_state_dict:
            sharded_state_dict[output_layer_bias_key].allow_shape_mismatch = True

        return sharded_state_dict

    _lm_mod.LanguageModule.sharded_state_dict = _patched_sharded_state_dict


# ── 3. Memory-mapped checkpoint loading ─────────────────────────────────────


def install_mmap_checkpoint():
    """Inject ``mmap=True`` into Megatron's ``torch.load`` calls.

    Prevents 8 ranks from each loading a full checkpoint into CPU RAM.
    """
    import megatron.training.checkpointing as _ckpt_mod

    if getattr(_ckpt_mod, "_lumen_mmap_patched", False):
        return

    _orig_torch_load = torch.load

    def _mmap_torch_load(*args, **kwargs):
        kwargs.setdefault("mmap", True)
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    if hasattr(_ckpt_mod, "torch"):
        _ckpt_mod.torch.load = _mmap_torch_load
    _ckpt_mod._lumen_mmap_patched = True


# ── 4. requires_grad fix for LoRA + activation checkpointing ───────────────


def install_requires_grad_fix():
    """Force ``hidden_states.requires_grad_(True)`` after ``make_viewless_tensor``.

    With LoRA + activation checkpointing, frozen embeddings produce
    ``hidden_states.requires_grad=False``, breaking the backward chain
    (grad_norm=0).
    """
    from megatron.core.transformer import transformer_block as _tb_mod
    from megatron.core.utils import make_viewless_tensor as _orig_mvt

    if getattr(_tb_mod, "_lumen_requires_grad_patched", False):
        return

    def _make_viewless_with_grad(inp, requires_grad, keep_graph):
        out = _orig_mvt(inp=inp, requires_grad=requires_grad, keep_graph=keep_graph)
        if not out.requires_grad and torch.is_grad_enabled():
            out = out.detach().requires_grad_(True)
        return out

    _tb_mod.make_viewless_tensor = _make_viewless_with_grad
    _tb_mod._lumen_requires_grad_patched = True


# ── 5. SwiGLU FP8 dtype + chunked backward ─────────────────────────────────


def install_swiglu_fp8():
    """Replace Megatron's ``SwiGLUFunction`` with a version that:

    1. Uses ``float8_e4m3fnuz`` on MI300X (instead of hard-coded ``float8_e4m3fn``).
    2. Processes the backward in 1024-row chunks when FP8 input store is active,
       reducing peak transient memory from 896 MiB to 112 MiB for Llama-70B.
    """
    try:
        from megatron.core.fusions import fused_bias_swiglu as _swiglu_mod
    except ImportError:
        return

    if getattr(_swiglu_mod, "_lumen_fp8_dtype_patched", False):
        return

    from lumen.quantize.config import _get_float8_e4m3

    _fp8_e4m3 = _get_float8_e4m3()

    class _PatchedSwiGLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, fp8_input_store=False, cpu_offload_input=False):
            if fp8_input_store:
                input_for_backward = input.to(_fp8_e4m3)
            else:
                input_for_backward = input
            if cpu_offload_input:
                input_for_backward.activation_offloading = True
            ctx.save_for_backward(input_for_backward)
            ctx.ori_input_dtype = input.dtype
            ctx.fp8_input_store = fp8_input_store
            return _swiglu_mod.swiglu(input)

        @staticmethod
        def backward(ctx, grad_output):
            input_saved = ctx.saved_tensors[0]
            if not ctx.fp8_input_store:
                return _swiglu_mod.swiglu_back(grad_output, input_saved), None, None
            dtype = ctx.ori_input_dtype
            M = input_saved.shape[0]
            chunk_sz = min(1024, M)
            n_chunks = (M + chunk_sz - 1) // chunk_sz
            result = torch.empty(
                M,
                input_saved.shape[1],
                dtype=dtype,
                device=input_saved.device,
            )
            for i in range(n_chunks):
                s = i * chunk_sz
                e = min(s + chunk_sz, M)
                inp_chunk = input_saved[s:e].to(dtype)
                grad_chunk = grad_output[s:e]
                result[s:e] = _swiglu_mod.swiglu_back(grad_chunk, inp_chunk)
                del inp_chunk
            return result, None, None

    _swiglu_mod.SwiGLUFunction = _PatchedSwiGLUFunction
    _swiglu_mod._lumen_fp8_dtype_patched = True


# ── 6. MLP-only recompute ──────────────────────────────────────────────────


def install_mlp_recompute():
    """Wrap the MLP sublayer of non-recomputed ``TransformerLayer`` with
    ``tensor_parallel.checkpoint`` when ``LUMEN_MLP_RECOMPUTE=1``.

    This frees intermediate BF16 tensors (FC1_out, SwiGLU_out) that
    would otherwise stay alive in the autograd graph (~389 MiB/layer).
    """
    import os

    try:
        from megatron.core.transformer import transformer_layer as _tl_mod
    except ImportError:
        return

    if getattr(_tl_mod, "_lumen_mlp_recompute_patched", False):
        return

    try:
        from megatron.core.transformer.identity_op import IdentityOp as _IdentityOp
    except ImportError:
        _IdentityOp = type(None)

    _OrigTransformerLayer = _tl_mod.TransformerLayer
    _orig_forward = _OrigTransformerLayer.forward

    def _patched_forward(self, *args, **kwargs):
        if os.environ.get("LUMEN_MLP_RECOMPUTE", "0") != "1" or not self.training:
            return _orig_forward(self, *args, **kwargs)

        if not hasattr(self, "_lumen_mlp_recompute_installed"):
            self._lumen_mlp_recompute_installed = True
            if (
                hasattr(self, "mlp")
                and not isinstance(self.mlp, _IdentityOp)
                and not getattr(self, "recompute_mlp", False)
            ):
                _orig_mlp_fwd = self.mlp.forward

                def _recompute_mlp_forward(hidden_states, *a, **kw):
                    from megatron.core import tensor_parallel

                    return tensor_parallel.checkpoint(_orig_mlp_fwd, False, hidden_states, *a, **kw)

                self.mlp.forward = _recompute_mlp_forward

        return _orig_forward(self, *args, **kwargs)

    _OrigTransformerLayer.forward = _patched_forward
    _tl_mod._lumen_mlp_recompute_patched = True


# ── 7. LoRA checkpoint key remapping ────────────────────────────────────────


def remap_lora_state_dict(module, state_dict):
    """Remap checkpoint keys for LoRA-wrapped layers.

    Transforms ``decoder.layers.N.linear_qkv.weight`` →
    ``decoder.layers.N.linear_qkv.base_layer.weight`` to match the model's
    state dict when LoRA wrapping adds a ``.base_layer.`` level.

    Returns the (possibly modified) state dict.
    """
    import re

    inner = module
    while hasattr(inner, "module"):
        inner = inner.module

    inner_keys = set(inner.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if len(ckpt_keys.intersection(inner_keys)) >= len(ckpt_keys):
        return state_dict

    lora_parents = set()
    for ik in inner_keys:
        m = re.match(r"(.+)\.base_layer\.weight$", ik)
        if m:
            lora_parents.add(m.group(1))
    if not lora_parents:
        return state_dict

    new_sd = {}
    mapped = 0
    for ck, cv in state_dict.items():
        parts = ck.rsplit(".", 1)
        if len(parts) == 2 and parts[0] in lora_parents:
            new_sd[f"{parts[0]}.base_layer.{parts[1]}"] = cv
            mapped += 1
        else:
            new_sd[ck] = cv

    if mapped > 0:
        try:
            from megatron.training import print_rank_0

            print_rank_0(f"> Lumen: remapped {mapped} LoRA base_layer checkpoint keys")
        except ImportError:
            pass
    return new_sd


# ── Public API ──────────────────────────────────────────────────────────────


def install_all():
    """Apply every Megatron compatibility patch.

    Must be called **before** ``GPTModel`` / ``TransformerBlock`` is imported.
    Safe to call multiple times (each installer is idempotent).
    """
    install_fused_layer_norm()
    install_language_module_checkpoint_guard()
    install_mmap_checkpoint()
    install_requires_grad_fix()
    install_swiglu_fp8()
    install_mlp_recompute()
