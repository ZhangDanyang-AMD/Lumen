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

import os

import torch
from torch.autograd.function import once_differentiable

_FUSED_SWIGLU_QUANT = os.environ.get("LUMEN_FUSED_SWIGLU_QUANT", "0") == "1"


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
    3. When ``LUMEN_FUSED_SWIGLU_QUANT=1``, runs AITER's fused SiLU+mul+FP8 quant
       after SwiGLU forward so the next FP8 GEMM can skip separate activation quant.
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
            bf16_out = _swiglu_mod.swiglu(input)
            if _FUSED_SWIGLU_QUANT:
                try:
                    from lumen.models._swiglu_fp8_fuse import try_fused_swiglu_fp8

                    try_fused_swiglu_fp8(input, bf16_out)
                except Exception:
                    pass
            return bf16_out

        @staticmethod
        @once_differentiable
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


# ── 5b. Fused SwiGLU activation via Triton ────────────────────────────────

_FUSED_SWIGLU = os.environ.get("LUMEN_FUSED_SWIGLU", "0") == "1"


def install_fused_swiglu_triton():
    """Replace Megatron's ``swiglu`` / ``swiglu_back`` with single-kernel
    Triton fused implementations.

    The original ``@jit_fuser`` implementations expand into 6-8 separate
    elementwise kernels per call (chunk, silu, sigmoid, mul, add, cat).
    The Triton kernels handle the full forward and backward in one launch
    each, saving ~400-600ms/step at LLaMA-70B scale.

    Controlled by ``LUMEN_FUSED_SWIGLU=1``.
    """
    if not _FUSED_SWIGLU:
        return

    try:
        from megatron.core.fusions import fused_bias_swiglu as _swiglu_mod
    except ImportError:
        return

    if getattr(_swiglu_mod, "_lumen_triton_swiglu_patched", False):
        return

    from lumen.ops.fused_swiglu import _probe_aiter_swiglu

    if not _probe_aiter_swiglu():
        import logging

        logging.getLogger(__name__).warning("LUMEN_FUSED_SWIGLU=1 but AITER SwiGLU kernels not available — skipping")
        return

    from lumen.ops.fused_swiglu import fused_swiglu, fused_swiglu_backward

    _swiglu_mod.swiglu = fused_swiglu
    _swiglu_mod.swiglu_back = fused_swiglu_backward
    _swiglu_mod._lumen_triton_swiglu_patched = True


# ── 6. MLP-only recompute ──────────────────────────────────────────────────


def install_mlp_recompute():
    """Wrap the MLP sublayer of non-recomputed ``TransformerLayer`` with
    ``tensor_parallel.checkpoint`` when ``LUMEN_MLP_RECOMPUTE=1``.

    This frees intermediate BF16 tensors (FC1_out, SwiGLU_out) that
    would otherwise stay alive in the autograd graph (~1.3 GiB/layer).

    ``LUMEN_MLP_RECOMPUTE_LAYERS`` controls how many non-recomputed layers
    get MLP-only recompute (default: all).  Set to e.g. ``20`` to only
    recompute MLP for the first 20 non-recomputed layers, balancing memory
    savings (~1.3 GiB/layer) vs compute overhead (~30ms/layer).
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

    _mlp_recompute_count = [0]
    _max_mlp_recompute = int(os.environ.get("LUMEN_MLP_RECOMPUTE_LAYERS", "9999"))

    def _patched_forward(self, *args, **kwargs):
        if os.environ.get("LUMEN_MLP_RECOMPUTE", "0") != "1" or not self.training:
            return _orig_forward(self, *args, **kwargs)

        if not hasattr(self, "_lumen_mlp_recompute_installed"):
            self._lumen_mlp_recompute_installed = True
            should_recompute = (
                hasattr(self, "mlp")
                and not isinstance(self.mlp, _IdentityOp)
                and not getattr(self, "recompute_mlp", False)
                and _mlp_recompute_count[0] < _max_mlp_recompute
            )
            if should_recompute:
                _orig_mlp_fwd = self.mlp.forward
                _mlp_recompute_count[0] += 1

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


# ── Fused RoPE (apex → Megatron rope_utils) ────────────────────────────────

_FUSED_ROPE_INSTALLED = False


def install_fused_rope():
    """Register apex fused RoPE into Megatron's rope_utils.

    Enables ``apply_rope_fusion=True`` without TransformerEngine.  The apex
    kernel delegates to AITER on ROCm (MI300X), matching TE's fused path.

    Megatron passes ``interleaved=`` kwarg that apex doesn't accept, so we
    wrap to strip unsupported keywords.
    """
    global _FUSED_ROPE_INSTALLED
    if _FUSED_ROPE_INSTALLED:
        return
    try:
        import megatron.core.models.common.embeddings.rope_utils as rope_utils
        from apex.transformer.functional.fused_rope import fused_apply_rotary_pos_emb as _apex_rope

        def _compat_fused_rope(t, freqs, **kwargs):
            return _apex_rope(t, freqs)

        rope_utils.fused_apply_rotary_pos_emb = _compat_fused_rope
        _FUSED_ROPE_INSTALLED = True
    except ImportError:
        pass


# ── 8. Force activation recompute during eval ───────────────────────────────


def install_eval_recompute():
    """Keep activation checkpointing active during ``model.eval()`` so that
    evaluation forward passes use the same 21-layer recompute pattern as
    training instead of storing all 80 layers' activations.

    Without this, the eval forward pass at 96%+ GPU memory creates a
    different allocation pattern that fragments the ROCm block cache.
    The fragmentation persists after ``model.train()`` resumes, causing
    a permanent +20% step-time regression.

    Controlled by ``LUMEN_EVAL_RECOMPUTE=1``.
    """
    if os.environ.get("LUMEN_EVAL_RECOMPUTE", "0") != "1":
        return

    try:
        from megatron.core.transformer import transformer_block as _tb_mod
    except ImportError:
        return

    if getattr(_tb_mod, "_lumen_eval_recompute_patched", False):
        return

    _OrigTransformerBlock = _tb_mod.TransformerBlock
    _orig_forward = _OrigTransformerBlock.forward

    def _forward_with_eval_recompute(self, *args, **kwargs):
        if not self.training and self.config.recompute_granularity == "full":
            self.training = True
            try:
                return _orig_forward(self, *args, **kwargs)
            finally:
                self.training = False
        return _orig_forward(self, *args, **kwargs)

    _OrigTransformerBlock.forward = _forward_with_eval_recompute
    _tb_mod._lumen_eval_recompute_patched = True


# ── 9. Post-eval memory cache clear ─────────────────────────────────────────


def install_post_eval_cache_clear():
    """Call ``torch.cuda.empty_cache()`` after every evaluation run to
    reset the ROCm allocator's cached block pool.

    This prevents fragmented eval-phase blocks from polluting the
    training allocation pattern.  There is a brief re-warmup cost on the
    first post-eval training step, but subsequent steps return to
    pre-eval speed.

    Controlled by ``LUMEN_POST_EVAL_CACHE_CLEAR=1``.
    """
    if os.environ.get("LUMEN_POST_EVAL_CACHE_CLEAR", "0") != "1":
        return

    try:
        import megatron.training.training as _train_mod
    except ImportError:
        return

    if getattr(_train_mod, "_lumen_post_eval_cache_clear_patched", False):
        return

    _orig_evaluate = _train_mod.evaluate
    _do_rewarm = os.environ.get("LUMEN_POST_EVAL_REWARM", "0") == "1"

    def _evaluate_with_cache_clear(*args, **kwargs):
        result = _orig_evaluate(*args, **kwargs)
        torch.cuda.empty_cache()
        if _do_rewarm:
            _post_eval_rewarm()
        return result

    _train_mod.evaluate = _evaluate_with_cache_clear
    _train_mod._lumen_post_eval_cache_clear_patched = True


def _post_eval_rewarm():
    """Allocate and free training-shaped tensors after eval + empty_cache to
    re-prime the ROCm allocator with the training block layout.

    This forces the allocator to create block splits that match training
    tensor sizes, so the first real training step after eval doesn't
    suffer from suboptimal allocations.

    Controlled by ``LUMEN_POST_EVAL_REWARM=1`` (requires POST_EVAL_CACHE_CLEAR).
    """
    try:
        from megatron.training import get_args
    except ImportError:
        return

    args = get_args()
    seq_len = getattr(args, "seq_length", 8192)
    mbs = getattr(args, "micro_batch_size", 1)
    hidden = getattr(args, "hidden_size", 8192)

    shapes = [
        (mbs * seq_len, hidden),
        (mbs * seq_len, hidden * 4),
        (mbs * seq_len, hidden * 2),
        (mbs, seq_len),
    ]
    buffers = []
    for shape in shapes:
        buffers.append(torch.empty(shape, dtype=torch.bfloat16, device="cuda"))
    del buffers
    torch.cuda.synchronize()


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
    install_fused_swiglu_triton()
    install_mlp_recompute()
    install_fused_rope()
    install_eval_recompute()
    install_post_eval_cache_clear()
