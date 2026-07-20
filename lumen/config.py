###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unified Lumen configuration — single dataclass covering quantization,
attention, norm replacement, and execution/fusion features.

Usage::

    from lumen.config import LumenConfig

    # Programmatic — flat kwargs, one call
    cfg = LumenConfig(format="fp8_e4m3", scaling="delayed",
                      fp8_attn="dpa", lumen_norm=True, fused_mlp=True)
    cfg.enable(model)

    # From CLI argparse namespace
    cfg = LumenConfig.from_args(args)
    cfg.enable(model, dp_group=dp_group)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, replace
from typing import Optional

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
# Arg-name mapping: LumenConfig field → argparse attribute name(s)
# ---------------------------------------------------------------------------

_ARG_MAP: dict[str, tuple[str, ...]] = {
    "format": ("linear_fp8_format", "lumen_fp8_format"),
    "scaling": ("linear_fp8_scaling",),
    "block_size": ("linear_fp8_block_size",),
    "amax_algo": ("linear_fp8_amax_algo",),
    "margin": ("linear_fp8_margin",),
    "reduce_amax": ("linear_fp8_reduce_amax",),
    "history_len": ("linear_fp8_amax_history",),
    "quantize_activation": ("linear_fp8_activation",),
    "fp8_wgrad": ("linear_fp8_wgrad",),
    "cache_frozen_weight": ("linear_fp8_cache_frozen_weight",),
    "bpreshuffle_gemm": ("linear_fp8_bpreshuffle",),
    "quantize_grad": ("grad_quant_type",),
    "first_last_layers_bf16": ("first_last_layers_bf16",),
    "num_layers_at_start_in_bf16": ("num_layers_at_start_in_bf16",),
    "num_layers_at_end_in_bf16": ("num_layers_at_end_in_bf16",),
    "num_layers": ("num_layers",),
    "use_sdma": ("use_sdma",),
    "fp8_attn": ("lumen_fp8_attn",),
    "attn_backend": ("lumen_attn_backend",),
    "attn_quant_type": ("lumen_fp8_quant_type",),
    "lumen_norm": ("lumen_norm",),
    "hf_attn_patch": ("hf_attn_patch",),
    "lumen_linear": ("lumen_linear",),
    "fused_mlp": ("lumen_fused_mlp",),
    "fp8_activation_store": ("lumen_fp8_activation_store",),
    "cpu_offload": ("lumen_cpu_offload",),
    "delay_wgrad": ("lumen_delay_wgrad",),
    "gradient_accumulation_fusion": ("lumen_gradient_accumulation_fusion",),
    "fp8_param_gather": ("lumen_fp8_param_gather",),
    "fp8_weight_cache": ("lumen_fp8_weight_cache",),
    "fused_rope": ("lumen_fused_rope",),
    "fused_moe": ("lumen_fused_moe",),
    "hip_graphs": ("lumen_hip_graphs",),
    "fp8_checkpoint": ("lumen_fp8_checkpoint",),
    "fp8_param_manager": ("fp8_param_manager",),
    "lora_rank": ("lora_rank",),
    "lora_alpha": ("lora_alpha",),
    "lora_dropout": ("lora_dropout",),
    "use_8bit_adam": ("use_8bit_adam",),
}


@dataclass
class LumenConfig:
    """Unified configuration for all Lumen training features.

    Groups functionality behind a single flat interface:

    * **Tier 0 — Weight storage & adapters:** ``fp8_param_manager``,
      ``lora_rank`` / ``lora_alpha`` / ``lora_dropout``.  Applied first
      (FP8ParamManager before LoRA) so adapter weights stay BF16.
    * **Tier 1 — Linear FP8:** ``format``, ``scaling``, ``block_size``, etc.
      These are forwarded to :class:`~lumen.quantize.QuantConfig`.
    * **Tier 2 — Attention FP8 & norms:** ``fp8_attn``, ``attn_backend``,
      ``attn_quant_type``, ``lumen_norm``.
    * **Tier 3 — Execution / fusion:** ``fused_mlp``, ``cpu_offload``,
      ``delay_wgrad``, ``hip_graphs``, etc.
    * **Optimizer hint:** ``use_8bit_adam`` — not applied to the model,
      read by the trainer to select 8-bit Adam from bitsandbytes.
    """

    # -- Tier 1: Linear FP8 (forwarded to QuantConfig) --
    format: str = "fp8_e4m3"
    scaling: str = "delayed"
    block_size: int = 128
    amax_algo: str = "max"
    margin: int = 0
    reduce_amax: bool = False
    history_len: int = 16
    quantize_activation: bool = True
    fp8_wgrad: bool = True
    cache_frozen_weight: bool = False
    bpreshuffle_gemm: bool = False
    quantize_grad: Optional[str] = None
    first_last_layers_bf16: bool = False
    num_layers_at_start_in_bf16: int = 1
    num_layers_at_end_in_bf16: int = 1
    num_layers: int = 0
    use_sdma: bool = False

    # -- Tier 2: Attention FP8 --
    fp8_attn: str = "none"
    attn_backend: str = "auto"
    attn_quant_type: str = "blockwise"

    # -- Tier 2: Norm replacement --
    lumen_norm: bool = False

    # -- Tier 2: Attention patching --
    hf_attn_patch: bool = False

    # -- Tier 2: Linear GEMM patching (BF16) --
    lumen_linear: bool = False

    # -- Tier 3: Execution / fusion --
    fused_mlp: bool = False
    fp8_activation_store: bool = False
    cpu_offload: bool = False
    delay_wgrad: bool = False
    gradient_accumulation_fusion: bool = False
    fp8_param_gather: bool = False
    fp8_weight_cache: bool = False
    fused_rope: bool = False
    fused_moe: bool = False
    hip_graphs: bool = False
    fp8_checkpoint: bool = False

    # -- Tier 0: FP8 weight storage (applied before everything else) --
    fp8_param_manager: bool = False

    # -- Tier 0: LoRA via HuggingFace PEFT (after FP8ParamManager) --
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # -- Optimizer hint (not applied to model, read by trainer) --
    use_8bit_adam: bool = False

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def quant_config(self):
        """Build the inner :class:`~lumen.quantize.QuantConfig`."""
        from lumen.quantize import QuantConfig

        return QuantConfig.from_str(
            format=self.format,
            scaling=self.scaling,
            block_size=self.block_size,
            amax_algo=self.amax_algo,
            margin=self.margin,
            reduce_amax=self.reduce_amax,
            history_len=self.history_len,
            quantize_activation=self.quantize_activation,
            fp8_wgrad=self.fp8_wgrad,
            cache_frozen_weight=self.cache_frozen_weight,
            bpreshuffle_gemm=self.bpreshuffle_gemm,
            quantize_grad=self.quantize_grad,
            first_last_layers_bf16=self.first_last_layers_bf16,
            num_layers_at_start_in_bf16=self.num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=self.num_layers_at_end_in_bf16,
            num_layers=self.num_layers,
            use_sdma=self.use_sdma,
            fp8_dpa=self.fp8_attn in ("dpa", "mha"),
            fp8_mha=self.fp8_attn == "mha",
        )

    # -----------------------------------------------------------------------
    # Orchestrated enablement
    # -----------------------------------------------------------------------

    @property
    def has_any_features(self) -> bool:
        """Return True if any Lumen feature is enabled."""
        return (
            self.quant_config.is_quantized
            or self.lumen_norm
            or self.hf_attn_patch
            or self.lumen_linear
            or self.fp8_param_manager
            or self.lora_rank > 0
            or self.fused_mlp
            or self.fp8_activation_store
            or self.cpu_offload
            or self.fp8_param_gather
            or self.fp8_weight_cache
            or self.fp8_checkpoint
            or self.fused_rope
            or self.fused_moe
            or self.hip_graphs
            or self.delay_wgrad
            or self.gradient_accumulation_fusion
        )

    def enable(self, model, *, dp_group=None, ep_group=None,
               ep_size: int = 1, ep_rank: int = 0,
               backend: str = "auto"):
        """Apply all Lumen features to *model* in the correct order.

        Orchestration:
          0a. FP8ParamManager — quantize linear weights to FP8 storage
          0b. LoRA (PEFT) — wrap linears with trainable adapters
          1.  Fused RoPE (module-level monkey-patch, before EP sharding)
          1a. Norm patching (before quant so new norm modules get patched)
          1b. Attention patching
          1c. Linear GEMM patching (BF16 AITER)
          2.  EP sharding (replaces MoE blocks with EPShardedMoeBlock)
          2a. Fused MoE (patches EPShardedMoeBlock local expert compute)
          3.  Pre-quant module flags (delay_wgrad, grad-accum fusion, etc.)
          4.  ``quant.enable()`` — FP8 linear patching
          5.  Post-quant features (fp8_checkpoint, fp8_param_gather)
          6.  Attach config to model for downstream reads

        FP8ParamManager runs **before** LoRA so that only base ``nn.Linear``
        weights are quantized; the LoRA adapter weights (``lora_A``, ``lora_B``)
        created afterwards stay in BF16 and remain trainable.

        Args:
            model: The model to patch.
            dp_group: DP process group for FSDP reduce_amax.
            ep_group: EP process group for expert parallelism.  When provided
                with ``ep_size > 1``, MoE blocks are sharded across the group.
            ep_size: Number of GPUs in the EP group (default 1 = no EP).
            ep_rank: This GPU's rank within the EP group.
            backend: GEMM backend selection.

        Returns:
            ``(manager, model)`` — the :class:`~lumen.quantize.ScalingManager`
            (or ``None``) and the model (may be a new PEFT wrapper).
        """
        import torch

        # 0a. FP8 param storage (replaces weight.data with FP8, freezes)
        fp8pm_mgr = None
        if self.fp8_param_manager:
            fp8pm_mgr = self._apply_fp8_param_manager(model)

        # 0b. LoRA adapters (PEFT)
        if self.lora_rank > 0:
            model = self._apply_lora(model)

        # 1. Fused RoPE (module-level patch, before EP sharding)
        if self.fused_rope:
            self._patch_fused_rope(model)

        # 1a. Norm patching
        if self.lumen_norm:
            self._patch_norms(model)

        # 1b. Attention patching (monkey-patch F.scaled_dot_product_attention)
        if self.hf_attn_patch:
            self._patch_sdpa()

        # 1c. Linear GEMM patching (BF16 AITER Triton)
        if self.lumen_linear:
            self._patch_linear(model)

        # 2. EP sharding (replaces HF MoE blocks with EPShardedMoeBlock)
        if ep_size > 1 and ep_group is not None:
            from lumen.modules.ep_moe import shard_experts_ep

            model = shard_experts_ep(model, ep_size, ep_rank, ep_group)

        # 2a. Fused MoE local-expert compute (AITER Triton fused GEMM)
        if self.fused_moe:
            self._patch_fused_moe(model)

        # 2. Pre-quant module attributes
        self._apply_pre_quant(model)

        # 3. FP8 linear quantization
        qcfg = self.quant_config
        manager = None
        if qcfg.is_quantized:
            import torch.distributed as dist

            import lumen.quantize as quant

            if dp_group is None and qcfg.reduce_amax and dist.is_initialized():
                dp_group = dist.group.WORLD
            manager = quant.enable(
                model,
                config=qcfg,
                backend=backend,
                dp_group=dp_group if qcfg.reduce_amax else None,
            )

        # 4. Post-quant features
        self._apply_post_quant(model, manager)

        # 5. Attach config + FP8ParamManager reference
        model._lumen_config = self
        if fp8pm_mgr is not None:
            model._fp8_param_manager = fp8pm_mgr

        self._log_summary(manager)
        return manager, model

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _apply_fp8_param_manager(self, model):
        """Replace linear weights with FP8 storage via FP8ParamManager.

        Targets ``nn.Linear`` and Megatron ``ColumnParallelLinear`` /
        ``RowParallelLinear``.  Must run before LoRA wrapping so that
        adapter weights stay BF16/trainable.
        """
        import torch

        from lumen.quantize.fp8_params import FP8ParamManager

        mgr = FP8ParamManager(fp8_dtype=torch.float8_e4m3fn)
        count = mgr.quantize_params(model)
        hooks = mgr.register_dequant_hooks(model)
        _rank0_print(f"> FP8ParamManager: quantized {count} params, registered {hooks} hooks")
        return mgr

    def _apply_lora(self, model):
        """Apply LoRA adapters via HuggingFace PEFT and freeze the base model.

        When FP8ParamManager is active, PEFT's ``_move_adapter_to_device_of_base_layer``
        casts LoRA adapter weights to FP8 (matching the quantized base weight dtype).
        We fix this by re-casting all LoRA adapter parameters back to BF16 so they
        remain trainable with standard arithmetic.
        """
        import torch
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.lora_target_modules),
        )
        model = get_peft_model(model, peft_config)

        if self.fp8_param_manager:
            fixed = 0
            for name, param in model.named_parameters():
                if "lora_" in name and param.dtype in (
                    torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                    torch.float8_e5m2, torch.float8_e5m2fnuz,
                ):
                    param.data = param.data.to(torch.bfloat16)
                    param.requires_grad_(True)
                    fixed += 1
            if fixed:
                _rank0_print(f"> Fixed {fixed} LoRA params cast to FP8 by PEFT -> BF16")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _rank0_print(
            f"> LoRA applied (rank={self.lora_rank}, alpha={self.lora_alpha}) — "
            f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )
        return model

    def _patch_norms(self, model) -> None:
        import torch
        from lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

        count = 0
        for _name, module in model.named_modules():
            for attr_name, child in list(module.named_children()):
                cls_name = type(child).__name__
                if cls_name in (
                    "RMSNorm",
                    "LlamaRMSNorm",
                    "MistralRMSNorm",
                    "Qwen2RMSNorm",
                    "Qwen3RMSNorm",
                ):
                    hidden = child.weight.shape[0]
                    eps = getattr(child, "eps", getattr(child, "variance_epsilon", 1e-6))
                    target_dtype = child.weight.dtype if not child.weight.is_meta else torch.bfloat16
                    repl = LumenRMSNorm(hidden, eps=eps, grad_quant_type=self.quantize_grad)
                    if not child.weight.is_meta:
                        repl.weight.data.copy_(child.weight.data)
                    repl.to(target_dtype)
                    if child.weight.is_meta:
                        repl.to(device="meta")
                    setattr(module, attr_name, repl)
                    count += 1
                elif cls_name in ("LayerNorm",):
                    hidden = child.weight.shape[0] if child.weight is not None else child.normalized_shape[0]
                    eps = getattr(child, "eps", 1e-5)
                    repl = LumenLayerNorm(hidden, eps=eps, grad_quant_type=self.quantize_grad)
                    is_meta = child.weight is not None and child.weight.is_meta
                    target_dtype = child.weight.dtype if (child.weight is not None and not is_meta) else torch.bfloat16
                    if child.weight is not None and not is_meta:
                        repl.weight.data.copy_(child.weight.data)
                    if hasattr(child, "bias") and child.bias is not None and repl.bias is not None and not is_meta:
                        repl.bias.data.copy_(child.bias.data)
                    repl.to(target_dtype)
                    if is_meta:
                        repl.to(device="meta")
                    setattr(module, attr_name, repl)
                    count += 1

        if count:
            _rank0_print(f"> Replaced {count} norm modules with Lumen implementations")

    def _patch_sdpa(self) -> None:
        """Monkey-patch ``F.scaled_dot_product_attention`` with AITER attention."""
        from lumen.ops.attention.hf_patch import patch_sdpa

        patch_sdpa()

    @staticmethod
    def _patch_fused_rope(model) -> None:
        """Monkey-patch HuggingFace ``apply_rotary_pos_emb`` with AITER fused RoPE.

        Detects the model's architecture and patches the corresponding
        HF modeling module's ``apply_rotary_pos_emb`` function.
        """
        from lumen.ops.rope import apply_rotary_qk_autograd

        arch = getattr(model.config, "model_type", "")
        _module_map = {
            "qwen3_moe": "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "qwen2_moe": "transformers.models.qwen2_moe.modeling_qwen2_moe",
            "qwen2": "transformers.models.qwen2.modeling_qwen2",
            "llama": "transformers.models.llama.modeling_llama",
            "mistral": "transformers.models.mistral.modeling_mistral",
        }
        mod_name = _module_map.get(arch)
        if mod_name is None:
            _rank0_print(f"> fused_rope: unsupported architecture '{arch}', skipping")
            return

        import importlib

        try:
            hf_mod = importlib.import_module(mod_name)
        except ImportError:
            _rank0_print(f"> fused_rope: could not import {mod_name}, skipping")
            return

        def _lumen_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return apply_rotary_qk_autograd(q, k, cos, sin)

        hf_mod.apply_rotary_pos_emb = _lumen_rope
        _rank0_print(f"> Fused RoPE: {mod_name}.apply_rotary_pos_emb -> AITER autograd RoPE")

    @staticmethod
    def _auto_tune_bf16_gemm(model) -> None:
        """Placeholder — BF16 GEMM uses torch.mm fallback (safe on MI350).

        hipBLASLt and ASM JIT both SIGABRT on MI350 gfx950, so we skip
        tuned GEMM injection and let ``tuned_gemm.gemm_a16w16`` fall back
        to ``torch.mm`` (which internally uses hipBLASLt via PyTorch).
        """
        pass

    def _patch_linear(self, model) -> None:
        """Replace ``nn.Linear`` forward with AITER GEMM (BF16, autograd-safe).

        Runs ``_auto_tune_bf16_gemm`` first to ensure hipBLASLt tuning
        configs exist, then patches forward with ``dispatch_gemm``.
        """
        import torch
        import torch.nn as nn

        self._auto_tune_bf16_gemm(model)

        from lumen.ops.quantize.linear import dispatch_gemm

        class _GemmA16W16Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight, bias):
                ctx.save_for_backward(input, weight)
                ctx.has_bias = bias is not None
                output = dispatch_gemm(input, weight, scaling_type="none", bias=bias)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                input, weight = ctx.saved_tensors
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                input_flat = input.reshape(-1, input.shape[-1])
                # grad_input = grad @ W  (NN layout — torch.mm handles this natively)
                grad_input = torch.mm(grad_flat, weight).reshape(input.shape)
                # grad_weight = grad^T @ input  (NN layout)
                grad_weight = torch.mm(grad_flat.t(), input_flat)
                grad_bias = (grad_flat.sum(0) if ctx.has_bias else None)
                return grad_input, grad_weight, grad_bias

        skip_ids = set()
        if self.fused_moe:
            for _name, module in model.named_modules():
                cls_name = type(module).__name__
                if cls_name in ("Qwen3MoeSparseMoeBlock", "EPShardedMoeBlock"):
                    for _sub_name, sub in module.named_modules():
                        skip_ids.add(id(sub))

        count = 0
        for _name, module in model.named_modules():
            if isinstance(module, nn.Linear) and id(module) not in skip_ids:

                def _make_aiter_forward(mod):
                    def _aiter_forward(input):
                        return _GemmA16W16Fn.apply(input, mod.weight, mod.bias)

                    return _aiter_forward

                module.forward = _make_aiter_forward(module)
                count += 1

        if count:
            _rank0_print(f"> Replaced {count} nn.Linear forward with AITER GEMM (ASM→HIP→Triton)")

    def _patch_fused_moe(self, model) -> None:
        """Replace sequential per-expert loops with AITER fused MoE kernels.

        Targets HuggingFace Qwen3 MoE blocks (``Qwen3MoeSparseMoeBlock``)
        and EP-sharded wrappers.  Replaces the Python-level expert-by-expert
        loop with AITER's ``fused_topk`` (ASM fused softmax+topk) and
        ``fused_moe_triton`` (Triton fused token-sort + grouped GEMM).

        Gate routing stays on the original ``nn.Linear`` gate; only the
        expert compute is fused.
        """
        from lumen.ops.dispatch import (
            _probe_aiter_moe_topk_softmax,
            _probe_aiter_triton_fused_moe,
            _probe_aiter_triton_moe_align,
        )

        has_topk = _probe_aiter_moe_topk_softmax()
        has_fused = _probe_aiter_triton_fused_moe() and _probe_aiter_triton_moe_align()

        if not has_topk:
            _rank0_print("> fused_moe: AITER topk_softmax not available, skipping fused routing")
        if not has_fused:
            _rank0_print("> fused_moe: AITER fused_moe triton not available, skipping fused expert GEMM")

        if not has_topk and not has_fused:
            return

        hf_count = 0
        ep_count = 0
        for _name, module in model.named_modules():
            cls_name = type(module).__name__
            if cls_name == "Qwen3MoeSparseMoeBlock":
                self._patch_hf_moe_block(module, has_topk, has_fused)
                hf_count += 1
            elif cls_name == "EPShardedMoeBlock":
                self._patch_ep_moe_block(module, has_fused)
                ep_count += 1

        total = hf_count + ep_count
        if total:
            features = []
            if has_topk and hf_count:
                features.append("fused_topk")
            if has_fused:
                features.append("fused_expert_gemm")
            label = (
                f"{hf_count} HF + {ep_count} EP-sharded"
                if hf_count and ep_count
                else f"{ep_count} EP-sharded" if ep_count
                else f"{hf_count} HF"
            )
            _rank0_print(
                f"> Fused MoE: patched {label} MoE blocks "
                f"({', '.join(features)})"
            )
        else:
            _rank0_print("> fused_moe: no MoE blocks found to patch")

    @staticmethod
    def _patch_hf_moe_block(moe_block, has_topk, has_fused) -> None:
        """Patch a single HuggingFace MoE block with fused kernels.

        Forward uses AITER ``fused_moe_triton`` for all three expert GEMMs.
        Backward computes gradients per-expert using ``torch.mm`` so that
        gradients flow to the original expert parameters.
        """
        import torch
        import torch.nn.functional as F

        original_forward = moe_block.forward
        gate = moe_block.gate
        experts = moe_block.experts
        num_experts = moe_block.num_experts
        top_k = moe_block.top_k
        norm_topk_prob = moe_block.norm_topk_prob

        if has_topk:
            from lumen.ops.moe.fused_routing import fused_topk as aiter_fused_topk

        if has_fused:
            from lumen.ops.moe.fused_moe import (
                _align_tokens,
                _get_moe_config,
                fused_moe_triton,
                moe_wgrad_triton,
            )

            class _FusedHFExpertsFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, hidden_flat, topk_indices, topk_weights,
                            w_gate, w_up, w_down):
                    topk_ids_i32 = topk_indices.to(torch.int32)
                    topk_w_f32 = topk_weights.float()

                    M = hidden_flat.shape[0]
                    fwd_config = _get_moe_config(hidden_flat.dtype, M)
                    block_size_m = fwd_config["BLOCK_SIZE_M"]

                    align_topk = _align_tokens(topk_ids_i32, num_experts, block_size_m)

                    gate_out = fused_moe_triton(
                        hidden_flat, w_gate, topk_ids_i32, topk_w_f32,
                        num_experts, top_k, mul_routed_weight=False,
                        precomputed_alignment=align_topk,
                        config=fwd_config,
                    )
                    up_out = fused_moe_triton(
                        hidden_flat, w_up, topk_ids_i32, topk_w_f32,
                        num_experts, top_k, mul_routed_weight=False,
                        precomputed_alignment=align_topk,
                        config=fwd_config,
                    )
                    silu_gate = F.silu(gate_out)
                    act_out = silu_gate * up_out

                    flat_ids_down = topk_ids_i32.reshape(-1, 1)
                    ones_down = torch.ones(
                        flat_ids_down.shape[0], 1,
                        dtype=torch.float32, device=hidden_flat.device,
                    )
                    M_flat = flat_ids_down.shape[0]
                    fwd_config_flat = _get_moe_config(hidden_flat.dtype, M_flat)
                    block_size_m_flat = fwd_config_flat["BLOCK_SIZE_M"]
                    align_flat = _align_tokens(flat_ids_down, num_experts, block_size_m_flat)

                    down_out = fused_moe_triton(
                        act_out.reshape(-1, act_out.shape[-1]),
                        w_down,
                        flat_ids_down,
                        ones_down,
                        num_experts, 1, mul_routed_weight=False,
                        precomputed_alignment=align_flat,
                        config=fwd_config_flat,
                    ).reshape(hidden_flat.shape[0], top_k, -1)

                    output = (down_out * topk_weights.unsqueeze(-1)).sum(dim=1)

                    ctx.save_for_backward(
                        hidden_flat, topk_indices, topk_weights,
                        w_gate, w_up, w_down,
                        gate_out, up_out, down_out,
                    )
                    return output

                @staticmethod
                def backward(ctx, grad_output):
                    (hidden_flat, topk_indices, topk_weights,
                     w_gate, w_up, w_down,
                     gate_out, up_out, down_out) = ctx.saved_tensors

                    T = hidden_flat.shape[0]
                    hidden_dim = hidden_flat.shape[1]
                    inter_dim = gate_out.shape[-1]

                    flat_ids = topk_indices.reshape(-1)
                    flat_hidden = hidden_flat.unsqueeze(1).expand(
                        -1, top_k, -1).reshape(-1, hidden_dim)
                    flat_gate = gate_out.reshape(-1, inter_dim)
                    flat_up = up_out.reshape(-1, inter_dim)

                    flat_grad_down = (
                        grad_output.unsqueeze(1) * topk_weights.unsqueeze(-1)
                    ).reshape(-1, hidden_dim)

                    fused_ids = flat_ids.unsqueeze(1).to(torch.int32)
                    ones = torch.ones(
                        flat_ids.shape[0], 1,
                        dtype=torch.float32, device=hidden_flat.device,
                    )

                    M_bwd = flat_ids.shape[0]
                    bwd_config = _get_moe_config(hidden_flat.dtype, M_bwd)
                    bwd_block_m = bwd_config["BLOCK_SIZE_M"]
                    alignment = _align_tokens(fused_ids, num_experts, bwd_block_m)

                    # dgrad: stride-swap (view, no copy) makes kernel compute A @ B
                    flat_grad_act = fused_moe_triton(
                        flat_grad_down, w_down.transpose(1, 2), fused_ids, ones,
                        num_experts, 1, mul_routed_weight=False,
                        precomputed_alignment=alignment,
                        config=bwd_config,
                    ).squeeze(1)

                    # SwiGLU backward
                    sg = F.silu(flat_gate)
                    flat_grad_up = flat_grad_act * sg
                    flat_grad_silu = flat_grad_act * flat_up
                    sig = torch.sigmoid(flat_gate)
                    flat_grad_gate = flat_grad_silu * sig * (
                        1.0 + flat_gate * (1.0 - sig))

                    flat_grad_h = fused_moe_triton(
                        flat_grad_gate, w_gate.transpose(1, 2), fused_ids, ones,
                        num_experts, 1, mul_routed_weight=False,
                        precomputed_alignment=alignment,
                        config=bwd_config,
                    ).squeeze(1)
                    flat_grad_h += fused_moe_triton(
                        flat_grad_up, w_up.transpose(1, 2), fused_ids, ones,
                        num_experts, 1, mul_routed_weight=False,
                        precomputed_alignment=alignment,
                        config=bwd_config,
                    ).squeeze(1)

                    grad_hidden = flat_grad_h.reshape(T, top_k, hidden_dim).sum(dim=1)

                    # --- grad_topk_weights (reuse saved down_out) ---
                    flat_ao = sg * flat_up
                    grad_topk_weights = (
                        grad_output.unsqueeze(1)
                        * down_out
                    ).sum(dim=-1)

                    # --- Weight grads via Triton wgrad kernel (autotuned) ---
                    sorted_token_ids, expert_ids, num_tokens_post_pad = alignment
                    grad_w_gate = moe_wgrad_triton(
                        flat_grad_gate, flat_hidden,
                        sorted_token_ids, expert_ids, num_tokens_post_pad,
                        num_experts, 1, w_gate.shape,
                        block_size_m=bwd_block_m,
                    )
                    grad_w_up = moe_wgrad_triton(
                        flat_grad_up, flat_hidden,
                        sorted_token_ids, expert_ids, num_tokens_post_pad,
                        num_experts, 1, w_up.shape,
                        block_size_m=bwd_block_m,
                    )
                    grad_w_down = moe_wgrad_triton(
                        flat_grad_down, flat_ao,
                        sorted_token_ids, expert_ids, num_tokens_post_pad,
                        num_experts, 1, w_down.shape,
                        block_size_m=bwd_block_m,
                    )

                    return (grad_hidden, None, grad_topk_weights,
                            grad_w_gate, grad_w_up, grad_w_down)

        def _fused_forward(hidden_states):
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_flat = hidden_states.view(-1, hidden_dim)

            router_logits = gate(hidden_flat)

            if has_topk and not torch.is_grad_enabled():
                topk_weights, topk_indices = aiter_fused_topk(
                    router_logits, top_k, softmax_first=True,
                )
                topk_weights = topk_weights.to(hidden_flat.dtype)
            else:
                if has_topk:
                    _, topk_indices = aiter_fused_topk(
                        router_logits, top_k, softmax_first=True,
                    )
                    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                    topk_weights = routing_weights.gather(1, topk_indices)
                else:
                    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                    topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
                if norm_topk_prob:
                    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                topk_weights = topk_weights.to(hidden_flat.dtype)

            if has_fused:
                w_gate = torch.stack([e.gate_proj.weight for e in experts])
                w_up = torch.stack([e.up_proj.weight for e in experts])
                w_down = torch.stack([e.down_proj.weight for e in experts])

                output = _FusedHFExpertsFn.apply(
                    hidden_flat, topk_indices, topk_weights,
                    w_gate, w_up, w_down,
                )
                return output.view(batch_size, seq_len, hidden_dim)
            else:
                return original_forward(hidden_states)

        moe_block.forward = _fused_forward

    @staticmethod
    def _patch_ep_moe_block(ep_block, has_fused) -> None:
        """Replace ``EPShardedMoeBlock._compute_local_experts`` with fused kernels.

        Forward uses AITER ``fused_moe_triton`` for the three expert GEMMs
        (gate, up, down).  Backward computes gradients per-expert using
        standard ``torch.mm`` so that gradients flow to the original
        parameters for training.
        """
        if not has_fused:
            return

        import torch
        import torch.nn.functional as F

        from lumen.ops.moe.fused_moe import (
            _align_tokens,
            _get_moe_config,
            fused_moe_triton,
            moe_wgrad_triton,
        )

        experts_per_gpu = ep_block.experts_per_gpu
        has_module_list = ep_block.local_experts is not None

        class _FusedEPExpertsFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, recv_hidden, recv_expert_ids, recv_weights,
                        w_gate, w_up, w_down):
                topk_ids = recv_expert_ids.unsqueeze(1).to(torch.int32)
                ones = torch.ones(
                    recv_hidden.shape[0], 1,
                    dtype=torch.float32, device=recv_hidden.device,
                )

                M = recv_hidden.shape[0]
                fwd_config = _get_moe_config(recv_hidden.dtype, M)
                block_size_m = fwd_config["BLOCK_SIZE_M"]
                alignment = _align_tokens(topk_ids, experts_per_gpu, block_size_m)

                gate_out = fused_moe_triton(
                    recv_hidden, w_gate, topk_ids, ones,
                    experts_per_gpu, 1, mul_routed_weight=False,
                    precomputed_alignment=alignment,
                    config=fwd_config,
                ).squeeze(1)
                up_out = fused_moe_triton(
                    recv_hidden, w_up, topk_ids, ones,
                    experts_per_gpu, 1, mul_routed_weight=False,
                    precomputed_alignment=alignment,
                    config=fwd_config,
                ).squeeze(1)

                silu_gate = F.silu(gate_out)
                act_out = silu_gate * up_out

                down_out = fused_moe_triton(
                    act_out, w_down, topk_ids, ones,
                    experts_per_gpu, 1, mul_routed_weight=False,
                    precomputed_alignment=alignment,
                    config=fwd_config,
                ).squeeze(1)

                output = down_out * recv_weights.unsqueeze(-1)

                ctx.save_for_backward(
                    recv_hidden, recv_expert_ids, recv_weights,
                    w_gate, w_up, w_down,
                    gate_out, up_out, down_out,
                )
                return output

            @staticmethod
            def backward(ctx, grad_output):
                (recv_hidden, recv_expert_ids, recv_weights,
                 w_gate, w_up, w_down,
                 gate_out, up_out, down_out) = ctx.saved_tensors

                topk_ids = recv_expert_ids.unsqueeze(1).to(torch.int32)
                ones = torch.ones(
                    recv_hidden.shape[0], 1,
                    dtype=torch.float32, device=recv_hidden.device,
                )

                M_bwd = recv_hidden.shape[0]
                bwd_config = _get_moe_config(recv_hidden.dtype, M_bwd)
                bwd_block_m = bwd_config["BLOCK_SIZE_M"]
                alignment = _align_tokens(topk_ids, experts_per_gpu, bwd_block_m)

                grad_down = grad_output * recv_weights.unsqueeze(-1)

                # dgrad: stride-swap (view, no copy) makes kernel compute A @ B
                grad_act = fused_moe_triton(
                    grad_down, w_down.transpose(1, 2), topk_ids, ones,
                    experts_per_gpu, 1, mul_routed_weight=False,
                    precomputed_alignment=alignment,
                    config=bwd_config,
                ).squeeze(1)

                # SwiGLU backward (vectorized)
                sg = F.silu(gate_out)
                grad_up_all = grad_act * sg
                grad_silu = grad_act * up_out
                sig = torch.sigmoid(gate_out)
                grad_gate_all = grad_silu * sig * (1.0 + gate_out * (1.0 - sig))

                grad_hidden = fused_moe_triton(
                    grad_gate_all, w_gate.transpose(1, 2), topk_ids, ones,
                    experts_per_gpu, 1, mul_routed_weight=False,
                    precomputed_alignment=alignment,
                    config=bwd_config,
                ).squeeze(1)
                grad_hidden += fused_moe_triton(
                    grad_up_all, w_up.transpose(1, 2), topk_ids, ones,
                    experts_per_gpu, 1, mul_routed_weight=False,
                    precomputed_alignment=alignment,
                    config=bwd_config,
                ).squeeze(1)

                # --- grad_recv_weights (reuse saved down_out) ---
                ao = sg * up_out
                grad_recv_weights = (grad_output * down_out).sum(dim=-1)

                # --- Weight grads via Triton wgrad kernel (autotuned) ---
                sorted_token_ids, expert_ids, num_tokens_post_pad = alignment
                grad_w_gate = moe_wgrad_triton(
                    grad_gate_all, recv_hidden,
                    sorted_token_ids, expert_ids, num_tokens_post_pad,
                    experts_per_gpu, 1, w_gate.shape,
                    block_size_m=bwd_block_m,
                )
                grad_w_up = moe_wgrad_triton(
                    grad_up_all, recv_hidden,
                    sorted_token_ids, expert_ids, num_tokens_post_pad,
                    experts_per_gpu, 1, w_up.shape,
                    block_size_m=bwd_block_m,
                )
                grad_w_down = moe_wgrad_triton(
                    grad_down, ao,
                    sorted_token_ids, expert_ids, num_tokens_post_pad,
                    experts_per_gpu, 1, w_down.shape,
                    block_size_m=bwd_block_m,
                )

                return (grad_hidden, None, grad_recv_weights,
                        grad_w_gate, grad_w_up, grad_w_down)

        def _fused_compute_local_experts(self, recv_hidden, recv_expert_ids, recv_weights):
            if recv_hidden.shape[0] == 0:
                return torch.zeros_like(recv_hidden)

            # Build weight tensors from live parameters each forward call
            # so that autograd can track gradients back to the originals.
            if has_module_list:
                w_gate = torch.stack([e.gate_proj.weight for e in self.local_experts])
                w_up = torch.stack([e.up_proj.weight for e in self.local_experts])
                w_down = torch.stack([e.down_proj.weight for e in self.local_experts])
            else:
                half = self.local_gate_up_proj.shape[1] // 2
                w_gate = self.local_gate_up_proj[:, :half, :].contiguous()
                w_up = self.local_gate_up_proj[:, half:, :].contiguous()
                w_down = self.local_down_proj

            return _FusedEPExpertsFn.apply(
                recv_hidden, recv_expert_ids, recv_weights,
                w_gate, w_up, w_down,
            )

        import types

        ep_block._compute_local_experts = types.MethodType(
            _fused_compute_local_experts, ep_block
        )

    def _apply_pre_quant(self, model) -> None:
        """Set module attributes that must exist before ``quant.enable()``.

        ``delay_wgrad`` and ``gradient_accumulation_fusion`` are only
        meaningful on Lumen-native parallel linear types.

        ``fp8_activation_store`` is set on both Lumen-native types **and**
        standard ``nn.Linear`` so that ``_replace_forward`` captures the
        flag and passes it into ``QuantizedLinearFunction``.
        """
        import torch.nn as nn

        has_lumen_only = self.delay_wgrad or self.gradient_accumulation_fusion
        has_act_store = self.fp8_activation_store

        if not (has_lumen_only or has_act_store):
            return

        lumen_count = 0
        if has_lumen_only:
            try:
                from lumen.modules.grouped_linear import LumenGroupedLinear
                from lumen.modules.layernorm_linear import LumenLayerNormLinear
                from lumen.modules.parallel_linear import (
                    LumenColumnParallelLinear,
                    LumenRowParallelLinear,
                )

                lumen_types = (
                    LumenColumnParallelLinear,
                    LumenRowParallelLinear,
                    LumenLayerNormLinear,
                    LumenGroupedLinear,
                )
            except ImportError:
                lumen_types = ()

            for module in model.modules():
                if lumen_types and isinstance(module, lumen_types):
                    if self.delay_wgrad and hasattr(module, "delay_wgrad"):
                        module.delay_wgrad = True
                    if self.gradient_accumulation_fusion and hasattr(module, "gradient_accumulation_fusion"):
                        module.gradient_accumulation_fusion = True
                    if has_act_store:
                        module.fp8_activation_store = True
                    lumen_count += 1

        linear_count = 0
        if has_act_store:
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.fp8_activation_store = True
                    linear_count += 1

        opts = []
        if self.delay_wgrad:
            opts.append("delay_wgrad")
        if self.gradient_accumulation_fusion:
            opts.append("gradient_accumulation_fusion")
        if has_act_store:
            opts.append("fp8_activation_store")
        total = lumen_count + linear_count
        if total:
            _rank0_print(
                f"> Pre-quant optimizations ({', '.join(opts)}) applied to "
                f"{total} modules ({lumen_count} Lumen, {linear_count} nn.Linear)"
            )

    def _apply_post_quant(self, model, manager) -> None:
        """Apply features that require the ScalingManager from ``quant.enable()``."""
        if self.fp8_checkpoint:
            if manager is not None:
                self._enable_fp8_checkpoint(manager)
            else:
                _rank0_print("> WARNING: fp8_checkpoint requires FP8 quantization")

        if self.fp8_param_gather:
            if manager is not None:
                manager.enable_fp8_params(model)
                _rank0_print(f"> FP8 param gather enabled ({manager.num_fp8_params} params)")
            else:
                _rank0_print("> WARNING: fp8_param_gather requires FP8 quantization")

        if self.fp8_weight_cache:
            if manager is not None:
                import lumen.quantize as quant

                count = quant.store_weights_fp8(model)
                _rank0_print(f"> FP8 weight cache enabled ({count} layers cached)")
            else:
                _rank0_print("> WARNING: fp8_weight_cache requires FP8 quantization")

    @staticmethod
    def _enable_fp8_checkpoint(manager) -> None:
        """Monkey-patch Megatron checkpoint to preserve FP8 scaling state.

        No-op if Megatron is not installed.
        """
        try:
            import megatron.core.tensor_parallel as tp_module
            import megatron.core.tensor_parallel.random as tp_random
        except ImportError:
            return

        from lumen.utils.checkpoint import _FP8ScalingContext

        if hasattr(tp_module, "_lumen_fp8_checkpoint_patched"):
            return

        _original = tp_random.checkpoint

        def _patched(function, distribute_saved_activations, *args):
            ctx = _FP8ScalingContext()
            ctx.save(manager)
            orig_fn = function

            def wrapped(*a, **kw):
                ctx.restore(manager)
                return orig_fn(*a, **kw)

            return _original(wrapped, distribute_saved_activations, *args)

        tp_random.checkpoint = _patched
        tp_module.checkpoint = _patched
        tp_module._lumen_fp8_checkpoint_patched = True
        tp_module._lumen_fp8_checkpoint_original = _original
        _rank0_print("> FP8-aware activation checkpointing enabled")

    def _log_summary(self, manager) -> None:
        parts = [f"format={self.format}", f"scaling={self.scaling}"]
        if self.fp8_param_manager:
            parts.append("fp8_param_manager")
        if self.lora_rank > 0:
            parts.append(f"lora_rank={self.lora_rank}")
        if self.fp8_attn != "none":
            parts.append(f"fp8_attn={self.fp8_attn}")
        if self.lumen_norm:
            parts.append("lumen_norm")
        if self.hf_attn_patch:
            parts.append("hf_attn_patch")
        if self.lumen_linear:
            parts.append("lumen_linear")
        if self.use_8bit_adam:
            parts.append("use_8bit_adam")
        tier3 = [
            name
            for name in (
                "fused_mlp",
                "fused_moe",
                "cpu_offload",
                "delay_wgrad",
                "gradient_accumulation_fusion",
                "fp8_param_gather",
                "fp8_weight_cache",
                "fused_rope",
                "hip_graphs",
                "fp8_checkpoint",
                "fp8_activation_store",
            )
            if getattr(self, name, False)
        ]
        if tier3:
            parts.append(f"features=[{', '.join(tier3)}]")
        _rank0_print(f"> LumenConfig enabled ({', '.join(parts)})")

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> LumenConfig:
        """Build from an argparse ``Namespace`` (replaces manual ``getattr`` blocks).

        Iterates :data:`_ARG_MAP` to find matching attributes on *args*.
        Unknown / missing attributes are silently skipped (defaults apply).

        Respects the ``linear_fp8`` boolean gate: when it is ``False``, the
        FP8 Linear quantization fields (format, scaling, etc.) are suppressed
        so that ``quant_config.is_quantized`` remains ``False``.
        """
        linear_fp8_enabled = getattr(args, "linear_fp8", None)

        kwargs: dict = {}
        for field_name, arg_names in _ARG_MAP.items():
            for arg_name in arg_names:
                val = getattr(args, arg_name, None)
                if val is not None:
                    kwargs[field_name] = val
                    break

        if linear_fp8_enabled is False:
            kwargs["scaling"] = "none"

        return cls(**kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> LumenConfig:
        """Build from a plain dict (e.g. YAML / JSON config)."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
