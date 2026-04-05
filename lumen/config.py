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
from dataclasses import dataclass, fields
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
    "fused_mlp": ("lumen_fused_mlp",),
    "fp8_activation_store": ("lumen_fp8_activation_store",),
    "cpu_offload": ("lumen_cpu_offload",),
    "delay_wgrad": ("lumen_delay_wgrad",),
    "gradient_accumulation_fusion": ("lumen_gradient_accumulation_fusion",),
    "fp8_param_gather": ("lumen_fp8_param_gather",),
    "fused_rope": ("lumen_fused_rope",),
    "hip_graphs": ("lumen_hip_graphs",),
    "fp8_checkpoint": ("lumen_fp8_checkpoint",),
}


@dataclass
class LumenConfig:
    """Unified configuration for all Lumen training features.

    Groups three tiers of functionality behind a single flat interface:

    * **Tier 1 — Linear FP8:** ``format``, ``scaling``, ``block_size``, etc.
      These are forwarded to :class:`~lumen.quantize.QuantConfig`.
    * **Tier 2 — Attention FP8 & norms:** ``fp8_attn``, ``attn_backend``,
      ``attn_quant_type``, ``lumen_norm``.
    * **Tier 3 — Execution / fusion:** ``fused_mlp``, ``cpu_offload``,
      ``delay_wgrad``, ``hip_graphs``, etc.
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

    # -- Tier 3: Execution / fusion --
    fused_mlp: bool = False
    fp8_activation_store: bool = False
    cpu_offload: bool = False
    delay_wgrad: bool = False
    gradient_accumulation_fusion: bool = False
    fp8_param_gather: bool = False
    fused_rope: bool = False
    hip_graphs: bool = False
    fp8_checkpoint: bool = False

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

    def enable(self, model, *, dp_group=None, backend: str = "auto"):
        """Apply all Lumen features to *model* in the correct order.

        Orchestration:
          1. Norm patching (before quant so new norm modules get patched)
          2. Pre-quant module flags (delay_wgrad, grad-accum fusion, etc.)
          3. ``quant.enable()`` — FP8 linear patching
          4. Post-quant features (fp8_checkpoint, fp8_param_gather)
          5. Attach config to model for downstream reads

        Returns:
            The :class:`~lumen.quantize.ScalingManager` if FP8 is active,
            else ``None``.
        """
        # 1. Norm patching
        if self.lumen_norm:
            self._patch_norms(model)

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

        # 5. Attach config
        model._lumen_config = self

        self._log_summary(manager)
        return manager

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _patch_norms(self, model) -> None:
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
                ):
                    hidden = child.weight.shape[0]
                    eps = getattr(child, "eps", getattr(child, "variance_epsilon", 1e-6))
                    repl = LumenRMSNorm(hidden, eps=eps, grad_quant_type=self.quantize_grad)
                    repl.weight.data.copy_(child.weight.data)
                    setattr(module, attr_name, repl)
                    count += 1
                elif cls_name in ("LayerNorm",):
                    hidden = child.weight.shape[0] if child.weight is not None else child.normalized_shape[0]
                    eps = getattr(child, "eps", 1e-5)
                    repl = LumenLayerNorm(hidden, eps=eps, grad_quant_type=self.quantize_grad)
                    if child.weight is not None:
                        repl.weight.data.copy_(child.weight.data)
                    if hasattr(child, "bias") and child.bias is not None and repl.bias is not None:
                        repl.bias.data.copy_(child.bias.data)
                    setattr(module, attr_name, repl)
                    count += 1

        if count:
            _rank0_print(f"> Replaced {count} norm modules with Lumen implementations")

    def _apply_pre_quant(self, model) -> None:
        """Set module attributes that must exist before ``quant.enable()``."""
        if not (self.delay_wgrad or self.gradient_accumulation_fusion or self.fp8_activation_store):
            return

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
            return

        count = 0
        for module in model.modules():
            if isinstance(module, lumen_types):
                if self.delay_wgrad and hasattr(module, "delay_wgrad"):
                    module.delay_wgrad = True
                if self.gradient_accumulation_fusion and hasattr(module, "gradient_accumulation_fusion"):
                    module.gradient_accumulation_fusion = True
                if self.fp8_activation_store:
                    module.fp8_activation_store = True
                count += 1

        if count:
            opts = []
            if self.delay_wgrad:
                opts.append("delay_wgrad")
            if self.gradient_accumulation_fusion:
                opts.append("gradient_accumulation_fusion")
            if self.fp8_activation_store:
                opts.append("fp8_activation_store")
            _rank0_print(f"> Pre-quant optimizations ({', '.join(opts)}) applied to {count} modules")

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
        if self.fp8_attn != "none":
            parts.append(f"fp8_attn={self.fp8_attn}")
        if self.lumen_norm:
            parts.append("lumen_norm")
        tier3 = [
            name
            for name in (
                "fused_mlp",
                "cpu_offload",
                "delay_wgrad",
                "gradient_accumulation_fusion",
                "fp8_param_gather",
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
        """
        kwargs: dict = {}
        for field_name, arg_names in _ARG_MAP.items():
            for arg_name in arg_names:
                val = getattr(args, arg_name, None)
                if val is not None:
                    kwargs[field_name] = val
                    break
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> LumenConfig:
        """Build from a plain dict (e.g. YAML / JSON config)."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
