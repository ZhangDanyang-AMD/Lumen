###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""VERL + Lumen entrypoint.

This module provides the Lumen-patched VERL training entrypoint. It
monkey-patches VERL's FSDP and Megatron workers to inject Lumen FP8/norm/attn
optimizations into the model build pipeline.  When LoRA is requested for the
Megatron backend, it monkey-patches ``make_megatron_module`` to inject a
LoRA callback into the model creation pipeline (applied before DDP wrapping).

Usage:
    python -m lumen.rl.verl.verl_entry --config-path configs/grpo_fsdp_lumen.yaml
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _resolve_verl_fsdp_module():
    """Return the VERL module that owns ``apply_fsdp2``.

    VERL moved the FSDP2 helper from ``verl.workers.fsdp_workers`` to
    ``verl.workers.engine.fsdp.transformer_impl``.  Support both layouts so
    the Lumen entrypoint keeps working across the versions used in this repo.
    """
    try:
        import verl.workers.fsdp_workers as fsdp_module

        return fsdp_module
    except ImportError:
        from verl.workers.engine.fsdp import transformer_impl as fsdp_module

        return fsdp_module


def patch_verl_fsdp_workers(lumen_args):
    """Monkey-patch VERL FSDP workers to inject Lumen optimizations.

    Hooks into apply_fsdp2 to inject Lumen FP8/norm/attn optimizations
    right before FSDP2 sharding is applied to the actor model.

    When fp8_weight_cache is enabled, also hooks into the optimizer
    creation to register post-step hooks that refresh FP8 weight caches.
    """
    fsdp_module = _resolve_verl_fsdp_module()

    if getattr(lumen_args, "lumen_fp8_weight_cache", False):
        _patch_optimizer_for_weight_cache(fsdp_module)

    logger.info("> VERL FSDP2 workers patched with Lumen optimizations")


def patch_verl_megatron_workers(lumen_args):
    """Monkey-patch verl.workers.megatron_workers to inject Lumen FP8.

    Hooks into ``ActorRolloutRefWorker.init_model`` to apply
    ``LumenConfig.enable()`` to the actor model after Megatron builds it.
    This enables Lumen FP8 features for Megatron + vLLM/SGLang combos.
    """
    try:
        import verl.workers.megatron_workers as meg_module
    except ImportError:
        logger.warning("> verl.workers.megatron_workers not available — skipping Megatron patch")
        return

    worker_cls = getattr(meg_module, "ActorRolloutRefWorker", None)
    if worker_cls is None:
        logger.warning("> ActorRolloutRefWorker not found in megatron_workers — skipping")
        return

    _original_init_model = worker_cls.init_model

    def _patched_init_model(self, *args, **kwargs):
        from dataclasses import replace as _replace

        from lumen.config import LumenConfig

        result = _original_init_model(self, *args, **kwargs)

        actor_module = getattr(self, "actor_module", None)
        if actor_module is not None:
            logger.info("> Applying Lumen FP8 to Megatron actor module")
            cfg = LumenConfig.from_args(lumen_args)
            cfg_no_lora = _replace(cfg, lora_rank=0)
            cfg_no_lora.enable(actor_module)
        return result

    worker_cls.init_model = _patched_init_model
    logger.info("> VERL Megatron workers patched with Lumen FP8 optimizations")


def _patch_optimizer_for_weight_cache(fsdp_module):
    """Hook into VERL's ActorRolloutRefWorker to register FP8 weight cache
    optimizer hooks after the optimizer is created."""
    from lumen.quantize import register_fp8_weight_optimizer_hooks

    if hasattr(fsdp_module, "ActorRolloutRefWorker"):
        worker_cls = fsdp_module.ActorRolloutRefWorker
        _original_init_model = getattr(worker_cls, "init_model", None)

        if _original_init_model is not None:
            def _patched_init_model(self, *args, **kwargs):
                result = _original_init_model(self, *args, **kwargs)
                model = getattr(self, "actor_module", None) or getattr(self, "model", None)
                optimizer = getattr(self, "actor_optimizer", None) or getattr(self, "optimizer", None)
                if model is not None and optimizer is not None:
                    register_fp8_weight_optimizer_hooks(model, optimizer)
                    logger.info("> FP8 weight cache optimizer hooks registered")
                return result

            worker_cls.init_model = _patched_init_model


def _patch_megatron_lora(lora_rank: int, lora_alpha: float, lora_dropout: float) -> None:
    """Monkey-patch ``make_megatron_module`` to inject LoRA before DDP wrapping.

    Hooks into ``bridge.get_model`` (mbridge path) to append a
    ``post_model_creation_callback`` that applies
    :func:`lumen.models.lora_adapter.apply_megatron_lora` to the freshly
    created GPTModel **before** Megatron wraps it with DDP.  Only the actor
    build (``wrap_with_ddp=True``) is affected; the reference model is left
    unchanged.
    """
    try:
        import verl.utils.megatron_utils as mutils
    except ImportError:
        logger.warning("> verl.utils.megatron_utils not available — skipping LoRA patch")
        return

    _original_make = mutils.make_megatron_module

    def _lora_callback(model, **_kwargs):
        from lumen.models.lora_adapter import apply_megatron_lora

        apply_megatron_lora(model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    def _patched_make(wrap_config, *args, **kwargs):
        bridge = kwargs.get("bridge")
        provider = kwargs.get("provider")

        is_actor_build = getattr(wrap_config, "wrap_with_ddp", False)
        is_mbridge_path = bridge is not None and provider is None

        if is_actor_build and is_mbridge_path:
            _original_get_model = bridge.get_model

            def _lora_get_model(*gm_args, **gm_kwargs):
                callbacks = list(gm_kwargs.get("post_model_creation_callbacks", []))
                callbacks.append(_lora_callback)
                gm_kwargs["post_model_creation_callbacks"] = callbacks
                return _original_get_model(*gm_args, **gm_kwargs)

            bridge.get_model = _lora_get_model
            try:
                return _original_make(wrap_config, *args, **kwargs)
            finally:
                bridge.get_model = _original_get_model

        return _original_make(wrap_config, *args, **kwargs)

    mutils.make_megatron_module = _patched_make
    logger.info(
        "> Patched make_megatron_module for LoRA injection: rank=%d, alpha=%.1f, dropout=%.2f",
        lora_rank, lora_alpha, lora_dropout,
    )


def main():
    """Lumen-aware VERL entrypoint.

    Parses lumen-specific config from the VERL YAML, patches the workers,
    then delegates to VERL's standard training loop.

    When ``LORA_RANK`` > 0, ``make_megatron_module`` is monkey-patched to
    inject a LoRA callback that runs before DDP wrapping.  Lumen's own
    PEFT LoRA (``LumenConfig._apply_lora``) is always skipped for Megatron
    — the custom adapter handles insertion before DDP.
    """
    from lumen.rl.verl.config import VerlLumenArgs

    lumen_fp8 = os.environ.get("LUMEN_FP8", "0") == "1"
    lumen_fp8_format = os.environ.get("LUMEN_FP8_FORMAT", "fp8_e4m3")
    if lumen_fp8_format == "fp8":
        lumen_fp8_format = "fp8_e4m3"
    lumen_fp8_scaling = os.environ.get("LUMEN_FP8_SCALING", "delayed")
    lumen_fp8_block_size = int(os.environ.get("LUMEN_FP8_BLOCK_SIZE", "128"))
    lumen_norm = os.environ.get("LUMEN_NORM", "0") == "1"
    lumen_fp8_attn = os.environ.get("LUMEN_FP8_ATTN", "none")
    lumen_fp8_quant_type = os.environ.get("LUMEN_FP8_QUANT_TYPE", "blockwise")
    lumen_attn_backend = os.environ.get("LUMEN_ATTN_BACKEND", "auto")
    lumen_fp8_weight_cache = os.environ.get("LUMEN_FP8_WEIGHT_CACHE", "0") == "1"
    lumen_fp8_activation_store = os.environ.get("LUMEN_FP8_ACTIVATION_STORE", "0") == "1"
    lumen_fp8_param_gather = os.environ.get("LUMEN_FP8_PARAM_GATHER", "0") == "1"
    fp8_param_manager = os.environ.get("FP8_PARAM_MANAGER", "0") == "1"
    use_8bit_adam = os.environ.get("USE_8BIT_ADAM", "0") == "1"
    lumen_rollout = os.environ.get("LUMEN_ROLLOUT", "")
    force_lumen_fsdp = os.environ.get("LUMEN_FORCE_FSDP", "0") == "1"
    model_path = os.environ.get("MODEL_NAME", "")

    lora_rank = int(os.environ.get("LORA_RANK", "0"))
    lora_alpha = float(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.0"))

    any_lumen = (
        lumen_fp8 or lumen_norm or lumen_fp8_attn != "none"
        or lumen_fp8_weight_cache or lumen_fp8_activation_store
        or lumen_fp8_param_gather or fp8_param_manager
        or bool(lumen_rollout) or force_lumen_fsdp or lora_rank > 0
    )
    if any_lumen:
        lumen_args = VerlLumenArgs(
            model_name_or_path=model_path,
            linear_fp8=lumen_fp8,
            linear_fp8_format=lumen_fp8_format,
            linear_fp8_scaling=lumen_fp8_scaling,
            linear_fp8_block_size=lumen_fp8_block_size,
            lumen_norm=lumen_norm,
            lumen_fp8_attn=lumen_fp8_attn,
            lumen_fp8_quant_type=lumen_fp8_quant_type,
            lumen_attn_backend=lumen_attn_backend,
            lumen_fp8_weight_cache=lumen_fp8_weight_cache,
            lumen_fp8_activation_store=lumen_fp8_activation_store,
            lumen_fp8_param_gather=lumen_fp8_param_gather,
            lumen_rollout=lumen_rollout,
            fp8_param_manager=fp8_param_manager,
            use_8bit_adam=use_8bit_adam,
        )
        patch_verl_fsdp_workers(lumen_args)
        patch_verl_megatron_workers(lumen_args)

    if lora_rank > 0:
        _patch_megatron_lora(lora_rank, lora_alpha, lora_dropout)

    # Select the downstream VERL training entrypoint.  Default keeps the
    # standard PPO trainer; ``LUMEN_VERL_MAIN=dapo`` delegates to
    # ``recipe.dapo.main_dapo`` so DAPO dynamic sampling (filter_groups) is
    # preserved while Lumen worker patches stay active.  The Lumen monkey-patch
    # targets ``verl.workers`` which both trainers share, so it applies either way.
    verl_main_target = os.environ.get("LUMEN_VERL_MAIN", "ppo").strip().lower()
    if verl_main_target == "dapo":
        logger.info("> Lumen entry: delegating to recipe.dapo.main_dapo (DAPO dynamic sampling)")
        from recipe.dapo.main_dapo import main as verl_main
    else:
        from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
