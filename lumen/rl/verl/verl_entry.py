###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""VERL + Lumen entrypoint.

This module provides the Lumen-patched VERL training entrypoint. It
monkey-patches VERL's FSDP and Megatron workers to inject Lumen FP8/norm/attn
optimizations into the model build pipeline.

Usage:
    python -m lumen.rl.verl.verl_entry --config-path configs/grpo_fsdp_lumen.yaml
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def patch_verl_fsdp_workers(lumen_args):
    """Monkey-patch verl.workers.fsdp_workers to inject Lumen optimizations.

    Hooks into apply_fsdp2 to inject Lumen FP8/norm/attn optimizations
    right before FSDP2 sharding is applied to the actor model.

    When fp8_weight_cache is enabled, also hooks into the optimizer
    creation to register post-step hooks that refresh FP8 weight caches.
    """
    import verl.workers.fsdp_workers as fsdp_module

    _original_apply_fsdp2 = fsdp_module.apply_fsdp2

    def _patched_apply_fsdp2(model, fsdp_kwargs, fsdp_config):
        from lumen.config import LumenConfig
        logger.info("> Applying Lumen optimizations before FSDP2 wrapping")
        cfg = LumenConfig.from_args(lumen_args)
        cfg.enable(model)
        return _original_apply_fsdp2(model, fsdp_kwargs, fsdp_config)

    fsdp_module.apply_fsdp2 = _patched_apply_fsdp2

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


def main():
    """Lumen-aware VERL entrypoint.

    Parses lumen-specific config from the VERL YAML, patches the workers,
    then delegates to VERL's standard training loop.
    """
    from lumen.rl.verl.config import VerlLumenArgs

    lumen_fp8 = os.environ.get("LUMEN_FP8", "0") == "1"
    lumen_norm = os.environ.get("LUMEN_NORM", "0") == "1"
    lumen_fp8_attn = os.environ.get("LUMEN_FP8_ATTN", "none")
    lumen_fp8_weight_cache = os.environ.get("LUMEN_FP8_WEIGHT_CACHE", "0") == "1"
    lumen_fp8_activation_store = os.environ.get("LUMEN_FP8_ACTIVATION_STORE", "0") == "1"
    lumen_fp8_param_gather = os.environ.get("LUMEN_FP8_PARAM_GATHER", "0") == "1"
    fp8_param_manager = os.environ.get("FP8_PARAM_MANAGER", "0") == "1"
    use_8bit_adam = os.environ.get("USE_8BIT_ADAM", "0") == "1"
    model_path = os.environ.get("MODEL_NAME", "")

    any_lumen = (
        lumen_fp8 or lumen_norm or lumen_fp8_attn != "none"
        or lumen_fp8_weight_cache or lumen_fp8_activation_store
        or lumen_fp8_param_gather or fp8_param_manager
    )
    if any_lumen:
        lumen_args = VerlLumenArgs(
            model_name_or_path=model_path,
            linear_fp8=lumen_fp8,
            lumen_norm=lumen_norm,
            lumen_fp8_attn=lumen_fp8_attn,
            lumen_fp8_weight_cache=lumen_fp8_weight_cache,
            lumen_fp8_activation_store=lumen_fp8_activation_store,
            lumen_fp8_param_gather=lumen_fp8_param_gather,
            fp8_param_manager=fp8_param_manager,
            use_8bit_adam=use_8bit_adam,
        )
        patch_verl_fsdp_workers(lumen_args)
        patch_verl_megatron_workers(lumen_args)

    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
