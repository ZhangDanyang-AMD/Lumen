# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3-30B-A3B pretraining on Megatron-LM with SonicMoE experts over MORI EP.

Launched by ``run_pretrain_qwen3_30b_a3b_megatron.sh``. On top of the generic
Lumen GPT builder this entry point:

* sets every ``SonicMoEExperts.precision_recipe`` from ``EXPERT_RECIPE``
  (``bf16`` | ``w4a16`` | ``a4w4`` | ``a8w4`` | ``a8w4f``; Lumen has no CLI
  switch for the expert recipe);
* installs the per-optimizer-step lifecycle of the SonicMoE MXFP4 packed-weight
  cache (``lumen.modules.mxfp4_pack_cache``); without it the cache stays off and
  the experts repack their weights on every call;
* serves Megatron's fused RoPE from ``lumen.patches.runtime.megatron_rope``
  (``LUMEN_ROPE_IMPL=apex|aiter``) and turns ``apply_rope_fusion`` on for the
  built model;
* passes ``gpu_per_node`` to MORI's dispatch/combine config, which Megatron
  leaves at MORI's default, so expert parallelism over fewer GPUs than the node
  has builds an intra-node config of the right size, and releases MORI's
  symmetric heap at exit;
* with ``ATTN_AITER_TRITON_MHA=1`` serves ``aiter.ops.mha.flash_attn_func``
  (Lumen's ``csrc`` attention backend) from AITER's Triton MHA, for AITER
  builds without the CK MHA kernels;
* runs MORI dispatch/combine on the current stream (``async_finish=False``,
  ``allocate_on_comm_stream=False``), the path the reference runs used; with
  Megatron's default communication-stream path the first step does not finish.

MXFP4 attention projections are selected by ``LUMEN_ATTN_LINEAR_RECIPE`` inside
Lumen's builder and need no code here.
"""

import atexit
import functools
import os

from megatron.core.enums import ModelType
from megatron.training import pretrain, print_rank_0

from lumen.models.llama31.megatron import forward_step, train_valid_test_datasets_provider
from lumen.models.megatron import (
    add_common_megatron_args,
    apply_fp8_training,
    apply_lora,
    install_fp8_param_gather_hook,
    make_lumen_model_provider,
)
from lumen.models.megatron import lumen_gpt_builder as _lumen_gpt_builder
from lumen.patches.runtime import megatron_rope

EXPERT_RECIPES = ("bf16", "w4a16", "a4w4", "a8w4", "a8w4f")


def _expert_recipe() -> str:
    recipe = os.environ.get("EXPERT_RECIPE", "bf16").strip().lower()
    if recipe not in EXPERT_RECIPES:
        raise ValueError(f"EXPERT_RECIPE={recipe!r} is not one of {EXPERT_RECIPES}")
    return recipe


def qwen3_moe_gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    return _lumen_gpt_builder(
        args,
        pre_process,
        post_process,
        vp_stage=vp_stage,
        config=config,
        pg_collection=pg_collection,
        model_name="Qwen3-30B-A3B",
    )


def _apply_expert_recipe(model, recipe):
    experts = [m for m in model.modules() if type(m).__name__ == "SonicMoEExperts"]
    if recipe != "bf16" and not experts:
        raise RuntimeError(f"EXPERT_RECIPE={recipe} needs SonicMoE experts (--lumen-sonic-moe)")
    for module in experts:
        module.precision_recipe = recipe
    return len(experts)


def _with_qwen3_moe_setup(provider):
    recipe = _expert_recipe()

    @functools.wraps(provider)
    def model_provider(*args, **kwargs):
        model = provider(*args, **kwargs)
        count = _apply_expert_recipe(model, recipe)
        rope = megatron_rope.enable_on_config(model)
        print_rank_0(
            f"> Qwen3-30B-A3B: expert_recipe={recipe} on {count} SonicMoE expert modules, "
            f"attn_linear_recipe={os.environ.get('LUMEN_ATTN_LINEAR_RECIPE', 'bf16')}, "
            f"rope_impl={megatron_rope.report()['impl']} (apply_rope_fusion={rope})"
        )
        return model

    return model_provider


def _install_pack_cache_lifecycle():
    import megatron.training.training as mt_training
    from megatron.training.utils import unwrap_model

    from lumen.modules.mxfp4_pack_cache import collect_cache_modules, install_mxfp4_pack_cache_lifecycle

    setup = mt_training.setup_model_and_optimizer

    @functools.wraps(setup)
    def setup_model_and_optimizer(*args, **kwargs):
        model, optimizer, scheduler = setup(*args, **kwargs)
        root = unwrap_model(model[0] if isinstance(model, (list, tuple)) else model)
        if collect_cache_modules(root):
            handle = install_mxfp4_pack_cache_lifecycle(root, optimizer)
            print_rank_0(f"> SonicMoE MXFP4 pack cache: {handle}")
        return model, optimizer, scheduler

    mt_training.setup_model_and_optimizer = setup_model_and_optimizer


def _use_aiter_triton_mha():
    # aiter.ops.mha.flash_attn_func dispatches on this module global; the
    # process-wide ENABLE_CK=0 env would also move other AITER ops off CK.
    if os.environ.get("ATTN_AITER_TRITON_MHA", "0") == "1":
        import aiter.ops.mha

        aiter.ops.mha.ENABLE_CK = False


def _install_mori_current_stream():
    from megatron.core.transformer.moe.token_dispatcher import _MoriManager

    def on_current_stream(method):
        @functools.wraps(method)
        def wrapped(self, hidden_states, async_finish=True, allocate_on_comm_stream=True):
            return method(self, hidden_states, False, False)

        return wrapped

    _MoriManager.dispatch = on_current_stream(_MoriManager.dispatch)
    _MoriManager.combine = on_current_stream(_MoriManager.combine)


def _install_mori_gpu_per_node():
    try:
        import mori.ops
    except ImportError:
        return
    config_cls = mori.ops.EpDispatchCombineConfig
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))

    @functools.wraps(config_cls)
    def ep_dispatch_combine_config(**kwargs):
        if local_world_size > 0 and "gpu_per_node" not in kwargs:
            kwargs["gpu_per_node"] = min(int(kwargs["world_size"]), local_world_size)
        return config_cls(**kwargs)

    mori.ops.EpDispatchCombineConfig = ep_dispatch_combine_config

    from megatron.core.transformer.moe import fused_a2a

    finalize = getattr(fused_a2a, "finalize_mori_shmem", None)
    if finalize is not None:
        atexit.register(finalize)


def main():
    model_provider = _with_qwen3_moe_setup(
        make_lumen_model_provider(
            qwen3_moe_gpt_builder,
            lora_applier=apply_lora,
            fp8_applier=apply_fp8_training,
        )
    )
    _use_aiter_triton_mha()
    install_fp8_param_gather_hook()
    _install_pack_cache_lifecycle()
    _install_mori_gpu_per_node()
    _install_mori_current_stream()
    megatron_rope.install()
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_common_megatron_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )


if __name__ == "__main__":
    main()
