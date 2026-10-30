###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSpeed integration helpers for Lumen quantized training.

``LumenConfig.enable`` remains the only model-patching entry point.  This
module owns only DeepSpeed-specific validation, model-provider assembly, and
runtime wiring.

The support contract covers ZeRO-1/2/3 with BF16
master parameters and Lumen FP8 E4M3 Linear compute using dynamic or
blockwise2d scaling.  Parameter/communication quantization and offload are not
yet safe under DeepSpeed and are rejected before the model is patched.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Callable, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from lumen.config import LumenConfig
from lumen.models.training_contract import (
    add_fsdp_fp8_contract_args,
    add_fsdp_runtime_contract_args,
)

__all__ = [
    "add_common_deepspeed_args",
    "build_deepspeed_config",
    "get_zero_stage",
    "initialize_deepspeed",
    "make_lumen_deepspeed_model_provider",
    "prepare_deepspeed_model",
    "register_deepspeed_quant_hooks",
    "validate_deepspeed_quant_config",
]


def _load_deepspeed():
    try:
        import deepspeed
    except ImportError as exc:
        raise ImportError(
            "DeepSpeed support requires the optional 'deepspeed' package. "
            "Install the ROCm-compatible DeepSpeed build used by your image."
        ) from exc
    return deepspeed


def _as_dict(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if hasattr(config, "config") and isinstance(config.config, Mapping):
        return config.config
    raise TypeError("deepspeed_config must be a mapping or expose a mapping '.config'")


def get_zero_stage(deepspeed_config: Mapping[str, Any]) -> int:
    """Return the configured ZeRO stage (0 when ZeRO is absent)."""
    zero = _as_dict(deepspeed_config).get("zero_optimization", {})
    if isinstance(zero, int):
        return int(zero)
    if not isinstance(zero, Mapping):
        raise TypeError("DeepSpeed 'zero_optimization' must be an int or mapping")
    return int(zero.get("stage", 0))


def _offload_enabled(zero: Mapping[str, Any], key: str) -> bool:
    value = zero.get(key)
    if not value:
        return False
    if isinstance(value, Mapping):
        return str(value.get("device", "none")).lower() != "none"
    return bool(value)


def validate_deepspeed_quant_config(
    config: LumenConfig,
    deepspeed_config: Mapping[str, Any],
) -> None:
    """Validate the initial DeepSpeed support contract before model mutation."""
    ds_config = _as_dict(deepspeed_config)
    stage = get_zero_stage(ds_config)
    if stage not in (1, 2, 3):
        raise ValueError(
            f"Lumen DeepSpeed supports ZeRO stages 1, 2, and 3; got stage {stage}."
        )

    zero = ds_config.get("zero_optimization", {})
    zero = zero if isinstance(zero, Mapping) else {}
    if _offload_enabled(zero, "offload_param") or _offload_enabled(zero, "offload_optimizer"):
        raise ValueError("DeepSpeed CPU/NVMe parameter or optimizer offload is not supported yet")

    native_fp8 = ds_config.get("fp8", {})
    native_fp8_enabled = (
        native_fp8.get("enabled", False)
        if isinstance(native_fp8, Mapping)
        else bool(native_fp8)
    )
    if native_fp8_enabled:
        raise ValueError("DeepSpeed native FP8 and Lumen FP8 cannot be enabled together")

    unsupported = {
        "fp8_param_manager": config.fp8_param_manager,
        "fp8_param_gather": config.fp8_param_gather,
        "fp8_weight_cache": config.fp8_weight_cache,
        "cpu_offload": config.cpu_offload,
        "gradient quantization": config.quantize_grad is not None,
        "FP8 attention": config.fp8_attn != "none",
        "LoRA": config.lora_rank > 0,
        "HIP graphs": config.hip_graphs,
        "delayed weight gradients": config.delay_wgrad,
        "gradient accumulation fusion": config.gradient_accumulation_fusion,
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError(
            "The initial DeepSpeed integration does not support: " + ", ".join(enabled)
        )

    qcfg = config.quant_config
    if stage == 3 and qcfg.is_quantized:
        raise ValueError(
            "DeepSpeed ZeRO-3 is currently validated for BF16 only; FP8 scaling "
            "state is not yet synchronized with stage-3 parameter release"
        )
    if qcfg.is_quantized:
        if config.lumen_linear:
            raise ValueError(
                "lumen_linear is the BF16 Linear path and cannot be combined with "
                "DeepSpeed FP8 Linear quantization"
            )
        if config.format not in ("fp8_e4m3", "hybrid"):
            raise ValueError(
                "DeepSpeed quantized training currently supports format='fp8_e4m3' "
                "or 'hybrid' only"
            )
        if config.scaling not in ("dynamic", "delayed", "blockwise2d"):
            raise ValueError(
                "DeepSpeed quantized training currently supports scaling='dynamic' "
                "'delayed', or 'blockwise2d' only"
            )
        if config.scaling == "delayed" and config.format != "hybrid":
            raise ValueError(
                "DeepSpeed delayed scaling on the ROCm 7.0 image requires "
                "format='hybrid'"
            )
        if config.scaling != "delayed" and config.format != "fp8_e4m3":
            raise ValueError(
                "DeepSpeed hybrid format is currently supported only with delayed scaling"
            )

    if os.environ.get("LUMEN_WEIGHT_QUANT_ONCE", "0") == "1":
        raise ValueError(
            "LUMEN_WEIGHT_QUANT_ONCE is not supported by the initial DeepSpeed "
            "integration because cache invalidation is not wired to ZeRO updates"
        )


def _parameter_ids(model: nn.Module) -> dict[str, int]:
    return {name: id(param) for name, param in model.named_parameters()}


def prepare_deepspeed_model(
    model: nn.Module,
    config: LumenConfig,
    deepspeed_config: Mapping[str, Any],
    *,
    dp_group=None,
    backend: str = "aiter",
):
    """Apply Lumen before ``deepspeed.initialize`` and preserve parameters."""
    validate_deepspeed_quant_config(config, deepspeed_config)
    before = _parameter_ids(model)
    original_model = model
    config._preserve_parameter_identity = True
    try:
        manager, model = config.enable(model, dp_group=dp_group, backend=backend)
    finally:
        del config._preserve_parameter_identity

    if model is not original_model:
        raise RuntimeError(
            "LumenConfig.enable() replaced the model object. The initial DeepSpeed "
            "path requires an in-place model patch."
        )
    after = _parameter_ids(model)
    if before != after:
        raise RuntimeError(
            "LumenConfig.enable() changed Parameter identity before DeepSpeed "
            "initialization; this configuration is not supported"
        )
    return manager, model


def make_lumen_deepspeed_model_provider(
    model_builder: Callable[..., nn.Module],
    config: LumenConfig,
    deepspeed_config: Mapping[str, Any],
    *,
    backend: str = "aiter",
) -> Callable[..., tuple[Any, nn.Module]]:
    """Return a Megatron-style provider that builds and patches one model.

    ZeRO-3 constructs the model inside ``deepspeed.zero.Init`` so parameters
    are partitioned as they are created instead of materializing the full model
    on every rank.
    """
    validate_deepspeed_quant_config(config, deepspeed_config)

    def model_provider(*args, **kwargs):
        if get_zero_stage(deepspeed_config) == 3:
            deepspeed = _load_deepspeed()
            build_context = deepspeed.zero.Init(
                config_dict_or_path=dict(_as_dict(deepspeed_config)),
            )
        else:
            build_context = nullcontext()
        with build_context:
            model = model_builder(*args, **kwargs)
            dp_group = dist.group.WORLD if config.reduce_amax and dist.is_initialized() else None
            return prepare_deepspeed_model(
                model,
                config,
                deepspeed_config,
                dp_group=dp_group,
                backend=backend,
            )

    return model_provider


def register_deepspeed_quant_hooks(
    engine,
    manager,
    *,
    enable_fp8_weight_cache: bool = False,
) -> None:
    """Attach runtime state that only exists after engine initialization."""
    if manager is None:
        return

    group = getattr(engine, "data_parallel_group", None)
    if group is None:
        optimizer = getattr(engine, "optimizer", None)
        group = getattr(optimizer, "dp_process_group", None)
    if group is not None and manager.config.reduce_amax:
        manager.set_dp_group(group)

    if enable_fp8_weight_cache:
        # ZeRO-1/2 keeps complete parameters resident, so the manager can cache
        # one quantized weight descriptor per optimizer update and reuse it
        # across accumulation micro-batches. The optimizer hook marks every
        # descriptor stale only after DeepSpeed performs a real update.
        manager.enable_fp8_params(engine.module)
    if enable_fp8_weight_cache or manager.needs_optimizer_step_hook:
        manager.register_fp8_optimizer_hook(engine.optimizer)


def build_deepspeed_config(args, *, world_size: int) -> dict[str, Any]:
    """Build the supported ZeRO-1/2/3 configuration."""
    train_batch_size = (
        args.micro_batch_size * args.gradient_accumulation_steps * world_size
    )
    zero_config = {
        "stage": args.zero_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
    }
    if args.zero_stage == 3:
        zero_config.update(
            {
                "stage3_prefetch_bucket_size": 50_000_000,
                "stage3_param_persistence_threshold": 100_000,
                "stage3_max_live_parameters": 1_000_000_000,
                "stage3_max_reuse_distance": 1_000_000_000,
            }
        )
    return {
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_batch_size": train_batch_size,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_optimization": zero_config,
        "gradient_clipping": args.max_grad_norm,
        "zero_allow_untested_optimizer": True,
        "steps_per_print": max(int(args.log_interval), 1),
        "wall_clock_breakdown": False,
    }


def initialize_deepspeed(
    model: nn.Module,
    args,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    manager=None,
):
    """Initialize DeepSpeed around an already Lumen-patched model."""
    deepspeed = _load_deepspeed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    ds_config = build_deepspeed_config(args, world_size=world_size)
    if args.zero_stage == 3 and getattr(args, "deepspeed_fp8_weight_cache", False):
        raise ValueError(
            "DeepSpeed FP8 weight cache is not supported with ZeRO-3 because "
            "parameters are released after each module forward"
        )
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )
    register_deepspeed_quant_hooks(
        engine,
        manager,
        enable_fp8_weight_cache=getattr(
            args, "deepspeed_fp8_weight_cache", False
        ),
    )
    return engine, optimizer, scheduler


def add_common_deepspeed_args(parser):
    """Register model-independent DeepSpeed training and FP8 arguments."""
    training = parser.add_argument_group("training")
    training.add_argument("--micro-batch-size", type=int, default=1)
    training.add_argument("--gradient-accumulation-steps", type=int, default=1)
    training.add_argument("--max-steps", type=int, default=50)
    training.add_argument("--lr", type=float, default=1e-5)
    training.add_argument("--min-lr", type=float, default=0.0)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument("--log-interval", type=int, default=1)
    training.add_argument(
        "--step-time-atol",
        type=float,
        default=0.0,
        help="Fail when steady-state mean step time exceeds this many milliseconds.",
    )
    training.add_argument(
        "--performance-warmup-steps",
        type=int,
        default=2,
        help="Number of initial steps excluded from the performance mean.",
    )
    training.add_argument(
        "--profile-step",
        type=int,
        default=0,
        help="Capture a rank-0 torch profiler table for this global step.",
    )
    training.add_argument(
        "--profile-output",
        type=str,
        default="",
        help="Profiler table output path (empty chooses a file under save-dir).",
    )
    training.add_argument("--save-interval", type=int, default=0)
    training.add_argument("--save-dir", type=str, default="./checkpoints")
    training.add_argument("--seed", type=int, default=1234)
    training.add_argument("--num-workers", type=int, default=2)
    add_fsdp_runtime_contract_args(training)

    deepspeed_group = parser.add_argument_group("deepspeed")
    deepspeed_group.add_argument(
        "--zero-stage",
        type=int,
        choices=[1, 2, 3],
        default=2,
    )
    deepspeed_group.add_argument(
        "--linear-backend",
        choices=["auto"],
        default="auto",
        help="Use Lumen's validated automatic AITER/Triton fallback chain.",
    )
    deepspeed_group.add_argument("--resume-from-checkpoint", type=str, default=None)
    deepspeed_group.add_argument(
        "--deepspeed-fp8-weight-cache",
        action="store_true",
        default=False,
        help="Quantize each ZeRO-1/2 FP8 weight once per optimizer update.",
    )
    deepspeed_group.add_argument(
        "--hf-attn-patch",
        action="store_true",
        default=False,
        help="Route HuggingFace SDPA through Lumen/AITER attention.",
    )
    deepspeed_group.add_argument(
        "--no-hf-attn-patch",
        dest="hf_attn_patch",
        action="store_false",
    )
    deepspeed_group.add_argument(
        "--lumen-fused-rope",
        action="store_true",
        default=False,
        help="Patch HuggingFace LLaMA RoPE with the AITER autograd kernel.",
    )
    deepspeed_group.add_argument(
        "--no-lumen-fused-rope",
        dest="lumen_fused_rope",
        action="store_false",
    )
    deepspeed_group.add_argument(
        "--lumen-fused-mlp",
        action="store_true",
        default=False,
        help="Fuse HuggingFace LLaMA SwiGLU activation forward/backward.",
    )
    deepspeed_group.add_argument(
        "--no-lumen-fused-mlp",
        dest="lumen_fused_mlp",
        action="store_false",
    )
    deepspeed_group.add_argument(
        "--lumen-norm",
        action="store_true",
        default=False,
        help="Patch RMSNorm/LayerNorm forward in place with Lumen kernels.",
    )
    deepspeed_group.add_argument(
        "--no-lumen-norm",
        dest="lumen_norm",
        action="store_false",
    )
    deepspeed_group.add_argument(
        "--lumen-linear",
        action="store_true",
        default=False,
        help="Route BF16 nn.Linear forward through Lumen/AITER GEMM.",
    )

    fp8 = parser.add_argument_group("linear-fp8")
    add_fsdp_fp8_contract_args(fp8)
    fp8.add_argument("--linear-fp8-wgrad", action="store_true", default=True)
    fp8.add_argument(
        "--no-linear-fp8-wgrad",
        dest="linear_fp8_wgrad",
        action="store_false",
    )
    fp8.add_argument(
        "--linear-fp8-quantize-output-layer",
        action="store_true",
        default=False,
    )
    return parser
