###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSpeed integration helpers for Lumen quantized training.

``LumenConfig.enable`` remains the only model-patching entry point.  This
module owns only DeepSpeed-specific validation, model-provider assembly, and
runtime wiring.

The initial support contract is deliberately narrow: ZeRO-1/2 with BF16
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
    if stage not in (1, 2):
        raise ValueError(
            f"Lumen DeepSpeed currently supports ZeRO stages 1 and 2; got stage {stage}. "
            "ZeRO-3 parameter materialization is not validated yet."
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
    if qcfg.is_quantized:
        if config.lumen_linear:
            raise ValueError(
                "lumen_linear is the BF16 Linear path and cannot be combined with "
                "DeepSpeed FP8 Linear quantization"
            )
        if config.format != "fp8_e4m3":
            raise ValueError(
                "DeepSpeed quantized training currently supports format='fp8_e4m3' only"
            )
        if config.scaling not in ("dynamic", "blockwise2d"):
            raise ValueError(
                "DeepSpeed quantized training currently supports scaling='dynamic' "
                "or 'blockwise2d' only"
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
    manager, model = config.enable(model, dp_group=dp_group, backend=backend)

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

    The provider is the canonical extension point for future ZeRO-3 support.
    The current contract accepts only ZeRO-1/2, so model construction uses a
    null context; once ZeRO-3 is validated this is where ``zero.Init`` belongs.
    """
    validate_deepspeed_quant_config(config, deepspeed_config)

    def model_provider(*args, **kwargs):
        with nullcontext():
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


def register_deepspeed_quant_hooks(engine, manager) -> None:
    """Attach runtime state that only exists after engine initialization."""
    if manager is None:
        return

    group = getattr(engine, "data_parallel_group", None)
    if group is None:
        optimizer = getattr(engine, "optimizer", None)
        group = getattr(optimizer, "dp_process_group", None)
    if group is not None and manager.config.reduce_amax:
        manager.set_dp_group(group)


def build_deepspeed_config(args, *, world_size: int) -> dict[str, Any]:
    """Build the supported ZeRO-1/2 configuration from training arguments."""
    train_batch_size = (
        args.micro_batch_size * args.gradient_accumulation_steps * world_size
    )
    return {
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_batch_size": train_batch_size,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": args.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        },
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
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )
    register_deepspeed_quant_hooks(engine, manager)
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
    training.add_argument("--save-interval", type=int, default=0)
    training.add_argument("--save-dir", type=str, default="./checkpoints")
    training.add_argument("--seed", type=int, default=1234)
    training.add_argument("--num-workers", type=int, default=2)
    add_fsdp_runtime_contract_args(training)

    deepspeed_group = parser.add_argument_group("deepspeed")
    deepspeed_group.add_argument("--zero-stage", type=int, choices=[1, 2], default=2)
    deepspeed_group.add_argument(
        "--linear-backend",
        choices=["aiter", "triton", "auto"],
        default="aiter",
        help="Lumen quantization backend. Use 'aiter' to require AITER.",
    )
    deepspeed_group.add_argument("--resume-from-checkpoint", type=str, default=None)
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
