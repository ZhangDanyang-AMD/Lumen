###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared FSDP training helpers for Lumen models.

This module consolidates FP8 and LoRA helper functions **and common CLI
argument groups** that are shared by all PyTorch-FSDP-based training scripts
(LLaMA2 SFT, LLaMA 3.1 pretraining, etc.).  Model-specific code (model
building, trainer class, model-specific CLI args) remains in the per-model
subpackages.
"""

import logging

import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Common CLI argument groups
# ---------------------------------------------------------------------------


def add_common_fsdp_args(parser):
    """Register CLI argument groups shared by all FSDP model scripts.

    Covers: backend, training basics, data, FSDP sharding, LoRA,
    linear-fp8 training, warmup / early-stop.

    Model-specific arguments (architecture sizes, checkpoint management,
    etc.) should be registered **after** calling this function.
    """
    parser.add_argument(
        "--backend",
        type=str,
        default="fsdp",
        choices=["megatron", "fsdp"],
        help="Training backend.",
    )

    # -- Training --
    t = parser.add_argument_group("training")
    t.add_argument("--micro-batch-size", type=int, default=1)
    t.add_argument("--gradient-accumulation-steps", type=int, default=8)
    t.add_argument("--max-steps", type=int, default=800)
    t.add_argument("--lr", type=float, default=4e-4)
    t.add_argument("--min-lr", type=float, default=0.0)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--log-interval", type=int, default=10)
    t.add_argument("--save-interval", type=int, default=0, help="Save checkpoint every N steps. 0 = disabled.")
    t.add_argument("--save-dir", type=str, default="./checkpoints")
    t.add_argument("--num-workers", type=int, default=4)

    # -- Data --
    d = parser.add_argument_group("data")
    d.add_argument("--train-data-path", type=str, default=None)
    d.add_argument("--val-data-path", type=str, default=None)
    d.add_argument("--train-samples", type=int, default=10000)
    d.add_argument("--val-samples", type=int, default=500)

    # -- FSDP --
    f = parser.add_argument_group("fsdp")
    f.add_argument(
        "--sharding-strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard"],
    )

    # -- LoRA --
    lora = parser.add_argument_group("lora")
    lora.add_argument("--lora-rank", type=int, default=0, help="LoRA rank. 0 = disabled (full fine-tuning).")
    lora.add_argument("--lora-alpha", type=float, default=32.0)
    lora.add_argument("--lora-dropout", type=float, default=0.1)

    # -- Linear FP8 training --
    lfp8 = parser.add_argument_group("linear-fp8")
    lfp8.add_argument(
        "--linear-fp8", action="store_true", default=False, help="Enable FP8 quantised training for Linear layers."
    )
    lfp8.add_argument(
        "--linear-fp8-format", type=str, default="fp8_e4m3", choices=["fp8_e4m3", "fp8_e5m2", "hybrid", "mxfp8"]
    )
    lfp8.add_argument(
        "--linear-fp8-scaling",
        type=str,
        default="delayed",
        choices=["dynamic", "delayed", "blockwise", "per_token", "none"],
    )
    lfp8.add_argument("--linear-fp8-block-size", type=int, default=128)
    lfp8.add_argument("--linear-fp8-amax-algo", type=str, default="max", choices=["max", "most_recent"])
    lfp8.add_argument("--linear-fp8-reduce-amax", action="store_true", default=False)
    lfp8.add_argument("--linear-fp8-amax-history", type=int, default=16)
    lfp8.add_argument("--linear-fp8-margin", type=int, default=0, help="Margin for FP8 scaling factor computation.")
    lfp8.add_argument("--linear-fp8-activation", action="store_true", default=True)
    lfp8.add_argument("--no-linear-fp8-activation", dest="linear_fp8_activation", action="store_false")
    lfp8.add_argument("--linear-fp8-wgrad", action="store_true", default=True)
    lfp8.add_argument(
        "--no-linear-fp8-wgrad",
        dest="linear_fp8_wgrad",
        action="store_false",
        help="Execute weight gradient GEMM in higher precision (BF16) even for FP8 runs.",
    )
    lfp8.add_argument(
        "--grad-quant-type",
        type=str,
        default=None,
        choices=["fp8", "mxfp8", "fp4"],
        help="Gradient quantization type (None=disabled). " "Applies to Linear, Attention, and RMSNorm.",
    )
    lfp8.add_argument(
        "--first-last-layers-bf16",
        action="store_true",
        default=False,
        help="Keep first and last N transformer layers in BF16 during FP8 training.",
    )
    lfp8.add_argument("--num-layers-at-start-in-bf16", type=int, default=1)
    lfp8.add_argument("--num-layers-at-end-in-bf16", type=int, default=1)

    # -- Norm replacement --
    norm = parser.add_argument_group("norm")
    norm.add_argument(
        "--tl-norm",
        action="store_true",
        default=False,
        help="Replace all norm modules (RMSNorm and LayerNorm) with Lumen implementations.",
    )

    # -- Warmup + Early stopping --
    wes = parser.add_argument_group("warmup-early-stop")
    wes.add_argument("--warmup-steps", type=int, default=0)
    wes.add_argument("--val-loss-target", type=float, default=None)

    return parser


# ---------------------------------------------------------------------------
# FP8 quantised training
# ---------------------------------------------------------------------------


def patch_norms(model: nn.Module, args) -> None:
    """Replace norm modules in the model with Lumen implementations.

    Works for both HuggingFace and other FSDP-compatible models.
    """
    if not getattr(args, "tl_norm", False):
        return

    from lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

    grad_quant_type = getattr(args, "grad_quant_type", None)
    count = 0
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in ("RMSNorm", "LlamaRMSNorm", "MistralRMSNorm", "Qwen2RMSNorm"):
                hidden_size = child.weight.shape[0]
                eps = getattr(child, "eps", getattr(child, "variance_epsilon", 1e-6))
                replacement = LumenRMSNorm(hidden_size, eps=eps, grad_quant_type=grad_quant_type)
                replacement.weight.data.copy_(child.weight.data)
                setattr(module, attr_name, replacement)
                count += 1
            elif cls_name in ("LayerNorm",):
                hidden_size = child.weight.shape[0] if child.weight is not None else child.normalized_shape[0]
                eps = getattr(child, "eps", 1e-5)
                replacement = LumenLayerNorm(hidden_size, eps=eps, grad_quant_type=grad_quant_type)
                if child.weight is not None:
                    replacement.weight.data.copy_(child.weight.data)
                if hasattr(child, "bias") and child.bias is not None and replacement.bias is not None:
                    replacement.bias.data.copy_(child.bias.data)
                setattr(module, attr_name, replacement)
                count += 1

    _rank0_print(f"> Replaced {count} norm modules with Lumen implementations")


def apply_fp8_training(model: nn.Module, args, dp_group=None) -> None:
    """Enable FP8 quantised training via Lumen.

    Args:
        dp_group: Data-parallel process group used for amax reduction.
            If ``None`` and ``args.fp8_reduce_amax`` is true, falls back
            to ``dist.group.WORLD``.
    """
    import lumen.quantize as quant
    from lumen.quantize import (
        AmaxAlgo,
        QuantConfig,
        QuantFormat,
        ScalingType,
    )

    fmt = getattr(args, "linear_fp8_format", "fp8_e4m3")
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
    use_sdma = getattr(args, "use_sdma", False)

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
        use_sdma=use_sdma,
    )

    if dp_group is None and config.reduce_amax and dist.is_initialized():
        dp_group = dist.group.WORLD

    # Patch norms before enabling quant
    patch_norms(model, args)

    quant.enable(
        model,
        config=config,
        dp_group=dp_group if config.reduce_amax else None,
    )
    bf16_str = f", first_last_bf16=start:{bf16_start}/end:{bf16_end}" if first_last_bf16 else ""
    _rank0_print(
        f"> FP8 training enabled (format={fmt}, scaling={scaling}, "
        f"amax_algo={amax_algo}, activation={quant_act}, "
        f"fp8_wgrad={fp8_wgrad}, grad_quant={grad_quant_type}{bf16_str})"
    )


def reset_fp8_state(model: nn.Module) -> None:
    """Reset FP8 scaling state after warmup.

    Unwraps nested ``.module`` attributes (e.g. from FSDP or DDP wrappers)
    before calling :meth:`apply`.
    """

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
    _rank0_print("> FP8 state reset after warmup")


# ---------------------------------------------------------------------------
# LoRA (via HuggingFace PEFT)
# ---------------------------------------------------------------------------


def apply_lora(model: nn.Module, args) -> nn.Module:
    """Apply LoRA adapters via HuggingFace PEFT and freeze the base model."""
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _rank0_print(
        f"> LoRA applied (rank={args.lora_rank}, alpha={args.lora_alpha}) — "
        f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    return model
