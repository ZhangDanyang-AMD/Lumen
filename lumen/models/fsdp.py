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

import torch
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
    f.add_argument(
        "--fsdp-version",
        type=int,
        default=1,
        choices=[1, 2],
        help="FSDP version: 1 (legacy FullyShardedDataParallel) or 2 (fully_shard API).",
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

    # -- Attention FP8 --
    afp8 = parser.add_argument_group("attention-fp8")
    afp8.add_argument(
        "--lumen-attn-backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "csrc", "asm"],
        help="Lumen attention kernel backend. 'auto' prefers csrc with triton fallback.",
    )
    afp8.add_argument(
        "--lumen-fp8-attn",
        type=str,
        default="none",
        choices=["none", "dpa", "mha"],
        help="FP8 attention scope: 'none' = BF16, 'dpa' = FP8 dot-product, " "'mha' = FP8 for full MHA block.",
    )
    afp8.add_argument(
        "--lumen-fp8-quant-type",
        type=str,
        default="blockwise",
        choices=["dynamic", "delayed", "blockwise", "blockwise2d", "per_token", "none", "mxfp8"],
        help="FP8 quantisation type for FP8 attention backends.",
    )

    # -- Context parallelism --
    cp = parser.add_argument_group("context-parallelism")
    cp.add_argument(
        "--lumen-cp-comm-type",
        type=str,
        default="a2a",
        choices=["a2a", "p2p"],
        help="Context parallelism comm type: 'a2a' or 'p2p' (ring).",
    )

    # -- Comm overlap --
    overlap = parser.add_argument_group("comm-overlap")
    overlap.add_argument(
        "--lumen-tp-comm-overlap",
        action="store_true",
        default=False,
        help="Overlap TP communication (SDMA) with GEMM computation.",
    )

    # -- Fused MLP --
    fmlp = parser.add_argument_group("fused-mlp")
    fmlp.add_argument(
        "--lumen-fused-mlp",
        action="store_true",
        default=False,
        help="Use fused MLP modules for reduced kernel launch overhead.",
    )

    # -- FP8 activation store --
    fp8act = parser.add_argument_group("fp8-activation-store")
    fp8act.add_argument(
        "--lumen-fp8-activation-store",
        action="store_true",
        default=False,
        help="Store MLP activations in FP8 during forward.",
    )

    # -- CPU offload --
    cpuoff = parser.add_argument_group("cpu-offload")
    cpuoff.add_argument(
        "--lumen-cpu-offload",
        action="store_true",
        default=False,
        help="Offload activations to CPU during forward, prefetch in backward.",
    )

    # -- Delay wgrad --
    dwg = parser.add_argument_group("delay-wgrad")
    dwg.add_argument(
        "--lumen-delay-wgrad",
        action="store_true",
        default=False,
        help="Defer weight gradient to overlap with next layer comm.",
    )

    # -- Gradient accumulation fusion --
    gaf = parser.add_argument_group("grad-accum-fusion")
    gaf.add_argument(
        "--lumen-gradient-accumulation-fusion",
        action="store_true",
        default=False,
        help="Fuse weight gradient accumulation into GEMM backward.",
    )

    # -- FP8 param all-gather --
    fp8p = parser.add_argument_group("fp8-params")
    fp8p.add_argument(
        "--lumen-fp8-param-gather",
        action="store_true",
        default=False,
        help="Store and all-gather parameters in FP8 for reduced comm volume.",
    )

    # -- Fused RoPE --
    rope = parser.add_argument_group("fused-rope")
    rope.add_argument(
        "--lumen-fused-rope",
        action="store_true",
        default=False,
        help="Use AITER fused RoPE kernel for rotary positional embeddings.",
    )

    # -- HIP graphs --
    hg = parser.add_argument_group("hip-graphs")
    hg.add_argument(
        "--lumen-hip-graphs",
        action="store_true",
        default=False,
        help="Graph-capture training steps to reduce kernel launch overhead.",
    )

    # -- FP8 checkpoint --
    ckpt = parser.add_argument_group("fp8-checkpoint")
    ckpt.add_argument(
        "--lumen-fp8-checkpoint",
        action="store_true",
        default=False,
        help="Use FP8-aware activation checkpointing.",
    )

    # -- MoE routing --
    moe = parser.add_argument_group("moe-routing")
    moe.add_argument(
        "--lumen-fused-moe-routing",
        action="store_true",
        default=False,
        help="Use fused MoE token routing.",
    )

    # -- Norm replacement --
    norm = parser.add_argument_group("norm")
    norm.add_argument(
        "--lumen-norm",
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
    if not getattr(args, "lumen_norm", False):
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
    fp8_attn = getattr(args, "lumen_fp8_attn", "none")
    fp8_dpa = fp8_attn in ("dpa", "mha")
    fp8_mha = fp8_attn == "mha"

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
        fp8_dpa=fp8_dpa,
        fp8_mha=fp8_mha,
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


def apply_fsdp2(
    model: nn.Module,
    args,
    dp_group=None,
) -> nn.Module:
    """Apply PyTorch FSDP2 (fully_shard) to the model.

    Uses torch.distributed.fsdp.fully_shard() which provides per-parameter
    sharding with lazy initialization and better composability.

    Args:
        model: The model to shard.
        args: CLI arguments.
        dp_group: Data-parallel process group.

    Returns:
        FSDP2-wrapped model.
    """
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError as e:
        raise ImportError(
            "FSDP2 (fully_shard) requires PyTorch 2.4+. "
            "Install a compatible PyTorch version or use --fsdp-version 1."
        ) from e

    # Build mixed precision policy
    mp_policy = None
    if getattr(args, "linear_fp8", False):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

    # Apply fully_shard to each transformer layer (bottom-up)
    for name, module in model.named_children():
        if hasattr(module, "layers") or "layers" in name:
            for layer_name, layer in module.named_children():
                fully_shard(
                    layer,
                    mesh=dp_group,
                    mp_policy=mp_policy,
                )

    # Shard the top-level model
    fully_shard(
        model,
        mesh=dp_group,
        mp_policy=mp_policy,
    )

    _rank0_print(f"> FSDP2 applied (fully_shard, mp_policy={mp_policy})")
    return model


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
