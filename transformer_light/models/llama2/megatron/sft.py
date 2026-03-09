###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LLaMA2 Supervised Fine-Tuning components for Megatron-LM-AMD.

This module provides the building blocks for SFT on LLaMA2 models using
Megatron-LM-AMD as the training backbone and Transformer Light for the
core dot-product attention (AITER / Triton / FP8).

Features:
    - Full fine-tuning or LoRA (parameter-efficient fine-tuning, with A2A comm opt)
    - FP8 quantised training (weight/activation quantisation via Transformer Light)
    - Transformer Light attention backends: AITER, Triton, Triton-FP8
    - Fine-grained MXFP8 block configuration (6 independent block sizes)
    - Context Parallelism, Tensor Parallelism, Pipeline Parallelism, VP, SP
    - Packed sequences with cross-sample attention boundary tracking (seq_start_id)
    - Answer-only loss masking for SFT
    - Synthetic warmup with FP8 state reset
    - Early stopping based on validation loss target

Example::

    from transformer_light.models.llama2.megatron import (
        add_finetune_args,
        forward_step,
        tl_gpt_builder,
        train_valid_test_datasets_provider,
    )
"""

import logging
import os
from functools import partial
from typing import Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
    is_first_or_last_pipeline_stage,
)
from transformer_light.modules.attention_megatron import (
    TransformerLightDotProductAttention,
)
from transformer_light.models.llama2.dataset import LLaMA2SFTDataset
from transformer_light.models.utils import safe_add_argument

__all__ = [
    "LLaMA2SFTDataset",
    "add_finetune_args",
    "apply_fp8_training",
    "apply_lora",
    "forward_step",
    "get_batch",
    "loss_func",
    "reset_fp8_state",
    "tl_gpt_builder",
    "train_valid_test_datasets_provider",
]

logger = logging.getLogger(__name__)

stimer = StragglerDetector()


# ---------------------------------------------------------------------------
# Custom GPT builder that injects Transformer Light attention
# ---------------------------------------------------------------------------

def tl_gpt_builder(args, pre_process, post_process, vp_stage=None, config=None):
    """Build a GPTModel with Transformer Light attention replacing the default
    DotProductAttention in every layer.

    Always uses the Megatron-Core local spec (no Transformer Engine dependency).
    The core_attention submodule is patched to TransformerLightDotProductAttention.
    When ``--tl-rmsnorm`` is set, RMSNorm modules are also replaced with the
    Triton-accelerated :class:`TransformerLightRMSNorm`.
    """
    print_rank_0("building GPT model with Transformer Light attention ...")

    if config is None:
        args.apply_rope_fusion = False
        config = core_transformer_config_from_args(args)
        config.persist_layer_norm = False
        config.bias_swiglu_fusion = False

    transformer_layer_spec = get_gpt_layer_local_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        args.qk_layernorm,
        args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
        normalization=args.normalization,
        use_kitchen=config.use_kitchen,
    )

    _patch_core_attention(transformer_layer_spec)
    _patch_norms_in_spec(transformer_layer_spec)
    transformer_layer_spec = _wrap_as_block_spec(transformer_layer_spec)
    _force_block_norm_default()

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
    )

    if getattr(args, "tl_rmsnorm", False):
        _patch_rmsnorm(model, getattr(args, "grad_quant_type", None))

    return model


def _patch_core_attention(spec):
    """Recursively walk a ModuleSpec tree and replace every ``core_attention``
    submodule with ``TransformerLightDotProductAttention``."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    if hasattr(spec, "submodules") and spec.submodules is not None:
        subs = spec.submodules
        if hasattr(subs, "self_attention") and subs.self_attention is not None:
            sa = subs.self_attention
            if hasattr(sa, "submodules") and sa.submodules is not None:
                sa_subs = sa.submodules
                if hasattr(sa_subs, "core_attention"):
                    sa_subs.core_attention = ModuleSpec(
                        module=TransformerLightDotProductAttention
                    )
        if hasattr(subs, "layer_specs"):
            for layer_spec in subs.layer_specs:
                _patch_core_attention(layer_spec)


class _MegatronCompatibleTLRMSNorm(torch.nn.Module):
    """Wrapper that adapts :class:`TransformerLightRMSNorm` to the Megatron-Core
    norm construction signature ``(config, hidden_size, eps=...)``.

    Unlike ``WrappedTorchNorm``, this does **not** assert on
    ``persist_layer_norm`` or ``sequence_parallel`` — RMSNorm normalises
    over the hidden dimension only, so each position is independent and
    works correctly with SP-scattered inputs.
    """

    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        from transformer_light.ops.normalization import TransformerLightRMSNorm
        self._norm = TransformerLightRMSNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


_NORM_ATTRS = (
    "input_layernorm", "pre_mlp_layernorm",
    "pre_cross_attn_layernorm", "post_cross_attn_layernorm",
    "final_layernorm",
)


def _patch_norms_in_spec(spec):
    """Replace **all** norm classes in a spec tree with
    :class:`_MegatronCompatibleTLRMSNorm` so that ``WrappedTorchNorm``
    and ``FusedLayerNorm`` (which reject SP / persist_layer_norm /
    RMSNorm) are never instantiated.

    Handles both:
    - Block-level specs (``TransformerBlockSubmodules`` with
      ``layer_specs`` and ``final_layernorm``)
    - Layer-level specs (``ModuleSpec`` with ``submodules``)
    """
    for attr in _NORM_ATTRS:
        if getattr(spec, attr, None) is not None:
            setattr(spec, attr, _MegatronCompatibleTLRMSNorm)

    if hasattr(spec, "submodules") and spec.submodules is not None:
        for attr in _NORM_ATTRS:
            if getattr(spec.submodules, attr, None) is not None:
                setattr(spec.submodules, attr, _MegatronCompatibleTLRMSNorm)

    layer_specs = getattr(spec, "layer_specs", None)
    if layer_specs is None and hasattr(spec, "submodules"):
        layer_specs = getattr(spec.submodules, "layer_specs", None)
    if layer_specs:
        for layer_spec in layer_specs:
            _patch_norms_in_spec(layer_spec)


def _force_block_norm_default():
    """Override the ``TransformerBlockSubmodules`` dataclass default for
    ``final_layernorm`` so that ``FusedLayerNorm`` (which rejects RMSNorm)
    is never instantiated, even when ``TransformerBlock`` creates a default
    ``TransformerBlockSubmodules`` internally."""
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    fields = TransformerBlockSubmodules.__dataclass_fields__
    if "final_layernorm" in fields:
        fields["final_layernorm"].default = _MegatronCompatibleTLRMSNorm


def _wrap_as_block_spec(spec):
    """Wrap a layer-level ``ModuleSpec`` in a ``TransformerBlockSubmodules``
    so that ``TransformerBlock.__init__`` does not default
    ``final_layernorm`` to ``FusedLayerNorm``.

    If *spec* is already a ``TransformerBlockSubmodules``, return it as-is.
    """
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    if isinstance(spec, TransformerBlockSubmodules):
        return spec
    return TransformerBlockSubmodules(
        layer_specs=[spec],
        final_layernorm=_MegatronCompatibleTLRMSNorm,
    )


def _patch_rmsnorm(model, grad_quant_type=None):
    """Replace all Megatron-Core RMSNorm modules with Transformer Light's
    Triton-accelerated :class:`TransformerLightRMSNorm`."""
    from transformer_light.ops.normalization import TransformerLightRMSNorm

    count = 0
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in ("RMSNorm", "MegatronRMSNorm", "TENorm",
                            "_MegatronCompatibleTLRMSNorm"):
                hidden_size = child.weight.shape[0]
                eps = getattr(child, "eps", getattr(child, "epsilon", 1e-6))
                replacement = TransformerLightRMSNorm(
                    hidden_size, eps=eps, grad_quant_type=grad_quant_type,
                )
                replacement.weight.data.copy_(child.weight.data)
                setattr(module, attr_name, replacement)
                count += 1

    print_rank_0(f"> Replaced {count} RMSNorm modules with TransformerLightRMSNorm")


# ---------------------------------------------------------------------------
# LoRA (Parameter-Efficient Fine-Tuning)
# ---------------------------------------------------------------------------

def apply_lora(model: GPTModel, args) -> None:
    """Wrap linear layers with LoRA adapters for parameter-efficient fine-tuning."""
    from megatron.core.transformer.lora_adapter import LoraAdapter

    common = {
        "config": model.config,
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
    }

    if hasattr(model, "embedding") and model.embedding is not None:
        model.embedding.word_embeddings = LoraAdapter(
            model.embedding.word_embeddings, **common
        )

    if hasattr(model, "decoder") and model.decoder is not None:
        for layer in model.decoder.layers:
            layer.self_attention.linear_qkv = LoraAdapter(
                layer.self_attention.linear_qkv, **common
            )
            layer.self_attention.linear_proj = LoraAdapter(
                layer.self_attention.linear_proj, **common
            )
            if hasattr(layer, "mlp") and layer.mlp is not None:
                layer.mlp.linear_fc1 = LoraAdapter(
                    layer.mlp.linear_fc1, **common
                )
                layer.mlp.linear_fc2 = LoraAdapter(
                    layer.mlp.linear_fc2, **common
                )

    if hasattr(model, "output_layer") and model.output_layer is not None:
        model.output_layer = LoraAdapter(model.output_layer, **common)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_rank_0(
        f"> LoRA applied (rank={args.lora_rank}, alpha={args.lora_alpha}) — "
        f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )


# ---------------------------------------------------------------------------
# FP8 quantised training
# ---------------------------------------------------------------------------

def apply_fp8_training(model: GPTModel, args) -> None:
    """Enable FP8 quantised training via Transformer Light's non-invasive patching."""
    import transformer_light.quantize as quant
    from transformer_light.quantize import (
        AmaxAlgo, QuantConfig, QuantFormat, ScalingType,
    )

    fmt = getattr(args, "tl_fp8_format", "fp8_e4m3")
    scaling = getattr(args, "fp8_scaling", "delayed")
    block_size = getattr(args, "fp8_block_size", 128)
    amax_algo = getattr(args, "fp8_amax_algo", "max")
    reduce_amax = getattr(args, "fp8_reduce_amax", False)
    history_len = getattr(args, "fp8_amax_history", 16)
    margin = getattr(args, "fp8_margin", 0)
    quant_act = getattr(args, "fp8_activation", True)
    grad_quant_type = getattr(args, "grad_quant_type", None)

    config = QuantConfig(
        format=QuantFormat(fmt),
        scaling=ScalingType(scaling),
        block_size=block_size,
        amax_algo=AmaxAlgo(amax_algo),
        margin=margin,
        reduce_amax=reduce_amax,
        history_len=history_len,
        quantize_activation=quant_act,
        quantize_grad=grad_quant_type,
    )

    dp_group = None
    if reduce_amax:
        import torch.distributed as dist
        from megatron.core import parallel_state
        if dist.is_initialized():
            dp_group = parallel_state.get_data_parallel_group()

    quant.enable(model, config=config, dp_group=dp_group)
    print_rank_0(
        f"> FP8 training enabled (format={fmt}, scaling={scaling}, "
        f"block_size={block_size}, amax_algo={amax_algo}, "
        f"reduce_amax={reduce_amax}, history={history_len}, "
        f"activation={quant_act}, grad_quant={grad_quant_type})"
    )


# ---------------------------------------------------------------------------
# Synthetic warmup + FP8 state reset
# ---------------------------------------------------------------------------

_warmup_step_counter = 0
_warmup_completed = False


def _get_synthetic_batch(args):
    """Generate a synthetic batch for GPU kernel warmup."""
    seq_length = args.seq_length
    mbs = args.micro_batch_size

    tokens = torch.ones(mbs, seq_length, dtype=torch.long, device="cuda") * 3545
    tokens[:, -1] = 2
    labels = tokens.clone()
    loss_mask = torch.ones(mbs, seq_length, dtype=torch.float, device="cuda")
    loss_mask[:, -1] = 0
    attention_mask = torch.ones(mbs, 1, seq_length, seq_length, dtype=torch.bool, device="cuda")
    position_ids = torch.arange(seq_length, dtype=torch.long, device="cuda").unsqueeze(0).expand(mbs, -1)

    return tokens, labels, loss_mask, attention_mask, position_ids


def reset_fp8_state(model):
    """Reset FP8 scaling state in all Transformer Light quantised layers."""

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
    print_rank_0("> FP8 state reset after warmup")


# ---------------------------------------------------------------------------
# Dataset provider
# ---------------------------------------------------------------------------

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, validation, and test SFT datasets."""
    args = get_args()

    tokenizer_obj = get_tokenizer()
    is_hf = hasattr(tokenizer_obj, "_tokenizer") and hasattr(
        tokenizer_obj._tokenizer, "encode"
    )
    raw_tokenizer = tokenizer_obj._tokenizer if is_hf else tokenizer_obj

    train_path = args.train_data_path[0] if args.train_data_path else None
    valid_path = args.valid_data_path[0] if args.valid_data_path else None
    test_path = args.test_data_path[0] if args.test_data_path else None

    print_rank_0("> building train, validation, and test SFT datasets ...")

    train_ds = LLaMA2SFTDataset(
        train_val_test_num_samples[0], train_path, args.seq_length,
        raw_tokenizer, is_hf,
    )
    valid_ds = LLaMA2SFTDataset(
        train_val_test_num_samples[1], valid_path, args.seq_length,
        raw_tokenizer, is_hf,
    )
    test_ds = LLaMA2SFTDataset(
        train_val_test_num_samples[2], test_path, args.seq_length,
        raw_tokenizer, is_hf,
    )

    print_rank_0("> finished creating SFT datasets ...")
    return train_ds, valid_ds, test_ds


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def get_batch(data_iterator, vp_stage=None):
    """Generate a batch with answer-only loss masking and packed sequence params."""
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    args = get_args()

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(
        ["input_ids", "loss_mask"], data, torch.int64
    )

    tokens_ = data_b["input_ids"]
    tokens = tokens_[:, : args.seq_length].contiguous()
    labels = tokens_[:, 1 : args.seq_length + 1].contiguous()
    answer_loss_mask = data_b["loss_mask"][:, 1 : args.seq_length + 1].contiguous()

    tokenizer = get_tokenizer()
    if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "eos_token_id"):
        eod_token = tokenizer._tokenizer.eos_token_id
    else:
        eod_token = tokenizer.eod

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, eod_token, eod_token,
        args.reset_position_ids, args.reset_attention_mask,
        args.eod_mask_loss, False,
    )
    loss_mask = loss_mask * answer_loss_mask.to(dtype=loss_mask.dtype)

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


# ---------------------------------------------------------------------------
# Loss function + early stopping
# ---------------------------------------------------------------------------

_val_loss_ema: Optional[float] = None
_early_stop_logged = False


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """SFT loss with answer-only masking and optional early stopping."""
    global _val_loss_ema, _early_stop_logged

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    args = get_args()
    val_target = getattr(args, "val_loss_target", None)
    if val_target is not None and not _early_stop_logged:
        avg_loss = (loss.clone().detach() / max(num_tokens.item(), 1)).item()
        if _val_loss_ema is None:
            _val_loss_ema = avg_loss
        else:
            _val_loss_ema = 0.9 * _val_loss_ema + 0.1 * avg_loss
        if _val_loss_ema < val_target:
            print_rank_0(
                f"> [Early Stop] Loss EMA ({_val_loss_ema:.4f}) < "
                f"target ({val_target:.4f}). "
                f"Setting train_iters to current iteration to stop training."
            )
            if hasattr(args, "iteration"):
                args.train_iters = args.iteration
            _early_stop_logged = True

    return loss, num_tokens, {"lm loss": reporting}


# ---------------------------------------------------------------------------
# Forward step
# ---------------------------------------------------------------------------

def forward_step(data_iterator, model: GPTModel):
    """Forward step for SFT training with optional synthetic warmup."""
    global _warmup_step_counter, _warmup_completed

    args = get_args()
    timers = get_timers()
    warmup_steps = getattr(args, "warmup_steps", 0)

    timers("batch-generator", log_level=2).start()
    global stimer
    with stimer(bdata=True):
        if warmup_steps > 0 and not _warmup_completed:
            _warmup_step_counter += 1
            if _warmup_step_counter <= warmup_steps:
                tokens, labels, loss_mask, attention_mask, position_ids = (
                    _get_synthetic_batch(args)
                )
                if data_iterator is not None:
                    try:
                        next(data_iterator)
                    except StopIteration:
                        pass
            else:
                if getattr(args, "fp8_training", False):
                    reset_fp8_state(model)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                _warmup_completed = True
                print_rank_0(
                    f"> Synthetic warmup complete ({warmup_steps} steps). "
                    f"Resuming with real data."
                )
                vp_stage = get_attr_wrapped_model(model, "vp_stage")
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
                    data_iterator, vp_stage
                )
        else:
            vp_stage = get_attr_wrapped_model(model, "vp_stage")
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
                data_iterator, vp_stage
            )
    timers("batch-generator").stop()

    with stimer:
        output_tensor = model(
            tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
        )

    return output_tensor, partial(loss_func, loss_mask, model=model)


# ---------------------------------------------------------------------------
# Extra CLI arguments
# ---------------------------------------------------------------------------

def add_finetune_args(parser):
    """Add finetune-specific arguments."""

    safe_add_argument(parser, "--backend", type=str, default="megatron",
                       choices=["megatron", "fsdp"], help="Training backend.")

    tl = parser.add_argument_group(title="transformer-light")
    safe_add_argument(
        tl, "--tl-attn-backend", type=str, default="aiter",
        choices=["aiter", "triton", "triton_fp8"],
        help="Transformer Light attention backend.",
    )
    safe_add_argument(
        tl, "--tl-fp8-quant-type", type=str, default="fp8_blockwise",
        choices=["fp8_blockwise", "mxfp8"],
        help="FP8 quantisation type for triton_fp8 backend.",
    )
    safe_add_argument(
        tl, "--tl-rmsnorm", action="store_true", default=False,
        help="Replace RMSNorm with Transformer Light Triton-accelerated RMSNorm.",
    )

    mxfp8 = parser.add_argument_group(title="mxfp8-block-config")
    safe_add_argument(mxfp8, "--mxfp8-block-m-fwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-fwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-m-dq-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-dq-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-m-dkv-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-dkv-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-quant-block-size", type=int, default=128)

    lora = parser.add_argument_group(title="lora")
    safe_add_argument(lora, "--lora-rank", type=int, default=0,
                       help="LoRA rank. 0 = disabled (full fine-tuning).")
    safe_add_argument(lora, "--lora-alpha", type=float, default=32.0)
    safe_add_argument(lora, "--lora-dropout", type=float, default=0.1)
    safe_add_argument(lora, "--lora-a2a", action="store_true", default=False,
                       help="Enable LoRA all-to-all communication optimisation.")

    fp8 = parser.add_argument_group(title="fp8-training")
    safe_add_argument(fp8, "--fp8-training", action="store_true", default=False)
    safe_add_argument(fp8, "--tl-fp8-format", type=str, default="fp8_e4m3",
                       choices=["fp8_e4m3", "fp8_e5m2", "hybrid", "mxfp8"])
    safe_add_argument(fp8, "--fp8-scaling", type=str, default="delayed",
                       choices=["dynamic", "delayed", "blockwise"])
    safe_add_argument(fp8, "--fp8-block-size", type=int, default=128)
    safe_add_argument(fp8, "--fp8-amax-algo", type=str, default="max",
                       choices=["max", "most_recent"])
    safe_add_argument(fp8, "--fp8-reduce-amax", action="store_true", default=False)
    safe_add_argument(fp8, "--fp8-amax-history", type=int, default=16)
    safe_add_argument(fp8, "--fp8-margin", type=int, default=0,
                       help="Margin for FP8 scaling factor computation (TE-compatible).")
    safe_add_argument(fp8, "--fp8-activation", action="store_true", default=True)
    safe_add_argument(fp8, "--no-fp8-activation", dest="fp8_activation",
                       action="store_false")
    safe_add_argument(fp8, "--grad-quant-type", type=str, default=None,
                       choices=["fp8", "mxfp8", "fp4"],
                       help="Gradient quantization type (None=disabled).")

    sft = parser.add_argument_group(title="sft-training")
    safe_add_argument(sft, "--warmup-steps", type=int, default=0)
    safe_add_argument(sft, "--val-loss-target", type=float, default=None)

    return parser
