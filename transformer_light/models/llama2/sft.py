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

    from transformer_light.models.llama2 import (
        add_finetune_args,
        forward_step,
        tl_gpt_builder,
        train_valid_test_datasets_provider,
    )
"""

import json
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional

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
from transformer_light.pytorch.ops.attention.attention_megatron import (
    TransformerLightDotProductAttention,
)

__all__ = [
    "LLaMA2SFTDataset",
    "add_finetune_args",
    "forward_step",
    "get_batch",
    "loss_func",
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
    """
    print_rank_0("building GPT model with Transformer Light attention ...")

    if config is None:
        config = core_transformer_config_from_args(args)

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


# ---------------------------------------------------------------------------
# LoRA (Parameter-Efficient Fine-Tuning)
# ---------------------------------------------------------------------------

def apply_lora(model: GPTModel, args) -> None:
    """Wrap linear layers with LoRA adapters for parameter-efficient fine-tuning.

    Freezes the base model weights and injects low-rank adapter matrices into
    the embedding, attention projections, MLP layers, and output layer.
    """
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
    """Enable FP8 quantised training via Transformer Light's non-invasive patching.

    Reads the following args (with defaults matching MLPerf/TE conventions):
      --fp8-format, --fp8-scaling, --fp8-block-size,
      --fp8-amax-algo, --fp8-reduce-amax, --fp8-amax-history, --fp8-activation
    """
    import transformer_light.quantize as quant
    from transformer_light.quantize import (
        AmaxAlgo, QuantConfig, QuantFormat, ScalingType,
    )

    fmt = getattr(args, "fp8_format", "fp8_e4m3")
    scaling = getattr(args, "fp8_scaling", "delayed")
    block_size = getattr(args, "fp8_block_size", 128)
    amax_algo = getattr(args, "fp8_amax_algo", "max")
    reduce_amax = getattr(args, "fp8_reduce_amax", False)
    history_len = getattr(args, "fp8_amax_history", 16)
    quant_act = getattr(args, "fp8_activation", True)

    config = QuantConfig(
        format=QuantFormat(fmt),
        scaling=ScalingType(scaling),
        block_size=block_size,
        amax_algo=AmaxAlgo(amax_algo),
        reduce_amax=reduce_amax,
        history_len=history_len,
        quantize_activation=quant_act,
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
        f"activation={quant_act})"
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
# SFT Dataset
# ---------------------------------------------------------------------------

class LLaMA2SFTDataset(torch.utils.data.Dataset):
    """SFT dataset that loads jsonl data and packs sequences.

    Each jsonl line should have the format::

        {"input": "<prompt text>", "output": "<completion text>"}

    or::

        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Sequences are tokenized, packed to ``seq_length``, and loss is masked
    so that only the completion (output/assistant) tokens contribute.
    """

    LLAMA2_CHAT_TEMPLATE = "[INST] {input} [/INST] {output}"

    def __init__(
        self,
        num_samples: int,
        data_path: Optional[str],
        seq_length: int,
        tokenizer,
        is_hf_tokenizer: bool = False,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer
        self.indexed_dataset: List[Dict[str, list]] = []
        self._raw_idx = 0

        if data_path is None:
            self._raw_samples = []
            return

        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                self._raw_samples = [json.loads(line) for line in f if line.strip()]
        elif data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                self._raw_samples = json.load(f)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        print_rank_0(f"> Loaded {len(self._raw_samples)} raw SFT samples from {data_path}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        while idx >= len(self.indexed_dataset):
            packed = self._pack_next()
            if packed is None:
                break
            self.indexed_dataset.append(packed)

        idx = idx % max(len(self.indexed_dataset), 1)
        sample = self.indexed_dataset[idx]
        out: Dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            out[k] = torch.LongTensor(v)
        return out

    # -- internal helpers --

    def _tokenize(self, text: str) -> List[int]:
        if self.is_hf_tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.tokenize(text)

    def _get_eos_id(self) -> int:
        if self.is_hf_tokenizer:
            return self.tokenizer.eos_token_id
        return self.tokenizer.eod

    def _process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Tokenize one raw sample and compute the answer-only loss mask."""
        if "messages" in sample:
            messages = sample["messages"]
            if len(messages) < 2:
                return None
            prompt_parts, completion_parts = [], []
            for msg in messages:
                if msg["role"] in ("user", "system"):
                    prompt_parts.append(msg["content"])
                elif msg["role"] == "assistant":
                    completion_parts.append(msg["content"])
            input_text = " ".join(prompt_parts)
            output_text = " ".join(completion_parts)
        elif "input" in sample and "output" in sample:
            input_text = sample["input"]
            output_text = sample["output"]
        else:
            return None

        prompt_str = self.LLAMA2_CHAT_TEMPLATE.format(input=input_text, output="")
        prompt_ids = self._tokenize(prompt_str)
        completion_ids = self._tokenize(output_text)
        eos_id = self._get_eos_id()

        input_ids = prompt_ids + completion_ids + [eos_id]
        loss_mask = [0] * len(prompt_ids) + [1] * len(completion_ids) + [0]

        if len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            loss_mask = loss_mask[: self.seq_length]

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "token_count": len(input_ids),
        }

    def _pack_next(self) -> Optional[Dict[str, list]]:
        """Pack multiple samples into one fixed-length sequence.

        Tracks sample boundaries via ``seq_start_id`` — a list of cumulative
        token offsets marking where each packed sample begins.  This enables
        proper cross-sample attention masking when used with flash-attention
        ``cu_seqlens`` APIs.
        """
        required = self.seq_length + 1
        all_ids: List[int] = []
        all_mask: List[int] = []
        seq_start_id: List[int] = [0]
        total = 0

        while total < required:
            if self._raw_idx >= len(self._raw_samples):
                if total == 0:
                    return None
                break
            sample = self._raw_samples[self._raw_idx]
            self._raw_idx += 1
            processed = self._process_sample(sample)
            if processed is None:
                continue
            all_ids.extend(processed["input_ids"])
            all_mask.extend(processed["loss_mask"])
            total += processed["token_count"]
            seq_start_id.append(total)

        eos_id = self._get_eos_id()
        while len(all_ids) < required:
            all_ids.append(eos_id)
            all_mask.append(0)

        seq_start_id = [min(s, required) for s in seq_start_id]
        if seq_start_id[-1] != required:
            seq_start_id.append(required)

        return {
            "input_ids": all_ids[:required],
            "loss_mask": all_mask[:required],
            "seq_start_id": seq_start_id,
        }


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

    # -- Transformer Light attention ----------------------------------------
    tl = parser.add_argument_group(title="transformer-light-attention")
    tl.add_argument(
        "--tl-attn-backend", type=str, default="aiter",
        choices=["aiter", "triton", "triton_fp8"],
        help="Transformer Light attention backend. "
             "'aiter' uses AITER flash-attention (fastest on MI300X), "
             "'triton' uses Triton flash-attention, "
             "'triton_fp8' uses Triton FP8 quantised attention.",
    )
    tl.add_argument(
        "--tl-fp8-quant-type", type=str, default="fp8_blockwise",
        choices=["fp8_blockwise", "mxfp8"],
        help="FP8 quantisation type for triton_fp8 backend.",
    )

    # -- Fine-grained MXFP8 block configuration ----------------------------
    mxfp8 = parser.add_argument_group(title="mxfp8-block-config")
    mxfp8.add_argument(
        "--mxfp8-block-m-fwd", type=int, default=128,
        help="Block size for query seq dim in MXFP8 forward pass.",
    )
    mxfp8.add_argument(
        "--mxfp8-block-n-fwd", type=int, default=128,
        help="Block size for key/value seq dim in MXFP8 forward pass.",
    )
    mxfp8.add_argument(
        "--mxfp8-block-m-dq-bwd", type=int, default=128,
        help="Block size for dQ seq dim in MXFP8 backward pass.",
    )
    mxfp8.add_argument(
        "--mxfp8-block-n-dq-bwd", type=int, default=128,
        help="Block size for dQ key dim in MXFP8 backward pass.",
    )
    mxfp8.add_argument(
        "--mxfp8-block-m-dkv-bwd", type=int, default=128,
        help="Block size for dKV seq dim in MXFP8 backward pass.",
    )
    mxfp8.add_argument(
        "--mxfp8-block-n-dkv-bwd", type=int, default=128,
        help="Block size for dKV key dim in MXFP8 backward pass.",
    )
    mxfp8.add_argument(
        "--mxfp8-quant-block-size", type=int, default=128,
        help="Quantisation block size for MXFP8 scaling.",
    )

    # -- LoRA / PEFT --------------------------------------------------------
    lora = parser.add_argument_group(title="lora")
    lora.add_argument(
        "--lora-rank", type=int, default=0,
        help="LoRA rank. 0 = disabled (full fine-tuning). Typical: 8, 16, 32.",
    )
    lora.add_argument(
        "--lora-alpha", type=float, default=32.0,
        help="LoRA scaling alpha.",
    )
    lora.add_argument(
        "--lora-dropout", type=float, default=0.1,
        help="Dropout applied to LoRA adapter outputs.",
    )
    lora.add_argument(
        "--lora-a2a", action="store_true", default=False,
        help="Enable all-to-all communication optimisation for LoRA. "
             "Distributes LoRA forward computation across DP ranks. "
             "Effective when multiple GPUs participate in data parallelism.",
    )

    # -- FP8 quantised training (weight/activation) -------------------------
    fp8 = parser.add_argument_group(title="fp8-training")
    fp8.add_argument(
        "--fp8-training", action="store_true", default=False,
        help="Enable FP8 quantised training for linear layers "
             "(weight + activation quantisation via Transformer Light).",
    )
    fp8.add_argument(
        "--fp8-format", type=str, default="fp8_e4m3",
        choices=["fp8_e4m3", "fp8_e5m2", "mxfp8"],
        help="FP8 number format for weight/activation quantisation.",
    )
    fp8.add_argument(
        "--fp8-scaling", type=str, default="delayed",
        choices=["dynamic", "delayed", "blockwise"],
        help="FP8 scaling strategy.",
    )
    fp8.add_argument(
        "--fp8-block-size", type=int, default=128,
        help="Block size for blockwise FP8 scaling.",
    )
    fp8.add_argument(
        "--fp8-amax-algo", type=str, default="max",
        choices=["max", "most_recent"],
        help="Amax algorithm for delayed scaling. "
             "'max' uses the maximum over the entire history window (TE default). "
             "'most_recent' uses only the latest recorded amax (MLPerf default).",
    )
    fp8.add_argument(
        "--fp8-reduce-amax", action="store_true", default=False,
        help="All-reduce amax across data-parallel ranks before computing "
             "the FP8 scale factor. Useful for large-scale runs where per-rank "
             "amax values can diverge.",
    )
    fp8.add_argument(
        "--fp8-amax-history", type=int, default=16,
        help="Length of the amax history window for delayed scaling. "
             "Shorter windows (e.g. 4) react faster to distribution changes.",
    )
    fp8.add_argument(
        "--fp8-activation", action="store_true", default=True,
        help="Quantise activations (inputs to linear layers) in addition to "
             "weights. Use --no-fp8-activation for weight-only FP8.",
    )
    fp8.add_argument(
        "--no-fp8-activation", dest="fp8_activation", action="store_false",
        help="Disable activation quantisation (weight-only FP8 mode).",
    )

    # -- Warmup + Early stopping --------------------------------------------
    sft = parser.add_argument_group(title="sft-training")
    sft.add_argument(
        "--warmup-steps", type=int, default=0,
        help="Number of synthetic-data warmup steps before real training. "
             "Warms up GPU kernels and FP8 scaling state, then resets FP8 stats.",
    )
    sft.add_argument(
        "--val-loss-target", type=float, default=None,
        help="Early stop when loss EMA drops below this value.",
    )

    return parser
