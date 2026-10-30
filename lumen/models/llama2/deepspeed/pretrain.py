###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LLaMA2 causal-language-model pretraining with DeepSpeed + Lumen."""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from lumen.config import LumenConfig
from lumen.models.deepspeed import (
    add_common_deepspeed_args,
    build_deepspeed_config,
    initialize_deepspeed,
    make_lumen_deepspeed_model_provider,
)
from lumen.models.fsdp import build_cosine_warmup_scheduler
from lumen.models.llama31.dataset import PretrainTextDataset, check_sample_budget

__all__ = ["DeepSpeedTrainer", "build_model", "get_args"]

logger = logging.getLogger(__name__)


def build_model(args) -> nn.Module:
    """Build a HuggingFace LLaMA2 model from a checkpoint or from scratch."""
    from transformers import LlamaConfig, LlamaForCausalLM

    if args.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        config = LlamaConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            vocab_size=args.vocab_size,
            max_position_embeddings=args.max_position_embeddings,
            rope_theta=args.rope_theta,
            rms_norm_eps=1e-5,
            attention_bias=False,
            hidden_act="silu",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_cache=False,
        )
        model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)

    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


class DeepSpeedTrainer:
    """LLaMA2 pretraining driver for validated ZeRO-1/2 and BF16 ZeRO-3."""

    def __init__(self, args, *, model_builder=build_model):
        self.args = args
        self._setup_distributed()

        self.deepspeed_config = build_deepspeed_config(
            args,
            world_size=self.world_size,
        )
        self.lumen_config = LumenConfig.from_args(args)
        provider = make_lumen_deepspeed_model_provider(
            model_builder,
            self.lumen_config,
            self.deepspeed_config,
            backend=args.linear_backend,
        )
        self.manager, model = provider(args)
        self._verify_patch_installation(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        scheduler = build_cosine_warmup_scheduler(optimizer, args)
        self.engine, self.optimizer, self.scheduler = initialize_deepspeed(
            model,
            args,
            optimizer=optimizer,
            scheduler=scheduler,
            manager=self.manager,
        )
        if args.deepspeed_fp8_weight_cache:
            if self.manager is None or self.manager.num_fp8_params <= 0:
                raise RuntimeError(
                    "DeepSpeed FP8 weight cache was requested but no parameters "
                    "were registered"
                )
            if self.rank == 0:
                logger.info(
                    "DeepSpeed FP8 per-update weight cache enabled for %d parameters",
                    self.manager.num_fp8_params,
                )

        self.global_step = 0
        self.last_train_loss = float("inf")
        self.train_loader = self._build_dataloader(
            args.train_data_path,
            args.train_samples,
            shuffle=True,
        )
        self.val_loader = (
            self._build_dataloader(args.val_data_path, args.val_samples, shuffle=False)
            if args.val_data_path
            else None
        )
        local_val_batches = (
            min(len(self.val_loader), args.eval_iters)
            if self.val_loader is not None
            else 0
        )
        common_val_batches = torch.tensor(
            local_val_batches,
            dtype=torch.int64,
            device=self.engine.device,
        )
        dist.all_reduce(common_val_batches, op=dist.ReduceOp.MIN)
        self.val_batches = int(common_val_batches.item())

        self.train_batches_in_epoch = 0
        self._resume_generator_state = None
        self._resume_batches_in_epoch = 0
        if args.resume_from_checkpoint:
            _, client_state = self.engine.load_checkpoint(args.resume_from_checkpoint)
            if self.manager is not None:
                self.manager.mark_fp8_params_stale()
            if client_state:
                self.global_step = int(client_state.get("global_step", 0))
                rank_states = client_state.get("rank_states", [])
                if self.rank < len(rank_states):
                    rank_state = rank_states[self.rank]
                    self._resume_generator_state = rank_state.get("generator_state")
                    self._resume_batches_in_epoch = int(
                        rank_state.get("batches_in_epoch", 0)
                    )
                    if self.manager is not None and rank_state.get("scaling_manager"):
                        self.manager.load_state_dict(rank_state["scaling_manager"])
        if self.rank == 0:
            parameter_count = sum(
                int(getattr(param, "ds_numel", param.numel()))
                for param in self.engine.module.parameters()
            )
            logger.info(
                "DeepSpeed %s ready | params=%d | world=%d | zero=%d | "
                "micro_batch=%d | grad_accum=%d | seq=%d",
                getattr(args, "model_label", "LLaMA2"),
                parameter_count,
                self.world_size,
                args.zero_stage,
                args.micro_batch_size,
                args.gradient_accumulation_steps,
                args.seq_length,
            )

    def _setup_distributed(self) -> None:
        try:
            import deepspeed
        except ImportError as exc:
            raise ImportError(
                "LLaMA2 DeepSpeed pretraining requires a ROCm-compatible DeepSpeed installation"
            ) from exc

        if not dist.is_initialized():
            deepspeed.init_distributed(dist_backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        torch.manual_seed(self.args.seed)

    def _build_dataloader(self, data_path, max_samples, *, shuffle: bool):
        from transformers import AutoTokenizer

        tokenizer_path = self.args.tokenizer_name_or_path or self.args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        dataset = PretrainTextDataset(
            data_path=data_path,
            seq_length=self.args.seq_length,
            tokenizer=tokenizer,
            is_hf_tokenizer=True,
            max_samples=max_samples,
            rank=self.rank,
            world_size=self.world_size,
            allow_repeat=shuffle and self.args.repeat_corpus,
        )
        if shuffle and max_samples:
            warning = check_sample_budget(
                dataset.n_samples_read,
                max_samples,
                self.args.repeat_corpus,
            )
            if warning and self.rank == 0:
                logger.warning(warning)
        generator = torch.Generator()
        generator.manual_seed(self.args.seed + self.rank)
        if shuffle:
            self.train_generator = generator
        return DataLoader(
            dataset,
            batch_size=self.args.micro_batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=shuffle,
        )

    def _new_train_iterator(self, *, resume: bool = False):
        if resume and self._resume_generator_state is not None:
            self.train_generator.set_state(self._resume_generator_state)
        self._train_epoch_generator_state = self.train_generator.get_state()
        data_iter = iter(self.train_loader)
        skip_batches = self._resume_batches_in_epoch if resume else 0
        for _ in range(skip_batches):
            try:
                next(data_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Checkpoint data-loader position exceeds the current corpus"
                ) from exc
        self.train_batches_in_epoch = skip_batches
        return data_iter

    def _next_batch(self, data_iter):
        try:
            batch = next(data_iter)
        except StopIteration as exc:
            if not self.args.repeat_corpus:
                raise RuntimeError(
                    "Training corpus exhausted; provide more data or pass "
                    "--repeat-corpus"
                ) from exc
            data_iter = self._new_train_iterator()
            batch = next(data_iter)
        self.train_batches_in_epoch += 1
        return batch, data_iter

    def _verify_patch_installation(self, model: nn.Module) -> None:
        if self.args.hf_attn_patch:
            import torch.nn.functional as functional

            if functional.scaled_dot_product_attention.__name__ != "_lumen_sdpa":
                raise RuntimeError("Lumen/AITER SDPA patch was requested but is not installed")
        if self.args.lumen_fused_rope:
            import transformers.models.llama.modeling_llama as modeling_llama

            if modeling_llama.apply_rotary_pos_emb.__name__ != "_lumen_rope":
                raise RuntimeError("Lumen/AITER RoPE patch was requested but is not installed")
        if self.args.lumen_fused_mlp:
            mlps = [
                module
                for module in model.modules()
                if type(module).__name__ == "LlamaMLP"
            ]
            if not mlps or any(
                not getattr(module, "_lumen_fused_mlp_patched", False)
                for module in mlps
            ):
                raise RuntimeError(
                    "Lumen fused MLP was requested but not every LlamaMLP was patched"
                )
        if self.args.lumen_norm:
            norms = [
                module
                for module in model.modules()
                if hasattr(module, "_lumen_norm_calls")
            ]
            if not norms:
                raise RuntimeError(
                    "Lumen norm was requested but no norm module was patched"
                )
        if self.args.lumen_linear:
            linear_modules = [
                module
                for module in model.modules()
                if isinstance(module, nn.Linear)
                and getattr(module, "_lumen_linear_patched", False)
            ]
            if not linear_modules:
                raise RuntimeError(
                    "Lumen BF16 Linear was requested but no nn.Linear was patched"
                )
        if self.args.linear_fp8:
            quantized_linears = [
                module
                for module in model.modules()
                if isinstance(module, nn.Linear)
                and getattr(module, "_quant_enabled", False)
            ]
            if not quantized_linears:
                raise RuntimeError(
                    "Lumen FP8 Linear was requested but no nn.Linear was patched"
                )
            if self.args.linear_backend != "auto" and any(
                getattr(module, "_quant_backend", None) != self.args.linear_backend
                for module in quantized_linears
            ):
                raise RuntimeError(
                    "Lumen FP8 Linear backend does not match the requested backend"
                )

        model._lumen_deepspeed_patch_contract = {
            "hf_attn_patch": bool(self.args.hf_attn_patch),
            "fused_rope": bool(self.args.lumen_fused_rope),
            "fused_mlp": bool(self.args.lumen_fused_mlp),
            "lumen_norm": bool(self.args.lumen_norm),
            "lumen_linear": bool(self.args.lumen_linear),
            "linear_fp8": bool(self.args.linear_fp8),
            "gradient_checkpointing": bool(self.args.gradient_checkpointing),
        }

    def _verify_runtime_patch_calls(self) -> None:
        sdpa_calls = 0
        rope_calls = 0
        mlp_calls = 0
        norm_calls = 0
        linear_calls = 0
        fp8_linear_calls = 0
        shared_quant_hits = 0
        if self.args.hf_attn_patch:
            from lumen.ops.attention.hf_patch import get_sdpa_patch_calls

            sdpa_calls = get_sdpa_patch_calls()
            if sdpa_calls <= 0:
                raise RuntimeError(
                    "Lumen/AITER SDPA was installed but no training call reached it"
                )
        if self.args.lumen_fused_rope:
            import transformers.models.llama.modeling_llama as modeling_llama

            rope_calls = int(getattr(modeling_llama, "_lumen_rope_calls", 0))
            if rope_calls <= 0:
                raise RuntimeError(
                    "Lumen/AITER RoPE was installed but no training call reached it"
                )
        if self.args.lumen_fused_mlp:
            mlp_calls = sum(
                int(getattr(module, "_lumen_fused_mlp_calls", 0))
                for module in self.engine.module.modules()
            )
            if mlp_calls <= 0:
                raise RuntimeError(
                    "Lumen fused MLP was installed but no training call reached it"
                )
        if self.args.lumen_norm:
            norm_calls = sum(
                int(getattr(module, "_lumen_norm_calls", 0))
                for module in self.engine.module.modules()
            )
            if norm_calls <= 0:
                raise RuntimeError(
                    "Lumen norm was installed but no training call reached it"
                )
        if self.args.lumen_linear:
            linear_calls = sum(
                int(getattr(module, "_lumen_linear_calls", 0))
                for module in self.engine.module.modules()
            )
            if linear_calls <= 0:
                raise RuntimeError(
                    "Lumen BF16 Linear was installed but no training call reached it"
                )
        if self.args.linear_fp8:
            fp8_linear_calls = sum(
                int(getattr(module, "_lumen_quant_linear_calls", 0))
                for module in self.engine.module.modules()
            )
            if fp8_linear_calls <= 0:
                raise RuntimeError(
                    "Lumen FP8 Linear was installed but no training call reached it"
                )
            if os.environ.get("LUMEN_SHARED_INPUT_QUANT", "0") == "1":
                shared_quant_modules = [
                    module
                    for module in self.engine.module.modules()
                    if hasattr(module, "_lumen_shared_quant")
                ]
                shared_quant_hits = sum(
                    int(getattr(module, "_lumen_shared_quant_hits", 0))
                    for module in shared_quant_modules
                )
                if shared_quant_modules and shared_quant_hits <= 0:
                    raise RuntimeError(
                        "Shared FP8 input quantization was enabled but never reused"
                    )
        if self.rank == 0:
            logger.info(
                "Verified Lumen runtime patches | sdpa_calls=%d | rope_calls=%d | "
                "mlp_calls=%d | norm_calls=%d | linear_calls=%d | "
                "fp8_linear_calls=%d | shared_quant_hits=%d",
                sdpa_calls,
                rope_calls,
                mlp_calls,
                norm_calls,
                linear_calls,
                fp8_linear_calls,
                shared_quant_hits,
            )

    def train(self) -> None:
        args = self.args
        self.engine.train()
        data_iter = self._new_train_iterator(resume=True)
        measured_step_times = []
        torch.cuda.reset_peak_memory_stats(self.engine.device)

        while self.global_step < args.max_steps:
            torch.cuda.synchronize(self.engine.device)
            step_start = time.perf_counter()
            profiler = None
            if self.rank == 0 and args.profile_step == self.global_step + 1:
                profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                )
                profiler.__enter__()
            accumulated_loss = 0.0
            for micro_step in range(args.gradient_accumulation_steps):
                batch, data_iter = self._next_batch(data_iter)
                input_ids = batch["input_ids"].to(self.engine.device, non_blocking=True)
                # HuggingFace LlamaForCausalLM performs the one-token shift
                # internally. The dataset's pre-shifted labels are for
                # Megatron-style consumers and would otherwise be shifted twice.
                labels = input_ids
                outputs = self.engine(input_ids=input_ids, labels=labels)
                if not torch.isfinite(outputs.loss).item():
                    raise RuntimeError(
                        f"Non-finite forward loss at global step "
                        f"{self.global_step + 1}, micro step {micro_step + 1}: "
                        f"{outputs.loss.item()}"
                    )
                accumulated_loss += outputs.loss.detach().float().item()
                self.engine.backward(outputs.loss)
                self.engine.step()

            torch.cuda.synchronize(self.engine.device)
            if profiler is not None:
                profiler.__exit__(None, None, None)
                profile_output = args.profile_output or os.path.join(
                    args.save_dir,
                    f"profile_step_{self.global_step + 1}.txt",
                )
                os.makedirs(os.path.dirname(profile_output) or ".", exist_ok=True)
                with open(profile_output, "w", encoding="utf-8") as output:
                    output.write(
                        profiler.key_averages().table(
                            sort_by="self_cuda_time_total",
                            row_limit=100,
                        )
                    )
                    output.write("\n\n")
                    output.write(
                        profiler.key_averages(group_by_input_shape=True).table(
                            sort_by="self_cuda_time_total",
                            row_limit=200,
                        )
                    )
                logger.info("Profiler table written to %s", profile_output)
            step_time_ms = (time.perf_counter() - step_start) * 1000.0
            self.global_step += 1
            if self.global_step > args.performance_warmup_steps:
                measured_step_times.append(step_time_ms)
            loss_sum = torch.tensor(
                accumulated_loss / args.gradient_accumulation_steps,
                dtype=torch.float64,
                device=self.engine.device,
            )
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            mean_loss = (loss_sum / self.world_size).item()
            if not torch.isfinite(loss_sum).item():
                raise RuntimeError(
                    f"Non-finite training loss at global step {self.global_step}: "
                    f"{mean_loss}"
                )
            self.last_train_loss = mean_loss
            if self.global_step % args.log_interval == 0 and self.rank == 0:
                lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else args.lr
                tokens_per_second = (
                    args.micro_batch_size
                    * args.gradient_accumulation_steps
                    * self.world_size
                    * args.seq_length
                    / (step_time_ms / 1000.0)
                )
                peak_memory_gib = torch.cuda.max_memory_allocated(self.engine.device) / 2**30
                logger.info(
                    "step %d/%d | loss %.6f | lr %.2e | step_time_ms %.1f | "
                    "tokens_per_second %.1f | peak_memory_gib %.2f",
                    self.global_step,
                    args.max_steps,
                    mean_loss,
                    lr,
                    step_time_ms,
                    tokens_per_second,
                    peak_memory_gib,
                )

            if (
                self.val_batches > 0
                and args.eval_interval > 0
                and self.global_step % args.eval_interval == 0
            ):
                val_loss = self.validate()
                if self.rank == 0:
                    logger.info("step %d | val_loss %.4f", self.global_step, val_loss)

            if args.save_interval > 0 and self.global_step % args.save_interval == 0:
                self.save_checkpoint()

        self._verify_runtime_patch_calls()
        if measured_step_times:
            mean_step_time_ms = statistics.fmean(measured_step_times)
            median_step_time_ms = statistics.median(measured_step_times)
            if self.rank == 0:
                logger.info(
                    "Steady-state performance | mean_step_time_ms %.1f | "
                    "median_step_time_ms %.1f | measured_steps %d",
                    mean_step_time_ms,
                    median_step_time_ms,
                    len(measured_step_times),
                )
            if args.step_time_atol > 0 and mean_step_time_ms > args.step_time_atol:
                raise RuntimeError(
                    f"Steady-state mean step time {mean_step_time_ms:.1f} ms exceeds "
                    f"the aligned limit {args.step_time_atol:.1f} ms"
                )
        if args.target_loss > 0 and self.last_train_loss > args.target_loss:
            raise RuntimeError(
                f"Training did not reach target loss: final={self.last_train_loss:.6f}, "
                f"target={args.target_loss:.6f}"
            )
        if self.rank == 0:
            logger.info(
                "DeepSpeed pretraining complete at step %d | final_loss %.6f",
                self.global_step,
                self.last_train_loss,
            )

    def validate(self) -> float:
        self.engine.eval()
        total = torch.zeros(2, dtype=torch.float64, device=self.engine.device)
        data_iter = iter(self.val_loader)
        with torch.no_grad():
            for _ in range(self.val_batches):
                batch = next(data_iter)
                input_ids = batch["input_ids"].to(self.engine.device, non_blocking=True)
                loss = self.engine(input_ids=input_ids, labels=input_ids).loss
                total[0] += loss.detach().double()
                total[1] += 1
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        self.engine.train()
        return (total[0] / total[1].clamp_min(1)).item()

    def save_checkpoint(self) -> None:
        os.makedirs(self.args.save_dir, exist_ok=True)
        local_state = {
            "generator_state": self._train_epoch_generator_state,
            "batches_in_epoch": self.train_batches_in_epoch,
            "scaling_manager": (
                self.manager.state_dict() if self.manager is not None else None
            ),
        }
        rank_states = [None] * self.world_size
        dist.all_gather_object(rank_states, local_state)
        self.engine.save_checkpoint(
            self.args.save_dir,
            tag=f"global_step{self.global_step}",
            client_state={
                "global_step": self.global_step,
                "rank_states": rank_states,
            },
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLaMA2 pretraining with DeepSpeed + Lumen",
    )
    model = parser.add_argument_group("model")
    model.add_argument("--model-name-or-path", type=str, default=None)
    model.add_argument("--tokenizer-name-or-path", type=str, required=True)
    model.add_argument("--hidden-size", type=int, default=4096)
    model.add_argument("--intermediate-size", type=int, default=11008)
    model.add_argument("--num-layers", type=int, default=32)
    model.add_argument("--num-attention-heads", type=int, default=32)
    model.add_argument("--num-key-value-heads", type=int, default=32)
    model.add_argument("--vocab-size", type=int, default=32000)
    model.add_argument("--max-position-embeddings", type=int, default=4096)
    model.add_argument("--rope-theta", type=float, default=10000.0)
    model.add_argument("--seq-length", type=int, default=4096)

    data = parser.add_argument_group("data")
    data.add_argument("--train-data-path", type=str, required=True)
    data.add_argument("--val-data-path", type=str, default=None)
    data.add_argument("--train-samples", type=int, default=None)
    data.add_argument("--val-samples", type=int, default=None)
    data.add_argument("--repeat-corpus", action="store_true", default=False)
    data.add_argument("--eval-iters", type=int, default=10)
    data.add_argument(
        "--target-loss",
        type=float,
        default=0.0,
        help="Fail at the end when final training loss is above this value; 0 disables.",
    )

    add_common_deepspeed_args(parser)
    parser.set_defaults(
        linear_fp8_scaling="dynamic",
        linear_fp8_format="fp8_e4m3",
        gradient_checkpointing=False,
        hf_attn_patch=True,
        lumen_fused_rope=True,
    )
    return parser.parse_args()
