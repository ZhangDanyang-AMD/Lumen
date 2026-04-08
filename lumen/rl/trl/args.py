###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pure argument contract for the TRL + Lumen integration."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "TrlLumenArgs",
    "build_accelerate_config_path",
    "build_grpo_config_kwargs",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ACCELERATE_DIR = _REPO_ROOT / "examples" / "rl" / "trl" / "accelerate"


@dataclass(slots=True)
class TrlLumenArgs:
    """Stable TRL contract shared by launchers, tests, and runtime helpers."""

    model_name_or_path: str
    output_dir: str
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    dataset_split: str = "train"
    dataset_prompt_column: str = "prompt"
    dataset_completion_column: str = "completion"
    reward_mode: str = "function"
    tokenizer_name_or_path: str | None = None
    reward_model_name_or_path: str | None = None
    train_dataset: Any | None = None
    eval_dataset: Any | None = None
    backend: str = "fsdp"
    algorithm: str = "grpo"
    fsdp_version: int = 1
    sharding_strategy: str = "full_shard"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 800
    lr: float = 1e-6
    lr_warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 0
    seed: int = 1234
    seq_length: int = 4096
    max_prompt_length: int = 2048
    max_completion_length: int = 1024
    num_generations: int = 4
    beta: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    gradient_checkpointing: bool = True
    warmup_steps: int = 0
    report_to: str = "none"
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    linear_fp8: bool = False
    linear_fp8_format: str = "fp8_e4m3"
    linear_fp8_scaling: str = "delayed"
    linear_fp8_block_size: int = 128
    linear_fp8_amax_algo: str = "max"
    linear_fp8_reduce_amax: bool = False
    linear_fp8_amax_history: int = 16
    linear_fp8_margin: int = 0
    linear_fp8_activation: bool = True
    linear_fp8_wgrad: bool = True
    grad_quant_type: str | None = None
    first_last_layers_bf16: bool = False
    num_layers_at_start_in_bf16: int = 1
    num_layers_at_end_in_bf16: int = 1
    use_sdma: bool = False
    lumen_norm: bool = False
    lumen_fp8_attn: str = "none"
    lumen_fp8_quant_type: str = "blockwise"
    lumen_attn_backend: str = "auto"
    lumen_fp8_activation_store: bool = False
    lumen_fp8_param_gather: bool = False
    lumen_fused_mlp: bool = False
    lumen_cpu_offload: bool = False
    lumen_delay_wgrad: bool = False
    lumen_gradient_accumulation_fusion: bool = False
    lumen_fused_rope: bool = False
    lumen_hip_graphs: bool = False
    lumen_fp8_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.algorithm != "grpo":
            raise ValueError(f"Unsupported algorithm {self.algorithm!r}; v1 only supports 'grpo'.")
        if self.backend != "fsdp":
            raise ValueError(f"Unsupported backend {self.backend!r}; v1 only supports 'fsdp'.")
        if self.fsdp_version not in (1, 2):
            raise ValueError(f"Unsupported fsdp_version {self.fsdp_version!r}; expected 1 or 2.")
        if self.sharding_strategy not in ("full_shard", "shard_grad_op", "no_shard"):
            raise ValueError(
                f"Unsupported sharding_strategy {self.sharding_strategy!r}; "
                "expected full_shard, shard_grad_op, or no_shard."
            )
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if self.beta != 0.0:
            raise ValueError("beta must stay at 0.0 in the v1 TRL + Lumen runner.")
        if self.max_prompt_length < 1:
            raise ValueError("max_prompt_length must be >= 1.")
        if self.max_completion_length < 1:
            raise ValueError("max_completion_length must be >= 1.")
        if self.max_prompt_length + self.max_completion_length > self.seq_length:
            raise ValueError("max_prompt_length + max_completion_length must be <= seq_length.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_accelerate_config_path(args: TrlLumenArgs) -> str:
    """Return the accelerate config path for the selected FSDP generation."""

    if args.fsdp_version not in (1, 2):
        raise ValueError(f"Unsupported fsdp_version {args.fsdp_version!r}; expected 1 or 2.")
    return (_ACCELERATE_DIR / f"fsdp{args.fsdp_version}.yaml").as_posix()


def build_grpo_config_kwargs(args: TrlLumenArgs) -> dict[str, Any]:
    """Translate the stable Lumen contract into GRPOConfig keyword arguments."""

    save_steps = args.save_interval if args.save_interval > 0 else args.max_steps + 1
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "per_device_train_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "warmup_steps": args.lr_warmup_steps,
        "logging_steps": args.log_interval,
        "save_steps": save_steps,
        "bf16": True,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "beta": args.beta,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "seed": args.seed,
    }
    if args.save_interval <= 0:
        kwargs["save_strategy"] = "no"

    # Pin rollout to Transformers' default model.generate() path.
    # TRL also supports vLLM and paged-attention backends; disable them
    # explicitly so a future TRL default cannot silently change rollout mode.
    kwargs["use_vllm"] = False

    return kwargs
