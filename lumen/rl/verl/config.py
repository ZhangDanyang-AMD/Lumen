###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Configuration contract for the VERL + Lumen integration.

This module defines VerlLumenArgs (the stable interface between VERL config
YAML and Lumen model builders) and the backend matrix validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "VerlLumenArgs",
    "validate_backend_matrix",
]

_VALID_TRAIN_BACKENDS = ("fsdp", "fsdp2", "megatron")
_VALID_ROLLOUT_BACKENDS = ("hf",)


@dataclass(slots=True)
class VerlLumenArgs:
    """Stable Lumen contract for VERL integration.

    This is intentionally separate from TrlLumenArgs — VERL has its own
    config structure (Hydra/OmegaConf) and different model lifecycle.
    """

    model_name_or_path: str
    output_dir: str = "./outputs/verl-lumen"

    train_backend: str = "fsdp2"
    critic_backend: str = "fsdp2"
    rollout_backend: str = "hf"

    fsdp_version: int = 2
    gradient_checkpointing: bool = True

    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    linear_fp8: bool = False
    linear_fp8_format: str = "fp8_e4m3"
    linear_fp8_scaling: str = "delayed"
    linear_fp8_activation: bool = True
    linear_fp8_wgrad: bool = True
    linear_fp8_reduce_amax: bool = False

    lumen_norm: bool = False
    lumen_fp8_attn: str = "none"
    lumen_fp8_activation_store: bool = False
    lumen_fp8_param_gather: bool = False
    lumen_fp8_weight_cache: bool = False

    fp8_param_manager: bool = False
    use_8bit_adam: bool = False

    def __post_init__(self) -> None:
        if self.train_backend not in _VALID_TRAIN_BACKENDS:
            raise ValueError(
                f"Unsupported train_backend {self.train_backend!r}; "
                f"expected one of {_VALID_TRAIN_BACKENDS}"
            )
        if self.rollout_backend not in _VALID_ROLLOUT_BACKENDS:
            raise ValueError(
                f"Unsupported rollout_backend {self.rollout_backend!r}; "
                f"v1 only supports {_VALID_ROLLOUT_BACKENDS}"
            )
        validate_backend_matrix(self)


def validate_backend_matrix(args: VerlLumenArgs) -> None:
    """Enforce v1 backend restrictions."""
    if args.train_backend == "megatron" and args.critic_backend == "megatron":
        raise ValueError(
            "v1 does not support critic_backend='megatron'. "
            "Keep critic on 'fsdp' or 'fsdp2' for the Megatron milestone."
        )
    if args.rollout_backend != "hf":
        raise ValueError(
            f"v1 only supports rollout_backend='hf', got {args.rollout_backend!r}. "
            "vLLM/SGLang rollout is deferred to post-v1."
        )


def from_verl_config(cfg: Any) -> VerlLumenArgs:
    """Build VerlLumenArgs from a VERL OmegaConf config dict.

    Extracts Lumen-relevant fields from VERL's nested config structure.
    """
    model_cfg = cfg.get("actor_rollout_ref", {}).get("model", {})
    actor_cfg = cfg.get("actor_rollout_ref", {}).get("actor", {})
    lumen_cfg = cfg.get("lumen", {})

    model_path = model_cfg.get("path", "")
    strategy = actor_cfg.get("strategy", "fsdp2")

    kwargs: dict[str, Any] = {
        "model_name_or_path": model_path,
        "train_backend": strategy,
    }

    if "output_dir" in cfg:
        kwargs["output_dir"] = cfg["output_dir"]

    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg.get("rank", 0) > 0:
        kwargs["lora_rank"] = lora_cfg["rank"]
        kwargs["lora_alpha"] = lora_cfg.get("alpha", 32.0)

    for key in (
        "linear_fp8", "linear_fp8_format", "linear_fp8_scaling",
        "lumen_norm", "lumen_fp8_attn", "lumen_fp8_activation_store",
        "lumen_fp8_param_gather", "lumen_fp8_weight_cache",
        "fp8_param_manager", "use_8bit_adam",
    ):
        if key in lumen_cfg:
            kwargs[key] = lumen_cfg[key]

    return VerlLumenArgs(**kwargs)
