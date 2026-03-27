###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TRL integration helpers for Lumen RL training."""

import importlib

from lumen.rl.trl.args import (
    TrlLumenArgs,
    build_accelerate_config_path,
    build_grpo_config_kwargs,
)

__all__ = [
    "TrlLumenArgs",
    "build_accelerate_config_path",
    "build_actor_model",
    "build_grpo_config_kwargs",
    "run_grpo",
    "build_reference_model",
    "build_reward_model",
    "maybe_run_synthetic_warmup",
]


def __getattr__(name: str):
    if name in {"build_actor_model", "build_reference_model", "build_reward_model"}:
        module = importlib.import_module("lumen.rl.trl.modeling")
        return getattr(module, name)
    if name == "run_grpo":
        module = importlib.import_module("lumen.rl.trl.runner")
        return getattr(module, name)
    if name == "maybe_run_synthetic_warmup":
        module = importlib.import_module("lumen.rl.trl.warmup")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
