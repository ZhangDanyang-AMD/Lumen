###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Lightweight helpers for the shared Lumen training contract.

This module is intentionally argparse-only so it can be imported without
bringing in the heavier Megatron, Transformers, or quantization stacks.
"""

from .utils import safe_add_argument

__all__ = [
    "add_fsdp_contract_args",
    "add_fsdp_fp8_contract_args",
    "add_fsdp_runtime_contract_args",
    "add_shared_checkpoint_args",
    "add_shared_experiment_args",
]


def add_shared_checkpoint_args(parser_or_group):
    """Register checkpoint-management flags shared across training stacks."""
    safe_add_argument(parser_or_group, "--use-ckpt", action="store_true", default=False)
    safe_add_argument(parser_or_group, "--save-ckpt", action="store_true", default=False)
    safe_add_argument(parser_or_group, "--resume-from-hf", action="store_true", default=False)
    safe_add_argument(parser_or_group, "--continual-ckpt-path", type=str, default=None)
    safe_add_argument(parser_or_group, "--ckpt-start-step", type=int, default=0)
    safe_add_argument(parser_or_group, "--fp8-params", action="store_true", default=False)
    safe_add_argument(parser_or_group, "--initial-ckpt-path", type=str, default=None)
    return parser_or_group


def add_shared_experiment_args(
    parser_or_group,
    *,
    default_target_log_ppl: float = 3.3,
    default_step_time_atol: int = 18000,
):
    """Register experiment-management flags shared across training stacks."""
    safe_add_argument(parser_or_group, "--tag", type=str, default="")
    safe_add_argument(parser_or_group, "--target-log-ppl", type=float, default=default_target_log_ppl)
    safe_add_argument(parser_or_group, "--step-time-atol", type=int, default=default_step_time_atol)
    safe_add_argument(parser_or_group, "--eval-every", type=int, default=0)
    safe_add_argument(parser_or_group, "--start-eval-at", type=int, default=0)
    return parser_or_group


def add_fsdp_runtime_contract_args(parser_or_group):
    """Register FSDP runtime args shared across FSDP training entrypoints."""
    safe_add_argument(parser_or_group, "--lr-warmup-steps", type=int, default=0)
    safe_add_argument(parser_or_group, "--eval-interval", type=int, default=0)
    safe_add_argument(
        parser_or_group,
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing / activation recomputation.",
    )
    safe_add_argument(
        parser_or_group,
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing / activation recomputation.",
    )
    return parser_or_group


def add_fsdp_fp8_contract_args(parser_or_group):
    """Register shared linear quantization and legacy FSDP launcher aliases."""
    from lumen.cli import add_linear_quant_args

    return add_linear_quant_args(
        parser_or_group, fsdp_aliases=True, fp8_format_default="fp8_e4m3",
    )


def add_fsdp_contract_args(parser_or_group):
    """Register the full shared FSDP contract on a parser or group."""
    add_fsdp_runtime_contract_args(parser_or_group)
    add_fsdp_fp8_contract_args(parser_or_group)
    add_shared_checkpoint_args(parser_or_group)
    add_shared_experiment_args(parser_or_group)
    return parser_or_group
