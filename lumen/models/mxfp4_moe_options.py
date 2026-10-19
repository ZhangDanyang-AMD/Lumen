# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""The single MXFP4 routed-expert selector and P0 fixture eligibility checks."""

import os

from lumen.utils.mxfp4_moe_ledger import Mxfp4StateError


def add_mxfp4_moe_arguments(parser, *, pytest=False):
    # One selector definition for argparse callers and the actual pytest fixture CLI.
    group = (
        parser.getgroup("Lumen routed-expert MXFP4")
        if pytest
        else parser.add_argument_group("Lumen routed-expert MXFP4")
    )
    add = group.addoption if pytest else group.add_argument
    add("--lumen-moe-mxfp4", action="store_true")
    add(
        "--lumen-moe-mxfp4-backend",
        choices=("auto", "grouped", "sequential"),
        default="auto",
    )
    add("--lumen-moe-mxfp4-strict", action="store_true")
    add(
        "--lumen-moe-mxfp4-grouped-tuning-policy",
        choices=("require-certified-bucket", "allow-validated-heuristic"),
        default="require-certified-bucket",
    )
    return parser


def validate_p0_selector(args):
    if not args.lumen_moe_mxfp4:
        raise Mxfp4StateError("FEATURE_NOT_REQUESTED")
    if args.lumen_moe_mxfp4_backend != "sequential" or not args.lumen_moe_mxfp4_strict:
        raise Mxfp4StateError("P0_REQUIRES_EXPLICIT_STRICT_SEQUENTIAL")
    if args.lumen_moe_mxfp4_grouped_tuning_policy not in (
        "require-certified-bucket",
        "allow-validated-heuristic",
    ):
        raise Mxfp4StateError("INVALID_GROUPED_TUNING_POLICY")
    if (
        os.environ.get("LUMEN_DSV4_LINEAR_FP8")
        or os.environ.get("LUMEN_MXFP4_DGRAD_HADAMARD") == "1"
    ):
        raise Mxfp4StateError("INCOMPATIBLE_QUANTIZATION_FLAGS")


def validate_p0_options(args, config):
    validate_p0_selector(args)
    if (
        config.add_bias_linear
        or config.delay_wgrad_compute
        or config.expert_tensor_parallel_size != 1
    ):
        raise Mxfp4StateError("UNSUPPORTED_ROUTED_FC")
    return {
        "scope": "routed_experts",
        "requested_backend": "sequential",
        "strict": True,
        "grouped_tuning_policy": args.lumen_moe_mxfp4_grouped_tuning_policy,
        "grouped_tuning_policy_applies": False,
        "kernel_family": "TRITON",
        "precision": "MXFP4",
    }
