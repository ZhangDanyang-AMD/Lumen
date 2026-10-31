#!/usr/bin/env python3
"""Focused Triton blockscale tuning for Llama2-7B training shapes."""

import argparse
import copy
import json
from pathlib import Path

import torch


def _base_config():
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": None,
        "NUM_KSPLIT": 1,
    }


def _candidates():
    base = _base_config()
    candidates = [base]

    variants = [
        {"BLOCK_SIZE_M": 64},
        {"BLOCK_SIZE_M": 256},
        {"BLOCK_SIZE_N": 64},
        {"BLOCK_SIZE_N": 128},
        {"BLOCK_SIZE_N": 512},
        {"GROUP_SIZE_M": 4},
        {"GROUP_SIZE_M": 8},
        {"GROUP_SIZE_M": 16},
        {"num_warps": 8},
        {"num_stages": 1},
        {"waves_per_eu": 2},
        {"waves_per_eu": 4},
        {"cache_modifier": ".cg"},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "num_warps": 8},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "num_warps": 8},
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "num_warps": 8},
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 4},
    ]
    for variant in variants:
        candidates.append({**base, **variant})
    for block_m, block_n in ((128, 256), (256, 128)):
        for group_m in (1, 4, 8):
            for waves in (1, 2, 4):
                for cache_modifier in (None, ".cg"):
                    for stages in (1, 2):
                        candidates.append(
                            {
                                **base,
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "GROUP_SIZE_M": group_m,
                                "num_warps": 8,
                                "num_stages": stages,
                                "waves_per_eu": waves,
                                "cache_modifier": cache_modifier,
                            }
                        )
    unique = {}
    for candidate in candidates:
        unique[json.dumps(candidate, sort_keys=True)] = candidate
    return list(unique.values())


def _measure(call, warmup=2, repeats=5):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return sorted(samples)[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale,
    )

    dtype = torch.float8_e4m3fn
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16).to(dtype)
    weight = torch.randn(
        args.n, args.k, device="cuda", dtype=torch.bfloat16
    ).to(dtype)
    x_scale = torch.ones(
        args.m, args.k // 128, device="cuda", dtype=torch.float32
    )
    weight_scale = torch.ones(
        args.n // 128, args.k // 128, device="cuda", dtype=torch.float32
    )
    output = torch.empty(
        args.m, args.n, device="cuda", dtype=torch.bfloat16
    )

    results = []
    for config in _candidates():
        candidate = copy.deepcopy(config)
        try:
            latency_ms = _measure(
                lambda: gemm_a8w8_blockscale(
                    x,
                    weight,
                    x_scale,
                    weight_scale,
                    dtype=torch.bfloat16,
                    y=output,
                    config=copy.deepcopy(candidate),
                    backend="triton",
                )
            )
            results.append({"latency_ms": latency_ms, "config": candidate})
        except Exception as exc:
            results.append({"error": str(exc), "config": candidate})

    results.sort(key=lambda item: item.get("latency_ms", float("inf")))
    payload = {
        "shape": {"M": args.m, "N": args.n, "K": args.k},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["results"][:5], indent=2))


if __name__ == "__main__":
    main()
