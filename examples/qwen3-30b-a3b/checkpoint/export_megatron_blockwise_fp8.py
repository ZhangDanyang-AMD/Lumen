"""Export a BF16 Megatron checkpoint as Lumen blockwise2d FP8 weights.

The output is an inference artifact, not a resumable Megatron training
checkpoint: optimizer and RNG state are intentionally omitted. Quantized
weights keep their original state-dict key and receive a sibling
``<module>.weight_scale`` tensor containing the dequantization factor consumed
by Lumen's blockscale GEMM.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Mapping

import torch


BLOCK_SIZE = 128
FORMAT_NAME = "lumen_blockwise2d_fp8_v1"
SCALE_SUFFIX = "_scale"


def should_quantize_weight(name: str, tensor: torch.Tensor) -> bool:
    """Return whether *tensor* is a blockwise2d GEMM weight.

    Embeddings, output heads, norms, and MoE routers remain in their checkpoint
    dtype. Norms are one-dimensional already; explicit name checks make the
    export contract clear and protect future two-dimensional implementations.
    """
    if not name.endswith(".weight") or tensor.ndim != 2 or not tensor.is_floating_point():
        return False
    lowered = name.lower()
    skipped = (
        "embedding",
        "output_layer",
        "lm_head",
        "layernorm",
        "layer_norm",
        ".norm.",
        ".router.",
    )
    return not any(token in lowered for token in skipped)


def scale_key_for(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"expected a .weight key, got {weight_key!r}")
    return f"{weight_key}{SCALE_SUFFIX}"


def quantize_model_state(
    model_state: Mapping[str, object],
    quantize_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[OrderedDict[str, object], list[str]]:
    """Quantize eligible weights while preserving all other model entries."""
    output: OrderedDict[str, object] = OrderedDict()
    quantized: list[str] = []
    for name, value in model_state.items():
        if not isinstance(value, torch.Tensor) or not should_quantize_weight(name, value):
            output[name] = value
            continue
        if value.shape[-2] % BLOCK_SIZE or value.shape[-1] % BLOCK_SIZE:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} is not divisible by "
                f"{BLOCK_SIZE}x{BLOCK_SIZE}"
            )
        fp8_weight, scale = quantize_fn(value)
        if fp8_weight.shape != value.shape:
            raise RuntimeError(
                f"quantizer changed {name} shape from {tuple(value.shape)} "
                f"to {tuple(fp8_weight.shape)}"
            )
        expected_scale_shape = (
            value.shape[-2] // BLOCK_SIZE,
            value.shape[-1] // BLOCK_SIZE,
        )
        if tuple(scale.shape) != expected_scale_shape:
            raise RuntimeError(
                f"quantizer returned scale shape {tuple(scale.shape)} for {name}; "
                f"expected {expected_scale_shape}"
            )
        output[name] = fp8_weight.cpu()
        output[scale_key_for(name)] = scale.float().cpu()
        quantized.append(name)
    return output, quantized


def _latest_iteration(checkpoint_dir: Path) -> int:
    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError(f"missing Megatron iteration marker: {marker}")
    return int(marker.read_text(encoding="utf-8").strip())


def _rank_checkpoints(checkpoint_dir: Path, iteration: int) -> list[Path]:
    iteration_dir = checkpoint_dir / f"iter_{iteration:07d}"
    paths = sorted(iteration_dir.glob("mp_rank_*/model_optim_rng.pt"))
    if not paths:
        raise FileNotFoundError(f"no rank checkpoints found under {iteration_dir}")
    return paths


def _production_quantizer(device: torch.device):
    if device.type != "cuda":
        raise ValueError(
            "the production blockwise2d quantizer requires --device cuda; "
            "run this utility in the Lumen training image on an AMD GPU"
        )
    from lumen.ops.quantize.linear import _quant_blockwise2d_weight
    from lumen.quantize.config import _get_float8_e4m3

    fp8_dtype = _get_float8_e4m3()

    def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight_device = weight.to(device=device, dtype=torch.bfloat16)
        fp8_weight, scale = _quant_blockwise2d_weight(
            weight_device, fp8_dtype, BLOCK_SIZE
        )
        return fp8_weight, scale

    return quantize, str(fp8_dtype).removeprefix("torch.")


def export_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    iteration: int | None = None,
    device: str = "cuda",
) -> dict:
    """Export all TP/PP/EP rank files from one Megatron iteration."""
    iteration = _latest_iteration(checkpoint_dir) if iteration is None else iteration
    inputs = _rank_checkpoints(checkpoint_dir, iteration)
    quantize_fn, fp8_dtype_name = _production_quantizer(torch.device(device))
    output_iteration = output_dir / f"iter_{iteration:07d}"
    output_iteration.mkdir(parents=True, exist_ok=True)

    rank_reports = []
    for input_path in inputs:
        checkpoint = torch.load(
            input_path, map_location="cpu", mmap=True, weights_only=False
        )
        if "model" not in checkpoint:
            raise KeyError(f"{input_path} has no 'model' state dict")
        model, quantized = quantize_model_state(checkpoint["model"], quantize_fn)
        artifact = {
            "format": FORMAT_NAME,
            "block_size": [BLOCK_SIZE, BLOCK_SIZE],
            "fp8_dtype": fp8_dtype_name,
            "scale_semantics": "dequantization_factor",
            "source_iteration": iteration,
            "checkpoint_version": checkpoint.get("checkpoint_version"),
            "model": model,
        }
        rank_dir = output_iteration / input_path.parent.name
        rank_dir.mkdir(parents=True, exist_ok=True)
        output_path = rank_dir / "model_fp8.pt"
        temporary_path = output_path.with_suffix(".pt.tmp")
        torch.save(artifact, temporary_path)
        os.replace(temporary_path, output_path)
        rank_reports.append(
            {
                "rank": input_path.parent.name,
                "file": str(output_path.relative_to(output_dir)),
                "quantized_weights": len(quantized),
            }
        )
        del artifact, model, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_dir / "latest_checkpointed_iteration.txt").write_text(
        str(iteration), encoding="utf-8"
    )
    manifest = {
        "format": FORMAT_NAME,
        "source": str(checkpoint_dir),
        "iteration": iteration,
        "block_size": [BLOCK_SIZE, BLOCK_SIZE],
        "fp8_dtype": fp8_dtype_name,
        "scale_semantics": "dequantization_factor",
        "resumable_training_checkpoint": False,
        "rank_files": rank_reports,
    }
    manifest_path = output_dir / "fp8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration to export (default: latest_checkpointed_iteration.txt).",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_checkpoint(
        args.load_dir,
        args.save_dir,
        iteration=args.iteration,
        device=args.device,
    )
    total = sum(rank["quantized_weights"] for rank in manifest["rank_files"])
    print(
        f"Exported {len(manifest['rank_files'])} rank files and {total} FP8 "
        f"weights to {args.save_dir}"
    )


if __name__ == "__main__":
    main()
