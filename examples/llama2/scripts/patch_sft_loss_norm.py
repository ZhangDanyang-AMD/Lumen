"""Patch Megatron-LM-AMD to enable --sft loss normalization.

NeMo MLPerf reference uses `sample_weight='constant'` + `validation_drop_last=False`
which normalizes val_loss as mean-of-per-microbatch-means (each microbatch weighted equally).

Megatron's default (non-sft) path uses global sum(loss)/sum(tokens) which gives more
weight to microbatches with more valid tokens.

The --sft flag in Megatron switches to per-microbatch averaging for both training
reporting and validation, matching NeMo's behavior.

This patch adds the --sft argument to Megatron's argument parser and ensures it
defaults to True.
"""

import sys
from pathlib import Path

SFT_ARG = "group.add_argument('--sft', action=\"store_true\", help='Megatron SFT training')"
PATCHED_SFT_ARG = (
    "group.add_argument('--sft', action=\"store_true\", default=True, "
    "help='Megatron SFT training (patched default=True for MLPerf loss norm)')"
)


def main():
    megatron_root = Path(sys.argv[1])

    args_file = megatron_root / "megatron" / "training" / "arguments.py"
    text = args_file.read_text()

    marker = "group.add_argument('--sft', action=\"store_true\""
    if marker not in text:
        old = "def _add_sft_args(parser):\n    group = parser.add_argument_group(title='sft')"
        new = (
            "def _add_sft_args(parser):\n"
            "    group = parser.add_argument_group(title='sft')\n"
            "    # --- patched: default=True to match NeMo sample_weight=constant ---"
        )
        if old in text:
            text = text.replace(old, new)
            text = text.replace(SFT_ARG, PATCHED_SFT_ARG)
            args_file.write_text(text)
            print("[patch_sft_loss_norm] Set --sft default=True in arguments.py")
        else:
            print("[patch_sft_loss_norm] WARN: could not find _add_sft_args pattern, trying direct injection")
            text = text.replace(SFT_ARG, PATCHED_SFT_ARG)
            args_file.write_text(text)
            print("[patch_sft_loss_norm] Set --sft default=True in arguments.py (direct)")
    else:
        if "default=True" not in text.split(marker)[0].split("\n")[-1] + marker:
            text = text.replace(SFT_ARG, PATCHED_SFT_ARG)
            args_file.write_text(text)
            print("[patch_sft_loss_norm] Set --sft default=True in arguments.py")
        else:
            print("[patch_sft_loss_norm] --sft default=True already set, skipping")


if __name__ == "__main__":
    main()
