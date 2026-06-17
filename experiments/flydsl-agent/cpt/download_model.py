"""Download Qwen2.5-Coder-32B to /dev/shm for fast loading.

/dev/shm is tmpfs (RAM-backed), so model loading during training startup
is significantly faster than loading from disk.

Usage::

    python download_model.py
    python download_model.py --model Qwen/Qwen2.5-Coder-32B --output /dev/shm/qwen2.5-coder-32b
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Download Qwen2.5-Coder-32B")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/dev/shm/qwen2.5-coder-32b",
        help="Local download directory.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (if needed for gated models).",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print(f"Downloading {args.model} to {args.output} ...")

    snapshot_download(
        repo_id=args.model,
        local_dir=args.output,
        local_dir_use_symlinks=False,
        token=args.token,
    )

    size_gb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(args.output)
        for f in fns
    ) / (1024 ** 3)

    print(f"Done. Model saved to {args.output} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
