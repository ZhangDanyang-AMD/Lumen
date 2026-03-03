# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Download LLaMA2 model from HuggingFace.

Downloads the HuggingFace checkpoint that will later be converted
to Megatron format via tools/checkpoint/convert.py.

Usage:
    python download_model.py --model_name meta-llama/Llama-2-70b-hf --output_dir /data/model
    python download_model.py --output_dir /data/model --verify
"""

import argparse
import hashlib
import os


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Download LLaMA2 from HuggingFace")
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-2-70b-hf",
        help="HuggingFace model name (e.g. meta-llama/Llama-2-7b-hf, "
             "meta-llama/Llama-2-13b-hf, meta-llama/Llama-2-70b-hf)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/data/model",
        help="Directory to save the downloaded model",
    )
    parser.add_argument(
        "--verify", action="store_true", default=False,
        help="Compute and print SHA-256 hashes of downloaded weight files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"Downloading {args.model_name} to {args.output_dir} ...")
    snapshot_download(
        repo_id=args.model_name,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Download complete: {args.output_dir}")

    if args.verify:
        print("\nVerifying file hashes ...")
        for root, _dirs, files in os.walk(args.output_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                relpath = os.path.relpath(fpath, args.output_dir)
                digest = _sha256(fpath)
                print(f"  {relpath}: {digest}")
        print("Verification complete.")


if __name__ == "__main__":
    main()
