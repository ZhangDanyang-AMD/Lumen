# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Download the Gov Report dataset from HuggingFace.

This downloads the scrolls/gov_report dataset used for summarization SFT,
matching the data used by the llama2_finetune example.

Usage:
    python download_dataset.py --output_dir /data/gov_report
    python download_dataset.py --output_dir /data/gov_report --verify
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
    parser = argparse.ArgumentParser(description="Download Gov Report dataset")
    parser.add_argument(
        "--output_dir", type=str, default="/data/gov_report",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="tau/scrolls",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--subset", type=str, default="gov_report",
        help="Dataset subset name",
    )
    parser.add_argument(
        "--verify", action="store_true", default=False,
        help="Compute and print SHA-256 hashes of output files for verification.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from datasets import load_dataset

    print(f"Downloading {args.dataset_name}/{args.subset} ...")
    dataset = load_dataset(args.dataset_name, args.subset)

    for split_name, split_data in dataset.items():
        output_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        split_data.to_json(output_path)
        print(f"  Saved {split_name}: {len(split_data)} samples -> {output_path}")
        if args.verify:
            digest = _sha256(output_path)
            print(f"    SHA-256: {digest}")

    print("Download complete.")


if __name__ == "__main__":
    main()
