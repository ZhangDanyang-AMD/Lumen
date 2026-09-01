###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Apply registered SOURCE-phase patches to a Megatron-LM checkout."""

from __future__ import annotations

import argparse
import sys

from lumen.patches.registry import PatchPhase, apply_patches, list_patches
from lumen.patches.types import PatchResult


def _ensure_source_patches_registered() -> None:
    import lumen.patches.source  # noqa: F401


def apply_megatron_source_patches(
    megatron_root: str,
    *,
    tags: set[str] | None = None,
    default_only: bool = True,
    dry_run: bool = False,
) -> dict[str, PatchResult]:
    """Apply SOURCE patches to *megatron_root*."""
    _ensure_source_patches_registered()
    return apply_patches(
        PatchPhase.SOURCE,
        megatron_root=megatron_root,
        tags=tags,
        default_only=default_only,
        dry_run=dry_run,
    )


def print_patch_report(megatron_root: str, results: dict[str, PatchResult], *, dry_run: bool = False) -> None:
    title = "Would patch" if dry_run else "Patched"
    print(f"{title} ROCm Megatron at {megatron_root}:")
    for name in sorted(results):
        result = results[name]
        if result.reason == "dry-run":
            status = "would apply"
        elif result.skipped:
            status = f"skipped ({result.reason})"
        elif not result.applied or result.return_value is False:
            status = "skipped"
        else:
            status = "PATCHED"
        print(f"  {status}: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply Lumen SOURCE patches to a Megatron-LM checkout.",
    )
    parser.add_argument(
        "megatron_root",
        nargs="?",
        help="Path to Megatron-LM root",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered SOURCE patches and exit",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=None,
        help="Apply only patches with this tag (repeatable)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Apply non-default patches too",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which patches would be applied without modifying files",
    )
    args = parser.parse_args(argv)

    _ensure_source_patches_registered()

    if args.list:
        tag_set = set(args.tags) if args.tags else None
        for spec in list_patches(PatchPhase.SOURCE, tags=tag_set):
            gate = "enabled" if spec.enabled() else "disabled"
            print(
                f"{spec.name:40} [{gate}] default={spec.default} "
                f"tags={','.join(sorted(spec.tags)) or '-'}"
            )
            if spec.description:
                print(f"  {spec.description}")
        return 0

    if not args.megatron_root:
        parser.error("megatron_root is required unless --list is set")

    tag_set = set(args.tags) if args.tags else None
    # Explicit --tag includes opt-in (default=False) patches; bare apply keeps defaults.
    default_only = not args.all and tag_set is None

    results = apply_megatron_source_patches(
        args.megatron_root,
        tags=tag_set,
        default_only=default_only,
        dry_run=args.dry_run,
    )
    print_patch_report(args.megatron_root, results, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
