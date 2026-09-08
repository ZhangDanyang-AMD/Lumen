#!/usr/bin/env python3
"""Add the aiter modules Lumen needs but upstream aiter does not ship.

Lumen imports 8 module paths at the top level of its own modules that only exist
on its aiter fork (ZhangDanyang-AMD/aiter @ lumen/triton_kernels). Without them
`import lumen` raises ModuleNotFoundError. Dockerfile.lumen solves this by
overlaying individual files onto an existing aiter install, and warns against
copying more than that: newer aiter Python expects newer CK/csrc source, which
breaks the runtime JIT build. This does the same overlay against the aiter that
is already installed in this container.

Safety: only files that do not already exist are written, so nothing aiter
currently ships is modified. Anything already present is reported and skipped --
use --force only if you have a reason, since overwriting an installed file is
what the Dockerfile comment warns about.

    python overlay_aiter.py           # dry run, lists what would be written
    python overlay_aiter.py --apply
    python overlay_aiter.py --revert

The files under aiter_overlay/ are vendored from Lumen's aiter fork
(third_party/aiter, ZhangDanyang-AMD/aiter @ lumen/triton_kernels). They are
copied into the example rather than taken from the submodule so that
reproducing this does not require cloning several hundred MB of aiter for nine
small files.
"""

import argparse
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("AITER_OVERLAY_SRC", os.path.join(_HERE, "aiter_overlay", "aiter"))
MANIFEST = os.environ.get(
    "AITER_OVERLAY_MANIFEST",
    os.path.join(os.environ.get("LOG_DIR", "/tmp"), "aiter_overlay_manifest.json"),
)


def aiter_root():
    import aiter

    return os.path.dirname(aiter.__file__)


def planned(root):
    """Yield (src, dst, rel) for every file in the overlay tree."""
    for dirpath, _, files in os.walk(SRC):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, SRC)
            yield src, os.path.join(root, rel), rel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the files")
    ap.add_argument("--force", action="store_true", help="also overwrite existing files")
    ap.add_argument("--revert", action="store_true", help="remove what --apply added")
    args = ap.parse_args()

    root = aiter_root()
    print(f"aiter install: {root}")
    print(f"overlay source: {SRC}\n")

    if args.revert:
        if not os.path.exists(MANIFEST):
            print(f"no manifest at {MANIFEST}; nothing recorded to revert")
            return 1
        with open(MANIFEST) as fh:
            added = json.load(fh)["added"]
        for rel in added:
            dst = os.path.join(root, rel)
            if os.path.exists(dst):
                os.remove(dst)
                print(f"  removed {rel}")
            else:
                print(f"  already gone {rel}")
        # Drop stale bytecode so the removal takes effect.
        for dirpath, dirnames, _ in os.walk(root):
            for d in list(dirnames):
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)
        os.remove(MANIFEST)
        print(f"\nreverted {len(added)} files, manifest removed")
        return 0

    to_add, existing = [], []
    for src, dst, rel in planned(root):
        (existing if os.path.exists(dst) else to_add).append((src, dst, rel))

    print(f"{len(to_add)} new, {len(existing)} already present\n")
    for _, _, rel in to_add:
        print(f"  NEW      {rel}")
    for _, _, rel in existing:
        print(f"  PRESENT  {rel}   (skipped; --force to overwrite)")

    targets = to_add + (existing if args.force else [])
    if not args.apply:
        print(f"\ndry run -- would write {len(targets)} files. Re-run with --apply.")
        return 0

    written = []
    for src, dst, rel in targets:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # A package dir needs __init__.py to be importable; aiter ships them for
        # existing dirs, but fusions/ may be new here.
        pkg = os.path.dirname(dst)
        while pkg != root and not os.path.exists(os.path.join(pkg, "__init__.py")):
            open(os.path.join(pkg, "__init__.py"), "a").close()
            written.append(os.path.relpath(os.path.join(pkg, "__init__.py"), root))
            print(f"  created  {written[-1]}  (package marker)")
            pkg = os.path.dirname(pkg)
        shutil.copy2(src, dst)
        written.append(rel)
        print(f"  wrote    {rel}")

    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump({"aiter_root": root, "added": written}, fh, indent=2)
    print(f"\nwrote {len(written)} files; manifest -> {MANIFEST}")
    print(f"revert with: python {os.path.abspath(__file__)} --revert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
