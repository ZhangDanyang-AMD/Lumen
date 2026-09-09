#!/usr/bin/env python3
"""Find every convolution a VeOmni example can construct, and which ones a GEMM kernel helps.

The conv patch only reaches convolutions, so the first question about any VeOmni
example is whether it has any. Answering that by running each example is
expensive and, for the configs whose weights are not public, impossible. It is
also unnecessary: a convolution has to be constructed somewhere in the source,
and the arguments it is constructed with already decide whether an
implicit-GEMM kernel can beat torch.

This walks the source with ``ast`` and reports each ``nn.ConvNd`` construction
with its kernel, stride and groups, classified as:

    dense      a real overlapping convolution over many channels. The only
               class where this kernel is the right tool.
    matmul     stride == kernel, so receptive fields do not overlap. This is a
               reshape plus one GEMM however it is spelled, and a convolution
               kernel has nothing to win. Every VLM's ViT patch-embed is this.
    depthwise  groups == channels. Implicit GEMM turns a convolution into one
               large matrix multiply, and a depthwise convolution has no such
               matrix -- it is memory-bound, which is why it gets a dedicated
               kernel elsewhere (Qwen3.5's GatedDeltaNet routes its short
               convolution to causal_conv1d).

Two things are examined, and the distinction matters: VeOmni implements the
transformers itself, but the DiT examples get their VAE -- which is where all of
their convolution is -- from installed diffusers. Reporting only VeOmni's own
source would put "no convolutions" next to Qwen-Image and Wan, which is exactly
backwards.

The diffusers half is counted differently, from an instantiated module tree
rather than from source, because a static scan cannot see it: those VAEs define
``QwenImageCausalConv3d(nn.Conv3d)`` and ``WanCausalConv3d(nn.Conv3d)`` and
construct *those*, so an ``nn.ConvNd`` scan finds only the handful of plain
resamplers. Building the VAE on the meta device needs its config but not its
weights, so it also works for a model whose weights are not on this machine.

This says nothing about how much time each convolution takes; that needs a trace
against real shapes (trace_vae_convs.py, trace_video_vae_convs.py). It is the
cheap half of the question: which examples are candidates at all.

    python3 survey_veomni_convs.py --veomni /work/VeOmni \
        --vae "Qwen-Image=$QWEN_IMAGE_DIR" --vae "Wan2.1=$WAN_DIR"
"""

import argparse
import ast
import collections
import importlib.util
import os
import re

CONV = {"Conv1d": 1, "Conv2d": 2, "Conv3d": 3}



def literal(node):
    """Source-ish text for an argument, without evaluating anything."""
    if node is None:
        return None
    try:
        return repr(ast.literal_eval(node))
    except (ValueError, SyntaxError):
        return ast.unparse(node)


def _digits(text):
    return re.findall(r"\d+", text or "")


def classify(kernel, stride, groups):
    """Which of the three classes is this construction, from its arguments alone?"""
    if groups not in ("1", "None") and not groups.isdigit():
        # An expression, e.g. self.conv_dim. A constant fraction of the channels
        # would have been written as a number, so this is the depthwise case.
        return "depthwise"
    if groups.isdigit() and int(groups) > 1:
        return "depthwise"
    if kernel is not None and (kernel == stride or _digits(kernel) == _digits(stride) and _digits(kernel)):
        return "matmul"
    if _digits(kernel) == ["1"]:
        return "matmul"  # pointwise: a GEMM already
    return "dense"


def scan(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name not in CONV:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        pos = node.args
        # nn.ConvNd(in_channels, out_channels, kernel_size, stride, padding, ...)
        kernel = literal(kw.get("kernel_size") or (pos[2] if len(pos) > 2 else None))
        stride = literal(kw.get("stride") or (pos[3] if len(pos) > 3 else None)) or "1"
        groups = literal(kw.get("groups")) or "1"
        found.append(
            {
                "rank": CONV[name],
                "kernel": kernel,
                "stride": stride,
                "groups": groups,
                "klass": classify(kernel, stride, groups),
            }
        )
    return found


def scan_tree(root, depth):
    """{group: [hit]} for every .py under root, grouped by its first `depth` path parts."""
    per_group = collections.defaultdict(list)
    groups_seen = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            continue
        group = "/".join(rel.split(os.sep)[:depth])
        groups_seen.add(group)
        for fn in filenames:
            if fn.endswith(".py"):
                per_group[group].extend(scan(os.path.join(dirpath, fn)))
    return per_group, groups_seen


def census_vae(label, path):
    """Count the convolutions of a real VAE, built on meta from its config alone."""
    import json

    import diffusers
    import torch
    import torch.nn as nn

    cfg_path = os.path.join(path, "vae", "config.json")
    if not os.path.exists(cfg_path):
        print(f"\n{label}: no vae/config.json under {path} -- weights not on this machine")
        return
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    klass = getattr(diffusers, cfg.get("_class_name", ""), None)
    if klass is None:
        print(f"\n{label}: diffusers has no {cfg.get('_class_name')!r}")
        return
    with torch.device("meta"):
        vae = klass.from_config(cfg)

    rows = []
    for _name, m in vae.named_modules():
        if not isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            continue
        rank = 3 if isinstance(m, nn.Conv3d) else (2 if isinstance(m, nn.Conv2d) else 1)
        # A causal module has moved its real padding into _padding; report the
        # kernel and stride, which is what decides the class.
        rows.append(
            {
                "rank": rank,
                "kernel": str(tuple(m.kernel_size)),
                "stride": str(tuple(m.stride)),
                "groups": str(m.groups),
                "klass": classify(str(tuple(m.kernel_size)), str(tuple(m.stride)), str(m.groups)),
                "type": type(m).__name__,
            }
        )
    report(f"{label}  [{cfg['_class_name']}, {len(rows)} conv modules]", rows, with_type=True)


def report(title, rows, with_type=False):
    print(f"\n{title}")
    extra = f"{'module':>26}" if with_type else ""
    print(f"  {'rank':>4}{'n':>4}{'kernel':>26}{'stride':>14}{'groups':>8}{extra}  class")
    key = lambda h: (h["rank"], h["kernel"], h["stride"], h["groups"], h["klass"], h.get("type", ""))  # noqa: E731
    seen = collections.Counter(key(h) for h in rows)
    for (rank, kernel, stride, groups, klass, kind), n in sorted(seen.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        col = f"{kind[:26]:>26}" if with_type else ""
        print(f"  {rank:>4}{n:>4}{str(kernel)[:26]:>26}{str(stride)[:14]:>14}{str(groups)[:8]:>8}{col}  {klass}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veomni", default=os.environ.get("VEOMNI_DIR", "/work/VeOmni"))
    ap.add_argument(
        "--vae",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="a model directory with a vae/ subfolder; repeatable",
    )
    args = ap.parse_args()

    root = os.path.join(args.veomni, "veomni", "models")
    if not os.path.isdir(root):
        raise SystemExit(f"no veomni/models under {args.veomni}")

    per_arch, all_arch = scan_tree(root, depth=2)
    all_arch = {a for a in all_arch if a.count("/") == 1}

    print("=== VeOmni's own model implementations, with no convolution ===")
    empty = sorted(a for a in all_arch if not per_arch.get(a))
    for a in empty:
        print(f"  {a}")
    print(f"  {len(empty)} of {len(all_arch)} architecture packages")
    print("  For the transformers/* entries this settles it: configs/text/* build")
    print("  these and nothing else, so the patch is a no-op by construction rather")
    print("  than by measurement. For the diffusers/* entries it does not: VeOmni")
    print("  wires up the DiT and takes the VAE from installed diffusers, and the")
    print("  VAE is where all of their convolution lives. See the second table.")

    print("\n=== VeOmni's own model implementations, with convolutions ===")
    for arch in sorted(per_arch):
        if per_arch[arch]:
            report(arch, per_arch[arch])

    print("\n\n=== the diffusers VAEs the DiT configs load, built on meta ===")
    if not args.vae:
        print("  none given; pass --vae LABEL=DIR for each model directory")
    elif importlib.util.find_spec("diffusers") is None:
        print("  diffusers not importable here -- rerun inside the container")
    else:
        for spec_str in args.vae:
            label, _, path = spec_str.partition("=")
            census_vae(label or path, path)
        print("\n  The causal 3-D modules are the patch's target and the plain nn.Conv2d")
        print("  resamplers are its video-only extra; the 1x1 modules are left on torch.")

    print("\n=== summary by class, VeOmni source only ===")
    tally = collections.Counter(h["klass"] for hits in per_arch.values() for h in hits)
    for klass in ("dense", "matmul", "depthwise"):
        print(f"  {klass:10} {tally[klass]:>4} constructions")
    print("\n  Only 'dense' is a candidate for this kernel. Counts are constructions")
    print("  in source, not calls at runtime, and a model with generated per-device")
    print("  copies is counted once per copy.")


if __name__ == "__main__":
    main()
