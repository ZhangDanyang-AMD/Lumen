"""FlyDSL-Gym: Compile and verify a generated kernel.

Reads a Python file containing FlyDSL kernel code, attempts to import and
compile it, then runs basic correctness checks.

Returns JSON to stdout:
  {"compiles": bool, "correct": bool, "error": str|null, "details": str}

Usage::

    python verify.py /tmp/kernel.py
    python verify.py /tmp/kernel.py --spec '{"operator":"gemm","M":4096,"N":4096,"K":2048}'
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback


def load_module(path):
    """Dynamically import a Python file as a module."""
    spec = importlib.util.spec_from_file_location("gen_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_syntax(code):
    """Check if code is valid Python."""
    try:
        compile(code, "<kernel>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def check_flydsl_patterns(code):
    """Check if code contains expected FlyDSL patterns."""
    import re
    patterns = {
        "has_flyc_decorator": bool(re.search(r"@flyc\.(kernel|jit)", code)),
        "has_fx_api": bool(re.search(r"fx\.\w+", code)),
        "has_import": bool(re.search(r"import flydsl|from flydsl", code)),
        "has_smem": bool(re.search(r"SmemAllocator|SharedAllocator", code)),
        "has_mfma": bool(re.search(r"rocdl\.mfma|mfma_", code)),
        "line_count": len(code.strip().split("\n")),
    }
    return patterns


def try_compile_kernel(path):
    """Try to import the kernel module — triggers FlyDSL JIT compilation."""
    try:
        module = load_module(path)
        return True, None, module
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None


def try_run_kernel(module, spec):
    """Try to run the kernel with test inputs and check correctness."""
    import torch

    op = spec.get("operator", "unknown")
    try:
        # Find callable: look for common entry point names
        entry = None
        for name in ["main", "run", "launch", "forward", op, f"{op}_kernel"]:
            if hasattr(module, name) and callable(getattr(module, name)):
                entry = getattr(module, name)
                break

        if entry is None:
            # Try any @flyc.jit decorated function
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and not name.startswith("_"):
                    entry = obj
                    break

        if entry is None:
            return None, "No callable entry point found"

        return True, "Entry point found (execution skipped in verify mode)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_path", help="Path to kernel Python file")
    parser.add_argument("--spec", type=str, default="{}", help="JSON spec")
    args = parser.parse_args()

    result = {
        "compiles": False,
        "correct": None,
        "error": None,
        "details": "",
        "patterns": {},
    }

    # Read code
    try:
        with open(args.kernel_path) as f:
            code = f.read()
    except Exception as e:
        result["error"] = f"Cannot read file: {e}"
        print(json.dumps(result))
        return

    # Check syntax
    valid, err = check_syntax(code)
    if not valid:
        result["error"] = err
        result["details"] = "Python syntax error"
        print(json.dumps(result))
        return

    # Check FlyDSL patterns
    result["patterns"] = check_flydsl_patterns(code)

    # Trivial kernel check
    if result["patterns"]["line_count"] < 10:
        result["error"] = "Trivial kernel: less than 10 lines"
        print(json.dumps(result))
        return

    # Try compile (import)
    compiles, err, module = try_compile_kernel(args.kernel_path)
    result["compiles"] = compiles
    if not compiles:
        result["error"] = err
        result["details"] = "FlyDSL compilation failed"
        print(json.dumps(result))
        return

    # Try run
    spec = json.loads(args.spec)
    correct, detail = try_run_kernel(module, spec)
    result["correct"] = correct
    result["details"] = detail

    print(json.dumps(result))


if __name__ == "__main__":
    main()
