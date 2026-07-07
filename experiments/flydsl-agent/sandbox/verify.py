"""FlyDSL-Gym: Compile, run, and verify a generated kernel.

Three verification levels:
  1. Compilation: import triggers FlyDSL JIT → catches syntax/type/import errors
  2. Runtime: construct inputs, find entry point, call kernel → catches crashes/OOM
  3. Correctness: compare output against PyTorch reference → catches wrong results

Returns JSON to stdout:
  {"compiles": bool, "runs": bool, "correct": bool|null,
   "error": str|null, "details": str, "patterns": dict}

Usage::
    python verify.py /tmp/kernel.py --spec '{"operator":"gemm","M":128,"N":128,"K":128}'
"""

import argparse
import importlib.util
import json
import os
import re
import signal
import sys
import traceback


# ── Timeout handler ──────────────────────────────────────────────────────────

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Kernel execution timed out")


# ── Module loading ───────────────────────────────────────────────────────────

def load_module(path):
    spec = importlib.util.spec_from_file_location("gen_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_syntax(code):
    try:
        compile(code, "<kernel>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def check_flydsl_patterns(code):
    return {
        "has_flyc_decorator": bool(re.search(r"@flyc\.(kernel|jit)", code)),
        "has_fx_api": bool(re.search(r"fx\.\w+", code)),
        "has_import": bool(re.search(r"import flydsl|from flydsl", code)),
        "has_smem": bool(re.search(r"SmemAllocator|SharedAllocator", code)),
        "has_mfma": bool(re.search(r"rocdl\.mfma|mfma_", code)),
        "line_count": len(code.strip().split("\n")),
    }


def try_compile_kernel(path):
    try:
        module = load_module(path)
        return True, None, module
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None


# ── Entry point detection ────────────────────────────────────────────────────

def find_entry_points(module, op_name):
    """Find callable entry points in module.

    Returns list of (func, name, kind) where kind is 'jit', 'launcher', or 'plain'.
    Priority: @flyc.jit > named launcher > any callable.
    """
    from flydsl.compiler.jit_function import JitFunction
    from flydsl.compiler.kernel_function import KernelFunction

    entries = []

    # Priority 1: @flyc.jit decorated functions (host launchers — these can run kernels)
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if isinstance(obj, JitFunction):
            entries.append((obj, name, "jit"))

    # Priority 2: known launcher function names (plain Python, not @flyc.kernel)
    launcher_patterns = [
        f"launch_{op_name}", f"run_{op_name}", f"{op_name}_launcher",
        "launch", "run", "forward", "main",
        f"launch_{op_name}_kernel", f"run_{op_name}_kernel",
    ]
    for pat in launcher_patterns:
        if hasattr(module, pat):
            obj = getattr(module, pat)
            if callable(obj) and not isinstance(obj, (KernelFunction, JitFunction, type)):
                entries.append((obj, pat, "launcher"))

    # Priority 3: any non-kernel, non-jit callable
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if callable(obj) and not isinstance(obj, (KernelFunction, JitFunction, type)):
            if not any(e[1] == name for e in entries):
                entries.append((obj, name, "plain"))

    return entries


# ── Operator-specific input constructors and references ──────────────────────

def _default_dtype(spec):
    dtype_str = spec.get("dtype", "bf16")
    mapping = {
        "bf16": "torch.bfloat16", "fp16": "torch.float16", "fp32": "torch.float32",
        "fp8": "torch.float8_e4m3fnuz", "bfloat16": "torch.bfloat16",
        "float16": "torch.float16", "float32": "torch.float32",
    }
    import torch
    return eval(mapping.get(dtype_str, "torch.bfloat16"))


def construct_inputs(op, spec):
    """Construct input tensors on GPU for the given operator."""
    import torch
    torch.manual_seed(42)
    device = "cuda"

    # Use small sizes to avoid OOM
    if op == "gemm":
        M = min(spec.get("M", 128), 256)
        N = min(spec.get("N", 128), 256)
        K = min(spec.get("K", 128), 256)
        dtype = _default_dtype(spec)
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        C = torch.zeros(M, N, dtype=dtype, device=device)
        return {"A": A, "B": B, "C": C, "M": M, "N": N, "K": K}

    elif op == "softmax":
        B = spec.get("batch", 1)
        S = min(spec.get("seq_len", 128), 256)
        H = min(spec.get("num_heads", 8), 8)
        D = spec.get("head_dim", 64)
        x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        return {"x": x, "B": B, "S": S, "H": H, "D": D}

    elif op in ("rmsnorm", "layernorm"):
        B = min(spec.get("batch", 4), 8)
        D = min(spec.get("hidden_dim", 1024), 2048)
        x = torch.randn(B, D, dtype=torch.bfloat16, device=device)
        w = torch.ones(D, dtype=torch.bfloat16, device=device)
        return {"x": x, "weight": w, "B": B, "D": D, "eps": 1e-5}

    elif op == "rope":
        B = min(spec.get("batch", 4), 8)
        S = 128
        H = 32
        D = 64
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        cos = torch.randn(S, D, dtype=torch.bfloat16, device=device)
        sin = torch.randn(S, D, dtype=torch.bfloat16, device=device)
        return {"q": q, "k": k, "cos": cos, "sin": sin}

    elif op == "topk":
        N = min(spec.get("tokens", 128), 256)
        D = min(spec.get("hidden_size", 1024), 2048)
        K = min(spec.get("k", 8), 16)
        x = torch.randn(N, D, dtype=torch.bfloat16, device=device)
        return {"x": x, "k": K, "N": N, "D": D}

    elif op == "quant":
        rows = min(spec.get("rows", 128), 256)
        cols = min(spec.get("cols", 128), 256)
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
        return {"x": x, "rows": rows, "cols": cols}

    elif op == "flash_attn":
        B = spec.get("batch", 1)
        H = min(spec.get("heads", 8), 8)
        S = min(spec.get("seq_len", 128), 256)
        D = min(spec.get("head_dim", 64), 128)
        q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        return {"q": q, "k": k, "v": v, "B": B, "H": H, "S": S, "D": D}

    elif op == "moe":
        M = min(spec.get("M", 128), 256)
        N = min(spec.get("N", 128), 256)
        K = min(spec.get("K", 128), 256)
        dtype = _default_dtype(spec)
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(K, N, dtype=dtype, device=device)
        return {"x": x, "w": w, "M": M, "N": N, "K": K}

    elif op == "mla":
        B = spec.get("batch", 1)
        S = min(spec.get("seq_len", 128), 256)
        H = min(spec.get("num_heads", 16), 16)
        D = min(spec.get("v_head_dim", 128), 128)
        q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        return {"q": q, "k": k, "v": v}

    elif op == "paged_attn":
        B, H, S, D = 1, 8, 128, 64
        q = torch.randn(B, H, 1, D, dtype=torch.bfloat16, device=device)
        k_cache = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        v_cache = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        return {"q": q, "k_cache": k_cache, "v_cache": v_cache}

    else:
        return None


def compute_reference(op, inputs):
    """Compute reference output using PyTorch."""
    import torch
    import torch.nn.functional as F

    if op == "gemm":
        A = inputs["A"].float()
        B = inputs["B"].float()
        return torch.mm(A, B)

    elif op == "softmax":
        return torch.softmax(inputs["x"].float(), dim=-1)

    elif op == "rmsnorm":
        x = inputs["x"].float()
        w = inputs["weight"].float()
        eps = inputs["eps"]
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return (x / rms) * w

    elif op == "layernorm":
        x = inputs["x"].float()
        D = inputs["D"]
        return F.layer_norm(x, [D])

    elif op == "topk":
        x = inputs["x"].float()
        k = inputs["k"]
        vals, indices = torch.topk(x, k, dim=-1)
        return vals

    elif op == "flash_attn":
        q = inputs["q"].float()
        k = inputs["k"].float()
        v = inputs["v"].float()
        D = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    elif op == "quant":
        x = inputs["x"].float()
        amax = x.abs().max()
        scale = amax / 448.0  # fp8_e4m3 max
        return (x / scale.clamp(min=1e-12)).clamp(-448.0, 448.0)

    elif op == "moe":
        x = inputs["x"].float()
        w = inputs["w"].float()
        return torch.mm(x, w)

    elif op == "rope":
        q = inputs["q"].float()
        cos = inputs["cos"].float()
        sin = inputs["sin"].float()
        # Simple RoPE: rotate pairs
        q1, q2 = q[..., ::2], q[..., 1::2]
        c = cos[:q.shape[1], :q.shape[-1]//2]
        s = sin[:q.shape[1], :q.shape[-1]//2]
        return torch.stack([q1 * c - q2 * s, q1 * s + q2 * c], dim=-1).flatten(-2)

    elif op in ("mla", "paged_attn"):
        q = inputs["q"].float()
        k = inputs.get("k", inputs.get("k_cache", torch.zeros_like(q))).float()
        v = inputs.get("v", inputs.get("v_cache", torch.zeros_like(q))).float()
        D = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    else:
        return None


def get_tolerance(op):
    tolerances = {
        "gemm": (1e-2, 1e-2),
        "softmax": (1e-3, 1e-3),
        "rmsnorm": (1e-3, 1e-3),
        "layernorm": (1e-3, 1e-3),
        "rope": (1e-3, 1e-3),
        "topk": (1e-3, 1e-3),
        "quant": (1e-1, 1e-1),
        "flash_attn": (1e-2, 1e-2),
        "moe": (1e-2, 1e-2),
        "mla": (1e-2, 1e-2),
        "paged_attn": (1e-2, 1e-2),
    }
    return tolerances.get(op, (1e-2, 1e-2))


# ── Runtime execution ────────────────────────────────────────────────────────

def _try_call(func, args_list):
    """Try calling func with multiple argument strategies. Returns (output, detail) or raises."""
    errors = []
    for args, kwargs, desc in args_list:
        try:
            output = func(*args, **kwargs)
            return output, desc
        except TypeError as e:
            errors.append(f"{desc}: {e}")
            continue
    raise TypeError("; ".join(errors))


def try_run_kernel(module, spec):
    """Try to run the kernel with test inputs and check correctness.

    Returns (runs: bool, correct: bool|None, detail: str)
    """
    import torch

    op = spec.get("operator", "unknown")

    entries = find_entry_points(module, op)
    if not entries:
        return False, None, "No callable entry point found"

    inputs = construct_inputs(op, spec)
    if inputs is None:
        return False, None, f"No input constructor for operator '{op}'"

    tensor_args = [v for v in inputs.values() if isinstance(v, torch.Tensor)]
    int_args = [v for v in inputs.values() if isinstance(v, int)]
    all_args = tensor_args + int_args
    tensor_kwargs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    all_kwargs = {k: v for k, v in inputs.items() if isinstance(v, (torch.Tensor, int, float))}

    # Build calling strategies (tried in order)
    call_strategies = [
        (all_args, {}, "positional(*tensors, *ints)"),
        (tensor_args, {}, "positional(*tensors)"),
        ([], all_kwargs, "kwargs(**all)"),
        ([], tensor_kwargs, "kwargs(**tensors)"),
        (tensor_args[:3], {}, "positional(*tensors[:3])"),
        (tensor_args[:2] + int_args[:1], {}, "positional(*tensors[:2], int)"),
    ]

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60s total timeout for all attempts

    # Try each entry point
    last_error = ""
    for func, name, kind in entries:
        try:
            output, call_desc = _try_call(func, call_strategies)
            torch.cuda.synchronize()
            signal.alarm(0)

            detail_prefix = f"{name} [{kind}] via {call_desc}"

            # Check correctness
            if output is None:
                # In-place kernel: check if any input tensor was modified
                ref = compute_reference(op, inputs)
                if ref is not None:
                    atol, rtol = get_tolerance(op)
                    for tname, t in tensor_kwargs.items():
                        try:
                            t_f = t.float().cpu()
                            r_f = ref.float().cpu()
                            # Try matching shapes via reshape/view
                            if t_f.numel() == r_f.numel():
                                if torch.allclose(t_f.reshape(-1), r_f.reshape(-1), atol=atol, rtol=rtol):
                                    return True, True, f"{detail_prefix} — in-place CORRECT on '{tname}'"
                        except Exception:
                            continue
                return True, None, f"{detail_prefix} — returned None (in-place?)"

            if isinstance(output, tuple):
                output = output[0] if isinstance(output[0], torch.Tensor) else output
            if not isinstance(output, torch.Tensor):
                return True, None, f"{detail_prefix} — output is {type(output).__name__}"

            ref = compute_reference(op, inputs)
            if ref is None:
                return True, None, f"{detail_prefix} — no reference for '{op}'"

            atol, rtol = get_tolerance(op)
            out_f = output.float().cpu()
            ref_f = ref.float().cpu()

            if out_f.shape != ref_f.shape:
                # Try flatten comparison if element count matches
                if out_f.numel() == ref_f.numel():
                    if torch.allclose(out_f.reshape(-1), ref_f.reshape(-1), atol=atol, rtol=rtol):
                        return True, True, f"{detail_prefix} — CORRECT (shape differs but values match)"
                return True, False, (
                    f"{detail_prefix} — shape mismatch: {tuple(out_f.shape)} "
                    f"vs ref {tuple(ref_f.shape)}, numel {out_f.numel()} vs {ref_f.numel()}"
                )

            correct = torch.allclose(out_f, ref_f, atol=atol, rtol=rtol)
            if correct:
                return True, True, f"{detail_prefix} — CORRECT (atol={atol})"
            else:
                max_diff = (out_f - ref_f).abs().max().item()
                return True, False, (
                    f"{detail_prefix} — INCORRECT (max_diff={max_diff:.4f}, atol={atol})"
                )

        except TimeoutError:
            signal.alarm(0)
            return False, None, f"{name}: timed out (60s)"
        except torch.cuda.OutOfMemoryError:
            signal.alarm(0)
            return False, None, f"{name}: CUDA OOM"
        except TypeError as e:
            last_error = f"{name}: all call strategies failed — {e}"
            continue
        except RuntimeError as e:
            err_str = str(e)
            if "only be called inside @jit" in err_str:
                last_error = f"{name}: is @flyc.kernel, not @flyc.jit (cannot call directly)"
                continue
            signal.alarm(0)
            return False, None, f"{name}: RuntimeError: {e}"
        except Exception as e:
            signal.alarm(0)
            return False, None, f"{name}: {type(e).__name__}: {e}"

    signal.alarm(0)
    return False, None, last_error or "All entry points failed"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_path", help="Path to kernel Python file")
    parser.add_argument("--spec", type=str, default="{}", help="JSON spec")
    parser.add_argument("--skip-runtime", action="store_true",
                        help="Skip runtime execution (compile-only mode)")
    args = parser.parse_args()

    result = {
        "compiles": False,
        "runs": None,
        "correct": None,
        "error": None,
        "details": "",
        "patterns": {},
    }

    try:
        with open(args.kernel_path) as f:
            code = f.read()
    except Exception as e:
        result["error"] = f"Cannot read file: {e}"
        print(json.dumps(result))
        return

    valid, err = check_syntax(code)
    if not valid:
        result["error"] = err
        result["details"] = "Python syntax error"
        print(json.dumps(result))
        return

    result["patterns"] = check_flydsl_patterns(code)

    if result["patterns"]["line_count"] < 10:
        result["error"] = "Trivial kernel: less than 10 lines"
        print(json.dumps(result))
        return

    compiles, err, module = try_compile_kernel(args.kernel_path)
    result["compiles"] = compiles
    if not compiles:
        result["error"] = err
        result["details"] = "FlyDSL compilation failed"
        print(json.dumps(result))
        return

    if args.skip_runtime:
        result["details"] = "Compilation passed (runtime skipped)"
        print(json.dumps(result))
        return

    spec = json.loads(args.spec)
    runs, correct, detail = try_run_kernel(module, spec)
    result["runs"] = runs
    result["correct"] = correct
    result["details"] = detail

    print(json.dumps(result))


if __name__ == "__main__":
    main()
