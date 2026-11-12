"""Run numbered MXFP4 examples with explicit settings and saved evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

LUMEN = Path(__file__).resolve().parents[3]


def parse_result(case, text, returncode):
    number = r"(?:[\d.Ee+-]+|nan|inf)"
    if case == "12":
        rows = re.findall(r"iteration\s+(\d+)/\s+20.*?elapsed time per iteration \(ms\): (" + number + r").*?lm loss: (" + number + r").*?grad norm: (" + number + r")", text, re.I)
        invalid = any(int(n) != 0 for n in re.findall(r"(?:number of skipped iterations|number of nan iterations):\s*(\d+)", text, re.I))
    else:
        rows = [(n, ms, loss, grad) for n, loss, grad, ms in re.findall(
            r"step (\d+)/20 \| loss (" + number + r") \| grad_norm (" + number + r").*?step_time_ms (" + number + r")", text, re.I)]
        invalid = False
    passed = (returncode == 0 and [int(r[0]) for r in rows] == list(range(1, 21))
              and not invalid and all(math.isfinite(float(x)) for r in rows for x in r[1:]))
    result = {"exit_code": returncode, "steps": len(rows), "passed": passed,
              "legacy_messages": [line for line in text.splitlines() if "legacy Lumen kernel" in line]}
    if rows:
        result.update(loss20=float(rows[-1][2]), grad_norm20=float(rows[-1][3]),
                      median_ms_steps2_20=statistics.median(float(r[1]) for r in rows[1:]) if len(rows) > 1 else None)
    return result


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["12", "13", "all"], default="all")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model, data, output, megatron = (
        p.resolve() for p in (args.model_path, args.data_dir, args.results_dir, args.megatron_root)
    )
    cases = ["12", "13"] if args.case == "all" else [args.case]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("LUMEN_", "AITER_", "WANDB_")) or key in (
            "EXTRA_ARGS", "EXTRA_TRAIN_ARGS", "TRAIN_JSONL", "VALID_JSONL", "DRY_RUN"):
            env.pop(key)
    env.update(
        NPROC="8", MBS="1", GBS="8", GRAD_ACCUM="1", SEQ_LEN="2048", PRECISION="mxfp4", LR="4e-4",
        TRAIN_STEPS="20", WARMUP_STEPS="2", SEED="1234", NUM_LAYERS="36", TAIL_BF16="5",
        LAUNCH="native", LUMEN_DIR=str(LUMEN), LUMEN_ROOT=str(LUMEN),
        MEGATRON_ROOT=str(megatron), TOKENIZER_DIR=str(model), MODEL_PATH=str(model),
        TRAIN_DATA_PATH=str(data / "train.jsonl"), VALID_DATA_PATH=str(data / "validation.jsonl"),
        MXFP4_COMM="0", EVAL_INTERVAL="2", EVAL_ITERS="2", LUMEN_SKIP_BACKEND_SYNC="0",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=f"{LUMEN}:{megatron}:" + env.get("PYTHONPATH", ""),
        NCCL_IB_DISABLE="1", NCCL_SOCKET_IFNAME="lo", NCCL_DEBUG="WARN",
    )
    # The LoRA SFT row keeps its original evaluation cadence (no eval within 20 steps).
    commands = {
        "12": ["bash", str(LUMEN / "examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh")],
        "13": ["bash", str(LUMEN / "examples/qwen3/run_qwen3_fsdp_mxfp4_sft.sh")],
    }
    for case in cases:
        print(json.dumps({"case": case, "command": commands[case], "settings": {
            k: env[k] for k in ("NPROC", "MBS", "GBS", "SEQ_LEN", "TRAIN_STEPS", "SEED")
        }}, indent=2), flush=True)
    if args.dry_run:
        return
    for path in (model / "config.json", model / "tokenizer.json", data / "train.jsonl", data / "validation.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("results directory must be empty; use a fresh directory for each rerun")
    output.mkdir(parents=True, exist_ok=True)
    for key, directory in (("AITER_JIT_DIR", "aiter-jit"), ("TRITON_CACHE_DIR", "triton-cache"),
                           ("TORCH_EXTENSIONS_DIR", "torch-extensions"), ("TMPDIR", "tmp")):
        env[key] = str(output / directory)
        (output / directory).mkdir()
        os.environ[key] = env[key]
    expected_data = {"train.jsonl": "c35818c02caa36405863ce0929fe1de28166b00369765f38bdbb79f90db41ec8",
                     "validation.jsonl": "93fb7f68d96a01581abe3ea42634cd36cbef471cebbe471e4dbe69f53e1cface"}
    for name, expected_hash in expected_data.items():
        if digest(data / name) != expected_hash:
            raise RuntimeError(f"{name} differs from the fixed Alpaca split; run prepare_mxfp4_example_data.py")

    import aiter
    import torch
    from aiter.ops.triton.fusions.activation import fused_silu_mul, silu_and_mul_backward

    assert callable(fused_silu_mul) and callable(silu_and_mul_backward)
    aiter_root = Path(aiter.__file__).resolve().parents[1]
    expected = git(LUMEN, "ls-tree", "HEAD", "third_party/aiter").split()[2]
    if git(aiter_root, "rev-parse", "HEAD") != expected:
        raise RuntimeError(f"imported AITER does not match Lumen gitlink {expected}")
    if torch.cuda.device_count() != 8:
        raise RuntimeError("numbered runs require exactly eight visible gfx950 GPUs")
    for device in range(8):
        prop = torch.cuda.get_device_properties(device)
        if not prop.gcnArchName.startswith("gfx950"):
            raise RuntimeError("MXFP4 examples require gfx950")
        free, total = torch.cuda.mem_get_info(device)
        if free / total < 0.90:
            raise RuntimeError(f"GPU {device} has occupied memory; wait for the existing job")
    manifest = {
        "lumen": git(LUMEN, "rev-parse", "HEAD"), "lumen_status": git(LUMEN, "status", "--short"),
        "aiter": expected, "aiter_path": aiter.__file__, "megatron": git(megatron, "rev-parse", "HEAD"),
        "image": env.get("MXFP4_RUNTIME_IMAGE", "native"),
        "packages": {p: importlib.metadata.version(p) for p in (
            "torch", "triton", "transformers", "peft", "accelerate", "flydsl"
        )},
        "inputs": {str(p): digest(p) for p in (
            model / "config.json", model / "tokenizer.json", data / "train.jsonl", data / "validation.jsonl"
        )}, "environment": {k: v for k, v in env.items() if k.startswith(("LUMEN_", "AITER_"))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "gpu-preflight.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "pytest", "-q", str(LUMEN / "tests/ops/test_mxfp4_aiter_integration.py")],
                       env=env, cwd=LUMEN, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = {}
    for case in cases:
        case_env = dict(env)
        case_dir = output / ("case" + case)
        case_dir.mkdir()
        case_env.update(RESULTS_DIR=str(case_dir),
                        LUMEN_MXFP4_AUTOTUNE_CACHE=str(case_dir / "autotune.json"))
        if case == "13":
            case_env["EVAL_INTERVAL"] = "50"
        with (case_dir / "console.log").open("w") as log:
            run = subprocess.run(commands[case], env=case_env, cwd=LUMEN, stdout=log, stderr=subprocess.STDOUT)
        (case_dir / "launch.json").write_text(json.dumps({"command": commands[case], "environment": {
            k: v for k, v in case_env.items() if k in env and env[k] != os.environ.get(k)
        }}, indent=2) + "\n")
        text = (case_dir / "console.log").read_text()
        summary[case] = parse_result(case, text, run.returncode)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary[case], indent=2), flush=True)
        if not summary[case]["passed"]:
            raise RuntimeError(f"case {case} failed; inspect {case_dir / 'console.log'}")


if __name__ == "__main__":
    main()
