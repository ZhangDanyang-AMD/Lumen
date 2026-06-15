#!/bin/bash
# Master script: run full data pipeline + benchmark in Docker
# Usage: bash run_in_docker.sh [--skip-llvm] [--skip-benchmark]
#
# Expects container "flydsl-build" already running with:
#   - rocm/vllm-dev:nightly as base (PyTorch, ROCm)
#   - /FlyDSL mounted from host
#   - /aiter mounted from host
#   - /dataprocess mounted from host
#   - /output mounted to host home

set -e
CONTAINER="flydsl-build"
SKIP_LLVM=false
SKIP_BENCHMARK=false

for arg in "$@"; do
    case $arg in
        --skip-llvm) SKIP_LLVM=true ;;
        --skip-benchmark) SKIP_BENCHMARK=true ;;
    esac
done

dexec() {
    docker exec "$CONTAINER" bash -c "$1"
}

echo "=== Step 0: Verify container ==="
dexec "python3 -c 'import torch; print(\"PyTorch\", torch.__version__); print(\"GPUs:\", torch.cuda.device_count())'"

if [ "$SKIP_LLVM" = false ]; then
    echo ""
    echo "=== Step 1: Build LLVM/MLIR (~30 min) ==="
    if dexec "test -d /FlyDSL/../llvm-project/mlir_install && echo exists" 2>/dev/null | grep -q exists; then
        echo "  LLVM already built, skipping."
    else
        echo "  Starting LLVM build..."
        dexec "cd /FlyDSL && bash scripts/build_llvm.sh -j128"
        echo "  LLVM build complete."
    fi
fi

echo ""
echo "=== Step 2: Build FlyDSL (~5 min) ==="
if dexec "python3 -c 'import flydsl; print(\"OK\")'" 2>/dev/null | grep -q OK; then
    echo "  FlyDSL already installed."
else
    dexec "cd /FlyDSL && bash scripts/build.sh -j128 && pip install -e ."
    dexec "python3 -c 'import flydsl; print(\"FlyDSL OK\")'"
fi

echo ""
echo "=== Step 3: Run base pipeline (manifest + CPT + SFT + RL) ==="
dexec "git config --global --add safe.directory /FlyDSL 2>/dev/null; git config --global --add safe.directory /aiter 2>/dev/null; true"
dexec "FLYDSL_ROOT=/FlyDSL AITER_ROOT=/aiter OUTPUT_DIR=/output python3 /dataprocess/process_all_v2.py"

if [ "$SKIP_BENCHMARK" = false ]; then
    echo ""
    echo "=== Step 4: Benchmark quality grading ==="
    dexec "FLYDSL_ROOT=/FlyDSL AITER_ROOT=/aiter python3 /dataprocess/benchmark_filter.py \
        --manifest /output/manifest.json \
        --output /output/graded_manifest.json \
        --max-tests 30"

    echo ""
    echo "=== Step 5: Re-run pipeline with graded manifest ==="
    dexec "FLYDSL_ROOT=/FlyDSL AITER_ROOT=/aiter OUTPUT_DIR=/output python3 /dataprocess/process_all_v2.py"
fi

echo ""
echo "=== Done! ==="
echo "Dataset at: /home/danyzhan/flydsl-agent-dataset/"
ls -lh /home/danyzhan/flydsl-agent-dataset/data/*/*.jsonl 2>/dev/null
