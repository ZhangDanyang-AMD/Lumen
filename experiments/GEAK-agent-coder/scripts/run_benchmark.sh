#!/bin/bash
set -e
MULTITUNE="/home/danyzhan/Lumen-RL/experiments/multi-tune-agent"
MODELS_DIR="/home/danyzhan/Lumen/experiments/GEAK-agent-coder"
export PYTHONPATH="$MULTITUNE/src:$MULTITUNE:$MULTITUNE/scripts:${PYTHONPATH:-}"

kill_vllm() {
    kill -9 $(pgrep -f vllm) 2>/dev/null || true
    sleep 10
    python3 -c "import torch; [torch.cuda.empty_cache() for i in range(8)]" 2>/dev/null || true
    sleep 20
}

start_vllm() {
    local model_path="$1"
    export HIP_VISIBLE_DEVICES=0
    nohup vllm serve "$model_path" \
        --port 8000 --served-model-name "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
        --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.95 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder --enforce-eager --trust-remote-code \
        > /home/danyzhan/vllm_bench.log 2>&1 &
    for i in $(seq 1 120); do
        if curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "model"; then
            echo "vLLM ready after ${i}0s"
            return 0
        fi
        sleep 10
    done
    echo "vLLM failed"; return 1
}

for model_label in sft4 sft2 base; do
    case $model_label in
        sft4) model_path="$MODELS_DIR/outputs/qwen3-coder-full2000-4epoch-merged" ;;
        sft2) model_path="$MODELS_DIR/outputs/qwen3-coder-full2000-2epoch-merged" ;;
        base) model_path="$MODELS_DIR/models/Qwen3-Coder-30B-A3B-Instruct" ;;
    esac
    echo ">>> ${model_label}-v7"
    kill_vllm
    start_vllm "$model_path" || continue
    cd "$MULTITUNE"
    python3 scripts/run_held_out_benchmark.py "${model_label}-v7" both 2>&1 | tee "/home/danyzhan/benchmark_heldout_${model_label}-v7.log"
done
echo ">>> ALL V7 BENCHMARK COMPLETE"
kill_vllm
