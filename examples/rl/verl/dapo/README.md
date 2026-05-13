# DAPO RL Training — Qwen3-8B on MI350X

Reproduces VERL's DAPO benchmark ([Qwen3-8B-Base, AIME-2024](https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md#qwen3-8b-base-dense-model))
on 8× AMD Instinct MI350X GPUs using VERL + FSDP2 + vLLM.

## Prerequisites

```bash
docker exec -it lumen_verl_test bash
cd /workspace/Lumen/examples/rl/verl/dapo
```

**Container:** `lumen_verl_test` (rocm/sgl-dev, vLLM 0.9.2rc2, ROCm 7.0)

**Data & model** (must be on host `/dev/shm`):
- Model: `/dev/shm/model/qwen3-8b-base`
- Train: `/dev/shm/data/dapo-math-17k.parquet`
- Val:   `/dev/shm/data/aime-2024.parquet`

## Experiments

### V5b — Conservative baseline (stable, slower)

```bash
bash run_dapo_qwen3_8b_bf16_v5b.sh
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `gpu_memory_utilization` | **0.3** | Small KV cache; more headroom for training |
| `max_num_seqs` | 64 | Lower concurrency |
| Dynamic sampling | Off | All prompts processed every step |
| Avg throughput | 379 tok/s | |
| Avg step time | 810 s | |
| Stability | Survived 1,021K seqlen | Ran 32 steps, no OOM |

### V5e — Fast mode with dynamic sampling

```bash
bash run_dapo_qwen3_8b_bf16_v5e.sh
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `gpu_memory_utilization` | **0.6** | 2× KV cache vs V5b |
| `max_num_seqs` | 128 | 2× concurrency |
| Dynamic sampling | **On** | `VERL_FILTER_GROUPS_ENABLE=1` filters uninformative prompts |
| Avg throughput | 888 tok/s | 2.3× V5b |
| Avg step time | 437 s | -46% vs V5b |
| Stability | Survived 1,507K seqlen | Ran 60 steps, OOM'd at step 61 |

### Which to use

- **V5b** for maximum stability on long runs (>100 steps). Slower but handles extreme sequence lengths.
- **V5e** for faster iteration. Dynamic sampling controls memory by filtering easy prompt groups (~22–56% filtered). May OOM on very long runs when sequences grow past ~1000K tokens.

## Configuration

All scripts source `common.sh`. Key env vars you can override:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/dev/shm/model/qwen3-8b-base` | HuggingFace model path |
| `GPU_MEM_UTIL` | per-script | vLLM `gpu_memory_utilization` |
| `TOTAL_STEPS` | 275 | Training steps |
| `TEST_FREQ` | 5 | Validation every N steps |
| `SAVE_FREQ` | 20 | Checkpoint save frequency |

Checkpoints: `/root/ckpts/${PROJECT_NAME}/${EXP_NAME}/` with `resume_mode=auto`.

## Dynamic Sampling (V5e)

Controlled by environment variables in the V5e script:

```bash
VERL_FILTER_GROUPS_ENABLE=1    # enable filtering
VERL_FILTER_GROUPS_METRIC=acc  # filter by accuracy
VERL_FILTER_GROUPS_MAX_GEN=10  # max re-generation attempts
```

Filters out prompt groups where all responses are correct or all incorrect (uninformative for GRPO advantage estimation), reducing effective batch memory during training.

## ROCm Adaptations

These adaptations are pre-applied in the `lumen_verl_test` container:

- **`free_cache_engine=True`**: vLLM's `_rocm_sleep`/`_rocm_wake_up` offloads model weights and frees KV cache between rollout and training phases.
- **`enforce_eager=True`**: CUDA graph capture is disabled on ROCm.
- **`VLLM_USE_V1=1`**: Uses vLLM V1 engine architecture.

Patches in `patches/` are reference copies — they are already installed in the container.

## Metrics Dashboard

Training metrics are tracked in:

```
Lumen/outputs/fp8_training_alignment/1a_metrics_dashboard.html
```

Open in a browser to see live charts comparing V5b, V5d, and V5e
(throughput, accuracy, reward, seqlen, step time, etc.).
