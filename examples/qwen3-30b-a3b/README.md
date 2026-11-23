# Qwen3-30B-A3B — MXFP4 MoE pretraining (Megatron backend)

Continued pretraining of [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) on one
MI350X/MI355X node (gfx950) with 4 GPUs, expert parallelism over MORI and Lumen's SonicMoE
experts. The MXFP4 recipe runs the expert GEMMs and the attention projections in MXFP4
(1x32 E8M0 scales) and keeps every Wgrad in BF16; the BF16 reference uses the same script
with the recipe switches flipped.

| File | Purpose |
| --- | --- |
| `config_MI350X_ep4_mxfp4.sh` | Parallelism, hyperparameters, recipe switches, ROCm runtime env |
| `run_pretrain_qwen3_30b_a3b_megatron.sh` | `torchrun` launcher with the full Megatron argument set |
| `pretrain_qwen3_30b_a3b_megatron.py` | Megatron entry point (Lumen GPT builder + SonicMoE / MORI setup) |
| `run_docker.sh` | Optional container wrapper (image, paths and GPUs as variables) |
| `scripts/download_model.py` | Download the HF checkpoint / tokenizer |

## Model and parallelism

48 layers, hidden 2048, 32 query / 4 KV heads (head dim 128), per-head q/k RMSNorm,
128 experts with top-8 routing (expert FFN 768), untied embeddings, RoPE base 1e6.
EP4 / TP1 / PP1 / CP1 (DP4 for the dense part), sequence 4096, micro batch 1, global batch 8,
constant LR 1e-5 (cosine with `min_lr == lr`), no warmup, weight decay 0.1, clip 1.0,
Adam (0.9, 0.95, 1e-8), aux-loss load balancing (1e-3), BF16 params with FP32 main grads
and the distributed optimizer.

## Recipe switches

| Variable | MXFP4 (default) | BF16 reference | Meaning |
| --- | --- | --- | --- |
| `EXPERT_RECIPE` | `a4w4` | `bf16` | SonicMoE expert GEMMs (`w4a16`, `a8w4`, `a8w4f` also accepted) |
| `LUMEN_ATTN_LINEAR_RECIPE` | `a4w4` | `bf16` | Attention QKV / output projections, BF16 Wgrad |
| `LUMEN_ROPE_IMPL` | `apex` | `aiter` | Fused RoPE kernel (`lumen.patches.runtime.megatron_rope`) |
| `SONIC_MOE_MXFP4_WGRAD` | `0` | `0` | Expert Wgrad stays BF16 ... |
| `SONIC_MOE_BF16_WGRAD_MAIN_GRAD` | `1` | `1` | ... and accumulates straight into `main_grad` |
| `SONIC_MOE_PACK_CACHE`, `R80_CACHE_GENERATION` | `1` | `1` | Cache packed MXFP4 expert weights once per optimizer step |
| `R80_DUAL_WEIGHT_PACK`, `AITER_GROUPED_MXFP4_DUAL_V2` | `1` | `1` | Dual-orientation weight pack for Fprop/Dgrad |
| `LUMEN_MORI_PRESYNC_PACK` | `1` | `1` | Pack expert weights while the host waits on the MORI count sync |
| `LUMEN_ATTN_BACKEND`, `LUMEN_ATTN_AITER_TRITON_MHA` | `csrc`, `1` | `csrc`, `1` | Attention through `aiter.ops.mha.flash_attn_func`, served by AITER's Triton MHA (`0` for CK-enabled AITER builds) |
| `LUMEN_ATTN_AITER_FUSED_BWD` | `1` | `1` | AITER's fused dq/dk/dv attention backward |
| `CUDA_DEVICE_MAX_CONNECTIONS`, `GPU_MAX_HW_QUEUES`, `OMP_NUM_THREADS` | `32`, `4`, `4` | `32`, `4`, `4` | ROCm runtime; `run_docker.sh` passes them explicitly because the image presets `1`, `8`, `1` |

`LUMEN_ROPE_IMPL=apex` needs `USE_ROCM_AITER_ROPE_BACKEND=0` before apex is first imported;
`run_docker.sh` sets it, and Lumen raises at start-up if apex picked its AITER backend anyway.
Expert placement and expert-count logging (`LUMEN_EXPERT_PLACEMENT`,
`LUMEN_LOG_EXPERT_COUNTS`) are off unless set explicitly.

The entry point adds what Lumen and Megatron do not expose as switches: it sets
`precision_recipe` on the 48 SonicMoE expert modules from `EXPERT_RECIPE`, installs the
packed-weight cache lifecycle after the optimizer is built, turns `apply_rope_fusion` on,
passes `gpu_per_node` to MORI's dispatch config, and runs MORI dispatch/combine on the
current stream (with Megatron's default communication-stream path the first step does not
complete).

## Software

| Component | Version |
| --- | --- |
| Image | `docker.io/zhangdanyangamd/lumen@sha256:d6f91f8ba5ee967055dc8d1eba45b457afcacd9b251a2253c09d02d1560a757d` (PyTorch 2.9.1 / ROCm 7.2, apex, TransformerEngine) |
| Megatron-LM | `a1b00d42` (in the image at `/workspace/Megatron-LM`) |
| AITER | `third_party/aiter` submodule of this checkout |
| MORI | `third_party/mori` submodule, built into a Python site (the image has none) |
| Triton | 3.8.0 (ROCm 7.2 build) was used for the reference runs; the image ships 3.7.0 |
| tensorboard | only for W&B / TensorBoard logging (the image has none) |

Packages the image lacks are installed into host directories once (`pip install --target`
inside the image; MORI built from `third_party/mori` as in `examples/dsv4/bootstrap_env.sh`)
and mounted with `EXTRA_SITE`, a colon-separated list prepended to `PYTHONPATH`. Check out
the submodules first:

```bash
git submodule update --init --recursive third_party/aiter third_party/mori
```

## Prepare data, tokenizer and the initial checkpoint

1. **Weights / tokenizer**: `python scripts/download_model.py --output_dir /path/to/Qwen3-30B-A3B`.
   The HF directory doubles as the tokenizer (`TOKENIZER_DIR`).
2. **Initial checkpoint**: a Megatron torch-format checkpoint for TP1 / PP1 / EP4
   (`iter_0000001/mp_rank_00_00{0..3}/model_optim_rng.pt` plus
   `latest_checkpointed_iteration.txt`), converted from the HF weights with Megatron's
   `tools/checkpoint/convert.py --model-type GPT --loader <qwen3-moe HF loader>
   --saver <qwen3-moe saver> --target-tensor-parallel-size 1
   --target-pipeline-parallel-size 1 --target-expert-parallel-size 4`.
   The Qwen3-MoE loader/saver plugins are not part of this repository; verify the result
   tensor-by-tensor against the HF safetensors. Training starts with `--finetune`
   (iteration counter reset, no optimizer or RNG state).
3. **Data**: JSONL files with one `{"text": "..."}` document per line, read by
   `lumen.models.llama31.dataset` (packed to 4096-token samples, cyclic). The reference
   runs used a ShareGPT (`anon8231489123/ShareGPT_Vicuna_unfiltered`) selection of 4096
   training and 256 validation documents (`train.jsonl` sha256 `54da91d5…f797a`,
   `validation.jsonl` sha256 `00ae256b…b5682f`); with them the log shows
   `Built 1523 pretraining samples … (6241236 tokens total)` for a 3000-iteration schedule
   and `Built 64 pretraining samples … (262489 tokens total)` for validation. Any other
   corpus works the same way but is not comparable with the numbers below.

## Run

```bash
cd examples/qwen3-30b-a3b

# MXFP4 (default recipe), 3000 iterations, checkpoint every 1000
DATA_DIR=/path/to/data TOKENIZER_DIR=/path/to/Qwen3-30B-A3B CKPT_DIR=/path/to/ckpt-ep4 \
RESULTS_DIR=/path/to/results EXTRA_SITE=/path/to/site \
    bash run_docker.sh

# BF16 reference
EXPERT_RECIPE=bf16 LUMEN_ATTN_LINEAR_RECIPE=bf16 LUMEN_ROPE_IMPL=aiter \
DATA_DIR=... TOKENIZER_DIR=... CKPT_DIR=... RESULTS_DIR=... EXTRA_SITE=... bash run_docker.sh

# 10-step smoke on the 3000-iteration data schedule, no checkpoint
TRAIN_ITERS=3000 EXIT_INTERVAL=10 SAVE_INTERVAL=0 ... bash run_docker.sh

# W&B (needs tensorboard in EXTRA_SITE and outbound network)
WANDB_PROJECT=my-project WANDB_ENTITY=my-entity WANDB_EXP_NAME=qwen3-mxfp4 NETRC=$HOME/.netrc ... bash run_docker.sh
```

Inside an environment that already has everything importable, run
`bash run_pretrain_qwen3_30b_a3b_megatron.sh` directly (paths via `TRAIN_DATA`, `VALID_DATA`,
`TOKENIZER`, `LOAD_DIR`, `RESULTS_DIR`, `SAVE_DIR`). `RENDER_NODES` restricts the container
to specific GPUs (`/dev/dri/renderD<N>`).

The training samples, and therefore the batch every step sees, depend on `TRAIN_ITERS`
only. Use `EXIT_INTERVAL` for short runs that should match the first steps of a full run.

## Expected results

Reference runs (4 x MI350X, ShareGPT selection above, 3000 iterations, checkpoint every 1000).
They used the same Megatron arguments, recipe and ROCm runtime settings, with the fused
attention backward bound into AITER out of tree; this example reaches the same step time
with `LUMEN_ATTN_AITER_FUSED_BWD=1` (the further host-side patches of those runs measured
within noise and are not part of the example):

| | MXFP4 (W&B `8yxwuewg`) | BF16 (W&B `ga5slrwf`) |
| --- | --- | --- |
| Step-1 loss / grad norm | 1.095539 / 8.352 | 1.051953 / 7.876 |
| Final validation loss (iteration 3000) | 1.459992 | 1.420308 |
| Median step time, same node | 811.0 ms | 955.2 ms (1.178x) |

The two 3000-iteration runs come from different nodes of the same type; the step times are
an interleaved A/B on one node. Training loss on this small corpus falls below 0.02 by
iteration 3000 for both recipes (MXFP4 stays about 0.008 above BF16 over the last 500
iterations), and validation loss rises over the run, so the numbers are a numerics and
throughput check, not a convergence result.

This example as shipped, through `run_docker.sh`, on the 3000-iteration schedule
(`TRAIN_ITERS=3000 EXIT_INTERVAL=100`, median of iterations 51-100, one 4-GPU group; the
reference stack measured 798.5 ms MXFP4 on the same node in the same session):

| | MXFP4 | BF16 |
| --- | --- | --- |
| Step-1 loss / grad norm | 1.095539 / 8.66 | 1.051953 / 7.878 |
| Median step time | 794.0 / 794.8 ms | 926.6 ms (1.17x) |

The step-1 losses match the reference runs exactly (the forward pass is the same). The
step-1 grad norm varies by up to about 3% between runs: the fused attention backward
accumulates dK/dV with atomics, so its reduction order is not fixed.

## Known issues

* The first start JIT-compiles AITER / Triton kernels (several minutes); keep `CACHE_DIR`
  between runs.
* RCCL initialisation occasionally fails with `hipIpcGetMemHandle failed: invalid argument`
  before the first step on some nodes; retry the run.
* W&B needs outbound network access from the container (`NETWORK=host`, the default).
  Megatron logs to W&B through its TensorBoard writer, so `tensorboard` must be importable.
* `--lumen-attn-backend triton` does not work with this AITER revision (Lumen's Triton
  attention imports an AITER FP8 attention kernel module that is not present); use `csrc`.
* `LUMEN_ATTN_AITER_TRITON_MHA=1` only switches `aiter.ops.mha.flash_attn_func`; the
  process-wide `ENABLE_CK=0` would also move other AITER ops off CK. With a CK-enabled
  AITER build set `LUMEN_ATTN_AITER_TRITON_MHA=0` to use the CK MHA kernels.
* `LUMEN_ATTN_AITER_FUSED_BWD=1` needs an AITER that zero-initialises dK/dV for the fused
  kernel (the `third_party/aiter` revision of this checkout); older revisions accumulate
  into uninitialised memory and produce wrong dK/dV.
* Images may preset `CUDA_DEVICE_MAX_CONNECTIONS=1`, `GPU_MAX_HW_QUEUES=8` and
  `OMP_NUM_THREADS=1`; with those the MXFP4 step is about 17% slower (961 vs 794 ms).
  `run_docker.sh` overrides them; export the recipe values yourself when launching
  `run_pretrain_qwen3_30b_a3b_megatron.sh` inside such an image.
* The step-1 loss depends on `TRAIN_ITERS` (see above), not only on the recipe.
