# Qwen-Image FlowGRPO full-parameter RL training: baseline and Lumen-accelerated

> 中文版：[README.md](README.md)

Reproduces verl-omni's official recipe
[`run_qwen_image_ocr_veomni.sh`](https://github.com/verl-project/verl-omni/blob/9ab544d81f93cbf49754aabacfa67f6fcf01b505/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh)
on one node with 8× AMD Instinct MI350X: full-parameter FlowGRPO on Qwen-Image (20B DiT), VeOmni training
engine, vLLM-Omni rollout, Qwen3-VL-8B as the OCR reward, 300 steps.

- **Baseline**: the official recipe with only paths, GPU count and ROCm attention backends changed (§4).
- **Lumen**: the baseline plus three Lumen patches on the actor (the VeOmni training engine) (§5); nothing else changes.

## 1. Results

8× MI350X, same seed. Validation reward (OCR test split, 1018 prompts):

| step | official | baseline | Lumen |
|---:|---:|---:|---:|
| 0 | 0.406 | 0.402 | 0.403 |
| 30 | | 0.644 | 0.619 |
| 60 | | 0.815 | 0.864 |
| 90 | | 0.901 | 0.940 |
| 120 | 0.937 | 0.928 | |
| 300 | | 0.949 | |

Performance (median of steps 2–90, validation and checkpoint steps excluded):

| | baseline | Lumen |
|---|---:|---:|
| actor (`old_log_prob` + `update_actor`) | 31.1 s | 27.7 s (−11%) |
| whole step | 129.5 s | 125.2 s (−3.3%) |
| throughput (images/GPU/s) | 0.494 | 0.511 |

- The official reference is verl-omni's [`docs/algo/performance.md`](https://github.com/verl-project/verl-omni/blob/9ab544d81f93cbf49754aabacfa67f6fcf01b505/docs/algo/performance.md):
  the validation curve of the VeOmni engine on 64× H100, and 0.510 images/GPU/s on 4× H200.
- The 300-step baseline takes about 12.5 h end to end (with validation and checkpoints); the Lumen run is 90 steps.
- Lumen speeds up only the actor, about 24% of a step; generation (~53 s) and OCR scoring (~39 s) are unchanged,
  so the whole step gains about 3%.
- Learning: the Lumen run follows the baseline (train reward over steps 61–90: 0.776 vs 0.770). Separate runs of one
  configuration already vary noticeably, so compare on the same machine with the same seed.

## 2. Requirements

- 8× MI350X (gfx950, 288 GB); peak memory about 208 GB per GPU.
- Host memory ≥ 1.5 TB: the 8 sleeping rollout replicas keep their pipelines in host memory (~900 GB).
- ≥ 1 TB free under `$WORK`: models 71 GB; each checkpoint 228 GB, the last 2 are kept.
- Docker; a host ROCm driver compatible with ROCm 7.2 in the image; access to GitHub, Hugging Face, Docker Hub and PyPI.

## 3. Steps

```bash
export WORK=/path/to/work                       # code, models, data and run outputs
git clone https://github.com/ZhangDanyang-AMD/Lumen.git $WORK/src/Lumen
EX=$WORK/src/Lumen/examples/perf-veomni/rl-flowgrpo-qwen-image

docker pull vllm/vllm-openai-rocm:v0.27.0
WORK=$WORK bash $EX/launch_container.sh         # container qwen-image-rl
docker exec qwen-image-rl bash $EX/setup_env.sh # ~30 min; ends with 19653 / 1018 rows and SETUP OK
```

Training (each run uses all 8 GPUs, run them one after the other; log in `$WORK/runs/<RUN_NAME>/train.log`):

```bash
docker exec -d qwen-image-rl bash -c "RUN_NAME=base bash $EX/run_flowgrpo.sh"           # baseline
docker exec -d qwen-image-rl bash -c "RUN_NAME=lumen LUMEN=1 bash $EX/run_flowgrpo.sh"  # Lumen
docker exec qwen-image-rl python3 $EX/compare.py $WORK/runs/base $WORK/runs/lumen      # vs official and each other
```

- Quick comparison: append `trainer.total_training_steps=90` (validation at steps 30/60/90), about 3.5 h per run
  (the Lumen column above uses this setting).
- W&B: add `-e WANDB_API_KEY=<key> -e WANDB_PROJECT=<project>` to `docker exec` for online logging; without a key
  the run logs offline into its directory.
- Check that Lumen is active: `docker exec qwen-image-rl bash -c 'grep -h lumen-rl /tmp/ray/session_latest/logs/worker-*.out | head'`
  shows `installed ... ['mask_all_true_to_none', 'rmsnorm_fuse', 'local_adamw']` and non-zero call counts.
- Stop a run: `docker exec qwen-image-rl bash -c 'ray stop --force; pkill -9 -f VLLM'`, then check `rocm-smi --showpids`.

## 4. Differences from the official recipe

No algorithm or hyper-parameter changes; only:

| item | official | here | why |
|---|---|---|---|
| GPUs | script default 4 | 8 (`NUM_GPUS`) | one 8-GPU node |
| actor param / optimizer offload | on | off | fits in 288 GB; with it host memory runs out and the offload copies are slow |
| attention backends | default | actor `native` (torch SDPA); rollout `TORCH_SDPA`, `enforce_eager=True` | ROCm |
| validation before training | off | on | a step-0 reference point |
| seeds | default | data and rollout fixed to 42 | same-seed comparisons |

Everything is pinned in `setup_env.sh`: the image `vllm/vllm-openai-rocm:v0.27.0`, the commits of verl, verl-omni,
VeOmni, vllm-omni and flow_grpo, and the revisions of both models.

## 5. Lumen patches (`LUMEN=1`)

Installed only in the actor processes that build the VeOmni training engine; the rollout and the judge run in their
own processes and are not touched.

| `LUMEN_PATCH` | what | numerics |
|---|---|---|
| `mask_all_true_to_none` | a training micro-batch holds the samples of one prompt, so its text mask is all True; dropping it lets attention use the flash kernel | mathematically exact; the flash path is deterministic |
| `rmsnorm_fuse` | diffusers RMSNorm forward fused by `torch.compile` ([`../veomni_patches/rmsnorm.py`](../veomni_patches/rmsnorm.py)) | 55.6 dB vs FP32, the original 52.6 dB |
| `local_adamw` | fused AdamW on the FSDP2 local shards ([`../veomni_patches/local_adamw.py`](../veomni_patches/local_adamw.py)) | bit-identical |

The other perf-veomni optimizations are not used here: the actor does not run the VAE; turning off gradient
checkpointing or reshard takes memory the rollout needs; the aiter fused backward behind `sdpa_efficient` /
`attn_triton_fwd` produces non-finite gradients on the actual RL activations.

## 6. Files

| file | runs on | purpose |
|---|---|---|
| `launch_container.sh` | host | start the container with `$WORK` mounted at the same path |
| `setup_env.sh` | container | pinned sources, the RL dependencies (image torch / vLLM / triton untouched), models, data |
| `run_flowgrpo.sh` | container | the official recipe plus the §4 overrides; `LUMEN=1` adds the §5 patches |
| `lumen_rl_patch.py` | Ray workers | installs the Lumen patches in the actor processes only and prints call counts |
| `compare.py` | anywhere | validation reward, train reward, per-phase time and throughput, against the official reference |
