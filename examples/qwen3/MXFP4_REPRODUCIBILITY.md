# Numbered MXFP4 examples 12 and 13

These are eight-GPU, 20-step smoke tests, not convergence or speedup evidence.
Row 12 is Megatron pretraining from random initialization with a Qwen3 tokenizer,
36 layers and the last five layers kept BF16. Its generated mock corpus is shared
by train/validation/test: the validation loss is **not held-out validation**.
Row 13 loads Qwen3-8B pretrained weights and trains LoRA with FSDP2. MXFP4
communication is disabled (`MXFP4_COMM=0`); evaluation interval 50 means the
20-step test does **not** produce a validation loss. Full-parameter FSDP
pretraining is a different, unnumbered example.

## Fixed runtime

- Runtime image: `zhangdanyangamd/lumen@sha256:d6f91f8ba5ee967055dc8d1eba45b457afcacd9b251a2253c09d02d1560a757d`
- AITER: `537314057fc0537144ef572eb28322d18a5d4b9e` (Lumen gitlink)
- Megatron-LM: `a1b00d4259e92dc4a07a0be2c24088fe827f4b6e`
- torch `2.9.1+rocm7.2.0.lw.git7e1940d4`, Triton `3.7.0+amd.rocm7.2.0.git89002410`
- transformers `5.16.0`, peft `0.20.0`, accelerate `1.14.0`, numpy `2.4.1`
- Install `flydsl==0.5.0` into an isolated directory ahead of the image packages;
  the image's FlyDSL 0.3.2 cannot import this AITER revision.
- Model/tokenizer: `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`.

Mount the reviewed Lumen checkout at `/rerun/Lumen`, the exact AITER source at
`/rerun/aiter`, writable task results/dependencies under `/rerun`, and the model
snapshot read-only at `/model`. Start the image with `/dev/kfd`, `/dev/dri`,
`--ipc=host --network=host` and video/render groups. The image supplies
`/workspace/Megatron-LM`; use a clean clone to exclude any image/local patches.
For mounts owned by a different UID, configure Git safe.directory for the exact
trusted paths (do not use a wildcard). Example inside the container:

```bash
pip install --no-cache-dir --no-deps --target /rerun/deps flydsl==0.5.0
git clone --no-hardlinks /workspace/Megatron-LM /rerun/Megatron-clean
git -C /rerun/Megatron-clean checkout --detach a1b00d4259e92dc4a07a0be2c24088fe827f4b6e
git config --global --add safe.directory /rerun/Lumen
git config --global --add safe.directory /rerun/aiter
export PYTHONPATH=/rerun/deps:/rerun/aiter:/rerun/Lumen:/rerun/Megatron-clean
export MXFP4_RUNTIME_IMAGE=zhangdanyangamd/lumen@sha256:d6f91f8ba5ee967055dc8d1eba45b457afcacd9b251a2253c09d02d1560a757d
python /rerun/Lumen/examples/qwen3/scripts/prepare_mxfp4_example_data.py /rerun/data
python /rerun/Lumen/examples/qwen3/scripts/run_mxfp4_examples.py \
  --case all --model-path /model --data-dir /rerun/data \
  --megatron-root /rerun/Megatron-clean --results-dir /rerun/results-clean
git -C /rerun/Megatron-clean diff > /rerun/results-clean/megatron-launcher.patch
cp /rerun/Megatron-clean/megatron/core/transformer/megatron_fused_rmsnorm.py \
  /rerun/results-clean/megatron_fused_rmsnorm.py
```

Use a fresh results directory and eight idle gfx950 GPUs. `--dry-run` prints the
commands without requiring a GPU. The runner checks the imported AITER gitlink,
input hashes, architecture/free memory, then runs the MXFP4 GPU integration tests
before training. It uses independent AITER/Triton/torch JIT caches and independent
per-case autotune caches. It checks exit status, steps 1–20, finite loss/gradient
norm/timing and skipped/NaN iteration counters. Preserve `manifest.json`,
`gpu-preflight.log`, `summary.json`, `case*/launch.json` and `case*/console.log`.
The manifest records source commits, local modifications, packages and input
hashes. All tracked source files must be clean at startup. With apex available,
the checked-in row-12 launcher automatically applies Lumen's RMSNorm/ROPE source
bridge via `examples/llama2/scripts/patch_gpt_layer_specs.py`; the resulting
Megatron diff and generated RMSNorm module must be saved with the evidence.
This automatic integration step is required for this runtime and differs from
the old ad hoc compatibility edits. Start each rerun from a fresh Megatron clone.

Both cases use MBS 1, sequence length 2048, seed 1234 and 20 steps; row 12 GBS 8,
row 13 gradient accumulation 1 and LR 4e-4 with two warmup steps. Do not use the
row-12 launcher's larger defaults (MBS 2, GBS 128, sequence 8192, 50 steps) to
reproduce this smoke table. Synchronization remains enabled for validation.

The data preparer verifies the public Stanford Alpaca source SHA256
`2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419`, splits
the first 51,490 rows for training and last 512 for validation, without shuffling.
The runner verifies both JSONL hashes, rather than accepting a changed dataset.

## Kernel scope and evidence

The normal 32-element MXFP4 quantization and fused Hadamard/requantization paths,
and SwiGLU forward/backward, call the pinned AITER public APIs. Packed transpose
on already shuffled input still uses the documented Lumen implementation: AITER's
public transpose API does not express this input-layout contract. Non-32 block
quantization and 2D stochastic rounding also retain explicit legacy paths.
Thus these runs must not be described as using exclusively AITER kernels.
See [integration mapping](../../docs/aiter_public_api_integration_2026-11-10.md).

On 2026-11-12, both rows passed from committed Lumen
`00edddf545fffdba2dcb2d6c9bb00776a1decb27` and clean initial AITER/Megatron
sources on `smci355-ccs-aus-n01-13` (8×MI355X). No manual compatibility edits were
applied. The checked-in Megatron bridge described above was applied automatically.

| Row | Steps / exit | Median ms, steps 2–20 | Loss at step 20 | Gradient norm at step 20 | Validation |
|---|---|---|---|---|---|
| 12 Megatron MXFP4 | 20 / 0 | 322.5 | 3.095294 | 38.790 | 3.054681, final validation-set pass; shared mock corpus |
| 13 FSDP2 MXFP4 LoRA | 20 / 0 | 523.3 | 1.1998 | 0.3203 | Not run within 20 steps |

CPU regressions: 3 passed. CPU/GPU initializer dispatch and stride metadata:
2 passed. MXFP4 AITER GPU integration: 13 passed. Bash syntax and dry-run passed.
Both training logs contain finite loss/gradient norms, and row 12 has zero skipped
and NaN iterations. Both still exercise the documented shuffled-input packed
transpose fallback. Evidence: [manifest and compressed logs](../../docs/evidence/mxfp4-2026-11-12/).

The earlier MI355X runs (346.8 ms row 12, 564.9 ms row 13) used temporary runtime
compatibility patches; they remain historical debugging evidence. The old
13-row MI350X table is also historical, not a paired baseline for this runtime.
