# Qwen-Image DiT SFT on Lumen's FlyDSL 3D convolution — integration report and runbook

> 中文版（默认）：[README.md](README.md)

A record of one complete operator integration: the convolutions in the frozen
Qwen-Image VAE are handed to Lumen's FlyDSL implicit-GEMM kernel, run inside a
real 8-GPU VeOmni training job, and compared item by item against the same run
without it. Everything runs from committed code; reproducing it involves no
editing.

---

## 1. Findings

On 8× AMD Instinct MI355X (gfx950) at 1024×1024, all three goals are met, with
one boundary that has to be stated alongside them.

| # | Finding | Evidence | Result |
|---|---|---|---|
| 1 | **The FlyDSL 3D convolution really runs** | The training log prints `backend for conv2d: FLYDSL`, and 52 `QwenImageCausalConv3d` modules are taken over | ✅ |
| 2 | **The results are correct** | Against an FP32 encode: **51.7 dB** patched, 51.6 dB for the stock BF16 model | ✅ no accuracy loss |
| 3 | **End-to-end speedup** | One whole VAE encode: **33.05 ms → 17.22 ms, 1.92x** | ✅ |
| — | **No measurable change in step time** | The VAE convolutions are 0.56 % of a step; repeat runs of one configuration vary by ±6 % | ⚠️ see §4 |

### 1.1 The kernel really runs

Lumen's dispatcher demotes to torch on its own when a problem is outside the
kernel's range. A demoted run finishes normally with a normal loss, and is
externally indistinguishable from a successful integration that happened to
change nothing. So the criterion here is not "it ran"; the entry point reports
which backend won:

```
[lumen] vae_conv: patched 52 QwenImageCausalConv3d, skipped 9
[lumen] backend for conv2d: FLYDSL
```

The second line is the only direct evidence the kernel executed. It caught a
real silent demotion during this work — see the BF16 subsection in §5.

### 1.2 Correctness

An FP32 encode is the reference, rather than bit-identity: changing the
convolution algorithm changes the reduction order, and a BF16 rounding
difference is expected.

| Comparison | SNR |
|---|---|
| Stock BF16 model vs FP32 | 51.6 dB |
| **Patched BF16 vs FP32** | **51.7 dB** |
| One layer: conv3d rewritten as conv2d (algebra only) | 90.4 dB |
| One layer: torch conv2d vs Lumen conv2d (kernel only) | 51.0 dB |

The third row says the causal rewrite itself is clean; the whole difference
comes from the kernel's reduction order and sits at BF16 rounding level. In
training, the per-step loss deviates by 0.945 % on average, against a
reproducibility floor of 0.28 % for repeat runs of the same configuration.

### 1.3 End-to-end performance

| Scope | Stock | Patched | Speedup |
|---|---|---|---|
| **Whole VAE encode** | 33.05 ms | 17.22 ms | **1.92x** |
| Convolutions only, NCHW | 18.76 ms | 6.12 ms | 3.07x |
| Convolutions only, NHWC | 18.76 ms | 3.78 ms | 4.97x |

The encode figure comes from `verify_vae_patch.py`, which times it directly with
internal repeats, and is the most trustworthy performance number here.

Where the gain actually comes from is worth stating: it is mostly not "FlyDSL
convolves faster than torch". It is that torch's `conv3d` is poor on a tensor
whose time extent is 1; dropping to a 2-D convolution accounts for the bulk, and
the FlyDSL kernel supplies the rest. In FP32, the rewrite alone — no FlyDSL
involved — takes one encode from 9.474 ms to 4.180 ms.

### 1.4 Boundary

**This report does not claim a training speedup.** The VAE convolutions are
0.56 % of a step, the best case saves about 0.44 %, and repeat runs of the same
configuration vary by ±6 % — more than an order of magnitude larger. The
argument and two worked counter-examples are in §4.

This is a correctness-and-plumbing milestone, and the measurement baseline for
the operator and Lumen work that follows.

---

## 2. Requirements

- Hardware: 1 node, 8× AMD Instinct MI355X (gfx950), ROCm host driver
- Disk: ~150 GB (76 GB image + 54 GB model)
- Time: ~40 min, mostly the model download and two ~4-minute training runs

The default resolution is **1024×1024**, matching the official
`configs/dit/qwen_image_sft.yaml`. Raising the resolution costs almost nothing
on this node (step 3.95 s → 4.01 s, memory +1 GB); see §4. `RES=256` selects the
smaller configuration.

---

## 3. Procedure

### 3.0 Work directory and container

Everything lives under one host directory bind-mounted at `/work`, on a disk
with 150 GB free.

```bash
export ROOT=/mnt/nvme/$USER/flydsl-demo      # change me
mkdir -p "$ROOT"

docker pull amdagi/veomni:rocm7.14_torch2.12_py3.12
# expected digest:
# sha256:fb0c497921af3df3faae09c96712d375b0b690dc05349f0c8e638ed377c8c956

docker run -d --name lumen-flydsl-demo \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --ipc host --shm-size 32g \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --ulimit memlock=-1:-1 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1 \
  -e FLA_TILELANG=0 \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$ROOT":/work \
  -w /work \
  amdagi/veomni:rocm7.14_torch2.12_py3.12 sleep infinity
```

Shorthand for the rest:

```bash
dx() { docker exec -w /work lumen-flydsl-demo bash -lc "$*"; }
dx 'rocm-smi --showid | head -12'      # expect 8 devices
```

⚠️ `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` must both be set. Setting
only one aborts with `Conflicting visibility of agent-N`.

### 3.1 Get the code

```bash
dx 'git clone -b dev/flydsl-conv3d https://github.com/ZhangDanyang-AMD/Lumen.git /work/Lumen'

# VeOmni pinned to the commit this example was validated against. Later commits
# may move the trainer hook points that train_dit_lumen.py wraps.
dx 'git clone https://github.com/ByteDance-Seed/VeOmni.git /work/VeOmni &&
    cd /work/VeOmni &&
    git checkout 573848a00fcd7329c2411346c6f4a983e9f67e3f &&
    git status --short && echo "(clean if nothing above)"'
```

From here on, `env.sh` is the only place paths are defined:

```bash
export EX=/work/Lumen/examples/qwen-image-flydsl-conv
dx ". $EX/env.sh && echo \$LUMEN_PYTHONPATH"
```

**No file in either repository is modified after this point.** The runs record
`git status --short | wc -l` for both, and it must stay `0`.

### 3.2 Prepare the environment

```bash
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/setup_env.sh"
```

Three things, all idempotent:

1. flydsl 0.3.2 into its own directory (`/work/pyenv/flydsl-0.3.2`)
2. nine files added to aiter, which Lumen needs in order to import
3. VeOmni editable install with `--no-deps`

The first two are explained in §5; neither modifies a file the image shipped.

Expected tail:

```
torch       2.12.0+rocm7.14.0a20260608
torch.hip   7.14.60850
torch.cuda  None (must be None on ROCm)
devices     8
flydsl conv3d probe: True (must be True)
SETUP OK
```

> ❌ Never run `uv sync`, `pip install -e '.[gpu]'`, or install/upgrade
> torch·torchvision·triton or any `cu12`/`cu13`/`nvidia-*` package. Any of them
> replaces the ROCm torch with a CUDA build and the node is done.
> ❌ Do not set `HSA_OVERRIDE_GFX_VERSION`; gfx950 is natively recognised.

### 3.3 Model and data

The 54 GB download is the long pole. The dataset is generated locally in seconds.

```bash
dx ". $EX/env.sh && hf download Qwen/Qwen-Image --local-dir \$QWEN_IMAGE_DIR"
dx ". $EX/env.sh && du -sh \$QWEN_IMAGE_DIR && ls \$QWEN_IMAGE_DIR"
```

Expect ~54 GB with `transformer`, `text_encoder`, `vae`, `tokenizer`,
`scheduler` and `model_index.json` present.

```bash
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/make_data.py 80 \$DATA_DIR/train_80.jsonl"
```

80 = 10 steps × 8 ranks, and **the record count is load-bearing**. The DiT path
forces `dyn_bsz=False`, so each rank gets `floor(N / dp_size)` batches and the
trainer breaks out of the epoch when they run out — silently, with no error and
a zero exit code. `run_10steps.sh` generates and checks the file, so this step
is optional; it is here to make the constraint visible.

### 3.4 Verify the operator on one GPU

Three checks, about a minute total. Run them in order and stop at the first
failure — each localises something different.

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_lumen_conv.py"
```

The Lumen conv op on its own: wiring, correctness against FP32, throughput,
autograd routing, argument validation. Ends in `PASS`:

```
frequency-weighted, one VAE forward (1024x1024)
  torch bf16 (baseline)    33.84 ms   1.00x
  lumen conv2d NCHW        18.74 ms   1.86x
  lumen conv2d NHWC        14.42 ms   2.44x
correctness: worst SNR 85.0 dB (threshold 25)
```

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/trace_vae_convs.py"
```

Hooks the real `AutoencoderKLQwenImage` and lists the convolutions one encode
actually performs — 30 calls in 16 configurations — with per-layer timings. At
the default 1024, expect about
`torch 18.763 ms → lumen NCHW 6.115 ms (3.07x) / NHWC 3.778 ms (4.97x)`.

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_vae_patch.py --dtype bf16"
```

The patch itself, and **the check that matters most**. Ends in `PASS`:

```
52 convolutions on FlyDSL
accuracy: 51.7 dB vs FP32, against 51.6 dB for the stock BF16 model
speed   : whole encode 33.050 -> 17.217 ms (1.92x)
conv2d locked to: FLYDSL
```

To see the silent-demotion failure mode deliberately, run the same script in
FP32 — which is how VeOmni loads the VAE:

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_vae_patch.py --dtype fp32"
```

The backend cache comes back empty and `torch conv2d vs lumen conv2d : inf dB`
— identical, because in FP32 Lumen *is* torch. See §5.

### 3.5 Two training runs, 10 steps each

One run per mode, at 1024 by default, about 4 minutes each.

```bash
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_10steps.sh baseline"
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_10steps.sh flydsl"
```

Both share the entry point, config, dataset, seed, parallelism and interpreter
environment. **Only `LUMEN_PATCH` differs**, so any difference is attributable
to the patch.

The first ~110 s is model load, FSDP2 wrap and RCCL init, and step 1 then takes
another 70–90 s selecting kernels. **Do not kill it early.** Steady state is
about 4 s/step and a whole run is about 250 s.

The flydsl run must print:

```
[lumen] vae_conv: patched 52 QwenImageCausalConv3d, skipped 9
[lumen] backend for conv2d: FLYDSL
```

**Without the second line the kernel did not run** — go to §6.

Add a `bf16-only` run to price the BF16 cast separately from the kernel, or
`RES=256` for the smaller configuration; artifacts carry the resolution, so
runs never overwrite each other.

### 3.6 Compare

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/compare_runs.py baseline-1024 flydsl-1024"
```

Output from running exactly these steps on 8× MI355X (your numbers will differ;
the magnitudes are what matter):

```
=== did the patch actually run? ===
  baseline-1024  no [lumen] patch lines (this is expected for the baseline)
  flydsl-1024    vae_conv: patched 52 QwenImageCausalConv3d, skipped 9
  flydsl-1024    backend for conv2d: FLYDSL

=== total_loss per step ===
 step       baseline-1024         flydsl-1024
    1         0.005066125         0.005102025
    2         0.007112541         0.007285969
   ...
   10         0.005530797         0.005613535

=== loss deviation from the control ===
  flydsl-1024  max 2.440%   mean 0.945%

=== timing ===
  mode            step 1    steady     wall
  baseline-1024     73.8s     4.01s     246s
  flydsl-1024       83.4s     3.91s     247s

=== peak memory (torch max_memory_allocated) ===
  baseline-1024   38.76 GB
  flydsl-1024     38.63 GB
```

**Reading it**

- **Backend line present** — the integration works, and this is the only thing a
  single pair of runs settles outright.
- **Loss deviation 0.945 %** — unchanged in practice. Repeat runs of the same
  mode differ by 0.28 % mean / 0.86 % max, so this is the same order, and the
  two curves track step for step.
- **Steady step time** — one pair per mode cannot support a conclusion; see §4.
- **Step 1** — kernel autotune, plus JIT compilation for flydsl. One series put
  that at about 12 s; in the pair above it is invisible. Neither number should
  be quoted.
- **Peak memory** — flat. The few hundred MB the BF16 VAE saves are invisible
  against 288 GB per card.

---

## 4. Analysis

### What a single run can and cannot settle

For the first two findings, one run per mode is enough — "did the kernel run"
and "did the numbers change" are both decidable from a single run.

For step time it is not, and the gap is wide:

| | value |
|---|---|
| VAE convolutions, whole encode | 18.76 ms |
| Best case saving per step | ~15.0 ms = **0.44 %** of a step |
| Run-to-run variation of one configuration | **±6 %** |

The noise is more than an order of magnitude larger than the effect, and this
trap sprang twice during the work. At 256, one A/B pair showed the patched run
8.8 % slower, which looked like a real regression; three repeats per mode put
both configurations at the same mean (4.06 s vs 4.04 s) with fully overlapping
ranges — drift. At 1024, a 4-step pair had it 17 % slower and a 10-step pair had
it 2.5 % faster, in opposite directions. Across every run made so far, both
modes land between 3.4 and 4.3 s/step with no separation.

So: **do not quote a training speedup from this report.** The measurement that
resolves the kernel is `verify_vae_patch.py`. If a step-time number is genuinely
needed, run each mode at least three times and compare distributions.

### Why 1024

Two reasons: the kernel's advantage is much larger there, and the resolution
costs almost nothing on this node.

Same code, same 8× MI355X, the only difference being `condition_model_cfg`
(`height`/`width` 256→1024, `max_sequence_length` 64→512):

| Metric | 256 | 1024 |
|---|---|---|
| Whole VAE encode, stock → patched | 4.44 → 3.83 ms (1.16x) | **33.05 → 17.22 ms (1.92x)** |
| Convolutions only, NCHW / NHWC | 2.23x / 2.67x | **3.07x / 4.97x** |
| Convolutions as a share of a step | 0.068 % | **0.56 %** |
| Steady step time (baseline) | 3.95 s | 4.01 s |
| Peak memory | 37.79 GB | 38.76 GB |

At 256 most of these layers are launch-bound — they sit at a 30–40 µs floor — so
the kernel's efficiency has nothing to work with. At 1024 the tensors are large
enough for it to show.

**The resolution costs less than expected.** Going from 256 to 4096 image tokens
and from 64 to 512 text tokens moved the steady step from 3.95 s to 4.01 s and
peak memory by 1 GB. The step is *overhead-bound rather than compute-bound*:
MFU is around 0.2 % and GEMM is about 1 % of a step, so 16x the token work
barely registers.

That implies something more important for the work ahead: **in this training
configuration, optimising operators cannot make training faster.** 99 % of the
time is not in them. Shortening the step starts with profiling that 99 %.

---

## 5. Implementation notes

### Two necessary pieces of environment handling

**First, flydsl 0.3.2 in its own directory.** The image ships flydsl 0.1.6,
whose JIT cannot link (`could not find path component of main program:
'ld.lld'`). 0.3.2 works, but installing it over the image's copy breaks aiter,
which imports `fly_values` — a symbol 0.3.2 removed — disabling aiter's CK and
HIP ops. So 0.3.2 goes to its own directory and onto `PYTHONPATH` for this
example only; the image's environment stays byte-identical. VeOmni is
unaffected, since its source references `aiter` in exactly zero places.

One visible consequence: with that directory active, diffusers' attention
backend probe fails to import `aiter.flash_attn_func` and logs `Falling back to
native attention`. **The baseline run therefore uses the same `PYTHONPATH`**,
keeping that difference out of the comparison.
⚠️ Do not compare a flydsl run against one made without it.

**Second, nine files added to aiter.** Lumen imports triton modules at package
level that exist only on its own aiter fork; upstream aiter's wheel does not
carry them and `import lumen` fails without them. `overlay_aiter.py` only writes
files that do not already exist, never overwrites anything aiter ships, and can
be undone with `--revert`. This has nothing to do with conv.

### Why the flydsl mode also casts to BF16

`LUMEN_PATCH=vae_conv` on its own triggers no FlyDSL code at all:

> VeOmni loads the VAE with `torch_dtype=torch.float32`
> (`modeling_qwen_image_condition.py:83`, hardcoded), while the FlyDSL kernel is
> BF16-only. Lumen's capability check rejects FP32 before the dispatcher is
> reached, so every call goes to torch — no error, no warning, empty backend
> cache.

`vae_bf16` casts the frozen VAE to BF16 so the kernel is reachable. It is **not
part of Lumen and not free**: it changes what the trainer computes, which is why
it is a separate switch and why `bf16-only` exists to price it on its own.
Measured, it moves the loss 1.28 % from the control, about 4.5x the
reproducibility floor, while the conv patch alone stays near it.

The cast cannot be done naively. `vae.to(bfloat16)` by itself gives:

```
IndexError: index 0 is out of bounds for dimension 0 with size 0
```

The condition model derives the diffusion timestep from the latents' dtype, and
`scale_noise` looks it up in the FP32 schedule by exact equality, which a BF16
timestep never matches. `train_dit_lumen.py` therefore wraps `encode()` to
return FP32 latents, so every downstream dtype matches the baseline and the only
variable is the precision the VAE convolves in.

**Making the kernel accept FP32 is the one change that removes this whole
complication**, and it heads the list below.

### How the patch works

`lumen_vae_conv.py` rebinds `forward` on 52 `QwenImageCausalConv3d` modules,
touching no parameters and no module structure — which is what makes it safe
under `init_device=meta` and FSDP2.

`QwenImageCausalConv3d` pads time asymmetrically — `kT-1` zero frames in front,
none behind — then convolves with `padding=0`. With one real frame, every kernel
slice but the last multiplies zeros, so

```
conv3d(pad(x), w)  ==  conv2d(x, w[:, :, -1])
```

is an exact identity, not an approximation. Symmetric `padding=1` on a 3-D
convolution is not causal and gives a different answer. Anything that breaks the
premise — real video with `T > 1`, an inference feature cache, a strided or
dilated time axis — falls back to the module's own forward.

Nine modules are deliberately left on torch: the 1×1×1 pointwise convolutions
and the `time_conv` layers, where Lumen measures *slower* (0.85–0.94x) because
they are launch-bound. The model's plain `nn.Conv2d` downsamplers are untouched
for the same reason.

---

## 6. Appendix

### Files

| File | Role |
|---|---|
| `env.sh` | every path, one place |
| `setup_env.sh` | flydsl directory, aiter files, VeOmni install |
| `lumen_vae_conv.py` | **the patch** — T=1 causal rewrite and layer selection |
| `train_dit_lumen.py` | VeOmni entry point; `LUMEN_PATCH` = `vae_conv` / `vae_bf16` / `linear` |
| `run_10steps.sh` | one training run, `baseline` / `flydsl` / `bf16-only`, `RES` picks resolution |
| `compare_runs.py` | per-step loss, timing, memory across runs |
| `verify_lumen_conv.py` | the op on its own |
| `verify_vae_patch.py` | patch numerics, and proof FlyDSL ran |
| `trace_vae_convs.py` | real per-layer convolution inventory and timings (image VAE) |
| `trace_video_vae_convs.py` | the same for a video VAE, split into the T=1 and T>1 shares |
| `overlay_aiter.py`, `aiter_overlay/` | the nine files Lumen needs from its aiter fork |
| `qwen_image_1024.yaml` | **the default config**, 1024×1024 with `max_sequence_length` 512 |
| `make_data.py`, `qwen_image_smoke.yaml` | smoke dataset, and the 256 config (`RES=256`) |

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` under `aiter.ops.triton` on `import lumen` | aiter files not added | `python3 $EX/overlay_aiter.py --apply` |
| `could not find path component of main program: 'ld.lld'` | using the image's flydsl 0.1.6 | put `$FLYDSL_SIDECAR` first on `PYTHONPATH` |
| `cannot import name 'fly_values'` | aiter meeting flydsl 0.3.2 | expected and harmless; VeOmni never imports aiter |
| flydsl run has no `backend for conv2d: FLYDSL` | the VAE is FP32 | use `flydsl` mode, not `LUMEN_PATCH=vae_conv` alone |
| `IndexError: index 0 is out of bounds for dimension 0 with size 0` | BF16 VAE without the FP32 restore on `encode()` | use `run_10steps.sh`, which sets `vae_bf16` correctly |
| `Conflicting visibility of agent-N` | only one visibility variable set | set `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` together |
| fewer steps than requested, exit code 0 | too few JSONL records | need ≥ `steps × ranks`; regenerate with `make_data.py` |
| host cannot delete `__pycache__` | written by root in the container | `docker exec lumen-flydsl-demo rm -rf <path>` |
| `FATAL: run_10steps.sh is already running as pid N` | a run is in flight | wait; two concurrent 8-GPU runs share the GPUs and corrupt both results. Delete `$LOG_DIR/.run.lock` only once that pid is gone |
| baseline log contains `[lumen]` patch lines | two runs overlapped, both writing `$VEOMNI_DIR/log.txt` | rerun serially; the lock above now prevents it |

### Next

In priority order:

1. **Extend the patch to the T>1 video path.** Already quantified: VeOmni's Wan
   uses `AutoencoderKLWan`, and at 17 frames of 480×832 **78.6 % of one
   encode's convolution time is in shapes whose time extent exceeds 1**, which
   this example's patch does not touch — while FlyDSL is already **1.54x**
   torch on them. `lumen.ops.conv3d` takes 5-D filters; the obstacle is one
   precondition in the patch, since the T=1 identity does not hold for video.
   Lifting it means keeping the module's causal `F.pad` and making a real 3-D
   call. `trace_video_vae_convs.py` reproduces the measurement.
2. **FP32 support in the FlyDSL conv.** Removes the `vae_bf16` complication
   entirely and makes the kernel usable in this trainer as configured.
3. **A VAE-dominated workload.** At 0.56 % of a step even at 1024, the kernel
   cannot show up in training no matter how fast it gets. VeOmni's
   `offline_embedding` task, batch encoding and VAE-only serving are where it
   would.
4. **Keep the whole VAE stage in channels-last.** NHWC is another 38 % over
   NCHW (6.115 → 3.778 ms), but only if the transposes are not paid per layer.
5. **Upstream the conv3d→conv2d rewrite to diffusers.** It benefits every T=1
   (image) user, needs no FlyDSL, and changes no numerics.
6. **Backward kernels**, if the VAE ever needs training. Frozen today, so not
   blocking.
