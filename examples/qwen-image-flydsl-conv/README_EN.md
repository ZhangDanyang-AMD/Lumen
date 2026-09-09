# Qwen-Image / Wan2.1 VAE convolution on Lumen FlyDSL

> Public integration report and reproducible runbook. 中文版（默认）：[README.md](README.md)

This report evaluates Lumen's FlyDSL implicit-GEMM convolution in two real
VeOmni workloads: whether the kernel actually executes, whether numerics remain
sound, and how much of the local operator gain reaches end-to-end execution.

- Qwen-Image: causal Conv3d with a degenerate `T=1` axis, exactly reducible to Conv2d.
- Wan2.1: real `T>1` video and feature-cache paths that must retain full causal Conv3d.

All training measurements were collected on one node with 8× AMD Instinct
MI355X (gfx950). The document uses only public repositories, public model IDs,
and a generic `/work` container layout; it contains no cluster- or user-specific
paths.

---

## 1. Executive summary

### 1.1 Final conclusions

| Item | Qwen-Image | Wan2.1 |
|---|---|---|
| Proof the kernel ran | `backend for conv2d: FLYDSL` | `backend for conv2d: FLYDSL` and `backend for conv3d: FLYDSL` |
| Local convolution gain | 3.07× NCHW and 4.97× NHWC at 1024 | 1.59× on 81-frame T>1/cache convolutions |
| Whole VAE encode | `33.05 → 17.22 ms`, **1.92×** | `705.8 → 556.1 ms`, **1.27×** |
| Optimizable share of a training step | Whole encode about 0.82%; convolutions about 0.56% | BF16 encode about 11.1% |
| Theoretical training-step gain | About **0.40%** | About **2.05%** |
| End-to-end conclusion | Below ±6% repeat-run noise; no training-speedup claim | Below 2.2–4.3% within-mode spread; no training-speedup claim |

The conclusions supported by the data are:

1. The FlyDSL kernels execute inside real 8-GPU training and offline-embedding jobs; these are not silent torch fallbacks.
2. BF16 encode accuracy does not regress relative to FP32 for either Qwen-Image or Wan.
3. Whole VAE encode improves by 1.92× and 1.27× respectively.
4. FlyDSL's independent step-time effect is smaller than run-to-run noise in the current training configurations.
5. Wan's reproducible `4.31 → 3.42 s/step` gain (−20.6%) comes from changing the VAE from FP32 to BF16, **not from the FlyDSL kernel**. A `bf16-only` control is therefore mandatory.

### 1.2 Why a large operator gain becomes a small training gain

The end-to-end ceiling follows Amdahl's law:

```text
end-to-end time saved = optimized fraction × (1 - 1 / local speedup)
```

| Scenario | Optimized component | Local change | Share of step | Theoretical step saving |
|---|---|---:|---:|---:|
| Qwen 256 | VAE convolutions | `2.771 → 1.007 ms` | 0.068% | **0.043%** |
| Qwen 1024 | Whole VAE encode | `33.05 → 17.22 ms` | 0.824% | **0.395%** |
| Wan training | BF16 VAE encode | `380 → 310 ms` | 11.1% | **2.047%** |
| Wan offline embedding | BF16 VAE encode | `381 → 307 ms` | 20.9% | **4.066%** |

Qwen's large local gain is primarily an **algorithmic dimension reduction**.
With causal Conv3d at `T=1`, only the last temporal kernel slice contributes,
so the operation can be rewritten exactly as Conv2d. This removes work on zero
frames and avoids torch Conv3d's inefficient path for a degenerate 5-D shape;
individual layers reach 2–3.7×. But the frozen VAE runs once per step, followed
by forward, backward, optimizer, and FSDP communication for a 20.43B DiT. The
result is only about 16 ms available to save from a roughly four-second step.

Wan cannot use that reduction after the first chunk because its feature cache
contains real activations. Its 1.59× result is a genuine Conv3d kernel gain. At
81 frames, convolutions are about 58.7% of the whole encode. Amdahl's law predicts
that changing convolution from `414.38` to `261.34 ms` should reduce encode to
about 553 ms; the measured result is 556.1 ms. The local gain is preserved and
is simply diluted by the VAE's non-convolution work to 1.27×.

At the training-step level, Wan saves 70 ms from a 3.42-second BF16 step, or
2.05%. That is comparable to the 2.2–4.3% repeat-run spread. The measured
`3.42 vs 3.44 s` therefore proves neither a regression nor a speedup. Removing
the DiT and backward pass raises the offline-embedding ceiling to about 4.1%,
but the three FlyDSL runs still have a 14.4% spread.

### 1.3 Four stages dilute the local gain

1. **Only suitable convolutions are replaced.** Pointwise layers, some resamplers, and unsupported dtypes remain on torch.
2. **Convolution is only part of a VAE.** Padding, cache concatenation, normalization, activation, resampling, and latent-distribution work remain.
3. **The VAE is only part of a step.** DiT compute, backward, optimizer, FSDP/RCCL, text encoding, input processing, and output writes occupy the rest.
4. **Short jobs also pay JIT cost.** Qwen's first compile costs about 10–11 seconds; a cold Wan cache costs about 4.5 minutes. At 70 ms saved per Wan step, that cold start takes roughly 3,850 steps to amortize.

### 1.4 Scope boundary

Operator timing, whole-encode timing, numerical checks, and backend execution
are supported claims. A stable training speedup is not. Qwen's 500-step run
cycles only eight images and demonstrates plumbing and overfitting rather than
generalization. Wan long-run convergence, generated-video quality, 14B, I2V,
LoRA, VAE decode, and VAE-training backward are outside the tested scope.

---

## 2. Experimental design

### 2.1 Fixed environment

| Item | Configuration |
|---|---|
| GPU | One node, 8× AMD Instinct MI355X, gfx950 |
| Container | `amdagi/veomni:rocm7.14_torch2.12_py3.12` |
| VeOmni | `573848a00fcd7329c2411346c6f4a983e9f67e3f`, no source changes |
| FlyDSL | 0.3.2 in an isolated sidecar directory |
| Parallelism | FSDP2, eight ranks, micro batch 1 |
| Shared settings | Eager attention / RoPE, gradient checkpointing, `torch_compile=false` |

### 2.2 Completed test matrix

| Model / task | Tests completed |
|---|---|
| Qwen-Image baseline | 256×256 full-parameter SFT for 2, 10, and 500 steps; all exit 0 |
| Qwen-Image + Lumen | Eight 10-step variants/repeats at 256; independent 1024 reproduction, operator validation, and 10-step A/B |
| Wan single GPU | 17-/81-frame layer traces, whole encode, FP32/BF16 numerics, and backend checks |
| Wan training | Three modes × three repeats × 10 steps: nine runs, all exit 0 |
| Wan offline embedding | Three modes × three repeats × 10 steps: nine runs, all exit 0 |
| Wan VAE diagnostics | Both tasks × three modes × one 10-step run: six runs |

### 2.3 Three-mode attribution

| Mode | VAE dtype | Convolution | Purpose |
|---|---|---|---|
| `baseline` | FP32 | torch | Current VeOmni behavior |
| `bf16-only` | BF16, encode output restored to FP32 | torch | Measures the dtype gain alone |
| `flydsl` | BF16, encode output restored to FP32 | Lumen FlyDSL | Measures the incremental kernel effect at the same dtype |

FlyDSL convolution is currently BF16-only. Comparing the FP32 baseline directly
with BF16+FlyDSL would incorrectly attribute the dtype gain to the kernel. All
modes use the same sidecar environment, configuration, data, seed, and
parallelism.

### 2.4 Measurement rules

- A `backend for conv2d/conv3d: FLYDSL` log line is the proof of kernel execution.
- Numerics use an FP32 encode of the same input as the reference and report SNR; BF16 bit identity is not expected.
- Training steady state uses wandb timestamp intervals for steps 3–10; step 1 JIT/autotune is excluded.
- A mode-to-mode gap smaller than the within-mode repeat spread is reported as unresolved.
- `LUMEN_TIME_VAE=1` synchronizes the GPU per encode and is used only for encode timing, never for step-time comparison.

---

## 3. Measured results

### 3.1 Qwen-Image

#### Numerics and backend

```text
[lumen] vae_conv: patched 52 QwenImageCausalConv3d, skipped 9
[lumen] backend for conv2d: FLYDSL
```

| Comparison | SNR |
|---|---:|
| Stock BF16 vs FP32 | 51.6 dB |
| **Patched BF16 vs FP32** | **51.7 dB** |
| One-layer conv3d→conv2d rewrite | 90.4 dB |
| One-layer torch conv2d vs Lumen conv2d | 51.0 dB |

#### Performance and contribution

| Metric | 256×256 | 1024×1024 |
|---|---:|---:|
| Whole VAE encode, stock→patched | `4.44 → 3.83 ms` (1.16×) | `33.05 → 17.22 ms` (**1.92×**) |
| Convolutions only, NCHW / NHWC | 2.23× / 2.67× | **3.07× / 4.97×** |
| Convolution share of training step | 0.068% | 0.56% |
| Baseline steady step | 3.95 s | 4.01 s |

At 256, three baseline and three patched runs average 4.06 and 4.04 s, with
fully overlapping ranges. A single 1024 A/B measured 4.01 and 3.91 s, but the
theoretical ceiling is only about 0.4%, so that 2.5% gap cannot be attributed to
the kernel. Repeat runs of an identical configuration vary by about ±6%.

The Qwen baseline's 500-step mean loss falls from `0.02083` to `0.00621`
(−70.2%). All 500 values are finite, with no OOM, hang, or growing memory
high-water mark. This is an optimizer/plumbing and overfitting check.

### 3.2 Wan2.1 single-GPU VAE

`vae_conv_video` takes over 58 convolution modules and skips 13 whose spatial
kernel is 1:

```text
[lumen] vae_conv_video: patched 58 convolutions, skipped 13
[lumen] backend for conv2d: FLYDSL
[lumen] backend for conv3d: FLYDSL
```

All measurements below are BF16 at 480×832, with five internal repeats for each
whole encode.

| Metric | 17 frames | 81 frames |
|---|---:|---:|
| T>1 or cached calls as a share of stock convolution time | 83.3% | 91.2% |
| T>1/cache convolutions, torch→Lumen | `75.64 → 47.41 ms` (1.60×) | `377.97 → 237.23 ms` (1.59×) |
| All convolutions, stock→video patch | `90.81 → 54.69 ms` (1.66×) | `414.38 → 261.34 ms` (1.59×) |
| **Whole VAE encode** | **`153.2 → 117.1 ms` (1.31×)** | **`705.8 → 556.1 ms` (1.27×)** |
| Stock BF16 vs FP32 | 50.2 dB | 49.7 dB |
| Video-patched BF16 vs FP32 | 50.4 dB | 50.2 dB |

An intentionally incorrect symmetric-padding control scores only −2.1 dB on
one layer, showing that the validation detects temporal semantic errors. Calls
that the older image-only patch can truly reduce to 2-D account for only
10.4% / 2.3% of convolution time at 17 / 81 frames.

### 3.3 Wan 8-GPU training and offline embedding

The workload is full-parameter Wan2.1-T2V-1.3B SFT at 81 frames. The measured
VAE input is `(1, 3, 81, 368, 544)`. Training and offline embedding each run all
three modes three times.

| Task / mode | n | Steady mean | Three-run range | Spread | Peak torch memory |
|---|---:|---:|---:|---:|---:|
| Training: baseline (FP32) | 3 | 4.31 s | 4.24–4.36 s | 2.9% | 21.19 GB |
| Training: bf16-only | 3 | 3.42 s | 3.38–3.46 s | 2.2% | 20.95 GB |
| Training: BF16 + FlyDSL | 3 | 3.44 s | 3.35–3.50 s | 4.3% | 21.16 GB |
| Offline embedding: baseline | 3 | 2.73 s | 2.65–2.78 s | 4.8% | 14.73 GB |
| Offline embedding: bf16-only | 3 | 1.82 s | 1.78–1.87 s | 4.6% | 12.79 GB |
| Offline embedding: BF16 + FlyDSL | 3 | 1.76 s | 1.65–1.90 s | 14.4% | 12.85 GB |

The training baseline→bf16-only reduction of 20.6% is well above the repeat
spread. The bf16-only→FlyDSL change of +0.6% is inside that spread and is not a
regression. Offline embedding improves by 3.3% in the means, but the intervals
overlap, so it is not a stable speedup claim either.

Mean / maximum per-step training-loss deviations from baseline-r1 are
0.335%/0.648% and 0.680%/3.658% for the repeated baselines,
0.670%/2.170% for bf16-only, and 0.689%/2.354% for BF16+FlyDSL. The patch stays
within the range observed between baseline repeats; ten steps do not establish
long-run equivalence. Offline-embedding loss and grad norm are always zero and
are not numerical or convergence evidence.

### 3.4 Wan VAE timing and cold start

These synchronized diagnostics report the mean of the nine calls after the
first:

| Task | FP32 encode | BF16 encode | BF16 + FlyDSL | Kernel saving |
|---|---:|---:|---:|---:|
| Training | 1272 ms | 380 ms | 310 ms | 70 ms, about 2.0% of the BF16 step |
| Offline embedding | 1268 ms | 381 ms | 307 ms | 74 ms, about 4.1% of the BF16 step |

The video VAE has about 66 distinct convolution shapes. A cold-cache FlyDSL run
spends about 4.5 minutes compiling in step 1. With a warm cache, step 1 is only
about 1–2 seconds behind baseline. The formal performance matrix uses a warm
cache and excludes step 1.

---

## 4. Implementation

### 4.1 Qwen: exact T=1 reduction

`QwenImageCausalConv3d` prepends `kT-1` zero frames in time, appends none, and
then calls convolution with `padding=0`. With one real frame:

```text
conv3d(causal_pad(x), w) == conv2d(x, w[:, :, -1])
```

This is an exact identity, not an approximation. The patch rebinds `forward`
without changing parameters or the module tree, which keeps it safe under
`init_device=meta` and FSDP2. Cached calls, `T>1`, or temporal stride/dilation do
not use this reduction.

### 4.2 Wan: preserve causal/cache logic and replace only the kernel

Later Wan chunks prepend real cached activations, so the other temporal kernel
slices cannot be discarded. `vae_conv_video` does not reimplement the causal
logic; it rebinds `nn.Conv2d/3d._conv_forward` underneath it:

```python
def _conv_forward(self, input, weight, bias):
    return lumen_conv(
        input,
        weight,
        bias,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
        groups=self.groups,
    )
```

The module's padding, cache concatenation, and chunking remain unchanged.
Pointwise layers with spatial kernel 1 measure about 0.99–1.00× and remain on
torch.

### 4.3 BF16 constraint

VeOmni loads both VAEs in FP32, while FlyDSL convolution supports BF16 only.
FP32 is rejected before the dispatcher and silently uses torch, leaving the
backend cache empty. `flydsl` mode therefore combines:

```text
vae_bf16 + vae_conv          # Qwen
vae_bf16 + vae_conv_video    # Wan
```

A plain `vae.to(bfloat16)` breaks the downstream diffusion-timestep lookup in
the FP32 schedule. The example computes inside the VAE in BF16 and restores
latents to FP32 at the `encode()` boundary, preserving downstream dtypes. Native
FP32 convolution support is the clean way to remove this complication.

### 4.4 Sidecar environment

The image contains flydsl 0.1.6. This example installs 0.3.2 in an isolated
directory and enables it only through `PYTHONPATH`, without replacing the image
environment. `overlay_aiter.py` adds only nine missing files required for Lumen
to import and does not overwrite files shipped by aiter. Every A/B mode uses the
same sidecar environment.

---

## 5. Runbook

### 5.1 Start the container

The full two-model setup needs one 8× MI355X node, a ROCm host driver, and about
200 GB of free storage. Replace `ROOT` with any fast local-storage directory.

```bash
export ROOT=/path/to/fast-storage/lumen-flydsl-demo
mkdir -p "$ROOT"

docker pull amdagi/veomni:rocm7.14_torch2.12_py3.12

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
  -v "$ROOT":/work -w /work \
  amdagi/veomni:rocm7.14_torch2.12_py3.12 sleep infinity

dx() { docker exec -w /work lumen-flydsl-demo bash -lc "$*"; }
dx 'rocm-smi --showid | head -12'
```

Set `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` together.

### 5.2 Clone and set up

```bash
dx 'git clone -b dev/flydsl-conv3d https://github.com/ZhangDanyang-AMD/Lumen.git /work/Lumen'

dx 'git clone https://github.com/ByteDance-Seed/VeOmni.git /work/VeOmni &&
    cd /work/VeOmni &&
    git checkout 573848a00fcd7329c2411346c6f4a983e9f67e3f'

export EX=/work/Lumen/examples/qwen-image-flydsl-conv
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/setup_env.sh"
```

Expected output includes:

```text
torch.cuda  None (must be None on ROCm)
devices     8
flydsl conv3d probe: True (must be True)
SETUP OK
```

Do not run `uv sync` or `pip install -e '.[gpu]'`, install CUDA/NVIDIA wheels,
or set `HSA_OVERRIDE_GFX_VERSION`.

### 5.3 Qwen-Image validation and training

```bash
# Model and 10-step data
dx ". $EX/env.sh && hf download Qwen/Qwen-Image --local-dir \$QWEN_IMAGE_DIR"
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/make_data.py 80 \$DATA_DIR/train_80.jsonl"

# Operator, real convolution trace, and full VAE patch
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_lumen_conv.py"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/trace_vae_convs.py"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \$EXAMPLE_DIR/verify_vae_patch.py --dtype bf16"

# Fast functional A/B
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_10steps.sh baseline"
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_10steps.sh flydsl"
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py baseline-1024 flydsl-1024"
```

Use at least three repeats for performance comparison. The scripts serialize
runs to prevent two 8-GPU jobs from overlapping.

```bash
for r in r1 r2 r3; do
  dx ". $EX/env.sh && RUN_SUFFIX=$r bash \$EXAMPLE_DIR/run_10steps.sh baseline"
  dx ". $EX/env.sh && RUN_SUFFIX=$r bash \$EXAMPLE_DIR/run_10steps.sh flydsl"
done

dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py \
  baseline-1024-r1 baseline-1024-r2 baseline-1024-r3 \
  flydsl-1024-r1 flydsl-1024-r2 flydsl-1024-r3"
```

Set `RES=256` for the small-resolution configuration. The dataset must contain
at least `steps × ranks` records; the script checks this automatically.

### 5.4 Wan model and data

```bash
dx ". $EX/env.sh && hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir \$WAN_DIR"

# Download public manifests and the first 400 video entries
dx '. /work/Lumen/examples/qwen-image-flydsl-conv/env.sh &&
    RAW=$WORK/data/tom-and-jerry &&
    mkdir -p "$RAW" &&
    hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --repo-type dataset \
      --include captions.txt --local-dir "$RAW" &&
    hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --repo-type dataset \
      --include videos.txt --local-dir "$RAW" &&
    head -n 400 "$RAW/videos.txt" > "$RAW/videos.head.txt" &&
    xargs -a "$RAW/videos.head.txt" -n 25 \
      hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset \
      --repo-type dataset --local-dir "$RAW" --quiet'
```

VeOmni's converter requires aligned `captions.txt` and `videos.txt` manifests.
Build a subset from files that downloaded successfully, then convert it to
parquet:

```bash
docker exec -i -w /work lumen-flydsl-demo python3 - <<'PY'
from pathlib import Path

src = Path('/work/data/tom-and-jerry')
dst = Path('/work/data/tom-and-jerry-subset')
dst.mkdir(parents=True, exist_ok=True)
link = dst / 'videos'
if not link.exists():
    link.symlink_to(src / 'videos', target_is_directory=True)
videos = src.joinpath('videos.txt').read_text().splitlines()
captions = src.joinpath('captions.txt').read_text().splitlines()
keep = [(v, c) for v, c in list(zip(videos, captions))[:400] if (src / v).exists()]
dst.joinpath('videos.txt').write_text(''.join(f'{v}\n' for v, _ in keep))
dst.joinpath('captions.txt').write_text(''.join(f'{c}\n' for _, c in keep))
print(f'{len(keep)} clips kept')
assert len(keep) >= 80
PY

dx ". $EX/env.sh && python3 \$VEOMNI_DIR/scripts/multimodal/convert_data/tom-and-jerry.py \
  --dataset_path \$WORK/data/tom-and-jerry-subset \
  --output_dir \$WAN_DATA_DIR"
```

`WAN_DATA_DIR` must contain at least 80 parquet rows; 400 leaves useful margin.

### 5.5 Wan single-GPU validation

```bash
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \
  \$EXAMPLE_DIR/trace_video_vae_convs.py --model \$WAN_DIR --frames 17"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \
  \$EXAMPLE_DIR/trace_video_vae_convs.py --model \$WAN_DIR --frames 81"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \
  \$EXAMPLE_DIR/verify_video_vae_patch.py --model \$WAN_DIR --frames 17 --dtype bf16"
dx ". $EX/env.sh && PYTHONPATH=\$LUMEN_PYTHONPATH python3 \
  \$EXAMPLE_DIR/verify_video_vae_patch.py --model \$WAN_DIR --frames 81 --dtype bf16"
```

The last command should end in `PASS` with both conv2d and conv3d locked to
`FLYDSL`. Use `--dtype fp32` to confirm the expected silent fallback; the
backend cache should remain empty.

### 5.6 Wan training and offline embedding

Fast functional check:

```bash
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_wan_10steps.sh baseline"
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_wan_10steps.sh bf16-only"
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/run_wan_10steps.sh flydsl"

dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py \
  wan-baseline wan-bf16-only wan-flydsl"
```

Formal performance comparison requires repeats:

```bash
for r in r1 r2 r3; do
  for mode in baseline bf16-only flydsl; do
    dx ". $EX/env.sh && RUN_SUFFIX=$r bash \$EXAMPLE_DIR/run_wan_10steps.sh $mode"
  done
done

dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py \
  wan-baseline-r1 wan-baseline-r2 wan-baseline-r3 \
  wan-bf16-only-r1 wan-bf16-only-r2 wan-bf16-only-r3 \
  wan-flydsl-r1 wan-flydsl-r2 wan-flydsl-r3"
```

Switch to offline embedding with `WAN_TASK=offline_embedding`:

```bash
for r in r1 r2 r3; do
  for mode in baseline bf16-only flydsl; do
    dx ". $EX/env.sh && WAN_TASK=offline_embedding RUN_SUFFIX=$r \
      bash \$EXAMPLE_DIR/run_wan_10steps.sh $mode"
  done
done

dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/compare_runs.py \
  wanemb-baseline-r1 wanemb-baseline-r2 wanemb-baseline-r3 \
  wanemb-bf16-only-r1 wanemb-bf16-only-r2 wanemb-bf16-only-r3 \
  wanemb-flydsl-r1 wanemb-flydsl-r2 wanemb-flydsl-r3"
```

Set `LUMEN_TIME_VAE=1` only for a separate encode-timing diagnostic. Do not use
that run's step time in the formal comparison.

### 5.7 Acceptance criteria

- Every training log reaches the requested step count and exits 0.
- FlyDSL modes print the corresponding `backend ...: FLYDSL`; a `patched` line alone is insufficient.
- BF16 SNR remains in the same range as stock BF16 relative to FP32.
- Performance claims use repeat distributions; a gap below within-mode spread is reported as unresolved.
- Step 1 and cold-cache compilation are reported separately from steady state.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `aiter.ops.triton` module missing while importing Lumen | aiter overlay was not applied | Rerun `setup_env.sh` |
| `ld.lld` path error | The image's flydsl 0.1.6 is active | Put `$FLYDSL_SIDECAR` first on `PYTHONPATH` |
| `cannot import name 'fly_values'` | aiter and sidecar flydsl APIs differ | Expected here; VeOmni does not use aiter CK/HIP ops |
| No `backend for conv*: FLYDSL` | FP32, unsupported shape, or inactive sidecar | Check dtype, `PYTHONPATH`, and backend cache |
| Timestep lookup `IndexError` after casting the VAE | Encode output was not restored to FP32 | Use this example's entry point and `vae_bf16` mode |
| Fewer steps than requested with exit code 0 | Too few data rows | Provide at least `steps × ranks` records |
| `Conflicting visibility of agent-N` | HIP and CUDA visibility disagree | Set both visibility variables together |
| Baseline log contains patch lines | Runs overlapped or inherited the wrong environment | Run serially; the script clears `LUMEN_PATCH` for baseline |
| Wan appears stuck in step 1 on first use | FlyDSL is compiling many shapes | Let compilation finish and preserve the disk cache |

---

## 7. File index and next work

| File | Purpose |
|---|---|
| `lumen_vae_conv.py` | Exact T=1 reduction, T>1/cache kernel replacement, and layer selection |
| `train_dit_lumen.py` | VeOmni hook, BF16 VAE wrapper, backend report, and VAE timing |
| `run_10steps.sh` | Qwen three-mode training with `RES` and `RUN_SUFFIX` |
| `run_wan_10steps.sh` / `wan_video.yaml` | Wan training and offline-embedding entry point |
| `verify_vae_patch.py` | Qwen numerics, whole encode, and backend validation |
| `verify_video_vae_patch.py` | Wan 17-/81-frame numerics, encode, and conv3d backend validation |
| `trace_vae_convs.py` / `trace_video_vae_convs.py` | Per-layer shape, coverage, and performance attribution |
| `compare_runs.py` | Loss, steady timing, repeat spread, and memory summary |
| `survey_veomni_convs.py` | Convolution coverage survey across VeOmni architectures and VAEs |

Priorities for follow-up work:

1. Increase Wan repeat count and steady-state length, using interleaved paired A/B runs to control drift.
2. Add FP32 support to FlyDSL convolution and remove `vae_bf16` as a confounding variable.
3. Measure batched VAE encode and VAE-only serving workloads where convolution dominates.
4. Keep the complete VAE stage channels-last instead of paying per-layer layout conversions.
5. Upstream Qwen's exact T=1 Conv3d→Conv2d rewrite to diffusers.
6. Add dgrad/wgrad only if the VAE is trained in the future; the current frozen VAE needs forward only.
