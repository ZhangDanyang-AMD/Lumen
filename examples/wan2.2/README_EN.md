# SimWAM Wan2.2 VAE (NAVSIM GRPO): torch BF16 vs aiter FlyDSL convolution

> 中文版：[README.md](README.md)

Convolution performance of SimWAM's Wan2.2-TI2V-5B VAE (`WanVideoVAE38`) in the NAVSIM GRPO case: native torch BF16
against aiter's FlyDSL implicit-GEMM convolution. The FlyDSL integration, `flydsl_conv.py`, lives in this example
directory only; the Lumen library is unchanged.

The GRPO case is the FlowGRPO rollout's condition encode: each sample's current frame is encoded once and nothing is
decoded (`trainer_grpo.py`, `simwam_grpo.build_action_condition`). The navtest action-only eval encodes the same frame
the same way. So this report covers that one call:

| | shape |
|---|---|
| input | (1, 3, 1, 384, 672), BF16 |
| latent | (1, 48, 1, 24, 42) |
| decode | none |
| convolutions per encode | 30 calls, 16 distinct (11 Conv3d, 5 Conv2d) |

---

## 1. Summary

| | torch BF16 | aiter FlyDSL |
|---|---:|---:|
| 1-frame encode (`cudnn.deterministic=False`, mean of two rounds' medians, low host load) | 7.372 ms | **5.404 ms (1.36x)** |
| 1-frame encode (torch with `cudnn.deterministic=True`, as SimWAM's eval sets it) | 15.708 ms | **5.404 ms (2.91x)** |
| GPU kernel time per encode (torch.profiler) | 7.116 ms | 3.101 ms |
| of which convolution kernels | 3.816 ms | 1.234 ms (3.09x) |
| convolution calls: aiter / torch | 0 / 30 | 30 / 0 |
| tuned-table lookups: exact / borrowed / heuristic | – | 25 / 0 / 0 |
| ternary bit-exact check of the 16 convolutions | – | 16 / 16 bit-exact |
| latent SNR against FP32 | 44.6 dB | 45.0 dB |
| full-navtest PDMS (12146 tokens) | 90.27 | 90.30 (+0.03, p = 0.48, not significant) |

- The FlyDSL path is 1.36x faster than torch BF16, and 2.91x faster than torch with the `cudnn.deterministic=True`
  that SimWAM's eval uses.
- The FlyDSL encode is host-bound: 3.10 ms of GPU kernels in a 5.40 ms encode. Its time therefore moves with the host's
  CPU load -- 5.82-6.00 ms while a 6-GPU navtest ran on the same host -- while torch's encode is GPU-bound and barely
  moves (§2.1).
- No accuracy regression: every convolution's kernel is bit-exact, the latent is no further from the FP32 reference
  than torch BF16's, and the full-navtest PDMS cannot be told apart from torch BF16's. Against torch BF16 at
  rtol = atol = 1e-5, 55.9% of the elements are outside, max|d| 0.031: the BF16 rounding of a different summation
  order, see §2.3.
- In GRPO training the VAE is about 0.4% of a step; measured, FlyDSL saves about 0.4 ms per step, about 0.07% of it (a
  rough 2-GPU, 20-step measurement, §2.5).
- All 25 lookups hit exactly, provided [ROCm/aiter#7372](https://github.com/ROCm/aiter/pull/7372)'s Wan2.2 tuned
  table is installed; without it every lookup takes the heuristic tile, the encode is slower, and the log says so
  (§2.4).

Scope: AMD Instinct MI350X (gfx950), image `amdagi/veomni:rocm7.14_torch2.12_py3.12` (ROCm 7.14 / torch 2.12), aiter main
`78a14ae5d` plus #7372's Wan2.2 table at `db4c8d83`, flydsl 0.5.0, SimWAM `1433ee7`, `Wan2.2_VAE.pth`. Measured
2026-11-17.

---

## 2. Data

### 2.1 1-frame encode time

Same-card A/B: each round times the three variants one after another on one card, in one process, on the same seeded
BF16 frame; each variant 10 warmup + 100 synchronized encodes, median. The second round is on another card in the
reverse order, and the two rounds' medians are averaged. No other process was on the cards during either round, and
the host's 1-minute load average was 40-75 (384 logical cores). The command is in §3.3 (`bench_encode.py`).

| variant | round 1 (GPU 2) | round 2 (GPU 4) | mean | vs torch BF16 |
|---|---:|---:|---:|---:|
| torch BF16, `cudnn.deterministic=False` | 7.344 ms | 7.400 ms | 7.372 ms | 1.00 |
| torch BF16, `cudnn.deterministic=True` | 15.733 ms | 15.682 ms | 15.708 ms | 0.47 |
| aiter FlyDSL (this example's `flydsl_conv.py`) | 5.386 ms | 5.421 ms | **5.404 ms** | **1.36** |

Round 1 ran torch -> torch (deterministic) -> FlyDSL, round 2 FlyDSL -> torch (deterministic) -> torch. Every
coefficient of variation was within 1.2%.

**Host load**: the same two rounds, run while a 6-GPU navtest loaded the host (load average about 500), gave torch
7.359 / 7.329 ms and FlyDSL 5.997 / 5.820 ms (mean 5.909 ms, 1.24x), with FlyDSL's coefficient of variation at 3.3% /
3.9% and a two-peaked distribution (round 1's p10 was 5.557 ms). torch moved by under 1% between the two measurements,
FlyDSL by 9%, because the FlyDSL encode is host-bound (§2.2). Timed interleaved in one process, this example's patch and
Lumen `ccaeadc`'s `lumen_vae_conv.py` (the patch SimWAM's eval used) gave 5.390 / 5.367 ms at load average 450 and
5.412 / 5.379 ms at 64, with bit-identical latents: the difference is the host load, not the two patches' code paths.

### 2.2 Kernel-level evidence

`bench_encode.py` ends by recording 20 encodes with torch.profiler, summing every GPU kernel's time and splitting it by
kernel name into convolutions and the rest. Timing each convolution alone with a synchronize around it measures mostly
launch and host time (a 1x1 conv looks as long as a 3x3 one that way), so the profiler is what is reported. The two
rounds of §2.1, per encode:

| | torch BF16 (round 1 / round 2) | aiter FlyDSL (round 1 / round 2) |
|---|---:|---:|
| total GPU kernel time | 7.086 / 7.146 ms | 3.111 / 3.091 ms |
| convolution kernels | 3.784 / 3.847 ms (25 launches) | 1.238 / 1.230 ms (25 launches) |
| other kernels | 3.302 / 3.300 ms | 1.874 / 1.861 ms |
| wall time (§2.1) | 7.344 / 7.400 ms | 5.386 / 5.421 ms |

- Convolution kernels: torch's are mostly CK grouped-conv kernels (16 launches 2.67 ms, 4 launches 0.83 ms); FlyDSL's
  are 25 `conv3d_implicit_kernel` launches, 1.23 ms in total, 3.09x faster. Both convolution columns count 25 launches;
  the kernels of the five 1x1 convs carry no convolution keyword in their names and are counted under the rest.
- The other kernels also drop by about 1.43 ms. Inference (not broken down item by item): the stock path pads every
  CausalConv3d input with two zero frames in time and then runs a kT=3 convolution; after the T=1 rewrite those padded
  tensors are gone.
- torch's GPU time is close to its wall time, so it is GPU-bound; FlyDSL's GPU time is 57% of its wall time, the rest
  being the GPU waiting for the host to issue kernels, which is why FlyDSL is more sensitive to host CPU load (§2.1).
### 2.3 Numerics

| check | result |
|---|---|
| call counts | each 1-frame encode makes 30 calls, 30 on the FlyDSL kernel, 0 falling back to torch (`test_flydsl_encode_matches_stock`) |
| ternary bit-exact check | each of the 16 convolutions the FlyDSL path runs (in the form after the T=1 rewrite) on {-1, 0, 1} BF16 data: 16 / 16 bit-exact against a CPU fp32 reference (`test_flydsl_kernels_exact_on_grpo_conv_shapes`) |
| SNR against FP32 | stock torch BF16 44.6 dB, FlyDSL 45.0 dB (`verify_vae.py --backend flydsl`) |
| against torch BF16 at rtol = atol = 1e-5 | not met: 55.9% of the latent elements are outside, max|d| 0.031. Every layer rounds its fp32 accumulator to BF16, so a different summation order moves an element by a BF16 ulp and later layers carry it on. The criterion is kept as it is and the result reported as it comes out |
| the T=1 rewrite itself | on CPU (every call on torch), a small model's 1-frame encode before and after the rewrite: max|d| 3.3e-7, within 1e-5 (`test_flydsl_patch_binds_every_conv_and_t1_rewrite_is_exact`) |

**Full-navtest PDMS** (12146 tokens, cv350, 2026-11-17):

| metric | torch BF16 | aiter FlyDSL (main + #7372's table) |
|---|---:|---:|
| **PDMS** | **90.27** | **90.30** |
| NC / DAC / EP / TTC | 98.65 / 98.04 / 84.02 / 95.78 | 98.64 / 98.05 / 84.02 / 95.82 |
| ADE against GT | 0.5798 m | 0.5799 m |

- Paired comparison: +0.025, bootstrap 95% CI -0.045 to +0.097, paired t-test p = 0.48, not significant. 38 tokens flip
  a gate (21 favourable, 15 unfavourable, sign test p = 0.41). Action SNR against torch BF16: median 52.2 dB. Every one of
  the 12146 tokens ran all 30 convolutions on FlyDSL.
- Against FlyDSL without #7372's table (aiter main, every lookup heuristic): 90.30 vs 90.30, -0.003, p = 0.94.
- Against strict fp64: cv350 has no per-token fp64 result. Strict fp64 on the full navtest on node 53 is 89.73; both runs
  here are about 0.55 above it, in line with BF16's +0.57 over fp64 on node 53 -- the bias of BF16 inference as a whole,
  independent of the convolution backend.
- Configuration: SimWAM's official `eval_navsim.py`, `SimWAM.pt`, action-only, 10 steps, `seed=0`, noise drawn in fp32 on
  CPU, `cudnn.deterministic=True`, `cudnn.benchmark=False`. FlyDSL through the patch SimWAM's eval used (Lumen `ccaeadc`'s
  `patch_vae_convs(min_spatial_kernel=1, video=True)`, with the same T=1 rewrite as this example), aiter main
  `78a14ae5d` plus #7372's 11-row Wan2.2 table (byte-identical to `db4c8d83`), flydsl 0.5.0. The navtest was split into
  10 shards by token and run on GPU 0-5.

### 2.4 Lookups: why #7372's table is needed

The FlyDSL kernel looks each convolution's launch tile up in `aiter/configs/model_configs/*_bf16_tuned_conv3d.csv`. Of a
1-frame encode's 30 calls, 25 look up the table; the other 5 are 1x1 convs on the matmul fast path, which do not.

| aiter | lookups: exact / borrowed / heuristic | 1-frame encode (one batch, 15:28-15:36, GPU 4 / GPU 5) |
|---|---|---:|
| aiter main `78a14ae5d` (no Wan2.2 table) | 0 / 0 / 25 | 5.686 / 5.779 ms |
| plus #7372 `db4c8d83` (11-row Wan2.2 table) | 25 / 0 / 0 | 5.459 / 5.660 ms |

- Without #7372's table the encode is still correct, only slower. `flydsl_conv.warn_if_untuned()` logs a warning with the
  number of shapes on the heuristic; `verify_vae.py --backend flydsl` fails on any heuristic lookup, and so does
  `test_flydsl_encode_matches_stock`, with the reason in the message.
- The table's 11 rows are exactly the 11 keys a 1-frame encode looks up: the encoder's three 3x3 stride-2 Conv2d
  downsamplers and eight convolutions in the T=1 form.
- The two timings above were measured on the afternoon of 2026-11-17 with the same FlyDSL patch as SimWAM's eval (Lumen
  `ccaeadc`'s `lumen_vae_conv.py`, with the same T=1 rewrite as this example), with the table at 59 rows; trimmed to 11,
  the encode's 25 lookups return the same config call for call and the latent is bit-identical. That batch did not
  record the host load and FlyDSL differed by 1.6%-3.7% between the two cards, so it only compares with and without
  the table within the batch; §2.1 has the absolute times.

### 2.5 The VAE in a whole GRPO training step (2 GPUs, 20 steps, a rough estimate)

A separate measurement with a different configuration from the rest of this report: two MI350X on cv350 (GPU 4, 5),
Lumen-RL `d0ada15`'s `examples/Embodied_SimWAM_NAVSIM`, SimWAM `1433ee7`, Lumen `293629f`, `batch_size=1` and
`group_size=8` per GPU (the official run uses 4 GPUs), 20 optimizer steps per run with the first rollout cycle dropped
and the last 16 steps counted, i.e. 4 rollout cycles (each cycle is 1 rollout and 4 inner updates; the VAE encodes one
frame once per rollout). Measured 2026-11-17; **a rough estimate from few steps**.

| run | aiter / flydsl | Lumen attention / linear | VAE conv | per 4-step cycle (median) | VAE encode (rank0 / rank1) | VAE share of the cycle |
|---|---|---|---|---:|---:|---:|
| gE (close to the example) | the image's `891788643` / 0.3.2 | on | torch BF16 | 5.27 s (about 1.32 s per step) | 20.8 / 25.3 ms | 0.39% / 0.48% |
| gA2 | main `78a14ae5d` + #7372 `147d1a4` / 0.5.0 | off | torch BF16 | 2.20 s | 8.5 / 8.1 ms | 0.39% / 0.37% |
| gB2 | as gA2 | off | FlyDSL (all 72 convs) | 2.24 s | 6.7 / 6.8 ms | 0.31% / 0.31% |

- **The VAE is about 0.4% of a GRPO step**: 0.39% (rank0) and 0.48% (rank1) in gE, the run closest to the example.
- **Measured saving from FlyDSL** (gA2 vs gB2, the same aiter main environment with only the VAE conv changed): the VAE
  encode drops from 8.3 ms to 6.75 ms (median of the two GPUs), about 1.5 ms per rollout, about 0.4 ms per optimizer
  step, about 0.07% of the step. The two runs' cycles are 2.20 s and 2.24 s; the FlyDSL run is 1.7% slower, but in the
  opposite direction from the VAE change and far larger than 1.5 ms, so it is run-to-run noise, not FlyDSL being slower.
- Inference (not measured): the 1.5 ms per rollout against gE's 5.27 s cycle is about 0.03%. gE's VAE encode takes 20.8 ms
  and FlyDSL's saving in that environment was not measured, so this is an order of magnitude only.
- gE's other components (per 4-step cycle): inner-update forward 2.18 s (of which the ref forward is 1.50 s), rollout
  denoising 1.45 s, backward 0.86 s, old_logp 0.44 s, PDM scoring 0.13 s, video prefill 60 ms.
- The FlyDSL A/B could only run with Lumen's attention and linear off; see the version note in §3.1. gB2's FlyDSL is Lumen
  `ccaeadc`'s `patch_vae_convs(min_spatial_kernel=1, video=True)`, with the same T=1 rewrite as this example's
  `flydsl_conv.py`.

### 2.6 How FlyDSL is wired in

- **The Lumen library has no FlyDSL option**: `LumenConfig(aiter_conv=True)` / `lumen.ops.conv` is aiter's Triton conv.
  So this example carries `flydsl_conv.py`, written the way Lumen main's
  `examples/perf-veomni/veomni_patches/aiter_conv.py` is: it calls `aiter.ops.flydsl.conv_kernels.flydsl_conv_implicit`
  directly and wraps `_lookup_tuned_tile` to count each lookup as exact, borrowed or heuristic. BF16, on the GPU and
  without a gradient goes to the kernel, anything else to torch, and both are counted.
- **Two rewrites, both exact**:
  - T=1 without a feature cache (the 1-frame encode, or the first chunk of any encode): a `CausalConv3d` prepends
    `kT - 1` zero frames, so only the last temporal tap meets data, and the call is the 2-D convolution with
    `weight[:, :, -1]`. 60 of the 62 `CausalConv3d` take it; the other two are stride-2 temporal downsamplers.
  - Every other call keeps the module's own causal padding and cache concat; only the convolution under it moves to
    the kernel.
- **Causal convs are recognised by their `_padding` 6-tuple**, not by class name. perf-veomni's `lumen_vae_conv.py`
  matches the class names `QwenImageCausalConv3d` / `WanCausalConv3d`, which miss SimWAM's `CausalConv3d`, so it cannot
  be used as is.
- **The Triton path stays**: Lumen's convention is that library features are enabled through `LumenConfig.enable`, and
  the review asked for the conv patch to go through `lumencfg.enable()`; FlyDSL is an example-local patch for now, not
  in the library. So `LumenConfig(aiter_conv=True)` (Triton) remains this example's default library path, with its
  tests; FlyDSL is the optional backend (`verify_vae.py --backend flydsl`), and this README's performance data is all
  FlyDSL.

---

## 3. Runbook

### 3.1 Environment

- Image: `amdagi/veomni:rocm7.14_torch2.12_py3.12` (torch 2.12 + ROCm 7.14), gfx950.
- aiter: main `78a14ae5d` or newer, plus [ROCm/aiter#7372](https://github.com/ROCm/aiter/pull/7372)'s
  `aiter/configs/model_configs/wan22_vae_bf16_tuned_conv3d.csv` (the 11-row `db4c8d83` version).
- flydsl 0.5.0 (what aiter main's `setup.py` pins).
- A SimWAM `1433ee7` checkout and `Wan2.2_VAE.pth`:

```bash
git clone https://github.com/H-EmbodVis/SimWAM.git $SIMWAM_DIR
git -C $SIMWAM_DIR checkout 1433ee7e06a62bd8b89da74455df0d82bda7b726
hf download Wan-AI/Wan2.2-TI2V-5B --include Wan2.2_VAE.pth --local-dir $(dirname $WAN22_VAE)
export PYTHONPATH=/path/to/Lumen:$PYTHONPATH
```

`SIMWAM_DIR` defaults to `/work/SimWAM`, `WAN22_VAE` to `/work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`.

Note: Lumen `293629f` is incompatible with aiter main: the CK `mha_fwd` signature does not match and the Triton path lacks
`get_shape_from_layout`, so on aiter main Lumen's attention falls back to torch entirely; aiter main's bf16 tuned GEMM table
also has none of the GRPO shapes, so linear runs on torch too. This example is not affected (its VAE conv patch is this directory's `flydsl_conv.py` or `lumen.ops.conv`, not
Lumen's attention / linear), but keep it in mind when a whole GRPO step also enables Lumen's attention / linear (§2.5 used
the image's aiter `891788643`).

### 3.2 Correctness

```bash
python3 -m pytest -v test_wan22_vae.py           # CPU tests, plus the Triton and FlyDSL tests with a GPU
python3 verify_vae.py --backend flydsl --iters 50
python3 verify_vae.py --backend triton --iters 50
```

### 3.3 Timing

```bash
python3 bench_encode.py --order torch,torchdet,flydsl --out round1.json   # one card
python3 bench_encode.py --order flydsl,torchdet,torch --out round2.json   # another card, reverse order
```

Each round prints the three variants' medians, the lookups of one encode, and the GPU kernel time from torch.profiler.
Check that no other process is on the card first, and time at low host load: the FlyDSL encode is host-bound and
runs about 10% slower when the CPUs are saturated (§2.1).

### 3.4 Acceptance

- `verify_vae.py --backend flydsl`: `flydsl_conv.patch_vae: 62 CausalConv3d (60 with the T=1 -> conv2d rewrite) and 10 Conv2d bound`;
  `FlyDSL tuned-table lookups: {'exact': 25, 'borrowed': 0, 'heuristic': 0, 'unkeyed': 0} over 16 distinct convolutions`;
  `conv calls {'aiter': 30, 'torch': 0}`; ternary check `16 exact`; patched SNR against FP32 no lower than stock. The
  1e-5 line reads FAIL (§2.3).
- `pytest`: `test_flydsl_encode_matches_stock` fails only at the 1e-5 `assert_close` (the call-count, 16-convolution and
  25-exact-lookup assertions come before it and pass); the other FlyDSL tests pass.
- Timing: in each round, FlyDSL's median is below torch BF16's.

### 3.5 Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `aiter FlyDSL conv unavailable` | aiter too old, or flydsl not 0.5.0 | install aiter main `78a14ae5d` or newer and flydsl 0.5.0 |
| `no exact tuned row` in the log, or `heuristic` lookups | #7372's Wan2.2 table is missing | put #7372's `wan22_vae_bf16_tuned_conv3d.csv` in `aiter/configs/model_configs/` |
| `torch` calls in `conv calls` | input not BF16, on the CPU, or needs a gradient | keep the VAE and input BF16 on the GPU and encode under `torch.no_grad()` |
| the first FlyDSL run sits in JIT compilation | every new tile compiles once | let it finish; the disk cache (`FLYDSL_RUNTIME_CACHE_DIR`) is reused afterwards |
| Lumen's attention / linear fall back to torch in a GRPO step | Lumen `293629f` is incompatible with aiter main | train the wh| FlyDSL encode slower than §2.1, coefficient of variation above 2% | high host CPU load (the FlyDSL encode is host-bound, §2.2) | check the load average with `uptime` and re-time at low load; compare torch and FlyDSL interleaved in one process |
ole step with an aiter that matches Lumen (§2.5 used the image's `891788643`); this example's VAE conv is not affected |
| flydsl 0.5.0 fails hashing the bitcode, file not found | dangling `*.bc` symlinks in the ROCm SDK | point `ROCM_PATH` at a copy of the ROCm tree without the dangling links |

---

## 4. Files

| file | |
|---|---|
| `wan22_vae.py` | loads SimWAM's `wan_video_vae.py` from a checkout and the weights (strict); the GRPO shapes; `ConvCalls` (aiter / torch counts on the Triton path); the Triton path's ternary check |
| `flydsl_conv.py` | this example's FlyDSL conv: `patch_vae`, the T=1 rewrite, call and lookup counts, `warn_if_untuned`, `Trace`, the ternary check |
| `verify_vae.py` | single-GPU check, `--backend triton` / `flydsl`: call counts, lookups, 1e-5, SNR against FP32, ms, ternary check |
| `bench_encode.py` | same-card A/B timing of the 1-frame encode (torch, torch deterministic, FlyDSL) and torch.profiler kernel time |
| `test_wan22_vae.py` | pytest: CPU structure, binding and T=1 rewrite; GPU Triton and FlyDSL tests |
| `README.md` | the Chinese version of this report |
