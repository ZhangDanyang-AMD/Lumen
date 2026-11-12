# Lumen upstream AITER integration

Base: Lumen main 7b5931d. AITER pin: 537314057fc0537144ef572eb28322d18a5d4b9e.
The implementation follows the pending integration plan in Lumen dae77b240,
updated for the final upstream module names and both H16 fusion PRs.

| AITER PR | Public implementation | Lumen consumer |
| --- | --- | --- |
| #5531 | quant.quant.dynamic_mxfp4_quant, use_sr | convert_to_mxfp4 |
| #5538 | quant.fast_transpose.transpose_packed_fp4 | transpose_packed_fp4 |
| #5542 | fusions.activation.silu_and_mul_backward | fused_swiglu_backward and Megatron chunked out buffers |
| #5548 | quant.quant.dynamic_mxfp4_quant_blockscale | convert_to_mxfp4_2d, 32x32 RTN |
| #6458 | quant.fused_quant_transpose_hadamard_mxfp4 | dual_layout_quant_mxfp4 |
| #6476 | quant.dequant_hadamard_quant_mxfp4 | dequant_hadamard_quant_mxfp4 |

Each MXFP4 API is independently probed. Upstream is attempted first and legacy
fallbacks are logged. Fallback-cache keys distinguish shapes, dtype, device and
layout/SR options, but exclude random seeds and offsets to avoid unbounded
per-launch cache growth. Callers still own packed-data/scale layout markers and
weight-cache invalidation; those lifecycles are unchanged.

## Numerical contract

H16 group size, signs, activation RTN and gradient SR decisions remain unchanged.
Explicit Philox seed/offset values reach the upstream API. Upstream H16 wrappers
reserve disjoint SR streams using their own allocation contract; their random
mapping is not bit-identical to the legacy seven-round Philox implementation.
Upstream row quantization is forced to Triton to avoid automatic backend drift.
Upstream handles E8M0 endpoints according to its public contract; old loss and
byte baselines must not be presented as equivalent without paired measurement.

## Remaining legacy capabilities

- Non-32 block sizes and 2D stochastic rounding: upstream blockscale is 32x32 RTN.
- Transpose with an already shuffled input: upstream transpose reads row-major.
- Standalone fused Hadamard quantization and dequant-transpose.
- Coalesced scale swizzle and expanded tile-scale swizzle.

For row-major transpose or 2D RTN with shuffled output, the adapter invokes
upstream then performs the existing public AITER weight shuffle. These calls
trade an extra layout pass for upstream ownership; performance requires measurement.

## SwiGLU and packaging

Forward calls fusions.activation.fused_silu_mul; backward calls the public
silu_and_mul_backward with optional out. Megatron uses out slices only after
the fused installer is active. Hugging Face FSDP retains its separate gate/up
activation path, so it does not gain #5542 kernel coverage.

The vendor activation files are removed. Both affected image paths reinstall
the complete AITER pin with PREBUILD_KERNELS=1 instead of mixing new Python with
old compiled extensions. Rebuild images after initializing the upstream submodule.

## Validation

CPU forwarding tests cover the new public names, out identity and H16 layout/SR
argument forwarding. GPU tests in tests/ops/test_mxfp4_aiter_integration.py forbid
legacy fallback and compare outputs with public wrappers, check reconstruction
SNR, packed transpose round trips and SwiGLU gradients against PyTorch autograd.
The six CPU forwarding/probe/config regression tests passed on alola3. Changed
Python files passed Ruff E4/E7/E9/F/I checks, formatting checks and diff whitespace
checks; train_pretrain.sh passed bash syntax checking.

Independent Bugbot review found one P1 in the MXFP8 compatibility adaptation:
32-value scales were paired with the tuned GEMM default K tile of 128. The
adapter now requests a public tuning config and sets BLOCK_SIZE_K to 32. A CPU
regression reproduces the old assertion and verifies the fix without mutating
the tuning cache; a GPU matmul regression is included.

GPU numerical validation is pending: Slurm job 68343968 was PENDING (Priority)
at handoff. It runs the new integration suite and existing MXFP4 suite from
/scratch/users/xdai/Lumen-aiter-integration-20261110 with the exact AITER pin.
The first MI355X GPU validation on 2026-11-12 exposed an incorrect SwiGLU
backward module path and inactive Philox arguments on RTN calls. The corrected
adapter clears seed/offset for RTN and preserves explicit values for SR. Both
SwiGLU functions live in fusions.activation at this pin. After these fixes all
13 integration GPU tests passed, and six CPU regressions passed.

The original queued alola3 job status above is historical. Both numbered MXFP4
runbook cases completed 20 steps on eight MI355X GPUs with exit code zero:

| Case | Loss at step 20 | Median ms, steps 2-20 | Validation |
| --- | --- | --- | --- |
| #12 Megatron mock pretrain | 3.077231 | 346.8 | 3.036381; zero skipped/NaN iterations |
| #13 FSDP2 LoRA SFT | 1.1821 | 564.9 | finite loss and gradient norm |

Both used micro-batch 1, seq 2048 and seed 1234; #12 used GBS 8 and five
BF16 tail layers, #13 used gradient accumulation 1 and warmup 2. The existing
image was 8110221cb3ac (Torch 2.9.1/ROCm 7.2.0), with AITER extensions built
from the exact source pin in a fresh JIT directory and FlyDSL 0.5.0 supplied
in an isolated dependency directory. Its Megatron revision was a1b00d4259e.
The test worktree required compatibility patches for an inactive MoE keyword,
LayerNormLinear initialization stride, and absent attention/MLP padding masks.
Those patches are recorded with the logs; the public API fix is commit 4680446.
Both training paths used the documented legacy packed transpose for shuffled
input. Completion does not establish that every kernel ran upstream-only.
The image and newly downloaded Alpaca split differ from the old runbook, so
these numbers are not a paired loss/performance comparison. No Docker rebuild
or full training convergence result is claimed.
