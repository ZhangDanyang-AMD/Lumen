# MXFP4 AITER integration

Lumen pins AITER at `537314057fc0537144ef572eb28322d18a5d4b9e` through
`third_party/aiter`. Each public API is independently probed; adapters preserve
Lumen's packed-data/scale layout markers and weight-cache lifecycle.

| AITER PR | Public API | Lumen consumer |
|---|---|---|
| #5531 | `aiter.ops.triton.quant.quant.dynamic_mxfp4_quant` | `convert_to_mxfp4` |
| #5538 | `aiter.ops.triton.quant.fast_transpose.transpose_packed_fp4` | `transpose_packed_fp4` |
| #5542 | `aiter.ops.triton.fusions.activation.fused_silu_mul`, `silu_and_mul_backward` | Fused SwiGLU forward/backward, including Megatron output-buffer slices |
| #5548 | `aiter.ops.triton.quant.quant.dynamic_mxfp4_quant_blockscale` | `convert_to_mxfp4_2d`, 32×32 RTN |
| #6458 | `aiter.ops.triton.quant.fused_quant_transpose_hadamard_mxfp4.fused_quant_transpose_hadamard_mxfp4` | `dual_layout_quant_mxfp4` |
| #6476 | `aiter.ops.triton.quant.dequant_hadamard_quant_mxfp4.dequant_hadamard_quant_mxfp4` | `dequant_hadamard_quant_mxfp4` |

H16 group size/signs and activation RTN/gradient SR choices are preserved.
RTN passes inactive seed `None` and offset zero; SR forwards explicit seed/offset.
Upstream SR reserves streams according to its allocation contract and is not
bit-identical to the legacy seven-round Philox implementation. Row quantization
selects the upstream Triton backend explicitly. Endpoint handling follows AITER's
public contract; historical byte/loss baselines require paired verification.

## Remaining Lumen paths

- Non-32 block quantization and 2D stochastic rounding: the upstream blockscale
  API implements 32×32 RTN only.
- Packed transpose of already shuffled input: the upstream transpose reads
  row-major payloads and does not expose this input-layout contract. Both numbered
  training examples exercise this fallback.
- Standalone Hadamard quantization and dequant-transpose utilities remain local;
  the integrated fused H16 APIs serve the corresponding production fusion paths.
- Coalesced and expanded tile-scale swizzles remain local layout utilities.

Fallbacks report their reason. For supported row-major transpose and 2D RTN
with shuffled output, adapters call upstream and then AITER's public weight
shuffle, which introduces an extra layout pass. Performance needs measurement.
The Hugging Face FSDP gate/up activation path is separate from fused SwiGLU and
does not gain #5542 coverage from the Megatron integration.

## Packaging and verification

Vendored activation overlays are removed. The affected Dockerfiles install the
complete AITER pin with `PREBUILD_KERNELS=1`; Python sources and compiled
extensions must come from the same revision. The changed Dockerfiles have not
been rebuilt in the current verification. The tested runtime instead uses a
fixed existing image digest, isolated dependencies and source JIT compilation.

The GPU integration tests compare public-wrapper outputs, reconstruction SNR,
transpose round trips and SwiGLU gradients, and forbid legacy fallback for the
supported cases. The two training rows do not cover every related FP8/MXFP8,
grouped GEMM, attention or cross-entropy compatibility change.

See [numbered example reproduction](MXFP4_REPRODUCIBILITY.md) for the fixed
runtime, commands, latest smoke results and limits. Raw run outputs belong in the
chosen results directory; task plans and run logs are not tracked project files.
