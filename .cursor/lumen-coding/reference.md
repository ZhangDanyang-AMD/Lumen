# Lumen Feature Parity Reference

## TE → Lumen Module Mapping

| TE Class | Lumen Class | Notes |
|----------|-------------|-------|
| `TEColumnParallelLinear` | `LumenColumnParallelLinear` | |
| `TERowParallelLinear` | `LumenRowParallelLinear` | |
| `TELayerNormColumnParallelLinear` | `LumenLayerNormLinear` | |
| `TEGroupedLinear` | `LumenGroupedLinear` | MoE expert GEMM |
| `TEFusedMLP` | `LumenFusedMLP` / `LumenGatedMLP` | Gate+up+activation+down |
| `TEActivationOp` | `_ACTIVATION_TO_AITER` | SwiGLU/GeGLU/ReGLU |
| `TEDotProductAttention` | `LumenDotProductAttention` | Flash attention |
| `TEDotProductAttentionMLA` | `LumenDotProductAttentionMLA` | Multi-Latent Attention |
| `TESpecProvider` | `LumenSpecProvider` | Megatron layer specs |
| `te.pytorch.RMSNorm` | `LumenRMSNorm` | CK/Triton |
| `te.pytorch.LayerNorm` | `LumenLayerNorm` | CK/Triton |
| `make_graphed_callables()` | `LumenGraphedCallable` | HIP Graphs |
| `te_checkpoint` | `lumen_checkpoint` | FP8-aware activation recompute |
| `get_cpu_offload_context` | `CPUOffloadManager` | Optimizer state offload |

## Attention Features

| Feature | Status | Backend |
|---------|--------|---------|
| Flash Attention | Supported | AITER CK / Triton |
| FP8 DPA | Supported | `--lumen-fp8-attn dpa` |
| FP8 MHA | Supported | `--lumen-fp8-attn mha` + Blockwise2D |
| MLA | Supported | K/V pad |
| CP A2A | Supported | SDMA option |
| CP P2P | Supported | Ring send/recv |
| CP A2A+P2P | **Missing** | Hierarchical |
| Sliding Window | **Partial** | csrc only, no Triton |
| Attention Bias | Supported | All modules |
| ALiBi | Supported | csrc + Triton |
| GQA | Supported | HQ > HKV |
| Return LSE | Supported | |
| Softmax Variants | **Deferred** | Needs AITER kernel changes |

## FP8 Quantization Features

| Feature | Status | Notes |
|---------|--------|-------|
| Delayed Scaling | Supported | Default TE path |
| Dynamic (per-tensor) | Supported | |
| Block Scaling | Supported | |
| Blockwise 2D | Supported | **Lumen-only** |
| MXFP8 | Supported | gfx950+ |
| Per-Token | Supported | **Lumen-only** |
| E4M3 / E5M2 / Hybrid | Supported | fnuz on gfx94x |
| FP4 | **Deferred** | No AMD hardware support |
| Amax History | Supported | `history_len`, `AmaxAlgo` |
| FP8 Margin | Supported | |
| FP8 Weight Gradients | Supported | |
| First/Last Layers BF16 | Supported | |

## Linear / GEMM Features

| Feature | Status | Notes |
|---------|--------|-------|
| Column/Row Parallel | Supported | |
| LayerNorm + Linear | Supported | |
| Grouped Linear (MoE) | Supported | |
| Fused MLP | Supported | Gate+up+act+down |
| FP8 GEMM | Supported | ASM/CK/Triton/hipBLASLt |
| FP8 Activation Store | Supported | Memory savings |
| TP Comm-GEMM Overlap | **Partial** | SDMA only, no chunk pipeline |
| Grad Accumulation Fusion | Supported | |
| Delay wgrad | **Partial** | Infra exists, not wired |

## FP8 GEMM Backend Dispatch

| Scaling Type | Backends (priority order) | Tuning Config |
|---|---|---|
| delayed/dynamic | hipBLASLt → CK (`gemm_a8w8_CK`) → Triton | `a8w8_tuned_gemm.csv` |
| per_token | Triton only (`gemm_a8w8_per_token_scale`) | None |
| blockwise | CK (`gemm_a8w8_blockscale`) → Triton | `a8w8_blockscale_tuned_gemm.csv` |
| mxfp8 | Triton only (`gemm_mxfp8`) | None |

## Grouped GEMM

- BF16: `aiter.ops.triton.gmm`
- FP8 per-token / mxfp8: fused MOE GEMM kernels (`moe_gemm_per_token`, `moe_gemm_mxfp8`)
- Sequential fallback: iterates per-group when no fused kernel
- wgrad: `ptgmm` (permuted-tensor GMM)

## Cross Entropy

- Triton kernel: `BLOCK_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(V))`
- Large vocab may need chunking (shared memory limit)
- SDMA all-gather route available (`use_sdma=True`, Lumen-only)

## Communication Features

| Feature | Status | Notes |
|---------|--------|-------|
| TP All-Gather / Reduce-Scatter | Supported | NCCL + SDMA option |
| TP Comm-GEMM Overlap | **Partial** | SDMA async, no chunk pipeline |
| Sequence Parallelism | Supported | |
| SDMA Collectives | Supported | **Lumen-only** (MI300X shared mem) |
| FP8 Param All-Gather | **Partial** | Manager exists, not wired |

## Lumen-Only Features (No TE Equivalent)

1. **Blockwise2D quantization** — 2D block FP8 scaling for attention Q/K/V
2. **Per-Token FP8 scaling** — per-row FP8 quantization
3. **SDMA-based TP collectives** — System DMA on MI300X shared memory
4. **SDMA cross-entropy** — SDMA-routed all-gather in parallel cross-entropy

## Checklist Before Submitting Code

- [ ] Every constant traces to a hardware fact, math property, or documented requirement
- [ ] No unnecessary abstraction layers
- [ ] Graceful degradation (fallback backends, clear error messages)
- [ ] Performance-critical path free of Python-level overhead
- [ ] No TE replacement language in docstrings/comments
- [ ] Tests use `compute_snr` or `check_close` against reference implementations
- [ ] New AITER imports guarded by probe functions
