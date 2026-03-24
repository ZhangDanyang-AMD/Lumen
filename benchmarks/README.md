# Lumen Benchmarks

Performance benchmarks validating Lumen's key feature optimizations.

All benchmarks use **Llama 3.1 8B** dimensions by default:
`hidden_size=4096`, `intermediate_size=14336`, `num_attention_heads=32`,
`num_key_value_heads=8`, `head_dim=128`.

## Benchmarks

| # | File | Lumen Features Exercised | Requirements |
|---|------|--------------------------|-------------|
| 1 | `bench_kernel_launch.py` | FP8 quantized_linear (7 scaling modes), fused MoE, FP8 activation store, attention backends (csrc/triton), norm+GEMM pipeline | 1 GPU + AITER |
| 2 | `bench_comm_overlap.py` | SdmaTpComm async allgather/reduce-scatter, column/row parallel overlap patterns, NCCL vs SDMA comparison | 2+ GPUs |
| 3 | `bench_fp8_param_allgather.py` | FP8ParamManager on LumenGatedMLP/LumenFusedMLP, dequant hooks, param compression SNR | 1 GPU |
| 4 | `bench_rope_fusion.py` | apply_rotary_pos_emb (1D), fused_rope (Q+K), 2D vision RoPE, 3D video RoPE, GQA configs, NeoX vs GPT-J | 1 GPU + AITER |
| 5 | `bench_wgrad_delay.py` | _DeferredWgrad API, wgrad-forward overlap, two-layer pipeline, gradient_accumulation_fusion, wgrad-comm overlap | 1 GPU |
| 6 | `bench_fused_pipeline.py` | Pipelined AG+GEMM / GEMM+RS overlap, NCCL vs SDMA backend, backward overlap isolation, chunk sweep | 2+ GPUs |
| 7 | `bench_e2e_fusion.py` | End-to-end transformer layer with all fusion strategies, TP scaling sweep, bandwidth reporting | 2+ GPUs (8 for TP scaling) |

## Quick Start

### Single GPU

```bash
# Run all single-GPU benchmarks
pytest benchmarks/ -v -s -k "not Distributed and not NCCL and not Sdma"

# Individual benchmarks
python -m benchmarks.bench_kernel_launch
python -m benchmarks.bench_rope_fusion
python -m benchmarks.bench_fp8_param_allgather
python -m benchmarks.bench_wgrad_delay
```

### Multi-GPU

```bash
# 2-GPU communication overlap
torchrun --nproc_per_node=2 -m benchmarks.bench_comm_overlap

# 8-GPU
torchrun --nproc_per_node=8 -m benchmarks.bench_comm_overlap
```

### Fused Pipeline

```bash
# Forward overlap
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k fwd
# NCCL vs SDMA
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k SdmaColumn
# Backward isolation
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_fused_pipeline.py -v -s -k BackwardOverlap
```

### E2E Transformer Layer

```bash
# Single layer with all fusion strategies
torchrun --nproc_per_node=2 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TransformerLayer
# TP scaling sweep (requires 8 GPUs)
torchrun --nproc_per_node=8 -m pytest benchmarks/bench_e2e_fusion.py -v -s -k TPScaling
```

## Tracing

Generate Chrome/Perfetto-compatible `.json` trace files that visualize the GPU
timeline (kernel names, durations, stream concurrency).

```bash
# All trace groups
python -m benchmarks.run_traces

# Single group
python -m benchmarks.run_traces --only kernel   # kernel launch
python -m benchmarks.run_traces --only comm     # comm-compute overlap
python -m benchmarks.run_traces --only wgrad    # wgrad delay
python -m benchmarks.run_traces --only rope     # RoPE fusion
python -m benchmarks.run_traces --only fp8      # FP8 param

# Single trace function (or multiple)
python -m benchmarks.run_traces --run trace_fused_moe
python -m benchmarks.run_traces --run trace_fused_moe trace_fused_qk_rope

# List all available trace function names
python -m benchmarks.run_traces --list
```

Trace files are written to `benchmarks/traces/` (git-ignored). Open them at
`chrome://tracing` or https://ui.perfetto.dev.

| Group | Trace Files | What to Look For |
|-------|-------------|------------------|
| kernel | `kernel_fused_moe.json`, `kernel_manual_moe.json`, `kernel_scaling_*.json`, `kernel_attn_*.json` | Kernel count reduction, kernel names, fused vs decomposed |
| comm | `comm_column_*.json`, `comm_pipeline_*.json` | Stream concurrency: allgather on comm stream overlapping GEMM on compute stream |
| wgrad | `wgrad_sequential.json`, `wgrad_overlapped.json`, `wgrad_4layer_*.json` | Deferred wgrad on secondary stream overlapping forward on default stream |
| rope | `rope_fused_s2048.json`, `rope_pytorch_s2048.json`, `rope_fused_qk.json` | Single fused kernel vs 4+ decomposed PyTorch ops |
| fp8 | `fp8_quant.json`, `fp8_dequant.json`, `fp8_forward_*.json` | Quant/dequant kernel overhead, BF16 vs FP8 param forward pipeline |

## Feature Coverage

### 1. Single-GPU Kernel Launch Reduction (`bench_kernel_launch.py`)

| Feature | What's Measured |
|---------|----------------|
| `quantized_linear` scaling modes | Latency of all 7 modes: none, delayed, dynamic, per_token, blockwise, blockwise2d, mxfp8 |
| `fused_moe_triton` | End-to-end fused MoE vs individual `fused_topk` + per-expert GEMMs + `fused_unpermute` |
| `fp8_activation_store` | `LumenGatedMLP(fp8_activation_store=True)` vs `False`: fwd, bwd, and activation memory |
| Attention backends | `aiter_csrc` vs `aiter_triton`: causal, GQA (32Q/8KV), sliding window, seqlen sweep |
| Norm + GEMM pipeline | `rmsnorm` → `quantized_linear` with different FP8 scaling modes |

### 2. Multi-GPU Comm-Compute Overlap (`bench_comm_overlap.py`)

| Feature | What's Measured |
|---------|----------------|
| Column-parallel overlap | Allgather on comm stream, local-shard GEMM on compute stream |
| Row-parallel overlap | Reduce-scatter on comm stream, GEMM on compute stream |
| `SdmaTpComm` async | `allgather_dim0_async` / `reduce_scatter_dim0_async` with true SDMA hardware |
| NCCL vs SDMA | Direct comparison of overlap ratios |

Overlap ratio: `1 - (T_overlapped / (T_comm + T_compute))`

### 3. FP8 Param All-Gather (`bench_fp8_param_allgather.py`)

| Feature | What's Measured |
|---------|----------------|
| `FP8ParamManager` | Applied to `LumenGatedMLP`, `LumenFusedMLP`, multi-layer stacks |
| Dequant hooks | Forward latency with BF16 vs FP8 params + dequant overhead |
| Param compression | ~2x memory reduction, per-layer bandwidth savings |
| Round-trip SNR | Quantization quality (>15 dB param, >10 dB output) |

### 4. RoPE Fusion (`bench_rope_fusion.py`)

| Feature | What's Measured |
|---------|----------------|
| `apply_rotary_pos_emb` | 1D RoPE across S={128..8192}, NeoX vs GPT-J interleaved |
| `fused_rope` | Q+K fused vs separate `apply_rotary_pos_emb` x2 |
| `apply_rotary_pos_emb_2d` | Vision 2D RoPE (14x14, 16x16, 32x32) |
| `apply_rotary_pos_emb_3d` | Video 3D RoPE (4x8x8) |
| GQA configs | H_kv = {1, 4, 8, 32} with H_q = 32 |

### 5. Wgrad Delay (`bench_wgrad_delay.py`)

| Feature | What's Measured |
|---------|----------------|
| `_DeferredWgrad` API | `defer()` + `execute()` vs eager wgrad |
| Wgrad-forward overlap | Deferred dW on secondary stream while next-layer forward runs |
| Two-layer pipeline | End-to-end: layer-2 dW overlaps layer-1 dX |
| `gradient_accumulation_fusion` | `w.main_grad.add_(dw)` vs `w.grad = dw` |
| Wgrad-comm overlap | Deferred dW overlaps with reduce-scatter |

### 6. Fused Pipeline Comm-GEMM Overlap (`bench_fused_pipeline.py`)

| Feature | What's Measured |
|---------|----------------|
| Column-parallel AG+GEMM | Fused vs sequential forward latency, overlap ratio |
| Row-parallel GEMM+RS | Fused vs sequential forward latency, overlap ratio |
| NCCL vs SDMA backend | Head-to-head fused pipeline latency comparison |
| Backward overlap isolation | dgrad+RS and AG+dgrad measured separately |
| Chunk sweep | Pipeline chunk count (1, 2, 4, 8) sensitivity |

### 7. End-to-End Transformer Layer Fusion (`bench_e2e_fusion.py`)

| Feature | What's Measured |
|---------|----------------|
| Single-layer fwd+bwd | Naive vs Fused NCCL vs Fused SDMA total latency |
| Forward-only | Isolated forward with all fusion strategies |
| Backward-only | Isolated backward with dgrad+RS overlap + deferred wgrad |
| TP scaling | TP=2/4/8 latency, speedup, bandwidth at each scale |
| Bandwidth utilization | AG and RS effective bandwidth in GB/s |
