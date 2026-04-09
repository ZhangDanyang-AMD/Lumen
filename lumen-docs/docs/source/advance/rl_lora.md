# RL Training with LoRA and FP8

Last updated: 04/09/2026.

Lumen supports **LoRA** (Low-Rank Adaptation) combined with **FP8 quantized training** for reinforcement learning algorithms such as GRPO, PPO, and DAPO. This combination enables RL post-training of very large models (70B+) on modest hardware by stacking two orthogonal memory savings:

- **FP8 base model weights** — ~50% parameter memory reduction via `quant.enable(model)`
- **LoRA adapters** — only low-rank A/B matrices are trainable, drastically reducing optimizer state

The benefits this brings include:

- Reinforcement learning with very large models (e.g. 70B+) on 8× MI300X GPUs
- Larger batch sizes due to reduced memory usage
- Simplified deployment — only LoRA adapters need to be saved and distributed
- Composable with all Lumen FP8 features: FP8 all-gather, FP8 attention, FP8 activation store

This guide explains how to enable LoRA in RL training and configure related parameters across both TRL and VERL backends.

## Usage Guide

### TRL Backend (FSDP1/FSDP2)

1. LoRA is available through Lumen's TRL integration in `lumen.rl.trl.runner`. The high-level entry point is `run_grpo()`.

2. LoRA is applied via HuggingFace PEFT. The runner handles LoRA initialization automatically when `lora_r > 0`.

3. Required configurations for LoRA:

   - `lora_r`: int, set to a value greater than 0 (e.g., 16, 32, 64)
   - `lora_alpha`: float, the scaling factor (effective LR multiplier = alpha / r)
   - `quant_format`: the FP8 format for the frozen base weights (e.g., `"fp8_e4m3"`)

4. Example:

   ```python
   from lumen.rl.trl.runner import run_grpo

   run_grpo(
       model_name="meta-llama/Llama-2-70b-hf",
       quant_format="fp8_e4m3",
       scaling="delayed",
       fsdp_strategy="FULL_SHARD",
       lora_r=32,
       lora_alpha=32,
   )
   ```

### VERL Backend (FSDP2 + SGLang/vLLM)

1. LoRA is available through VERL's `RayPPOTrainer` with the FSDP2 backend. When Lumen FP8 features are enabled, VERL uses `lumen.rl.verl.verl_entry` as the entry point.

2. Required configurations for LoRA:

   - `actor_rollout_ref.model.lora_rank`: int, set to a reasonable value (e.g., 32, 64, 128)
   - `actor_rollout_ref.model.lora_alpha`: float, the alpha term in LoRA
   - `actor_rollout_ref.model.target_modules`: the target modules for LoRA, typically `"all-linear"`

3. Recommended options:

   - `LUMEN_FP8=1`: enable FP8 quantized linear for the actor model
   - `FP8_PARAM_MANAGER=1`: enable FP8 parameter manager for ~62% peak memory reduction
   - `actor_rollout_ref.ref.fsdp_config.param_offload=true`: offload reference model to CPU

4. Example:

   ```bash
   LUMEN_FP8=1 FP8_PARAM_MANAGER=1 \
   bash examples/rl/verl/run_grpo_fsdp2.sh
   ```

## Best Practices and Notes

1. **Learning rate**: When using LoRA, it is recommended to increase the learning rate by an order of magnitude compared to full fine-tuning. For example, use `3e-5` instead of `5e-7`.

2. **LoRA Rank**:

   - Too small a rank can hurt convergence, especially for RL where reward signals are noisy.
   - Recommendations:
     - For models ≤8B: `lora_rank >= 16` is sufficient
     - For models 8B–32B: `lora_rank >= 32` recommended
     - For models 70B+: `lora_rank >= 64` recommended for RL convergence parity with full fine-tuning

3. **FP8 + LoRA interaction**: The frozen base weights are quantized to FP8 for forward/backward, while LoRA adapter weights (A and B matrices) remain in BF16 for numerical stability. This is handled automatically by `quant.enable()`.

4. **Gradient checkpointing**: Always enable gradient checkpointing for RL with large models. It is critical for fitting 70B+ models in memory.

5. **Reference model**: For PPO/GRPO, the reference model should use `param_offload=true` to reduce GPU memory. With LoRA, the reference model shares the same frozen base weights and only differs in the absence of LoRA adapters.

## Configuration Reference

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `lora_r` / `lora_rank` | Rank of the low-rank matrices | 16, 32, 64, 128 |
| `lora_alpha` | Scaling factor (effective LR multiplier = alpha / r) | Equal to rank, or 2× rank |
| `lora_target_modules` / `target_modules` | Which layers to apply LoRA to | `"all-linear"` or `"qkv_proj,o_proj,gate_proj,up_proj,down_proj"` |
| `lora_dropout` | Dropout on LoRA paths | 0.0 (default for RL) |
| `quant_format` | FP8 format for frozen base weights | `"fp8_e4m3"`, `"fp8_e5m2"` |
| `scaling` | FP8 scaling strategy | `"delayed"`, `"dynamic"` |

## Memory Comparison

Peak memory per GPU for RL training (GRPO, 8× MI300X):

| Model | BF16 Full (TRL) | FP8 Full (TRL) | FP8 + LoRA r=32 (TRL) | FP8 + LoRA r=32 (VERL) |
|-------|-----------------|-----------------|------------------------|-------------------------|
| Llama-3.1-8B | 34.57 GB | ~18 GB | ~12 GB | 11.76 GB |
| Qwen2.5-32B | 124.18 GB | ~68 GB | ~42 GB | 38.43 GB |

## Example Scripts

### TRL + GRPO + LoRA (single-node, 8× MI300X)

```bash
cd examples/rl/trl/benchmark/llama-3.1-8b
bash run.sh R1
```

### VERL + GRPO + LoRA (FSDP2 + SGLang, 8× MI300X)

```bash
# BF16 baseline
bash examples/rl/verl/run_grpo_fsdp2.sh

# With Lumen FP8
LUMEN_FP8=1 bash examples/rl/verl/run_grpo_fsdp2.sh

# With FP8 + LoRA + FP8 param manager
LUMEN_FP8=1 FP8_PARAM_MANAGER=1 bash examples/rl/verl/run_grpo_fsdp2.sh
```

### VERL + Megatron + SGLang (multi-node, MoE/TP/PP)

```bash
bash examples/rl/verl/run_grpo_megatron_sglang.sh
```

## Supported Algorithms

| Algorithm | Framework | LoRA + FP8 Status |
|-----------|-----------|-------------------|
| GRPO | TRL | Validated |
| GRPO | VERL (FSDP2) | Validated |
| GRPO | VERL (Megatron) | Supported |
| PPO | TRL / VERL | Supported |
| DAPO | TRL | Supported |
