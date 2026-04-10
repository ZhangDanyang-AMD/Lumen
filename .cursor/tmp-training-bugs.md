# Temporary Training Bug Notes

This file lives at `.cursor/tmp-training-bugs.md` relative to the `Lumen` repo root. Read the whole file at the start of every new Lumen training debug session.

Use it to keep track of possible bugs found during testing. Do not treat any entry here as proof. Re-check against the current reference diff and current repro before acting.

Treat any fresh return to the same debugging problem as a new debug session:

- a new chat or agent session
- a new day or work block
- returning after unrelated work
- starting a new round of debug after prior tests finished

Write back only meaningful tests or experiments that change confidence in a hypothesis, such as a new repro, written diff, backend toggle, layerwise compare, kernel test, or targeted integration check. Do not log every identical rerun. Do log negative results that rule a suspicion out.

## Open

### [2026-04-10 fp8-training-alignment-repro]
- Goal: Reproduce VERL FP8 RL benchmarks using Lumen on MI350X, showing FP8 training aligns with BF16 when both use FP8 rollout.
- Reference: https://github.com/verl-project/verl/blob/main/docs/advance/fp8.md (Qwen3-8B-Base, Qwen3-30B-A3B-Base, Qwen3-30B-A3B)
- Setup: VERL 0.8.0.dev, vLLM 0.9.2rc2, 8x MI350X, DAPO recipe via main_ppo with DAPO overrides
- Environment check:
  - DAPO reward manager: YES
  - Decoupled clipping: YES (cliprange_low, cliprange_high)
  - Token-level loss: YES (loss_agg_mode=token-mean)
  - Rollout correction (TIS): YES (rollout_is=token, rollout_is_threshold=2.0)
  - Overlong reward buffer: YES (DAPORewardManager)
  - vLLM FP8 rollout: YES (quantization=fp8)
  - Qwen3 + Qwen3MoE support: YES (vLLM + HF transformers 4.57.1)
  - Dynamic sampling (filter_groups): config exists, NOT in RayPPOTrainer loop (both BF16/FP8 skip it, comparison remains fair)
- Models downloaded: qwen3-8b-base (16GB), qwen3-30b-a3b-base (57GB), qwen3-30b-a3b (57GB) to /dev/shm/model/
- Data downloaded: dapo-math-17k.parquet (286MB), aime-2024.parquet (29KB) to Lumen/data/
- Scripts: Lumen/examples/rl/verl/dapo/ (7 experiment scripts + common.sh + smoke_test.sh)
- Experiment structure (restructured): 
  - Exp 1 (8B dense): 1A BF16, 1B FP8 rollout+TIS, 1C FP8 rollout no TIS, 1D FP8 E2E (Lumen FP8PM)
  - Exp 2 (30B MoE unified): 2A BF16+TIS (shared baseline), 2B FP8 rollout+TIS, 2C FP8 E2E (Lumen FP8PM)
  - Old exp 3 (separate 30B baseline) eliminated — 2A serves as shared baseline for all 30B comparisons
- Risks: vLLM FP8 rollout untested on ROCm; Qwen3 MoE + FSDP2 untested; FP8PM + FSDP2 + MoE untested
- Next: Run smoke test (2-step BF16 with Qwen3-8B), then full experiments
- Status: open (setup complete, no runs yet)

### [2026-04-09 fp8pm-fsdp2-memory-regression] — FIXED
- Symptom: FSDP2+SGLang with FP8ParamManager uses MORE GPU memory than BF16 baseline when offloading is enabled (69.18 GB vs 48.06 GB peak).
- Root cause: `_FP8LinearFunc.forward` calls `ctx.save_for_backward(fp8_weight, scale)` which pins a reference to the allgathered parameter tensor. FSDP2 `param_offload` expects to reclaim allgathered memory after each module's forward, but the autograd reference prevents this. Result: ALL layers' allgathered FP8 weights accumulate on GPU until backward.
- Fix: Changed `ctx.save_for_backward(fp8_weight, scale)` to `ctx.save_for_backward(fp8_weight.clone(), scale)` in `lumen/quantize/fp8_params.py`. The clone creates an independent copy (FP8, 1 byte/elem, ~0.5 GB total for Qwen 0.5B) so FSDP2 can free the allgathered buffer after each layer's forward.
- Evidence (all Qwen 0.5B, 4 GPU, FSDP2+SGLang):
  | Config | Offload | Peak VRAM (max GPU) | vs BF16 (same offload) |
  |---|---|---|---|
  | BF16 | Yes | 48.06 GB | baseline |
  | FP8PM (before fix) | Yes | 69.18 GB | **+44% REGRESSION** |
  | **FP8PM (after clone fix)** | **Yes** | **45.50 GB** | **-5% SAVINGS** |
  | BF16 | No | 73.49 GB | baseline |
  | FP8PM | No | 54.87 GB | **-25% SAVINGS** |
- Note: `USE_8BIT_ADAM=1` env var was set in previous tests but did NOT actually change the optimizer — VERL's `build_optimizer` uses its own config (`FSDPOptimizerConfig`), not the env var. All tests used standard `torch.optim.AdamW`. bitsandbytes `AdamW8bit` is incompatible with FSDP2 DTensor.
- Throughput: FP8PM no-offload: 913 tok/s; FP8PM offload (clone fix): 942 tok/s.
- Status: fixed (clone fix in `_FP8LinearFunc.forward`)

### [2026-04-09 megatron-fp8-lora-incompatible]
- Symptom: Lumen FP8ParamManager and LoRA (PEFT) cannot be applied to Megatron-Core models.
- Root cause: Megatron-Core uses `ColumnParallelLinear`/`RowParallelLinear` which inherit from `nn.Module` directly, NOT from `nn.Linear`. Lumen's `FP8ParamManager.quantize_params()` only targets `nn.Linear` modules. Similarly, PEFT's LoRA expects HuggingFace model structure.
- FP8PM fix (2026-04-09 session 4):
  1. Extended `FP8ParamManager` to discover and target Megatron's `ColumnParallelLinear`/`RowParallelLinear` via `_get_quantizable_types()` classmethod.
  2. Created `_FP8MegatronLinearFunc` autograd function that quantizes BF16 weights to FP8 **on-the-fly** during forward (not in-place). Saves [input, FP8_weight, scale] for backward instead of [input, BF16_weight], halving weight portion of autograd graph.
  3. Key design: Megatron params stay BF16 (NOT quantized in-place) to preserve compatibility with Megatron's distributed optimizer and DDP. In-place FP8 quantization broke the distributed optimizer (caused `Failed to unpickle serialized exception` crash).
  4. Injection point: `verl/workers/engine/megatron/transformer_impl.py:_build_megatron_module()` — NOT `megatron_workers.py` (VERL 0.8.0 uses engine-based architecture).
  5. Gate: `FP8_PARAM_MANAGER=1` env var + `not self.engine_config.forward_only` (skip ref model).
- Evidence:
  - **Megatron+SGLang FP8PM (Qwen 0.5B, 4 GPU, TP=2, offload=true): PASSED** (2/2 steps, exit code 0). Step 2 throughput: 369 tok/s. Peak VRAM: 50.06 GB (-29% vs BF16 70.52 GB).
  - **Megatron+vLLM FP8PM (Qwen 0.5B, 4 GPU, TP=2, offload=true, rollout TP=1, gpu_util=0.3): PASSED** (2/2 steps, exit code 0). Step 2 throughput: 291 tok/s (vs BF16 338 tok/s — -14% regression). Peak VRAM: 86.89 GB (vs BF16 85.01 GB — within noise, dominated by vLLM KV cache).
  - All 4 Ray workers show `[LUMEN] FP8PM applied to Megatron module` in both SGLang and vLLM tests.
  - v2 (in-place quantization) crashed distributed optimizer. v3 (on-the-fly) works.
  - Megatron+vLLM with rollout TP=2 hangs during weight update (confirmed: vLLM TP>=2 still broken on ROCm). Must use TP=1.
- Known issue: on-the-fly quantization causes throughput regression (369 vs 704 tok/s for SGLang, 291 vs 338 tok/s for vLLM = -14%). Memory savings are significant (-29% with SGLang), negligible with vLLM (vLLM KV cache dominates).
- LoRA with Megatron: RESOLVED. Implemented `MegatronLoraAdapter` (`lumen/models/lora_adapter.py`) — a custom TP-aware LoRA wrapper for `ColumnParallelLinear`/`RowParallelLinear`. Injected via `post_model_creation_callbacks` in `make_megatron_module` (before DDP wrapping). Env var: `LORA_RANK=32`. Bypasses PEFT entirely; uses direct parameter injection with correct TP all-reduce for `RowParallelLinear`.
- Status: FP8PM resolved (memory savings confirmed); LoRA resolved (custom adapter); throughput regression noted as known tradeoff

### [2026-04-08 verl-megatron-nccl-hip-init]
- Symptom: VERL Megatron + SGLang test fails with `ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!` in Ray worker processes
- Possible bug: ROCm 6.2 runtime in container cannot initialize HIP GPUs (`hipGetDeviceCount` returns 0, error 100 = `hipErrorNoDevice`) despite host driver 6.18.1 seeing 8 MI300X GPUs. `torch.cuda.is_available()` returns False, `torch.cuda.device_count()` returns 8 (via KFD/DRM).
- Evidence so far:
  1. FSDP2 + SGLang works (both BF16 and FP8 Linear) — FSDP workers use the same `init_process_group` call but somehow succeed
  2. Megatron workers fail at `ProcessGroupNCCL()` constructor which checks GPU availability more strictly
  3. `rocm-smi` sees 8 GPUs, `/dev/kfd` and `/dev/dri/renderD*` exist with world-readable permissions
  4. Direct `ctypes.CDLL("libamdhip64.so").hipGetDeviceCount` returns 0 devices
  5. Patching `verl.utils.device.get_device_name()` to return "cuda" fixes the backend string but doesn't fix the underlying NCCL GPU detection
  6. **New container (`rocm/sgl-dev:v0.5.9-rocm700`) with ROCm 7.0 and PyTorch 2.9.0**: torch.cuda.is_available()=True, torch.cuda.device_count()=8, NCCL works for torchrun (tested BF16, LoRA, FP8PM+LoRA)
- Additional fixes applied to VERL 0.7.1 (in old `lumen_unit_test` container):
  - `verl/utils/fsdp_utils.py`: Added `DTensorSpec = None` fallback for torch < 2.6
  - `verl/checkpoint_engine/__init__.py`: Wrapped `hccl_checkpoint_engine` import in try/except
  - `aiter/ops/triton/gluon/pa_decode_gluon.py`: Added `hasattr(gluon, "jit")` check
- Resolution: Root cause was ROCm 6.2 vs host driver 6.18.1 mismatch. Fixed by using `rocm/sgl-dev` image with ROCm 7.0.
- Status: resolved (in new container)

### [2026-04-08 verl-vllm-rocm-import-crash]
- Symptom: `vllm-rocm 0.6.3` import crashes with `RuntimeError: No HIP GPUs are available` from `w8a8_utils.py:11` where `TORCH_DEVICE_IDENTITY = torch.ones(1).cuda() if is_hip() else None` runs at module level
- Possible bug: Same HIP init issue as above — module-level CUDA tensor creation fails when HIP runtime can't find GPUs
- Evidence: Uninstalling vllm-rocm fixes all other imports; the package poisons any import chain that touches `verl.checkpoint_engine.hccl_checkpoint_engine` → `vllm.distributed.utils`
- Resolution: Resolved by using new container with ROCm 7.0. The `rocm/sgl-dev` image includes vLLM 0.9.2rc2 built for ROCm 7.0 which works correctly.
- Status: resolved (in new container)

### [2026-04-08 verl-sglang-hybrid-server-crash]
- Symptom: VERL FSDP2+SGLang full pipeline fails in new container. SGLang HTTP server actors die during initialization.
- Root causes found and fixed:
  1. `torch_memory_saver` package (pip) links to `libcuda.so.1`/`libcudart.so.12` which don't exist on ROCm → uninstalled, SGLang falls back to noop
  2. VERL hardcodes `attention_backend: "fa3"` for SGLang but FA3 not available in ROCm build → patched to `"triton"`
  3. SGLang `init_memory_pool` fails with `Not enough memory` even with 270 GB per GPU → indicates deeper incompatibility between VERL 0.7.1 and SGLang 0.5.9-dev in hybrid mode
- Evidence: torchrun with direct FSDP2/DDP training works (BF16, LoRA, FP8PM+LoRA all complete successfully with 8 GPUs), proving the GPU/NCCL stack is healthy. Only the VERL+SGLang hybrid rollout integration fails.
- Update (2026-04-09): Upgraded VERL to 0.8.0.dev. With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `param_offload=true`, `optimizer_offload=true`, 4 GPUs, and Qwen 0.5B model, **FSDP2+SGLang pipeline completes successfully** (2/2 training steps). The 8B model still OOMs because another container uses ~53% of GPU memory (143 GB / 270 GB per MI300X).
- Resolution: FSDP2+SGLang works when GPU memory is not contended. Key settings: `expandable_segments=True`, offload actor params/optimizer to CPU, low `gpu_memory_utilization` for SGLang.
- Status: resolved (with memory constraint workaround)

### [2026-04-09 verl-vllm-hang-rocm]
- Symptom: Both FSDP2+vLLM and Megatron+vLLM hang indefinitely during weight updates. Two distinct hang modes:
  1. **TP >= 2**: vLLM V1 engine initialization hangs after "Initializing a V1 LLM engine...". The `spawn` multiprocessing method (required in Ray actors) fails to properly initialize child TP workers on ROCm.
  2. **TP = 1**: V1 engine initializes OK, reaches "Training from scratch", then hangs at `update_weights`. The `collective_rpc("update_weights_from_ipc")` call never completes — the utility RPC goes through AsyncMPClient → input_socket → engine core process → executor → workers, and gets stuck in this pipeline.
- Root cause: vLLM V1's architecture uses multi-process IPC (`AsyncMPClient` with ZMQ sockets to engine core subprocess). On ROCm:
  - `CuMemAllocator` is unavailable (NVIDIA-only), so `sleep()`/`wake_up()` are no-ops
  - The `collective_rpc` utility channel hangs, likely because the engine core process event loop is blocked or not processing utility messages during weight updates
  - VERL 0.8.0 requires V1 API (`AsyncLLM`); setting `VLLM_USE_V1=0` causes `ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False`
- Evidence:
  1. V1 TP=2: Hangs >4 min at engine init with `VLLM_USE_V1=1`. No error, just silence.
  2. V1 TP=1: Passes init, prints "Training from scratch", then hangs >2 min at `update_weights`.
  3. V0 attempt (`VLLM_USE_V1=0`): Immediately fails — VERL 0.8.0 uses `AsyncLLM.from_vllm_config` which enforces V1.
  4. `free_cache_engine=false` does not help (bypasses sleep/wake_up but hang persists).
  5. Qwen 0.5B model with offloading, 4 GPUs, low gpu_memory_utilization=0.1.
  6. SGLang rollout works perfectly with identical settings (proves actor/weights are OK).
- Patches applied (not sufficient):
  - `vllm/v1/worker/gpu_worker.py`: `sleep()` and `wake_up()` skip if `cumem_available` is false
  - `verl/workers/rollout/vllm_rollout/vllm_async_server.py`: `enable_sleep_mode` disabled on ROCm
  - `vllm/platforms/rocm.py`: Added `get_device_uuid` method
- Update (2026-04-09 session 2): Retested both Megatron+vLLM and FSDP2+vLLM with clean code (no debug logging errors). Findings:
  1. **Megatron+vLLM (TP=1)**: Only 1 of 4 vLLM V1 engines initialized successfully. The others hung at engine core process spawn (Gloo init stuck). No training progress made. Killed after 300s timeout.
  2. **FSDP2+vLLM (TP=1)**: Only 1 of 4 V1 engines initialized. "Training from scratch" appeared but generation hung — presumably because not all rollout servers were ready. Killed after 240s timeout.
  3. The collective_rpc path DOES work when the engine initializes — a debug logging test showed the worker function `update_weights_from_ipc` was called successfully. The previous "hang at update_weights" was likely caused by the engine not initializing properly (silent failure).
  4. Root cause refined: vLLM V1's `spawn` multiprocessing (required in Ray actors) has a race condition on ROCm where most spawned engine core processes fail to initialize (Gloo process group init hangs). Only ~25% of engines succeed randomly.
- Resolution: **No workaround found.** vLLM V1's spawned engine core processes cannot reliably initialize on ROCm inside Ray actors. Would require upstream vLLM fixes for ROCm multiprocessing or an in-process engine core mode.
- **Root cause found (2026-04-09 session 3)**: Missing `@with_amdsmi_context` decorator on vLLM ROCm platform `get_device_uuid()` method (`vllm/platforms/rocm.py:313`). Without this decorator, `amdsmi_init()` is never called before `amdsmi_get_gpu_asic_info()`, so in VERL worker processes (where no other amdsmi function has run yet), the call fails silently and falls back to a generic UUID (`rocm-gpu-0`). Meanwhile, in vLLM EngineCore subprocesses (where amdsmi was previously initialized by other platform functions), the real hardware UUID is returned (`0xDE72E6A9A0230550`). This creates mismatched ZMQ IPC socket paths between the weight sender (VERL worker) and receiver (EngineCore), causing an indefinite deadlock during `update_weights_from_ipc`.
- Fix: Add `@with_amdsmi_context` decorator to `RocmPlatform.get_device_uuid()` in `vllm/platforms/rocm.py`. One-line change.
- Verified results:
  | Config | GPUs | Result | Throughput |
  |---|---|---|---|
  | FSDP2+vLLM TP=1 | 1 | PASSED (2/2 steps) | 1537 tok/s |
  | FSDP2+vLLM TP=1 | 4 | PASSED (2/2 steps) | 1813 tok/s |
  | Megatron+vLLM TP=1 (actor TP=2) | 4 | PASSED (2/2 steps) | 1020 tok/s |
- The TP>=2 init hang (multiple EngineCore processes per vLLM server) is a separate issue from this UUID mismatch and may still exist. All tests used rollout TP=1.
- Status: resolved (for rollout TP=1; rollout TP>=2 untested after fix)

### [2026-04-09 verl-megatron-rocm-integration]
- Symptom: Megatron-Core integration on ROCm requires extensive patches for TransformerEngine absence.
- Issues and fixes applied:
  1. **TransformerEngine not available**: Patched `mbridge/core/llm_bridge.py` to set `use_transformer_engine=False` when `transformer_impl=local`
  2. **FusedLayerNorm vs RMSNorm**: Patched `megatron/core/models/gpt/gpt_layer_specs.py` to use `WrappedTorchNorm` instead of `FusedLayerNorm` for local impl
  3. **Sequence parallelism**: Disabled via override config (`sequence_parallel=false`) since `WrappedTorchNorm` doesn't support it
  4. **Weight name mapping (LLaMA)**: Patched `mbridge/core/bridge.py` and `mbridge/models/llama.py` to add `input_layernorm.weight` and `pre_mlp_layernorm.weight` mappings for local (non-TE) parameter names
  5. **Weight name mapping (Qwen2)**: Patched `mbridge/models/qwen2.py` to add `pre_mlp_layernorm.weight` mapping
  6. **Attention mask shape [FIXED]**: Megatron-Core's local `FusedSoftmax` expects 4D attention mask `[b,1,s,s]` but VERL passes 2D `[b,s]`. Patched both `model_forward_gen` (inference path) and `gptmodel_forward_model_engine` (training engine path) in `verl/models/mcore/model_forward.py` to convert 2D padding mask to 4D causal mask: `~(causal & padding)` where causal = lower-triangular, padding = `mask.unsqueeze(1).unsqueeze(2)`.
  7. **TE meta package**: Uninstalled broken `transformer-engine` meta package that caused import errors
- Evidence:
  - **Megatron+SGLang with Qwen 0.5B: PASSED** (2/2 training steps, 4 GPUs, TP=2, param_offload=true, optimizer_offload=true). Step 1 throughput: 167.2 tokens/s, Step 2 throughput: 402.4 tokens/s. Exit code 0.
  - Megatron+vLLM: blocked by vLLM V1 hang on ROCm (see `verl-vllm-hang-rocm`)
- Resolution: Megatron+SGLang works after all patches. Megatron+vLLM blocked on upstream vLLM V1 ROCm support.
- Comparison (Qwen 0.5B, 4 GPU, param_offload=true):
  - FSDP2+SGLang BF16: Peak 48.06 GB, 797 tok/s
  - Megatron+SGLang BF16: Peak 70.52 GB, 704 tok/s
  - FSDP2 uses 32% less memory and is 13% faster than Megatron for the same model.
- Status: resolved (for SGLang); blocked (for vLLM)

## Ruled Out

Move disproved suspicions here instead of deleting them.

## Resolved

### [2026-04-08 fp8pm-dequant-memory-leak]
- Symptom: FP8PM + LoRA (210.4 GB) used **more** memory than BF16 + LoRA (142.9 GB) on 70B model
- Root cause: Two compounding issues in `FP8ParamManager._wrap_forward_to_use_dequant`:
  1. The old `_make_dequant_hook` created a full BF16 dequantized copy per `nn.Linear` module and stored it as `module._dequantized_weight`. These were cleaned up after each forward, but…
  2. `F.linear(input, dequant_weight)` caused PyTorch autograd to **save the BF16 dequant_weight** for backward (needed to compute `grad_input = grad_output @ weight`). This meant all 225 layers' BF16 copies (~140 GB for 70B) accumulated until backward ran.
  3. Additionally, PyTorch autograd saved the `input` tensor for each linear's backward, adding further overhead.
- Fix: Replaced `F.linear` with custom `_FP8LinearFunc(torch.autograd.Function)`:
  - `save_for_backward` stores only FP8 weight (1 byte/elem) + scalar scale — not the full BF16 weight
  - Does not save `input` (not needed since FP8 weights are frozen, no grad_weight computed)
  - Reconstructs BF16 weight from FP8 on-the-fly during backward
  - Removed redundant pre/post hooks (`_make_dequant_hook`, `_make_cleanup_hook`)
- Evidence (70B, 8-GPU DDP, 3 steps):
  | Config | Before Fix | After Fix |
  |--------|-----------|-----------|
  | BF16 + LoRA r=16 | 142.73 GB | 142.73 GB |
  | FP8PM + LoRA r=16 + 8-bit Adam | 210.43 GB | **73.48 GB** |
  Memory saved: 137 GB (65% reduction). FP8PM+LoRA now **49% better** than BF16+LoRA.
- Evidence (8B, single GPU, 3 steps):
  | Config | Before Fix | After Fix |
  |--------|-----------|-----------|
  | BF16 + LoRA r=16 | 17.07 GB | 17.07 GB |
  | FP8PM + LoRA r=16 + 8-bit Adam | 27.22 GB | **13.31 GB** |
- Status: resolved

### [2026-04-08 fp8pm-lora-compat-fix]
- Symptom: FP8ParamManager + LoRA crashed with `NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'`
- Root cause: PEFT's `_move_adapter_to_device_of_base_layer` casts LoRA adapter weights (`lora_A`, `lora_B`) to match base layer dtype. Since FP8ParamManager converts base weights to FP8, LoRA adapters were also cast to FP8.
- Fix: Added post-PEFT fixup in `LumenConfig._apply_lora()` to re-cast any LoRA params with FP8 dtype back to BF16. Also fixed `LumenConfig.from_args()` to respect `linear_fp8=False` by setting `scaling="none"`.
- Status: resolved

### [2026-04-08 fp8pm-multi-gpu-integration]
- Symptom: FP8ParamManager works on single GPU but needs multi-GPU support.
- Fixes applied:
  1. FSDP1/FSDP2 incompatible — switched to DDP (MULTI_GPU)
  2. Device mismatch in dequant hook: scale tensor stayed on CPU after model.cuda(); fixed by using `weight.data` and `scale.to(device)`
  3. DDP unused-parameter error: set `requires_grad=False` on FP8 params
- Results (8-GPU DDP, Llama-3.1-8B, 20 steps):
  | Config | Peak Mem/GPU | vs BF16 DDP |
  |--------|-------------|-------------|
  | BF16 DDP | 80.5 GB | baseline |
  | FP8ParamManager DDP | 31.4 GB | **-61%** |
  | FP8PM + 8-bit Adam | 30.4 GB | **-62%** |
- 70B result: FP8PM + 8-bit Adam = 210.8 GB (fits MI300X 256 GB), BF16 DDP = OOM
- Limitation: FP8ParamManager freezes nn.Linear weights (only ~1.3% params trainable)
- Status: resolved

### [2026-04-08 fp8-memory-savings-experiment]
- Symptom: Need to prove FP8 training saves memory. Previous runs on wrong branch showed no savings.
- Reference: BF16 baseline single-GPU Llama-3.1-8B, AdamW, gradient checkpointing, SDPA, bs=2, seq=256.
- Fix: Synced dev/RL to Docker, fixed FP8ParamManager to skip nn.Embedding (was crashing on bias attr), synced descriptor.py.
- Results (Llama-3.1-8B, 3 steps, single MI300X GPU, **process-isolated** — each config in its own python process starting from 0 MB):
  | Config                 | Peak Alloc (MB) | Peak Res (MB) | Steady-State (MB) | vs BF16   |
  |------------------------|-----------------|---------------|-------------------|-----------|
  | BF16 baseline (AdamW)  | 76,861          | 78,728        | 46,228            | baseline  |
  | FP8ParamManager        | 28,909          | 32,178        | 24,757            | **-62.4%** peak, **-46.5%** steady |
  | FP8Param + 8-bit Adam  | 27,923          | 30,172        | 23,769            | **-63.7%** peak, **-48.6%** steady |
  | FP8 Attention (dpa)    | 76,860          | 77,414        | 46,227            | -0.0% peak |
- Detailed memory breakdown (process-isolated):
  | Component              | BF16 Baseline | FP8ParamManager | FP8+8bit Adam | FP8 Attn  |
  |------------------------|---------------|-----------------|---------------|-----------|
  | Param storage          | 15,317 MB     | 8,160 MB        | 8,160 MB      | 15,317 MB |
  |   FP8 params           | 0 MB          | 7,157 MB        | 7,157 MB      | 0 MB      |
  |   BF16 params          | 15,317 MB     | 1,003 MB        | 1,003 MB      | 15,317 MB |
  | Optimizer states       | 30,633 MB     | 2,005 MB        | 1,018 MB      | 30,633 MB |
  |   state dtype          | bf16          | bf16            | uint8+fp32    | bf16      |
- Key findings:
  1. FP8ParamManager saves ~48 GB peak (62.4%). Savings come from BOTH weights AND optimizer states:
     - Weights: bf16->fp8 saves ~7.2 GB (15,317 -> 8,160 MB)
     - Optimizer states: AdamW creates exp_avg/exp_avg_sq matching param dtype. FP8 params get bf16 states sized to numel() but the numel() is the same — the saving is from dtype: the optimizer internally matches the param dtype (bf16 for the few remaining bf16 params only). The FP8 params have 1-byte elements so AdamW allocates bf16 states with the same numel but for FP8 params the optimizer states are much smaller because the parameter tensor's element_size is 1 byte.
     - **Correction**: AdamW allocates states based on param shape, but uses bf16 (not fp32) for all states in this PyTorch version. For FP8 params, AdamW only tracks non-FP8 params (lm_head, embedding, norms). The 225 quantized Linear params are FP8 and their gradients still flow, but optimizer states are allocated as bf16 tensors matching param numel — total only 2,005 MB vs 30,633 MB.
  2. Adding 8-bit Adam saves another ~1 GB (optimizer states 2,005->1,018 MB as uint8).
  3. FP8 Attention alone saves negligible peak memory with gradient checkpointing enabled.
- Status: resolved

| Config | Peak Mem/GPU | Elapsed | vs BF16 full |
|--------|-------------|---------|-------------|
| BF16 full (baseline) | 34.57 GB | 122.7s | baseline |
| FP8 Linear only | 38.85 GB | 1279.2s | +12% mem, 10.4x slower |
| FP8 Linear + FP8 Attn (dpa) | **30.92 GB** | 1586.5s | **-11% mem**, 12.9x slower |
| FP8 Linear + FP8 Attn + Act Store | **30.89 GB** | 1581.3s | **-11% mem**, 12.9x slower |
| FP8 Linear + FP8 Attn + Param Gather | CRASH | — | AITER quant kernel crash |
| FP8 Linear + FP8 Attn + Lumen Norm | CRASH | — | FSDP1 mixed dtype flatten error |

### [2026-04-08 fp8-architectural-fixes]
- Symptom: Three FP8 features marked "NOT FEASIBLE" in BENCHMARK_RESULTS.md
- Fix 1 (FP8 Weight Cache): Wired store_weights_fp8() into LumenConfig + VERL. Result: actor -2%, throughput +3.3% vs FP8-only.
- Fix 2 (FP8 Activation Store): Extended _apply_pre_quant to nn.Linear. Result: no measurable effect on full FP8 path.
- Fix 3 (AITER Kernel Crash): Added weight.contiguous() + TORCH_CHECK. Result: crash resolved, actor -1.7%, throughput +1.4%.
- Combined: actor -2.4%, throughput +2.8% vs FP8-only.
- Status: resolved

## Entry Template

```markdown
### [YYYY-MM-DD session-name]
- Symptom:
- Possible bug:
- Evidence so far:
- Next check:
- Status: open | ruled out | resolved
```
