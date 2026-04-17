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
- Data: moved from Lumen/data/ to /dev/shm/data/ (ramdisk for I/O speed)
- Scripts: Lumen/examples/rl/verl/dapo/ (10 experiment scripts + common.sh + smoke_test.sh)
- Experiment structure: 
  - Exp 1 (8B dense): 1A BF16, 1B FP8 rollout+TIS, 1C FP8 rollout no TIS, 1D FP8 E2E (full Lumen stack)
  - Exp 2 (30B MoE Base): 2A BF16+TIS, 2B FP8 rollout+TIS, 2C FP8 E2E (full Lumen stack)
  - Exp 3 (30B MoE Instruct): 3A BF16+TIS, 3B FP8 rollout+TIS, 3C FP8 E2E (full Lumen stack)
  - Exp 2 uses Qwen3-30B-A3B-Base, Exp 3 uses Qwen3-30B-A3B (instruct) — independent confirmation on different model variants
- Code update (2026-04-10, commit 4c5edeb: merge main perf opts into dev/RL):
  - Analyzed new performance code from main. Key changes affecting E2E experiments:
  - FP8Descriptor dataclass with lazy transpose cache — transparent improvement
  - _safe_fp8_desc() NaN protection for zero-scale warmup steps — transparent fix
  - LUMEN_FUSED_QUANT_AMAX=1: fused quant+amax single Triton kernel (halves memory bandwidth for delayed scaling)
  - LUMEN_FUSED_CAST_TRANSPOSE=1: fused FP8 cast+transpose Triton kernel
  - LUMEN_FUSED_QUANT_SCALE=1: fused static per-tensor quant via AITER Triton
  - LUMEN_FP8=1 now wired through verl_entry.py → VerlLumenArgs → LumenConfig.enable() → replaces nn.Linear forward with FP8 GEMM
  - FP8PM + FP8 Linear work together: FP8PM pre-quantizes weights (FP8Descriptor), FP8 Linear detects via _fp8_desc and uses directly for GEMM (no double-quant)
  - Updated 1D/2C scripts: now use full Lumen stack (FP8PM + FP8 GEMM + Lumen Norm + fused kernels)
  - Updated common.sh: data paths → /dev/shm, auto-enable fused kernel flags when LUMEN_FP8=1
  - 2C additionally enables LUMEN_FP8_ACTIVATION_STORE=1 (FP8 MLP activation store for MoE memory)
- Risks: vLLM FP8 rollout untested on ROCm; Qwen3 MoE + FSDP2 untested; FP8PM + FP8 GEMM + FSDP2 untested; fused kernels untested on MI350
- Memory issues resolved:
  - GPU OOM during compute_log_prob: vLLM sleep() is no-op on ROCm → KV cache holds memory during FSDP ops
  - Fix: free_cache_engine=True + gpu_memory_utilization=0.3 + log_prob_micro_batch_size_per_gpu=1
  - Smoke test passed (2 steps, BSZ=16, n=2, MAX_RESPONSE=512, GPU_MEM=0.5)
  - Full pipeline OOM at GPU_MEM=0.9/0.5/0.4 with offload=true/false → only stable at GPU_MEM=0.3 + free_cache_engine=True
  - Hydra config fix: +data.gen_batch_size (not in schema, needs + prefix); rollout.quantization IS in schema (no + prefix)
- Exp 1A attempt 2: crashed at step 55/275 (2026-04-10 ~17:55 UTC)
  - Cause: GPU OOM during FSDP worker — HSA_STATUS_ERROR_OUT_OF_RESOURCES: Available Free mem: 0 MB
  - Root cause: response_length grew from ~850 (step 1) to ~2750 (step 54), global_seqlen/max hit 280K
  - Missing config vs reference: use_dynamic_bsz=True not enabled (reference uses it to auto-adjust micro batch sizes)
  - Fix: added use_dynamic_bsz=True, ppo_max_token_len_per_gpu=21504, log_prob_max_token_len_per_gpu=21504 to common.sh
  - This is a real config diff from reference, not a parameter tweak
- Exp 1A attempt 3: relaunched with dynamic batching (2026-04-11 ~02:35 UTC)
  - Config: GPU_MEM_UTIL=0.3, offload=true, free_cache_engine=True, log_prob_micro_bsz=1, gen_bsz=32, use_dynamic_bsz=True
  - **STEP 55 MILESTONE CLEARED** (2026-04-11 ~09:00 UTC)
  - Step 55/275: rewards -0.902→-0.049, val_acc 4.58%→20.42%
  - Dynamic batching fix VALIDATED — survived seqlen 369K (31% above 280K crash)
  - Attempt 3 crash at ~step 60: HOST OOM (kernel oom-kill), NOT GPU OOM
    - Root cause: /dev/shm checkpoints (255 GB) + FSDP offload filling host RAM
    - Fix: moved CKPTS_DIR from /dev/shm to /root/ckpts (disk), max_ckpt_to_keep=2
  - Attempt 4: resumed from step 40 checkpoint (2026-04-11 ~12:00 UTC)
    - Config: same as attempt 3 + disk-backed checkpoints at /root/ckpts/
    - Step 60 passed safely — disk checkpoint saved without host OOM
    - val_acc: 17.6% (s45) → 21.0% (s50) → 18.9% (s55) → 23.2% (s60)
    - rewards: -0.367 → -0.264 → -0.087 → -0.031
    - resp_len: 1800 → 3140 → 2258 → 3117, seqlen_max up to 335K
    - All previous crash points cleared (step 55 GPU OOM, step 60 host OOM)
  - Overall ETA: ~21h remaining for 275 steps (step 61/275)
- Exp 1A attempt 4 continued: reached step 192/275 (2026-04-12 ~14:52 UTC)
  - val_acc trajectory (every 5 steps from step 95 onward):
    23.3% → 23.3% → 22.0% → 23.9% → 23.1% → 24.8% → 22.8% → 24.3% → 25.8% → 25.1%
    → 25.3% → 25.6% → 25.4% → 25.8% → 25.1% → 25.9% → 25.6% → 27.1% → 25.7% → 26.8%
  - rewards at step 192: oscillating around 0.0 (range -0.2 to +0.1)
  - response_length at step 192: ~5000 tokens avg (up from ~3000 at step 60)
  - global_seqlen/max: mean=392K, peak=640K (step 189)
  - global_seqlen/balanced_max: mean=267K, peak=388K
  - Training was progressing well — val_acc improving, no divergence
- Attempt 4 crash at step ~193 (2026-04-12 14:59:36 UTC):
  - Cause: GPU OOM on vLLM rollout worker rank 3 (PID 735732)
  - Error: HSA_STATUS_ERROR_OUT_OF_RESOURCES: Available Free mem: 0 MB → SIGABRT → cascaded to kill all Ray workers
  - Last successful weight update: 14:52:28 on rank 3, then OOM ~7 min later during the next step's generation phase
  - Root cause: response lengths grew to ~5000 avg (from ~3000 at step 60), global_seqlen/max routinely >400K
  - Dynamic batching (ppo_max_token_len_per_gpu=21504) limits training micro-batch size but cannot limit total batch memory during vLLM generation phase
  - The 21504 value was the theoretical max (1024+20480) — too generous, leaves no memory headroom for growing response lengths
  - Fix: lowered ppo_max_token_len_per_gpu and log_prob_max_token_len_per_gpu from 21504 to 16384 in common.sh
    - This forces more, smaller micro-batches during training/log_prob, reducing peak GPU memory
    - Separated LOG_PROB_MAX_TOKEN_LEN from PPO_MAX_TOKEN_LEN for independent tuning
    - The generation phase (vLLM) is capped by gpu_memory_utilization=0.3 + free_cache_engine=True (unchanged)
  - Last checkpoint: global_step_180 (inside lumen_verl_test container at /root/ckpts/FP8-ALIGN/1A-qwen3-8b-bf16-rollout-bf16/)
- Attempt 5 (2026-04-12): assertion error with PPO_MAX_TOKEN_LEN=16384 — max_seq_len=20594 > max_token_len. Reverted to 21504.
- Attempt 6 (2026-04-13 ~02:34 UTC): resumed from step 180 with gpu_memory_utilization=0.25 (lowered from 0.3)
  - Steps 181-188 completed successfully. Metrics:
    step=181 rew=-0.123 rlen=5521 seqlen_max=472K
    step=182 rew=-0.185 rlen=5427 seqlen_max=488K
    step=183 rew=+0.014 rlen=4621 seqlen_max=448K
    step=184 rew=-0.216 rlen=5856 seqlen_max=470K
    step=185 rew=+0.066 rlen=4864 seqlen_max=427K val_acc=26.3%
    step=186 rew=+0.004 rlen=4554 seqlen_max=527K
    step=187 rew=+0.084 rlen=4865 seqlen_max=561K
    step=188 rew=+0.111 rlen=4302 seqlen_max=438K
  - Crash after step 188 (~04:10 UTC): GPU OOM on vLLM rollout worker PID 869078 (rank 0)
    - HSA_STATUS_ERROR_OUT_OF_RESOURCES → SIGABRT → cascaded kill of all Ray workers
    - Crash during generation phase (after weight update, during vLLM inference)
  - Analysis: gpu_memory_utilization=0.25 was WORSE than 0.3 — smaller KV cache forces more scheduling
    pressure when sequences are long. The OOM is from vLLM trying to allocate blocks for long sequences
    that don't fit in the reduced KV cache budget.
  - Root cause pattern: vLLM can schedule up to max_num_seqs=256 concurrent sequences. With n=16 and
    gen_batch_size=32, each generation step sends 512 prompts. When responses reach 5K+ tokens avg,
    the total KV cache needed for concurrent in-flight sequences exceeds the GPU memory budget.
  - Fix plan for attempt 7:
    1. Revert gpu_memory_utilization to 0.3 (attempt 4 survived to step 193 with this)
    2. Reduce max_num_seqs from 256 to 64 (limit concurrent sequences to reduce peak KV cache)
    3. Reduce gen_batch_size from 32 to 16 (fewer prompts per generation call, halving peak demand)
    4. These are ROCm-specific accommodations — VERL reference uses 0.9 on H100 where vLLM can
       reclaim memory via CuMemAllocator sleep/wake_up, which is a no-op on ROCm
  - Last checkpoint: global_step_180 (survives across attempts)
- VERL reference comparison at step 185:
  - val_acc=26.3% at step 185 — VERL BF16 reference shows ~25-30% at similar step count → ON TRACK
  - reward oscillating around 0 (±0.2) — matches VERL reference behavior for DAPO
  - response_length ~5K tokens — expected growth pattern (model learns longer reasoning chains)
  - No divergence or training instability detected; only infrastructure OOM issue
- Attempt 7 (2026-04-13 ~05:17 UTC): gen_batch_size=16 caused AssertionError: 32 % 64 != 0
  - ppo_mini_batch_size_per_gpu was computed as mini_batch_size // dp_size, but with gen_batch_size=16
    the PPO tensordict had batch_size=32 but mini_batch_size_per_gpu=64 (from wrong path). Reverted gen_bsz to 32.
- Attempt 8 (2026-04-13 ~05:17 UTC): gpu_memory_utilization=0.3, max_num_seqs=64, gen_batch_size=32
  - Config diff from attempt 4: only max_num_seqs reduced from 256 to 64
  - **STEP 193 MILESTONE CLEARED** — past both previous crash points (att 4 @ 193, att 6 @ 188)
  - Step 191/275 reached (2026-04-13 ~07:25 UTC), training stable
  - Metrics at step 190: val_acc=26.1%, reward=+0.082, resp_len=4724, seqlen_max=554K
  - Peak seqlen_max=614K at step 189 — survived (att 4 crashed at 640K)
  - Step time ~756s (~12.6 min) — slower than att 4 (~488s) due to max_num_seqs=64 limiting
    vLLM concurrent generation throughput. Trade-off: stability vs speed.
  - ETA: ~18h for remaining 84 steps
- VERL reference comparison at step 190:
  - val_acc=26.1% — VERL BF16 dark green curve reads ~25-28% at same step → ON TRACK
  - Accuracy plateau 22-27% since step 95, with 2-3% oscillation — matches VERL ref pattern
  - Reward oscillating ±0.25 around 0 — matches DAPO expected behavior
  - Response length ~5K tokens — expected CoT growth
- Attempt 8 crash at step ~249 (2026-04-13 ~18:17 UTC):
  - Cause: GPU OOM on vLLM rollout worker PID 946931
  - Error: HSA_STATUS_ERROR_OUT_OF_RESOURCES: Available Free mem: 0 MB → SIGABRT
  - Same pattern as attempts 4 and 6 — vLLM KV cache exhaustion during generation
  - Last logged step: 249 (seqlen_max=519K)
  - Last checkpoint: global_step_240
  - val_acc at step 245: 28.5%, step 240: 27.7%, step 235: 28.5%, step 230: 29.3%
  - Training metrics were healthy — only infrastructure OOM issue
  - This confirms the need for the ROCm sleep/wake fix to enable higher gpu_memory_utilization
- 1A-v2 attempt 1 crash (gpu_memory_utilization=0.85, gen_bsz=96, max_num_seqs=256):
  - Immediate OOM on step 0 — HSA_STATUS_ERROR_OUT_OF_RESOURCES on WorkerDict (FSDP training worker)
  - Root cause: gpu_memory_utilization=0.85 too aggressive; vLLM reserves 85% for KV cache but model weights
    also need GPU memory during generation. sleep/wake doesn't help if wake_up reloads weights into memory.
  - Also discovered: VERL calls wake_up before generation but never calls sleep after generation completes.
    The sleep/wake patches enable the mechanism but VERL doesn't use the sleep side.
    Need deeper VERL integration to call sleep between rollout→training phase transition.
- 1A-v2 attempt 2 (gpu_memory_utilization=0.5, gen_bsz=32, max_num_seqs=64):
  - Running, at step 8/275, ~3.75 min/step early on (sequences still short)
  - enable_sleep_mode=True on all workers (patches applied)
  - wake_up is being called but sleep is not (VERL integration gap)
  - Effectively testing gpu_memory_utilization=0.5 vs v1's 0.3 (67% more KV cache budget)
- Critical discovery (2026-04-14): VERL never calls release()/sleep() between rollout→training phase transition
  - fsdp_workers.py `generate_sequences()` calls `rollout_mode()` (wake_up) before generation, then
    `trainer_mode()` after generation — but `trainer_mode()` was NEVER DEFINED, it just doesn't error
    because the colocated async worker takes a different code path through `update_weights()`
  - `release()` method exists in vllm_rollout.py (calls sleep(level=...)) but is NEVER invoked
  - This means on ROCm: vLLM KV cache stays allocated during training even when sleep patches are applied
  - With gpu_memory_utilization=0.5: KV cache=126 GB, leaving 126 GB for training — works
  - With gpu_memory_utilization=0.9: KV cache=227 GB, leaving only 25 GB — OOM
  - On CUDA: CuMemAllocator transparently manages memory pools, so explicit release isn't needed
  - Fix: Added `trainer_mode()` method to `ActorRolloutRefWorker` that calls `self.rollout.release()`
    when `free_cache_engine=True`. This triggers `_rocm_sleep()` to offload weights + free KV cache.
  - Saved patched fsdp_workers.py to patches/verl_fsdp_workers.py
  - V3 script prepared: gpu_memory_utilization=0.9, max_num_seqs=256, with full sleep/wake cycle
  - Will launch after V2 completes or crashes
- V3 (gpu_memory_utilization=0.9, max_num_seqs=256, full ROCm sleep/wake):
  - Ran 79 steps (step 1-79) before OOM crash during compute_log_prob
  - Sleep/wake cycle worked flawlessly: 209 GiB freed per cycle
  - Val accuracy: 4.0% (s5) → 21.4% (s50) → 23.1% (s70) → 22.3% (s75)
  - Reward crossed zero at step 60 (+0.010)
  - Throughput: avg 403 tok/s, peak 645 tok/s
  - Gen latency: 0.17-0.67 ms/tok (avg 0.34)
  - Actor update: stable at ~0.021 ms/tok
  - Crash at step 79: OOM during compute_log_prob (training phase, not rollout)
    - Response lengths had grown to ~3000-3868 tokens avg
    - KV cache was correctly freed by sleep() — the OOM is from FSDP training activations
    - compute_log_prob requires full sequence padded to max_seqlen in the batch
    - global_seqlen/max at crash: ~419K tokens → ~419K × 8 (hidden) × 2 (bf16) ≈ large activations
  - Comparison to V1: V1 ran to step 249 with gpu_mem=0.3 but WITHOUT sleep/wake
    - V1 never needed sleep/wake because gpu_mem=0.3 only allocated ~75 GiB KV cache,
      leaving ~177 GiB for training. V1's OOM was during vLLM GENERATION (KV cache too small for long seqs).
    - V3 uses gpu_mem=0.9 → vLLM allocates ~209 GiB KV cache during rollout, then sleep()
      frees it, leaving ~252 GiB for training. But compute_log_prob at seqlen_max=419K with
      use_dynamic_bsz still hit OOM.
    - Key insight: V3's training OOM is a DIFFERENT failure mode from V1's rollout OOM.
      V1 died during generation (KV cache too small). V3 dies during training (activations too large).
  - Comparison to VERL reference (8×H100):
    - H100 has CuMemAllocator which does fine-grained memory pool management (not just sleep/wake)
    - CuMemAllocator can dynamically reclaim unused memory WITHIN the training phase
    - Our ROCm sleep/wake is all-or-nothing: sleep frees everything, wake restores everything
    - Between sleep and wake, there's no dynamic memory management — PyTorch default allocator
    - H100 also has 80 GiB HBM vs MI350X 288 GiB HBM — but they use FEWER GPUs of H100 memory
      per weight, so the ratio of available memory to model size may be similar
    - VERL ref uses gen_batch_size=32×3=96 (3 gen batches per train batch) vs our gen_batch_size=32
    - VERL ref uses train_batch_size=32 same as ours
    - VERL ref ran 500 steps to ~35%+ accuracy without crashing — suggests their memory management
      handles the growing response lengths better
  - Root cause analysis for V3 OOM:
    1. compute_log_prob pads all sequences to max_seqlen in the micro-batch
    2. With use_dynamic_bsz=True, the micro-batch token budget is PPO_MAX_TOKEN_LEN=21504
    3. If one sequence is 20K tokens, the entire micro-batch is just 1 sequence but padded to 20K
    4. The forward pass through the full model at 20K sequence length requires substantial activation memory
    5. On H100 with CuMemAllocator, PyTorch can borrow from vLLM's freed memory pool seamlessly
    6. On ROCm, even after sleep() frees 209 GiB, the PyTorch allocator may fragment the space
  - Possible fixes for V4:
    1. Lower ppo_max_token_len_per_gpu (e.g., 16384) to force smaller micro-batches
    2. Lower max_response_length (e.g., 16384 instead of 20480) to cap sequence growth
    3. Use gradient accumulation with smaller effective batch sizes
    4. Use gpu_memory_utilization=0.85 instead of 0.9 to leave more headroom
- V4 (gpu_memory_utilization=0.85, ROCm sleep/wake, dynamic sampling filter_groups):
  - Config: gen_batch_size=96, train_batch_size=32, max_num_seqs=256
  - Dynamic sampling: filter_groups enabled (metric=acc, max_gen=10)
  - Patches: vllm sleep/wake ROCm, verl trainer_mode release, verl ray_trainer filter_groups
  - Step 1-34 completed successfully (~4h), no OOM
  - Accuracy: 14% (s1) → 42% (s36) — faster learning than V1/V3 due to dynamic sampling filtering uninformative groups
  - Reward: -0.73 → -0.16 (approaching zero faster than V1)
  - Response length: 921 → 1612 tokens (growing normally)
  - Seqlen max: 221K → 423K (within safe range)
  - Throughput: avg 724 tok/s (peak 1035), gen latency avg 0.59 ms/tok
  - Dynamic sampling filter ratio: avg 35% (range 22-59%), 1 gen batch per step (96 prompts enough)
  - Disk-full incident: host disk hit 100% at ~step 34 (old V1/V2/V3 ckpts = 826G). Training blocked ~2.5h.
    Cleaned old ckpts, training resumed at step 35. Currently step 37+ and progressing.
  - Raylet warning "over 95% full" expected until ~5% of 3.5T (175G) is free
- Status: open (V4 running, need to reach step 79+ to confirm fix for V3 failure mode)
- V5 attempt (vLLM 0.16.0, new container lumen_v5_test):
  - Initial config was misaligned: copied from NPU reference (TP=2, SP=2, train_bsz=16, gen_bsz=48).
    Should have used GPU reference from FP8 docs: TP=1, SP=1, train_bsz=32, gen_bsz=96.
  - smoke10-14: ZMQ handle mismatch with ROLLOUT_TP=2 — FSDP workers use global GPU IDs,
    vLLM TP workers use local IDs. Fixed get_device_uuid() to map local→global via CUDA_VISIBLE_DEVICES.
  - smoke15: ZMQ fix confirmed (handles now GPU-0..GPU-7). New error: HIP IPC rebuild_ipc()
    fails with "HIP error: invalid argument" — CUDA IPC not supported on ROCm in this context.
    Moot point since reference uses TP=1 (no cross-process IPC needed).
  - Config realigned (2026-04-15): train_bsz=32, gen_bsz=96, mini_bsz=32, rollout_tp=1, sp_size=1,
    ppo_max_token_len=21504, free_cache_engine=true, gpu_mem_util=0.3, max_num_seqs=64.
    Matches VERL FP8 docs Qwen3-8B-Base GPU reference except ROCm memory adaptations.
- V5c CuMemAllocator investigation (2026-04-15):
  - Goal: Use native vLLM 0.16 CuMemAllocator on ROCm MI350X to enable proper sleep/wake memory management.
  - Approach 1: C++ per-chunk hipMemSetAccess patch
    - Patched cumem_allocator.cpp to call hipMemSetAccess per-chunk immediately after each hipMemMap
      (instead of once on entire VA range). Compiled and installed successfully.
    - Result: FAILED — still got "Memory access fault: Write access to a read-only page" on first
      generation after sleep/wake cycle. Confirms hipMemSetAccess silently fails on ROCm 7.12/MI350X.
  - Approach 2: Python monkey-patch CuMemAllocator.sleep/wake_up
    - Patched cumem.py to keep memory mapped during sleep (skip unmap_and_release), only copy data
      to/from CPU for offload/restore.
    - Result: FAILED — hit "Memory usage increased after sleeping" assertion in gpu_worker.py.
    - Patched assertion to warning on ROCm.
    - Result: Still FAILED — "Memory access fault: Write access to a read-only page" during first
      generation AFTER model init + weight update. This proves the bug is not just in sleep/wake
      but in the initial hipMemMap/hipMemSetAccess at allocation time.
  - Approach 3: Full CuMemAllocator bypass in gpu_worker.py
    - Patched _maybe_get_memory_pool_context() to return nullcontext() on ROCm (skip CuMemAllocator
      entirely). Model weights and KV cache allocated through standard PyTorch CUDA allocator.
    - Patched sleep()/wake_up() to manually offload/restore weights via CPU copy.
    - Result: No more "Memory access fault" — CuMemAllocator bypass works.
    - New problem: OOM during training. With standard allocation, vLLM EngineCore (separate child
      process) holds model weights + KV cache in its own GPU memory. Our gpu_worker patches only
      affect the worker process, not the child EngineCore. The KV cache (~150 GiB at gpu_mem_util=0.6)
      stays allocated in the child process during training, leaving no memory for FSDP.
    - Tried VLLM_ENABLE_V1_MULTIPROCESSING=0 — doesn't propagate to Ray worker processes.
  - Root cause conclusion: vLLM 0.16 V1 architecture couples sleep/wake to CuMemAllocator at a
    fundamental level. The EngineCore child process manages KV cache and model weights through
    CuMemAllocator's VMM-based allocation, which enables cross-process memory management.
    Without working CuMemAllocator (broken on ROCm due to hipMemSetAccess bug rocm-systems#2516),
    there is no mechanism to free KV cache in the child process during training.
  - Decision: Abandon vLLM 0.16 CuMemAllocator on ROCm. Fall back to V5b (vLLM 0.9.2) which uses
    a simpler architecture with custom _rocm_sleep/_rocm_wake_up that works correctly.
  - Files created: /home/danyzhan/cumem_patch/{cumem_allocator.cpp, cumem_allocator_compat.h,
    build.sh, patch_gpu_worker.py, patch_gpu_worker_v2.py}
  - Patches applied in lumen_v5_test container (not usable):
- V5b (gpu_memory_utilization=0.3, vLLM 0.9.2, ref-aligned config):
  - Config: gen_bsz=96, train_bsz=32, TP=1, SP=1, max_num_seqs=64
  - Container: lumen_verl_test (rocm/sgl-dev, vLLM 0.9.2rc2, ROCm 7.0)
  - V5b full7: Original _rocm_sleep/wake_up, fresh start, gpu_mem=0.3
  - Ran 32 steps before being stopped for V5d launch
  - Val accuracy: 7.0% (s5) → 23.0% (s30)
  - Reward: -0.885 → -0.003 (crossed zero at s30)
  - Response length: 812 → 3358 tokens (growing normally)
  - Seqlen max: 202K → 1021K (survived largest ever)
  - Throughput: avg 379 tok/s, gen latency avg 0.36 ms/tok
  - Step time: avg 810s (~13.5 min), increasing with resp length
  - Bottleneck: generation phase 88% of step time (straggler effect: min 5.4s, max 1376s at s30)
  - Stopped at step 32 to launch V5d with higher gpu_memory_utilization
- V5d (gpu_memory_utilization=0.6, vLLM 0.9.2, max_num_seqs=128):
  - Config: same as V5b except gpu_mem=0.6 (2x KV cache), max_num_seqs=128 (2x concurrency)
  - Rationale: V5b bottlenecked by generation straggler effect (~88% of step time)
    More KV cache → more concurrent sequences → less waiting for stragglers
  - VERL FP8 reference does NOT use dynamic sampling (filter_groups), so V5d skips it too
  - V5d full1: fresh start, gpu_mem=0.6, max_num_seqs=128
  - Steps 1-26 completed, CRASHED at step 27 during compute_log_prob
  - Val accuracy: s5=7.6%, s10=11.8%, s15=13.2%, s20=16.3%, s25=16.0%
    (V5b: 7.0%, 11.9%, 13.6%, 14.8%, 20.5%)
    Note: V5d s25 acc (16.0%) is lower than V5b s25 (20.5%).
  - Reward: -0.872 (s1) → -0.182 (s26), was improving steadily
  - Response length: 879 → 1788 tokens (growing, matches V5b pattern)
  - Seqlen max: 218-593K (s25 had 593K jump; V5b reached 1021K by s30)
  - Throughput: avg 703 tok/s (+86% vs V5b 379), peak 828 tok/s (s23)
  - Gen latency: avg 0.17 ms/tok (-53% vs V5b 0.36)
  - Step time: avg 387s (-52% vs V5b 810s)
  - CRASH DETAILS (step 27):
    - Error: HSA_STATUS_ERROR_OUT_OF_RESOURCES on WorkerDict pid=1987510
    - Phase: compute_log_prob (old_log_prob computation in training phase)
    - Cause: GPU OOM during training. With gpu_mem=0.6, vLLM allocates ~151 GiB
      KV cache. Even though _rocm_sleep frees KV cache before training, the vLLM
      model weights (~16 GiB) + EngineCore overhead remain on GPU, leaving ~214 GiB
      for FSDP training. As seqlen_max grew (593K at s25), the training phase's
      memory needs exceeded available memory.
    - Comparison: V5b (gpu_mem=0.3) survived seqlen_max up to 1021K because less
      vLLM overhead remained after sleep. V3 (gpu_mem=0.9) OOM'd at s79.
    - Ray error: ActorUnavailableError: keepalive watchdog timeout (actor hung on OOM)
  - Status: CRASHED at step 27
  - Next steps: 
    1. Can resume from checkpoint (step 26 should be saved)
    2. Options: (a) revert to gpu_mem=0.3 (V5b config, slower but stable), 
       (b) try gpu_mem=0.4 or 0.5 (compromise), or (c) enable dynamic sampling
       to limit sequence lengths
    - /opt/python/lib/python3.12/site-packages/vllm/cumem_allocator.abi3.so (per-chunk hipMemSetAccess)
    - /opt/python/lib/python3.12/site-packages/vllm/device_allocator/cumem.py (ROCm-safe sleep/wake)
    - /opt/python/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py (full CuMemAllocator bypass)
- ROCm sleep/wake_up investigation and fix (2026-04-13 ~14:00 UTC):
  - Problem: vLLM `gpu_memory_utilization=0.3` is a workaround because vLLM sleep/wake_up is disabled on ROCm.
    VERL reference uses 0.9 on H100 where CuMemAllocator sleep frees GPU memory between rollout and training.
    On ROCm, three gates prevent sleep mode:
    1. `vllm/platforms/interface.py:177`: `is_sleep_mode_available()` returns False for ROCm
    2. `vllm/config/__init__.py:582-584`: Raises ValueError if enable_sleep_mode=True on non-CUDA
    3. `verl/workers/rollout/vllm_rollout/vllm_async_server.py:222,259`: Forces enable_sleep_mode=False on ROCm
  - Existing code: Container already had a `_rocm_sleep`/`_rocm_wake_up` patch in gpu_worker.py (from prior session)
    that offloads weights to CPU pinned memory and frees KV cache tensors. This doesn't need CuMemAllocator.
  - Fix applied (4 patches):
    1. `vllm/platforms/interface.py`: `is_sleep_mode_available()` now returns True for ROCm
    2. `vllm/v1/worker/gpu_worker.py`: `_maybe_get_memory_pool_context()` returns nullcontext() when
       cumem_available=False (instead of trying CuMemAllocator.get_instance() which would assert)
    3. `vllm/v1/worker/gpu_worker.py`: `initialize_from_config()` same cumem_available guard
    4. `verl/workers/rollout/vllm_rollout/vllm_async_server.py`: Removed `and not torch.version.hip` gate
  - Expected behavior after fix: VERL passes enable_sleep_mode=True → vLLM accepts it on ROCm →
    _rocm_sleep offloads weights+frees KV between rollout and training → training phase has full GPU memory →
    gpu_memory_utilization can be 0.85+ (much higher KV cache budget for longer sequences)
  - 1A-v2 script prepared: gpu_memory_utilization=0.85, gen_batch_size=96, max_num_seqs=256
    Will launch after current 1A attempt 8 completes (or crashes).
  - Risk: _rocm_sleep weight offload/restore adds latency per step. wake_up reinitializes KV cache from scratch.
    But this is the same approach CuMemAllocator uses on CUDA (level 1 sleep offloads weights, discards KV).
  - Code review findings and additional fixes:
    1. [OK] Level 2 buffer save/restore already handled in _rocm_sleep (named_buffers saved for level==2)
    2. [FIXED] KV reinit: _rocm_wake_up now calls full `initialize_kv_cache()` instead of just
       `initialize_kv_cache_tensors()` — ensures attention backend rebinding, input batch reinit
    3. [FIXED] expandable_segments: VERL now calls `set_expandable_segments(True)` on ROCm even with
       sleep enabled (safe because ROCm doesn't use CuMemAllocator pool, only CUDA has the conflict)
    4. [Noted] Thread safety: relies on VERL draining requests before sleep; adding lock is optional
    5. [Noted] Memory leak risk over many cycles: monitor RSS + GPU across long runs
  - Patches saved to host: vllm/v1/worker/gpu_worker.py, vllm/platforms/interface.py,
    verl_patches/vllm_async_server.py

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

### [2026-04-12 V5e-dynamic-sampling]
- V5e (gpu_memory_utilization=0.6, vLLM 0.9.2, max_num_seqs=128, dynamic sampling):
  - Config: same as V5d + VERL_FILTER_GROUPS_ENABLE=1 (metric=acc, max_gen=10)
  - Rationale: V5d crashed at step 27 during compute_log_prob (GPU OOM, seqlen_max=593K).
    Dynamic sampling filters uninformative prompt groups (~49% filter ratio), reducing
    effective sequence load during training and mitigating OOM risk.
  - V5e full1: fresh start
  - Steps 1-60 completed, CRASHED at step 61 during compute_log_prob (GPU OOM)
  - Filter ratio: 16-56% per step (declining as model improves)
  - Val accuracy: s5=4.2%, s10=7.1%, s15=10.2%, s20=12.4%, s25=13.6%, s30=17.0%,
    s35=18.2%, s40=20.6%, s45=21.3%, s50=23.1% (matched V5b s30!), s55=22.8%, s60=21.8%
  - Reward: -0.715 (s1) → +0.006 (s30) → +0.133 (s57), oscillating near 0 from s30+
  - Response length: 728 → 4381 tokens (growing steadily, ~6x initial)
  - Seqlen max: 208-1507K (peak 1507K at s55, survived! V5d crashed at 593K)
  - Throughput: avg 888 tok/s, peak 1301 tok/s (s51). Consistently >1000 from s40+.
  - Gen latency: avg 0.46 ms/tok
  - Step time: avg 437s (-46% vs V5b's 810s). Increasing after s40 as responses grow longer.
  - CRASH DETAILS (step 61):
    - Error: ActorDiedError - worker killed during compute_log_prob
    - Phase: compute_log_prob (old_log_prob computation in training phase)
    - seqlen_max at crash step: 1145K (generation completed, crash during training)
    - Note: survived 1507K at s55 but crashed at 1145K at s61 — suggests the crash
      depends on both seqlen_max AND accumulated memory from multiple steps, or specific
      sequence distribution (min-max gap, balanced load) per step.
    - Same root cause as V5d: gpu_mem=0.6 leaves too much vLLM overhead on GPU
    - Dynamic sampling extended run from 26 steps (V5d) to 60 steps (+2.3x), demonstrating
      that filter_groups does help, but cannot fully prevent OOM at gpu_mem=0.6 when
      sequences grow long enough.
  - Status: CRASHED at step 61
  - Next steps:
    1. Resume from checkpoint (step 60 should be saved) with same config
    2. Try gpu_mem=0.4 or 0.5 with dynamic sampling for even better stability
    3. V5b (gpu_mem=0.3, no dyn samp) proved most stable (survived 1021K seqlen at s30)
       but was 2x slower. Trade-off: speed vs stability.

## Entry Template

```markdown
### [YYYY-MM-DD session-name]
- Symptom:
- Possible bug:
- Evidence so far:
- Next check:
- Status: open | ruled out | resolved
```
