# AITER public API integration plan

> Agentic execution: implement inline using executing-plans, then run the user-selected review.

**Goal:** Connect Lumen training to the upstream implementations from AITER #5531, #5538, #5542, #5548, #6458 and #6476.

**Architecture:** Preserve Lumen wrappers and parameter/cache lifecycle. Probe upstream public APIs and adapt arguments and physical layouts. Preserve legacy kernels only for capabilities the upstream API does not expose, with explicit fallback reporting.

**Tech stack:** Python, PyTorch, Triton, ROCm/AITER.

**Spec:** The user-authorized integration document at Lumen dae77b240, outputs/aiter_5542_lumen_integration/AITER_5542_LUMEN_INTEGRATION_PLAN.md, extended to the two H16 fusion PRs.

## Constraints and review focus

- Preserve H16 sign, group size, activation RTN and gradient SR choices.
- Never read shuffled payloads or swizzled scales as row-major.
- Preserve user-supplied Philox seeds and offsets; document upstream SR stream differences.
- Keep public Lumen signatures and forward/backward cache wiring.
- Missing APIs must produce explicit fallback reasons; required SwiGLU preflight fails when fusion is requested.
- Fixed upstream revision must actually contain all six merged implementations.
- Packaging, Python sources and compiled AITER extensions must come from the same revision.

## Tasks

- [x] Add failing CPU forwarding/probe tests and GPU numerical regression tests for public API selection, layout options and SR reproducibility.
- [x] Adapt current AITER compatibility surfaces using PR #30's focused compatibility commit, excluding its examples runbook changes.
- [x] Replace SwiGLU temporary names; support out buffers and update Megatron chunked backward, probes and preflight.
- [x] Route 1D quantization, 32x32 RTN quantization, row-major transpose and both H16 fusion wrappers through upstream APIs. Retain only documented unsupported legacy variants.
- [x] Pin a verified upstream commit, remove vendor activation overlays and rebuild installations consistently.
- [x] Run static checks, CPU tests, then GPU numerical/integration checks in an authorized compute allocation if available. Record unavailable checks explicitly.
- [x] Run review, resolve verified issues, commit and push an independent Lumen branch.

Validation status: six CPU tests and scoped static checks passed. GPU job 68343968 is queued; GPU, Docker and full training validation remain pending. Bugbot's one P1 was reproduced and fixed with a regression test.
