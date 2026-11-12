# MXFP4 example reproducibility implementation plan

> Execute inline with executing-plans; run one independent whole-change review.

Goal: numbered examples 12 and 13 describe the current upstream AITER integration accurately and run from checked-in code without temporary patches.

Architecture: retain the 13-row runbook and historical non-MXFP4 measurements. Add the missing FSDP2 LoRA launcher, formalize version-compatible Megatron calls, and provide an explicit 8-GPU/20-step runner with environment and data manifests. Pin the tested runtime image and AITER source; rebuild extensions in an isolated JIT directory. Report legacy shuffled-input transpose and avoid claiming paired speedups or convergence.

- [x] Write failing CPU regressions for old/new spec signatures, partition stride, and nonempty padding-mask preservation.
- [x] Fix spec keyword compatibility, LayerNormLinear stride, and residual attention/MLP mask forwarding.
- [x] Add case13 entry point and reproducibility runner/preflight/data preparation; make backend synchronization overridable.
- [x] Restore English/Chinese 13-row runbook, correcting MXFP4 defaults and replacing ambiguous current performance claims with dated evidence.
- [x] Test clean checked-in sources: CPU regressions, GPU integration, shell syntax, then numbered cases12/13 at eight GPUs when available.
- [x] Independent review, resolve findings, save logs/manifests, commit and push the integration branch.

Constraints: no new GPU kernels; no changes to model dimensions, loss masks, LR, seed, H16 signs, activation RTN or gradient SR. Preserve nonempty MoE masks or reject unsupported APIs explicitly. GPU availability is checked before the 8-GPU run; do not take occupied cards.

Verification: Lumen 00edddf, CPU 3 passed, initializer contracts 2 passed, AITER GPU 13 passed, both 8-GPU training cases completed 20 steps / exit 0. Evidence saved under docs/evidence/mxfp4-2026-11-12. Reviewed and all findings resolved. Publication remains on the integration branch.
