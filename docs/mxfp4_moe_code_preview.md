# MXFP4 MoE code preview

This is a source-reading development snapshot, not a release or a runtime qualification.
It contains L0 and the reviewed P0a adapter, including full-forward/backward microbatch `no_sync`, the host source-path follow-up, and the separately reviewed publication lint/host-boundary refactor.

## Repository dependency

The `third_party/aiter` gitlink selects AITER snapshot `8ce21a4e8b45cec65b279bfeb31019e76a5cf61f` from https://github.com/DaiXindi-AMD/aiter.git . The AITER URL is changed only on this preview branch so the commit can be fetched. MORI is unchanged. This is not a change to the existing M1 source worktree or its pins.

## Suggested reading order

1. [Controlled Core startup](../lumen/models/mxfp4_moe_startup.py) and [real host driver](../lumen/models/mxfp4_moe_driver.py): selector validation, DDP installation, accumulation windows and complete F+B microbatch contexts.
2. [Core expert adapter](../lumen/models/mxfp4_moe.py) and [binding](../lumen/models/mxfp4_moe_binding.py): actual routed counts, parameters, leases and original Core call ordering.
3. [Autograd and real Torch operations](../lumen/ops/mxfp4_moe.py), [FC orchestration](../lumen/ops/mxfp4_moe_execution.py), [device materialization](../lumen/ops/mxfp4_moe_materialize.py), and [shared host materialization boundary](../lumen/ops/mxfp4_moe_materialization_boundary.py).
4. [Existing-hook DDP handoff](../lumen/utils/mxfp4_moe_ddp.py), [obligation ledger](../lumen/utils/mxfp4_moe_ledger.py), and [fixed-schema failure reporter](../lumen/utils/mxfp4_moe_diagnostics.py). Public return means host submission, not device completion. Unexpected diagnostic errors preserve the original exception and expose the secondary cause.
5. [Paired numerical entry](../tests/env/test_mxfp4_moe_paired_numeric.py), [exact receiver-M entry](../tests/env/test_mxfp4_moe_exact_expert_m.py), and [two-microbatch real Core entry](../tests/env/test_mxfp4_moe_microbatch_runtime.py). These GPU entries are authored, not executed evidence.

For multiple microbatches, enclose each complete forward and backward, including checkpoint replay, in `runtime.microbatch_scope(index)`. Do not use a forward-only no_sync context.

The explicit fixture CLI plugin is `-p tests.env.mxfp4_moe_pytest`, with `--lumen-moe-mxfp4 --lumen-moe-mxfp4-backend sequential --lumen-moe-mxfp4-strict`. The general parser defaults are not changed to force P0a implicitly. Runtime commands require the separately authorized world8 environment, signed contract, budgets, launcher/store/timer setup and run identifiers.

## Evidence and limitations

The actual repository configuration was used to check every changed Python file: AITER 17 and Lumen 76; Ruff 0.16.0 and Black 26.10.1 (`--workers 1`) plus staged whitespace checks passed. This is not a full-repository or GitHub CI pass. No new ignore/noqa or relaxed lint configuration was used.

The [host runner](../tests/run_mxfp4_host_checks.py) imports complete real host modules without importing the Lumen/ops initializers, Torch, AITER or Core, and without fake parent packages or AST-exec loaders. The publication refactor passed 52 related host tests; one guarded normal-Core-import class was skipped and is unrun. The final secondary-cause clarification passed 15 affected diagnostic tests. These counts overlap and are not a new combined 67-test run. The Core-body import case needs the authorized environment and explicit `MXFP4_CORE_SOURCE_ROOT` matching the two pinned source files; no configured source is not a functional pass.

No real Lumen/Torch/AITER/Core import integration, pytest collection, AITER prepare/JIT/build, GPU numerical/EP/DDP/reducer execution, stream/resource traces, fault/exit behavior, training or benchmarks were validated by this publication. Do not present this snapshot as PR-ready or production-ready. M1 job/runtime/source worktrees were not changed or operated on.
