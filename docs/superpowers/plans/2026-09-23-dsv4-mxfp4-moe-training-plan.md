# Lumen MXFP4 A4W4 DeepSeek-V4 MoE — Development Plan

**Date:** 2026-09-23

**Status:** Ready to execute after M0 clean-lineage approval

**Design source:** docs/superpowers/specs/2026-09-21-dsv4-mxfp4-moe-training-design.md

**Lumen integration branch:** mxfp4-moe

**PR status:** No PR requested

---

## 1. Objective

Deliver an opt-in, experimental Lumen-native MXFP4 A4W4 training path for DeepSeek-V4 routed-expert FC1/FC2 on gfx950, covering:

~~~text
EP8 / local-E32
EP4 / local-E64
ETP=1
FC1 [M,4096] x [4096,4096]
FC2 [M,2048] x [2048,4096]
Fprop + DGrad + eager WGrad
~~~

The implementation must preserve Megatron routing, dispatch/combine, SwiGLU/clamp, probability gradient, recompute and checkpoint semantics. New GPU kernels belong to AITER; Lumen owns integration and training lifecycle.

The plan is complete when:

1. the source/build lineage is clean and reproducible;
2. P0 sequential MXFP4 is a strict executable reference;
3. P1 grouped operators pass numerical, lifecycle, distributed and resume gates;
4. the mandatory matched-scale A8W4 quality control exists;
5. explicitly claimed quality cells pass their pre-registered statistical gate;
6. grouped strict runs pass route-bucket certification and performance promotion gates;
7. all evidence is tied to an auditable manifest.

---

## 2. Non-negotiable execution rules

1. Do not commit the current Lumen or AITER dirty workspace as one baseline.
2. Do not implement against two AITER truth sources. Lumen gitlink, imported source and loaded extension must resolve to one pinned lineage.
3. Do not add GPU kernels under Lumen.
4. Do not use --linear-fp4 for this feature; use --lumen-moe-mxfp4.
5. Do not use A8W4 or BF16 as a strict A4W4 fallback.
6. Do not fallback after BoundExecution commit.
7. Do not use Parameter._version as cache correctness authority.
8. Do not use Python/Torch global RNG for MXFP4 SR.
9. Do not enable deferred WGrad in V1.
10. Do not tune before kernel correctness passes.
11. Do not use synthetic balanced routing as certification evidence.
12. Do not call a 100-step run convergence.
13. Do not promote a quality result without a pre-registered power analysis.
14. Do not publish a formal candidate/control result with fallback, skip, OOM, NaN/Inf or an unverified selected backend. A current-FP8 run with truthful fallback counters is diagnostic-only.
15. Do not modify Megatron core unless the existing provider/hook boundary is proven insufficient and the minimal change is separately reviewed.

---

## 3. Dependency graph

~~~text
M0 extract clean prerequisites
  |
  +-> M1 pinned clean implementation baseline
        |
        +-> N0 numeric formats / selector / golden vectors
        +-> R0 Lumen Philox allocator / ledger / weight epoch
        +-> A0 AITER ABI and capability schema
          |
          +-> P0 sequential Lumen reference
          |     |
          |     +-> Q0 matched-scale A8W4 numeric control
          |
          +-> A1 grouped quant/cache builder
          +-> A2 grouped Fprop
          +-> A3 grouped DGrad
          +-> A4 grouped eager WGrad
          +-> A5 grouped matched-scale A8W4 control
                    |
                    +-> P1 transactional grouped integration
                           |
                           +-> M2 ABI-freeze forward-port
                                  |
                                  +-> T0 final route certification
                                         |
                                         +-> C0 final replay/reshard --+
                                         +-> I0 final EP stability ----+-> QG powered quality
                                                                          +-> B1 full-step performance
                                                                                     |
                                                                                     +-> experimental promotion
~~~

Independent work may run in parallel only after its prerequisite contract is frozen. In particular:

- numeric golden vectors, Lumen state machinery and AITER ABI review can proceed in parallel after M0;
- Fprop, DGrad and WGrad kernel work can proceed in parallel after A0/N0, but each must keep one shared ABI/layout contract;
- tuning starts only after the corresponding kernel passes correctness;
- quality and performance formal runs start only after the exact release candidate and manifest are frozen.

---

## 4. M0 — Establish the clean implementation lineage

### M0.1 Inventory both dirty workspaces

Produce separate Lumen and AITER inventories:

~~~text
tracked diff
untracked file list
per-file content digest
dependency/capability supplied
tests already available
tests still missing
keep / split / reject decision
~~~

For each dirty change, map it to a spec requirement. Examples of potentially reusable classes include numeric selectors, dual-layout quantization, cache invalidation and transpose/shuffle helpers; similarity alone is not enough to keep a change.

Acceptance:

- every retained line has one named prerequisite owner;
- unrelated tuning, benchmarks, experiments and local artifacts are excluded;
- no generated binary, profile output or cache is included.

### M0.2 Extract minimal AITER prerequisites

Create independent, DCO-signed commits for each reusable prerequisite. A commit must contain its focused tests and no unrelated formatting.

Candidate split, subject to the inventory:

1. shared numeric codec/selector helpers and byte-exact tests;
2. shared transpose/shuffle/H16 helpers and tests;
3. any required cache/layout builder primitive and tests;
4. any independent bug fix needed by the public grouped API.

Run the smallest relevant AITER tests after every commit. A failed unrelated existing test must be recorded with evidence; it cannot be described as passing.

### M0.3 Extract minimal Lumen prerequisites

Create independent commits for:

1. cache invalidation behavior needed by committed_weight_epoch;
2. reusable quantization facade changes required by P0;
3. dispatch/probe fixes required by the new public AITER API;
4. tests that expose existing grouped-linear parameter API drift;
5. unrelated changes must remain outside this lineage.

Each commit gets its own focused tests. Do not combine all dirty changes into baseline.

### M0.4 Pin and prove the runtime

Build or install AITER from the selected clean commit and record:

~~~text
source commit and tree digest
Lumen gitlink
aiter.__file__
loaded extension path
extension SHA256/build-id
ROCm/compiler/PyTorch/RCCL
GPU arch/CU
container digest
dirty=false
~~~

Update the Lumen gitlink only to the exact tested AITER commit.

### M0.5 Integrate prerequisites into mxfp4-moe

The existing mxfp4-moe branch is currently the design-document carrier, not the implementation baseline. After M0:

1. merge or cherry-pick the reviewed prerequisite commits in dependency order;
2. preserve their individual commit boundaries;
3. avoid a force rewrite unless a separate branch-maintenance decision explicitly authorizes it;
4. regenerate the manifest and rerun prerequisite tests on the final branch head.

M0 exit:

- Lumen, AITER and Megatron are pinned and clean;
- runtime source/build identity is singular;
- all prerequisite commits and test evidence are reviewable independently.

### M0.6 Declare M1

M1 is the immutable clean implementation baseline produced by M0: exact Lumen/AITER/Megatron commits, matching Lumen gitlink/imported AITER/build-id, toolchain/container digests and `dirty=false`. N0 through P1 are developed and first qualified on this lineage. Changing any pinned component creates a new M1 manifest rather than mutating the old record.

---

## 5. N0 — Freeze numeric and layout conformance

### N0.1 Publish versioned definitions

Implement versioned definitions and golden vectors for:

~~~text
lumen_fp4_e2m1_e8m0_tile32x32_s1024_v1
lumen_fp4_e2m1_e8m0_row1x32_v1
fp4_e2m1_e8m0_selector_v1
E2M1 payload codec
E4M3FN payload codec
E8M0 scale codec
mxfp4_weight_tile32x32_v1
mxfp4_segmented_row_and_t_v1
hadamard16_all_plus_v1
~~~

The definitions must cover amax, exponent rounding/ties, zero/signed zero, NaN, Inf, subnormal, clamp/saturation, nibble order, endianness, scale placement and alignment.

### N0.2 Golden-vector suite

Add byte-exact fixtures for:

- normal, boundary and tie values;
- all zero, signed zero, NaN/Inf/subnormal;
- positive/negative saturation;
- row and tile boundaries;
- ragged M and logical-zero padding;
- forward and transpose views;
- identical BF16 source producing independent row/transpose quantization;
- a test that fails if row FP4 is decoded and requantized for transpose.

### N0.3 A8W4 matched-scale control

Implement the quality-only recipe lumen_w4_s1024_a8_nonweight_operands_v1:

- one A4 selector invocation per BF16 group;
- identical E8M0 scale bytes supplied to A4 and E4M3FN quantizers;
- identical W4 bytes in fixed-snapshot tests;
- A8 Fprop and DGrad with W4;
- A8 x A8 WGrad;
- same accumulator, reduction order and destinations as A4.

N0 exit:

- all numeric/layout golden vectors are byte exact;
- IDs and ABI versioning are documented;
- A4/A8 fixed-snapshot scale and W4 equality tests pass.

---

## 6. R0 — Lumen training-state foundations

### R0.1 Committed weight epoch

Add a single owner for:

~~~text
committed_weight_epoch
cache_generation
optimizer-step readiness event
external weight commit
checkpoint restore
~~~

Required tests:

- successful update advances epoch only after BF16 sync event;
- overflow/skipped step does not advance;
- all microbatches in one step see one epoch;
- DCP in-place restore invalidates cache despite unchanged identity/_version;
- mark_mxfp4_weights_dirty is rejected during accumulation or active consumers;
- naked .data mutation is documented unsupported.

### R0.2 Budgeted cache allocator

Implement:

~~~text
planner prediction
persistent reservation ledger
peak-build reservation ledger
active/retired generation accounting
unique-storage de-duplication
hard quota enforcement
~~~

Tests must cover 0.75x default miss, explicit <=1.0x override, old-generation peak accounting, allocator plan breach and no hidden post-bind allocation.

### R0.3 Philox logical-draw allocator

Implement rank/domain-separated seeds, declaration-time canonical ordering, atomic reservations, SRTokenV1 receipts and checkpoint state.

Tests:

- declaration order invariance;
- candidate probing consumes no draws;
- retry/recompute reuses token;
- unused reservation is not reclaimed;
- row/transpose ranges do not overlap;
- padding consumes slots, empty expert consumes zero;
- A4/A8 corresponding elements share draws;
- uint64 overflow fails before launch;
- resume restores next_draw and invocation IDs.

### R0.4 Obligation ledger

Implement logical and physical IDs, terminal state conservation, stream-event completion and all-rank digest aggregation.

Keep process-local binding IDs, addresses, timestamps and branch UUIDs out of the canonical replay hash. Hash a stable logical-binding key plus canonical config/route/weight/RNG/layout/status fields. At every checkpoint boundary, rotate both the uninterrupted run and any later resume branches onto the same deterministic logical child segment rooted at the saved parent frontier; store physical attempt identity only in external artifacts.

Tests:

- declared equation holds for success, empty, prelaunch failure and abort;
- child kernel launches do not change logical counts;
- exact checkpoint rejects open entries;
- recompute creates physical attempts without duplicate logical obligations.

R0 exit:

- weight/cache/RNG/ledger behavior passes without grouped kernels;
- state is serializable in a versioned Lumen checkpoint sidecar;
- no Python random is used by the new path.

---

## 7. A0 — Freeze AITER public API and compiled ABI

### A0.1 Reuse audit

Search public wrappers, private kernel bodies, quantizers, shuffle helpers, split-K reductions, tests, benchmarks and config loaders. Record the concrete semantic or hardware gap before adding a new kernel.

### A0.2 ABI symbols

Define _v1 symbols for:

~~~text
weight-cache build
row quantization
transpose/H16 quantization
grouped Fprop
grouped DGrad
grouped WGrad
~~~

Low-level types are restricted to Tensor, Tensor[], fixed-width scalars/enums and fixed flat tuples. Mutation, workspace, alignment, capacity and alias contracts are explicit.

### A0.3 Capability/error schema

Expose:

~~~text
ABI major/minor
numeric/layout support
static capability result
workspace requirements
alignment/capacity requirements
capture-safe flag
typed candidate miss
typed run-fatal errors
~~~

Do not use arbitrary Python exceptions as fallback detection.

### A0.4 Four-stage facade

Implement and test:

~~~text
query_static
prepare_kernel
bind_microbatch
launch
~~~

The facade may use Python dataclasses, but compiled calls receive expanded tensors/scalars. No opaque Python plan handle crosses the ABI.

Define and test the low-level capture contract. A capture-safe plan binds immutable RoutingMetadata identity, counts/offsets and every layout-specific padded-offset tensor, operand capacities, caller-owned workspace, stable tensor addresses, all numeric/physical/prepared layout IDs, and SR draws_reserved/draws_used. routing_fingerprint is diagnostic only. If SR state is not an explicit ABI operand with a unique non-overlapping reservation per replay, or any address/capacity/layout invariant cannot be held, query_static returns capture_safe=false. V1 does not claim dynamic-routing end-to-end replay.

A0 exit:

- ABI review is signed off by Lumen and AITER owners;
- schema/version mismatch tests fail closed;
- query is pure and launch has no lookup/JIT/allocation path.

---

## 8. P0 — Sequential MXFP4 reference

### P0.1 CLI and scope

Add:

~~~text
--lumen-moe-mxfp4
--lumen-moe-mxfp4-backend=sequential
--lumen-moe-mxfp4-strict
~~~

Reject LUMEN_DSV4_LINEAR_FP8, LUMEN_MXFP4_DGRAD_HADAMARD=1, deferred WGrad, unsupported arch/shape/local-E/ETP and bias. Log scope=routed_experts.

### P0.2 Module and autograd

Use the existing expert-sorted interface and call the dense MXFP4 semantics per expert. Preserve weight0..weightN Parameter/state_dict keys.

P0 must implement:

- exact routing segment handling;
- same quantizers/layout IDs/RNG tokens as P1;
- forward cache and defined transpose view;
- direct ACCUMULATE_FP32 main-grad;
- all empty/ragged semantics;
- full recompute token/cache reuse.

### P0.3 Tests

Run the full design-spec section 15.3 shape/routing matrix on both local-E values. Add Megatron provider, CLI, state_dict, router-gradient, activation order and checkpoint tests.

### P0.4 Distributed stability

Run:

- 4L TP8/EP8/ETP1 strict sequential, 100 optimizer steps;
- 43L TP4/PP4/EP4/ETP1 strict sequential, 100 optimizer steps when resource-ready.

P0 exit:

- every Fprop/DGrad/WGrad obligation completes sequential MXFP4 or EMPTY;
- fallback and open ledger counts are zero;
- P0 artifacts become the executable reference for P1.

---

## 9. A1–A5 — AITER grouped implementation

### A1 — Weight/activation builders

Implement canonical weight packing, transpose/prepared views, row quantization and per-expert H16/transpose/pad quantization. Accept caller-owned metadata/workspace.

Acceptance:

- operand parity with N0 trusted quantizer is bitwise;
- local-E32/64 and all ragged/empty cases pass;
- memory requirements are queryable before allocation;
- every launchable kernel has repr, tests and benchmark.

### A2 — Grouped Fprop

Implement FC1/FC2 grouped A4W4 Fprop over expert-sorted segments.

Acceptance:

- fixed packed-operand parity oracle passes;
- all-empty is no-op;
- no cross-segment read/write;
- config_id and selected backend are observable;
- production lookup path, not a literal test config, is exercised.

### A3 — Grouped DGrad

Implement dY row-SR x cached weight transpose view.

Acceptance:

- no BF16 re-quantization of updated weights;
- late dY attachment contract passes;
- staging is bind-planned when needed;
- RNG mapping does not depend on tile/config.

### A4 — Grouped eager WGrad

Implement OVERWRITE and ACCUMULATE_FP32.

Acceptance:

- direct FP32 main-grad write;
- M_e=0 OVERWRITE zero-fill and ACCUMULATE_FP32 unchanged semantics;
- all-empty WGrad OVERWRITE may run only the bound zero-fill child operation, never a 0-M GEMM; Fprop/DGrad/ACCUMULATE remain no-op;
- illegal aliases rejected;
- deterministic/fixed reduction required by exact cells;
- no BF16 intermediate in production accumulation;
- no deferred queue or closure.

### A5 — Grouped matched-scale A8W4 control

Extend the same grouped execution structure to the quality-control recipe:

- Fprop A8 x W4;
- DGrad A8 x W4 transpose;
- WGrad A8 x A8;
- X uses E4M3FN RNE; dY uses E4M3FN SR; FNUZ/E5M2 are rejected;
- external scale bytes from the shared A4 selector;
- same routing, workspace, reduction and output contracts as A4;
- no registration in the A4 runtime fallback chain.

Until A5 passes, A8 sequential results are diagnostic only and cannot enter confirmatory quality analysis.

Each AITER kernel commit includes its public wrapper, relevant tests, benchmark and config support. Shared helpers go to utils/common; do not duplicate kernels.

---

## 10. P1 — Transactional grouped Lumen integration

### P1.1 Static preparation

Add guarded AITER probes and cacheable static plans. The runtime_compatible predicate checks ABI major/minor policy, backend build/version range, numeric/physical/prepared layout IDs, required capability bits, GPU arch/CU/ROCm conditions, descriptor/bucket/envelope/capacity and workspace/alignment. Source path, commit and artifact provenance remain separate certification fields. Static preparation may JIT/lookup outside the timed path.

### P1.2 Routing metadata

Lumen/Megatron owns semantic routing counts, caller-owned buffers and object lifetime. For each routing epoch it invokes the AITER-owned RoutingMetadataV1 physical-schema builder/validator, wraps the returned tensors/fields in an immutable Lumen facade, and keeps that object alive through all asynchronous FC1/FC2/backward consumers.

### P1.3 Transactional bind

For every microbatch:

1. declare obligations and reserve SR ranges;
2. receive the immutable candidate plans produced by P1.1;
3. side-effect-free preflight all candidates;
4. apply grouped tuning policy;
5. select one backend bundle;
6. reserve/build cache, workspace and staging;
7. validate all F/D/W destinations and schemas;
8. return immutable BoundExecutionV1 as commit point.

After step 8, no backend/config change or fallback is allowed.

### P1.4 Backward attachment

Use attach_runtime_operands_v1 only for live tensor validation. Any mismatch is run-fatal. The resulting EphemeralOperandView owns no storage and is not checkpointed.

### P1.5 State machine tests

Exercise all six backend/strict combinations and both tuning policies. Distinguish:

~~~text
requested backend
selected backend
auto selection
backend fallback
precision fallback
forced sequential
~~~

P1 exit:

- grouped-vs-P0 parity passes all required routes/shapes/passes;
- no post-commit fallback path exists;
- full recompute and direct FP32 main-grad pass;
- the 4L explicit validated-config grouped strict stability run has zero fallback; final 4L/43L certified-config stability is an M2 -> T0 -> I0 gate.

---

## 11. T0 — Routing bucket and tuning certification

Before M2, this stage may build tooling and collect exploratory evidence only. Final route capture, tuning, held-out validation and certification MUST be rerun on the M2 release-candidate manifest; pre-M2 rows cannot be promoted by copying their labels or timing results forward.

### T0.1 Capture corpus

Capture ordered group_sizes from:

- BF16 4L EP8 and 43L EP4;
- frozen P1 strict runs;
- multiple layers, steps, seeds and data shards.

Store complete counts plus seghist-v1 descriptor and provenance. Split by complete capture group into tuning, validation and final held-out sets.

### T0.2 Learn bucketizer

Use collision/regret analysis to propose route_bucketizer_version. Do not pre-freeze a hand-written 3x4 classifier. Different pass/op variants may use different bucketizers.

Rules:

- all-empty bypasses grouped GEMM config lookup; WGrad OVERWRITE still executes/guarantees the bound zero-fill child operation;
- no wildcard route fields;
- no nearest bucket;
- no cross-CU relaxation;
- max 12 buckets is a target, not a requirement;
- split a bucket or choose a generic config when ranking flips or regret fails.

The immutable bucketizer artifact must define, per op variant:

~~~text
descriptor and bucketizer versions
candidate-independent work quantum
padded_work integer formula
integer bucket boundaries and inclusivity
route classifier rules
canonical serialization/digest
OOD result
~~~

V1 uses work quantum 16 rows for Fprop/DGrad and semantic pad32 for WGrad. prepare loads/parses/compiles the static registry; bind performs only deterministic in-memory exact-map selection for the current route.

### T0.3 Tune candidates

Tune only kernels that already pass correctness. Keep normal execution pinned to one config; autotune is an offline tool only.

### T0.4 Held-out certification

Each exact key/bucket needs real center, boundary and extreme artifacts from at least two capture runs.

Pass:

~~~text
weighted mean regret U95 <= 3%
every center/boundary/extreme regret U95 <= 5%
runtime capacity/envelope checks pass
correctness remains valid
~~~

Use route-representatives-v1 from the spec to select center/boundary/extreme. The artifact manifest records the selected IDs, envelope min/max for all scalar/histogram fields, config capacity, candidate oracle, invocation frequencies and the process-first hierarchical bootstrap inputs.

Store artifact_set_id, config ID, route/bucket versions, candidate set, raw timing and immutable digests.

T0 exit:

- runtime compatibility predicate is implemented and tested; its value is recomputed at each bind;
- correctness_validated=true;
- tuning_config_certified=true;
- artifact_provenance_certified=true;
- every non-empty grouped route seen in formal strict runs maps to an exact certified row; all-empty is the specified no-op.

---

## 12. C0 — Checkpoint, exact replay and reshard

The harness and smoke cases may be developed on M1, but every formal replay_class or reshard qualification MUST be rerun after M2 against the final build/kernel/config digests.

### C0.1 Checkpoint sidecar

Add a versioned Lumen sidecar containing:

~~~text
committed weight epoch
RNG domain seeds/next_draw/next invocation IDs
ledger schema/epoch/counts/hash-chain frontier
last committed optimizer step
execution/certificate manifest digest
~~~

Do not persist packed caches.

### C0.2 warm_start

Verify model-only load and explicit reset of optimizer/scheduler/scaler/RNG/data cursor/weight epoch.

### C0.3 exact_continue

Require an explicit backend plus a frozen, versioned mapping from complete static key and exact route-bucket key to config_id/prepared_layout_id, bound to the run-level certificate and artifact digest. Auto, an incomplete mapping, heuristic selection, fallback, autotune, alternative-candidate selection and mapping changes are forbidden. A fresh resume process may deterministically rebuild StaticPlan/JIT code from that exact certified artifact and must verify mapping/kernel digests. Checkpoint load clears derived caches, so the first resumed microbatch performs a normal transactional bind using the frozen mapping and rebuilt cache; it must not reuse a pre-checkpoint BoundExecution or treat prepare/bind as a selection opportunity. At each checkpoint boundary compare one uninterrupted branch with two independent resume branches from the same checkpoint:

- 2 paired seeds minimum;
- 2 non-identical quiescent checkpoint boundaries;
- 100 successful post-resume optimizer steps per boundary;
- exact topology cells:
  - 4L world8 TP8/PP1/EP8/ETP1 -> same;
  - 43L world16 TP4/PP4/EP4/ETP1 -> same;
- load-boundary canonical bitwise equality for BF16 model, FP32 master, optimizer moments, scheduler/scaler, global step, consumed samples/tokens, cursor, router/expert-bias state, all RNG domains, committed weight epoch and ledger frontier;
- load leaves derived caches absent/invalid; first forward rebuilds from restored BF16 Parameters and matches fresh quantization packed/scales/layout IDs bitwise;
- per-step bitwise equality for sample/data-order digest, route/immutable-metadata digest, FP4 RNG token/frontier, packed-weight hash, loss, gradients, BF16 model, FP32 master, optimizer/scheduler/scaler state, ledger leaf and hash-chain state;
- final model/master/optimizer/RNG/cursor bitwise equality as a redundant terminal assertion.

Any mismatch is run-fatal for the exact attempt. The failed execution domain may only be reported later as state_complete_continue after its separate state-completeness gate; it does not continue by silently downgrading.

### C0.4 state_complete_continue

Run an independent qualification rather than relabeling a failed exact attempt:

- use the same registered topology/rank mapping and at least 2 paired seeds x 2 quiescent checkpoint boundaries;
- require the complete load-boundary state comparison and cache-absence/fresh-rebuild checks listed in C0.3;
- restore-state, missing-state or cache-rebuild mismatch yields REPLAY_UNSUPPORTED;
- run 100 post-resume optimizer steps per boundary with finite loss/grad, ledger conservation, open=0, strict zero-fallback and no OOM/device fault;
- retain per-step uninterrupted-vs-resume numeric diagnostics, but record continuation stability separately from replay_class and make no bitwise or quality claim.

### C0.5 reshard_continue

Validate each registered direction separately:

~~~text
world16 TP4 PP4 EP4 ETP1 -> world16 TP8 PP2 EP8 ETP1
world16 TP8 PP2 EP8 ETP1 -> world16 TP4 PP4 EP4 ETP1
~~~

At load boundary, compare canonical global BF16 model, FP32 master, optimizer moments, scheduler/scaler, global step, consumed samples/tokens, data cursor and router/expert-bias state. Invalidate and rebuild caches from target-rank BF16 Parameters.

Define a versioned reshard_transition_id. Validate the source frontier/manifest, bind the target manifest, and start a new ledger epoch from the canonical source-frontier/source-manifest/target-manifest/transition digest. Do not require raw per-rank RNG equality across changed ownership. Instead:

- canonical-remap global identity domains;
- deterministically rebuild data-worker/model-parallel domains from global cursor, target rank mapping and transition ID;
- start target rank-local FP4 SR domains at next_draw=0 and next_logical_invocation_id=0, with seeds derived from run seed, source frontier, target manifest, target rank, domain and RNG schema; namespace both counters by the new target ledger/RNG epoch;
- archive the source state and complete target mapping/seed ledger.

Run 2 paired seeds x100 post-resume steps and require finite loss/grad, ledger conservation, open=0, strict zero-fallback and no OOM/device fault. Each direction is certified independently and does not imply trajectory parity, bitwise replay or a quality claim.

C0 exit:

- exact cells have bitwise replay certificate or are correctly renamed;
- STATE_COMPLETE_ONLY cells pass their independent load/cache/continuation gate; otherwise they are REPLAY_UNSUPPORTED;
- reshard claims list exact direction/topology only;
- checkpoint boundary has no open ledger entries.

---

## 13. I0 — Megatron integration and stability

Integration tests may run throughout development, but the registered 4L/43L formal stability matrix MUST be rerun after M2 on the release-candidate manifest.

### I0.1 Integration tests

Verify:

- router/top-k, dispatch/combine and shared expert stay on the existing path;
- FC1/FC2 activation/probability order is unchanged;
- routing probability gradient matches reference;
- parameter/state_dict names remain stable;
- EP all-to-all and local sort metadata align with RoutingMetadataV1;
- all-rank backend/config/fallback ledger is complete.

### I0.2 Stability matrix

At minimum:

| Model | Topology | Mode | Precision | Steps |
|---|---|---|---|---:|
| 4L | TP8/PP1/EP8/ETP1 | cold | BF16/A8/A4 | 100 |
| 4L | TP8/PP1/EP8/ETP1 | warm | BF16/A8/A4 | 100 |
| 43L | TP4/PP4/EP4/ETP1 | every supported mode | BF16/A8/A4 | 100 |

All low-precision formal runs use explicit backend, strict=true and zero fallback. A8 and A4 use the same execution structure when the result is intended for confirmatory quality.

Stability pass means only:

~~~text
all steps complete
loss/grad finite
route/ledger conservation
zero fallback
no OOM or device fault
checkpoint semantics as declared
~~~

---

## 14. M2 — ABI-freeze forward-port and release-candidate freeze

After the grouped ABI/layout contract is frozen and P0/P1 pass on the M1 clean baseline, pin one explicit latest-AITER target commit. Forward-port the AITER implementation to that commit, update the Lumen gitlink/adapter to the exact tested revision, and create a new clean manifest. Do not track a moving branch name or absorb unrelated AITER changes.

Repeat on the forward-ported lineage:

1. ABI/schema/layout compatibility and byte-exact numeric golden vectors;
2. AITER public-wrapper Fprop/DGrad/WGrad correctness over the design-spec section 15.3 matrix;
3. P0 sequential correctness and P1 grouped-vs-P0 parity;
4. typed capability/miss, workspace/alignment/alias and no-post-commit-fallback tests;
5. 4L world8 TP8/PP1/EP8/ETP1 grouped strict 100-step training gate with zero fallback and closed ledger;
6. checkpoint load/cache rebuild smoke under the new build.

The M2 manifest must record the new Lumen/AITER commits, gitlink, source tree digests, loaded extension path/SHA256/build-id, ABI/layout IDs, toolchain and `dirty=false`. If the target AITER cannot pass the frozen contract, stop and either fix it in focused commits or publish a reviewed manifest amendment; do not silently certify the older implementation lineage.

M2 exit freezes the exact release candidate used by final T0, C0, I0, QG and B1 execution. T0 must first freeze the certified route-to-config mapping consumed by exact C0 and formal I0; C0/I0 then gate QG/B1. No formal tuning, replay/reshard, stability, quality or performance evidence collected before this freeze can certify the release candidate.

---

## 15. QG — Pre-registered quality program

### QG.1 Pilot

Use independent historical/pilot seeds to estimate:

- paired NLL-difference variance;
- validation sampling noise;
- expected alternative mean;
- required n for each margin.

Freeze before formal data:

~~~text
claim cells
seed-generation rule and list
n >= 5
power >= 80%, target 90%
fixed cumulative non-padding-token endpoint
validation corpus/tokenizer/preprocess/order digests
primary and secondary eval paths
margins and per-seed caps
alpha and multiplicity method
infra-invalid criteria
~~~

Pilot data is not pooled into formal results.

### QG.2 Core formal cells

Run BF16/A8W4/A4W4 paired arms for:

1. 4L cold pretraining;
2. 4L warm-start continued training;
3. every 43L cell for which a public quality claim is desired.

Recommended minimum horizon is 500 matched optimizer updates, aligned by actual non-padding tokens. Endpoint selection is fixed; no best-checkpoint choice.

### QG.3 Evaluation

Primary:

~~~text
token-weighted validation NLL
common deterministic BF16 evaluation
approximately >=1M valid target tokens
pilot-sized to <=0.001 NLL sampling half-width
~~~

Secondary:

~~~text
deterministic native A4/A8 evaluation
~~~

### QG.4 Statistical decision

For paired seed s:

~~~text
d_s = endpoint NLL difference
U95 = mean(d_s) + t_(0.95,n-1) * sd(d_s) / sqrt(n)
~~~

All must pass:

~~~text
U95(A4-BF16) <= 0.010
U95(A8-BF16) <= 0.010
U95(A4-A8)   <= 0.005
~~~

Per-seed caps:

~~~text
A4-BF16 <= 0.030
A8-BF16 <= 0.030
A4-A8   <= 0.015
~~~

Without demonstrated power, a five-seed pass is only short_horizon_quality_regression_gate. Exact resume inherits quality only when checkpoint-to-endpoint replay is bitwise; all other continuation modes require their own full cell.

---

## 16. B1 — Performance program

### B1.1 Microbenchmarks

For every promoted route bucket:

- call the AITER public wrapper;
- use `n_micro` independent fresh process pairs, minimum 3 and default 5 only when power is sufficient;
- use 20 warmups and at least 200 timed samples per candidate/process;
- balance AB/BA or use ABBA/BAAB/ABBA;
- use process-first hierarchical paired bootstrap;
- retain raw samples and report mean/median/p95/p99/CV;
- record selected kernel/config and all provenance.

Estimate micro and full-step sample counts separately. For `j in {micro,full}`, use an independent pilot to freeze paired-log standard deviation `sigma_log_j`, expected mean and:

~~~text
delta_log_j = expected_mean_log_speedup_j - log(promotion_threshold_j)
n_j = max(3, ceil(((z_0.95 + z_power) * sigma_log_j / delta_log_j)^2))
~~~

Require delta_log_j > 0, one-sided alpha=0.05, power >=80% and target 90%. `n_micro` counts fresh process pairs; `n_full` counts fresh distributed paired replicate blocks. Pilot measurements are excluded from formal intervals.

### B1.2 Lumen local-block benchmark

Call the real Lumen module/API and time:

~~~text
hot cache lookup
FC1 quant + grouped GEMM
existing BF16 clamp/SwiGLU/probability
FC2 quant + grouped GEMM
~~~

Do not include router/dispatch/combine in this boundary. Report cold cache separately.

For every bootstrap draw, resample independent capture-group/process pairs first and contiguous timed blocks second; recompute route weights, per-route paired medians, weighted latency/speedup and weighted-route p90 ratio from that same draw. Use the 5th percentile for the speedup LCB and the 95th percentile for the p90-ratio UCB. Ranks, routes and repeated iterations are not independent top-level units.

### B1.3 Full-step benchmark

For each comparison:

- use `n_full` fresh distributed paired replicate blocks, minimum 3 and default 5 only when the B1.1 power calculation is sufficient;
- each block uses three fresh distributed launches in the mandatory order `C_before -> V_candidate -> C_after`;
- all three launches start from the same byte-identical BF16 model/FP32 master/optimizer/scheduler/scaler/RNG/data-cursor checkpoint, use the same seed/data slice, consume the same fixed warmup-step count and warmup data prefix, and execute the same fixed timed-step count;
- derived caches are absent from that checkpoint and each variant rebuilds them under its own frozen contract;
- at least 20 warmup optimizer steps;
- at least 100 consecutive timed optimizer steps;
- max-rank wall time and actual non-padding tokens/s;
- no eval/checkpoint/profiler/JIT/probe inside the window;
- no deleted steps;
- retain per-rank raw samples and aligned per-step vectors;
- use only the pre-registered two-level paired moving-block bootstrap defined below; do not choose a t interval after seeing results.

Any candidate/control fallback, skip, OOM, NaN/Inf or kernel/device failure fails the formal replicate. Any unregistered backend/precision/config change also fails the block. A current-FP8 run containing fallback is retained only as a deployment diagnostic and is excluded from formal intervals.

For every bracketed control pair:

~~~text
control_drift = abs(C_after-C_before) / ((C_after+C_before)/2)
~~~

Drift above 2% invalidates the paired block as an environment failure and the artifact remains retained.

For throughput/rate, freeze the estimator:

~~~text
C_hat_i = sqrt(C_before_i * C_after_i)
d_i = log(V_candidate_i) - log(C_hat_i)
speedup_i = exp(d_i)
~~~

Use 10,000 bootstrap draws with a pre-registered seed rule. Resample paired replicate blocks first. Inside each selected block, draw one contiguous circular step-block index sequence and apply that same sequence to C_before, V_candidate, C_after and every rank; ranks are one joint distributed observation and are never resampled independently. Recompute max-rank window elapsed, non-padding tokens/s, C_hat and mean paired log-speedup per draw. The one-sided LCB/UCB is the 5th/95th percentile after exponentiating the log-domain distribution. The pilot freezes block length, `n_full` and all resampling choices before formal data.

### B1.4 Promotion gates

~~~text
weighted_latency(v) = sum(route_frequency_r * median_latency(v,r))
local speedup = weighted_latency(P0) / weighted_latency(P1)
real-route local block speedup LCB >= 1.20x
4L full-step P1/P0 LCB          >= 1.10x
43L full-step P1/P0 LCB         >= 1.10x
43L full-step P1/BF16 LCB       >= 1.05x
4L P1/current-FP8 LCB           >= 1.00x, when a zero-fallback formal control exists
43L P1/current-FP8 LCB          >= 1.00x, when a zero-fallback formal control exists
A4/matched-A8 LCB if claimed    >= 1.00x
Q0.9(route latency P1) / Q0.9(route latency P0) UCB <= 1.05
control replicate drift         <= 2%
physical HBM headroom           >= 10%
~~~

Variant contracts are explicit:

- P1 candidate: grouped, strict, require-certified-bucket, backend fallback=0 and precision fallback=0;
- P0 control: sequential, strict, zero fallback; grouped tuning policy is not applicable;
- BF16 control: fixed BF16 grouped backend/config with no undeclared backend substitution;
- current blockwise-FP8 formal control: available only when the frozen workload runs under an explicit strict policy with backend fallback=0 and precision fallback=0;
- current deployed blockwise-FP8 with any fallback: truthful per-pass diagnostic only, excluded from promotion intervals and gates;
- matched-A8 diagnostic: grouped and strict zero-fallback when used for a formal ratio, otherwise diagnostic-only.

Every promoted ratio therefore has a strict grouped zero-fallback P1 candidate and a zero-fallback control. Controls follow their own fixed contract and preserve complete provenance; they are not mislabeled as grouped MXFP4. If no zero-fallback current-FP8 control exists, its comparison is marked unavailable for formal promotion and the deployed-path measurement remains diagnostic-only.

---

## 17. Planned commit series

Exact filenames must be revalidated after M0; use rg before creating new modules. The series below defines responsibilities, not permission to copy old layouts blindly.

### AITER series

1. **Prerequisite commits**
   - minimal shared numeric/layout/helper fixes;
   - each independently tested and DCO-signed.
2. **ABI commit**
   - public _v1 schemas, typed capability/errors, workspace query;
   - no performance kernel required yet.
3. **Quant/cache builder commit**
   - canonical and prepared layouts, golden/parity tests, benchmark.
4. **Grouped Fprop commit**
   - wrapper, kernel, tests, benchmark/config.
5. **Grouped DGrad commit**
   - wrapper, kernel, tests, benchmark/config.
6. **Grouped eager WGrad commit**
   - output modes, direct FP32 accumulation, tests, benchmark/config.
7. **Grouped matched-scale A8W4 control commit**
   - shared selector scales, A8 F/D and A8xA8 WGrad, parity tests.
8. **Routing/tuning commit**
   - seghist-v1 builder, strict loader, bucketizer, artifact sidecar.
9. **Certification-data commit**
   - only measured, reviewed rows; no heuristic rows labeled tuned.

### Lumen series

1. **Prerequisite commits**
   - only the audited reusable dirty changes.
2. **State foundations**
   - committed weight epoch, cache generation/budget allocator.
3. **RNG and ledger**
   - Philox domains, SR tokens, obligation state and checkpoint frontier.
4. **P0 sequential integration**
   - CLI, module/autograd, strict reference, tests.
5. **A8W4 matched-scale control**
   - shared selector/scale override, full F/D/W control path.
6. **P1 AITER adapter**
   - probes, four-stage facade, transactional bind, late attachment.
7. **Checkpoint modes**
   - warm/exact/state-complete/reshard and sidecar.
8. **Megatron integration tests**
   - EP8/EP4, recompute, main-grad, router semantics.
9. **Benchmarks and artifact tooling**
   - route capture, local block, full-step, manifests and reports.
10. **Documentation**
    - user flags, supported cells, limitations and exact claims.

Keep AITER and Lumen commits separate. A Lumen commit consuming a new AITER ABI must name the required AITER commit/build in its body and manifest.

---

## 18. Test and evidence ladder

Do not skip levels:

~~~text
L0 static/schema tests
L1 numeric golden vectors
L2 AITER operator correctness
L3 P0 sequential correctness
L4 P1 grouped-vs-P0 parity
L5 Lumen lifecycle/dispatch/autograd
L6 single-rank module integration
L7 EP8 and EP4 distributed stability
L8 resume/reshard conformance
L9 powered quality
L10 micro/local-block performance
L11 full-step performance
~~~

Every level records:

~~~text
command
exit status
pass/fail/skip counts
hardware/software manifest
selected backend/config
fallback and ledger summary
artifact path and digest
known limitations
~~~

A skipped GPU test is not a pass. Any formal P0/P1/BF16/matched-A8/current-FP8 result with nonzero runtime fallback is invalid. A deployed current-FP8 run with fallback is diagnostic-only.

---

## 19. Artifact layout

Store generated evidence outside source control under a manifest-addressed root:

~~~text
<artifact-root>/<execution-manifest-digest>/
  manifest/
  prerequisites/
  correctness/
  routing/
  tuning/
  checkpoint/
  quality/
  performance/
  ledgers/
  logs/
~~~

Each report references immutable digests rather than mutable path names. Tuning, validation and final certification route sets are separate. Formal attempts are append-only; failed quality/performance attempts are retained.

---

## 20. Stop and rollback rules

Stop the current stage immediately when:

- a numeric golden vector or P0 parity fails;
- the first localized mismatch points to an AITER op;
- weight/routing epoch or RNG token mismatches;
- bind performs a hidden allocation/lookup/JIT;
- a post-commit path attempts fallback;
- a strict run has any fallback or open ledger entry;
- NaN/Inf, OOM, illegal memory access or async GPU fault occurs;
- quality crosses a per-seed cap;
- performance control drift exceeds 2%;
- runtime source/build identity differs from the manifest.

Preserve the first failing artifact. Localize in order:

~~~text
manifest/config/data diff
-> numeric/operand parity
-> kernel pass
-> Lumen integration
-> layerwise forward/backward
-> distributed/end-to-end
~~~

Do not change an already aligned LR, warmup, clip, scaling, batch or optimizer setting to make a failing run green.

---

## 21. Final handoff checklist

Before declaring implementation complete:

- [ ] M0 clean lineage and pinned runtime are reproducible.
- [ ] Spec IDs and ABI versions match code and artifacts.
- [ ] P0 sequential and P1 grouped use the same numeric recipe.
- [ ] AITER kernels have public wrappers, repr, tests, benchmarks and configs.
- [ ] Lumen owns no new GPU kernel.
- [ ] Direct FP32 main-grad WGrad is verified.
- [ ] Deferred WGrad is rejected.
- [ ] Backend/strict/tuning state machine is fully tested.
- [ ] Cache budgets and allocator quotas are enforced.
- [ ] Route buckets are held-out certified without nearest matching.
- [ ] exact/state-complete/reshard naming matches evidence.
- [ ] Quality claims match only powered, tested cells.
- [ ] Performance claims use fresh-process paired evidence.
- [ ] Every formal P0/P1/BF16/matched-A8/current-FP8 run is zero-fallback; any deployed current-FP8 fallback result is labeled diagnostic-only.
- [ ] Documentation states S1024 A4W4 is not DSV4-report-equivalent.
- [ ] mxfp4-moe contains the reviewed Lumen series and pins the required AITER revision.

No PR is created by this plan. Publishing or merging later remains a separate action.
