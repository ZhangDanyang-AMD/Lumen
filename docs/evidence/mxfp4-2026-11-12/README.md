# MXFP4 formal-source verification, 2026-11-12

Tested Lumen: `00edddf545fffdba2dcb2d6c9bb00776a1decb27`.
Host: `smci355-ccs-aus-n01-13.cs-aus.dcgpu`, eight MI355X GPUs.
All Lumen/AITER/Megatron tracked sources were clean at startup. The formal
row-12 launcher then applied the recorded Megatron bridge. No ad hoc source
compatibility edits were used. Lumen and AITER remained clean after training.

`summary.json` contains the automatically validated training results.
`manifest.json` pins sources, package versions, runtime and input hashes.
Each case contains `launch.json` and `console.log.gz` (use `gzip -dc` to read).
`gpu-preflight.log` records 13 passing AITER integration tests.
`megatron-launcher.patch` plus `megatron_fused_rmsnorm.py` capture the automatic
source bridge. `mock-data.json` records row-12 generated input provenance.
The raw `.patch` retains Git's blank context lines exactly; these account for
the two whitespace warnings in `git diff --check` on the saved evidence file.

Other checks in this task:

- `python -m unittest discover -s tests -p test_mxfp4_example_compat.py`: 3 passed.
- `python -m pytest -q tests/modules/test_layernorm_linear_stride.py`: 2 passed.
- `bash -n` on the two launchers and shared Megatron body: passed.
- Runner `--dry-run --case all`: passed without GPU imports.
- Data preparer run against the saved original Alpaca response: reproduced
  train/validation hashes in the manifest, 51,490/512 rows; changed response rejected.
- Independent review found stride bias metadata and incomplete launch/runtime
  evidence; all findings fixed and re-reviewed without further blockers.

Full evidence, including generated mock corpus and driver logs, is backed up at
`C:\Users\xdai\lumen-mxfp4-results-20261112\evidence-clean-00edddf.tar.gz`.
Task runtime directory: `/dev/shm/xdai-lumen-mxfp4-rerun-20261112/`.
The saved source pin and reproduction recipe are sufficient to regenerate the
mock corpus; this Git directory omits the raw corpus and compiled caches.

These are 20-step smoke tests. Neither convergence nor paired performance
improvement over earlier results is established. Row 12's validation shares its
training corpus; row 13 performs no validation inside 20 steps. Both logs show
the documented packed transpose fallback for already shuffled MXFP4 input.
