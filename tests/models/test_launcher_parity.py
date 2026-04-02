###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Source-level parity tests for the Lumen shell launchers."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

LLAMA2_DOCKERFILE = REPO_ROOT / "examples" / "llama2" / "Dockerfile"
LLAMA31_LAUNCHER = REPO_ROOT / "examples" / "llama31" / "run_pretrain.sh"
LLAMA31_CONFIG = REPO_ROOT / "examples" / "llama31" / "config_MI355X_1x8x1.sh"
LLAMA31_MI300X_CONFIG = REPO_ROOT / "examples" / "llama31" / "config_MI300X_1x8x1.sh"
LLAMA31_DOCKERFILE = REPO_ROOT / "examples" / "llama31" / "Dockerfile"
LLAMA2_LAUNCHER = REPO_ROOT / "examples" / "llama2" / "run_finetune.sh"
LLAMA2_CONFIG = REPO_ROOT / "examples" / "llama2" / "config_MI355X_1x8x1.sh"
LLAMA2_MI300X_CONFIG = REPO_ROOT / "examples" / "llama2" / "config_MI300X_1x8x1.sh"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _between(text: str, start_marker: str, end_marker: str) -> str:
    return text.split(start_marker, 1)[1].split(end_marker, 1)[0]


class TestLauncherParity:
    def test_llama31_mi300x_config_exists_as_overlay(self):
        assert LLAMA31_MI300X_CONFIG.exists()
        source = _source(LLAMA31_MI300X_CONFIG)
        assert 'source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"' in source
        assert 'export MLPERF_SUBMISSION_PLATFORM="MI300X"' in source
        assert 'export LUMEN_ATTN_BACKEND="aiter_csrc"' in source

    def test_llama31_mi300x_config_uses_lf_line_endings(self):
        assert b"\r\n" not in LLAMA31_MI300X_CONFIG.read_bytes()

    def test_example_dockerfiles_align_root_training_stack_contract(self):
        for dockerfile in (LLAMA2_DOCKERFILE, LLAMA31_DOCKERFILE):
            source = _source(dockerfile)
            for snippet in (
                "apt-get update && apt-get install -y --no-install-recommends",
                "libopenmpi-dev libibverbs-dev rdma-core libpci-dev",
                "PREBUILD_KERNELS=1 pip install -e .",
                "third_party/mori",
                "pip install setuptools-scm",
                "pip install torchao>=0.8",
                "--force-reinstall --no-deps",
                "pip install -r /workspace/Lumen/requirements.txt",
                "ldconfig",
            ):
                assert snippet in source

    def test_example_launchers_expose_use_sdma_switch(self):
        for launcher in (LLAMA2_LAUNCHER, LLAMA31_LAUNCHER):
            source = _source(launcher)
            assert 'if [ "${USE_SDMA}" -gt 0 ]; then' in source
            assert 'CMD_SUFFIX="${CMD_SUFFIX} --use-sdma"' in source
            assert 'export MORI_ENABLE_SDMA="${USE_SDMA}"' in source

    def test_llama2_launcher_builds_shared_checkpoint_suffix(self):
        source = _source(LLAMA2_LAUNCHER)
        assert "CMD_SUFFIX" in source
        assert "--use-ckpt" in source
        assert "--resume-from-hf" in source
        assert "--save-ckpt" in source
        assert "--tag" in source
        assert "--fp8-params" in source

    def test_llama2_launcher_computes_eval_interval_from_eval_every(self):
        source = _source(LLAMA2_LAUNCHER)
        assert 'if [ "${EVAL_EVERY}" -gt 0 ] && [ "${EVAL_INTERVAL}" -eq 0 ]; then' in source
        assert "EVAL_INTERVAL=$(( (EVAL_EVERY + GBS - 1) / GBS ))" in source

    def test_llama2_megatron_launcher_emits_shared_experiment_flags(self):
        source = _source(LLAMA2_LAUNCHER)
        megatron_block = _between(
            source,
            "run_megatron() {",
            "###############################################################################\n# FSDP BACKEND",
        )
        for flag in (
            "--seed",
            "--eval-iters ${EVAL_ITERS}",
            "--continual-ckpt-path",
            "--target-log-ppl",
            "--step-time-atol",
            "--ckpt-start-step",
            "--eval-every",
            "--start-eval-at",
        ):
            assert flag in megatron_block

    def test_llama2_fsdp_launcher_emits_shared_contract_flags(self):
        source = _source(LLAMA2_LAUNCHER)
        fsdp_block = _between(
            source,
            "run_fsdp() {",
            "###############################################################################\n# DISPATCH",
        )
        for flag in (
            "--lr-warmup-steps",
            "--continual-ckpt-path",
            "--target-log-ppl",
            "--step-time-atol",
            "--ckpt-start-step",
            "--eval-every",
            "--start-eval-at",
            "--primus-turbo-fp8-attention",
            "--primus-turbo-mxfp8-attention",
            "--dbg-attn-output",
        ):
            assert flag in fsdp_block

    def test_llama2_config_defines_shared_launcher_env_defaults(self):
        source = _source(LLAMA2_CONFIG)
        for env_name in (
            "SEED",
            "EVAL_ITERS",
            "USE_SDMA",
            "LR_WARMUP_STEPS",
            "USE_CKPT",
            "FROM_HF",
            "SAVE_CKPT",
            "CONTINUAL_CKPT",
            "CKPT_START_STEP",
            "FP8_PARAMS",
            "TARGET_LOG_PPL",
            "STEP_TIME_ATOL",
            "EVAL_EVERY",
            "START_EVAL_AT",
            "PRIMUS_FP8_ATTN",
            "PRIMUS_MXFP8_ATTN",
            "DBG_ATTN_OUTPUT",
        ):
            assert f"export {env_name}" in source

    def test_llama2_mi300x_config_is_overlay_with_mi300x_contract(self):
        source = _source(LLAMA2_MI300X_CONFIG)
        assert 'source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"' in source
        assert "export EVAL_EVERY=" in source
        assert 'export LUMEN_ATTN_BACKEND="csrc"' in source
        assert 'export MLPERF_SUBMISSION_PLATFORM="MI300X"' in source

    def test_llama31_config_defines_use_sdma_default(self):
        source = _source(LLAMA31_CONFIG)
        assert "export USE_SDMA=0" in source
