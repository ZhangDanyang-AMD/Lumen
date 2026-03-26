###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Source-level parity tests for the Lumen shell launchers."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

LLAMA31_LAUNCHER = REPO_ROOT / "examples" / "llama31" / "run_pretrain.sh"
LLAMA2_LAUNCHER = REPO_ROOT / "examples" / "llama2" / "run_finetune.sh"
LLAMA2_CONFIG = REPO_ROOT / "examples" / "llama2" / "config_MI355X_1x8x1.sh"
LLAMA2_MI300X_CONFIG = REPO_ROOT / "examples" / "llama2" / "config_MI300X_1x8x1.sh"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _between(text: str, start_marker: str, end_marker: str) -> str:
    return text.split(start_marker, 1)[1].split(end_marker, 1)[0]


class TestLauncherParity:
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

    def test_llama2_mi300x_config_uses_eval_every_contract(self):
        source = _source(LLAMA2_MI300X_CONFIG)
        assert "export EVAL_EVERY=" in source
