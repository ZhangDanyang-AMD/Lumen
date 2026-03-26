###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Policy checks for the Lumen training stack touched by parity work."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

POLICY_TARGETS = [
    REPO_ROOT / "examples" / "llama31" / "pretrain_llama31.py",
    REPO_ROOT / "examples" / "llama2" / "finetune_llama2.py",
    REPO_ROOT / "lumen" / "models" / "training_contract.py",
    REPO_ROOT / "lumen" / "models" / "experiment_ops.py",
    REPO_ROOT / "lumen" / "models" / "fsdp.py",
    REPO_ROOT / "lumen" / "models" / "llama31" / "fsdp" / "pretrain.py",
    REPO_ROOT / "lumen" / "models" / "llama2" / "fsdp" / "sft.py",
]


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestTrainingStackPolicy:
    def test_policy_targets_have_no_direct_nemo_imports(self):
        pattern = re.compile(r"^\s*(?:from|import)\s+nemo\b", re.MULTILINE)
        for path in POLICY_TARGETS:
            assert pattern.search(_source(path)) is None, f"unexpected nemo import in {path}"

    def test_policy_targets_have_no_direct_transformer_engine_imports(self):
        pattern = re.compile(r"^\s*(?:from|import)\s+transformer_engine\b", re.MULTILINE)
        for path in POLICY_TARGETS:
            assert pattern.search(_source(path)) is None, f"unexpected transformer_engine import in {path}"
