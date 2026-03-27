###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fast runner wiring tests for the TRL integration."""

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LUMEN_DIR = REPO_ROOT / "lumen"
RL_DIR = LUMEN_DIR / "rl"
TRL_DIR = RL_DIR / "trl"


class _FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.padding_side = "right"


class _FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        tokenizer = _FakeTokenizer()
        cls.calls.append((model_name_or_path, tokenizer))
        return tokenizer


class _DummyTrainer:
    last_kwargs = None

    def __init__(self, *args, **kwargs):
        _DummyTrainer.last_kwargs = kwargs

    def train(self):
        return None


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_runner_module(events):
    _FakeAutoTokenizer.calls = []
    _DummyTrainer.last_kwargs = None

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)
    sys.modules["torch"] = torch_mod

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = _FakeAutoTokenizer
    sys.modules["transformers"] = transformers_mod

    def _grpo_config(**kwargs):
        events.append(("config", kwargs))
        return kwargs

    class _Trainer(_DummyTrainer):
        def __init__(self, *args, **kwargs):
            events.append("trainer_init")
            super().__init__(*args, **kwargs)

        def train(self):
            events.append("train")
            return None

    trl_mod = types.ModuleType("trl")
    trl_mod.GRPOConfig = _grpo_config
    trl_mod.GRPOTrainer = _Trainer
    sys.modules["trl"] = trl_mod

    args_mod = types.ModuleType("lumen.rl.trl.args")
    args_mod.build_grpo_config_kwargs = lambda args: {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "beta": 0.0,
    }
    sys.modules["lumen.rl.trl.args"] = args_mod

    modeling_mod = types.ModuleType("lumen.rl.trl.modeling")
    modeling_mod.build_actor_model = lambda args: events.append("build_model") or {"actor_model": True}
    sys.modules["lumen.rl.trl.modeling"] = modeling_mod

    warmup_mod = types.ModuleType("lumen.rl.trl.warmup")
    warmup_mod.maybe_run_synthetic_warmup = lambda model, args, *, device: events.append(("warmup", device))
    sys.modules["lumen.rl.trl.warmup"] = warmup_mod

    lumen_pkg = sys.modules.setdefault("lumen", types.ModuleType("lumen"))
    lumen_pkg.__path__ = [str(LUMEN_DIR)]

    rl_pkg = sys.modules.setdefault("lumen.rl", types.ModuleType("lumen.rl"))
    rl_pkg.__path__ = [str(RL_DIR)]

    trl_pkg = sys.modules.setdefault("lumen.rl.trl", types.ModuleType("lumen.rl.trl"))
    trl_pkg.__path__ = [str(TRL_DIR)]

    return _load_module("lumen.rl.trl.runner", TRL_DIR / "runner.py")


class TestTrlRunner:
    def test_run_grpo_wires_model_warmup_and_trainer(self):
        events = []
        mod = _load_runner_module(events)
        reward_fn = lambda **kwargs: [0.0]  # noqa: E731
        args = types.SimpleNamespace(
            model_name_or_path="meta-llama/Llama-2-70b-hf",
            tokenizer_name_or_path=None,
            output_dir="out",
            max_steps=2,
            train_dataset=[{"prompt": "hello"}],
        )

        trainer = mod.run_grpo(args, reward_fn=reward_fn)

        assert events == [
            "build_model",
            ("config", {"output_dir": "out", "max_steps": 2, "beta": 0.0}),
            "trainer_init",
            ("warmup", "cuda:0"),
            "train",
        ]
        tokenizer = _FakeAutoTokenizer.calls[0][1]
        assert tokenizer.padding_side == "left"
        assert tokenizer.pad_token == tokenizer.eos_token
        assert _DummyTrainer.last_kwargs["reward_funcs"] is reward_fn
        assert _DummyTrainer.last_kwargs["args"]["beta"] == 0.0
        assert _DummyTrainer.last_kwargs["train_dataset"] == [{"prompt": "hello"}]
        assert trainer is not None


@pytest.mark.skipif(
    os.environ.get("LUMEN_RUN_SLOW_RL_TESTS") != "1",
    reason="set LUMEN_RUN_SLOW_RL_TESTS=1 to run distributed RL smoke tests",
)
def test_grpo_fsdp1_example_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(repo_root / "examples/rl/trl/accelerate/fsdp1.yaml"),
        str(repo_root / "examples/rl/trl/run_grpo_fsdp.py"),
        "--model-name-or-path",
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "--dataset-name",
        "trl-lib/Capybara",
        "--output-dir",
        str(tmp_path / "out"),
        "--max-steps",
        "2",
        "--micro-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
    ]

    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        check=False,
    )

    assert completed.returncode == 0
