###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the TRL synthetic warmup helper."""

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LUMEN_DIR = REPO_ROOT / "lumen"
RL_DIR = LUMEN_DIR / "rl"
TRL_DIR = RL_DIR / "trl"


class _FakeTensor:
    def __init__(self, *, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device


class _FakeLoss:
    def __init__(self, events):
        self._events = events

    def backward(self):
        self._events.append("backward")


class _FakeWarmupModel:
    def __init__(self):
        self.events = []

    def parameters(self):
        return [object()]

    def __call__(self, *, input_ids, labels):
        self.events.append(("forward", input_ids.device, labels.device))
        return types.SimpleNamespace(loss=_FakeLoss(self.events))


class _FakeAdamW:
    instances = []

    def __init__(self, params, *, lr, betas, eps, weight_decay):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.zero_grad_calls = 0
        self.step_calls = 0
        _FakeAdamW.instances.append(self)

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_warmup_module(reset_events):
    _FakeAdamW.instances = []

    def _ones(*shape, dtype, device):
        return _FakeTensor(shape=shape, dtype=dtype, device=device)

    torch_mod = types.ModuleType("torch")
    torch_mod.long = "long"
    torch_mod.ones = _ones
    torch_mod.optim = types.SimpleNamespace(AdamW=_FakeAdamW)
    sys.modules["torch"] = torch_mod

    def _reset_fp8_state(model):
        reset_events.append(model)

    fsdp_mod = types.ModuleType("lumen.models.fsdp")
    fsdp_mod.reset_fp8_state = _reset_fp8_state
    sys.modules["lumen.models.fsdp"] = fsdp_mod

    lumen_pkg = sys.modules.setdefault("lumen", types.ModuleType("lumen"))
    lumen_pkg.__path__ = [str(LUMEN_DIR)]

    models_pkg = sys.modules.setdefault("lumen.models", types.ModuleType("lumen.models"))
    models_pkg.__path__ = [str(LUMEN_DIR / "models")]

    rl_pkg = sys.modules.setdefault("lumen.rl", types.ModuleType("lumen.rl"))
    rl_pkg.__path__ = [str(RL_DIR)]

    trl_pkg = sys.modules.setdefault("lumen.rl.trl", types.ModuleType("lumen.rl.trl"))
    trl_pkg.__path__ = [str(TRL_DIR)]

    return _load_module("lumen.rl.trl.warmup", TRL_DIR / "warmup.py")


class TestSyntheticWarmup:
    def test_runs_zero_lr_warmup_and_resets_fp8(self):
        reset_events = []
        mod = _load_warmup_module(reset_events)
        model = _FakeWarmupModel()
        args = types.SimpleNamespace(
            warmup_steps=2,
            micro_batch_size=1,
            seq_length=16,
            linear_fp8=True,
        )

        steps = mod.maybe_run_synthetic_warmup(model, args, device="cuda:3")

        assert steps == 2
        assert model.events == [
            ("forward", "cuda:3", "cuda:3"),
            "backward",
            ("forward", "cuda:3", "cuda:3"),
            "backward",
        ]
        assert len(_FakeAdamW.instances) == 1
        optimizer = _FakeAdamW.instances[0]
        assert optimizer.lr == 0.0
        assert optimizer.zero_grad_calls == 2
        assert optimizer.step_calls == 2
        assert reset_events == [model]

    def test_skips_when_warmup_disabled(self):
        reset_events = []
        mod = _load_warmup_module(reset_events)
        model = _FakeWarmupModel()
        args = types.SimpleNamespace(
            warmup_steps=0,
            micro_batch_size=1,
            seq_length=16,
            linear_fp8=True,
        )

        steps = mod.maybe_run_synthetic_warmup(model, args, device="cuda:0")

        assert steps == 0
        assert model.events == []
        assert _FakeAdamW.instances == []
        assert reset_events == []
