###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################
"""Tests for FSDP2 distributed training integration.

How to run::

    # All tests (unit + distributed); distributed tests require >= 2 GPUs:
    pytest tests/models/test_fsdp2.py -v

    # Distributed FSDP2 training test (launches torchrun --nproc_per_node=2 internally):
    pytest tests/models/test_fsdp2.py -v -k "dist"
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import pytest  # noqa: F401  # Task 2: @pytest.mark.parametrize
import torch
import torch.nn as nn

_DIST = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2+ GPUs required",
)

_BF16_TRAIN_SCRIPT = textwrap.dedent(
    """\
    import argparse
    import os
    import torch
    import torch.distributed as dist
    from transformers import LlamaConfig, LlamaForCausalLM

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(42)

    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        vocab_size=256,
    )
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).cuda()

    args = argparse.Namespace(
        linear_fp8=False,
        sharding_strategy="full_shard",
    )
    from lumen.models.fsdp import apply_fsdp2
    apply_fsdp2(model, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 256, (2, 16), device="cuda")
    labels = input_ids.clone()

    try:
        losses = []
        for step in range(10):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        if rank == 0:
            assert losses[-1] < losses[0] * 0.8, (
                f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
            )
            print(f"PASS: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
"""
)

_FP8_TRAIN_SCRIPT = textwrap.dedent(
    """\
    import argparse
    import os
    import torch
    import torch.distributed as dist
    from transformers import LlamaConfig, LlamaForCausalLM

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(42)

    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        vocab_size=256,
    )
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).cuda()

    args = argparse.Namespace(
        linear_fp8=True,
        sharding_strategy="full_shard",
    )
    from lumen.models.fsdp import apply_fp8_training, apply_fsdp2
    apply_fp8_training(model, args)
    apply_fsdp2(model, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 256, (2, 16), device="cuda")
    labels = input_ids.clone()

    try:
        losses = []
        for step in range(10):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        if rank == 0:
            assert losses[-1] < losses[0] * 0.9, (
                f"FP8 loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
            )
            print(f"PASS: FP8 loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
"""
)


class TestFSDP2CLIArgs:

    def test_fsdp_version_default(self):
        from lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args([])
        assert args.fsdp_version == 1

    def test_fsdp_version_2(self):
        from lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args(["--fsdp-version", "2"])
        assert args.fsdp_version == 2


class TestApplyFSDP2:

    def test_function_exists(self):
        from lumen.models.fsdp import apply_fsdp2

        assert callable(apply_fsdp2)

    def test_fully_shard_receives_device_mesh(self):
        from lumen.models.fsdp import apply_fsdp2

        model = nn.Sequential(nn.Linear(8, 4))
        args = argparse.Namespace(
            linear_fp8=False,
            sharding_strategy="full_shard",
        )

        with patch("torch.distributed.fsdp.fully_shard") as mock_fs, patch(
            "torch.distributed.device_mesh.init_device_mesh"
        ) as mock_mesh, patch("lumen.models.fsdp.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            mock_mesh.return_value = MagicMock()
            mock_fs.side_effect = lambda m, **kw: m
            apply_fsdp2(model, args)

            for call in mock_fs.call_args_list:
                mesh_arg = call.kwargs.get("mesh")
                assert mesh_arg is mock_mesh.return_value

    def test_no_shard_raises_with_fsdp2(self):
        from lumen.models.fsdp import apply_fsdp2

        model = nn.Linear(8, 4)
        args = argparse.Namespace(
            linear_fp8=False,
            sharding_strategy="no_shard",
        )
        with pytest.raises(ValueError, match="no_shard is not supported"):
            apply_fsdp2(model, args)

    @pytest.mark.parametrize(
        "strategy,expected_reshard",
        [
            ("full_shard", True),
            ("shard_grad_op", False),
        ],
    )
    def test_reshard_after_forward_mapping(self, strategy, expected_reshard):
        from lumen.models.fsdp import apply_fsdp2

        model = nn.Sequential(nn.Linear(8, 4))
        args = argparse.Namespace(
            linear_fp8=False,
            sharding_strategy=strategy,
        )

        with patch("torch.distributed.fsdp.fully_shard") as mock_fs, patch(
            "torch.distributed.device_mesh.init_device_mesh"
        ) as mock_mesh, patch("lumen.models.fsdp.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            mock_mesh.return_value = MagicMock()
            mock_fs.side_effect = lambda m, **kw: m
            apply_fsdp2(model, args)

            for c in mock_fs.call_args_list:
                assert c.kwargs["reshard_after_forward"] == expected_reshard

    def test_shards_each_decoder_layer(self):
        from lumen.models.fsdp import apply_fsdp2

        layer0 = nn.Linear(8, 8)
        layer1 = nn.Linear(8, 8)
        inner = nn.Module()
        inner.layers = nn.ModuleList([layer0, layer1])
        model = nn.Module()
        model.add_module("model", inner)
        model.add_module("lm_head", nn.Linear(8, 4))

        args = argparse.Namespace(
            linear_fp8=False,
            sharding_strategy="full_shard",
        )

        with patch("torch.distributed.fsdp.fully_shard") as mock_fs, patch(
            "torch.distributed.device_mesh.init_device_mesh"
        ) as mock_mesh, patch("lumen.models.fsdp.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            mock_mesh.return_value = MagicMock()
            mock_fs.side_effect = lambda m, **kw: m
            apply_fsdp2(model, args)

            assert mock_fs.call_count == 3
            sharded_modules = [c.args[0] for c in mock_fs.call_args_list]
            assert layer0 in sharded_modules
            assert layer1 in sharded_modules
            assert model in sharded_modules


class TestResetFp8StateFSDP2:

    def test_clears_fp8_flags_without_module_wrapper(self):
        """reset_fp8_state finds FP8 modules in FSDP2 (no .module wrapper)."""
        from lumen.models.fsdp import reset_fp8_state

        layer = nn.Linear(8, 4)
        layer.fp8_initialized = True
        model = nn.Sequential(layer)
        model.fp8_initialized = True

        reset_fp8_state(model)

        assert not getattr(model, "fp8_initialized", False)
        assert not getattr(layer, "fp8_initialized", False)


@_DIST
class TestFSDP2Integration:

    @staticmethod
    def _get_free_port():
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _run_training_script(self, script: str, timeout: int = 120):
        port = str(self._get_free_port())
        env = os.environ.copy()
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = port
        kwargs = dict(
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if sys.platform != "win32":
            kwargs["start_new_session"] = True

        fd, script_path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(script)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node=2",
                    "--master_addr=127.0.0.1",
                    f"--master_port={port}",
                    script_path,
                ],
                **kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            if sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(exc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            stdout = (exc.stdout or "")[:2000]
            stderr = (exc.stderr or "")[:2000]
            pytest.fail(
                f"Training script timed out after {timeout}s " f"(port {port}).\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        finally:
            os.unlink(script_path)
        return result

    def test_bf16_fsdp2_overfit(self):
        """2-GPU LLaMA mini BF16 + FSDP2: loss decreases over 10 steps."""
        result = self._run_training_script(_BF16_TRAIN_SCRIPT)
        assert result.returncode == 0, (
            f"Training failed (rc={result.returncode}):\n" f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_fp8_fsdp2_overfit(self):
        """2-GPU LLaMA mini FP8 + FSDP2: loss decreases over 10 steps."""
        result = self._run_training_script(_FP8_TRAIN_SCRIPT)
        assert result.returncode == 0, (
            f"FP8 training failed (rc={result.returncode}):\n" f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
