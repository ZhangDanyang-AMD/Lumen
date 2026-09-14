# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the Qwen3 FSDP full-pretraining entrypoint."""

import os
import subprocess
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.modeling_layers import (
    GradientCheckpointingLayer as _GRAD_CKPT_LAYER,
)

from examples.qwen3.train_qwen3_fsdp import (
    _apply_selective_grad_checkpointing,
    _pretrain_loss,
    _select_recompute_layers,
    _set_fsdp2_accumulation_state,
    _set_fsdp2_gradient_sync,
    _set_fsdp2_reshard_after_backward,
    _set_fsdp2_reshard_after_forward,
    _validation_batch_count,
    parse_args,
)


_BASE = [
    "--model-name-or-path",
    "Qwen/Qwen3-8B",
    "--train-data-path",
    "train.jsonl",
]


def _fake_torchrun_env(tmp_path):
    """Run the launcher against a torchrun that just echoes its arguments."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@"\n',
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)

    launcher = (
        Path(__file__).resolve().parents[2]
        / "examples/qwen3/run_qwen3_fsdp_mxfp4_pretrain.sh"
    )
    env = os.environ.copy()
    env.update(
        MODEL_PATH=str(tmp_path / "model"),
        TRAIN_DATA_PATH=str(tmp_path / "train.jsonl"),
        RESULTS_DIR=str(tmp_path / "results"),
        NPROC="1",
        MBS="1",
        GBS="1",
        TRAIN_STEPS="1",
        PATH=f"{fake_bin}:{env['PATH']}",
    )
    return env, launcher


@pytest.mark.parametrize(
    ("mxfp4_comm", "expect_compressed_comm"),
    [(None, False), ("1", True)],
)
def test_mxfp4_launcher_comm_compression_is_opt_in(
    tmp_path, mxfp4_comm, expect_compressed_comm
):
    env, launcher = _fake_torchrun_env(tmp_path)
    if mxfp4_comm is None:
        env.pop("MXFP4_COMM", None)
    else:
        env["MXFP4_COMM"] = mxfp4_comm

    result = subprocess.run(
        ["bash", str(launcher)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert ("--fsdp-mxfp4-comm" in args) is expect_compressed_comm


@pytest.mark.parametrize(
    ("retain", "gbs", "expect_flag", "expect_warning"),
    [
        (None, "16", False, False),
        ("1", "16", True, False),
        # Retention has nothing to save without accumulation, so the launcher
        # must not quietly pass a flag that changes memory for no benefit.
        ("1", "1", False, True),
    ],
)
def test_launcher_retains_accumulated_params_only_when_accumulating(
    tmp_path, retain, gbs, expect_flag, expect_warning
):
    env, launcher = _fake_torchrun_env(tmp_path)
    env["GBS"] = gbs
    env["MBS"] = "1"
    env["NPROC"] = "1"
    if retain is None:
        env.pop("RETAIN_ACCUM_PARAMS", None)
    else:
        env["RETAIN_ACCUM_PARAMS"] = retain

    result = subprocess.run(
        ["bash", str(launcher)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert ("--fsdp-retain-accumulated-params" in args) is expect_flag
    assert ("RETAIN_ACCUM_PARAMS ignored" in result.stderr) is expect_warning


def test_pretrain_mxfp4_defaults_to_full_parameters_and_bf16_tail():
    args = parse_args(_BASE + ["--task", "pretrain", "--mode", "mxfp4"])

    assert args.lora_rank == 0
    assert args.linear_fp4 is True
    assert args.linear_fp8 is False
    assert args.first_last_layers_bf16 is True
    assert args.num_layers_at_start_in_bf16 == 0
    assert args.num_layers_at_end_in_bf16 == 5


def test_sft_defaults_remain_lora_and_no_bf16_tail():
    args = parse_args(_BASE)

    assert args.task == "sft"
    assert args.lora_rank == 16
    assert args.first_last_layers_bf16 is False


def test_pretrain_rejects_lora():
    with pytest.raises(ValueError, match="requires --lora-rank 0"):
        parse_args(_BASE + ["--task", "pretrain", "--lora-rank", "16"])


def test_init_from_scratch_is_pretrain_only():
    with pytest.raises(ValueError, match="only valid with --task pretrain"):
        parse_args(_BASE + ["--init-from-scratch"])


def test_fsdp2_accumulation_sync_recurses():
    model = MagicMock()

    _set_fsdp2_gradient_sync(model, False)
    _set_fsdp2_gradient_sync(model, True)

    assert model.set_requires_gradient_sync.call_args_list[0].args == (False,)
    assert model.set_requires_gradient_sync.call_args_list[0].kwargs == {"recurse": True}
    assert model.set_requires_gradient_sync.call_args_list[1].args == (True,)


def test_fsdp2_reshard_after_backward_recurses():
    model = MagicMock()

    _set_fsdp2_reshard_after_backward(model, False)

    model.set_reshard_after_backward.assert_called_once_with(False, recurse=True)


def test_fsdp2_reshard_after_forward_skips_the_root_module():
    root = MagicMock()
    child_a, child_b = MagicMock(), MagicMock()
    plain = SimpleNamespace()  # a non-FSDP submodule must be ignored
    root.modules.return_value = [root, child_a, plain, child_b]

    _set_fsdp2_reshard_after_forward(root, False)

    root.set_reshard_after_forward.assert_not_called()
    child_a.set_reshard_after_forward.assert_called_once_with(False, recurse=False)
    child_b.set_reshard_after_forward.assert_called_once_with(False, recurse=False)


def test_fsdp2_reshard_after_forward_requires_a_wrapped_submodule():
    root = MagicMock()
    root.modules.return_value = [root, SimpleNamespace()]

    with pytest.raises(RuntimeError, match="set_reshard_after_forward"):
        _set_fsdp2_reshard_after_forward(root, False)


def _accumulation_calls(model, name):
    return [(c.args, c.kwargs) for c in getattr(model, name).call_args_list]


def test_fsdp2_retained_params_are_freed_only_on_the_final_microbatch():
    model = MagicMock()
    child = MagicMock()
    model.modules.return_value = [model, child]

    for micro in range(3):
        _set_fsdp2_accumulation_state(
            model,
            final_micro=micro == 2,
            retain_params=True,
            reshard_after_forward=True,
        )

    assert _accumulation_calls(model, "set_requires_gradient_sync") == [
        ((False,), {"recurse": True}),
        ((False,), {"recurse": True}),
        ((True,), {"recurse": True}),
    ]
    assert _accumulation_calls(model, "set_reshard_after_backward") == [
        ((False,), {"recurse": True}),
        ((False,), {"recurse": True}),
        ((True,), {"recurse": True}),
    ]
    # The wrapped policy is restored on the final micro-batch, not assumed.
    assert _accumulation_calls(child, "set_reshard_after_forward") == [
        ((False,), {"recurse": False}),
        ((False,), {"recurse": False}),
        ((True,), {"recurse": False}),
    ]


def test_fsdp2_retention_restores_shard_grad_op_policy():
    model = MagicMock()
    child = MagicMock()
    model.modules.return_value = [model, child]

    _set_fsdp2_accumulation_state(
        model, final_micro=True, retain_params=True, reshard_after_forward=False
    )

    # shard_grad_op never reshards after forward; retention must not turn that
    # into full_shard behaviour on the final micro-batch.
    child.set_reshard_after_forward.assert_called_once_with(False, recurse=False)


def test_fsdp2_accumulation_leaves_lifetime_untouched_by_default():
    model = MagicMock()
    model.modules.return_value = [model, MagicMock()]

    _set_fsdp2_accumulation_state(
        model, final_micro=False, retain_params=False, reshard_after_forward=True
    )

    model.set_requires_gradient_sync.assert_called_once_with(False, recurse=True)
    model.set_reshard_after_backward.assert_not_called()


def test_fsdp2_retain_accumulated_params_requires_fsdp2():
    with pytest.raises(ValueError, match="requires --fsdp-version 2"):
        parse_args(_BASE + ["--fsdp-retain-accumulated-params"])

    args = parse_args(
        _BASE + ["--fsdp-version", "2", "--fsdp-retain-accumulated-params"]
    )
    assert args.fsdp_retain_accumulated_params is True


@pytest.mark.parametrize(
    ("num_layers", "requested", "expected"),
    [
        (36, 36, list(range(36))),
        (36, 40, list(range(36))),
        (36, 0, []),
        (36, -1, []),
        # Spread across the depth rather than a prefix: every layer costs the
        # same recompute, so a prefix would only bias where memory is held.
        (36, 9, [0, 4, 8, 12, 16, 20, 24, 28, 32]),
        (36, 1, [0]),
    ],
)
def test_select_recompute_layers_spreads_across_depth(
    num_layers, requested, expected
):
    assert _select_recompute_layers(num_layers, requested) == expected


class _CountingLayer(_GRAD_CKPT_LAYER):
    """A layer that records how many times its forward actually runs."""

    def __init__(self, dim, counter):
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim)
        self.counter = counter

    def forward(self, x):
        self.counter["forwards"] += 1
        return torch.tanh(self.lin(x))


class _CountingStack(torch.nn.Module):
    def __init__(self, dim, depth, counter):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _CountingLayer(dim, counter) for _ in range(depth)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _run_counting_stack(requested, depth=4, dim=8):
    """Train-mode forward/backward with `requested` layers recomputed."""
    torch.manual_seed(0)
    counter = {"forwards": 0}
    model = _CountingStack(dim, depth, counter)
    model.train()
    for layer in model.layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = partial(
            torch.utils.checkpoint.checkpoint, use_reentrant=False
        )
    if requested is not None:
        _apply_selective_grad_checkpointing(model, requested)

    x = torch.randn(2, dim, requires_grad=True)
    model(x).square().sum().backward()
    grads = [layer.lin.weight.grad.clone() for layer in model.layers]
    return counter["forwards"], grads


@pytest.mark.parametrize("requested", [0, 2, 4])
def test_selective_recompute_runs_exactly_the_selected_layers_twice(requested):
    depth = 4
    forwards, grads = _run_counting_stack(requested, depth=depth)

    # One forward per layer, plus one extra for each recomputed layer. This is
    # what fails if the per-layer flag is set but never read, or if the whole
    # stack keeps recomputing.
    assert forwards == depth + requested

    _, reference = _run_counting_stack(0, depth=depth)
    for grad, expected in zip(grads, reference):
        torch.testing.assert_close(grad, expected)


def test_selective_recompute_rejects_layers_without_the_flag():
    model = SimpleNamespace()
    layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    model.modules = lambda: [SimpleNamespace(layers=layers)]

    with pytest.raises(RuntimeError, match="gradient_checkpointing"):
        _apply_selective_grad_checkpointing(model, 1)


def test_selective_recompute_requires_a_transformer_layer_list():
    model = SimpleNamespace(modules=lambda: [SimpleNamespace()])

    with pytest.raises(RuntimeError, match="ModuleList"):
        _apply_selective_grad_checkpointing(model, 1)


def test_grad_checkpoint_layers_conflicts_with_disabled_checkpointing():
    with pytest.raises(ValueError, match="--no-grad-checkpointing"):
        parse_args(
            _BASE + ["--no-grad-checkpointing", "--grad-checkpoint-layers", "9"]
        )

    with pytest.raises(ValueError, match=">= 0"):
        parse_args(_BASE + ["--grad-checkpoint-layers", "-2"])

    assert parse_args(_BASE).grad_checkpoint_layers is None
    assert parse_args(_BASE + ["--grad-checkpoint-layers", "9"]).grad_checkpoint_layers == 9


def test_pretrain_loss_does_not_shift_dataset_labels_twice():
    labels = torch.tensor([[1, 2, 3]])
    logits = torch.full((1, 3, 4), -20.0)
    logits.scatter_(-1, labels.unsqueeze(-1), 20.0)
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "labels": labels,
    }

    loss = _pretrain_loss(model, batch, "cpu")

    assert loss.item() < 1e-6
    model.assert_called_once_with(input_ids=batch["input_ids"])


def test_pretrain_loss_upcasts_bf16_logits_for_cross_entropy():
    logits = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    batch = {
        "input_ids": torch.randint(0, 8, (2, 3)),
        "labels": torch.randint(0, 8, (2, 3)),
    }

    loss = _pretrain_loss(model, batch, "cpu")

    assert loss.dtype == torch.float32


def test_eval_batches_must_be_positive():
    with pytest.raises(ValueError, match="--eval-batches must be >= 1"):
        parse_args(_BASE + ["--eval-batches", "0"])


def test_validation_batch_count_rejects_empty_local_loader():
    with pytest.raises(ValueError, match="no full micro-batch"):
        _validation_batch_count(0, 10, "cpu")


def test_validation_batch_count_uses_global_minimum():
    def set_remote_min(count, op):
        assert op == torch.distributed.ReduceOp.MIN
        count.fill_(3)

    with (
        patch("examples.qwen3.train_qwen3_fsdp.dist.is_initialized", return_value=True),
        patch(
            "examples.qwen3.train_qwen3_fsdp.dist.all_reduce",
            side_effect=set_remote_min,
        ),
    ):
        count = _validation_batch_count(8, 10, "cpu")

    assert count == 3


def test_pretrain_dataset_import_does_not_require_megatron():
    from lumen.models.llama31.dataset import PretrainTextDataset

    assert PretrainTextDataset.__name__ == "PretrainTextDataset"


def test_pretrain_dataset_tokenization_is_sharded_by_input_line(tmp_path):
    from lumen.models.llama31.dataset import PretrainTextDataset

    data_path = tmp_path / "train.txt"
    data_path.write_text("1\n2\n3\n4\n", encoding="utf-8")
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        encode=lambda text, add_special_tokens=False: [int(text)],
    )

    rank0 = PretrainTextDataset(
        str(data_path), seq_length=1, tokenizer=tokenizer,
        is_hf_tokenizer=True, rank=0, world_size=2,
    )
    rank1 = PretrainTextDataset(
        str(data_path), seq_length=1, tokenizer=tokenizer,
        is_hf_tokenizer=True, rank=1, world_size=2,
    )

    assert [rank0[i]["input_ids"].item() for i in range(len(rank0))] == [1, 3]
    assert [rank1[i]["input_ids"].item() for i in range(len(rank1))] == [2, 4]


def test_pretrain_dataset_stops_reading_once_sample_budget_is_met(tmp_path):
    from lumen.models.llama31.dataset import PretrainTextDataset

    data_path = tmp_path / "train.txt"
    data_path.write_text("".join(f"{i}\n" for i in range(1, 1001)), encoding="utf-8")
    seen = []

    def encode(text, add_special_tokens=False):
        seen.append(text)
        return [int(text)]

    tokenizer = SimpleNamespace(eos_token_id=0, encode=encode)

    ds = PretrainTextDataset(
        str(data_path), seq_length=1, tokenizer=tokenizer,
        is_hf_tokenizer=True, max_samples=2,
    )

    assert len(ds) == 2
    # Two samples need 4 tokens; the scan must not walk the whole file.
    assert len(seen) <= 4


def test_train_samples_is_pretrain_only():
    with pytest.raises(ValueError, match="--train-samples"):
        parse_args(_BASE + ["--task", "sft", "--train-samples", "8"])
