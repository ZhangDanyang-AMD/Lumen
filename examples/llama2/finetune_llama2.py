# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""LLaMA2 Supervised Fine-Tuning — unified entry point.

Supports two backends, selected via ``--backend``:

- **megatron** — Megatron-LM-AMD ``pretrain`` driver with TP/PP/CP/VP/SP.
  Requires ``pip install megatron-lm``.
- **fsdp** — PyTorch FSDP + HuggingFace Transformers.
  Requires ``pip install transformers peft``.

Usage:
    # Megatron backend (default)
    torchrun --nproc_per_node=8 finetune_llama2.py --backend megatron <megatron args>

    # FSDP backend
    torchrun --nproc_per_node=8 finetune_llama2.py --backend fsdp <fsdp args>

See run_finetune.sh for a complete launch example.
"""

import os
import sys

from lumen.models.utils import peek_backend


def _maybe_install_profiler():
    """Patch train_step to capture torch.profiler on steps [start, end].

    Activated by env LUMEN_PROF_START (default: off).
    LUMEN_PROF_END, LUMEN_PROF_OUTPUT, LUMEN_PROF_STOP_AFTER are optional.
    """
    prof_start = os.environ.get("LUMEN_PROF_START")
    if not prof_start:
        return

    import functools

    import megatron.training.training as _mtt
    import torch
    import torch.distributed as dist

    start = int(prof_start)
    end = int(os.environ.get("LUMEN_PROF_END", str(start + 2)))
    output = os.environ.get(
        "LUMEN_PROF_OUTPUT",
        "/data1/lumen/results/tp1_fp8/profile_summary_new.txt",
    )
    stop_after = int(os.environ.get("LUMEN_PROF_STOP_AFTER", "0"))

    _state = {"step": 0, "prof": None}
    _orig = _mtt.train_step

    @functools.wraps(_orig)
    def _profiled(*args, **kwargs):
        _state["step"] += 1
        step = _state["step"]
        rank = dist.get_rank() if dist.is_initialized() else 0

        if step == start and rank == 0:
            _state["prof"] = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            _state["prof"].__enter__()
            print(f"[PROFILER] Started at step {step}", flush=True)

        result = _orig(*args, **kwargs)

        if step == end and rank == 0 and _state["prof"] is not None:
            _state["prof"].__exit__(None, None, None)
            summary = _state["prof"].key_averages().table(sort_by="self_cuda_time_total", row_limit=200)
            with open(output, "w") as f:
                f.write(
                    f"Profile steps {start}-{end}, "
                    f"LUMEN_PREFER_HIPBLASLT="
                    f"{os.environ.get('LUMEN_PREFER_HIPBLASLT', '?')}\n\n"
                )
                f.write(summary)
            print(f"[PROFILER] Wrote {output}", flush=True)
            _state["prof"] = None

        if stop_after and step >= stop_after:
            if rank == 0:
                print(f"[PROFILER] Stopping after step {step}", flush=True)
            sys.exit(0)

        return result

    _mtt.train_step = _profiled
    print(f"[PROFILER] Will profile steps {start}-{end}, output={output}", flush=True)


def _run_megatron():
    from megatron.core.enums import ModelType
    from megatron.training import pretrain

    from lumen.models.llama2.megatron import (
        add_finetune_args,
        apply_fp8_training,
        apply_lora,
        forward_step,
        lumen_gpt_builder,
        train_valid_test_datasets_provider,
    )
    from lumen.models.megatron import (
        install_fp8_param_gather_hook,
        install_fp8_param_storage_hook,
        install_hip_graphs_hook,
        make_lumen_model_provider,
    )

    model_provider = make_lumen_model_provider(
        lumen_gpt_builder,
        lora_applier=apply_lora,
        fp8_applier=apply_fp8_training,
    )
    install_fp8_param_gather_hook()
    install_fp8_param_storage_hook()
    install_hip_graphs_hook()
    _maybe_install_profiler()

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_finetune_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )


def _run_fsdp():
    from lumen.models.llama2.fsdp import FSDPTrainer, get_args

    args = get_args()
    trainer = FSDPTrainer(args)
    trainer.train()


if __name__ == "__main__":
    backend = peek_backend()

    if backend == "megatron":
        _run_megatron()
    elif backend == "fsdp":
        _run_fsdp()
    else:
        print(f"ERROR: Unknown backend '{backend}'. Use 'megatron' or 'fsdp'.")
        sys.exit(1)
