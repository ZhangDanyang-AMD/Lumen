###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""SonicMoE expert adapter for Megatron-Core MoE layers."""

from __future__ import annotations

import os
from typing import Iterable, MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.utils import get_pg_rank


def _expert_weights(experts: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return FC1/FC2 weights in local-expert order."""
    if hasattr(experts, "linear_fc1") and hasattr(experts, "linear_fc2"):
        count = int(experts.num_local_experts)
        fc1 = [getattr(experts.linear_fc1, f"weight{i}") for i in range(count)]
        fc2 = [getattr(experts.linear_fc2, f"weight{i}") for i in range(count)]
        return fc1, fc2

    if hasattr(experts, "local_experts"):
        local_experts: Iterable[nn.Module] = experts.local_experts
        fc1 = [expert.linear_fc1.weight for expert in local_experts]
        fc2 = [expert.linear_fc2.weight for expert in local_experts]
        return fc1, fc2

    raise TypeError(f"Unsupported Megatron expert implementation: {type(experts).__name__}")


def _pop_expert_weights(
    state_dict: MutableMapping[str, torch.Tensor],
    prefix: str,
    stem: str,
    count: int,
) -> list[torch.Tensor] | None:
    """Remove and return per-expert checkpoint tensors for one linear layer."""
    sequential_keys = [
        f"{prefix}local_experts.{index}.{stem}.weight" for index in range(count)
    ]
    grouped_keys = [
        f"{prefix}{stem}.weight{index}" for index in range(count)
    ]
    keys = sequential_keys if all(key in state_dict for key in sequential_keys) else grouped_keys
    if not all(key in state_dict for key in keys):
        return None

    weights = [state_dict.pop(key) for key in keys]
    for index in range(count):
        for extra_key in (
            f"{prefix}local_experts.{index}.{stem}._extra_state",
            f"{prefix}{stem}._extra_state{'' if index == 0 else index}",
        ):
            state_dict.pop(extra_key, None)
    return weights


def _stacked_expert_factory(
    tensor: torch.Tensor,
    *,
    prefix: str,
    layer_name: str,
    ep_rank: int,
    ep_size: int,
    replica_id: tuple[int, int, int],
    sharded_offsets: tuple,
    split_swiglu: bool,
    singleton_local_shards: bool,
) -> ShardedTensorFactory:
    """Expose stacked Sonic weights using Megatron's per-expert checkpoint layout."""
    local_key = f"{prefix}{'w1' if layer_name == 'linear_fc1' else 'w2'}"
    num_local_experts = tensor.shape[0]
    num_global_experts = ep_size * num_local_experts

    @torch.no_grad()
    def build_fn(key, data, factory_replica_id, flattened_range):
        if flattened_range is not None:
            raise ValueError("Flattened optimizer ranges are not supported for SonicMoE")
        shards = []
        for local_index in range(num_local_experts):
            global_index = ep_rank * num_local_experts + local_index
            expert_tensor = data[local_index].transpose(0, 1)
            if singleton_local_shards:
                checkpoint_key = (
                    f"{prefix}experts.{global_index}.{layer_name}.weight"
                )
                expert_offsets = sharded_offsets
                prepend_axis_num = len(sharded_offsets)
            else:
                checkpoint_key = f"{prefix}experts.{layer_name}.weight"
                expert_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), global_index, num_global_experts),
                )
                prepend_axis_num = len(sharded_offsets) + 1

            if split_swiglu:
                gate, up = torch.chunk(expert_tensor, 2, dim=0)
                for part, value in enumerate((gate, up)):
                    part_key = (
                        f"{checkpoint_key}_{'w' if part == 0 else 'v'}"
                        if singleton_local_shards
                        else checkpoint_key
                    )
                    shards.append(
                        ShardedTensor.from_rank_offsets(
                            part_key,
                            value,
                            *expert_offsets,
                            (prepend_axis_num, part, 2),
                            replica_id=factory_replica_id,
                            prepend_axis_num=prepend_axis_num,
                        )
                    )
            else:
                shards.append(
                    ShardedTensor.from_rank_offsets(
                        checkpoint_key,
                        expert_tensor,
                        *expert_offsets,
                        replica_id=factory_replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )
                )
        return shards

    def merge_fn(loaded):
        if split_swiglu:
            experts = [
                torch.cat((loaded[index], loaded[index + 1]), dim=0)
                for index in range(0, len(loaded), 2)
            ]
        else:
            experts = list(loaded)
        return torch.stack(experts, dim=0).transpose(1, 2).contiguous()

    return ShardedTensorFactory(
        key=local_key,
        data=tensor,
        build_fn=build_fn,
        merge_fn=merge_fn,
        replica_id=replica_id,
    )


class SonicMoEExperts(nn.Module):
    """Megatron expert interface backed by AITER SonicMoE.

    Megatron dispatches and permutes token/expert assignments before invoking
    this module. SonicMoE's pre-routed entry point consumes those assignments
    and performs both expert GEMMs, SwiGLU, routed-weight multiplication, and
    their backward passes.
    """

    def __init__(self, experts: nn.Module):
        super().__init__()
        self.config = experts.config
        self.num_local_experts = int(experts.num_local_experts)
        self.ep_group = getattr(experts, "ep_group", None)
        self.tp_group = getattr(experts, "tp_group", None)
        self.dp_group = getattr(experts, "dp_group", None)
        self.gemm_backend = os.environ.get("SONIC_MOE_GEMM_BACKEND", "triton")
        if self.gemm_backend != "triton":
            raise ValueError(
                "SONIC_MOE_GEMM_BACKEND must be 'triton'; select its GEMM backend "
                "with SONIC_MOE_GROUPED_GEMM_BACKEND="
                "triton|hipblaslt|multistream|auto"
            )

        if getattr(self.config, "add_bias_linear", False):
            raise ValueError("SonicMoE integration currently requires --disable-bias-linear")
        if not getattr(self.config, "gated_linear_unit", False):
            raise ValueError("SonicMoE integration currently requires a gated activation (SwiGLU)")
        expert_tp = getattr(self.config, "expert_tensor_parallel_size", 1)
        if expert_tp != 1:
            raise ValueError("SonicMoE integration currently requires expert tensor parallel size 1")

        fc1, fc2 = _expert_weights(experts)
        self.w1 = nn.Parameter(
            torch.stack([weight.detach() for weight in fc1], dim=0)
            .transpose(1, 2)
            .contiguous()
        )
        self.w2 = nn.Parameter(
            torch.stack([weight.detach() for weight in fc2], dim=0)
            .transpose(1, 2)
            .contiguous()
        )
        for parameter in self.parameters():
            parameter.allreduce = False
        self._register_load_state_dict_pre_hook(self._remap_checkpoint_state)

        self.scaling_type = "none"
        self.scaling_manager = None
        from lumen.quantize.config import _get_float8_e4m3

        self.fp8_dtype = _get_float8_e4m3()
        self.block_size = 128
        self.gradient_accumulation_fusion = False
        self.delay_wgrad = False

    def _remap_checkpoint_state(
        self,
        state_dict: MutableMapping[str, torch.Tensor],
        prefix: str,
        *args,
    ) -> None:
        """Map SequentialMLP or TEGroupedMLP checkpoint keys to Sonic storage."""
        w1_key = f"{prefix}w1"
        w2_key = f"{prefix}w2"
        if w1_key not in state_dict:
            weights = _pop_expert_weights(
                state_dict, prefix, "linear_fc1", self.num_local_experts
            )
            if weights is not None:
                state_dict[w1_key] = (
                    torch.stack(weights, dim=0).transpose(1, 2).contiguous()
                )
        if w2_key not in state_dict:
            weights = _pop_expert_weights(
                state_dict, prefix, "linear_fc2", self.num_local_experts
            )
            if weights is not None:
                state_dict[w2_key] = (
                    torch.stack(weights, dim=0).transpose(1, 2).contiguous()
                )
        for key, parameter in ((w1_key, self.w1), (w2_key, self.w2)):
            value = state_dict.get(key)
            if value is not None and value.shape != parameter.shape:
                transposed = value.transpose(1, 2)
                if transposed.shape == parameter.shape:
                    state_dict[key] = transposed.contiguous()

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Return SequentialMLP-compatible distributed checkpoint shards."""
        metadata = metadata or {}
        singleton_local_shards = metadata.get("singleton_local_shards", False)
        ep_rank = get_pg_rank(self.ep_group)
        ep_size = self.ep_group.size() if self.ep_group is not None else 1
        replica_id = (
            0,
            get_pg_rank(self.tp_group),
            get_pg_rank(self.dp_group),
        )
        return {
            f"{prefix}w1": _stacked_expert_factory(
                self.w1,
                prefix=prefix,
                layer_name="linear_fc1",
                ep_rank=ep_rank,
                ep_size=ep_size,
                replica_id=replica_id,
                sharded_offsets=sharded_offsets,
                split_swiglu=True,
                singleton_local_shards=singleton_local_shards,
            ),
            f"{prefix}w2": _stacked_expert_factory(
                self.w2,
                prefix=prefix,
                layer_name="linear_fc2",
                ep_rank=ep_rank,
                ep_size=ep_size,
                replica_id=replica_id,
                sharded_offsets=sharded_offsets,
                split_swiglu=False,
                singleton_local_shards=singleton_local_shards,
            ),
        }

    def enable_fp8(
        self,
        scaling_manager=None,
        scaling_type="dynamic",
        fp8_dtype=None,
        block_size=None,
    ):
        """Use FP8 compute on expert GEMMs; master ``w1`` / ``w2`` stay BF16.

        ``blockwise2d`` is the official Qwen3-*-FP8 recipe (128×128 weights,
        dynamic 1×128 activations) on both forward and backward.
        """
        from lumen.quantize import QuantConfig, ScalingManager

        self.scaling_type = scaling_type
        self.scaling_manager = scaling_manager or ScalingManager(QuantConfig())
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if block_size is not None:
            self.block_size = block_size

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert,
        permuted_probs: torch.Tensor,
    ):
        cpu_counts = None
        gpu_counts = None
        if isinstance(tokens_per_expert, tuple):
            cpu_counts, gpu_counts = tokens_per_expert
        else:
            cpu_counts = tokens_per_expert

        if self.scaling_type != "none":
            return _fp8_pre_routed_forward(
                permuted_local_hidden_states,
                gpu_counts if gpu_counts is not None else cpu_counts,
                permuted_probs,
                self.w1,
                self.w2,
                scaling_manager=self.scaling_manager,
                scaling_type=self.scaling_type,
                fp8_dtype=self.fp8_dtype,
                block_size=self.block_size,
            ), None

        # Keep Megatron's dispatcher-produced host counts on the CPU. The
        # pre-routed entry point creates the GPU offsets needed by Triton while
        # the multi-stream hipBLASLt backend reuses host offsets, matching
        # TEGroupedMLP's one-D2H-per-layer metadata boundary.
        counts = torch.as_tensor(
            cpu_counts if cpu_counts is not None else gpu_counts,
            dtype=torch.int32,
        )

        from aiter.ops.triton.sonicmoe import (
            SonicMoEActivationType,
            moe_pre_routed_inputs,
        )

        output, _expert_frequency = moe_pre_routed_inputs(
            permuted_local_hidden_states,
            permuted_probs.reshape(-1).float(),
            counts,
            self.w1,
            None,
            self.w2,
            None,
            torch.cuda.current_stream().cuda_stream,
            SonicMoEActivationType.SWIGLU,
            False,
            True,
        )
        return output, None


def _group_sizes_on_device(tokens_per_expert, num_experts: int, device) -> torch.Tensor:
    counts = torch.as_tensor(tokens_per_expert)
    if counts.numel() != num_experts:
        raise ValueError(
            f"Expected {num_experts} local expert counts, got {counts.numel()}"
        )
    if counts.device.type == "cpu":
        return counts.to(device=device, dtype=torch.int32)
    return counts.to(dtype=torch.int32)


def _expert_token_counts(tokens_per_expert, num_experts: int) -> list[int]:
    counts = torch.as_tensor(tokens_per_expert, dtype=torch.int64)
    if counts.numel() != num_experts:
        raise ValueError(
            f"Expected {num_experts} local expert counts, got {counts.numel()}"
        )
    if counts.device.type != "cpu":
        counts = counts.cpu()
    return [int(value) for value in counts.tolist()]


def _quantized_expert_linear(
    tokens: torch.Tensor,
    weight_kn: torch.Tensor,
    *,
    scaling_manager,
    scaling_type: str,
    fp8_dtype,
    block_size: int,
    tensor_id: str,
):
    """Run one expert GEMM: ``tokens @ weight_kn`` with BF16-master FP8 compute.

    ``weight_kn`` is the SonicMoE slice ``[K, N]``. ``quantized_linear`` uses the
    Linear convention ``Y = X @ W.T`` with ``W = [N, K]``.
    """
    if tokens.numel() == 0:
        return tokens.new_empty(tokens.shape[0], weight_kn.shape[1])
    from lumen.ops.quantize.linear import quantized_linear

    return quantized_linear(
        tokens,
        weight_kn.transpose(0, 1),
        None,
        scaling_manager=scaling_manager,
        scaling_type=scaling_type,
        fp8_dtype=fp8_dtype,
        block_size=block_size,
        tensor_id=tensor_id,
    )


def _fp8_pre_routed_forward(
    hidden_states: torch.Tensor,
    tokens_per_expert,
    permuted_probs: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    scaling_manager,
    scaling_type: str,
    fp8_dtype,
    block_size: int,
) -> torch.Tensor:
    """SwiGLU expert MLP with official blockwise FP8 on expert-sorted tokens.

    Default path uses grouped FP8 GEMM for ``w1``/``w2`` forward and backward.
    Set ``SONIC_MOE_FP8_GROUPED=0`` to fall back to per-expert
    ``quantized_linear``.
    """
    num_experts = w1.shape[0]
    hidden = hidden_states.shape[-1]
    if hidden_states.numel() == 0:
        return hidden_states.new_empty(hidden_states.shape[0], hidden)

    use_grouped = os.environ.get("SONIC_MOE_FP8_GROUPED", "1") != "0"
    if use_grouped:
        group_sizes = _group_sizes_on_device(
            tokens_per_expert, num_experts, hidden_states.device
        )
        from lumen.ops.gemm.grouped_gemm import grouped_fp8_expert_mlp

        output = grouped_fp8_expert_mlp(
            hidden_states,
            w1,
            w2,
            group_sizes,
            scaling_type=scaling_type,
            fp8_dtype=fp8_dtype,
            block_size=block_size,
        )
    else:
        counts = _expert_token_counts(tokens_per_expert, num_experts)
        outputs = []
        offset = 0
        for expert_index, token_count in enumerate(counts):
            expert_tokens = hidden_states[offset : offset + token_count]
            fc1 = _quantized_expert_linear(
                expert_tokens,
                w1[expert_index],
                scaling_manager=scaling_manager,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
                tensor_id=f"sonic_w1.{expert_index}",
            )
            gate, up = fc1.chunk(2, dim=-1)
            hidden_act = F.silu(gate) * up
            fc2 = _quantized_expert_linear(
                hidden_act,
                w2[expert_index],
                scaling_manager=scaling_manager,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
                tensor_id=f"sonic_w2.{expert_index}",
            )
            outputs.append(fc2)
            offset += token_count
        output = (
            torch.cat(outputs, dim=0) if outputs else hidden_states.new_empty(0, hidden)
        )

    scores = permuted_probs.reshape(-1, 1).to(dtype=output.dtype)
    if scores.numel() != output.shape[0]:
        raise ValueError(
            f"Expected one router score per pre-routed token ({output.shape[0]}), "
            f"got {scores.numel()}"
        )
    return output * scores


def replace_megatron_moe_experts(model: nn.Module) -> int:
    """Replace Megatron ``MoELayer.experts`` modules with SonicMoE adapters."""
    replaced = 0
    for module in model.modules():
        if type(module).__name__ != "MoELayer" or not hasattr(module, "experts"):
            continue
        if isinstance(module.experts, SonicMoEExperts):
            continue
        module.experts = SonicMoEExperts(module.experts)
        replaced += 1
    return replaced


__all__ = ["SonicMoEExperts", "replace_megatron_moe_experts"]
