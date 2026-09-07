###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Qwen3-30B-A3B training with HuggingFace Transformers and PyTorch FSDP2.

All ranks form the dense/data-parallel group and also participate in
expert-parallel rows. Experts with the same local index are synchronized only
across expert-data-parallel replicas.
"""

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

QWEN3_30B_A3B_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1_000_000.0,
}


def _rank0_log(message: str, *args) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message, *args)


@torch.no_grad()
def _clip_grad_norm_mixed_mesh(
    parameters,
    max_norm: float,
) -> torch.Tensor:
    """Clip an FSDP2 model whose parameters use different device meshes."""
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.zeros((), device=torch.cuda.current_device())

    local_squared_norm = torch.zeros(
        (),
        device=grads[0].device,
        dtype=torch.float32,
    )
    for grad in grads:
        local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
        local_squared_norm += local_grad.float().square().sum()

    if dist.is_initialized():
        dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
    total_norm = local_squared_norm.sqrt()
    clip_coefficient = min(max_norm / (total_norm.item() + 1e-6), 1.0)
    if clip_coefficient < 1.0:
        for grad in grads:
            grad.mul_(clip_coefficient)
    return total_norm


@dataclass(frozen=True)
class ParallelGroups:
    """Dense-DP, EP, and expert-DP process groups and coordinates."""

    ep_group: Optional[dist.ProcessGroup]
    dp_group: Optional[dist.ProcessGroup]
    expert_dp_group: Optional[dist.ProcessGroup]
    ep_rank: int
    dp_rank: int
    expert_dp_rank: int
    ep_size: int
    dp_size: int
    expert_dp_size: int


@dataclass(frozen=True)
class _ParallelRankLayout:
    """Rank membership used to construct the parallel process groups."""

    ep_ranks: tuple[int, ...]
    dp_ranks: tuple[int, ...]
    expert_dp_ranks: tuple[int, ...]
    ep_rank: int
    dp_rank: int
    expert_dp_rank: int
    expert_dp_size: int


def _parallel_rank_layout(
    world_size: int,
    global_rank: int,
    ep_size: int,
    dp_size: Optional[int] = None,
) -> _ParallelRankLayout:
    """Return Megatron-compatible overlapping DP and EP rank membership."""
    if ep_size < 1 or world_size % ep_size:
        raise ValueError(f"ep_size={ep_size} must divide world_size={world_size}")
    if dp_size is not None and dp_size != world_size:
        raise ValueError(
            f"dp_size={dp_size} must equal world_size={world_size}; "
            "dense/data parallelism overlaps expert parallelism"
        )
    if not 0 <= global_rank < world_size:
        raise ValueError(f"global_rank={global_rank} must be in [0, {world_size})")

    expert_dp_size = world_size // ep_size
    ep_row = global_rank // ep_size
    ep_rank = global_rank % ep_size
    return _ParallelRankLayout(
        ep_ranks=tuple(range(ep_row * ep_size, (ep_row + 1) * ep_size)),
        dp_ranks=tuple(range(world_size)),
        expert_dp_ranks=tuple(
            replica * ep_size + ep_rank for replica in range(expert_dp_size)
        ),
        ep_rank=ep_rank,
        dp_rank=global_rank,
        expert_dp_rank=ep_row,
        expert_dp_size=expert_dp_size,
    )


def create_parallel_groups(
    ep_size: int,
    dp_size: Optional[int] = None,
    timeout_minutes: int = 60,
) -> ParallelGroups:
    """Create overlapping dense-DP, EP, and expert-DP process groups."""
    if not dist.is_initialized():
        if ep_size != 1 or dp_size not in (None, 1):
            raise RuntimeError(
                "torch.distributed must be initialized when DP or EP is greater than one"
            )
        return ParallelGroups(
            ep_group=None,
            dp_group=None,
            expert_dp_group=None,
            ep_rank=0,
            dp_rank=0,
            expert_dp_rank=0,
            ep_size=1,
            dp_size=1,
            expert_dp_size=1,
        )

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    layout = _parallel_rank_layout(world_size, global_rank, ep_size, dp_size)

    ep_group = None
    timeout = timedelta(minutes=timeout_minutes)
    for row in range(layout.expert_dp_size):
        ranks = list(range(row * ep_size, (row + 1) * ep_size))
        group = dist.new_group(ranks, timeout=timeout)
        if global_rank in ranks:
            ep_group = group

    expert_dp_group = None
    for column in range(ep_size):
        ranks = [
            row * ep_size + column for row in range(layout.expert_dp_size)
        ]
        group = dist.new_group(ranks, timeout=timeout)
        if global_rank in ranks:
            expert_dp_group = group

    return ParallelGroups(
        ep_group=ep_group,
        dp_group=dist.group.WORLD,
        expert_dp_group=expert_dp_group,
        ep_rank=layout.ep_rank,
        dp_rank=layout.dp_rank,
        expert_dp_rank=layout.expert_dp_rank,
        ep_size=ep_size,
        dp_size=world_size,
        expert_dp_size=layout.expert_dp_size,
    )


class _LocalExpertMLP(nn.Module):
    """One Qwen3 expert represented with patchable ``nn.Linear`` modules."""

    def __init__(
        self,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        activation,
    ):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            gate_up_weight.shape[1],
            gate_up_weight.shape[0],
            bias=False,
            device=gate_up_weight.device,
            dtype=gate_up_weight.dtype,
        )
        self.down_proj = nn.Linear(
            down_weight.shape[1],
            down_weight.shape[0],
            bias=False,
            device=down_weight.device,
            dtype=down_weight.dtype,
        )
        self.gate_up_proj.weight.data.copy_(gate_up_weight)
        self.down_proj.weight.data.copy_(down_weight)
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.activation(gate) * up)


def _local_weight(weight: torch.Tensor) -> torch.Tensor:
    """Unwrap FSDP2/DTensor views so Sonic grouped GEMM sees a dense tensor."""
    to_local = getattr(weight, "to_local", None)
    if not callable(to_local):
        return weight
    local = to_local()
    if tuple(local.shape) == tuple(weight.shape):
        return local
    full_tensor = getattr(weight, "full_tensor", None)
    if callable(full_tensor):
        return full_tensor()
    return local


class _FusedLocalExperts(nn.Module):
    """Convert a local packed-weight slice into patchable expert modules."""

    def __init__(self, experts: nn.Module, start: int, end: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                _LocalExpertMLP(
                    experts.gate_up_proj[expert_id].detach(),
                    experts.down_proj[expert_id].detach(),
                    experts.act_fn,
                )
                for expert_id in range(start, end)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate all assignments through one FSDP-wrapped module call."""
        output = torch.zeros_like(hidden_states)
        for expert_id, expert in enumerate(self.experts):
            positions = torch.where(expert_ids == expert_id)[0]
            if positions.numel() == 0:
                continue
            expert_output = expert(hidden_states[positions])
            output.index_copy_(
                0,
                positions,
                expert_output * routing_weights[positions].unsqueeze(-1),
            )
        return output


class _TEGroupedLocalExperts(nn.Module):
    """Local experts backed by Transformer Engine grouped GEMMs."""

    def __init__(self, experts: nn.Module, start: int, end: int):
        super().__init__()
        import transformer_engine.pytorch as te

        num_experts = end - start
        self.num_experts = num_experts
        gate_up_weight = experts.gate_up_proj[start:end]
        down_weight = experts.down_proj[start:end]
        hidden_size = gate_up_weight.shape[2]
        intermediate_size = down_weight.shape[2]
        common = {
            "num_gemms": num_experts,
            "sequence_parallel": False,
            "fuse_wgrad_accumulation": False,
            "bias": False,
            "return_bias": False,
            "params_dtype": gate_up_weight.dtype,
            "device": gate_up_weight.device,
        }
        self.gate_up_proj = te.GroupedLinear(
            in_features=hidden_size,
            out_features=2 * intermediate_size,
            **common,
        )
        self.down_proj = te.GroupedLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            **common,
        )
        self.activation = experts.act_fn
        for local_id, global_id in enumerate(range(start, end)):
            getattr(self.gate_up_proj, f"weight{local_id}").data.copy_(
                experts.gate_up_proj[global_id]
            )
            getattr(self.down_proj, f"weight{local_id}").data.copy_(
                experts.down_proj[global_id]
            )

    def forward_all(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate all local experts using two grouped GEMMs."""
        order = torch.argsort(expert_ids, stable=True)
        sorted_hidden = hidden_states[order]
        sorted_weights = routing_weights[order]
        splits = torch.bincount(
            expert_ids,
            minlength=self.num_experts,
        ).tolist()
        gate, up = self.gate_up_proj(sorted_hidden, splits).chunk(2, dim=-1)
        activated = self.activation(gate) * up
        sorted_output = self.down_proj(activated, splits)
        sorted_output = sorted_output * sorted_weights.unsqueeze(-1)
        output = torch.empty_like(sorted_output)
        output.index_copy_(0, order, sorted_output)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_all(hidden_states, expert_ids, routing_weights)


class _SonicLocalExperts(nn.Module):
    """Local HF expert slice backed by AITER SonicMoE."""

    def __init__(self, experts: nn.Module, start: int, end: int):
        super().__init__()
        self.num_experts = end - start
        gate, up = experts.gate_up_proj[start:end].detach().chunk(2, dim=1)
        self.w1 = nn.Parameter(
            torch.stack((gate, up), dim=2)
            .flatten(1, 2)
            .transpose(1, 2)
            .contiguous()
        )
        self.w2 = nn.Parameter(
            experts.down_proj[start:end].detach().transpose(1, 2).contiguous()
        )
        self.scaling_type = "none"
        self.scaling_manager = None
        self.fp8_dtype = None
        self.block_size = 128
        self.gemm_backend = os.environ.get("SONIC_MOE_GEMM_BACKEND", "triton")
        if self.gemm_backend != "triton":
            raise ValueError(
                "SONIC_MOE_GEMM_BACKEND must be 'triton'; select its GEMM backend "
                "with SONIC_MOE_GROUPED_GEMM_BACKEND="
                "triton|hipblaslt|multistream|auto"
            )

    def enable_fp8(
        self,
        scaling_manager=None,
        scaling_type="blockwise2d",
        fp8_dtype=None,
        block_size=None,
    ):
        """Use official blockwise FP8 on expert GEMMs; master w1/w2 stay BF16."""
        from lumen.quantize import QuantConfig, ScalingManager

        if scaling_type != "blockwise2d":
            raise ValueError(
                "FSDP Sonic expert FP8 only supports scaling_type='blockwise2d', "
                f"got {scaling_type!r}"
            )
        self.scaling_type = scaling_type
        self.scaling_manager = scaling_manager or ScalingManager(QuantConfig())
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if block_size is not None:
            self.block_size = block_size

    def forward_grouped(
        self,
        hidden_states: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate expert-major assignments without an internal permutation."""
        if hidden_states.shape[0] == 0:
            return hidden_states

        w1 = _local_weight(self.w1)
        w2 = _local_weight(self.w2)
        if self.scaling_type != "none":
            from lumen.ops.gemm.grouped_gemm import grouped_fp8_expert_mlp

            return grouped_fp8_expert_mlp(
                hidden_states,
                w1,
                w2,
                counts.to(device=hidden_states.device, dtype=torch.int32),
                scaling_type=self.scaling_type,
                fp8_dtype=self.fp8_dtype,
                block_size=self.block_size,
                # w1 stores per-column gate/up pairs, matching the BF16
                # moe_pre_routed_inputs call below.
                concat_layout=False,
            )

        from aiter.ops.triton import sonicmoe

        common = (
            w1,
            None,
            w2,
            None,
        )
        if not hasattr(sonicmoe, "moe_pre_routed_inputs"):
            raise RuntimeError(
                "This SonicMoE build lacks moe_pre_routed_inputs, which is required "
                "for the grouped GEMM backend"
            )

        kernel_weights = torch.ones(
            hidden_states.shape[0],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        output, _expert_frequency = sonicmoe.moe_pre_routed_inputs(
            hidden_states,
            kernel_weights,
            counts.to(dtype=torch.int32),
            *common,
            torch.cuda.current_stream().cuda_stream,
            sonicmoe.SonicMoEActivationType.SWIGLU,
            False,
            False,
        )
        return output

    def forward_all(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate all received expert assignments in one SonicMoE call."""
        if hidden_states.shape[0] == 0:
            return hidden_states

        order = torch.argsort(expert_ids, stable=True)
        sorted_hidden = hidden_states[order]
        sorted_weights = routing_weights[order]
        counts = torch.bincount(expert_ids, minlength=self.num_experts)

        # Keep score gradients in native PyTorch. Fusing this multiplication
        # into Sonic changes BF16 rounding and is not faster end to end.
        sorted_output = self.forward_grouped(sorted_hidden, counts)
        sorted_output = sorted_output * sorted_weights.unsqueeze(-1)

        output = torch.empty_like(sorted_output)
        output.index_copy_(0, order, sorted_output)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        *,
        counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if counts is not None:
            return self.forward_grouped(hidden_states, counts)
        if expert_ids is None or routing_weights is None:
            raise ValueError("expert_ids and routing_weights are required")
        return self.forward_all(hidden_states, expert_ids, routing_weights)


class _ModuleListLocalExperts(nn.Module):
    """Local slice of older Transformers Qwen3 expert modules."""

    def __init__(self, experts: nn.ModuleList, start: int, end: int):
        super().__init__()
        self.experts = nn.ModuleList(list(experts[start:end]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate all assignments through one FSDP-wrapped module call."""
        output = torch.zeros_like(hidden_states)
        for expert_id, expert in enumerate(self.experts):
            positions = torch.where(expert_ids == expert_id)[0]
            if positions.numel() == 0:
                continue
            expert_output = expert(hidden_states[positions])
            output.index_copy_(
                0,
                positions,
                expert_output * routing_weights[positions].unsqueeze(-1),
            )
        return output


class EPShardedMoeBlock(nn.Module):
    """Qwen3 sparse MoE block with differentiable all-to-all token dispatch."""

    def __init__(
        self,
        original_block: nn.Module,
        ep_rank: int,
        ep_size: int,
        ep_group: Optional[dist.ProcessGroup],
        expert_backend: str = "sequential",
    ):
        super().__init__()
        self.gate = original_block.gate
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.ep_group = ep_group
        self._lumen_fused_router = False
        self._lumen_moe_dispatch_overlap = False
        self._lumen_moe_global_expert_layout = False
        self._sonic_first_forward = expert_backend == "sonic"

        experts = original_block.experts
        num_experts = getattr(experts, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(self.gate, "num_experts", None)
        if num_experts is None:
            num_experts = len(experts)
        self.num_experts = int(num_experts)
        self.top_k = int(getattr(self.gate, "top_k", getattr(original_block, "top_k", 1)))
        if self.num_experts % ep_size:
            raise ValueError(f"num_experts={self.num_experts} must be divisible by ep_size={ep_size}")

        self.experts_per_rank = self.num_experts // ep_size
        self.local_expert_start = ep_rank * self.experts_per_rank
        end = self.local_expert_start + self.experts_per_rank
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            if expert_backend == "te_grouped":
                self.local_experts = _TEGroupedLocalExperts(
                    experts,
                    self.local_expert_start,
                    end,
                )
            elif expert_backend == "sonic":
                self.local_experts = _SonicLocalExperts(
                    experts,
                    self.local_expert_start,
                    end,
                )
            else:
                self.local_experts = _FusedLocalExperts(experts, self.local_expert_start, end)
        elif isinstance(experts, (nn.ModuleList, list)):
            if expert_backend in ("sonic", "te_grouped"):
                raise TypeError(
                    f"expert_backend={expert_backend} requires packed HF expert "
                    "weights (gate_up_proj/down_proj); "
                    f"got {type(experts).__name__}"
                )
            self.local_experts = _ModuleListLocalExperts(experts, self.local_expert_start, end)
        else:
            raise TypeError(f"Unsupported Qwen3 expert container: {type(experts).__name__}")

    def enable_lumen_fused_router(self) -> None:
        """Enable the router implementation installed by LumenConfig."""
        self._lumen_fused_router = True

    def enable_lumen_moe_dispatch_overlap(self) -> None:
        """Overlap count exchange with token permutation and payload packing."""
        self._lumen_moe_dispatch_overlap = True

    def enable_lumen_moe_global_expert_layout(self) -> None:
        """Use Megatron-style expert-major dispatch with SonicMoE."""
        if not hasattr(self.local_experts, "forward_grouped"):
            raise ValueError(
                "The global-expert dispatcher currently requires expert_backend=sonic"
            )
        self._lumen_moe_global_expert_layout = True

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_output = self.gate(hidden_states)
        if isinstance(gate_output, tuple) and len(gate_output) >= 3:
            return gate_output[1], gate_output[2]

        router_logits = gate_output
        if self._lumen_fused_router:
            from lumen.ops.moe import fused_topk_with_score_function

            _, routing_probs = fused_topk_with_score_function(
                router_logits,
                self.top_k,
                True,
                None,
                None,
                None,
                "softmax",
                None,
            )
            routing_weights, selected_experts = torch.topk(
                routing_probs, self.top_k, dim=-1
            )
            return routing_weights.to(hidden_states.dtype), selected_experts

        routing_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_scores, self.top_k, dim=-1)
        if getattr(self.gate, "norm_topk_prob", getattr(self, "norm_topk_prob", False)):
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights.to(hidden_states.dtype), selected_experts

    def _exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        if self.ep_size == 1:
            return send_counts
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        return recv_counts

    def _exchange_tensor(
        self,
        tensor: torch.Tensor,
        send_splits: list[int],
        recv_splits: list[int],
        *,
        differentiable: bool,
    ) -> torch.Tensor:
        if self.ep_size == 1:
            return tensor
        output = tensor.new_empty((sum(recv_splits), *tensor.shape[1:]))
        if differentiable:
            return dist_nn.all_to_all_single(
                output,
                tensor.contiguous(),
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self.ep_group,
            )
        dist.all_to_all_single(
            output,
            tensor.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.ep_group,
        )
        return output

    def _gather_expert_counts(self, local_counts: torch.Tensor) -> torch.Tensor:
        """Gather each sender's global-expert histogram on the EP group."""
        if self.ep_size == 1:
            return local_counts.unsqueeze(0)
        gathered = torch.empty(
            self.ep_size * self.num_experts,
            dtype=local_counts.dtype,
            device=local_counts.device,
        )
        dist.all_gather_into_tensor(
            gathered,
            local_counts.contiguous(),
            group=self.ep_group,
        )
        return gathered.view(self.ep_size, self.num_experts)

    def _forward_global_expert_layout(
        self,
        hidden_flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch expert-major tokens and avoid receiver-side token sorting."""
        from lumen.ops.moe.dispatch_layout import transpose_variable_chunks

        token_ids = (
            torch.arange(hidden_flat.shape[0], device=hidden_flat.device)
            .unsqueeze(1)
            .expand_as(selected_experts)
            .reshape(-1)
        )
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        order = torch.argsort(flat_experts, stable=True)
        send_token_ids = token_ids[order]
        send_weights = flat_weights[order]

        local_counts = torch.bincount(flat_experts, minlength=self.num_experts)
        global_counts = self._gather_expert_counts(local_counts)
        recv_counts_by_sender = global_counts[
            :,
            self.local_expert_start : self.local_expert_start
            + self.experts_per_rank,
        ].contiguous()
        send_splits = (
            local_counts.view(self.ep_size, self.experts_per_rank)
            .sum(dim=1)
            .tolist()
        )
        recv_splits = recv_counts_by_sender.sum(dim=1).tolist()

        recv_hidden = self._exchange_tensor(
            hidden_flat[send_token_ids],
            send_splits,
            recv_splits,
            differentiable=True,
        )
        grouped_hidden = transpose_variable_chunks(
            recv_hidden,
            recv_counts_by_sender,
            source_layout="sender_major",
            fused=False,
        )
        grouped_output = self.local_experts(
            grouped_hidden,
            counts=recv_counts_by_sender.sum(dim=0),
        )
        local_output = transpose_variable_chunks(
            grouped_output,
            recv_counts_by_sender,
            source_layout="expert_major",
            fused=False,
        )
        returned = self._exchange_tensor(
            local_output,
            recv_splits,
            send_splits,
            differentiable=True,
        )
        final_output = torch.zeros_like(hidden_flat)
        final_output.index_add_(
            0,
            send_token_ids,
            returned * send_weights.unsqueeze(-1),
        )
        return final_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route tokens to local experts and restore their original ordering."""
        input_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, input_shape[-1])
        routing_weights, selected_experts = self._route(hidden_flat)
        if self._lumen_moe_global_expert_layout:
            return self._forward_global_expert_layout(
                hidden_flat,
                routing_weights,
                selected_experts,
            ).reshape(input_shape)

        token_ids = (
            torch.arange(hidden_flat.shape[0], device=hidden_flat.device)
            .unsqueeze(1)
            .expand(-1, selected_experts.shape[1])
            .reshape(-1)
        )
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        destinations = torch.div(flat_experts, self.experts_per_rank, rounding_mode="floor")
        local_expert_ids = flat_experts.remainder(self.experts_per_rank)
        send_counts = torch.bincount(destinations, minlength=self.ep_size)
        pending_counts = None
        if self._lumen_moe_dispatch_overlap and self.ep_size > 1:
            from lumen.ops.moe.dispatch_overlap import begin_count_exchange

            pending_counts = begin_count_exchange(send_counts, self.ep_group)
        order = torch.argsort(destinations, stable=True)

        send_token_ids = token_ids[order]
        send_hidden = hidden_flat[send_token_ids]
        send_expert_ids = local_expert_ids[order]
        send_weights = flat_weights[order]

        # Exchange tokens and their two scalar metadata fields together. Local
        # expert ids are small integers and are exactly representable in BF16.
        send_payload = torch.cat(
            (
                send_hidden,
                send_weights.to(send_hidden.dtype).unsqueeze(-1),
                send_expert_ids.to(send_hidden.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        if pending_counts is not None:
            send_splits, recv_splits = pending_counts.wait_for_splits()
        else:
            recv_counts = self._exchange_counts(send_counts)
            send_splits = send_counts.tolist()
            recv_splits = recv_counts.tolist()
        recv_payload = self._exchange_tensor(
            send_payload, send_splits, recv_splits, differentiable=True
        )
        recv_hidden = recv_payload[:, :-2]
        recv_weights = recv_payload[:, -2]
        recv_expert_ids = recv_payload[:, -1].to(torch.int64)

        local_output = self.local_experts(
            recv_hidden,
            recv_expert_ids,
            recv_weights,
        )

        if self._sonic_first_forward and self.ep_size > 1:
            dist.barrier(group=self.ep_group)
            self._sonic_first_forward = False

        returned = self._exchange_tensor(
            local_output, recv_splits, send_splits, differentiable=True
        )
        final_output = torch.zeros_like(hidden_flat)
        final_output.index_add_(0, send_token_ids, returned)
        return final_output.reshape(input_shape)


def shard_moe_experts(
    model: nn.Module,
    groups: ParallelGroups,
    expert_backend: str = "sequential",
) -> nn.Module:
    """Replace all HuggingFace Qwen3 MoE blocks with EP-sharded blocks."""
    replaced = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"):
            layer.mlp = EPShardedMoeBlock(
                mlp,
                ep_rank=groups.ep_rank,
                ep_size=groups.ep_size,
                ep_group=groups.ep_group,
                expert_backend=expert_backend,
            )
            replaced += 1
    _rank0_log(
        "EP sharding replaced %d MoE blocks (%d local experts per rank)",
        replaced,
        QWEN3_30B_A3B_CONFIG["num_experts"] // groups.ep_size,
    )
    return model


def apply_fsdp2(
    model: nn.Module,
    groups: ParallelGroups,
    sharding: str,
) -> nn.Module:
    """Shard shared parameters over dense DP and experts over expert DP."""
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if groups.dp_group is None or groups.expert_dp_group is None:
        raise RuntimeError("FSDP2 requires an initialized torch.distributed process group")

    dense_mesh = DeviceMesh.from_group(
        groups.dp_group,
        "cuda",
        mesh_dim_names=("dp",),
    )
    expert_mesh = DeviceMesh.from_group(
        groups.expert_dp_group,
        "cuda",
        mesh_dim_names=("expert_dp",),
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    reshard_after_forward = sharding == "full_shard"
    if sharding not in {"full_shard", "shard_grad_op"}:
        raise ValueError(f"Unsupported sharding strategy: {sharding}")

    layers = list(model.model.layers)
    for layer in layers:
        local_experts = getattr(getattr(layer, "mlp", None), "local_experts", None)
        if local_experts is not None:
            fully_shard(
                local_experts,
                mesh=expert_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )
            local_experts.set_gradient_divide_factor(1.0)
        fully_shard(
            layer,
            mesh=dense_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        layer.set_gradient_divide_factor(1.0)
    fully_shard(
        model,
        mesh=dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    model.set_gradient_divide_factor(1.0)
    _rank0_log(
        "FSDP2 wrapped %d layers over dense DP=%d and experts over expert DP=%d "
        "(EP=%d, reshard_after_forward=%s)",
        len(layers),
        groups.dp_size,
        groups.expert_dp_size,
        groups.ep_size,
        reshard_after_forward,
    )
    return model


class AlpacaDataset(Dataset):
    """Convert Alpaca JSONL rows to Qwen3 chat tokens with answer-only loss."""

    def __init__(
        self,
        path: str,
        tokenizer,
        seq_length: int,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        with Path(path).open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.length = min(len(self.rows), max_samples) if max_samples else len(self.rows)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return self.length

    def _tokenize_chat(self, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": True,
        }
        try:
            output = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            output = self.tokenizer.apply_chat_template(messages, **kwargs)
        return list(output["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt = row["instruction"].strip()
        if row.get("input", "").strip():
            prompt = f"{prompt}\n{row['input'].strip()}"
        prompt_ids = self._tokenize_chat(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        full_ids = self._tokenize_chat(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": row["output"]},
            ],
            add_generation_prompt=False,
        )
        loss_mask = [0] * len(prompt_ids) + [1] * max(0, len(full_ids) - len(prompt_ids))
        sample_length = self.seq_length + 1
        full_ids = full_ids[:sample_length]
        loss_mask = loss_mask[:sample_length]
        padding = sample_length - len(full_ids)
        if padding:
            full_ids.extend([self.pad_id] * padding)
            loss_mask.extend([0] * padding)
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        }


def build_model(args: argparse.Namespace) -> nn.Module:
    """Load Qwen3-30B-A3B or initialize the matching architecture."""
    from transformers import (
        AutoModelForCausalLM,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            # Preserve FP32 sharded/master parameters for AdamW, matching
            # Megatron's distributed optimizer. FSDP mixed precision casts the
            # all-gathered compute parameters to BF16.
            dtype=torch.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model = model.to(torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))))
    else:
        config = Qwen3MoeConfig(
            vocab_size=QWEN3_30B_A3B_CONFIG["vocab_size"],
            hidden_size=QWEN3_30B_A3B_CONFIG["hidden_size"],
            intermediate_size=QWEN3_30B_A3B_CONFIG["intermediate_size"],
            moe_intermediate_size=QWEN3_30B_A3B_CONFIG["moe_intermediate_size"],
            num_hidden_layers=QWEN3_30B_A3B_CONFIG["num_hidden_layers"],
            num_attention_heads=QWEN3_30B_A3B_CONFIG["num_attention_heads"],
            num_key_value_heads=QWEN3_30B_A3B_CONFIG["num_key_value_heads"],
            num_experts=QWEN3_30B_A3B_CONFIG["num_experts"],
            num_experts_per_tok=QWEN3_30B_A3B_CONFIG["num_experts_per_tok"],
            max_position_embeddings=args.max_position_embeddings,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": QWEN3_30B_A3B_CONFIG["rope_theta"],
            },
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            norm_topk_prob=args.norm_topk_prob,
            output_router_logits=True,
            router_aux_loss_coef=(
                args.aux_loss_coeff / QWEN3_30B_A3B_CONFIG["num_experts_per_tok"]
            ),
            tie_word_embeddings=False,
            use_cache=False,
            dtype=torch.float32,
        )
        # Qwen3-30B-A3B uses a 128-wide attention head although
        # hidden_size / num_attention_heads is 64.
        config.head_dim = 128
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        with torch.device(f"cuda:{local_rank}"):
            model = Qwen3MoeForCausalLM(config)
    for module in model.modules():
        if hasattr(module, "top_k") and hasattr(module, "norm_topk_prob"):
            module.norm_topk_prob = args.norm_topk_prob
    if args.fuse_rope:
        import transformers.models.qwen3_moe.modeling_qwen3_moe as modeling_qwen3_moe
        from lumen.ops.rope import apply_rotary_qk_autograd

        def lumen_rotary_embedding(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            position_ids=None,
            unsqueeze_dim: int = 1,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del position_ids, unsqueeze_dim
            return apply_rotary_qk_autograd(query, key, cos, sin)

        modeling_qwen3_moe.apply_rotary_pos_emb = lumen_rotary_embedding
        _rank0_log("Enabled AITER fused rotary embedding")
    return model


def _enable_fsdp_sonic_fp8(model: nn.Module, lumen_config, manager) -> None:
    """Apply official blockwise FP8 to FSDP Sonic expert modules."""
    qcfg = lumen_config.quant_config
    scaling_type = (
        qcfg.scaling.value if hasattr(qcfg.scaling, "value") else str(qcfg.scaling)
    )
    count = 0
    for module in model.modules():
        if not isinstance(module, _SonicLocalExperts):
            continue
        module.enable_fp8(
            scaling_manager=manager,
            scaling_type=scaling_type,
            fp8_dtype=qcfg.torch_dtype,
            block_size=qcfg.block_size,
        )
        count += 1
    if count == 0:
        raise RuntimeError(
            "fp8_blockwise2d with --expert-backend sonic found no "
            "_SonicLocalExperts modules"
        )
    _rank0_log(
        "Enabled FP8 (scaling=%s) on %d FSDP Sonic expert modules",
        scaling_type,
        count,
    )


def _enable_lumen(
    model: nn.Module,
    args: argparse.Namespace,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> nn.Module:
    if args.mode == "fp8_blockwise2d" and args.expert_backend == "te_grouped":
        raise ValueError(
            "fp8_blockwise2d does not cover TE grouped experts; "
            "use --expert-backend sequential or sonic"
        )
    if (
        args.mode == "bf16"
        and not args.aiter_attn
        and not args.lumen_norm
        and not args.fused_router
        and not args.lumen_moe_dispatch_overlap
        and not args.lumen_moe_global_expert_layout
    ):
        return model

    from lumen.config import LumenConfig

    config = LumenConfig(
        format="fp8_e4m3",
        scaling=args.fp8_scaling if args.mode == "fp8_blockwise2d" else "none",
        block_size=128,
        amax_algo="max",
        history_len=16,
        reduce_amax=False,
        quantize_activation=True,
        fp8_wgrad=True,
        cache_frozen_weight=False,
        bpreshuffle_gemm=False,
        quantize_grad=None,
        first_last_layers_bf16=False,
        lumen_norm=args.lumen_norm,
        hf_attn_patch=args.aiter_attn,
        fused_router=args.fused_router,
        moe_dispatch_overlap=args.lumen_moe_dispatch_overlap,
        moe_global_expert_layout=args.lumen_moe_global_expert_layout,
    )
    manager, model = config.enable(model, dp_group=dp_group)
    if args.mode == "fp8_blockwise2d" and args.expert_backend == "sonic":
        _enable_fsdp_sonic_fp8(model, config, manager)
    return model


class FSDPTrainer:
    """Full-parameter Qwen3-30B-A3B trainer using FSDP2 and expert parallelism."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        if args.fsdp_fp8_param_storage:
            raise ValueError(
                "--fsdp-fp8-param-storage only supports frozen base weights; "
                "Qwen3-30B-A3B uses full-parameter training"
            )
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                timeout=timedelta(minutes=args.distributed_timeout_minutes),
            )
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        torch.manual_seed(args.seed)

        self.groups = create_parallel_groups(
            args.ep_size,
            args.dp_size,
            timeout_minutes=args.distributed_timeout_minutes,
        )
        self.model = build_model(args)
        self.model = shard_moe_experts(
            self.model,
            self.groups,
            expert_backend=args.expert_backend,
        )
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model = _enable_lumen(self.model, args, self.groups.dp_group)
        self.model = apply_fsdp2(self.model, self.groups, args.sharding)

        decay_parameters = []
        no_decay_parameters = []
        for parameter in self.model.parameters():
            if parameter.ndim <= 1:
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay_parameters,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                },
            ],
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._lr_multiplier,
        )
        self.tokenizer, self.train_loader, self.val_loader = self._build_data()

    def _lr_multiplier(self, step: int) -> float:
        if step < self.args.lr_warmup_steps:
            return float(step) / max(self.args.lr_warmup_steps, 1)
        if self.args.lr_decay_style == "constant":
            return 1.0
        progress = (step - self.args.lr_warmup_steps) / max(
            self.args.max_steps - self.args.lr_warmup_steps,
            1,
        )
        min_ratio = self.args.min_lr / self.args.lr if self.args.lr else 0.0
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _build_data(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_name_or_path or self.args.model_name_or_path
        )

        def make_loader(path: str, max_samples: Optional[int], train: bool) -> DataLoader:
            if self.args.data_format == "alpaca":
                dataset = AlpacaDataset(path, tokenizer, self.args.seq_length, max_samples)
            else:
                from lumen.models.llama31.dataset import PretrainTextDataset

                virtual_num_samples = (
                    self.args.max_steps
                    * self.args.micro_batch_size
                    * self.args.gradient_accumulation_steps
                    * self.groups.dp_size
                    if train
                    else None
                )
                dataset = PretrainTextDataset(
                    data_path=path,
                    seq_length=self.args.seq_length,
                    tokenizer=tokenizer,
                    is_hf_tokenizer=True,
                    max_samples=max_samples,
                    virtual_num_samples=virtual_num_samples,
                )
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.groups.dp_size,
                rank=self.groups.dp_rank,
                shuffle=train and self.args.shuffle_data,
                seed=self.args.data_seed,
            )
            return DataLoader(
                dataset,
                batch_size=self.args.micro_batch_size,
                sampler=sampler,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
            )

        train_loader = make_loader(self.args.train_data_path, None, True)
        val_loader = (
            make_loader(self.args.val_data_path, self.args.val_samples, False)
            if self.args.val_data_path
            else None
        )
        return tokenizer, train_loader, val_loader

    def _loss_components(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.args.data_format == "pretrain":
            input_ids = batch["input_ids"].to(self.local_rank, non_blocking=True)
            labels = batch["labels"].to(self.local_rank, non_blocking=True)
            outputs = self.model(input_ids=input_ids, output_router_logits=True)
            token_loss = F.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.shape[-1]),
                labels.reshape(-1),
            )
            aux_loss = outputs.aux_loss
            if outputs.aux_loss is not None:
                hf_aux_coefficient = (
                    self.args.aux_loss_coeff
                    / QWEN3_30B_A3B_CONFIG["num_experts_per_tok"]
                )
                total_loss = token_loss + hf_aux_coefficient * outputs.aux_loss
            else:
                aux_loss = token_loss.new_zeros(())
                total_loss = token_loss
            return total_loss, token_loss, aux_loss

        tokens = batch["input_ids"].to(self.local_rank, non_blocking=True)
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        loss_mask = batch["loss_mask"][:, 1:].to(self.local_rank, non_blocking=True)
        logits = self.model(input_ids=input_ids).logits
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        )
        lm_loss = (token_loss * loss_mask.reshape(-1)).sum() / loss_mask.sum().clamp_min(1)
        return lm_loss, lm_loss, lm_loss.new_zeros(())

    def _loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._loss_components(batch)[0]

    @torch.no_grad()
    def validate(self) -> float:
        """Return validation loss averaged over data-parallel replicas."""
        if self.val_loader is None:
            raise RuntimeError("Validation data was not configured")
        self.model.eval()
        total_loss = 0.0
        batches = 0
        for batch in self.val_loader:
            total_loss += self._loss(batch).item()
            batches += 1
            if batches >= 10:
                break
        self.model.train()
        result = torch.tensor(
            [total_loss, float(batches)],
            device=self.local_rank,
            dtype=torch.float64,
        )
        if self.groups.dp_size > 1:
            dist.all_reduce(result, group=self.groups.dp_group)
        return (result[0] / result[1].clamp_min(1)).item()

    def train(self) -> None:
        """Run the configured training loop."""
        self.model.train()
        data_iterator = iter(self.train_loader)
        profiler = None
        profile_output = os.environ.get("LUMEN_PROFILE_OUTPUT")
        profile_step = int(os.environ.get("LUMEN_PROFILE_STEP", "1"))
        for step in range(1, self.args.max_steps + 1):
            if profile_output and self.rank == 0 and step == profile_step:
                profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                )
                profiler.start()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.local_rank)
            start = time.perf_counter()
            self.optimizer.zero_grad()
            accumulated_loss = 0.0
            accumulated_lm_loss = 0.0
            accumulated_aux_loss = 0.0
            for micro_step in range(self.args.gradient_accumulation_steps):
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.train_loader)
                    batch = next(data_iterator)
                dump_dir = os.environ.get("QWEN_PARITY_DUMP_DIR")
                if dump_dir:
                    dump_path = (
                        Path(dump_dir)
                        / f"fsdp-batch{step - 1}-micro{micro_step}-rank{self.rank}.pt"
                    )
                    if not dump_path.exists():
                        dump_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "input_ids": batch["input_ids"].cpu(),
                                "labels": batch.get("labels", torch.empty(0)).cpu(),
                            },
                            dump_path,
                        )
                loss, lm_loss, aux_loss = self._loss_components(batch)
                if os.environ.get("QWEN_PARITY_LOG_LOCAL_LOSS", "0") == "1":
                    logger.warning(
                        "Qwen parity local loss: rank=%d step=%d micro=%d lm_loss=%.9f",
                        self.rank,
                        step,
                        micro_step,
                        lm_loss.item(),
                    )
                # Both dense-DP and expert-DP FSDP groups use SUM reductions.
                # Scaling each rank's local loss by dense DP produces the
                # global-batch mean for shared and EP-local expert parameters.
                (
                    loss
                    / (
                        self.args.gradient_accumulation_steps
                        * self.groups.dp_size
                    )
                ).backward()
                accumulated_loss += loss.item()
                accumulated_lm_loss += lm_loss.item()
                accumulated_aux_loss += aux_loss.item()

            if self.args.max_grad_norm > 0:
                grad_norm = _clip_grad_norm_mixed_mesh(
                    self.model.parameters(),
                    self.args.max_grad_norm,
                )
            else:
                grad_norm = torch.zeros((), device=f"cuda:{self.local_rank}")
            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            peak_memory_mb = torch.tensor(
                torch.cuda.max_memory_allocated(self.local_rank) / (1024**2),
                dtype=torch.float64,
                device=f"cuda:{self.local_rank}",
            )
            dist.all_reduce(peak_memory_mb, op=dist.ReduceOp.MAX)
            global_batch_size = (
                self.args.micro_batch_size
                * self.args.gradient_accumulation_steps
                * self.groups.dp_size
            )
            throughput = global_batch_size * 1000 / elapsed_ms

            if step % self.args.log_interval == 0:
                loss_metrics = torch.tensor(
                    [
                        accumulated_loss,
                        accumulated_lm_loss,
                        accumulated_aux_loss,
                    ],
                    dtype=torch.float64,
                    device=f"cuda:{self.local_rank}",
                )
                if self.groups.dp_size > 1:
                    dist.all_reduce(loss_metrics, group=self.groups.dp_group)
                    loss_metrics /= self.groups.dp_size
                _rank0_log(
                    "step %d/%d | loss %.6f | lm_loss %.6f | aux_loss %.6f | "
                    "grad_norm %.3f | lr %.2e | step_time_ms %.1f | "
                    "throughput_samples_per_sec %.3f | "
                    "peak_memory_mb %.1f",
                    step,
                    self.args.max_steps,
                    loss_metrics[0].item() / self.args.gradient_accumulation_steps,
                    loss_metrics[1].item() / self.args.gradient_accumulation_steps,
                    loss_metrics[2].item() / self.args.gradient_accumulation_steps,
                    grad_norm.item(),
                    self.scheduler.get_last_lr()[0],
                    elapsed_ms,
                    throughput,
                    peak_memory_mb.item(),
                )
            if (
                self.val_loader is not None
                and self.args.eval_interval > 0
                and step % self.args.eval_interval == 0
            ):
                _rank0_log("step %d | val_loss %.4f", step, self.validate())
            if profiler is not None:
                profiler.step()
                if step == profile_step:
                    profiler.stop()
                    output_path = Path(profile_output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    profiler.export_chrome_trace(str(output_path))
                    summary_path = output_path.with_suffix(".txt")
                    summary_path.write_text(
                        profiler.key_averages().table(
                            sort_by="self_cuda_time_total",
                            row_limit=200,
                        )
                    )
                    _rank0_log("Wrote kernel profile to %s", output_path)
                    profiler = None


def get_args() -> argparse.Namespace:
    """Parse arguments for Qwen3-30B-A3B FSDP2 training."""
    parser = argparse.ArgumentParser(
        description="Qwen3-30B-A3B training with Transformers + FSDP2 + expert parallelism"
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="HF checkpoint path. Omit to initialize Qwen3-30B-A3B from scratch.",
    )
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--val-data-path", default=None)
    parser.add_argument("--data-format", choices=["pretrain", "alpaca"], default="pretrain")
    parser.add_argument("--mode", choices=["bf16", "fp8_blockwise2d"], default="bf16")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-style", choices=["cosine", "constant"], default="constant")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--adam-eps", type=float, default=1e-5)
    parser.add_argument("--aux-loss-coeff", type=float, default=1e-3)
    parser.add_argument(
        "--norm-topk-prob",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize selected expert weights; enabled to match the Megatron TE flow.",
    )
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument(
        "--dp-size",
        type=int,
        default=None,
        help="Dense/data-parallel size; must equal the distributed world size.",
    )
    parser.add_argument(
        "--expert-backend",
        choices=["sequential", "te_grouped", "sonic"],
        default="te_grouped",
    )
    parser.add_argument(
        "--sharding",
        choices=["full_shard", "shard_grad_op"],
        default="full_shard",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fp8-scaling",
        choices=["blockwise2d", "delayed", "dynamic"],
        default="blockwise2d",
    )
    parser.add_argument("--aiter-attn", action="store_true")
    parser.add_argument("--lumen-norm", action="store_true")
    parser.add_argument("--fused-router", action="store_true")
    parser.add_argument("--lumen-moe-dispatch-overlap", action="store_true")
    parser.add_argument("--lumen-moe-global-expert-layout", action="store_true")
    parser.add_argument("--fuse-rope", action="store_true")
    parser.add_argument(
        "--fsdp-fp8-param-storage",
        action="store_true",
        help="Unsupported for full-parameter training; accepted to provide a clear error.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--shuffle-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="Sampler seed; Megatron's cyclic sampler starts at epoch seed 0.",
    )
    parser.add_argument("--distributed-timeout-minutes", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    """Launch Qwen3-30B-A3B FSDP training."""
    args = get_args()
    rank = int(os.environ.get("RANK", 0))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    trainer = FSDPTrainer(args)
    try:
        trainer.train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

