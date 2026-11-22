###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""SonicMoE expert adapter for Megatron-Core MoE layers."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, MutableMapping

logger = logging.getLogger(__name__)
_DUAL_PACK_NOTICE = {"logged": False}

import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.utils import get_pg_rank
from torch import nn

from lumen.utils.sonic_moe_handoff import consume_mapped_handoff


def _expert_weights(
    experts: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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

    raise TypeError(
        f"Unsupported Megatron expert implementation: {type(experts).__name__}"
    )


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
    grouped_keys = [f"{prefix}{stem}.weight{index}" for index in range(count)]
    keys = (
        sequential_keys
        if all(key in state_dict for key in sequential_keys)
        else grouped_keys
    )
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
            raise ValueError(
                "Flattened optimizer ranges are not supported for SonicMoE"
            )
        shards = []
        for local_index in range(num_local_experts):
            global_index = ep_rank * num_local_experts + local_index
            expert_tensor = data[local_index].transpose(0, 1)
            if singleton_local_shards:
                checkpoint_key = f"{prefix}experts.{global_index}.{layer_name}.weight"
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
            raise ValueError(
                "SonicMoE integration currently requires --disable-bias-linear"
            )
        if not getattr(self.config, "gated_linear_unit", False):
            raise ValueError(
                "SonicMoE integration currently requires a gated activation (SwiGLU)"
            )
        expert_tp = getattr(self.config, "expert_tensor_parallel_size", 1)
        if expert_tp != 1:
            raise ValueError(
                "SonicMoE integration currently requires expert tensor parallel size 1"
            )

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
        self.precision_recipe = "bf16"
        # Packed MXFP4 expert weights are reused for every microbatch of an optimizer
        # step. Each orientation costs ~0.3x the BF16 weight bytes, so the whole
        # 48-layer cache is ~9 GiB per rank; SONIC_MOE_PACK_CACHE=0 disables it.
        self._packed_weights = None
        self._packed_stamp = None
        self._pack_generation = 0
        self._pack_cache_requested = os.environ.get("SONIC_MOE_PACK_CACHE", "1") != "0"
        # Generation caching requires a proven optimizer + root-load lifecycle.
        # Before installation the public op packs fresh instead of trusting _version.
        self._pack_cache_enabled = False

    def _invalidate_packed_weights(self) -> None:
        self._packed_weights = None
        self._packed_stamp = None
        self._pack_generation = int(getattr(self, "_pack_generation", 0)) + 1

    def _cached_packed_weights(self):
        """Packed w1/w2 for the current weight values, or None when caching is off."""
        if not self._pack_cache_enabled:
            return None
        stamp = (
            int(getattr(self, "_pack_generation", 0)),
            self.w1.data_ptr(),
            self.w1._version,
            self.w2.data_ptr(),
            self.w2._version,
        )
        if self._packed_weights is None or self._packed_stamp != stamp:
            from lumen.ops.moe.mxfp4_sonic import (
                pack_pre_routed_weights,
                pack_pre_routed_weights_dual,
            )

            # One-read dual orientation pack, byte-identical to the 4-call loop; the loop stays
            # the logged fallback in the op layer and R80_DUAL_WEIGHT_PACK=0 restores it.
            if os.environ.get("R80_DUAL_WEIGHT_PACK", "1") == "1":
                if not _DUAL_PACK_NOTICE["logged"]:
                    _DUAL_PACK_NOTICE["logged"] = True
                    logger.info(
                        "R80_DUAL_WEIGHT_PACK=1: using AITER dual-orientation weight pack"
                    )
                self._packed_weights = pack_pre_routed_weights_dual(self.w1, self.w2)
            else:
                self._packed_weights = pack_pre_routed_weights(self.w1, self.w2)
            self._packed_stamp = stamp
        return self._packed_weights

    def mori_mapped_handoff_capability(self) -> tuple[bool, str]:
        """Return whether this instance can consume the optional MORI handoff."""
        if self.precision_recipe not in ("w4a16", "a4w4", "a8w4", "a8w4f"):
            return False, f"precision recipe is {self.precision_recipe!r}, not MXFP4"
        from lumen.ops.moe.mxfp4_sonic import mapped_pre_routed_available

        if not mapped_pre_routed_available():
            return False, "AITER mapped dual-quant API version 1 is unavailable"
        return True, ""

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

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert,
        permuted_probs: torch.Tensor,
    ):
        if isinstance(tokens_per_expert, tuple):
            cpu_counts, gpu_counts = tokens_per_expert
            tokens_per_expert = cpu_counts if cpu_counts is not None else gpu_counts

        handoff = consume_mapped_handoff(
            permuted_local_hidden_states,
            permuted_probs,
            tokens_per_expert,
        )

        if self.precision_recipe in ("w4a16", "a4w4", "a8w4", "a8w4f"):
            from lumen.ops.moe.mxfp4_sonic import (
                pre_routed_quantized,
                pre_routed_quantized_mapped,
            )

            # Preserve the dispatcher's pinned host tensor so AITER can upload
            # counts asynchronously while it validates the same host values.
            counts = (
                tokens_per_expert
                if isinstance(tokens_per_expert, torch.Tensor)
                else torch.as_tensor(tokens_per_expert, dtype=torch.int32)
            )
            native_wgrad = (
                self.precision_recipe == "a4w4"
                and os.environ.get("SONIC_MOE_MXFP4_WGRAD", "0") == "1"
            )
            bf16_main_grad = (
                self.precision_recipe == "a4w4"
                and not native_wgrad
                and handoff is None
                and os.environ.get("SONIC_MOE_BF16_WGRAD_MAIN_GRAD", "0") == "1"
            )
            main_grads = None
            if (native_wgrad or bf16_main_grad) and torch.is_grad_enabled():
                buffers = tuple(
                    getattr(p, "main_grad", None) for p in (self.w1, self.w2)
                )
                if all(buffer is not None for buffer in buffers):
                    main_grads = buffers
                elif any(buffer is not None for buffer in buffers):
                    raise ValueError(
                        "Sonic Wgrad requires both main_grad buffers or neither"
                    )
            common_kwargs = {
                "wgrad_mode": "mxfp4" if native_wgrad else "bf16",
                "main_grads": main_grads,
                "wgrad_fallback": os.environ.get(
                    "SONIC_MOE_MXFP4_WGRAD_FALLBACK", "error"
                ),
                "wgrad_backend": (
                    os.environ.get("SONIC_MOE_MXFP4_WGRAD_BACKEND", "triton")
                    if native_wgrad
                    else "triton"
                ),
            }
            if handoff is not None:
                output = pre_routed_quantized_mapped(
                    self.precision_recipe,
                    handoff.source_tokens,
                    handoff.scores,
                    handoff.counts,
                    handoff.gather_map,
                    self.w1,
                    self.w2,
                    self._cached_packed_weights(),
                    **common_kwargs,
                )
            else:
                output = pre_routed_quantized(
                    self.precision_recipe,
                    permuted_local_hidden_states,
                    permuted_probs.reshape(-1).float(),
                    counts,
                    self.w1,
                    self.w2,
                    self._cached_packed_weights(),
                    **common_kwargs,
                )
            return (
                output,
                None,
            )
        if handoff is not None:
            raise RuntimeError("mapped MORI handoff reached a non-MXFP4 Sonic expert")
        if self.precision_recipe != "bf16":
            raise ValueError(f"Unsupported expert precision: {self.precision_recipe}")

        counts = torch.as_tensor(tokens_per_expert, dtype=torch.int32)
        from aiter.ops.triton.moe.sonicmoe import (
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
    if os.environ.get("LUMEN_MORI_SONIC_MXFP4_HANDOFF", "0") == "1":
        from lumen.modules.mori_sonic_handoff import install_mori_sonic_handoff

        install_mori_sonic_handoff()
    return replaced


__all__ = ["SonicMoEExperts", "replace_megatron_moe_experts"]
