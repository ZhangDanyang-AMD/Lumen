###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""FP8 parameter storage, all-gather communication, and dequant hooks.

**Layer 1** — local FP8 cache + lazy re-quant (wired via
:class:`~lumen.quantize.scaling_manager.ScalingManager`).

**Layer 2** — FP8 all-gather communication: each rank quantizes its
local weight shard to FP8 (1 byte/elem), all-gathers the FP8 shards
(half bandwidth vs BF16), then dequantizes per-shard back to BF16.
Supports NCCL and SDMA backends.
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def quantize_param_to_fp8(
    param: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple:
    """Quantize a parameter tensor to FP8 with per-tensor scale.

    Returns:
        Tuple of (fp8_tensor, scale) where scale is a scalar float32.
    """
    amax = param.abs().amax().clamp(min=1e-12)
    if fp8_dtype == torch.float8_e4m3fn:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
    elif fp8_dtype == torch.float8_e4m3fnuz:
        fp8_max = torch.finfo(torch.float8_e4m3fnuz).max
    else:
        fp8_max = 448.0
    scale = fp8_max / amax
    fp8_param = (param.float() * scale).to(fp8_dtype)
    return fp8_param, scale


def dequantize_param_from_fp8(
    fp8_param: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 parameter back to target dtype."""
    return (fp8_param.to(torch.float32) / scale).to(target_dtype)


def fp8_allgather_weight(
    weight_shard: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    target_dtype: torch.dtype = torch.bfloat16,
    use_sdma: bool = False,
) -> torch.Tensor:
    """All-gather a weight shard via FP8, dequantize per-shard to *target_dtype*.

    This is the "Layer 2" FP8 param all-gather: each rank quantizes its
    local shard to FP8, communicates only the FP8 bytes (half the volume
    of BF16), then each rank independently dequantizes back to full
    precision.  Per-rank scales are also all-gathered so that each
    chunk is dequantized with its originating rank's scale.

    Args:
        weight_shard: Local weight shard ``[rows_local, cols]`` in BF16/FP32.
        group: Process group for the all-gather. Defaults to WORLD.
        fp8_dtype: Target FP8 dtype for quantization.
        target_dtype: Dtype to dequantize into after gathering.
        use_sdma: Use MORI SDMA allgather instead of NCCL.

    Returns:
        Full weight tensor ``[rows_full, cols]`` in *target_dtype*.
    """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)

    if world_size <= 1:
        return weight_shard.to(target_dtype)

    fp8_shard, local_scale = quantize_param_to_fp8(weight_shard, fp8_dtype)

    # All-gather FP8 data shards
    full_shape = list(fp8_shard.shape)
    full_shape[0] *= world_size
    fp8_full = torch.empty(full_shape, dtype=fp8_shard.dtype, device=fp8_shard.device)

    if use_sdma:
        from lumen.modules.sdma_comm import SdmaTpComm

        comm = SdmaTpComm.get(group)
        fp8_full = comm.allgather_dim0(fp8_shard)
    else:
        dist.all_gather_into_tensor(fp8_full, fp8_shard.contiguous(), group=group)

    # All-gather per-rank scales (tiny: one float32 scalar per rank)
    scale_scalar = local_scale.float().reshape(1).to(fp8_shard.device)
    all_scales = torch.empty(world_size, dtype=torch.float32, device=fp8_shard.device)
    dist.all_gather_into_tensor(all_scales, scale_scalar, group=group)

    chunks = fp8_full.chunk(world_size, dim=0)
    dequant_chunks = [dequantize_param_from_fp8(chunks[i], all_scales[i], target_dtype) for i in range(world_size)]
    return torch.cat(dequant_chunks, dim=0)


def fp8_allgather_weight_pipelined(
    weight_shards: list,
    group: Optional[dist.ProcessGroup] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    target_dtype: torch.dtype = torch.bfloat16,
    use_sdma: bool = False,
) -> list:
    """Pipelined FP8 all-gather for multiple weight shards.

    Overlaps allgather(i+1) on a comm stream with dequant(i) on the
    compute stream, hiding dequant latency behind communication.

    Args:
        weight_shards: List of local weight shards to all-gather.
        group: Process group. Defaults to WORLD.
        fp8_dtype: Target FP8 dtype.
        target_dtype: Dtype to dequantize into after gathering.
        use_sdma: Use MORI SDMA allgather instead of NCCL.

    Returns:
        List of full (dequantized) weight tensors.
    """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)

    if world_size <= 1:
        return [s.to(target_dtype) for s in weight_shards]

    N = len(weight_shards)
    device = weight_shards[0].device
    fp8_shards = []
    local_scales = []
    fp8_outs = []
    all_scales_list = []

    for shard in weight_shards:
        fp8_s, sc = quantize_param_to_fp8(shard, fp8_dtype)
        fp8_shards.append(fp8_s)
        local_scales.append(sc)
        full_shape = list(fp8_s.shape)
        full_shape[0] *= world_size
        fp8_outs.append(torch.empty(full_shape, dtype=fp8_s.dtype, device=device))

    # All-gather per-rank scales for each weight (batched, tiny payload)
    for sc in local_scales:
        sc_tensor = sc.float().reshape(1).to(device)
        gathered = torch.empty(world_size, dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(gathered, sc_tensor, group=group)
        all_scales_list.append(gathered)

    comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device)
    # Keep NCCL subgroup collectives on `comm_stream` ordered after the scale
    # all-gathers already enqueued on the current stream.
    comm_stream.wait_stream(compute_stream)

    results = [None] * N

    def _do_gather(idx):
        if use_sdma:
            from lumen.modules.sdma_comm import SdmaTpComm

            comm = SdmaTpComm.get(group)
            fp8_outs[idx] = comm.allgather_dim0(fp8_shards[idx])
        else:
            dist.all_gather_into_tensor(fp8_outs[idx], fp8_shards[idx].contiguous(), group=group)

    def _do_dequant(idx):
        chunks = fp8_outs[idx].chunk(world_size, dim=0)
        per_rank_scales = all_scales_list[idx]
        dq = [dequantize_param_from_fp8(chunks[r], per_rank_scales[r], target_dtype) for r in range(world_size)]
        results[idx] = torch.cat(dq, dim=0)

    with torch.cuda.stream(comm_stream):
        _do_gather(0)

    for i in range(N):
        compute_stream.wait_stream(comm_stream)
        if i + 1 < N:
            with torch.cuda.stream(comm_stream):
                _do_gather(i + 1)
        _do_dequant(i)

    compute_stream.wait_stream(comm_stream)
    return results


class FP8ParamManager:
    """Manages FP8 parameter quantization and all-gather hooks.

    When enabled, linear layer weights are stored in FP8 and
    dequantized on-the-fly before computation. All-gather operations
    communicate FP8 data (half the bandwidth of BF16).

    Args:
        fp8_dtype: Target FP8 dtype for parameters.
    """

    def __init__(self, fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        self.fp8_dtype = fp8_dtype
        self._param_scales: dict = {}
        self._original_dtypes: dict = {}
        self._hooks: list = []
        self._wrapped_modules: list = []

    def quantize_params(self, model: nn.Module) -> int:
        """Quantize all linear weights in the model to FP8.

        Returns:
            Number of parameters quantized.
        """
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or hasattr(module, "weight"):
                weight = getattr(module, "weight", None)
                if weight is None or not isinstance(weight, nn.Parameter):
                    continue
                if weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2):
                    continue

                self._original_dtypes[name] = weight.dtype
                fp8_weight, scale = quantize_param_to_fp8(weight.data, self.fp8_dtype)
                self._param_scales[name] = scale

                weight.data = fp8_weight.to(weight.device)
                weight._fp8_scale = scale
                weight._fp8_dtype = self.fp8_dtype
                weight._original_dtype = self._original_dtypes[name]
                count += 1

        logger.info("Quantized %d parameters to FP8 (%s)", count, self.fp8_dtype)
        return count

    def register_dequant_hooks(self, model: nn.Module) -> int:
        """Register forward pre-hooks to dequantize FP8 params before compute.

        Returns:
            Number of hooks registered.
        """
        count = 0
        for name, module in model.named_modules():
            if name in self._param_scales:
                hook = module.register_forward_pre_hook(self._make_dequant_hook(name))
                self._hooks.append(hook)
                self._wrap_forward_to_use_dequant(module)
                self._wrapped_modules.append(module)
                count += 1
        return count

    def _make_dequant_hook(self, param_name: str):
        original_dtype = self._original_dtypes[param_name]

        def hook(module, inputs):
            weight = module.weight
            if hasattr(weight, "_fp8_scale"):
                fp8_data = weight.data
                dequant = dequantize_param_from_fp8(fp8_data, weight._fp8_scale, original_dtype)
                module._dequantized_weight = dequant

        return hook

    def _wrap_forward_to_use_dequant(self, module: nn.Module) -> None:
        """Replace forward to use _dequantized_weight when set by pre-hook."""
        if hasattr(module, "_fp8_original_forward"):
            return
        original_forward = module.forward

        def fp8_aware_forward(*args, **kwargs):
            if hasattr(module, "_dequantized_weight"):
                weight = module._dequantized_weight
            else:
                weight = module.weight
            return torch.nn.functional.linear(args[0], weight, module.bias)

        module._fp8_original_forward = original_forward
        module.forward = fp8_aware_forward

    def remove_hooks(self):
        """Remove all registered hooks and restore original forwards."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for module in self._wrapped_modules:
            if hasattr(module, "_fp8_original_forward"):
                module.forward = module._fp8_original_forward
                del module._fp8_original_forward
        self._wrapped_modules.clear()

    def memory_savings_bytes(self, model: nn.Module) -> int:
        """Estimate memory savings from FP8 params (bytes)."""
        saved = 0
        for name in self._param_scales:
            parts = name.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p, None)
                if mod is None:
                    break
            if mod is not None and hasattr(mod, "weight"):
                numel = mod.weight.numel()
                saved += numel  # saving 1 byte per element (2 bytes BF16 -> 1 byte FP8)
        return saved
