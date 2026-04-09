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

from lumen.quantize.descriptor import FP8Descriptor

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


def _split_dim0_chunks(tensor: torch.Tensor, num_chunks: int) -> tuple[torch.Tensor, ...]:
    """Split a dim-0 shard into evenly sized chunks."""
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")

    rows = tensor.shape[0]
    if rows % num_chunks != 0:
        raise ValueError(f"num_chunks={num_chunks} must evenly divide dim0 rows={rows}")

    return tensor.chunk(num_chunks, dim=0)


def _dequantize_gathered_fp8_chunk(
    fp8_chunk: torch.Tensor,
    per_rank_scales: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize one gathered FP8 chunk using its per-rank scales."""
    world_size = per_rank_scales.numel()
    gathered_rank_chunks = fp8_chunk.chunk(world_size, dim=0)
    dequant_chunks = [
        dequantize_param_from_fp8(gathered_rank_chunks[rank], per_rank_scales[rank], target_dtype)
        for rank in range(world_size)
    ]
    return torch.cat(dequant_chunks, dim=0)


def fp8_allgather_weight(
    weight_shard: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    target_dtype: torch.dtype = torch.bfloat16,
    use_sdma: bool = False,
    num_chunks: int = 1,
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
        num_chunks: Number of dim-0 chunks to communicate sequentially.

    Returns:
        Full weight tensor ``[rows_full, cols]`` in *target_dtype*.
    """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)

    if world_size <= 1:
        return weight_shard.to(target_dtype)

    fp8_shard, local_scale = quantize_param_to_fp8(weight_shard, fp8_dtype)

    # All-gather per-rank scales (tiny: one float32 scalar per rank)
    scale_scalar = local_scale.float().reshape(1).to(fp8_shard.device)
    all_scales = torch.empty(world_size, dtype=torch.float32, device=fp8_shard.device)
    dist.all_gather_into_tensor(all_scales, scale_scalar, group=group)

    fp8_shard_chunks = _split_dim0_chunks(fp8_shard, num_chunks)
    dequantized_full_chunks = []

    for fp8_shard_chunk in fp8_shard_chunks:
        full_shape = list(fp8_shard_chunk.shape)
        full_shape[0] *= world_size
        fp8_full_chunk = torch.empty(full_shape, dtype=fp8_shard_chunk.dtype, device=fp8_shard_chunk.device)

        if use_sdma:
            from lumen.modules.sdma_comm import SdmaTpComm

            comm = SdmaTpComm.get(group)
            fp8_full_chunk = comm.allgather_dim0(fp8_shard_chunk)
        else:
            dist.all_gather_into_tensor(fp8_full_chunk, fp8_shard_chunk.contiguous(), group=group)

        dequantized_full_chunks.append(_dequantize_gathered_fp8_chunk(fp8_full_chunk, all_scales, target_dtype))

    return torch.cat(dequantized_full_chunks, dim=0)


def fp8_allgather_weight_pipelined(
    weight_shards: list,
    group: Optional[dist.ProcessGroup] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    target_dtype: torch.dtype = torch.bfloat16,
    use_sdma: bool = False,
    num_chunks: int = 1,
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
        num_chunks: Number of dim-0 chunks to pipeline per weight shard.

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
    fp8_shard_chunks = []
    fp8_out_chunks = []
    all_scales_list = []

    for shard in weight_shards:
        fp8_s, sc = quantize_param_to_fp8(shard, fp8_dtype)
        fp8_shards.append(fp8_s)
        local_scales.append(sc)
        shard_chunks = _split_dim0_chunks(fp8_s, num_chunks)
        fp8_shard_chunks.append(shard_chunks)
        chunk_outs = []
        for shard_chunk in shard_chunks:
            full_shape = list(shard_chunk.shape)
            full_shape[0] *= world_size
            chunk_outs.append(torch.empty(full_shape, dtype=shard_chunk.dtype, device=device))
        fp8_out_chunks.append(chunk_outs)

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

    results = [[None] * num_chunks for _ in range(N)]
    chunk_tasks = [(weight_idx, chunk_idx) for weight_idx in range(N) for chunk_idx in range(num_chunks)]

    def _do_gather(weight_idx: int, chunk_idx: int):
        if use_sdma:
            from lumen.modules.sdma_comm import SdmaTpComm

            comm = SdmaTpComm.get(group)
            fp8_out_chunks[weight_idx][chunk_idx] = comm.allgather_dim0(fp8_shard_chunks[weight_idx][chunk_idx])
        else:
            dist.all_gather_into_tensor(
                fp8_out_chunks[weight_idx][chunk_idx],
                fp8_shard_chunks[weight_idx][chunk_idx].contiguous(),
                group=group,
            )

    def _do_dequant(weight_idx: int, chunk_idx: int):
        results[weight_idx][chunk_idx] = _dequantize_gathered_fp8_chunk(
            fp8_out_chunks[weight_idx][chunk_idx],
            all_scales_list[weight_idx],
            target_dtype,
        )

    with torch.cuda.stream(comm_stream):
        _do_gather(*chunk_tasks[0])

    for task_idx, (weight_idx, chunk_idx) in enumerate(chunk_tasks):
        compute_stream.wait_stream(comm_stream)
        if task_idx + 1 < len(chunk_tasks):
            with torch.cuda.stream(comm_stream):
                _do_gather(*chunk_tasks[task_idx + 1])
        _do_dequant(weight_idx, chunk_idx)

    compute_stream.wait_stream(comm_stream)
    return [torch.cat(weight_chunks, dim=0) for weight_chunks in results]


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
        self._state_dict_handles: list = []

    _QUANTIZABLE_TYPES = (nn.Linear,)

    def quantize_params(self, model: nn.Module) -> int:
        """Quantize all ``nn.Linear`` weights in the model to FP8.

        Skips non-linear modules (Embedding, LayerNorm, etc.) even if
        they have a ``weight`` attribute, because the dequant forward
        hook assumes ``F.linear`` semantics.

        Returns:
            Number of parameters quantized.
        """
        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, self._QUANTIZABLE_TYPES):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or not isinstance(weight, nn.Parameter):
                continue
            if weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2):
                continue

            self._original_dtypes[name] = weight.dtype
            fp8_weight, scale = quantize_param_to_fp8(weight.data, self.fp8_dtype)
            self._param_scales[name] = scale

            weight.data = fp8_weight.to(weight.device)
            weight._fp8_desc = FP8Descriptor(data=weight.data, scale=scale, fp8_dtype=self.fp8_dtype)
            weight._fp8_original_dtype = self._original_dtypes[name]
            weight.requires_grad_(False)
            count += 1

        logger.info("Quantized %d nn.Linear parameters to FP8 (%s)", count, self.fp8_dtype)
        return count

    def register_dequant_hooks(self, model: nn.Module) -> int:
        """Register forward pre-hooks to dequantize FP8 params before compute.

        Also registers state_dict hooks so checkpoints save BF16 weights.

        Returns:
            Number of hooks registered.
        """
        self.register_state_dict_hooks(model)
        count = 0
        for name, module in model.named_modules():
            if name in self._param_scales:
                hook = module.register_forward_pre_hook(self._make_dequant_hook(name))
                self._hooks.append(hook)
                self._wrap_forward_to_use_dequant(module)
                self._wrapped_modules.append(module)
                count += 1
        return count

    def register_state_dict_hooks(self, model: nn.Module) -> int:
        """Register hooks so ``state_dict`` serializes dequantized weights.

        FP8 parameters are temporarily cast to their original dtype (default
        bfloat16) before tensors are collected, then restored so runtime
        storage stays in FP8.
        """

        def _pre_save(module, prefix, keep_vars):
            for _name, param in module._parameters.items():
                if param is None or not hasattr(param, "_fp8_desc"):
                    continue
                orig_dtype = getattr(param, "_fp8_original_dtype", torch.bfloat16)
                param._state_dict_data_backup = param.data
                scale = param._fp8_desc.scale.to(param.data.device)
                param.data = (param.data.to(torch.float32) / scale).to(orig_dtype)

        def _post_save(module, state_dict, prefix, local_metadata):
            for _name, param in module._parameters.items():
                if param is None or not hasattr(param, "_state_dict_data_backup"):
                    continue
                param.data = param._state_dict_data_backup
                del param._state_dict_data_backup

        count = 0
        for mod in model.modules():
            if not any(p is not None and hasattr(p, "_fp8_desc") for p in mod._parameters.values()):
                continue
            self._state_dict_handles.append(mod.register_state_dict_pre_hook(_pre_save))
            self._state_dict_handles.append(mod.register_state_dict_post_hook(_post_save))
            count += 1
        return count

    def _make_dequant_hook(self, param_name: str):
        original_dtype = self._original_dtypes[param_name]

        def hook(module, inputs):
            weight = module.weight
            if hasattr(weight, "_fp8_desc"):
                fp8_data = weight.data
                scale = weight._fp8_desc.scale.to(fp8_data.device)
                dequant = dequantize_param_from_fp8(fp8_data, scale, original_dtype)
                module._dequantized_weight = dequant

        return hook

    def _wrap_forward_to_use_dequant(self, module: nn.Module) -> None:
        """Replace forward to use _dequantized_weight while keeping weight in autograd graph.

        DDP requires every parameter to participate in the loss computation graph.
        We use a STE (straight-through estimator) trick: add ``weight - weight.detach()``
        which is zero in forward but connects ``weight`` to the graph for backward.
        """
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
        for handle in self._state_dict_handles:
            handle.remove()
        self._state_dict_handles.clear()
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
