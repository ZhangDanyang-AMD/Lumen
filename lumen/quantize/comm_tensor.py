###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch
import torch.utils._pytree as pytree

from lumen.quantize.fp8_params import quantize_param_to_fp8

__all__ = ["FP8CommTensor"]


class FP8CommTensor(torch.Tensor):
    """Thin tensor subclass for FSDP2 FP8 all-gather communication.

    Wraps a BF16 parameter and provides FSDP2 hooks that quantize the
    local shard to FP8 before all-gather, then dequantize after.  This
    halves the all-gather bandwidth without changing optimizer or
    gradient behaviour.
    """

    _fp8_dtype: torch.dtype

    @staticmethod
    def __new__(cls, data: torch.Tensor, fp8_dtype: torch.dtype):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
            requires_grad=data.requires_grad,
        )

    def __init__(self, data: torch.Tensor, fp8_dtype: torch.dtype):
        self._data = data
        self._fp8_dtype = fp8_dtype

    def __repr__(self):
        return f"FP8CommTensor(shape={list(self.shape)}, dtype={self.dtype}, fp8_dtype={self._fp8_dtype})"

    def __tensor_flatten__(self):
        return ["_data"], {"fp8_dtype": self._fp8_dtype}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        return cls(inner_tensors["_data"], metadata["fp8_dtype"])

    @staticmethod
    def fsdp_pre_all_gather(tensor) -> tuple[tuple[torch.Tensor, ...], dict]:
        fp8_data, scale = quantize_param_to_fp8(tensor._data, tensor._fp8_dtype)
        return (fp8_data, scale.float().reshape(1)), {}

    @staticmethod
    def fsdp_post_all_gather(
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: dict,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fp8_gathered, scales_gathered = all_gather_outputs
        world_size = scales_gathered.numel()
        chunks = fp8_gathered.chunk(world_size, dim=0)
        dequant_chunks = [
            (chunk.to(torch.float32) / scales_gathered[i]).to(param_dtype) for i, chunk in enumerate(chunks)
        ]
        result = torch.cat(dequant_chunks, dim=0)
        if out is not None:
            out.copy_(result)
            return out
        return result

    _FSDP2_SAFE_OPS = {
        torch.ops.aten.copy_.default,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.view.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.empty_like.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.detach.default,
    }

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}

        def _unwrap(x):
            return x._data if isinstance(x, FP8CommTensor) else x

        if func in cls._FSDP2_SAFE_OPS:
            unwrapped_args = pytree.tree_map(_unwrap, args)
            unwrapped_kwargs = pytree.tree_map(_unwrap, kwargs)
            result = func(*unwrapped_args, **unwrapped_kwargs)
            if isinstance(result, torch.Tensor) and not isinstance(result, FP8CommTensor):
                source = next(
                    (a for a in pytree.tree_leaves(args) if isinstance(a, FP8CommTensor)),
                    None,
                )
                if source is not None:
                    return FP8CommTensor(result, source._fp8_dtype)
            return result

        unwrapped_args = pytree.tree_map(_unwrap, args)
        unwrapped_kwargs = pytree.tree_map(_unwrap, kwargs)
        return func(*unwrapped_args, **unwrapped_kwargs)
