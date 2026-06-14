###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch
import torch.utils._pytree as pytree

from lumen.quantize.fp8_params import quantize_param_to_fp8

__all__ = ["FP8CommTensor", "Blockwise2DFP8Param"]


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


class Blockwise2DFP8Param(torch.Tensor):
    """FSDP2 all-gather wrapper for a **frozen** blockwise2d base weight stored as FP8.

    Unlike :class:`FP8CommTensor` (per-tensor scale, wraps trainable params,
    *dequantizes* back to BF16 after all-gather so the GEMM re-quantizes), this
    wraps a LoRA-frozen base weight that was quantized **once** to FP8 with a 2D
    ``(N/block, K/block)`` tile scale.  Because the weight never changes:

      - ``fsdp_pre_all_gather`` hands over the local FP8 shard **and** its scale
        rows verbatim — **no re-quantization** per step (kills the elementwise
        re-quant that dominates 70B);
      - ``fsdp_post_all_gather`` concatenates the gathered FP8 + scale and keeps
        them **FP8** (no dequant), returning a ``Blockwise2DFP8Param`` so the
        forward feeds the FP8 GEMM directly.

    Both inner tensors shard along dim 0; the FP8 weight ``(N, K)`` and the scale
    ``(N/block, K/block)`` stay consistent iff ``N % (block * world_size) == 0``
    (so a rank's FP8 rows map exactly onto its scale rows).
    """

    _fp8_dtype: torch.dtype
    _block_size: int

    @staticmethod
    def __new__(cls, fp8_data: torch.Tensor, scale: torch.Tensor, orig_dtype, block_size: int = 128):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            fp8_data.shape,
            dtype=orig_dtype,
            device=fp8_data.device,
            requires_grad=False,
        )

    def __init__(self, fp8_data: torch.Tensor, scale: torch.Tensor, orig_dtype, block_size: int = 128):
        self._fp8 = fp8_data
        self._scale = scale
        self._fp8_dtype = fp8_data.dtype
        self._orig_dtype = orig_dtype
        self._block_size = block_size

    def __repr__(self):
        return (
            f"Blockwise2DFP8Param(shape={list(self.shape)}, fp8={self._fp8_dtype}, "
            f"scale={list(self._scale.shape)}, block={self._block_size})"
        )

    def __tensor_flatten__(self):
        return ["_fp8", "_scale"], {"orig_dtype": self._orig_dtype, "block_size": self._block_size}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        return cls(
            inner_tensors["_fp8"],
            inner_tensors["_scale"],
            metadata["orig_dtype"],
            metadata["block_size"],
        )

    @staticmethod
    def fsdp_pre_all_gather(tensor) -> tuple[tuple[torch.Tensor, ...], dict]:
        # Frozen + pre-quantized: ship the FP8 shard and its scale rows as-is.
        return (tensor._fp8, tensor._scale), {
            "orig_dtype": tensor._orig_dtype,
            "block_size": tensor._block_size,
        }

    @staticmethod
    def fsdp_post_all_gather(
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: dict,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        fp8_full, scale_full = all_gather_outputs
        if out is not None:
            # In-place reconstruction path: refresh the existing subclass's inners.
            assert isinstance(out, Blockwise2DFP8Param)
            out._fp8.copy_(fp8_full)
            out._scale.copy_(scale_full)
            return
        result = Blockwise2DFP8Param(
            fp8_full, scale_full, metadata["orig_dtype"], metadata["block_size"]
        )
        # Second element: inner tensors FSDP2 may free after copy-in.
        return result, (fp8_full, scale_full)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}

        # Most FSDP2 plumbing ops (copy_/view/detach/clone/...) act on the FP8
        # storage; unwrap to _fp8 and re-wrap with the same scale when the result
        # is a plain tensor derived from this param.
        def _unwrap(x):
            return x._fp8 if isinstance(x, Blockwise2DFP8Param) else x

        source = next(
            (a for a in pytree.tree_leaves(args) if isinstance(a, Blockwise2DFP8Param)),
            None,
        )
        unwrapped_args = pytree.tree_map(_unwrap, args)
        unwrapped_kwargs = pytree.tree_map(_unwrap, kwargs)
        result = func(*unwrapped_args, **unwrapped_kwargs)
        if (
            source is not None
            and isinstance(result, torch.Tensor)
            and not isinstance(result, Blockwise2DFP8Param)
            and result.shape == source._fp8.shape
        ):
            return Blockwise2DFP8Param(
                result, source._scale, source._orig_dtype, source._block_size
            )
        return result
