###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import logging
from collections import defaultdict, deque
from typing import Dict, Optional, Set

import torch
import torch.nn as nn

from lumen.quantize.config import (
    AmaxAlgo,
    QuantConfig,
    QuantFormat,
    ScalingType,
    _get_float8_e4m3,
)

logger = logging.getLogger(__name__)


def _get_quant_ops():
    """Lazy import to avoid circular dependency with lumen.ops."""
    from lumen.ops.quantize import (
        convert_from_mxfp8,
        convert_to_mxfp8,
        quant_fp8_blockwise_impl,
    )

    return convert_to_mxfp8, convert_from_mxfp8, quant_fp8_blockwise_impl


# ---------------------------------------------------------------------------
# Gradient quantization helpers
# ---------------------------------------------------------------------------

GRAD_QUANT_TYPES = (None, "fp8", "mxfp8", "fp4")


def _round_to_fp8(tensor: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    """Per-tensor FP8 quant-dequant round-trip."""
    orig_dtype = tensor.dtype
    amax = tensor.abs().amax().clamp(min=1e-12)
    fp8_max = torch.finfo(fp8_dtype).max
    scale = fp8_max / amax
    tensor_fp8 = (tensor.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return tensor_fp8.to(orig_dtype) / scale


def _round_to_mxfp8(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Microscaling FP8 quant-dequant round-trip."""
    orig_dtype = tensor.dtype
    orig_shape = tensor.shape

    flat = tensor.reshape(-1, orig_shape[-1]).contiguous()
    M, N = flat.shape
    pad_n = (block_size - N % block_size) % block_size
    if pad_n > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_n))

    data_bf16 = flat.to(torch.bfloat16)
    convert_to_mxfp8, convert_from_mxfp8, _ = _get_quant_ops()
    data_lp, scales = convert_to_mxfp8(data_bf16, block_size=block_size, axis=-1)
    data_hp = convert_from_mxfp8(
        data_lp,
        scales,
        output_dtype=torch.bfloat16,
        block_size=block_size,
        axis=-1,
    )

    if pad_n > 0:
        data_hp = data_hp[:, :N]

    return data_hp.reshape(orig_shape).to(orig_dtype)


class ScalingManager:
    """Tracks per-tensor scaling factors across training iterations.

    Accepts either a :class:`QuantConfig` or legacy keyword arguments.

    Examples::

        # With QuantConfig
        mgr = ScalingManager(QuantConfig(format=QuantFormat.MXFP8,
                                         scaling=ScalingType.BLOCKWISE))

        # MLPerf-style: most_recent amax, history=4
        mgr = ScalingManager(QuantConfig(amax_algo=AmaxAlgo.MOST_RECENT,
                                         history_len=4))

        # Legacy style (still supported)
        mgr = ScalingManager(recipe="delayed")
    """

    def __init__(
        self,
        config: Optional[QuantConfig] = None,
        *,
        recipe: Optional[str] = None,
        history_len: int = 16,
        block_size: int = 32,
        fp8_dtype: Optional[torch.dtype] = None,
    ):
        if config is not None:
            self.config = config
        elif recipe is not None:
            self.config = QuantConfig.from_str(
                format="mxfp8" if recipe == "mxfp8" else "fp8_e4m3",
                scaling=recipe if recipe not in ("mxfp8", "block") else "blockwise",
                block_size=block_size,
                history_len=history_len,
            )
        else:
            self.config = QuantConfig(block_size=block_size, history_len=history_len)

        if fp8_dtype is None:
            from lumen.quantize.config import _get_float8_e4m3

            fp8_dtype = _get_float8_e4m3()

        self.fp8_dtype = self.config.torch_dtype or fp8_dtype
        self.fp8_dtype_bwd = self.config.torch_dtype_bwd or self.fp8_dtype
        self._fp8_max = self.config.fp8_max
        self._fp8_max_bwd = self.config.fp8_max_bwd
        self._margin = self.config.margin
        self.amax_history = defaultdict(lambda: deque(maxlen=self.config.history_len))
        self.scale_cache = {}
        self._dp_group = None
        self._use_sdma = config.use_sdma if config else False

        # FP8 param lifecycle state
        self._fp8_param_ids: Set[str] = set()
        self._fp8_params: Dict[str, nn.Parameter] = {}
        self._fp8_param_cache: Dict[str, tuple] = {}
        self._fp8_param_stale: Set[str] = set()
        self._fp8_step_counter: int = -1
        self._sdma_allgather = None

    @property
    def recipe(self) -> str:
        return self.config.recipe

    def set_dp_group(self, group) -> None:
        """Set the data-parallel process group for ``reduce_amax``."""
        self._dp_group = group

    def _compute_scale(self, amax: torch.Tensor, fp8_max: float) -> torch.Tensor:
        """Compute the quantization scale from *amax*, accounting for margin.

        Formula: ``sf = (fp8_max / amax) / (2 ** margin)``,
        returned as ``amax / (fp8_max / (2 ** margin))`` so that the caller
        can divide by the scale to quantize.
        Always returns fp32 to satisfy hipb_mm's scale dtype requirement.
        """
        amax = amax.float()
        effective_max = fp8_max / (2**self._margin)
        scale = amax / effective_max
        scale = torch.where(amax > 0.0, scale, torch.ones_like(scale))
        return scale

    def get_scale(self, tensor_id: str, tensor: torch.Tensor, *, backward: bool = False):
        """Return the scale factor for this tensor (None for block/mxfp8/per_token/none)."""
        recipe = self.recipe
        fp8_max = self._fp8_max_bwd if backward else self._fp8_max

        if recipe == "none":
            return None

        if recipe == "delayed":
            history = self.amax_history[tensor_id]
            if len(history) == 0:
                amax = tensor.abs().amax()
            elif self.config.amax_algo == AmaxAlgo.MOST_RECENT:
                amax = history[-1].to(device=tensor.device)
            else:
                amax = torch.stack(list(history)).amax().to(device=tensor.device)

            if self.config.reduce_amax and self._dp_group is not None:
                torch.distributed.all_reduce(
                    amax,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self._dp_group,
                )

            return self._compute_scale(amax, fp8_max)
        elif recipe == "dynamic":
            amax = tensor.abs().amax()
            if self.config.reduce_amax and self._dp_group is not None:
                torch.distributed.all_reduce(
                    amax,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self._dp_group,
                )
            return self._compute_scale(amax, fp8_max)
        elif recipe in ("blockwise", "mxfp8", "per_token"):
            return None

    def update_amax(self, tensor_id: str, tensor: torch.Tensor):
        """Record amax for delayed scaling (tensor-based, no .item() sync)."""
        self.amax_history[tensor_id].append(tensor.detach().abs().amax())

    def quantize(self, tensor_id: str, tensor: torch.Tensor, *, backward: bool = False):
        """Quantize tensor. Returns (quantized_tensor, scale).

        When FP8 param mode is active for *tensor_id*, this method
        returns the cached (and possibly lazily re-quantized) FP8 weight
        instead of quantizing on-the-fly.

        When *backward* is True and the format is HYBRID, E5M2 dtype and its
        corresponding FP8_MAX are used instead of the forward-pass values.
        """
        if not backward:
            cached = self.get_fp8_param_cached(tensor_id, tensor)
            if cached is not None:
                return cached

        return self._quantize_core(tensor_id, tensor, backward=backward)

    # ------------------------------------------------------------------
    # FP8 parameter lifecycle (replaces standalone FP8ParamManager)
    # ------------------------------------------------------------------

    @property
    def num_fp8_params(self) -> int:
        return len(self._fp8_params)

    def register_fp8_param(self, tensor_id: str, param: nn.Parameter) -> None:
        """Register a parameter for FP8 lifecycle management."""
        self._fp8_param_ids.add(tensor_id)
        self._fp8_params[tensor_id] = param

    def mark_fp8_params_stale(self) -> None:
        """Mark all cached FP8 params as stale.

        Call after ``optimizer.step()`` so that the next forward pass
        re-quantizes from the updated master weights.
        """
        self._fp8_param_stale.update(self._fp8_param_ids)

    def check_and_mark_fp8_stale(self, current_step: int) -> None:
        """Mark stale only when the training step counter advances.

        Safe to call on every micro-batch; actual staleness is flagged
        only once per optimizer step.
        """
        if current_step > self._fp8_step_counter:
            self._fp8_step_counter = current_step
            self.mark_fp8_params_stale()

    def quantize_fp8_params(self) -> None:
        """Eagerly re-quantize all registered master params to FP8.

        Populates the internal FP8 param cache.  Equivalent to
        Megatron's ``cast_master_weights_to_fp8``.

        For FSDP workloads prefer :meth:`mark_fp8_params_stale` (lazy)
        because the full unsharded param is only available during
        FSDP's forward pass.
        """
        if not self._fp8_params:
            return

        with torch.no_grad():
            for tid, param in self._fp8_params.items():
                self._fp8_param_stale.discard(tid)
                fp8_data, scale = self._quantize_core(tid, param.data)
                self._fp8_param_cache[tid] = (
                    fp8_data.detach() if isinstance(fp8_data, torch.Tensor) else fp8_data,
                    scale.detach() if isinstance(scale, torch.Tensor) else scale,
                )

        if self._dp_group is not None:
            if self._use_sdma:
                self._reduce_fp8_amax_sdma()
            else:
                self._reduce_fp8_amax_dist()

    def get_fp8_param_cached(self, tensor_id: str, tensor: torch.Tensor):
        """Return cached FP8 weight, re-quantizing lazily if stale.

        Called from the forward path instead of :meth:`quantize` when
        FP8 param mode is active for *tensor_id*.
        """
        if tensor_id not in self._fp8_param_ids:
            return None

        cached = self._fp8_param_cache.get(tensor_id)
        if cached is not None and tensor_id not in self._fp8_param_stale:
            return cached

        self._fp8_param_stale.discard(tensor_id)
        fp8_data, scale = self._quantize_core(tensor_id, tensor)
        entry = (
            fp8_data.detach() if isinstance(fp8_data, torch.Tensor) else fp8_data,
            scale.detach() if isinstance(scale, torch.Tensor) else scale,
        )
        self._fp8_param_cache[tensor_id] = entry
        return entry

    def _collect_amaxes(self):
        """Gather the latest amax values from all registered FP8 params."""
        amaxes = []
        tensor_ids = []
        for tid in self._fp8_params:
            history = self.amax_history.get(tid)
            if history and len(history) > 0:
                amaxes.append(history[-1])
                tensor_ids.append(tid)
        return amaxes, tensor_ids

    def _scatter_amaxes(self, max_amaxes, tensor_ids) -> None:
        """Write reduced amax values back into the history deques."""
        for i, tid in enumerate(tensor_ids):
            history = self.amax_history[tid]
            if len(history) > 0:
                history[-1] = max_amaxes[i]

    def _reduce_fp8_amax_sdma(self) -> None:
        """All-reduce (MAX) per-param amax across DP ranks via mori SDMA."""
        from lumen.ops.sdma import SdmaAllgather, sdma_allgather_max

        amaxes, tensor_ids = self._collect_amaxes()
        if not amaxes:
            return

        device = amaxes[0].device
        packed = torch.stack([a.to(device) for a in amaxes]).contiguous()

        if self._sdma_allgather is None:
            self._sdma_allgather = SdmaAllgather()
            logger.info(
                "ScalingManager: created SdmaAllgather handle (%d amax elements)",
                packed.numel(),
            )

        max_amaxes = sdma_allgather_max(packed, self._sdma_allgather)
        self._scatter_amaxes(max_amaxes, tensor_ids)

    def _reduce_fp8_amax_dist(self) -> None:
        """All-reduce (MAX) per-param amax across DP ranks via torch.distributed."""
        amaxes, tensor_ids = self._collect_amaxes()
        if not amaxes:
            return

        device = amaxes[0].device
        packed = torch.stack([a.to(device) for a in amaxes]).contiguous()
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.MAX, group=self._dp_group)
        self._scatter_amaxes(packed, tensor_ids)

    def register_fp8_optimizer_hook(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> "ScalingManager":
        """Register a post-step hook that marks FP8 params stale.

        The next forward pass will lazily re-quantize from the updated
        master weights — no explicit call needed in the training loop.
        """

        def _post_step(_opt, _args, _kwargs):
            self.mark_fp8_params_stale()

        optimizer.register_step_post_hook(_post_step)
        logger.info("ScalingManager: registered FP8 optimizer post-step hook")
        return self

    def enable_fp8_params(self, model: nn.Module) -> "ScalingManager":
        """Scan *model* for quantized layers and register their weights.

        Looks for modules with ``_quant_tensor_id`` (set during
        ``quant.enable()`` patching) and registers their ``.weight``
        parameters for FP8 lifecycle management.
        """
        count = 0
        for _name, module in model.named_modules():
            tensor_id = getattr(module, "_quant_tensor_id", None)
            if tensor_id is not None and hasattr(module, "weight"):
                self.register_fp8_param(tensor_id, module.weight)
                count += 1
        if count > 0:
            self.quantize_fp8_params()
            logger.info(
                "ScalingManager: FP8 param mode enabled (%d params)",
                count,
            )
        return self

    # ------------------------------------------------------------------
    # Quantize core (shared by both on-the-fly and FP8 param paths)
    # ------------------------------------------------------------------

    def _quantize_core(self, tensor_id: str, tensor: torch.Tensor, *, backward: bool = False):
        """Core quantization logic (delegates to format-specific paths)."""
        if self.config.scaling == ScalingType.NONE:
            return tensor, None

        scale = self.get_scale(tensor_id, tensor, backward=backward)
        fp8_max = self._fp8_max_bwd if backward else self._fp8_max
        dtype = self.fp8_dtype_bwd if backward else self.fp8_dtype

        if scale is None and self.config.format == QuantFormat.MXFP8:
            convert_to_mxfp8, _, _ = _get_quant_ops()
            return convert_to_mxfp8(
                tensor,
                block_size=self.config.block_size,
                axis=-1,
                float8_dtype_pt=dtype,
            )

        if scale is None and self.config.scaling == ScalingType.BLOCKWISE:
            _, _, quant_fp8_blockwise_impl = _get_quant_ops()
            orig_shape = tensor.shape
            flat = tensor.reshape(-1, orig_shape[-1])
            fp8_tensor, fp8_scales = quant_fp8_blockwise_impl(
                flat.contiguous(),
                dtype=dtype,
                axis=1,
                block_size=self.config.block_size,
            )
            return fp8_tensor.view(orig_shape), fp8_scales

        if scale is None and self.config.scaling == ScalingType.PER_TOKEN:
            orig_shape = tensor.shape
            flat = tensor.reshape(-1, orig_shape[-1])
            row_max = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            row_scale = row_max / fp8_max
            fp8_tensor = (flat / row_scale).clamp(-fp8_max, fp8_max).to(dtype)
            return fp8_tensor.view(orig_shape), row_scale

        fp8_tensor = (tensor * (1.0 / scale)).clamp(-fp8_max, fp8_max).to(dtype)
        self.update_amax(tensor_id, tensor)
        return fp8_tensor, scale

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all tracked state (e.g. after warmup)."""
        self.amax_history.clear()
        self.scale_cache.clear()
        self._fp8_param_cache.clear()
        self._fp8_param_stale.clear()
        self._fp8_step_counter = -1

    # ------------------------------------------------------------------
    # Gradient quantization
    # ------------------------------------------------------------------

    def quantize_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply gradient quantization based on ``self.config.quantize_grad``.

        Performs a quant-dequant round-trip that reduces the gradient tensor
        to the representable precision of the configured low-precision format,
        then returns a tensor with the original shape and dtype.

        The format is read from ``self.config.quantize_grad`` (one of
        ``None``, ``"fp8"``, ``"mxfp8"``, ``"fp4"``).  When ``None``, the
        tensor is returned unchanged.
        """
        return self.quantize_grad_tensor(
            tensor,
            self.config.quantize_grad,
            fp8_dtype=self.fp8_dtype,
            block_size=self.config.block_size,
        )

    @staticmethod
    def quantize_grad_tensor(
        tensor: torch.Tensor,
        grad_quant_type: Optional[str],
        fp8_dtype: Optional[torch.dtype] = None,
        block_size: int = 32,
    ) -> torch.Tensor:
        """Stateless gradient quant-dequant round-trip.

        Can be called without a :class:`ScalingManager` instance.

        Args:
            tensor: The gradient tensor.
            grad_quant_type: ``"fp8"``, ``"mxfp8"``, ``"fp4"``, or ``None``.
            fp8_dtype: Explicit FP8 dtype for the ``"fp8"`` path.  Auto-detects
                when ``None``.
            block_size: Block size for ``"mxfp8"`` quantization.
        """
        if grad_quant_type is None:
            return tensor

        if grad_quant_type == "fp8":
            if fp8_dtype is None:
                fp8_dtype = _get_float8_e4m3()
            return _round_to_fp8(tensor, fp8_dtype)

        if grad_quant_type == "mxfp8":
            return _round_to_mxfp8(tensor, block_size=block_size)

        if grad_quant_type == "fp4":
            raise NotImplementedError(
                "FP4 gradient quantization is not yet implemented. " "Use 'fp8' or 'mxfp8' for now."
            )

        raise ValueError(f"Unknown grad_quant_type={grad_quant_type!r}. " f"Valid options: {GRAD_QUANT_TYPES}")

    # ------------------------------------------------------------------
    # Attention quantization primitives
    # ------------------------------------------------------------------

    @staticmethod
    def quantize_per_tensor_fp8(
        tensor: torch.Tensor,
        float8_dtype: Optional[torch.dtype] = None,
    ):
        """Per-tensor FP8 quantization.

        Returns ``(tensor_fp8, descale)`` where *descale* has shape ``(1,)``
        matching the aiter per-tensor convention.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()
        dtype_max = torch.finfo(float8_dtype).max
        amax = tensor.abs().amax()
        amax = torch.where(
            amax == 0,
            torch.tensor(dtype_max, device=tensor.device, dtype=tensor.dtype),
            amax,
        )
        scale = dtype_max / amax
        tensor_fp8 = (tensor * scale).clamp(-dtype_max, dtype_max).to(float8_dtype)
        descale = (1.0 / scale).to(torch.float32).reshape(1)
        return tensor_fp8, descale

    @staticmethod
    def quantize_block_fp8(
        tensor: torch.Tensor,
        block_m: int,
        float8_dtype: Optional[torch.dtype] = None,
    ):
        """Per-block FP8 quantization for attention Q/K tensors.

        Expects input in ``[B, S, H, D]`` (BSHD) layout.  Internally permutes
        to ``[B, H, S, D]``, reshapes into blocks of *block_m* rows, computes
        per-block max for scaling, quantizes, and permutes back to BSHD.

        Returns ``(tensor_fp8, scale_inv)`` where *scale_inv* is the inverse
        scale suitable for dequantization.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()
        tensor = tensor.permute(0, 2, 1, 3)  # [B, H, S, D]
        B, H, L, D = tensor.shape
        MAX_FP8 = torch.finfo(float8_dtype).max
        tensor = tensor.reshape(B, H, L // block_m, block_m, D).reshape(
            B,
            H,
            L // block_m,
            block_m * D,
        )
        tensor_max = tensor.abs().max(dim=-1)[0]
        tensor_max = torch.where(tensor_max == 0, MAX_FP8, tensor_max)
        scale = MAX_FP8 / tensor_max
        tensor = tensor * scale.reshape(scale.shape + (1,))
        tensor = tensor.clamp(-MAX_FP8, MAX_FP8).to(float8_dtype)
        tensor = tensor.reshape(B, H, L, D).permute(0, 2, 1, 3).contiguous()
        return tensor, 1.0 / scale.to(torch.float32).contiguous()

    @staticmethod
    def quantize_v_fp8(
        v: torch.Tensor,
        float8_dtype: Optional[torch.dtype] = None,
    ):
        """Per-tensor FP8 quantization for attention V tensor.

        Returns ``(v_fp8, v_scale, p_scale)`` where *p_scale* is the FP8 max
        value used for softmax scaling in the attention kernel.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()
        range_v = torch.max(torch.abs(v))
        dtype_max = torch.finfo(float8_dtype).max
        v_scale = dtype_max / range_v
        p_scale = dtype_max
        finfo = torch.finfo(float8_dtype)
        v_fp8 = (v * v_scale).clamp(min=finfo.min, max=finfo.max).to(float8_dtype)
        return v_fp8, v_scale, p_scale

    @staticmethod
    def quantize_block_mxfp8(
        tensor: torch.Tensor,
        block_size: int,
        layout: str = "bshd",
        *,
        is_2d_block: bool = True,
        float8_dtype: Optional[torch.dtype] = None,
        cu_seqlens=None,
        max_seqlens=None,
    ):
        """MXFP8 block quantization with attention layout support.

        Supports ``"bhsd"``, ``"bshd"``, and ``"thd"`` layouts.

        Returns ``(tensor_mxfp8, scale)``.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()

        if layout == "bhsd":
            tensor_bhsd = tensor
            B, H, S, D = tensor_bhsd.shape
        elif layout == "bshd":
            tensor_bhsd = tensor.permute(0, 2, 1, 3).contiguous()
            B, H, S, D = tensor_bhsd.shape
        elif layout == "thd":
            assert cu_seqlens is not None, "thd layout requires cu_seqlens"
            assert tensor.dim() == 3, f"expected thd tensor shape [T,H,D], got {tensor.shape}"
            T, H, D = tensor.shape
            B = int(cu_seqlens.numel() - 1)
            assert max_seqlens is not None, "thd layout requires max_seqlens"
            max_seqlen = int(max_seqlens) if isinstance(max_seqlens, int) else int(max(max_seqlens))
            if is_2d_block:
                padded = ((max_seqlen + block_size - 1) // block_size) * block_size
            else:
                padded = max_seqlen
            tensor_bhsd = torch.zeros(
                (B, H, padded, D),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            for b in range(B):
                s = int(cu_seqlens[b].item())
                e = int(cu_seqlens[b + 1].item())
                tensor_bhsd[b, :, : e - s, :] = tensor[s:e].transpose(0, 1)
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        convert_to_mxfp8, _, _ = _get_quant_ops()
        quanted_bhsd, scale_bhsd = convert_to_mxfp8(
            tensor_bhsd,
            block_size=block_size,
            axis=-1,
            is_2d_block=is_2d_block,
            float8_dtype_pt=float8_dtype,
        )

        if layout == "bshd":
            return (
                quanted_bhsd.permute(0, 2, 1, 3).contiguous(),
                scale_bhsd.permute(0, 2, 1, 3).contiguous(),
            )

        if layout == "thd":
            qs, ss = [], []
            for b in range(B):
                s0 = int(cu_seqlens[b].item())
                e0 = int(cu_seqlens[b + 1].item())
                L = e0 - s0
                qs.append(quanted_bhsd[b, :, :L, :].transpose(0, 1).contiguous())
                if is_2d_block:
                    m_blocks = (L + block_size - 1) // block_size
                    ss.append(scale_bhsd[b, :, :m_blocks, :].transpose(0, 1).contiguous())
                else:
                    ss.append(scale_bhsd[b, :, :L, :].transpose(0, 1).contiguous())
            return torch.cat(qs, dim=0), torch.cat(ss, dim=0)

        return quanted_bhsd, scale_bhsd

    @staticmethod
    def compute_p_scale_mxfp8(
        float8_dtype: Optional[torch.dtype] = None,
    ) -> int:
        """Compute softmax P scale constant for MXFP8 attention kernels.

        Returns an integer scale derived from the FP8 format's exponent
        encoding, used by Triton MXFP8 attention kernels for correct
        softmax rescaling.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()
        p_scale_f = torch.finfo(float8_dtype).max
        if float8_dtype == torch.float8_e4m3fn:
            mask_s, mbits, s_bias = 0b1111, 3, 7
        else:
            mask_s, mbits, s_bias = 0b11111, 2, 15
        hp_ebias = 127
        raw = (
            torch.bitwise_right_shift(
                torch.tensor(p_scale_f).to(float8_dtype).view(torch.uint8),
                mbits,
            )
            & mask_s
        )
        return (raw - s_bias + hp_ebias).to(torch.uint32).item()
