"""Fuse Megatron MORI dispatch's receive-side routing into one AITER kernel."""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import textwrap

logger = logging.getLogger(__name__)

_ENV = "LUMEN_MORI_FUSED_RECV"
# MoriDispatch.forward of Megatron-LM a1b00d42 (Core_r0.17.0); any other body keeps the original.
_EXPECTED_FORWARD_SHA256 = frozenset(
    {"112a4c4ee1c6c975f7ab89675287e9b59c031ead01e9b2f18692e3b385e099b1"}
)
_STATE = {"installed": False, "reason": None}


def forward_fingerprint(forward) -> str:
    source = textwrap.dedent(inspect.getsource(forward)).strip()
    return hashlib.sha256(source.encode()).hexdigest()


def _fused_forward_factory(fused_a2a, recv_routing):
    import torch

    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        num_local_experts,
        router_topk,
        max_num_tokens_per_rank,
        fp8_dispatch=False,
        async_finish=True,
        allocate_on_comm_stream=True,
        kernel_type=None,
    ):
        op = fused_a2a.get_mori_op(
            group=group,
            hidden_dim=x.shape[1],
            num_local_experts=num_local_experts,
            router_topk=router_topk,
            max_num_tokens_per_rank=max_num_tokens_per_rank,
            data_type=x.dtype,
            kernel_type=kernel_type,
            fp8_dispatch=fp8_dispatch,
        )
        (
            dispatch_out,
            dispatch_weights,
            _,
            dispatch_indices,
            recv_num_token,
            routing_handle,
        ) = fused_a2a._run_mori_op_on_stream(
            lambda: op.dispatch(
                x,
                token_probs.float(),
                None,
                token_indices.to(torch.int32),
                return_routing=True,
            ),
            async_finish,
            allocate_on_comm_stream,
        )
        # Same values as Megatron's where/scatter_add_ chain: local ids with -1 for
        # padding rows and non-local slots, plus per-local-expert counts.
        recv_token_indices, tokens_per_expert = recv_routing(
            dispatch_indices.contiguous(),
            recv_num_token.flatten()[:1],
            torch.distributed.get_rank(group) * num_local_experts,
            num_local_experts,
        )

        ctx.group = group
        ctx.num_local_experts = num_local_experts
        ctx.router_topk = router_topk
        ctx.max_num_tokens_per_rank = max_num_tokens_per_rank
        ctx.fp8_dispatch = fp8_dispatch
        ctx.local_num_tokens = token_indices.shape[0]
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.routing_handle = routing_handle
        ctx.save_for_backward(token_indices)
        # dispatch_weights views MORI's reusable buffer; combine reads this private copy later.
        raw_dispatch_weights = dispatch_weights.detach().clone()
        ctx.mark_non_differentiable(
            recv_token_indices, tokens_per_expert, raw_dispatch_weights
        )
        return (
            dispatch_out,
            recv_token_indices,
            dispatch_weights,
            tokens_per_expert,
            raw_dispatch_weights,
            routing_handle,
        )

    return forward


def install_mori_fused_recv() -> bool:
    """Patch the exact Megatron MoriDispatch.forward; any mismatch keeps the original."""
    if _STATE["installed"]:
        return True
    if os.environ.get(_ENV, "1") == "0":
        _STATE["reason"] = f"{_ENV}=0"
        return False
    try:
        from megatron.core.transformer.moe import fused_a2a
    except ImportError as exc:
        _STATE["reason"] = f"Megatron fused_a2a import failed: {exc!r}"
        return False
    dispatch = getattr(fused_a2a, "MoriDispatch", None)
    if dispatch is None or not all(
        callable(getattr(fused_a2a, name, None))
        for name in ("get_mori_op", "_run_mori_op_on_stream")
    ):
        _STATE["reason"] = "MoriDispatch or its helpers are missing"
        return False
    try:
        fingerprint = forward_fingerprint(dispatch.forward)
    except (OSError, TypeError, ValueError) as exc:
        _STATE["reason"] = f"MoriDispatch.forward source unavailable: {exc!r}"
        return False
    if fingerprint not in _EXPECTED_FORWARD_SHA256:
        _STATE["reason"] = f"MoriDispatch.forward fingerprint {fingerprint} is not supported"
        logger.warning("MORI fused receive routing unavailable: %s", _STATE["reason"])
        return False
    try:
        from aiter.ops.triton.moe.mori_recv_routing import mori_recv_local_routing
    except ImportError as exc:
        _STATE["reason"] = f"AITER mori_recv_local_routing unavailable: {exc!r}"
        logger.warning("MORI fused receive routing unavailable: %s", _STATE["reason"])
        return False
    dispatch.forward = staticmethod(
        _fused_forward_factory(fused_a2a, mori_recv_local_routing)
    )
    _STATE["installed"] = True
    logger.info("Installed MORI fused receive routing")
    return True


__all__ = ["forward_fingerprint", "install_mori_fused_recv"]
