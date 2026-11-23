"""IMPORT-phase patch: replay Megatron's ``TopKRouter`` training forward/backward as CUDA graphs.

The router is ~25 small ops (gating GEMM, top-k, softmax, masks, aux loss and its logging)
whose launch cost (~240 us of host time per call) sits on the host-bound MoE forward path,
before the MORI dispatch. A graph replays the same kernels, so outputs, gradients and the
aux-loss tracker are bit-identical; the replay costs ~30 us.

Each router is captured with ``torch.cuda.make_graphed_callables`` on its third eligible call
(the first two run eagerly, so lazy state and the aux-loss backward scale exist). A graph's
static outputs and saved activations are overwritten by its next replay, so only the
configuration where every forward is followed by its backward before the next forward is
eligible: training with grad enabled, no padding mask, plain ``aux_loss`` balancing, no
recompute, pipeline size 1, a single-rank TPxCP group (no collective inside the graph) and no
MoE comm overlap. Anything else runs eagerly.

``LUMEN_GRAPHED_ROUTER=0`` disables the patch.
"""

from __future__ import annotations

import logging
import os

import torch

from lumen.patches.registry import PatchPhase, register_patch

logger = logging.getLogger(__name__)

_WARMUP_CALLS = 2


class _RouterGraph:
    __slots__ = ("key", "calls", "forward", "disabled", "capturing")

    def __init__(self):
        self.key = None
        self.calls = 0
        self.forward = None
        self.disabled = False
        self.capturing = False


def _config_eligible(router) -> bool:
    cfg = router.config
    lb_type = cfg.moe_router_load_balancing_type
    lb_types = lb_type if isinstance(lb_type, (list, tuple)) else [lb_type]
    if list(lb_types) != ["aux_loss"] or getattr(router, "routing_type", None) == "sinkhorn":
        return False
    if any(
        getattr(cfg, name, None)
        for name in (
            "moe_input_jitter_eps",
            "moe_z_loss_coeff",
            "moe_router_force_load_balancing",
            "moe_router_force_biased",
            "moe_expert_capacity_factor",
            "moe_router_fusion",
            "moe_router_enable_expert_bias",
            "overlap_moe_expert_parallel_comm",
            "recompute_granularity",
            "virtual_pipeline_model_parallel_size",
        )
    ):
        return False
    if getattr(router, "router_replay", None) is not None:
        return False
    if getattr(router, "expert_bias", None) is not None:
        return False
    tp_cp_group = getattr(router, "tp_cp_group", None)
    if tp_cp_group is not None and tp_cp_group.size() != 1:
        return False
    try:
        from megatron.core import parallel_state

        if parallel_state.get_pipeline_model_parallel_world_size() != 1:
            return False
    except (ImportError, AssertionError, RuntimeError):
        return False
    return True


def _call_eligible(router, input, padding_mask) -> bool:
    return (
        padding_mask is None
        and router.training
        and torch.is_grad_enabled()
        and input.is_cuda
        and input.requires_grad
        and not torch.cuda.is_current_stream_capturing()
    )


def _capture(router, input, state):
    from megatron.core.transformer.moe import moe_utils

    if moe_utils.MoEAuxLossAutoScaler.main_loss_backward_scale is None:
        return None
    if router._forward_hooks or router._forward_pre_hooks or router._backward_hooks:
        state.disabled = True
        return None
    tracker = moe_utils.get_moe_layer_wise_logging_tracker()
    saved = {name: entry["values"].clone() for name, entry in tracker.items()}
    sample = input.detach().clone().requires_grad_(True)
    state.capturing = True
    try:
        torch.cuda.make_graphed_callables(router, (sample,), num_warmup_iters=3)
        graphed = vars(router).pop("forward")
    finally:
        state.capturing = False
        for name, values in saved.items():
            tracker[name]["values"].copy_(values)
    return graphed


def install_graphed_router() -> None:
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError:
        return
    if getattr(TopKRouter, "_lumen_graphed_router_patched", False):
        return

    eager_forward = TopKRouter.forward

    def forward(self, input, padding_mask=None):
        state = self.__dict__.get("_lumen_router_graph")
        if state is None:
            state = _RouterGraph()
            state.disabled = not _config_eligible(self)
            self.__dict__["_lumen_router_graph"] = state
        if state.disabled or state.capturing or not _call_eligible(self, input, padding_mask):
            return eager_forward(self, input, padding_mask=padding_mask)
        key = (tuple(input.shape), input.dtype, input.device, input.stride())
        if state.forward is not None:
            if key == state.key:
                return state.forward(input)
            return eager_forward(self, input, padding_mask=padding_mask)
        if key != state.key:
            state.key, state.calls = key, 0
        state.calls += 1
        if state.calls > _WARMUP_CALLS:
            state.forward = _capture(self, input, state)
            if state.forward is not None:
                return state.forward(input)
        return eager_forward(self, input, padding_mask=padding_mask)

    TopKRouter.forward = forward
    TopKRouter._lumen_graphed_router_patched = True
    logger.info("Megatron TopKRouter training forward/backward replays as CUDA graphs")


register_patch(
    "graphed_router",
    PatchPhase.IMPORT,
    description="Replay Megatron TopKRouter forward/backward as CUDA graphs",
    enabled=lambda: os.environ.get("LUMEN_GRAPHED_ROUTER", "1") == "1",
    tags=frozenset({"core", "moe", "megatron"}),
)(install_graphed_router)
