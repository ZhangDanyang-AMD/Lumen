"""MORI EP token dispatcher for Megatron MoE (Phase C).

Uses ``mori.ops.EpDispatchCombineOp`` on the expert-parallel process group instead
of NCCL ``all_to_all`` in ``MoEAlltoAllTokenDispatcher``.  Enabled when
``LUMEN_DSV4_MOE_MORI=1`` — ``get_dsv4_spec`` patches ``MoELayer`` to select
``MoEMoriTokenDispatcher`` while keeping ``--moe-token-dispatcher-type alltoall``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig

from lumen.utils.sonic_moe_handoff import (
    mapped_handoff_enabled,
    mapped_handoff_scope,
    publish_mapped_handoff,
)

logger = logging.getLogger(__name__)

_MORI_GROUP_NAME = "lumen_dsv4_mori_ep"
_MORI_SHMEM_REGISTERED = False
_MORI_EP_GLOO_GROUP: dist.ProcessGroup | None = None
_MAPPED_FALLBACK_NOTICES: set[str] = set()


def mori_ep_enabled() -> bool:
    return os.environ.get("LUMEN_DSV4_MOE_MORI", "0") == "1"


def _require_mori():
    try:
        import mori  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LUMEN_DSV4_MOE_MORI=1 requires the mori package. "
            "Use lumen/tests:latest or lumen/dsv4-lumen:mi308x."
        ) from exc


def _get_ep_gloo_group(ep_group: dist.ProcessGroup) -> dist.ProcessGroup:
    """Gloo process group matching *ep_group* ranks (MORI shmem requires CPU backend)."""
    global _MORI_EP_GLOO_GROUP
    if _MORI_EP_GLOO_GROUP is not None:
        return _MORI_EP_GLOO_GROUP
    ranks = dist.get_process_group_ranks(ep_group)
    _MORI_EP_GLOO_GROUP = dist.new_group(ranks=ranks, backend="gloo")
    return _MORI_EP_GLOO_GROUP


def _init_mori_shmem(ep_group: dist.ProcessGroup) -> None:
    global _MORI_SHMEM_REGISTERED
    from mori import shmem

    gloo_group = _get_ep_gloo_group(ep_group)
    if not _MORI_SHMEM_REGISTERED:
        torch._C._distributed_c10d._register_process_group(_MORI_GROUP_NAME, gloo_group)
        _MORI_SHMEM_REGISTERED = True
    shmem.shmem_torch_process_group_init(_MORI_GROUP_NAME)


def _routing_to_topk(
    routing_map: torch.Tensor, probs: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Megatron multihot routing to MORI top-k indices/weights."""
    token_probs, token_indices = torch.topk(probs, topk, dim=-1)
    if routing_map.dtype == torch.bool:
        valid = routing_map.gather(1, token_indices)
        token_probs = token_probs * valid.to(token_probs.dtype)
    token_indices = token_indices.to(torch.int32)
    token_probs = token_probs.to(torch.float32)
    return token_indices, token_probs


def _mapped_local_routing(
    recv_weights: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    local_expert_indices: list[int],
):
    """Build expert-major scores and a GPU source-row gather map."""
    if recv_weights.shape != recv_topk_ids.shape or recv_topk_ids.dim() != 2:
        raise ValueError(
            "MORI routing weights and expert ids must have matching rank-2 shapes"
        )
    local_start = local_expert_indices[0]
    local_count = len(local_expert_indices)
    local_ids = recv_topk_ids.to(torch.long) - local_start
    valid = (local_ids >= 0) & (local_ids < local_count)
    positions = valid.nonzero(as_tuple=False)
    source_rows = positions[:, 0]
    slots = positions[:, 1]
    expert_ids = local_ids[source_rows, slots]
    order = torch.argsort(expert_ids, stable=True)
    source_rows = source_rows.index_select(0, order)
    expert_ids = expert_ids.index_select(0, order)
    scores = (
        recv_weights[source_rows, slots.index_select(0, order)]
        .to(torch.float32)
        .contiguous()
    )
    gather_map = source_rows.to(torch.int32).contiguous()
    counts_gpu = torch.bincount(expert_ids, minlength=local_count)
    counts = tuple(int(count) for count in counts_gpu.detach().cpu().tolist())
    if sum(counts) != gather_map.numel():
        raise RuntimeError("MORI mapped routing count mismatch")
    return scores, counts, gather_map


@dataclass
class _MappedCombineV1:
    gather_map: torch.Tensor
    source_rows: int
    gather_version: int
    api_version: int = 1


@lru_cache(maxsize=8)
def _create_mori_op(
    ep_size: int,
    ep_rank: int,
    hidden_dim: int,
    max_tokens: int,
    num_local_experts: int,
    topk: int,
    dtype: torch.dtype,
    enable_sdma: bool,
):
    import mori.ops as mori_ops

    kernel_type = mori_ops.EpDispatchCombineKernelType.IntraNode
    return (
        mori_ops.EpDispatchCombineOp(
            mori_ops.EpDispatchCombineConfig(
                data_type=dtype,
                rank=ep_rank,
                world_size=ep_size,
                hidden_dim=hidden_dim,
                scale_dim=1,
                scale_type_size=torch.float32.itemsize,
                max_token_type_size=dtype.itemsize,
                max_num_inp_token_per_rank=max_tokens,
                num_experts_per_rank=num_local_experts,
                num_experts_per_token=topk,
                warp_num_per_block=16,
                block_num=80,
                kernel_type=kernel_type,
                gpu_per_node=ep_size,
            )
        ),
        enable_sdma,
    )


class MoEMoriTokenDispatcher(MoETokenDispatcher):
    """Expert-parallel dispatch/combine via MORI ``EpDispatchCombineOp``."""

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: list[int],
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        _require_mori()
        super().__init__(config=config, pg_collection=pg_collection)
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        self.local_expert_indices = local_expert_indices
        self.router_topk = config.moe_router_topk

        input_chunk_idxs = torch.arange(
            self.num_experts * max(self.tp_size, 1), device="cuda"
        )
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        self._mori_op = None
        self._enable_sdma = os.environ.get("MORI_ENABLE_SDMA", "0") == "1"
        self._max_tokens = int(os.environ.get("LUMEN_DSV4_MORI_MAX_TOKENS", "4096"))
        self._mori_shmem_ready = False
        self._mapped_handoff_requested = (
            os.environ.get("LUMEN_MORI_SONIC_MXFP4_HANDOFF", "0") == "1"
        )
        self._mapped_combine: _MappedCombineV1 | None = None

    @contextmanager
    def expert_handoff_scope(self, experts):
        """Scope one optional dispatcher-to-expert handoff."""
        enabled = False
        if self._mapped_handoff_requested:
            capability = getattr(experts, "mori_mapped_handoff_capability", None)
            if callable(capability):
                enabled, reason = capability()
            else:
                reason = f"{type(experts).__name__} is not a SonicMoEExperts consumer"
            if not enabled and reason not in _MAPPED_FALLBACK_NOTICES:
                _MAPPED_FALLBACK_NOTICES.add(reason)
                logger.warning(
                    "MORI mapped MXFP4 handoff unavailable (%s); "
                    "materializing the local expert reorder",
                    reason,
                )
        try:
            with mapped_handoff_scope(enabled):
                yield
        except BaseException:
            self._mapped_combine = None
            raise

    def _get_mori_op(self, hidden_dim: int, dtype: torch.dtype):
        op, enable_sdma = _create_mori_op(
            self.ep_size,
            parallel_state.get_expert_model_parallel_rank(),
            hidden_dim,
            self._max_tokens,
            self.num_local_experts,
            self.router_topk,
            dtype,
            self._enable_sdma,
        )
        self._enable_sdma = enable_sdma
        return op

    def _ensure_mori_shmem(self) -> None:
        if not self._mori_shmem_ready:
            _init_mori_shmem(self.ep_group)
            self._mori_shmem_ready = True

    def dispatch_preprocess(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self.routing_map = routing_map
        self.topk_ids, self.topk_weights = _routing_to_topk(
            routing_map, probs, self.router_topk
        )
        return hidden_states, self.topk_weights

    def token_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        del probs
        self._ensure_mori_shmem()
        _, hidden_dim = hidden_states.shape
        self._mori_op = self._get_mori_op(hidden_dim, hidden_states.dtype)

        scale = torch.empty(0, device=hidden_states.device, dtype=torch.float32)
        if self._enable_sdma:
            dispatch_fn = self._mori_op.dispatch_send
        else:
            dispatch_fn = self._mori_op.dispatch

        (
            packed_hidden,
            recv_weights,
            _recv_scales,
            recv_topk_ids,
            packed_recv_count,
        ) = dispatch_fn(hidden_states, self.topk_weights, scale, self.topk_ids)

        if self._enable_sdma:
            self._mori_op.dispatch_recv()

        self._recv_topk_ids = recv_topk_ids
        self._recv_topk_weights = recv_weights
        if isinstance(packed_recv_count, torch.Tensor):
            self._tokens_per_expert = packed_recv_count.to(torch.long)
        else:
            self._tokens_per_expert = torch.tensor(
                packed_recv_count, device=hidden_states.device, dtype=torch.long
            )
        return packed_hidden, recv_weights

    def dispatch_postprocess(self, global_input_tokens, global_probs):
        if mapped_handoff_enabled():
            scores, counts, gather_map = _mapped_local_routing(
                self._recv_topk_weights,
                self._recv_topk_ids,
                self.local_expert_indices,
            )
            if publish_mapped_handoff(global_input_tokens, scores, counts, gather_map):
                self._mapped_combine = _MappedCombineV1(
                    gather_map=gather_map,
                    source_rows=global_input_tokens.shape[0],
                    gather_version=gather_map._version,
                )
                tokens_per_expert = torch.tensor(counts, dtype=torch.long)
                return global_input_tokens, tokens_per_expert, scores

        if self.num_local_experts > 1 and self._tokens_per_expert.numel() > 0:
            counts = self._tokens_per_expert
            if counts.dim() == 1 and counts.numel() == self.num_local_experts:
                global_input_tokens, global_probs = sort_chunks_by_idxs(
                    global_input_tokens,
                    counts,
                    self.sort_input_by_local_experts,
                    probs=global_probs,
                    fused=self.config.moe_permute_fusion,
                )
        tokens_per_expert = self._tokens_per_expert
        return global_input_tokens, tokens_per_expert, global_probs

    def combine_preprocess(self, hidden_states):
        if self._mapped_combine is not None:
            combine = self._mapped_combine
            self._mapped_combine = None
            if (
                combine.api_version != 1
                or combine.gather_map._version != combine.gather_version
            ):
                raise RuntimeError("MORI mapped combine metadata was mutated")
            if hidden_states.shape[0] != combine.gather_map.numel():
                raise RuntimeError("MORI mapped expert output row mismatch")
            source_output = hidden_states.new_zeros(
                (combine.source_rows, *hidden_states.shape[1:])
            )
            return source_output.index_add(
                0, combine.gather_map.to(torch.long), hidden_states
            )

        if self.num_local_experts > 1 and self._tokens_per_expert.numel() > 0:
            counts = self._tokens_per_expert
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                counts.T.ravel() if counts.dim() > 1 else counts,
                self.restore_output_by_local_experts,
                fused=self.config.moe_permute_fusion,
            )
        return hidden_states

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        del async_finish, allocate_on_comm_stream
        if self._enable_sdma:
            combined = self._mori_op.combine_send(
                hidden_states, None, self._recv_topk_ids
            )[0]
            self._mori_op.combine_recv()
        else:
            combined = self._mori_op.combine(hidden_states, None, self._recv_topk_ids)[
                0
            ]
        return combined

    def combine_postprocess(self, hidden_states: torch.Tensor):
        return hidden_states.view(self.hidden_shape)


def patch_megatron_moe_mori() -> None:
    """Route ``alltoall`` MoE dispatcher to MORI when env is set."""
    if not mori_ep_enabled():
        return
    if getattr(patch_megatron_moe_mori, "_done", False):
        return

    from megatron.core.transformer.moe import moe_layer
    from megatron.core.transformer.moe.moe_layer import get_default_pg_collection

    _orig_init = moe_layer.MoELayer.__init__
    _orig_routed_experts_compute = moe_layer.MoELayer.routed_experts_compute

    def _patched_init(
        self,
        config: TransformerConfig,
        submodules=None,
        layer_number=None,
        pg_collection=None,
    ):
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        _orig_init(self, config, submodules, layer_number, pg_collection)
        if config.moe_token_dispatcher_type != "alltoall":
            return
        self.token_dispatcher = MoEMoriTokenDispatcher(
            self.num_local_experts,
            self.local_expert_indices,
            config=config,
            pg_collection=pg_collection,
        )
        if self.use_shared_expert and self.shared_expert_overlap:
            self.token_dispatcher.set_shared_experts(self.shared_experts)

    moe_layer.MoELayer.__init__ = _patched_init

    def _patched_routed_experts_compute(self, hidden_states, probs, residual):
        dispatcher = self.token_dispatcher
        if (
            not isinstance(dispatcher, MoEMoriTokenDispatcher)
            or not dispatcher._mapped_handoff_requested
        ):
            return _orig_routed_experts_compute(self, hidden_states, probs, residual)
        with dispatcher.expert_handoff_scope(self.experts):
            return _orig_routed_experts_compute(self, hidden_states, probs, residual)

    moe_layer.MoELayer.routed_experts_compute = _patched_routed_experts_compute
    patch_megatron_moe_mori._done = True
    logger.info("[lumen-dsv4] MoE token dispatcher: MORI EP (EpDispatchCombineOp)")
