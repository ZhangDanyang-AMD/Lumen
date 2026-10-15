"""Place Qwen3 experts so each EP rank sees a similar token load.

One forward over the first batch counts the router's original expert
ids while the weights are still in their contiguous layout. The placement is
packed from those counts and written into the expert parameters before any
optimizer step. Rank ``r`` then owns slots ``[local * r, local * r + local)``.
A token keeps the expert it routed to; only the rank that stores it changes.

Megatron has already constructed its optimizer by the time the data iterator
exists, but that optimizer has not stepped. After the weights move,
``reload_model_params`` copies them into the fp32 master. Adam moments are
still the initial zeros, so nothing trained is migrated. FSDP creates Adam
only after this placement.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

_INVERSE: dict[tuple[int, str], torch.Tensor] = {}
_INSTALLED = False
_PLACED = False


def _enabled() -> bool:
    return os.environ.get("LUMEN_EXPERT_REMAP", "0") == "1"


def _inverse_index(layer: int, device: torch.device) -> torch.Tensor:
    from lumen.models.qwen3_30b_a3b.fsdp.expert_balance import inverse_ids

    key = (layer, str(device))
    cached = _INVERSE.get(key)
    if cached is None:
        cached = torch.tensor(inverse_ids(layer), dtype=torch.long, device=device)
        _INVERSE[key] = cached
    return cached


def _local_storage(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        local = tensor.to_local()
        if tuple(local.shape) != tuple(tensor.shape):
            raise RuntimeError(
                "runtime expert remap needs each expert weight unsharded on its rank"
            )
        return local
    return tensor


def _exchange_matrix(
    tensor: torch.Tensor,
    src: torch.Tensor,
    ep_group,
    num_local: int,
) -> None:
    """Move expert rows across the EP group."""
    tensor = _local_storage(tensor)
    ep_size = dist.get_world_size(ep_group)
    if tensor.ndim < 1 or tensor.shape[0] != num_local:
        raise RuntimeError(
            "expert weight is not laid out with the local experts on dimension 0, "
            f"got shape {tuple(tensor.shape)} and {num_local} local experts"
        )
    gathered = [torch.empty_like(tensor) for _ in range(ep_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=ep_group)
    stacked = torch.cat(gathered, dim=0)
    tensor.copy_(stacked.index_select(0, src))
    del gathered, stacked


def _targets(model):
    from lumen.models.qwen3_30b_a3b.fsdp.pretrain import EPShardedMoeBlock
    from lumen.modules.sonic_moe import SonicMoEExperts

    blocks = []
    experts = []
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        root = getattr(chunk, "module", chunk)
        for submodule in root.modules():
            if isinstance(submodule, EPShardedMoeBlock):
                blocks.append(submodule)
            elif isinstance(submodule, SonicMoEExperts):
                experts.append(submodule)
    if blocks and experts:
        raise RuntimeError("expert remap found both FSDP and Megatron expert modules")
    if not blocks and not experts:
        raise RuntimeError("expert remap found no expert modules")
    return blocks, experts


def _invert(slot: list[int]) -> list[int]:
    inverse = [0] * len(slot)
    for original, new_slot in enumerate(slot):
        inverse[new_slot] = original
    return inverse


def _finish_inflight_param_gathers(model) -> None:
    """Wait until probe communication has landed in the persistent parameters.

    A flag check is not enough. Megatron can leave a parameter all-gather
    queued after ``param_gather_dispatched`` is already false, and that gather
    writes the pre-placement weights back over the exchange.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        groups = []
        for name in ("bucket_groups", "expert_parallel_bucket_groups"):
            found = getattr(chunk, name, None)
            if found:
                groups.extend(found)
        for group in groups:
            config = getattr(group, "ddp_config", None)
            if not getattr(config, "overlap_param_gather", False):
                continue
            if getattr(group, "param_gather_dispatched", False):
                group.finish_param_sync(skip_next_bucket_dispatch=True)
    torch.cuda.synchronize()


def _drop_probe_weight_caches() -> None:
    """Drop expert GEMM weight snapshots taken while the layout was still identity.

    The probe is the first forward. FlyDSL keeps a Sonic operator built from
    those weights, and the FP8 expert path keeps a quantized copy keyed by
    parameter id and ``_version``. Writing ``param.data`` does not change
    ``_version``, so either snapshot would still be the original layout.
    """
    from lumen.ops.moe import flydsl_grouped

    for name in ("_PRESHUFFLE_CACHE", "_WEIGHT_KN_CACHE", "_NATIVE_ROUTES_CACHE"):
        cache = getattr(flydsl_grouped, name, None)
        if isinstance(cache, dict):
            cache.clear()
    from lumen.ops.gemm.grouped_gemm import _weight_fp8_cache

    _weight_fp8_cache.clear()


def _reshard_fsdp_parameters(model) -> None:
    """Put FSDP parameters back on the shard the optimizer updates.

    ``shard_grad_op`` keeps the probe's parameters in the all-gather buffer.
    Writing that buffer is discarded on the next reshard, so step 1 can look
    healthy and step 2 reads the original expert order again.
    """
    from torch.distributed.fsdp import FSDPModule

    chunks = model if isinstance(model, (list, tuple)) else [model]
    modules = []
    for chunk in chunks:
        root = getattr(chunk, "module", chunk)
        for submodule in root.modules():
            if isinstance(submodule, FSDPModule):
                modules.append(submodule)
    for module in reversed(modules):
        module.reshard()


def apply_counted_placement(model) -> None:
    """Permute expert weights from counts already recorded, then publish the table."""
    global _PLACED
    if _PLACED:
        return
    _finish_inflight_param_gathers(model)
    _reshard_fsdp_parameters(model)
    from lumen.models.qwen3_30b_a3b.fsdp.expert_balance import build_slots, commit_slots

    blocks, experts = _targets(model)
    if blocks:
        layers = [block._remap_layer for block in blocks]
        if layers != list(range(len(blocks))):
            raise RuntimeError(f"FSDP expert layers are {layers}, expected a dense 0-based range")
        for block in blocks:
            if int(block.experts_per_rank) != int(blocks[0].experts_per_rank):
                raise RuntimeError("expert remap expects the same expert count on every rank")
            if block.ep_group is not blocks[0].ep_group:
                raise RuntimeError("expert remap expects one expert-parallel group")
        sample = blocks[0]
        experts_per_rank = int(sample.experts_per_rank)
        group = sample.ep_group
    else:
        layers = list(range(len(experts)))
        for module in experts:
            if int(module.num_local_experts) != int(experts[0].num_local_experts):
                raise RuntimeError("expert remap expects the same expert count on every rank")
            if module.ep_group is not experts[0].ep_group:
                raise RuntimeError("expert remap expects one expert-parallel group")
        sample = experts[0]
        experts_per_rank = int(sample.num_local_experts)
        group = sample.ep_group
    if group is None:
        raise RuntimeError("expert remap requires an expert-parallel group")
    ep_size = dist.get_world_size(group)
    if ep_size <= 0:
        raise RuntimeError("expert remap requires a positive expert-parallel size")

    table, worst = build_slots(layers, experts_per_rank)
    width = len(table[layers[0]])
    if width != ep_size * experts_per_rank:
        raise RuntimeError(
            f"packed {width} experts but EP {ep_size} times {experts_per_rank} local experts "
            "does not match"
        )

    with torch.no_grad():
        if blocks:
            for block in blocks:
                local = int(block.experts_per_rank)
                inverse = _invert(table[block._remap_layer])
                src = torch.tensor(
                    inverse[block.local_expert_start : block.local_expert_start + local],
                    dtype=torch.long,
                    device=block.local_experts.w1.device,
                )
                for param in (block.local_experts.w1, block.local_experts.w2):
                    _exchange_matrix(param.data, src, block.ep_group, local)
        else:
            for layer, module in enumerate(experts):
                local = int(module.num_local_experts)
                inverse = _invert(table[layer])
                start = dist.get_rank(module.ep_group) * local
                src = torch.tensor(
                    inverse[start : start + local],
                    dtype=torch.long,
                    device=module.w1.device,
                )
                for param in (module.w1, module.w2):
                    _exchange_matrix(param.data, src, module.ep_group, local)
    torch.cuda.synchronize()
    _drop_probe_weight_caches()
    # Publish after the exchange. The next forward is what first reads the table.
    commit_slots(table)
    moved = 0
    for layer in layers:
        moved = max(
            moved,
            sum(expert != dest for expert, dest in enumerate(table[layer])),
        )
    if blocks:
        for block in blocks:
            block._expert_slot = torch.tensor(
                table[block._remap_layer],
                dtype=torch.long,
                device=block.gate.weight.device,
            )
            block._remap_collect = False
    _INVERSE.clear()
    _PLACED = True
    if dist.get_rank() == 0:
        print(
            "> Lumen expert remap: packed a placement from the first batch "
            "before optimizer state was created; heaviest rank is "
            f"{worst:.2f}x the mean load; {moved} experts changed seats. "
            "Each token still hits the same expert.",
            flush=True,
        )


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _probe_megatron(forward_step, model, iterators) -> None:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

    chunks = _as_list(model)
    wrapped = iterators if isinstance(iterators, list) else [iterators]
    if len(wrapped) != len(chunks):
        raise RuntimeError(
            f"expert remap probe has {len(chunks)} model chunks and {len(wrapped)} iterators"
        )
    microbatches = get_num_microbatches()
    if microbatches < 1:
        raise RuntimeError("expert remap probe requires at least one microbatch")
    for chunk in chunks:
        chunk.train()
    # Gradients stay enabled so activation checkpointing can run. The graph is
    # dropped with the output; this pass never backpropagates or steps.
    for _ in range(microbatches):
        for chunk, iterator in zip(chunks, wrapped):
            output = forward_step(iterator, chunk)
            del output
    return microbatches


class _CapturedBatches:
    """Remember batches read during the probe without replacing the outer iterator."""

    def __init__(self, inner):
        self.inner = inner
        self.saved = []

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.inner)
        self.saved.append(_clone_batch(item))
        return item


def _clone_batch(value):
    """Copy batch tensors so a later iterator write cannot change the replay."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_batch(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_batch(item) for item in value)
    if isinstance(value, list):
        return [_clone_batch(item) for item in value]
    return value


def _capture_probe(iterator) -> list[_CapturedBatches]:
    from megatron.core.rerun_state_machine import RerunDataIterator

    iterators = iterator if isinstance(iterator, list) else [iterator]
    captured = []
    for current in iterators:
        if not isinstance(current, RerunDataIterator):
            raise RuntimeError(
                "expert remap probe requires Megatron's RerunDataIterator, "
                f"got {type(current).__name__}"
            )
        wrapper = _CapturedBatches(current.iterable)
        current.iterable = wrapper
        captured.append(wrapper)
    return captured


def _rewind_probe(iterator, captured: list[_CapturedBatches], microbatches: int) -> None:
    """Replay the probe's batches so the first training step sees the same data."""
    iterators = iterator if isinstance(iterator, list) else [iterator]
    if len(iterators) != len(captured):
        raise RuntimeError("expert remap lost a probe iterator")
    for current, wrapper in zip(iterators, captured):
        if len(wrapper.saved) != microbatches:
            raise RuntimeError(
                "expert remap probe recorded "
                f"{len(wrapper.saved)} batches, expected {microbatches}"
            )
        current.iterable = wrapper.inner
        current.saved_microbatches = wrapper.saved
        current.replaying = True
        current.replay_pos = 0


def _clear_probe_aux_tracker() -> None:
    """Drop aux-loss totals recorded by the probe so step 1 logs only itself."""
    from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker

    clear_aux_losses_tracker()


def _refresh_megatron_master(optimizer) -> None:
    reload = getattr(optimizer, "reload_model_params", None)
    if not callable(reload):
        raise RuntimeError(
            "expert remap moved the weights but cannot copy them into the Megatron "
            "fp32 master"
        )
    reload()


def install_megatron_expert_remap() -> None:
    """Collect router counts, then permute columns once the placement exists."""
    global _INSTALLED
    if not _enabled() or _INSTALLED:
        return

    # Importing the qwen3 package here pulls Megatron back in while
    # ``lumen.models.megatron`` is still initializing.
    from megatron.core.transformer.moe.router import TopKRouter
    import megatron.training.training as training

    original_routing = TopKRouter.routing

    def routing(self, logits, padding_mask=None):
        from lumen.models.qwen3_30b_a3b.fsdp.expert_balance import (
            accumulate_layer_counts,
        )

        probs, routing_map = original_routing(self, logits, padding_mask)
        # ``_PLACED`` lives in this module. Gating on the balance module's
        # flag can miss the table when that module was imported twice.
        if not _PLACED:
            if routing_map is not None:
                counts = routing_map.detach().to(dtype=torch.float64)
                counts = counts.sum(dim=tuple(range(counts.ndim - 1)))
                accumulate_layer_counts(self.layer_number - 1, counts)
            return probs, routing_map
        index = _inverse_index(self.layer_number - 1, probs.device)
        return (
            probs.index_select(-1, index),
            routing_map.index_select(-1, index),
        )

    TopKRouter.routing = routing

    original_train = training.train

    def train(*args, **kwargs):
        if not _enabled() or _PLACED:
            return original_train(*args, **kwargs)
        forward_step = args[0] if args else kwargs["forward_step_func"]
        model = args[1] if len(args) > 1 else kwargs["model"]
        optimizer = args[2] if len(args) > 2 else kwargs["optimizer"]
        args_list = list(args)
        if len(args_list) > 4:
            incoming = args_list[4]
        else:
            incoming = kwargs["train_data_iterator"]
        captured = _capture_probe(incoming)
        microbatches = _probe_megatron(forward_step, model, incoming)
        apply_counted_placement(model)
        _refresh_megatron_master(optimizer)
        _rewind_probe(incoming, captured, microbatches)
        _clear_probe_aux_tracker()
        return original_train(*args, **kwargs)

    training.train = train
    _INSTALLED = True
