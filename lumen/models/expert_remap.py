"""Place Qwen3 experts so each EP rank sees a similar token load.

Counts are taken from the router's original expert ids while the weights are
still in their contiguous layout. After ``LUMEN_EXPERT_REMAP_AFTER_STEPS``
optimizer steps (default 1) the permutation is packed from those counts.
Rank ``r`` then owns slots ``[local * r, local * r + local)``. Token weights
stay on the original experts. Megatron aux loss is attached inside
``TopKRouter.routing`` before the column permute, so it still sees the logical
expert ids.

Moving a weight also moves the Adam moments that belong to it. The bf16
parameter and Megatron's fp32 master are permuted directly, so this does not
reload the master from the rounded bf16 copy.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

_INVERSE: dict[tuple[int, str], torch.Tensor] = {}
_INSTALLED = False
_PLACED = False
_OPTIMIZER_STEPS = 0


def _enabled() -> bool:
    return os.environ.get("LUMEN_EXPERT_REMAP", "0") == "1"


def _after_steps() -> int:
    raw = os.environ.get("LUMEN_EXPERT_REMAP_AFTER_STEPS", "1")
    try:
        steps = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"LUMEN_EXPERT_REMAP_AFTER_STEPS must be an integer, got {raw!r}"
        ) from exc
    if steps < 1:
        raise RuntimeError("LUMEN_EXPERT_REMAP_AFTER_STEPS must be at least 1")
    return steps


def _inverse_index(layer: int, device: torch.device) -> torch.Tensor:
    from lumen.models.qwen3_30b_a3b.fsdp.expert_balance import inverse_ids

    key = (layer, str(device))
    cached = _INVERSE.get(key)
    if cached is None:
        cached = torch.tensor(inverse_ids(layer), dtype=torch.long, device=device)
        _INVERSE[key] = cached
    return cached


def _walk_optimizers(optimizer):
    seen: set[int] = set()
    stack = [optimizer]
    found = []
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        found.append(current)
        for name in ("optimizer", "_inner", "inner_optimizer"):
            child = getattr(current, name, None)
            if child is not None and child is not current:
                stack.append(child)
        for name in ("chained_optimizers", "optimizers"):
            children = getattr(current, name, None)
            if children:
                stack.extend(children)
    return found


def _paired_shards(optimizer, param) -> list[torch.Tensor]:
    """Megatron keeps a 1-D fp32 master beside the bf16 expert parameter."""
    shards = []
    group_pairs = (
        ("model_float16_groups", "shard_fp32_from_float16_groups"),
        ("model_float16_groups", "shard_float16_groups"),
        ("model_fp32_groups", "shard_fp32_groups"),
        ("float16_groups", "fp32_from_float16_groups"),
    )
    for model_name, shard_name in group_pairs:
        model_groups = getattr(optimizer, model_name, None)
        shard_groups = getattr(optimizer, shard_name, None)
        if not model_groups or not shard_groups:
            continue
        for model_group, shard_group in zip(model_groups, shard_groups):
            for model_param, shard_param in zip(model_group, shard_group):
                if model_param is param and torch.is_tensor(shard_param):
                    shards.append(shard_param)
    return shards


def _state_tensors(optimizer, owner) -> list[torch.Tensor]:
    state = getattr(optimizer, "state", None)
    if state is None:
        return []
    bucket = state.get(owner)
    if not bucket:
        return []
    tensors = []
    for key in ("exp_avg", "exp_avg_sq"):
        value = bucket.get(key)
        if torch.is_tensor(value):
            tensors.append(value)
    return tensors


def _storages(param: torch.nn.Parameter, optimizer) -> list[tuple[torch.Tensor, bool]]:
    """Return ``(tensor, is_adam_moment)`` in a rank-independent order."""
    ordered: list[tuple[torch.Tensor, bool]] = [(param.data, False)]
    for opt in _walk_optimizers(optimizer):
        for moment in _state_tensors(opt, param):
            ordered.append((moment, True))
        for shard in _paired_shards(opt, param):
            ordered.append((shard, False))
            for owner in _walk_optimizers(opt):
                for moment in _state_tensors(owner, shard):
                    ordered.append((moment, True))
    unique: list[tuple[torch.Tensor, bool]] = []
    seen: set[int] = set()
    for tensor, is_moment in ordered:
        pointer = tensor.data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        unique.append((tensor, is_moment))
    return unique


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
    """Move expert rows across the EP group.

    A parameter keeps its leading expert dimension. Megatron's fp32 master and
    Adam moments are flat, but still expert-major, so they are viewed as rows.
    """
    tensor = _local_storage(tensor)
    ep_size = dist.get_world_size(ep_group)
    if tensor.ndim >= 1 and tensor.shape[0] == num_local:
        matrix = tensor
    elif tensor.ndim == 1 and tensor.is_contiguous() and tensor.numel() % num_local == 0:
        matrix = tensor.view(num_local, -1)
    else:
        raise RuntimeError("expert storage is not a contiguous expert-major layout")
    gathered = [torch.empty_like(matrix) for _ in range(ep_size)]
    dist.all_gather(gathered, matrix.contiguous(), group=ep_group)
    stacked = torch.cat(gathered, dim=0)
    matrix.copy_(stacked.index_select(0, src))
    del gathered, stacked


def _permute_param(param, src: torch.Tensor, ep_group, num_local: int, optimizer) -> None:
    storages = _storages(param, optimizer)
    if not any(is_moment for _, is_moment in storages):
        raise RuntimeError(
            "expert remap could not find Adam moments for an expert weight; "
            "refusing to move the weight without its optimizer state"
        )
    for tensor, _ in storages:
        if tensor.numel() != param.numel():
            raise RuntimeError(
                "runtime expert remap does not support an expert optimizer shard "
                f"of {tensor.numel()} elements for a parameter of {param.numel()}"
            )
        _exchange_matrix(tensor, src, ep_group, num_local)


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


def _place(model, optimizer) -> None:
    global _PLACED
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

    def invert(slot: list[int]) -> list[int]:
        inverse = [0] * len(slot)
        for original, new_slot in enumerate(slot):
            inverse[new_slot] = original
        return inverse

    with torch.no_grad():
        if blocks:
            for block in blocks:
                local = int(block.experts_per_rank)
                inverse = invert(table[block._remap_layer])
                src = torch.tensor(
                    inverse[block.local_expert_start : block.local_expert_start + local],
                    dtype=torch.long,
                    device=block.local_experts.w1.device,
                )
                for param in (block.local_experts.w1, block.local_experts.w2):
                    _permute_param(param, src, block.ep_group, local, optimizer)
        else:
            for layer, module in enumerate(experts):
                local = int(module.num_local_experts)
                inverse = invert(table[layer])
                start = dist.get_rank(module.ep_group) * local
                src = torch.tensor(
                    inverse[start : start + local],
                    dtype=torch.long,
                    device=module.w1.device,
                )
                for param in (module.w1, module.w2):
                    _permute_param(param, src, module.ep_group, local, optimizer)
    # Publish after the exchange. The next forward is what first reads the table.
    commit_slots(table)
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
            "> Lumen expert remap: packed a placement from token counts after "
            f"{_OPTIMIZER_STEPS} optimizer step(s); heaviest rank is {worst:.2f}x "
            "the mean load. Each token still hits the same expert.",
            flush=True,
        )


def note_optimizer_step(model, optimizer) -> None:
    """Pack and apply the placement once the warmup steps have been optimized."""
    global _OPTIMIZER_STEPS
    if not _enabled() or _PLACED:
        return
    _OPTIMIZER_STEPS += 1
    if _OPTIMIZER_STEPS < _after_steps():
        return
    _place(model, optimizer)


def install_megatron_expert_remap() -> None:
    """Collect router counts, then permute columns once the placement exists."""
    global _INSTALLED
    if not _enabled() or _INSTALLED:
        return

    from lumen.models.qwen3_30b_a3b.fsdp.expert_balance import (
        accumulate_layer_counts,
        slots_ready,
    )
    from megatron.core.transformer.moe.router import TopKRouter
    import megatron.training.training as training

    original_routing = TopKRouter.routing

    def routing(self, logits, padding_mask=None):
        probs, routing_map = original_routing(self, logits, padding_mask)
        if not slots_ready():
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
        model = args[1] if len(args) > 1 else kwargs["model"]
        optimizer = args[2] if len(args) > 2 else kwargs["optimizer"]
        original_step = optimizer.step

        def step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            note_optimizer_step(model, optimizer)
            return result

        optimizer.step = step
        try:
            return original_train(*args, **kwargs)
        finally:
            optimizer.step = original_step

    training.train = train
    _INSTALLED = True
