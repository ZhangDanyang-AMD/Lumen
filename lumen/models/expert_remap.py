"""Place Qwen3 experts so each EP rank sees a similar token load.

The table is the one measured for FSDP. ``slot[original]`` is the contiguous
id the dispatcher uses, and rank ``r`` still owns slots ``[16*r, 16*r+16)``.
Token weights stay on the original experts. Aux loss is attached inside
``TopKRouter.routing`` before this column permute, so it still sees the
logical expert ids.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

_INVERSE: dict[tuple[int, str], torch.Tensor] = {}
_INSTALLED = False


def _enabled() -> bool:
    return os.environ.get("LUMEN_EXPERT_REMAP", "0") == "1"


def _inverse_ids(layer: int) -> list[int]:
    from lumen.models.qwen3_30b_a3b.fsdp.expert_balance import expert_slot

    slot = expert_slot(layer)
    if len(slot) != 128 or sorted(slot) != list(range(128)):
        raise RuntimeError(f"expert placement for layer {layer} is not a permutation of 128 experts")
    inverse = [0] * 128
    for original, new_slot in enumerate(slot):
        inverse[new_slot] = original
    return inverse


def _inverse_index(layer: int, device: torch.device) -> torch.Tensor:
    key = (layer, str(device))
    cached = _INVERSE.get(key)
    if cached is None:
        cached = torch.tensor(_inverse_ids(layer), dtype=torch.long, device=device)
        _INVERSE[key] = cached
    return cached


def _exchange_expert_placement(model) -> bool:
    from lumen.modules.sonic_moe import SonicMoEExperts

    modules: list[SonicMoEExperts] = []
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        root = getattr(chunk, "module", chunk)
        for submodule in root.modules():
            if isinstance(submodule, SonicMoEExperts):
                modules.append(submodule)
    if len(modules) != 48:
        raise RuntimeError(f"expert remap expected 48 SonicMoE layers, found {len(modules)}")

    placed = False
    for layer, module in enumerate(modules):
        if getattr(module, "_lumen_experts_remapped", False):
            continue
        group = module.ep_group
        if group is None:
            raise RuntimeError("expert remap requires an expert-parallel group")
        ep_size = dist.get_world_size(group)
        local = int(module.num_local_experts)
        if ep_size * local != 128:
            raise RuntimeError(
                f"expert remap expects EP * local experts = 128, got {ep_size} * {local}"
            )
        start = dist.get_rank(group) * local
        src = _inverse_index(layer, module.w1.device)[start : start + local]
        for param in (module.w1, module.w2):
            gathered = [torch.empty_like(param.data) for _ in range(ep_size)]
            dist.all_gather(gathered, param.data.contiguous(), group=group)
            stacked = torch.cat(gathered, dim=0)
            param.data.copy_(stacked.index_select(0, src))
            del gathered, stacked
        module._lumen_experts_remapped = True
        placed = True

    if placed and dist.get_rank() == 0:
        print(
            "> Lumen expert remap enabled "
            "(permutation only; each token still hits the same expert)",
            flush=True,
        )
    return placed


def install_megatron_expert_remap() -> None:
    """Permute router columns and expert weights when ``LUMEN_EXPERT_REMAP=1``."""
    global _INSTALLED
    if not _enabled() or _INSTALLED:
        return

    from megatron.core.transformer.moe.router import TopKRouter
    import megatron.training.training as training

    original_routing = TopKRouter.routing

    def routing(self, logits, padding_mask=None):
        probs, routing_map = original_routing(self, logits, padding_mask)
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
        # A second train() call must not permute an already permuted layout,
        # and must not reload the fp32 master from the rounded bf16 copy.
        if _exchange_expert_placement(model):
            optimizer.reload_model_params()
        return original_train(*args, **kwargs)

    training.train = train
    _INSTALLED = True
