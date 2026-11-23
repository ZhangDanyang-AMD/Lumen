"""Append each MORI layer's host-synced local expert counts to a per-rank jsonl file."""

from __future__ import annotations

import atexit
import json
import os

_ENV = "LUMEN_LOG_EXPERT_COUNTS"
_STATE = {"file": None}


def _log_file():
    if _STATE["file"] is None:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        directory = os.environ[_ENV]
        os.makedirs(directory, exist_ok=True)
        _STATE["file"] = open(os.path.join(directory, f"expert-counts-rank{rank}.jsonl"), "a")
        _STATE["rank"] = rank
        atexit.register(_STATE["file"].close)
    return _STATE["file"]


def _iteration():
    try:
        from megatron.training import get_args

        return getattr(get_args(), "curr_iteration", None)
    except Exception:
        return None


def install_expert_count_log(moe_layer, layer_index: int) -> bool:
    """Log the counts the manager already synced to host; adds no device sync of its own."""
    if not os.environ.get(_ENV):
        return False
    manager = getattr(getattr(moe_layer, "token_dispatcher", None), "_comm_manager", None)
    if type(manager).__name__ != "_MoriManager":
        return False
    inner = vars(manager).get("_synced_tokens_per_expert")
    layer = getattr(moe_layer, "layer_number", layer_index)
    state = {"logged": None, "calls": 0}

    def _synced_tokens_per_expert():
        counts = inner() if inner is not None else type(manager)._synced_tokens_per_expert(manager)
        # Called twice per dispatch (permute, then the expert counts); log each dispatch once.
        dispatched = manager.tokens_per_expert
        if dispatched is not state["logged"] and not counts.is_cuda:
            state["logged"] = dispatched
            f = _log_file()
            f.write(
                json.dumps(
                    {
                        "it": _iteration(),
                        "call": state["calls"],
                        "layer": layer,
                        "rank": _STATE["rank"],
                        "counts": counts.tolist(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            state["calls"] += 1
        return counts

    manager._synced_tokens_per_expert = _synced_tokens_per_expert
    return True


__all__ = ["install_expert_count_log"]
