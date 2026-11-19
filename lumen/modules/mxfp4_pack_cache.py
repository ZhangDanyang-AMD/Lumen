# ruff: noqa: BLE001
"""Round-80 staged Lumen lifecycle: per-optimizer-step invalidation for the MXFP4 packed-weight
cache used by SonicMoE experts (w4a16 / a4w4 / a8w4 / a8w4f recipes).

CPU-only helper; it never packs and never touches tensor math.  It decides a cache strategy
(generation-stamped or fail-closed uncached), registers one optimizer post-step callback per
**optimizer** that only bumps per-module generation counters, and registers a checkpoint-load
callback that does the same.  Packing is deferred to the first forward after the optimizer's
main->model copy / param all-gather (see "Caller guard" below).

Why: the runtime module stamps its cache with ``(w1.data_ptr(), w1._version, ...)``.  Megatron's
distributed optimizer updates model params via flat-buffer ``.data.copy_`` (and the param
all-gather writes the same buffer directly), which does not bump ``Parameter._version`` in the
pinned Torch 2.9.1 image.  A version-only stamp can stay frozen across optimizer steps.  This
reuses Lumen's existing per-optimizer-step staleness pattern (the ``mark_fp8_params_stale``
analogue) but activates it for Sonic MXFP4 independently of the FP8 parameter count.

Environment contract (explicit; the legacy version-stamp cache is never kept silently):
  * ``R80_CACHE_GENERATION=1``  -> generation-stamped cache, but only if the full lifecycle
    (step hook + checkpoint-load hook) can be registered; otherwise fail-closed uncached.
  * unset / other               -> fail-closed **uncached** (repack every call, guaranteed fresh).
  * ``R80_CACHE_LEGACY=1``      -> explicit opt-in to the old version-only cache (debug/repro
    only; stale under Megatron optimizer writes).

Caller guard (deferred pack): a Torch ``optimizer.register_step_post_hook`` callback fires inside
``optimizer.step()``, i.e. **before** Megatron's ``_copy_main_params_to_model_params`` and the
param all-gather.  This helper therefore never packs in the hook; it only bumps generations, and
the next forward (after the copy/sync has run) repacks from the current weights.  Do not call
``pack_pre_routed_weights`` / ``_cached_packed_weights`` from any optimizer hook.

Install (training entry, after optimizer creation):
    from lumen.modules.mxfp4_pack_cache import install_mxfp4_pack_cache_lifecycle
    install_mxfp4_pack_cache_lifecycle(model, optimizer)
"""

from __future__ import annotations

import functools
import logging
import os
import weakref

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "collect_cache_modules",
    "install_mxfp4_pack_cache_lifecycle",
    "invalidate_mxfp4_pack_cache",
    "probe_mxfp4_pack_cache",
    "uninstall_mxfp4_pack_cache_lifecycle",
]

_ENV_GEN = "R80_CACHE_GENERATION"
_ENV_LEGACY = "R80_CACHE_LEGACY"

# optimizer id -> {"optimizer": weakref, "mechanism": str, "load_hook": bool,
#                  "models": [weakref per model]}
_INSTALLED: dict[int, dict] = {}


# --------------------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------------------
def _generation_enabled() -> bool:
    return os.environ.get(_ENV_GEN, "0") == "1"


def _legacy_enabled() -> bool:
    return os.environ.get(_ENV_LEGACY, "0") == "1"


def collect_cache_modules(model) -> list[torch.nn.Module]:
    """Modules exposing the SonicMoE packed-weight cache API."""
    return [
        m
        for m in model.modules()
        if hasattr(m, "_cached_packed_weights")
        and hasattr(m, "_invalidate_packed_weights")
    ]


def _get_entry(optimizer):
    """Registry lookup robust to Python id() reuse of dead optimizer objects."""
    key = id(optimizer)
    entry = _INSTALLED.get(key)
    if entry is None:
        return None
    holder = entry.get("optimizer")
    if holder is None or holder() is not optimizer:
        _INSTALLED.pop(key, None)
        return None
    return entry


def _pair_installed(model, optimizer) -> bool:
    entry = _get_entry(optimizer)
    if entry is None:
        return False
    return any(ref() is model for ref in entry["models"])


def _model_optimizer_ids(model) -> list:
    return list(getattr(model, "_r80_mxfp4_pack_cache_optimizer_ids", []))


def _remember_pair(model, optimizer, strategy: str) -> None:
    ids = _model_optimizer_ids(model)
    if id(optimizer) not in ids:
        ids.append(id(optimizer))
        model._r80_mxfp4_pack_cache_optimizer_ids = ids
    model._r80_mxfp4_pack_cache_strategy = strategy


def _model_mechanisms(model) -> list:
    out = []
    for oid in _model_optimizer_ids(model):
        entry = _INSTALLED.get(oid)
        if entry is not None:
            out.append(entry["mechanism"])
    return out


def probe_mxfp4_pack_cache(model) -> dict:
    """Structured read-only probe of the packed-weight cache capability on *model*."""
    mods = collect_cache_modules(model)
    return {
        "env_generation": _ENV_GEN,
        "env_legacy": _ENV_LEGACY,
        "generation_requested": _generation_enabled(),
        "legacy_requested": _legacy_enabled(),
        "modules": len(mods),
        "modules_with_generation": sum(
            1 for m in mods if hasattr(m, "_pack_generation")
        ),
        "modules_with_cache_switch": sum(
            1 for m in mods if hasattr(m, "_pack_cache_enabled")
        ),
        "optimizer_pairs": _model_optimizer_ids(model),
        "strategy": getattr(model, "_r80_mxfp4_pack_cache_strategy", None),
        "last_reason": getattr(model, "_r80_mxfp4_pack_cache_reason", None),
        "post_step_mechanisms": _model_mechanisms(model),
        "load_hooks": int(getattr(model, "_r80_mxfp4_pack_cache_load_hooks", 0)),
    }


# --------------------------------------------------------------------------------------
# generation / hooks
# --------------------------------------------------------------------------------------
def _mark_generation(module) -> None:
    """Deferred stale marker: bump generation only.  NEVER packs (may run inside optimizer)."""
    module._pack_generation = int(getattr(module, "_pack_generation", 0)) + 1


def invalidate_mxfp4_pack_cache(model) -> int:
    """Immediately mark all caches stale (checkpoint/eval/state-change edges); never packs."""
    mods = collect_cache_modules(model)
    for m in mods:
        _mark_generation(m)
    return len(mods)


def _make_wrapped_step(original_step, callback):
    @functools.wraps(original_step)
    def _wrapped_step(*args, **kwargs):
        out = original_step(*args, **kwargs)
        callback()  # deferred only: bump generation, no packing
        return out

    return _wrapped_step


def _register_post_step(optimizer, callback):
    """Return (mechanism|None, reason|None); never raises."""
    reg = getattr(optimizer, "register_step_post_hook", None)
    if callable(reg):
        try:
            reg(lambda _opt, _args, _kwargs: callback())
            return "register_step_post_hook", None
        except Exception as exc:  # some wrappers expose the attr but reject it
            logger.warning(
                "MXFP4 pack-cache: register_step_post_hook failed (%r); "
                "trying step wrapper",
                exc,
            )
    step = getattr(optimizer, "step", None)
    if callable(step):
        try:
            optimizer.step = _make_wrapped_step(step, callback)
            return "wrapped_optimizer_step", None
        except Exception as exc:
            return None, f"optimizer.step is not wrappable: {exc!r}"
    return None, "optimizer exposes neither a usable register_step_post_hook nor step()"


def _register_load_hook(model, callback):
    """Return (ok, reason); never raises.  Required for a generation-stamped cache."""
    reg = getattr(model, "register_load_state_dict_post_hook", None)
    if not callable(reg):
        return False, "model has no register_load_state_dict_post_hook"
    try:

        def _hook(*_args, **_kwargs):
            callback()
            # torch requires load-state post hooks to return None

        reg(_hook)
        return True, None
    except Exception as exc:
        return False, f"register_load_state_dict_post_hook failed: {exc!r}"


# --------------------------------------------------------------------------------------
# fail-closed
# --------------------------------------------------------------------------------------
def _force_uncached(mods) -> None:
    for m in mods:
        if hasattr(m, "_pack_cache_enabled"):
            m._pack_cache_enabled = False
        else:
            m._cached_packed_weights = (
                lambda: None
            )  # instance override, guaranteed fresh
        m._pack_generation = int(getattr(m, "_pack_generation", 0)) + 1


def _set_uncached_marker(model, reason: str) -> None:
    model._r80_mxfp4_pack_cache_strategy = "uncached_fresh"
    model._r80_mxfp4_pack_cache_reason = reason


def _model_reason(model) -> str:
    return getattr(model, "_r80_mxfp4_pack_cache_reason", "fail-closed uncached")


def _install_generation(model, optimizer, mods, handle: dict) -> dict:
    # Checkpoint hooks belong to each root, not to the shared optimizer. Joining an
    # existing optimizer must never inherit another root's checkpoint capability.
    own_load_hooks = int(getattr(model, "_r80_mxfp4_pack_cache_load_hooks", 0))
    if not own_load_hooks:
        model_ref = weakref.ref(model)

        def _on_load():
            obj = model_ref()
            if obj is not None:
                invalidate_mxfp4_pack_cache(obj)

        load_ok, load_reason = _register_load_hook(model, _on_load)
        if not load_ok:
            _force_uncached(mods)
            _set_uncached_marker(
                model,
                f"checkpoint-load hook unavailable ({load_reason}); "
                "fail-closed uncached",
            )
            handle.update(
                ok=True, strategy="uncached_fresh", reason=_model_reason(model)
            )
            return handle
        model._r80_mxfp4_pack_cache_load_hooks = 1

    entry = _get_entry(optimizer)
    if entry is None:
        entry = {
            "optimizer": weakref.ref(optimizer),
            "mechanism": None,
            "load_hook": True,
            "models": [],
        }

        def _on_step(_entry=entry):
            # Mark each module once even when two registered roots share a child.
            # Never pack before the optimizer's main->model copy/param all-gather.
            seen = set()
            for ref in list(_entry["models"]):
                obj = ref()
                if obj is None:
                    continue
                for m in collect_cache_modules(obj):
                    if id(m) not in seen:
                        seen.add(id(m))
                        _mark_generation(m)

        mechanism, reason = _register_post_step(optimizer, _on_step)
        if mechanism is None:
            _force_uncached(mods)
            _set_uncached_marker(model, f"{reason}; fail-closed uncached")
            handle.update(
                ok=True, strategy="uncached_fresh", reason=_model_reason(model)
            )
            return handle
        entry["mechanism"] = mechanism
        _INSTALLED[id(optimizer)] = entry
        logger.info(
            "MXFP4 pack-cache: post-step hook via %s, root load hook registered",
            mechanism,
        )

    entry["models"].append(weakref.ref(model))
    # Respect an explicit module cache opt-out; hook success is not permission to
    # override SONIC_MOE_PACK_CACHE=0. The first enabled forward consumes fresh weights.
    for m in mods:
        if hasattr(m, "_pack_cache_enabled"):
            m._pack_cache_enabled = bool(getattr(m, "_pack_cache_requested", True))
        _mark_generation(m)
    _remember_pair(model, optimizer, "generation")
    handle.update(
        ok=True,
        strategy="generation",
        reason="per-optimizer-step generation invalidation; repack deferred to next "
        "forward after optimizer copy/param sync",
        post_step_mechanism=entry["mechanism"],
        load_hook=True,
    )
    return handle


# --------------------------------------------------------------------------------------
# public install
# --------------------------------------------------------------------------------------
def install_mxfp4_pack_cache_lifecycle(model, optimizer=None) -> dict:
    """Install the lifecycle for (model, optimizer); idempotent per pair; never raises.

    - repeated call with the same (model, optimizer) -> {"reason": "already_installed"}
    - same model + a second optimizer -> a second step hook is registered for it
    - `optimizer=None` or any registration failure -> uncached (fresh) route
    - a later call with a real optimizer explicitly upgrades the uncached route to generation
      (generation bumped so the first forward repacks), keeping the model safe in between.
    """
    mods = collect_cache_modules(model)
    handle = {
        "ok": False,
        "env_generation": _ENV_GEN,
        "generation_requested": _generation_enabled(),
        "legacy_requested": _legacy_enabled(),
        "modules": len(mods),
        "strategy": None,
        "reason": None,
        "upgraded_from_uncached": False,
        "post_step_mechanism": None,
        "load_hook": False,
    }
    if not mods:
        handle.update(
            ok=True,
            strategy="noop_no_cache_found",
            reason="no module exposes _cached_packed_weights/_invalidate_packed_weights",
        )
        return handle

    if _legacy_enabled():
        # Explicit debug/repro opt-in: restore the old _version-based cache.
        # This is unsafe for Megatron's version-preserving parameter copies, so it
        # remains isolated from the generation strategy and is never selected by default.
        for m in mods:
            if hasattr(m, "_pack_cache_enabled"):
                m._pack_cache_enabled = bool(getattr(m, "_pack_cache_requested", True))
        handle.update(
            ok=True,
            strategy="legacy_version_stamp",
            reason=f"{_ENV_LEGACY}=1 explicit opt-in to the old version-only cache "
            "(stale under Megatron optimizer writes; debug only)",
        )
        logger.warning("MXFP4 pack-cache: %s", handle["reason"])
        return handle

    if optimizer is not None and _pair_installed(model, optimizer):
        entry = _get_entry(optimizer)
        handle.update(
            ok=True,
            strategy="generation",
            reason="already_installed",
            post_step_mechanism=entry["mechanism"],
            load_hook=entry.get("load_hook", False),
        )
        return handle

    if not _generation_enabled():
        _force_uncached(mods)
        _set_uncached_marker(
            model,
            f"{_ENV_GEN} not 1; fail-closed uncached repack (the old version-only "
            "stamp is stale under Megatron optimizer writes)",
        )
        handle.update(ok=True, strategy="uncached_fresh", reason=_model_reason(model))
        logger.warning("MXFP4 pack-cache: %s", handle["reason"])
        return handle

    missing = [type(m).__name__ for m in mods if not hasattr(m, "_pack_generation")]
    if missing:
        _force_uncached(mods)
        _set_uncached_marker(
            model,
            "generation requested but module lacks _pack_generation "
            f"({len(missing)} modules); fail-closed uncached",
        )
        handle.update(ok=True, strategy="uncached_fresh", reason=_model_reason(model))
        logger.warning("MXFP4 pack-cache: %s", handle["reason"])
        return handle

    if optimizer is None:
        was_uncached = (
            getattr(model, "_r80_mxfp4_pack_cache_strategy", None) == "uncached_fresh"
        )
        _force_uncached(mods)
        _set_uncached_marker(
            model,
            "generation requested but no optimizer supplied; "
            "fail-closed uncached until a real optimizer is installed",
        )
        handle.update(
            ok=True,
            strategy="uncached_fresh",
            reason=_model_reason(model),
            upgraded_from_uncached=was_uncached,
        )
        logger.warning("MXFP4 pack-cache: %s", handle["reason"])
        return handle

    was_uncached = (
        getattr(model, "_r80_mxfp4_pack_cache_strategy", None) == "uncached_fresh"
    )
    handle = _install_generation(model, optimizer, mods, handle)
    handle["upgraded_from_uncached"] = bool(
        was_uncached and handle.get("strategy") == "generation"
    )
    return handle


def uninstall_mxfp4_pack_cache_lifecycle(model, optimizer=None) -> None:
    """Test-only marker reset; torch-registered hooks remain (they cannot be removed safely)."""
    for attr in (
        "_r80_mxfp4_pack_cache_strategy",
        "_r80_mxfp4_pack_cache_reason",
        "_r80_mxfp4_pack_cache_optimizer_ids",
        "_r80_mxfp4_pack_cache_load_hooks",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
    if optimizer is not None:
        entry = _get_entry(optimizer)
        if entry is not None:
            entry["models"] = [
                r for r in entry["models"] if r() is not None and r() is not model
            ]
            if not entry["models"]:
                _INSTALLED.pop(id(optimizer), None)
