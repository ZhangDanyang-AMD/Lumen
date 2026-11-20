"""One-shot MORI-to-Sonic routing handoff."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock, local
from typing import Any

MAPPED_HANDOFF_API_VERSION = 1


class MappedHandoffError(RuntimeError):
    """Raised when the mapped routing handoff contract is violated."""


@dataclass(frozen=True)
class _TensorStamp:
    identity: int
    version: int | None
    shape: tuple[int, ...]
    device: str
    dtype: str

    @classmethod
    def capture(cls, tensor: Any) -> _TensorStamp:
        return cls(
            identity=id(tensor),
            version=getattr(tensor, "_version", None),
            shape=tuple(tensor.shape),
            device=str(tensor.device),
            dtype=str(tensor.dtype),
        )

    def validate(self, tensor: Any, name: str) -> None:
        if self != self.capture(tensor):
            raise MappedHandoffError(f"MUTATED_{name.upper()}")


@dataclass(frozen=True)
class MappedHandoffV1:
    """Versioned expert-major view over an unsorted source token tensor."""

    source_tokens: Any
    scores: Any
    counts: tuple[int, ...]
    gather_map: Any
    source_stamp: _TensorStamp
    scores_stamp: _TensorStamp
    gather_stamp: _TensorStamp
    api_version: int = MAPPED_HANDOFF_API_VERSION

    @classmethod
    def create(
        cls,
        source_tokens: Any,
        scores: Any,
        counts: Any,
        gather_map: Any,
    ) -> MappedHandoffV1:
        normalized_counts = _normalize_counts(counts)
        if not normalized_counts or any(count < 0 for count in normalized_counts):
            raise MappedHandoffError("INVALID_COUNTS")
        if len(gather_map.shape) != 1 or len(scores.shape) != 1:
            raise MappedHandoffError("INVALID_ROUTING_RANK")
        if gather_map.shape[0] != scores.shape[0]:
            raise MappedHandoffError("ROUTING_ROW_MISMATCH")
        if sum(normalized_counts) != gather_map.shape[0]:
            raise MappedHandoffError("COUNT_SUM_MISMATCH")
        if (
            source_tokens.device != scores.device
            or source_tokens.device != gather_map.device
        ):
            raise MappedHandoffError("ROUTING_DEVICE_MISMATCH")
        return cls(
            source_tokens=source_tokens,
            scores=scores,
            counts=normalized_counts,
            gather_map=gather_map,
            source_stamp=_TensorStamp.capture(source_tokens),
            scores_stamp=_TensorStamp.capture(scores),
            gather_stamp=_TensorStamp.capture(gather_map),
        )

    def validate(self, source_tokens: Any, scores: Any, counts: Any) -> None:
        if self.api_version != MAPPED_HANDOFF_API_VERSION:
            raise MappedHandoffError("UNSUPPORTED_HANDOFF_VERSION")
        if source_tokens is not self.source_tokens or scores is not self.scores:
            raise MappedHandoffError("ROUTING_IDENTITY_MISMATCH")
        if _normalize_counts(counts) != self.counts:
            raise MappedHandoffError("ROUTING_COUNTS_MISMATCH")
        self.source_stamp.validate(self.source_tokens, "source_tokens")
        self.scores_stamp.validate(self.scores, "scores")
        self.gather_stamp.validate(self.gather_map, "gather_map")


@dataclass
class _HandoffScope:
    enabled: bool
    handoff: MappedHandoffV1 | None = None
    published: bool = False
    consumed: bool = False


_ACTIVE_SCOPE: ContextVar[_HandoffScope | None] = ContextVar(
    "lumen_sonic_moe_mapped_handoff", default=None
)
_THREAD_SCOPE = local()
_PROCESS_SCOPE_LOCK = Lock()
_PROCESS_SCOPES: dict[int, _HandoffScope] = {}
_SOURCE_SCOPE_LOCK = Lock()
_SOURCE_SCOPES: dict[int, tuple[Any, _HandoffScope]] = {}


def _current_scope() -> _HandoffScope | None:
    """Return the context scope, or its same-thread compatibility fallback."""
    scope = _ACTIVE_SCOPE.get()
    if scope is not None:
        return scope
    scope = getattr(_THREAD_SCOPE, "scope", None)
    if scope is not None:
        return scope
    with _PROCESS_SCOPE_LOCK:
        if len(_PROCESS_SCOPES) == 1:
            return next(iter(_PROCESS_SCOPES.values()))
    return None


def _normalize_counts(counts: Any) -> tuple[int, ...]:
    if hasattr(counts, "tolist"):
        counts = counts.tolist()
    return tuple(int(count) for count in counts)


def register_mapped_handoff_source(scope: _HandoffScope, source_tokens: Any) -> None:
    """Bind one routed invocation to its exact dispatch tensor."""
    if not isinstance(scope, _HandoffScope):
        raise MappedHandoffError("INVALID_HANDOFF_SCOPE")
    key = id(source_tokens)
    with _SOURCE_SCOPE_LOCK:
        if key in _SOURCE_SCOPES:
            raise MappedHandoffError("DUPLICATE_HANDOFF_SOURCE")
        _SOURCE_SCOPES[key] = (source_tokens, scope)


def mapped_handoff_scope_for_source(source_tokens: Any) -> _HandoffScope | None:
    """Resolve a worker-thread scope by exact source tensor identity."""
    with _SOURCE_SCOPE_LOCK:
        entry = _SOURCE_SCOPES.get(id(source_tokens))
    if entry is None:
        return None
    registered_source, scope = entry
    if registered_source is not source_tokens:
        raise MappedHandoffError("HANDOFF_SOURCE_IDENTITY_COLLISION")
    return scope


def unregister_mapped_handoff_source(scope: _HandoffScope, source_tokens: Any) -> None:
    """Release one source binding after routed expert compute completes."""
    key = id(source_tokens)
    with _SOURCE_SCOPE_LOCK:
        entry = _SOURCE_SCOPES.pop(key, None)
    if entry is None or entry[0] is not source_tokens or entry[1] is not scope:
        raise MappedHandoffError("HANDOFF_SOURCE_REGISTRY_MISMATCH")


@contextmanager
def mapped_handoff_scope(enabled: bool):
    """Bound one optional handoff to a single routed-expert invocation."""
    if _current_scope() is not None:
        raise MappedHandoffError("NESTED_HANDOFF_SCOPE")
    scope = _HandoffScope(enabled=enabled)
    token = _ACTIVE_SCOPE.set(scope)
    _THREAD_SCOPE.scope = scope
    with _PROCESS_SCOPE_LOCK:
        _PROCESS_SCOPES[id(scope)] = scope
    failed = False
    try:
        yield scope
    except BaseException:
        failed = True
        raise
    finally:
        _ACTIVE_SCOPE.reset(token)
        del _THREAD_SCOPE.scope
        with _PROCESS_SCOPE_LOCK:
            removed = _PROCESS_SCOPES.pop(id(scope), None)
        if removed is not scope:
            raise MappedHandoffError("HANDOFF_SCOPE_REGISTRY_MISMATCH")
        if not failed and scope.enabled and scope.published and not scope.consumed:
            raise MappedHandoffError("UNCONSUMED_HANDOFF")


def publish_mapped_handoff(
    source_tokens: Any,
    scores: Any,
    counts: Any,
    gather_map: Any,
    *,
    scope: _HandoffScope | None = None,
) -> bool:
    """Publish one handoff, or return False when the surrounding scope disabled it."""
    scope = scope if scope is not None else _current_scope()
    if scope is None or not scope.enabled:
        return False
    if not isinstance(scope, _HandoffScope):
        raise MappedHandoffError("INVALID_HANDOFF_SCOPE")
    if scope.published:
        raise MappedHandoffError("DUPLICATE_HANDOFF")
    scope.handoff = MappedHandoffV1.create(source_tokens, scores, counts, gather_map)
    scope.published = True
    return True


def mapped_handoff_enabled(scope: _HandoffScope | None = None) -> bool:
    """Whether the current routed-expert scope accepts a mapped handoff."""
    scope = scope if scope is not None else _current_scope()
    return scope is not None and scope.enabled


def consume_mapped_handoff(
    source_tokens: Any,
    scores: Any,
    counts: Any,
) -> MappedHandoffV1 | None:
    """Consume the active handoff once; absence means use the normal expert path."""
    scope = _current_scope()
    if scope is None or not scope.enabled:
        return None
    if scope.consumed:
        raise MappedHandoffError("HANDOFF_ALREADY_CONSUMED")
    if scope.handoff is None:
        return None
    scope.handoff.validate(source_tokens, scores, counts)
    scope.consumed = True
    return scope.handoff


__all__ = [
    "MAPPED_HANDOFF_API_VERSION",
    "MappedHandoffError",
    "MappedHandoffV1",
    "consume_mapped_handoff",
    "mapped_handoff_enabled",
    "mapped_handoff_scope",
    "mapped_handoff_scope_for_source",
    "publish_mapped_handoff",
    "register_mapped_handoff_source",
    "unregister_mapped_handoff_source",
]
