"""One-shot MORI-to-Sonic routing handoff."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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


def _normalize_counts(counts: Any) -> tuple[int, ...]:
    if hasattr(counts, "tolist"):
        counts = counts.tolist()
    return tuple(int(count) for count in counts)


@contextmanager
def mapped_handoff_scope(enabled: bool):
    """Bound one optional handoff to a single routed-expert invocation."""
    if _ACTIVE_SCOPE.get() is not None:
        raise MappedHandoffError("NESTED_HANDOFF_SCOPE")
    scope = _HandoffScope(enabled=enabled)
    token = _ACTIVE_SCOPE.set(scope)
    failed = False
    try:
        yield scope
    except BaseException:
        failed = True
        raise
    finally:
        _ACTIVE_SCOPE.reset(token)
        if not failed and scope.enabled and scope.published and not scope.consumed:
            raise MappedHandoffError("UNCONSUMED_HANDOFF")


def publish_mapped_handoff(
    source_tokens: Any,
    scores: Any,
    counts: Any,
    gather_map: Any,
) -> bool:
    """Publish one handoff, or return False when the surrounding scope disabled it."""
    scope = _ACTIVE_SCOPE.get()
    if scope is None or not scope.enabled:
        return False
    if scope.published:
        raise MappedHandoffError("DUPLICATE_HANDOFF")
    scope.handoff = MappedHandoffV1.create(source_tokens, scores, counts, gather_map)
    scope.published = True
    return True


def mapped_handoff_enabled() -> bool:
    """Whether the current routed-expert scope accepts a mapped handoff."""
    scope = _ACTIVE_SCOPE.get()
    return scope is not None and scope.enabled


def consume_mapped_handoff(
    source_tokens: Any,
    scores: Any,
    counts: Any,
) -> MappedHandoffV1 | None:
    """Consume the active handoff once; absence means use the normal expert path."""
    scope = _ACTIVE_SCOPE.get()
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
    "publish_mapped_handoff",
]
