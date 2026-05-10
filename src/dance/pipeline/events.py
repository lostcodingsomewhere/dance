"""
Stage lifecycle events.

Subscribers register callbacks for `on_stage_started`, `on_stage_completed`,
`on_stage_failed`. Used by the dispatcher to surface progress, by tests to
assert order, and (later) by the companion app's WebSocket layer to push
live progress.

Keep it boring: in-process, synchronous, no thread-safety guarantees beyond
what a Python list gives you. If we ever need cross-process events we'll
revisit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from dance.core.database import Track

EventKind = Literal["started", "completed", "failed", "skipped"]


@dataclass(frozen=True)
class StageEvent:
    kind: EventKind
    stage_name: str
    track_id: int
    track_title: str | None
    duration_ms: int | None = None
    error: str | None = None


Listener = Callable[[StageEvent], None]


class EventBus:
    """Trivial pub/sub. One bus per dispatcher instance."""

    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def publish(self, event: StageEvent) -> None:
        for listener in self._listeners:
            listener(event)

    def emit(self, kind: EventKind, stage_name: str, track: Track, **kwargs) -> None:
        self.publish(
            StageEvent(
                kind=kind,
                stage_name=stage_name,
                track_id=track.id,
                track_title=track.title,
                **kwargs,
            )
        )
