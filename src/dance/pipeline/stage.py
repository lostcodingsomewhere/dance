"""
Stage protocol — the contract every pipeline stage implements.

A Stage is responsible for one transition in the Track lifecycle: it consumes
tracks in ``input_state`` and (on success) writes ``output_state`` back. The
dispatcher discovers registered stages and runs whatever is ready; stages know
nothing about each other.

This is deliberately tiny. Don't add complexity unless a real consumer
requires it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import Track, TrackState


@runtime_checkable
class Stage(Protocol):
    """A unit of pipeline work."""

    name: str
    input_state: TrackState
    output_state: TrackState
    error_state: TrackState = TrackState.ERROR

    def process(self, session: Session, track: Track, settings: Settings) -> bool:
        """Run the stage on one track.

        Implementations should:
        - Set ``track.state`` to an in-progress state if useful (optional).
        - On success: write the output_state, commit, return True.
        - On failure: set state to ``error_state`` with ``error_message``,
          commit, return False.

        Exceptions raised will be caught by the dispatcher; the stage doesn't
        need a top-level try/except for control flow.
        """
        ...
