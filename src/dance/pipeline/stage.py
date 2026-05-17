"""
Stage protocol — the contract every pipeline stage implements.

A Stage is responsible for one transition in the Track lifecycle: it consumes
tracks in ``input_state`` and (on success) writes ``output_state`` back. The
dispatcher discovers registered stages and runs whatever is ready; stages know
nothing about each other.

This is deliberately tiny. Don't add complexity unless a real consumer
requires it.

Concurrency knobs (used only by the parallel dispatcher path; sequential mode
ignores them):

- ``concurrency`` — how many worker threads to run for this stage. Set higher
  than 1 only for stages that release the GIL during compute (numpy / librosa /
  torch / mutagen all do).
- ``uses_gpu`` — when True, every ``process()`` call goes through a
  module-level semaphore so MPS isn't oversubscribed. ``separate`` and
  ``embed`` are the GPU stages today.
- ``inflight_state`` — the *-ING TrackState a track sits in while a worker is
  processing it. The dispatcher atomically flips ``input_state -> inflight_state``
  at claim time so two workers can never grab the same track. ``None`` means
  the legacy sequential dispatcher (no atomic claim — tracks stay in
  input_state until ``process()`` writes output_state). The dispatcher's
  parallel path requires this be non-None.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import Track, TrackState


@runtime_checkable
class Stage(Protocol):
    """A unit of pipeline work.

    Required surface (enforced by ``isinstance(x, Stage)``):

    - ``name``: str
    - ``input_state``: TrackState
    - ``output_state``: TrackState
    - ``error_state``: TrackState
    - ``process(session, track, settings) -> bool``

    Optional parallel-mode hints (read by the dispatcher via getattr with
    defaults, so existing stage classes work unchanged):

    - ``inflight_state: TrackState`` — the *-ING state a track sits in
      while a worker is processing it. The parallel dispatcher uses this
      to atomically claim tracks; without it, the stage falls back to
      sequential-only behavior.
    - ``concurrency: int`` — how many worker threads the parallel
      dispatcher gives this stage. Default: 1.
    - ``uses_gpu: bool`` — if True, the parallel dispatcher gates each
      ``process()`` call on a module-level semaphore so MPS isn't
      oversubscribed. Default: False.
    """

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
