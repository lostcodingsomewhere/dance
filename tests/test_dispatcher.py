"""Tests for the Stage protocol and Dispatcher.

These use synthetic in-memory stages so they're fast and don't require audio
processing libs. Fixtures come from ``conftest.py``.
"""

from __future__ import annotations

import pytest

from dance.config import Settings
from dance.core.database import Track, TrackState
from dance.pipeline.dispatcher import Dispatcher
from dance.pipeline.events import StageEvent
from dance.pipeline.stage import Stage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeStage:
    """Concrete stage that doesn't touch audio."""

    error_state = TrackState.ERROR

    def __init__(
        self,
        name: str,
        input_state: TrackState,
        output_state: TrackState,
        *,
        succeed: bool = True,
        crash: bool = False,
    ) -> None:
        self.name = name
        self.input_state = input_state
        self.output_state = output_state
        self.succeed = succeed
        self.crash = crash
        self.calls: list[int] = []

    def process(self, session, track, settings) -> bool:
        self.calls.append(track.id)
        if self.crash:
            raise RuntimeError("boom")
        if self.succeed:
            track.state = self.output_state.value
            session.commit()
            return True
        track.state = self.error_state.value
        track.error_message = "fake failure"
        session.commit()
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def dispatcher(settings, session) -> Dispatcher:
    d = Dispatcher(settings, session)
    d.clear()  # start with no default stages — tests register their own
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_protocol_compliance():
    s = FakeStage("x", TrackState.PENDING, TrackState.ANALYZED)
    assert isinstance(s, Stage)


def test_register_rejects_duplicate(dispatcher):
    dispatcher.register(FakeStage("a", TrackState.PENDING, TrackState.ANALYZED))
    with pytest.raises(ValueError):
        dispatcher.register(FakeStage("a", TrackState.ANALYZED, TrackState.SEPARATED))


def test_register_rejects_non_stage(dispatcher):
    with pytest.raises(TypeError):
        dispatcher.register("not a stage")  # type: ignore[arg-type]


def test_runs_stage_on_matching_track(dispatcher, session, make_track):
    stage = FakeStage("analyze", TrackState.PENDING, TrackState.ANALYZED)
    dispatcher.register(stage)
    track = make_track()
    session.commit()

    result = dispatcher.run()

    assert result["analyze"]["processed"] == 1
    session.refresh(track)
    assert track.state == TrackState.ANALYZED.value
    assert stage.calls == [track.id]


def test_chains_stages_in_one_run(dispatcher, session, make_track):
    """A single run() walks PENDING → ANALYZED → SEPARATED."""
    dispatcher.register(FakeStage("analyze", TrackState.PENDING, TrackState.ANALYZED))
    dispatcher.register(FakeStage("separate", TrackState.ANALYZED, TrackState.SEPARATED))
    track = make_track()
    session.commit()

    dispatcher.run()

    session.refresh(track)
    assert track.state == TrackState.SEPARATED.value


def test_skip_advances_state_without_running(dispatcher, session, make_track):
    """skip={'separate'} should fast-forward through the stage."""
    dispatcher.register(FakeStage("analyze", TrackState.PENDING, TrackState.ANALYZED))
    sep = FakeStage("separate", TrackState.ANALYZED, TrackState.SEPARATED)
    dispatcher.register(sep)
    dispatcher.register(FakeStage("post", TrackState.SEPARATED, TrackState.COMPLETE))
    track = make_track()
    session.commit()

    result = dispatcher.run(skip={"separate"})

    session.refresh(track)
    assert track.state == TrackState.COMPLETE.value
    assert sep.calls == []
    assert result["separate"]["skipped"] == 1


def test_failure_records_error_state(dispatcher, session, make_track):
    dispatcher.register(FakeStage("analyze", TrackState.PENDING, TrackState.ANALYZED, succeed=False))
    track = make_track()
    session.commit()

    result = dispatcher.run()

    session.refresh(track)
    assert track.state == TrackState.ERROR.value
    assert track.error_message == "fake failure"
    assert result["analyze"]["errors"] == 1


def test_exception_in_stage_is_caught(dispatcher, session, make_track):
    """An uncaught exception should mark the track as ERROR, not propagate."""
    dispatcher.register(FakeStage("crash", TrackState.PENDING, TrackState.ANALYZED, crash=True))
    track = make_track()
    session.commit()

    result = dispatcher.run()

    session.refresh(track)
    assert track.state == TrackState.ERROR.value
    assert "boom" in (track.error_message or "")
    assert result["crash"]["errors"] == 1


def test_track_id_filter(dispatcher, session, make_track):
    dispatcher.register(FakeStage("a", TrackState.PENDING, TrackState.ANALYZED))
    t1 = make_track()
    t2 = make_track()
    session.commit()

    dispatcher.run(track_id=t1.id)

    session.refresh(t1)
    session.refresh(t2)
    assert t1.state == TrackState.ANALYZED.value
    assert t2.state == TrackState.PENDING.value


def test_events_fire_in_order(dispatcher, session, make_track):
    """Subscribers see started → completed for success, started → failed for failure."""
    dispatcher.register(FakeStage("a", TrackState.PENDING, TrackState.ANALYZED))
    make_track()
    session.commit()

    events: list[StageEvent] = []
    dispatcher.events.subscribe(events.append)

    dispatcher.run()

    kinds = [e.kind for e in events]
    assert kinds == ["started", "completed"]
    assert events[1].duration_ms is not None
    assert events[1].duration_ms >= 0


def test_no_progress_terminates(dispatcher, session):
    """If nothing matches input_state, run() returns immediately without spinning."""
    dispatcher.register(FakeStage("noop", TrackState.PENDING, TrackState.ANALYZED))
    # No tracks at all.

    result = dispatcher.run()
    assert result["noop"]["processed"] == 0
    assert result["noop"]["errors"] == 0
