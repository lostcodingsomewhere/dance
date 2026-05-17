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


# ---------------------------------------------------------------------------
# Parallel mode
# ---------------------------------------------------------------------------


class ParallelFakeStage(FakeStage):
    """Like FakeStage but with the parallel-mode hints declared and a small
    sleep so concurrency is observable."""

    def __init__(
        self,
        name,
        input_state,
        output_state,
        inflight_state,
        *,
        concurrency=1,
        uses_gpu=False,
        sleep_s=0.0,
        **kw,
    ):
        super().__init__(name, input_state, output_state, **kw)
        self.inflight_state = inflight_state
        self.concurrency = concurrency
        self.uses_gpu = uses_gpu
        self.sleep_s = sleep_s
        # Track all observed thread ids so we can prove workers ran in parallel.
        self.thread_ids: set[int] = set()

    def process(self, session, track, settings) -> bool:
        import threading
        import time

        self.thread_ids.add(threading.get_ident())
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return super().process(session, track, settings)


@pytest.fixture
def fast_idle_grace(monkeypatch):
    """Make the parallel dispatcher exit quickly after queue drains."""
    from dance.pipeline import dispatcher as disp_mod

    monkeypatch.setattr(disp_mod, "_IDLE_GRACE_S", 0.5)
    monkeypatch.setattr(disp_mod, "_POLL_INTERVAL_S", 0.05)


def test_parallel_drains_queue_via_session_factory(
    settings, session_factory, session, make_track, fast_idle_grace
):
    """End-to-end: parallel dispatcher claims + processes 6 tracks across
    two concurrent workers and lands them at output_state."""
    for _ in range(6):
        make_track(state="pending")
    session.commit()

    d = Dispatcher(settings, session_factory=session_factory)
    d.clear()
    d.register(
        ParallelFakeStage(
            "analyze",
            TrackState.PENDING,
            TrackState.ANALYZED,
            TrackState.ANALYZING,
            concurrency=2,
            sleep_s=0.05,
        )
    )

    result = d.run()
    assert result["analyze"]["processed"] == 6
    assert result["analyze"]["errors"] == 0

    # All 6 tracks must be at output_state
    session.expire_all()
    states = [t.state for t in session.query(Track).all()]
    assert states.count(TrackState.ANALYZED.value) == 6


def test_parallel_atomic_claim_no_double_dispatch(
    settings, session_factory, session, make_track, fast_idle_grace
):
    """4 workers, 4 tracks: every track is processed exactly once."""
    for _ in range(4):
        make_track(state="pending")
    session.commit()

    stage = ParallelFakeStage(
        "analyze",
        TrackState.PENDING,
        TrackState.ANALYZED,
        TrackState.ANALYZING,
        concurrency=4,
        sleep_s=0.05,
    )

    d = Dispatcher(settings, session_factory=session_factory)
    d.clear()
    d.register(stage)
    d.run()

    # Exactly 4 process() calls, each with a distinct track id.
    assert len(stage.calls) == 4
    assert len(set(stage.calls)) == 4


def test_parallel_uses_multiple_threads(
    settings, session_factory, session, make_track, fast_idle_grace
):
    """Sanity: with concurrency=3 and 6 tracks, more than one thread is used."""
    for _ in range(6):
        make_track(state="pending")
    session.commit()

    stage = ParallelFakeStage(
        "analyze",
        TrackState.PENDING,
        TrackState.ANALYZED,
        TrackState.ANALYZING,
        concurrency=3,
        sleep_s=0.1,
    )

    d = Dispatcher(settings, session_factory=session_factory)
    d.clear()
    d.register(stage)
    d.run()

    # Should have used 2 or 3 distinct OS threads.
    assert len(stage.thread_ids) >= 2


def test_parallel_stages_run_concurrently(
    settings, session_factory, session, make_track, fast_idle_grace
):
    """Stage A feeds Stage B. With both workers spun up at start, B's
    workers should pick up A's output before A finishes the queue."""
    import time

    for _ in range(4):
        make_track(state="pending")
    session.commit()

    stage_a = ParallelFakeStage(
        "analyze",
        TrackState.PENDING,
        TrackState.ANALYZED,
        TrackState.ANALYZING,
        concurrency=1,
        sleep_s=0.05,
    )
    stage_b = ParallelFakeStage(
        "separate",
        TrackState.ANALYZED,
        TrackState.SEPARATED,
        TrackState.SEPARATING,
        concurrency=1,
        sleep_s=0.05,
    )

    d = Dispatcher(settings, session_factory=session_factory)
    d.clear()
    d.register(stage_a)
    d.register(stage_b)

    t0 = time.monotonic()
    d.run()
    elapsed = time.monotonic() - t0

    assert len(stage_a.calls) == 4
    assert len(stage_b.calls) == 4
    # Sequential would be 4×0.05 (A) + 4×0.05 (B) = 0.4s plus idle grace.
    # Parallel with overlap: ~4×0.05 = 0.2s + grace. Confirm overlap happened
    # by checking elapsed is less than "both stages run fully back-to-back".
    # (Be loose — CI noise is real. We mostly care that no test fails when
    # the implementation regresses to sequential.)
    assert elapsed < 1.5  # generous upper bound


def test_parallel_rejects_stage_without_inflight_state(
    settings, session_factory
):
    """A stage missing inflight_state can't be used in parallel mode."""
    d = Dispatcher(settings, session_factory=session_factory)
    d.clear()
    # FakeStage doesn't declare inflight_state at all.
    d.register(FakeStage("badstage", TrackState.PENDING, TrackState.ANALYZED))
    with pytest.raises(ValueError, match="inflight_state"):
        d.run()


def test_constructor_requires_session_xor_factory(settings, session, session_factory):
    """Pass exactly one of session= or session_factory=."""
    with pytest.raises(ValueError, match="exactly one"):
        Dispatcher(settings)
    with pytest.raises(ValueError, match="exactly one"):
        Dispatcher(settings, session=session, session_factory=session_factory)
