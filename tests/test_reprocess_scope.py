"""``dance reprocess`` touches only the tracks it selected.

Regression for a review finding: with more than one selected track the
reprocess loop called ``Dispatcher.run(track_id=None)``, so the dispatcher
processed every queued track in the library, and with ``--skip-embeddings``
its up-front auto-advance moved EVERY ``regions_detected`` track to
``complete`` with no CLAP embedding (``dance process`` never revisits a
``complete`` track, so those embeddings were lost for good).

The dispatcher now takes a collection of ids as its ``track_id`` scope, and
``run_until_settled`` passes the selection. These tests put unselected
tracks next to the selected ones (a ``pending`` one whose audio is missing,
so an analyze claim would error it, and a ``regions_detected`` one) and
check they don't move.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from dance import cli
from dance.config import Settings, get_settings
from dance.core.database import (
    StemFile,
    StemKind,
    Track,
    TrackState,
    get_session,
    init_db,
    now_utc,
)
from dance.pipeline import dispatcher as dispatcher_mod
from dance.pipeline import preflight
from dance.pipeline.dispatcher import Dispatcher
from dance.pipeline.reprocess import (
    advance_skipped,
    plan_reprocess,
    run_until_settled,
    tracks_with_stems_on_disk,
)

S = TrackState


class _FakeStage:
    """Parallel-capable stage that records which tracks it processed."""

    error_state = S.ERROR
    concurrency = 2
    uses_gpu = False

    def __init__(self, name: str, src: S, inflight: S, dst: S) -> None:
        self.name = name
        self.input_state = src
        self.inflight_state = inflight
        self.output_state = dst
        self.calls: list[int] = []

    def process(self, session, track, settings) -> bool:  # noqa: ARG002
        self.calls.append(int(track.id))
        track.state = self.output_state.value
        session.commit()
        return True


def _fake_pipeline() -> list[_FakeStage]:
    """The default pipeline's states, without any audio work."""
    return [
        _FakeStage("analyze", S.PENDING, S.ANALYZING, S.ANALYZED),
        _FakeStage("separate", S.ANALYZED, S.SEPARATING, S.SEPARATED),
        _FakeStage("analyze_stems", S.SEPARATED, S.ANALYZING_STEMS, S.STEMS_ANALYZED),
        _FakeStage("detect_regions", S.STEMS_ANALYZED, S.DETECTING_REGIONS, S.REGIONS_DETECTED),
        _FakeStage("embed", S.REGIONS_DETECTED, S.EMBEDDING, S.COMPLETE),
    ]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def fast_idle(monkeypatch):
    """Parallel workers exit quickly once their (scoped) queue is empty."""
    monkeypatch.setattr(dispatcher_mod, "_IDLE_GRACE_S", 0.3)
    monkeypatch.setattr(dispatcher_mod, "_POLL_INTERVAL_S", 0.02)


def _dispatcher(mode: str, settings, session, session_factory, stages=None) -> Dispatcher:
    if mode == "sequential":
        d = Dispatcher(settings, session)
    else:
        d = Dispatcher(settings, session_factory=session_factory)
    if stages is not None:
        d.clear()
        for stage in stages:
            d.register(stage)
    return d


def _states(session, *tracks: Track) -> list[str]:
    session.expire_all()
    return [session.get(Track, t.id).state for t in tracks]


# ---------------------------------------------------------------------------
# Dispatcher: a collection of ids is a scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_dispatcher_runs_only_the_ids_in_scope(
    mode, settings, session, session_factory, make_track, fast_idle
):
    t = [make_track(state=S.PENDING.value) for _ in range(4)]
    session.commit()
    stages = _fake_pipeline()
    d = _dispatcher(mode, settings, session, session_factory, stages)

    d.run(track_id={t[0].id, t[2].id})

    assert _states(session, *t) == ["complete", "pending", "complete", "pending"]
    for stage in stages:
        assert sorted(stage.calls) == [t[0].id, t[2].id], stage.name


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_dispatcher_scoped_skip_advances_only_the_scope(
    mode, settings, session, session_factory, make_track, fast_idle
):
    t = [make_track(state=S.REGIONS_DETECTED.value) for _ in range(3)]
    session.commit()
    stages = _fake_pipeline()
    d = _dispatcher(mode, settings, session, session_factory, stages)

    d.run(skip={"embed"}, track_id=[t[0].id, t[1].id])

    assert _states(session, *t) == ["complete", "complete", "regions_detected"]
    assert stages[-1].calls == []  # skipped, never processed


def test_dispatcher_empty_scope_and_single_id_still_work(
    settings, session, session_factory, make_track
):
    t = [make_track(state=S.PENDING.value) for _ in range(2)]
    session.commit()
    d = _dispatcher("sequential", settings, session, session_factory, _fake_pipeline())

    d.run(track_id=frozenset())
    assert _states(session, *t) == ["pending", "pending"]

    d.run(track_id=t[1].id)  # the original single-track form
    assert _states(session, *t) == ["pending", "complete"]


# ---------------------------------------------------------------------------
# run_until_settled: the review's reproduction
# ---------------------------------------------------------------------------


def test_skip_embeddings_does_not_complete_unselected_tracks(
    settings, session, session_factory, make_track, fast_idle, tmp_path
):
    """The review's exact scenario, through the REAL default stages and the
    parallel dispatcher the CLI uses: selected tracks 1-2 and unselected
    track 3 all wait in ``regions_detected``; unselected track 4 is
    ``pending`` with no audio on disk (an analyze claim would error it).
    Before the fix: ``[complete, complete, complete, error]``."""
    selected = [make_track(state=S.REGIONS_DETECTED.value) for _ in range(2)]
    parked = make_track(state=S.REGIONS_DETECTED.value)
    waiting = make_track(state=S.PENDING.value, file_path=str(tmp_path / "missing.mp3"))
    session.commit()
    d = Dispatcher(settings, session_factory=session_factory)

    totals = run_until_settled(d, [t.id for t in selected], skip={"embed"})

    assert _states(session, *selected, parked, waiting) == [
        "complete",
        "complete",
        "regions_detected",
        "pending",
    ]
    assert session.get(Track, waiting.id).error_message is None
    assert totals["embed"]["skipped"] == 2
    assert all(c["processed"] == 0 and c["errors"] == 0 for c in totals.values()), totals


def test_skipped_stage_reached_mid_run_is_advanced_for_the_selection_only(
    settings, session, session_factory, make_track, fast_idle
):
    """Selected tracks only reach ``regions_detected`` during the pass (after
    the dispatcher's up-front auto-advance), so ``run_until_settled`` advances
    them itself — still never the unselected ``regions_detected`` track."""
    selected = [make_track(state=S.PENDING.value) for _ in range(2)]
    parked = make_track(state=S.REGIONS_DETECTED.value)
    waiting = make_track(state=S.PENDING.value)
    session.commit()
    stages = _fake_pipeline()
    d = _dispatcher("parallel", settings, session, session_factory, stages)

    totals = run_until_settled(d, [t.id for t in selected], skip={"embed"})

    assert _states(session, *selected, parked, waiting) == [
        "complete",
        "complete",
        "regions_detected",
        "pending",
    ]
    assert totals["embed"]["skipped"] == 2
    for stage in stages[:-1]:
        assert sorted(stage.calls) == [t.id for t in selected], stage.name
    assert stages[-1].calls == []


def test_run_until_settled_with_nothing_selected_does_nothing(
    settings, session, session_factory, make_track
):
    waiting = make_track(state=S.PENDING.value)
    session.commit()
    stages = _fake_pipeline()
    d = _dispatcher("parallel", settings, session, session_factory, stages)

    assert run_until_settled(d, []) == {}
    assert _states(session, waiting) == ["pending"]
    assert all(stage.calls == [] for stage in stages)


def test_advance_skipped_is_scoped_and_chains(make_track, session):
    t = [make_track(state=S.STEMS_ANALYZED.value) for _ in range(2)]
    session.commit()
    stages = _fake_pipeline()

    moved = advance_skipped(session, stages, {"detect_regions", "embed"}, [t[0].id])

    assert moved == {"detect_regions": 1, "embed": 1}
    assert _states(session, *t) == ["complete", "stems_analyzed"]
    assert advance_skipped(session, stages, set(), [t[1].id]) == {}


# ---------------------------------------------------------------------------
# --from separate says when it won't re-run Demucs
# ---------------------------------------------------------------------------


def _give_stems(session, track: Track, root: Path, *, on_disk: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for kind in StemKind:
        path = root / f"{kind.value}.wav"
        if on_disk:
            path.write_bytes(b"RIFF")
        session.add(StemFile(track_id=track.id, kind=kind.value, path=str(path)))
    session.flush()


def test_plan_from_separate_notes_tracks_whose_stems_are_kept(make_track, session, tmp_path):
    from dance.pipeline.stages.analyze_stems import StemAnalysisStage
    from dance.pipeline.stages.separate import StemSeparationStage

    kept = make_track(state=S.COMPLETE.value)
    gone = make_track(state=S.COMPLETE.value)
    _give_stems(session, kept, tmp_path / "kept", on_disk=True)
    _give_stems(session, gone, tmp_path / "gone", on_disk=False)
    session.commit()

    assert tracks_with_stems_on_disk(session, [kept.id, gone.id]) == [kept.id]

    plan = plan_reprocess(session, StemSeparationStage())
    assert plan.track_ids == [kept.id, gone.id]
    assert len(plan.notes) == 1
    assert "1 of 2 selected track(s)" in plan.notes[0]
    assert "not re-run Demucs" in plan.notes[0]

    assert plan_reprocess(session, StemAnalysisStage()).notes == []


# ---------------------------------------------------------------------------
# CLI, end to end on a temp DB
# ---------------------------------------------------------------------------


def _cli_db():
    settings = get_settings()  # the conftest autouse fixture points this at tmp
    settings.ensure_directories()
    init_db(settings.db_url)
    return settings, get_session(settings.db_url)


def _add_track(session, n: int, path: Path, state: str) -> Track:
    track = Track(
        file_hash=f"{n:064d}",
        file_path=str(path),
        file_name=path.name,
        file_size_bytes=1024,
        title=f"Track {n}",
        artist="Artist",
        state=state,
        created_at=now_utc(),
        updated_at=now_utc(),
    )
    session.add(track)
    session.flush()
    return track


def test_cli_reprocess_track_ids_leave_every_other_track_alone(tmp_path, monkeypatch, fast_idle):
    """``dance reprocess --from embed -t A -t B --skip-embeddings`` with the
    real CLI and default stages: A and B go back to ``complete``; an
    unselected ``regions_detected`` track and an unselected ``pending`` one
    (audio missing) are untouched."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    settings, session = _cli_db()
    try:
        a = _add_track(session, 1, tmp_path / "a.wav", S.COMPLETE.value)
        b = _add_track(session, 2, tmp_path / "b.wav", S.COMPLETE.value)
        parked = _add_track(session, 3, tmp_path / "c.wav", S.REGIONS_DETECTED.value)
        waiting = _add_track(session, 4, tmp_path / "missing.wav", S.PENDING.value)
        session.commit()
        ids = [a.id, b.id, parked.id, waiting.id]
    finally:
        session.close()

    result = CliRunner().invoke(
        cli.main,
        ["reprocess", "--from", "embed", "-t", str(ids[0]), "-t", str(ids[1]), "--skip-embeddings"],
    )
    assert result.exit_code == 0, result.output
    assert "Rewound 2 track(s) to regions_detected" in result.output

    session = get_session(settings.db_url)
    try:
        got = {t.id: (t.state, t.error_message) for t in session.query(Track)}
    finally:
        session.close()
    assert got == {
        ids[0]: ("complete", None),
        ids[1]: ("complete", None),
        ids[2]: ("regions_detected", None),
        ids[3]: ("pending", None),
    }


def test_cli_dry_run_from_separate_prints_the_demucs_note(tmp_path):
    settings, session = _cli_db()
    try:
        track = _add_track(session, 1, tmp_path / "a.wav", S.COMPLETE.value)
        _give_stems(session, track, tmp_path / "stems", on_disk=True)
        session.commit()
    finally:
        session.close()

    result = CliRunner().invoke(cli.main, ["reprocess", "--from", "separate", "--all", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Note:" in result.output
    assert "not re-run Demucs" in " ".join(result.output.split())
