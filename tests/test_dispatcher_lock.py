"""Cross-process dispatcher lock, and placeholder rows never being claimed.

The lock tests use a REAL second process (``subprocess`` with ``PYTHONPATH``
pointed at this checkout's ``src``) to hold or contend for
``<data_dir>/dispatcher.lock``. flock semantics between processes are the
thing under test, so threads in one process wouldn't prove anything.

Every child gets ``HOME`` pointed at a tmp dir so its ``Settings`` can never
read the real ``~/.dance/.env``.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from dance.config import Settings
from dance.core.database import Track, TrackState
from dance.pipeline import dispatcher as disp_mod
from dance.pipeline.dispatcher import Dispatcher
from dance.pipeline.dispatcher_lock import (
    EXIT_BUSY,
    INHERITED_FD_ENV,
    DispatcherBusyError,
    DispatcherLock,
    dispatcher_lock,
    lock_path_for,
    read_holder,
)
from dance.pipeline.heal_ledger import HealLedger

SRC = Path(__file__).resolve().parents[1] / "src"
PLACEHOLDER = "pending:spotify123"


# ---------------------------------------------------------------------------
# Real second processes
# ---------------------------------------------------------------------------


def _child_env(home: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != INHERITED_FD_ENV}
    env["PYTHONPATH"] = str(SRC)
    env["HOME"] = str(home)
    env.update(extra)
    return env


class _Child:
    """A python subprocess whose stdout lines we can await with a timeout."""

    def __init__(self, script: str, args: list[str], env: dict[str, str], **popen_kw) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(script), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            **popen_kw,
        )
        self._lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line.rstrip("\n"))

    @property
    def pid(self) -> int:
        return self.proc.pid

    def expect(self, prefix: str, timeout: float = 30.0) -> str:
        deadline = time.monotonic() + timeout
        seen: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=0.1)
            except queue.Empty:
                if self.proc.poll() is not None and self._lines.empty():
                    break
                continue
            seen.append(line)
            if line.startswith(prefix):
                return line
        raise AssertionError(f"child never printed {prefix!r}; saw: {seen}")

    def send(self, text: str = "") -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def finish(self, timeout: float = 30.0) -> int:
        return self.proc.wait(timeout=timeout)

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)


# Holds the lock until a line arrives on stdin.
_HOLDER = """
    import sys
    from dance.pipeline.dispatcher_lock import DispatcherLock

    lock = DispatcherLock(sys.argv[1], command="test-holder")
    lock.acquire()
    print("LOCKED", flush=True)
    sys.stdin.readline()
    lock.release()
    print("RELEASED", flush=True)
"""

# One acquire attempt; reports the outcome.
_PROBE = """
    import os, sys
    from dance.pipeline.dispatcher_lock import DispatcherBusyError, DispatcherLock

    lock = DispatcherLock(sys.argv[1], command="test-probe")
    try:
        lock.acquire()
    except DispatcherBusyError as exc:
        print(f"BUSY {exc.holder.pid}", flush=True)
    else:
        print(f"ACQUIRED inherited={lock.inherited}", flush=True)
"""

# A real parallel Dispatcher whose stage blocks mid-track until a gate file
# appears, so the track sits in its inflight state owned by a live run.
_GATED_DISPATCHER = """
    import json, os, sys, time
    from pathlib import Path

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from dance.config import Settings
    from dance.core.database import TrackState
    from dance.pipeline import dispatcher as disp_mod
    from dance.pipeline.dispatcher import Dispatcher

    db_url, data_dir, gate, track_id = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
    disp_mod._IDLE_GRACE_S = 0.3
    disp_mod._POLL_INTERVAL_S = 0.05
    Dispatcher._register_default_stages = lambda self: None  # skip heavy imports

    class GatedStage:
        name = "analyze"
        input_state = TrackState.PENDING
        output_state = TrackState.ANALYZED
        inflight_state = TrackState.ANALYZING
        error_state = TrackState.ERROR
        concurrency = 1
        uses_gpu = False

        def process(self, session, track, settings):
            print(f"CLAIMED {track.id}", flush=True)
            deadline = time.monotonic() + 60
            while not gate.exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("gate never opened")
                time.sleep(0.05)
            track.state = self.output_state.value
            session.commit()
            return True

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    settings = Settings(data_dir=Path(data_dir), library_dir=Path(data_dir) / "lib",
                        stems_dir=Path(data_dir) / "stems")
    d = Dispatcher(settings, session_factory=sessionmaker(bind=engine))
    d.register(GatedStage())
    print("DONE " + json.dumps(d.run(track_id=track_id)), flush=True)
"""


@pytest.fixture
def child_home(tmp_path) -> Path:
    home = tmp_path / "child_home"
    home.mkdir()
    return home


@pytest.fixture
def spawn(child_home):
    """Start real child processes; anything still alive is killed at teardown."""
    children: list[_Child] = []

    def _spawn(script: str, *args: str, **kw) -> _Child:
        env = kw.pop("env", None) or _child_env(child_home)
        child = _Child(script, [str(a) for a in args], env, **kw)
        children.append(child)
        return child

    yield _spawn
    for c in children:
        c.kill()


@pytest.fixture
def hold_lock(spawn):
    """Start a real process that holds ``data_dir``'s lock until released."""

    def _hold(data_dir: Path) -> _Child:
        holder = spawn(_HOLDER, data_dir)
        holder.expect("LOCKED")
        return holder

    return _hold


def _release(holder: _Child) -> None:
    holder.send()
    holder.expect("RELEASED")
    assert holder.finish() == 0


# ---------------------------------------------------------------------------
# In-process stages + fixtures
# ---------------------------------------------------------------------------


class _Stage:
    name = "analyze"
    input_state = TrackState.PENDING
    output_state = TrackState.ANALYZED
    inflight_state = TrackState.ANALYZING
    error_state = TrackState.ERROR
    concurrency = 1
    uses_gpu = False

    def __init__(self) -> None:
        self.calls: list[int] = []

    def process(self, session, track, settings) -> bool:
        self.calls.append(int(track.id))
        track.state = self.output_state.value
        session.commit()
        return True


@pytest.fixture
def settings(tmp_path, db_url) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=db_url,
    )


@pytest.fixture
def fast_idle_grace(monkeypatch):
    monkeypatch.setattr(disp_mod, "_IDLE_GRACE_S", 0.3)
    monkeypatch.setattr(disp_mod, "_POLL_INTERVAL_S", 0.05)


def _dispatcher(settings, *, session=None, session_factory=None) -> tuple[Dispatcher, _Stage]:
    if session is not None:
        d = Dispatcher(settings, session)
    else:
        d = Dispatcher(settings, session_factory=session_factory)
    d.clear()
    stage = _Stage()
    d.register(stage)
    return d, stage


def _state(session, track_id: int) -> str:
    session.expire_all()
    return session.get(Track, track_id).state


# ---------------------------------------------------------------------------
# Dispatcher.run vs a lock held by another process
# ---------------------------------------------------------------------------


def test_run_refuses_while_other_process_holds_lock_and_leaves_its_tracks_alone(
    settings, session_factory, session, make_track, fast_idle_grace, hold_lock
):
    # The other run's in-flight track, plus ordinary pending work.
    theirs = make_track(state=TrackState.ANALYZING.value)
    pending = make_track(state=TrackState.PENDING.value)
    session.commit()

    holder = hold_lock(settings.data_dir)
    d, stage = _dispatcher(settings, session_factory=session_factory)

    with pytest.raises(DispatcherBusyError) as excinfo:
        d.run()

    assert excinfo.value.holder.pid == holder.pid
    assert f"pid {holder.pid}" in str(excinfo.value)
    assert "test-holder" in str(excinfo.value)
    # Nothing was healed, claimed, or counted.
    assert _state(session, theirs.id) == TrackState.ANALYZING.value
    assert _state(session, pending.id) == TrackState.PENDING.value
    assert stage.calls == []
    assert HealLedger(settings.data_dir).count_for(theirs.id, "analyzing") == 0

    # Holder gone: the next run proceeds. ``theirs`` is now a genuine orphan,
    # so healing it and processing it is exactly right.
    _release(holder)
    result = d.run()
    assert result["analyze"]["processed"] == 2
    assert sorted(stage.calls) == sorted([theirs.id, pending.id])
    assert _state(session, theirs.id) == TrackState.ANALYZED.value


def test_sequential_run_also_refuses_while_locked(settings, session, make_track, hold_lock):
    t = make_track()
    session.commit()
    holder = hold_lock(settings.data_dir)
    d, stage = _dispatcher(settings, session=session)

    with pytest.raises(DispatcherBusyError):
        d.run()
    assert stage.calls == []

    _release(holder)
    assert d.run()["analyze"]["processed"] == 1
    assert stage.calls == [t.id]


def test_live_dispatcher_in_another_process_is_not_double_processed(
    settings, db_url, session_factory, session, make_track, fast_idle_grace, spawn, tmp_path
):
    """The reported bug, end to end: a real parallel Dispatcher in another
    process has a track mid-stage; starting a run here must neither reset
    that track nor process it a second time."""
    theirs = make_track(state=TrackState.PENDING.value)
    ours = make_track(state=TrackState.PENDING.value)
    session.commit()
    gate = tmp_path / "gate"

    other = spawn(_GATED_DISPATCHER, db_url, settings.data_dir, gate, theirs.id)
    other.expect(f"CLAIMED {theirs.id}")
    assert _state(session, theirs.id) == TrackState.ANALYZING.value

    d, stage = _dispatcher(settings, session_factory=session_factory)
    with pytest.raises(DispatcherBusyError) as excinfo:
        d.run()
    assert excinfo.value.holder.pid == other.pid
    assert _state(session, theirs.id) == TrackState.ANALYZING.value  # not reset
    assert HealLedger(settings.data_dir).count_for(theirs.id, "analyzing") == 0
    session.expire_all()
    assert session.get(Track, theirs.id).error_message is None

    gate.write_text("go")
    done = other.expect("DONE")
    assert json.loads(done[len("DONE ") :])["analyze"]["processed"] == 1
    assert other.finish() == 0

    result = d.run()
    assert stage.calls == [ours.id]  # theirs was processed exactly once, over there
    assert result["analyze"]["processed"] == 1
    assert _state(session, theirs.id) == TrackState.ANALYZED.value
    assert _state(session, ours.id) == TrackState.ANALYZED.value


# ---------------------------------------------------------------------------
# Stale locks
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def test_leftover_lock_file_with_dead_pid_does_not_block(settings, session, make_track):
    dead = _dead_pid()
    path = lock_path_for(settings.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": dead, "started_at": "x", "command": "dance process"}))
    assert read_holder(path).pid == dead
    assert read_holder(path).alive is False

    t = make_track()
    session.commit()
    d, stage = _dispatcher(settings, session=session)
    assert d.run()["analyze"]["processed"] == 1
    assert stage.calls == [t.id]


def test_holder_killed_with_sigkill_releases_lock(settings, session, make_track, hold_lock):
    """flock dies with its process, even when nothing gets to clean up."""
    holder = hold_lock(settings.data_dir)
    holder.kill()  # SIGKILL: no finally, no release()
    # The file still names the dead holder, and that must not matter.
    assert read_holder(lock_path_for(settings.data_dir)).pid == holder.pid

    t = make_track()
    session.commit()
    d, stage = _dispatcher(settings, session=session)
    assert d.run()["analyze"]["processed"] == 1
    assert stage.calls == [t.id]


def test_clean_release_clears_holder_diagnostics(settings):
    with dispatcher_lock(settings.data_dir, command="dance process") as lock:
        holder = read_holder(lock.path)
        assert holder.pid == os.getpid()
        assert holder.command == "dance process"
        assert holder.started_at
    assert read_holder(lock_path_for(settings.data_dir)).pid is None


# ---------------------------------------------------------------------------
# Lock mechanics: re-entrancy, threads, wait, hand-off to a child
# ---------------------------------------------------------------------------


def test_reentrant_in_same_thread_but_not_across_threads(settings):
    errors: list[BaseException] = []

    def other_thread() -> None:
        try:
            with dispatcher_lock(settings.data_dir):
                pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with dispatcher_lock(settings.data_dir) as outer:
        with dispatcher_lock(settings.data_dir) as inner:
            assert inner is outer
        assert outer.held  # inner exit didn't release
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
    assert len(errors) == 1 and isinstance(errors[0], DispatcherBusyError)
    assert not outer.held


def test_wait_blocks_until_holder_exits(settings, hold_lock):
    holder = hold_lock(settings.data_dir)
    seen: list[int | None] = []
    threading.Timer(0.6, holder.send).start()

    lock = DispatcherLock(settings.data_dir)
    t0 = time.monotonic()
    lock.acquire(wait=True, poll_s=0.05, on_wait=lambda h: seen.append(h.pid))
    try:
        assert time.monotonic() - t0 >= 0.5
        assert seen and seen[0] == holder.pid
    finally:
        lock.release()


def test_wait_timeout_raises_busy(settings, hold_lock):
    holder = hold_lock(settings.data_dir)
    with pytest.raises(DispatcherBusyError) as excinfo:
        DispatcherLock(settings.data_dir).acquire(wait=True, poll_s=0.05, timeout_s=0.3)
    assert excinfo.value.holder.pid == holder.pid


def test_child_adopts_inherited_lock_and_strangers_stay_out(settings, spawn, child_home):
    parent = DispatcherLock(settings.data_dir, command="test-parent").acquire()
    try:
        fd = parent.fileno()
        inherit_env = _child_env(child_home, **{INHERITED_FD_ENV: str(fd)})
        child = spawn(_PROBE, settings.data_dir, env=inherit_env, pass_fds=(fd,))
        assert child.expect("ACQUIRED") == "ACQUIRED inherited=True"
        assert child.finish() == 0

        # A process without the descriptor is refused, and the child's exit
        # didn't drop the parent's lock.
        stranger = spawn(_PROBE, settings.data_dir)
        assert stranger.expect("BUSY") == f"BUSY {os.getpid()}"

        # A forged env var without the real descriptor doesn't get in either.
        forged = spawn(
            _PROBE, settings.data_dir, env=_child_env(child_home, **{INHERITED_FD_ENV: str(fd)})
        )
        assert forged.expect("BUSY") == f"BUSY {os.getpid()}"
    finally:
        parent.release()

    after = spawn(_PROBE, settings.data_dir)
    assert after.expect("ACQUIRED") == "ACQUIRED inherited=False"


# ---------------------------------------------------------------------------
# Placeholder rows (Spotify download still in flight) are never claimed
# ---------------------------------------------------------------------------


def test_parallel_never_claims_placeholder_rows(
    settings, session_factory, session, make_track, fast_idle_grace
):
    downloading = make_track(file_hash=PLACEHOLDER)
    ready = make_track()
    session.commit()

    d, stage = _dispatcher(settings, session_factory=session_factory)
    result = d.run()

    assert stage.calls == [ready.id]
    assert result["analyze"] == {"processed": 1, "errors": 0, "skipped": 0}
    assert _state(session, downloading.id) == TrackState.PENDING.value
    session.expire_all()
    assert session.get(Track, downloading.id).error_message is None

    # Once the download lands (real hash replaces the placeholder) it's claimable.
    session.get(Track, downloading.id).file_hash = "f" * 64
    session.commit()
    d.run()
    assert stage.calls == [ready.id, downloading.id]


def test_sequential_and_skip_never_touch_placeholder_rows(settings, session, make_track):
    downloading = make_track(file_hash=PLACEHOLDER)
    ready = make_track()
    session.commit()

    d, stage = _dispatcher(settings, session=session)
    d.run(skip={"analyze"})  # auto-advance path
    assert _state(session, ready.id) == TrackState.ANALYZED.value
    assert _state(session, downloading.id) == TrackState.PENDING.value

    d.run()  # claim path
    assert stage.calls == []
    assert _state(session, downloading.id) == TrackState.PENDING.value


# ---------------------------------------------------------------------------
# API: POST /pipeline/process and the watch loop
# ---------------------------------------------------------------------------


def pipeline_mod():
    from dance.api.routers import pipeline

    return pipeline


@pytest.fixture
def app(settings, session_factory):
    from dance.api import create_app
    from tests.test_api import FakeAbletonBridge

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return create_app(
        settings=settings, bridge=FakeAbletonBridge(), session_factory=session_factory
    )


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def _wait_job(client, job_id: str) -> dict:
    body: dict = {}
    for _ in range(200):
        body = client.get(f"/api/v1/pipeline/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished: {body}")


def test_api_process_returns_409_naming_the_other_process(client, settings, hold_lock, monkeypatch):
    from dance.api.jobs import get_job_registry

    ran: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **kw: ran.append(cmd))

    holder = hold_lock(settings.data_dir)
    r = client.post("/api/v1/pipeline/process")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert f"pid {holder.pid}" in detail
    assert "already in progress" in detail
    assert get_job_registry().active(kind="pipeline_run") == []
    assert ran == []

    _release(holder)
    # Free again: the run is accepted (subprocess stubbed out).
    monkeypatch.setattr(pipeline_mod(), "_track_state_signature", lambda _: "stable")

    class _Proc:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **kw: _Proc())
    r = client.post("/api/v1/pipeline/process")
    assert r.status_code == 200
    assert _wait_job(client, r.json()["id"])["status"] == "done"


def test_api_run_hands_its_lock_to_the_dance_process_child(
    client, settings, spawn, child_home, monkeypatch
):
    """``POST /process`` holds the lock itself and its ``dance process``
    child inherits it: the child gets in, a terminal run doesn't, and the
    lock is free again once the job ends."""
    real_run = subprocess.run
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(pipeline_mod(), "_track_state_signature", lambda _: "stable")

    class _Proc:
        returncode = 0

    def fake_run(cmd, *args, **kwargs):
        if "process" not in cmd:
            return _Proc()
        env = dict(kwargs["env"], PYTHONPATH=str(SRC), HOME=str(child_home))
        assert kwargs["pass_fds"] == (int(env[INHERITED_FD_ENV]),)
        assert env["DANCE_DATA_DIR"] == str(settings.data_dir)
        # Stand-in for ``dance process``: take the lock the same way it would.
        child = real_run(
            [sys.executable, "-c", textwrap.dedent(_PROBE), str(settings.data_dir)],
            env=env,
            pass_fds=kwargs["pass_fds"],
            capture_output=True,
            text=True,
        )
        seen.append(("child", child.stdout.strip()))
        terminal = real_run(
            [sys.executable, "-c", textwrap.dedent(_PROBE), str(settings.data_dir)],
            env=_child_env(child_home),
            capture_output=True,
            text=True,
        )
        seen.append(("terminal", terminal.stdout.strip()))
        return _Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = client.post("/api/v1/pipeline/process")
    assert r.status_code == 200
    assert _wait_job(client, r.json()["id"])["status"] == "done"

    # Two passes (signature stable on the 2nd); the lock was held across both.
    assert (
        seen
        == [
            ("child", "ACQUIRED inherited=True"),
            ("terminal", f"BUSY {os.getpid()}"),
        ]
        * 2
    )
    after = spawn(_PROBE, settings.data_dir)
    assert after.expect("ACQUIRED") == "ACQUIRED inherited=False"


def test_watch_tick_skips_while_another_process_holds_lock(
    settings, session, make_track, hold_lock, monkeypatch
):
    from dance.api import jobs as jobs_mod

    pm = pipeline_mod()
    monkeypatch.setattr(jobs_mod, "_REGISTRY", jobs_mod.JobRegistry())
    monkeypatch.setitem(pm._WATCH_STATE, "enabled", True)
    started: list[tuple[str, bool]] = []

    def fake_job(job_id, _settings, lock=None):
        started.append((job_id, bool(lock and lock.held)))
        if lock is not None:
            lock.release()

    monkeypatch.setattr(pm, "_run_dispatcher_job", fake_job)
    make_track()
    session.commit()

    holder = hold_lock(settings.data_dir)
    assert pm._watch_tick(settings) is None
    assert started == []
    assert jobs_mod.get_job_registry().active(kind="pipeline_run") == []

    _release(holder)
    job_id = pm._watch_tick(settings)
    assert job_id is not None
    for _ in range(100):
        if started:
            break
        time.sleep(0.02)
    assert started == [(job_id, True)]


def test_watch_tick_ignores_placeholder_only_backlog(settings, session, make_track, monkeypatch):
    from dance.api import jobs as jobs_mod

    pm = pipeline_mod()
    monkeypatch.setattr(jobs_mod, "_REGISTRY", jobs_mod.JobRegistry())
    monkeypatch.setitem(pm._WATCH_STATE, "enabled", True)
    monkeypatch.setattr(pm, "_run_dispatcher_job", lambda *a, **k: pytest.fail("started"))
    make_track(file_hash=PLACEHOLDER)
    session.commit()

    assert pm._pending_track_count(settings) == 0
    assert pm._watch_tick(settings) is None


# ---------------------------------------------------------------------------
# CLI: process / run / ingest
# ---------------------------------------------------------------------------


def _cli_env(tmp_path) -> dict[str, str]:
    return {
        "DANCE_DATA_DIR": str(tmp_path / "data"),
        "DANCE_LIBRARY_DIR": str(tmp_path / "lib"),
        "DANCE_STEMS_DIR": str(tmp_path / "stems"),
        "DANCE_ALS_OUTPUT_DIR": str(tmp_path / "als"),
    }


@pytest.fixture
def cli_stubs(monkeypatch):
    from dance import cli
    from dance.pipeline import preflight

    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    calls: list[str] = []
    monkeypatch.setattr(cli, "_run_sync", lambda *a, **k: calls.append("sync"))
    monkeypatch.setattr(cli, "_run_process", lambda *a, **k: calls.append("process") or {})
    monkeypatch.setattr(cli.build_graph, "callback", lambda *a, **k: calls.append("build-graph"))
    monkeypatch.setattr(cli.tag, "callback", lambda *a, **k: calls.append("tag"))
    monkeypatch.setattr(cli.export_als, "callback", lambda *a, **k: calls.append("export-als"))
    return cli, calls


@pytest.mark.parametrize(
    "args",
    [["process"], ["run", "--once", "--skip-sync"], ["ingest", "--skip-sync"]],
)
def test_cli_exits_75_naming_the_holder(args, tmp_path, cli_stubs, hold_lock):
    cli, calls = cli_stubs
    holder = hold_lock(tmp_path / "data")

    result = CliRunner().invoke(cli.main, args, env=_cli_env(tmp_path))

    assert result.exit_code == EXIT_BUSY, result.output
    assert f"pid {holder.pid}" in result.output
    assert "--wait" in result.output
    assert calls == []


def test_cli_process_wait_queues_behind_the_holder(tmp_path, cli_stubs, hold_lock):
    cli, calls = cli_stubs
    holder = hold_lock(tmp_path / "data")
    threading.Timer(1.5, holder.send).start()

    result = CliRunner().invoke(cli.main, ["process", "--wait"], env=_cli_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "Waiting for the running pipeline" in result.output
    assert calls == ["process"]


def test_cli_run_daemon_skips_a_busy_pass(tmp_path, cli_stubs, hold_lock, monkeypatch):
    cli, calls = cli_stubs
    hold_lock(tmp_path / "data")

    def stop(_seconds):
        raise KeyboardInterrupt

    # Swap the module's ``time`` (not ``time.sleep`` itself, which other
    # threads share) so the daemon loop ends after one pass.
    monkeypatch.setattr(cli, "time", SimpleNamespace(sleep=stop))
    result = CliRunner().invoke(cli.main, ["run", "--skip-sync"], env=_cli_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "Skipping this pass" in result.output
    assert calls == []


def test_real_dance_process_exits_75_while_locked(tmp_path, child_home, hold_lock):
    """The actual console entry point, in its own process."""
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("dance process preflights ffmpeg")
    holder = hold_lock(tmp_path / "data")
    env = _child_env(
        child_home, DANCE_DATABASE_URL=f"sqlite:///{tmp_path / 'cli.db'}", **_cli_env(tmp_path)
    )

    proc = subprocess.run(
        [sys.executable, "-m", "dance.cli", "process", "--skip-ingest"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    out = proc.stdout + proc.stderr
    assert proc.returncode == EXIT_BUSY, out
    assert f"pid {holder.pid}" in out
    assert "already in progress" in out
