"""
Cross-process dispatcher lock.

Only one dispatcher may drain the pipeline at a time, across every process
that touches the DB: a terminal ``dance process`` / ``dance run`` /
``dance ingest``, and the API's ``POST /pipeline/process`` and watch loop
(which run ``dance process`` as a subprocess). Two concurrent runs are worse
than wasted CPU. Every parallel ``Dispatcher.run`` starts by healing each
track it finds in a ``*-ing`` state, and with a second run live those are
that run's in-flight tracks. They get reset, claimed again, processed
twice, and the heal ledger can promote healthy tracks to ERROR ("repeated
native crash").

Mechanism: an exclusive, non-blocking ``fcntl.flock`` on
``<data_dir>/dispatcher.lock``. The kernel drops a flock when the last
descriptor for it closes, including when the holder dies (SIGKILL, SIGBUS
/ exit 138), so a leftover lock file never blocks anything. Its contents
(pid, start time, command) are diagnostics for the "busy" message only.

Handing the lock to a child: the API takes the lock in its own process, so
its 409 and the watch loop's skip can't race another run. It then runs
``dance process`` as a subprocess with the descriptor inherited
(``pass_fds``) and its number in ``DANCE_DISPATCHER_LOCK_FD``. Parent and
child then share one open file description, which means one lock. The child
adopts it instead of contending with its own parent, and the lock stays
held between the API's dispatcher passes.

flock is per open file description, not per process: a second ``open()``
+ ``flock`` in the *same* process conflicts too. So a thread in the API
process can't sneak past a lock another thread holds. The CLI, which takes
the lock for a whole command and then calls ``Dispatcher.run`` (which
takes it again), goes through :func:`dispatcher_lock`, which is re-entrant
per thread.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LOCK_FILENAME = "dispatcher.lock"

# Env var naming an inherited, already-locked descriptor (see module doc).
INHERITED_FD_ENV = "DANCE_DISPATCHER_LOCK_FD"

# Exit status for "another pipeline run holds the lock". EX_TEMPFAIL from
# sysexits.h: a temporary condition, so retrying later is the right move.
EXIT_BUSY = 75


@dataclass(frozen=True)
class LockHolder:
    """Who holds the lock, as recorded in the lock file. Every field may be
    ``None``: the file can be empty (the holder hasn't written yet) or
    unreadable."""

    pid: int | None = None
    started_at: str | None = None
    command: str | None = None

    @property
    def alive(self) -> bool | None:
        """Whether ``pid`` is a running process (``None`` if unknown)."""
        if self.pid is None:
            return None
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, owned by someone else
        except OSError:
            return None
        return True

    def describe(self) -> str:
        if self.pid is None:
            return "holder unknown"
        parts = [f"pid {self.pid}"]
        if self.command:
            parts.append(f"`{self.command}`")
        if self.started_at:
            parts.append(f"started {self.started_at}")
        if self.alive is False:
            # The lock is held, so someone has it: e.g. the API died and the
            # `dance process` child it handed the lock to is still running.
            parts.append("no longer running; a process it started still holds the lock")
        return ", ".join(parts)


class DispatcherBusyError(RuntimeError):
    """Another dispatcher holds the lock. Nothing was touched."""

    def __init__(self, lock_path: Path, holder: LockHolder) -> None:
        self.lock_path = lock_path
        self.holder = holder
        super().__init__(
            f"Another pipeline run is already in progress ({holder.describe()}). "
            "Only one dispatcher may run at a time, so wait for it to finish "
            f"(lock file: {lock_path})."
        )


def lock_path_for(data_dir: Path | str) -> Path:
    return Path(data_dir) / LOCK_FILENAME


def read_holder(path: Path) -> LockHolder:
    """Best-effort parse of the lock file's diagnostics."""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return LockHolder()
    if not raw:
        return LockHolder()
    try:
        data = json.loads(raw)
    except ValueError:
        return LockHolder()
    if not isinstance(data, dict):
        return LockHolder()
    pid = data.get("pid")
    started_at = data.get("started_at")
    command = data.get("command")
    return LockHolder(
        pid=pid if isinstance(pid, int) else None,
        started_at=started_at if isinstance(started_at, str) else None,
        command=command if isinstance(command, str) else None,
    )


def _default_command() -> str:
    argv = list(sys.argv) or ["?"]
    head = Path(argv[0]).name
    # ``python -m dance.cli ...`` shows up as ``.../dance/cli.py``.
    if head in ("cli.py", "__main__.py"):
        head = "dance"
    return " ".join([head, *argv[1:]])[:200]


class DispatcherLock:
    """One handle on ``<data_dir>/dispatcher.lock``.

    Not re-entrant on its own: acquiring while another handle (in this
    process or any other) holds the lock fails. Use :func:`dispatcher_lock`
    for the re-entrant, scoped form.
    """

    def __init__(self, data_dir: Path | str, *, command: str | None = None) -> None:
        self.path = lock_path_for(data_dir)
        self.command = command or _default_command()
        self._fd: int | None = None
        self._inherited = False

    # -- state -------------------------------------------------------------

    @property
    def held(self) -> bool:
        return self._fd is not None

    @property
    def inherited(self) -> bool:
        """True when this handle adopted a descriptor a parent passed down."""
        return self._inherited

    def fileno(self) -> int:
        if self._fd is None:
            raise RuntimeError("dispatcher lock is not held")
        return self._fd

    def holder(self) -> LockHolder:
        return read_holder(self.path)

    # -- acquire / release -------------------------------------------------

    def try_acquire(self) -> bool:
        """One non-blocking attempt. True if this handle now holds the lock."""
        if self._fd is not None:
            return True
        fd = self._adopt_inherited()
        if fd is not None:
            self._fd = fd
            self._inherited = True
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._write_holder(fd)
        return True

    def acquire(
        self,
        *,
        wait: bool = False,
        poll_s: float = 1.0,
        timeout_s: float | None = None,
        on_wait: Callable[[LockHolder], None] | None = None,
    ) -> DispatcherLock:
        """Take the lock or raise :class:`DispatcherBusyError`.

        With ``wait=True``, poll every ``poll_s`` until the holder lets go
        (or ``timeout_s`` passes). ``on_wait`` is called with the current
        holder when waiting starts and whenever the holder changes.
        """
        if self.try_acquire():
            return self
        if not wait:
            raise DispatcherBusyError(self.path, self.holder())
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        announced: LockHolder | None = None
        while True:
            holder = self.holder()
            if on_wait is not None and holder != announced:
                on_wait(holder)
                announced = holder
            if deadline is not None and time.monotonic() >= deadline:
                raise DispatcherBusyError(self.path, holder)
            time.sleep(poll_s)
            if self.try_acquire():
                return self

    def release(self) -> None:
        """Let go. Idempotent."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        if self._inherited:
            # The parent that handed this descriptor down shares the same
            # open file description, so LOCK_UN would drop *its* lock too.
            # Leave it: the lock ends when the last holder closes it.
            self._inherited = False
            return
        try:
            # Clear the diagnostics while still holding the lock, so the
            # file only names a past holder if that holder died.
            os.ftruncate(fd, 0)
        except OSError:
            pass
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> DispatcherLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()

    # -- internals ---------------------------------------------------------

    def _adopt_inherited(self) -> int | None:
        """Adopt a descriptor a parent passed via ``INHERITED_FD_ENV``.

        Only if it really is our lock file (same device + inode) and
        flocking it succeeds, which it does when the parent's lock is on the
        same open file description. Anything else falls back to a normal
        acquire.
        """
        raw = os.environ.get(INHERITED_FD_ENV)
        if not raw:
            return None
        try:
            fd = int(raw)
            st = os.fstat(fd)
            ref = os.stat(self.path)
        except (ValueError, OSError):
            return None
        if (st.st_dev, st.st_ino) != (ref.st_dev, ref.st_ino):
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return None
        # Our own grandchildren shouldn't inherit it by accident.
        os.set_inheritable(fd, False)
        return fd

    def _write_holder(self, fd: int) -> None:
        info = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": self.command,
        }
        data = (json.dumps(info) + "\n").encode("utf-8")
        try:
            os.ftruncate(fd, 0)
            os.pwrite(fd, data, 0)
        except OSError:
            pass  # diagnostics only


_tls = threading.local()


@contextmanager
def dispatcher_lock(
    data_dir: Path | str,
    *,
    wait: bool = False,
    poll_s: float = 1.0,
    on_wait: Callable[[LockHolder], None] | None = None,
    command: str | None = None,
) -> Iterator[DispatcherLock]:
    """Hold the dispatcher lock for the duration of the block.

    Re-entrant per thread: a CLI command that already holds the lock can
    call ``Dispatcher.run``, which takes it again. Other threads, and other
    processes, get :class:`DispatcherBusyError` (or wait, with ``wait=True``).
    """
    key = os.path.abspath(lock_path_for(data_dir))
    held: dict[str, DispatcherLock] = _tls.__dict__.setdefault("held", {})
    if key in held:
        yield held[key]
        return
    lock = DispatcherLock(data_dir, command=command)
    lock.acquire(wait=wait, poll_s=poll_s, on_wait=on_wait)
    held[key] = lock
    try:
        yield lock
    finally:
        held.pop(key, None)
        lock.release()
