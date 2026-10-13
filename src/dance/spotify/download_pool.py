"""One bounded worker pool shared by every ingest download path.

Before this, the three API ingest paths each had their own concurrency:
the CSV job ran a private 4-worker pool, ``/ingest/track`` spawned a thread,
and ``/ingest/playlist`` spawned **one thread per new track** — a 300-track
playlist launched 300 concurrent yt-dlp processes against YouTube. Two
overlapping jobs stacked on top of each other.

All three now submit to this pool, so the number of simultaneous downloads
is ``settings.download_workers`` (default 3) process-wide, however many jobs
are in flight. Work beyond that queues: interactive single-track adds
(Cmd-K, :meth:`DownloadPool.submit_interactive`) go ahead of bulk playlist /
CSV work, each class in submission order. Without that, a track added
live from the Booth waited behind every download of a 300-track playlist.

The workers are daemon threads (unlike ``ThreadPoolExecutor``'s, which the
interpreter joins at exit): the API used daemon threads before so that
stopping uvicorn never blocks on an in-flight download, and that is kept.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_DOWNLOAD_WORKERS = 3

#: Queue priorities — lower runs first. The pool size never changes; an
#: interactive job only jumps the *queue*, it never preempts a download.
PRIORITY_INTERACTIVE = 0
PRIORITY_BULK = 10
_PRIORITY_STOP = 100

_Work = tuple[Future[Any], Callable[..., Any], tuple, dict]


class DownloadPool:
    """Fixed-size daemon-thread pool returning ``concurrent.futures.Future``s."""

    def __init__(self, max_workers: int, name: str = "dance-download") -> None:
        self.max_workers = max(1, int(max_workers))
        # (priority, sequence, work): the sequence keeps FIFO order within a
        # priority and means two entries never compare their work items.
        self._queue: queue.PriorityQueue[tuple[int, int, _Work | None]] = queue.PriorityQueue()
        self._seq = itertools.count()
        self._threads = [
            threading.Thread(target=self._worker, name=f"{name}-{i}", daemon=True)
            for i in range(self.max_workers)
        ]
        for t in self._threads:
            t.start()

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        """Queue bulk work (playlist / CSV rows) in submission order."""
        return self._submit(PRIORITY_BULK, fn, args, kwargs)

    def submit_interactive(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        """Queue a user-initiated single download ahead of all bulk work."""
        return self._submit(PRIORITY_INTERACTIVE, fn, args, kwargs)

    def _submit(self, priority: int, fn: Callable[..., T], args: tuple, kwargs: dict) -> Future[T]:
        fut: Future[T] = Future()
        self._queue.put((priority, next(self._seq), (fut, fn, args, kwargs)))
        return fut

    def shutdown(self) -> None:
        """Stop the workers after the queued work drains (tests only)."""
        for _ in self._threads:
            self._queue.put((_PRIORITY_STOP, next(self._seq), None))

    def _worker(self) -> None:
        while True:
            _, _, item = self._queue.get()
            if item is None:
                return
            fut, fn, args, kwargs = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 — surfaced via the future
                fut.set_exception(exc)
            else:
                fut.set_result(result)


_POOL: DownloadPool | None = None
_POOL_LOCK = threading.Lock()


def get_download_pool(max_workers: int = DEFAULT_DOWNLOAD_WORKERS) -> DownloadPool:
    """Return the process-wide pool, creating it on first use.

    The size is fixed by the first caller (``settings.download_workers``);
    later calls with a different size get the existing pool rather than a
    second one, which would silently double the concurrency.
    """
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = DownloadPool(max_workers)
        elif _POOL.max_workers != max(1, int(max_workers)):
            logger.info(
                "download pool already running with %d workers; ignoring request for %d "
                "(restart the API to change DANCE_DOWNLOAD_WORKERS)",
                _POOL.max_workers,
                max_workers,
            )
        return _POOL


def _reset_download_pool_for_tests() -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown()
        _POOL = None


__all__ = [
    "DEFAULT_DOWNLOAD_WORKERS",
    "PRIORITY_BULK",
    "PRIORITY_INTERACTIVE",
    "DownloadPool",
    "get_download_pool",
]
