"""In-memory job tracker for background work kicked off by the API.

Used by the CSV ingest endpoint to run yt-dlp downloads off the request
thread while letting the UI poll (or subscribe via WebSocket) for status.

Deliberately minimal — no persistence, no cancellation, no priorities.
A job lives in the process that started it; if the API restarts, all jobs
are lost. This is fine for hobbyist ops where the user is in the room and
can re-trigger.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal


JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class JobItem:
    """One unit of work within a job."""

    label: str  # e.g. "Daft Punk - One More Time"
    status: Literal["pending", "ok", "skip", "fail"] = "pending"
    message: str = ""


@dataclass
class Job:
    id: str
    kind: str  # e.g. "csv_ingest"
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    items: list[JobItem] = field(default_factory=list)
    error: str | None = None
    # Bump on every mutation so WebSocket subscribers can detect change.
    revision: int = 0

    @property
    def counts(self) -> dict[str, int]:
        c = {"pending": 0, "ok": 0, "skip": 0, "fail": 0}
        for item in self.items:
            c[item.status] = c.get(item.status, 0) + 1
        return c

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "revision": self.revision,
            "counts": self.counts,
            "total": len(self.items),
            "items": [
                {"label": it.label, "status": it.status, "message": it.message}
                for it in self.items
            ],
        }


class JobRegistry:
    """Thread-safe in-memory job store + change broadcast."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[Job], None]] = []

    def create(self, kind: str, item_labels: list[str]) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            items=[JobItem(label=lbl) for lbl in item_labels],
        )
        with self._lock:
            self._jobs[job.id] = job
        self._broadcast(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            return sorted(
                self._jobs.values(), key=lambda j: j.created_at, reverse=True
            )[:limit]

    def update_item(
        self, job_id: str, index: int, status: str, message: str = ""
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or index >= len(job.items):
                return
            job.items[index].status = status  # type: ignore[assignment]
            job.items[index].message = message
            job.revision += 1
            snapshot = job
        self._broadcast(snapshot)

    def set_status(
        self,
        job_id: str,
        status: JobStatus,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = status
            job.error = error
            if status == "running" and job.started_at is None:
                job.started_at = time.time()
            if status in ("done", "error") and job.finished_at is None:
                job.finished_at = time.time()
            job.revision += 1
            snapshot = job
        self._broadcast(snapshot)

    def subscribe(self, callback: Callable[[Job], None]) -> Callable[[], None]:
        """Register a callback. Returns an ``unsubscribe()`` closure."""
        with self._lock:
            self._subscribers.append(callback)

        def _unsub() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return _unsub

    def _broadcast(self, job: Job) -> None:
        # Snapshot subscriber list under the lock, but call callbacks
        # outside it — callbacks may take time (WebSocket sends).
        with self._lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(job)
            except Exception:  # noqa: BLE001
                # Don't let one bad subscriber kill broadcast for others.
                pass


# Process-wide singleton. The API stashes it on app.state.job_registry too.
_REGISTRY = JobRegistry()


def get_job_registry() -> JobRegistry:
    return _REGISTRY


__all__ = ["Job", "JobItem", "JobRegistry", "JobStatus", "get_job_registry"]
