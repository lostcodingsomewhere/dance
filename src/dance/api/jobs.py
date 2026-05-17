"""Job tracker for background work kicked off by the API.

Used by:

- ``POST /pipeline/ingest/commit`` for yt-dlp download jobs (kind="csv_ingest").
- ``POST /pipeline/process`` for in-process dispatcher runs (kind="pipeline_run").

Each :class:`Job` owns a list of :class:`JobItem` units (one per CSV row,
one per pipeline stage, etc.). Updates flow through the registry under
a lock and are then broadcast to subscribers — the ``/ws/pipeline``
WebSocket pushes the full snapshot to UI clients on every mutation.

Persistence: jobs are serialized to ``<data_dir>/jobs.json`` after every
mutation (atomic write via tempfile rename). On startup the registry
loads the last 50 jobs. This is a JSON file rather than a SQLAlchemy
table on purpose: jobs are operational state, not analytical, and we
don't want to churn the schema or Alembic migrations for them.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal


logger = logging.getLogger(__name__)


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
    kind: str  # e.g. "csv_ingest" or "pipeline_run"
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

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        items = [JobItem(**it) for it in d.get("items", [])]
        # ``counts`` and ``total`` are derived properties — drop if persisted.
        return cls(
            id=d["id"],
            kind=d["kind"],
            status=d.get("status", "done"),
            created_at=float(d.get("created_at", time.time())),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            items=items,
            error=d.get("error"),
            revision=int(d.get("revision", 0)),
        )


# Cap on persisted history. Older jobs get dropped on each save to keep the
# file from growing unbounded.
_MAX_PERSISTED_JOBS = 50


class JobRegistry:
    """Thread-safe job store + change broadcast + optional disk persistence."""

    def __init__(self, persist_path: Path | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[Job], None]] = []
        self._persist_path = persist_path
        if persist_path is not None:
            self._load_from_disk()

    # ---- persistence ----

    def _load_from_disk(self) -> None:
        """Best-effort load. Missing or corrupt files are tolerated silently."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load jobs from %s: %s", self._persist_path, e)
            return
        if not isinstance(data, list):
            return
        for entry in data:
            try:
                job = Job.from_dict(entry)
                # Heal: any job left in ``running`` or ``queued`` from a previous
                # API process is dead. Move it to ``error`` so the UI can tell.
                if job.status in ("running", "queued"):
                    job.status = "error"
                    job.error = "API restarted while job was active"
                    job.finished_at = job.finished_at or time.time()
                self._jobs[job.id] = job
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping corrupt job entry: %s", e)

    def _save_to_disk(self) -> None:
        """Atomic write: dump to a tempfile then rename. Lock held by caller."""
        if self._persist_path is None:
            return
        # Order: newest first, capped.
        ordered = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[
            :_MAX_PERSISTED_JOBS
        ]
        # Rebuild dict so future ``list()`` calls don't see dropped jobs either.
        self._jobs = {j.id: j for j in ordered}

        payload = [j.to_dict() for j in ordered]
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=".jobs.", suffix=".json", dir=str(self._persist_path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.replace(tmp, self._persist_path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning("Could not persist jobs to %s: %s", self._persist_path, e)

    # ---- mutation ----

    def create(self, kind: str, item_labels: list[str]) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            items=[JobItem(label=lbl) for lbl in item_labels],
        )
        with self._lock:
            self._jobs[job.id] = job
            self._save_to_disk()
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

    def active(self, kind: str | None = None) -> list[Job]:
        """Jobs currently in ``queued`` or ``running``, optionally filtered by kind."""
        with self._lock:
            return [
                j
                for j in self._jobs.values()
                if j.status in ("queued", "running")
                and (kind is None or j.kind == kind)
            ]

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
            self._save_to_disk()
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
            self._save_to_disk()
        self._broadcast(snapshot)

    # ---- subscribers ----

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


# Process-wide singleton. Created without persistence here; the API replaces
# it via ``init_persisted_registry`` once settings are available.
_REGISTRY = JobRegistry()


def get_job_registry() -> JobRegistry:
    return _REGISTRY


def init_persisted_registry(persist_path: Path) -> JobRegistry:
    """Swap the global registry for one that writes to ``persist_path``.

    Called from ``create_app`` so subscribers (PipelineWSManager) attach to
    the same instance the API endpoints use. Idempotent: calling twice with
    the same path is fine.
    """
    global _REGISTRY
    _REGISTRY = JobRegistry(persist_path=persist_path)
    return _REGISTRY


__all__ = [
    "Job",
    "JobItem",
    "JobRegistry",
    "JobStatus",
    "get_job_registry",
    "init_persisted_registry",
]
