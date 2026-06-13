"""
Heal ledger — a tiny sidecar JSON that counts how many times each track has
been healed from a given inflight (``*-ing``) state.

Why a sidecar instead of a DB column? Adding a column to the ``tracks`` table
is a schema change, which (per the repo's workflow rules) needs a separate
proposal + Alembic migration. The heal counter is pure operational
bookkeeping — it never feeds recommendations, the API, or the ``.als`` writer —
so a small JSON file keyed by track id under ``settings.data_dir`` is the
right weight. It's also self-healing: a corrupt/missing file just resets the
counters, which at worst gives a track one extra retry.

The ledger exists to break a real infinite-crash loop: when a track crashes
hard mid-stage (e.g. a native SIGBUS / exit 138 during Demucs that the
dispatcher's ``except Exception`` can't catch), it's left in its ``*-ing``
state. On the next ``dance process`` startup ``_heal_inflight_orphans`` resets
it to the prior state AND nulls ``error_message`` — so it silently re-crashes
forever with nothing in ``dance status``. This ledger lets the heal loop notice
"this track has now been healed from ``separating`` N times" and promote it to
ERROR with an actionable message instead of re-queuing it again.

Layout (``<data_dir>/heal_ledger.json``)::

    {"<track_id>": {"state": "separating", "count": 2}}

``state`` is the inflight state the track was healed *from*; if a track later
gets stuck in a *different* inflight state the counter resets (a different
stage crashing is a different problem). The counter is also cleared on any
clean successful advance via :func:`clear`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

_LEDGER_NAME = "heal_ledger.json"


class _Entry(TypedDict):
    """One ledger row: the inflight state a track was healed from + how many
    times. ``state`` is ``None`` only for defensively-parsed corrupt rows."""

    state: str | None
    count: int


class HealLedger:
    """Counts repeated heals-from-inflight per track, persisted to JSON."""

    def __init__(self, data_dir: Path) -> None:
        self._path = Path(data_dir) / _LEDGER_NAME
        self._data: dict[str, _Entry] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, _Entry]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                out: dict[str, _Entry] = {}
                for k, v in raw.items():
                    if not isinstance(v, dict):
                        continue
                    state = v.get("state")
                    try:
                        count = int(v.get("count", 0))
                    except (TypeError, ValueError):
                        continue
                    out[str(k)] = {
                        "state": state if isinstance(state, str) else None,
                        "count": count,
                    }
                return out
        except (OSError, ValueError, TypeError) as e:
            # Corrupt/unreadable ledger: reset rather than crash the run. Worst
            # case a track gets one extra retry before being flagged.
            logger.warning("Heal ledger at %s unreadable (%s) — resetting", self._path, e)
        return {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=".heal_ledger.", suffix=".json", dir=str(self._path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False)
                os.replace(tmp, self._path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning("Could not persist heal ledger to %s: %s", self._path, e)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def record_heal(self, track_id: int, inflight_state: str) -> int:
        """Record that ``track_id`` was just healed from ``inflight_state``.

        Returns the running heal count for this (track, state). If the track
        was previously stuck in a *different* inflight state, the counter
        resets to 1 (a new stage crashing is a fresh problem).
        """
        key = str(track_id)
        entry = self._data.get(key)
        if entry is None or entry["state"] != inflight_state:
            count = 1
        else:
            count = entry["count"] + 1
        self._data[key] = {"state": inflight_state, "count": count}
        self._save()
        return count

    def count_for(self, track_id: int, inflight_state: str) -> int:
        """Current recorded heal count for (track, state), 0 if none/mismatch."""
        entry = self._data.get(str(track_id))
        if entry is None or entry["state"] != inflight_state:
            return 0
        return entry["count"]

    def clear(self, track_id: int) -> None:
        """Forget a track — call on any clean successful advance."""
        if self._data.pop(str(track_id), None) is not None:
            self._save()

    def stuck_tracks(self) -> dict[int, _Entry]:
        """Return {track_id: {'state': ..., 'count': ...}} for every tracked
        entry, so callers (e.g. ``dance status``) can surface repeated
        crashers. Empty dict on a clean ledger."""
        out: dict[int, _Entry] = {}
        for k, v in self._data.items():
            try:
                out[int(k)] = v
            except (TypeError, ValueError):
                continue
        return out
