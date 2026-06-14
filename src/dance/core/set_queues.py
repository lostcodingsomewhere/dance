"""Per-role plan-queue helpers for a Set.

A Set's plan is per-role queues of source tracks the DJ intends to bring in:
``{role: [track_id, ...]}``, stored as JSON on ``Set.plan``. The plan lives
inside the Booth rec grid — queued picks on top of each role column, live recs
below. No scenes, no coupling; you combine parts live.

Pure functions only: parse/validate the JSON and derive the context the
recommender needs to score what fits a role next.
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from dance.core.database import PlanRole, StemFile

# Role columns, left→right. ``song`` is the full-track anchor (== recommender
# column ``mix``); the four others are extracted stems.
PLAN_ROLES: tuple[str, ...] = tuple(r.value for r in PlanRole)
_STEM_ROLES: tuple[str, ...] = ("drums", "bass", "vocals", "other")


def role_to_column(role: str) -> str:
    """Map a plan role to a recommender/StemFile column. ``song`` → ``mix``."""
    return "mix" if role == PlanRole.SONG.value else role


def parse_plan(plan: str | None) -> dict[str, list[int]]:
    """Decode ``Set.plan`` JSON into ``{role: [track_id, ...]}``.

    Drops unknown roles and non-int ids; invalid JSON → empty plan rather than
    raising. Every known role is present (empty list when unqueued) so callers
    can iterate uniformly.
    """
    out: dict[str, list[int]] = {r: [] for r in PLAN_ROLES}
    if not plan:
        return out
    try:
        raw = json.loads(plan)
    except (ValueError, TypeError):
        return out
    if not isinstance(raw, dict):
        return out
    for role, ids in raw.items():
        if role in PLAN_ROLES and isinstance(ids, list):
            out[role] = [int(t) for t in ids if isinstance(t, int)]
    return out


def encode_plan(queues: dict[str, list[int]]) -> str | None:
    """Encode ``{role: [track_id]}`` to JSON; ``None`` when every queue empty."""
    clean = {
        r: [int(t) for t in queues.get(r, [])]
        for r in PLAN_ROLES
        if queues.get(r)
    }
    return json.dumps(clean) if clean else None


def plan_sequence(queues: dict[str, list[int]]) -> list[int]:
    """The plan's track lineup (first-appearance order) — the journey input."""
    out: list[int] = []
    seen: set[int] = set()
    # interleave by depth then role so order roughly follows how the set builds
    depth = max((len(q) for q in queues.values()), default=0)
    for i in range(depth):
        for role in PLAN_ROLES:
            q = queues.get(role, [])
            if i < len(q) and q[i] not in seen:
                seen.add(q[i])
                out.append(q[i])
    return out


def context_combo_stem_ids(
    session: Session,
    queues: dict[str, list[int]],
    *,
    exclude_role: str,
) -> list[int]:
    """Resolve the *tail* of every other role's queue to ``stem_files.id`` —
    the combo to score the next pick for ``exclude_role`` against.

    Uses the last queued track per other stem role (the plan's current end
    state). The ``song`` role is skipped (whole-track anchor, no single stem).
    """
    wanted: list[tuple[int, str]] = []
    for role in _STEM_ROLES:
        if role == exclude_role:
            continue
        q = queues.get(role, [])
        if q:
            wanted.append((q[-1], role))
    if not wanted:
        return []
    track_ids = {tid for tid, _ in wanted}
    rows = (
        session.query(StemFile.id, StemFile.track_id, StemFile.kind)
        .filter(StemFile.track_id.in_(track_ids))
        .all()
    )
    by_track_kind = {(int(t), str(k)): int(sid) for sid, t, k in rows}
    out: list[int] = []
    for tid, role in wanted:
        sid = by_track_kind.get((tid, role))
        if sid is not None:
            out.append(sid)
    return out


__all__ = [
    "PLAN_ROLES",
    "context_combo_stem_ids",
    "encode_plan",
    "parse_plan",
    "plan_sequence",
    "role_to_column",
]
