"""Re-run pipeline stages on tracks that already went through them.

``dance process`` only ever moves tracks *forward*, so a fix to a stage
(say, the analyzer) never reaches tracks that are already ``complete``.
Reprocessing rewinds selected tracks to a stage's ``input_state`` and then
lets the normal dispatcher carry them forward again — the same code path as
``dance process``, so idempotent stages short-circuit as usual (``separate``
skips Demucs when all four stems are already on disk).

Only tracks that have *reached* the chosen stage are rewound: a track still
upstream of it is left alone (rewinding it "back" would actually move it
forward past stages it never ran). Tracks marked ``duplicate_of`` are left
alone by default, and ``error`` tracks only on request, since their failure
point isn't recorded in their state.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from dance.core.database import Track, TrackState
from dance.pipeline.stage import Stage

logger = logging.getLogger(__name__)

#: Pipeline order. ``TrackState`` is declared in the order tracks advance,
#: so its declaration order (minus ``error``) is the canonical position.
STATE_ORDER: tuple[str, ...] = tuple(s.value for s in TrackState if s is not TrackState.ERROR)


def state_position(state: str) -> int | None:
    """Index of ``state`` in :data:`STATE_ORDER`; ``None`` for error/unknown."""
    try:
        return STATE_ORDER.index(state)
    except ValueError:
        return None


@dataclass(frozen=True)
class ReprocessRow:
    track_id: int
    title: str
    state: str


@dataclass
class ReprocessPlan:
    """What a reprocess would do. Built by :func:`plan_reprocess`."""

    stage_name: str
    reset_to: str
    selected: list[ReprocessRow] = field(default_factory=list)
    #: reason -> tracks left untouched for that reason
    skipped: dict[str, list[ReprocessRow]] = field(default_factory=dict)

    @property
    def track_ids(self) -> list[int]:
        return [r.track_id for r in self.selected]

    def skip(self, reason: str, row: ReprocessRow) -> None:
        self.skipped.setdefault(reason, []).append(row)


SKIP_UPSTREAM = "not yet at this stage"
SKIP_DUPLICATE = "duplicate_of set (use --include-duplicates)"
SKIP_ERROR = "in error (use --include-errors)"
SKIP_LIMIT = "over --limit"
SKIP_MISSING = "no such track"


def find_stage(stages: Sequence[Stage], name: str) -> Stage:
    for stage in stages:
        if stage.name == name:
            return stage
    known = ", ".join(s.name for s in stages)
    raise ValueError(f"Unknown stage {name!r}. Registered stages: {known}")


def plan_reprocess(
    session: Session,
    stage: Stage,
    *,
    track_ids: Iterable[int] | None = None,
    limit: int | None = None,
    include_duplicates: bool = False,
    include_errors: bool = False,
) -> ReprocessPlan:
    """Decide which tracks a reprocess from ``stage`` would rewind.

    ``track_ids=None`` means every track. Selection is by ascending id so a
    ``limit`` is deterministic.
    """
    plan = ReprocessPlan(stage_name=stage.name, reset_to=stage.input_state.value)
    from_pos = state_position(stage.input_state.value)
    if from_pos is None:  # pragma: no cover — guarded by the stage table test
        raise ValueError(f"Stage {stage.name!r} has an unordered input_state")

    query = session.query(Track).order_by(Track.id)
    wanted: list[int] | None = None
    if track_ids is not None:
        wanted = sorted({int(t) for t in track_ids})
        query = query.filter(Track.id.in_(wanted))
    tracks = query.all()

    if wanted is not None:
        found = {int(t.id) for t in tracks}
        for tid in wanted:
            if tid not in found:
                plan.skip(SKIP_MISSING, ReprocessRow(tid, "-", "-"))

    for track in tracks:
        row = ReprocessRow(int(track.id), track.title or track.file_name or "-", str(track.state))
        if track.duplicate_of is not None and not include_duplicates:
            plan.skip(SKIP_DUPLICATE, row)
            continue
        if track.state == TrackState.ERROR.value:
            if not include_errors:
                plan.skip(SKIP_ERROR, row)
                continue
        else:
            pos = state_position(str(track.state))
            if pos is None or pos < from_pos:
                plan.skip(SKIP_UPSTREAM, row)
                continue
        if limit is not None and len(plan.selected) >= limit:
            plan.skip(SKIP_LIMIT, row)
            continue
        plan.selected.append(row)
    return plan


def apply_reprocess(session: Session, plan: ReprocessPlan, *, data_dir: Path | None = None) -> int:
    """Rewind every selected track to ``plan.reset_to``. Returns the count.

    Clears ``error_message`` and, if ``data_dir`` is given, the track's
    heal-ledger entry (a deliberate re-run starts its crash count fresh).
    """
    ids = plan.track_ids
    if not ids:
        return 0
    n = (
        session.query(Track)
        .filter(Track.id.in_(ids))
        .update(
            {Track.state: plan.reset_to, Track.error_message: None},
            synchronize_session=False,
        )
    )
    session.commit()
    if data_dir is not None:
        from dance.pipeline.heal_ledger import HealLedger

        ledger = HealLedger(data_dir)
        for tid in ids:
            ledger.clear(tid)
    logger.info("Reprocess: rewound %d track(s) to %s", n, plan.reset_to)
    return int(n)


def unsettled_count(session: Session, track_ids: Iterable[int]) -> int:
    """How many of ``track_ids`` are neither ``complete`` nor ``error``."""
    ids = list(track_ids)
    if not ids:
        return 0
    return int(
        session.query(Track)
        .filter(
            Track.id.in_(ids),
            Track.state.notin_([TrackState.COMPLETE.value, TrackState.ERROR.value]),
        )
        .count()
    )


def run_until_settled(
    dispatcher,
    track_ids: Sequence[int],
    *,
    skip: Iterable[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Run the dispatcher until every rewound track is complete/error.

    One parallel ``Dispatcher.run`` pass can end with tracks still
    mid-pipeline: a downstream stage's workers give up after an idle grace
    period that can be shorter than the upstream work on a track (analysis +
    stem analysis + region detection on a long track). ``dance process``
    leaves that to a manual re-run; here we re-run while passes still make
    progress, bounded by the number of stages (every pass advances each
    unsettled track by at least one stage, since it starts in some stage's
    input state). Returns per-stage totals across passes.
    """
    skip_set = set(skip or ())
    only = track_ids[0] if len(track_ids) == 1 else None
    totals: dict[str, dict[str, int]] = {}
    for _ in range(len(dispatcher.stages) + 1):
        result = dispatcher.run(skip=skip_set, track_id=only)
        moved = 0
        for name, counts in result.items():
            agg = totals.setdefault(name, {"processed": 0, "errors": 0, "skipped": 0})
            for key, value in counts.items():
                agg[key] = agg.get(key, 0) + int(value)
                moved += int(value)
        session = (
            dispatcher.session if dispatcher.session is not None else dispatcher.session_factory()
        )
        try:
            remaining = unsettled_count(session, track_ids)
        finally:
            if dispatcher.session is None:
                session.close()
        if not moved or not remaining:
            break
        logger.info("Reprocess: %d track(s) still mid-pipeline; running another pass", remaining)
    return totals


def inflight_states(stages: Sequence[Stage]) -> set[str]:
    """Values of every stage's ``inflight_state`` (the ``*-ing`` states)."""
    out: set[str] = set()
    for stage in stages:
        inflight = getattr(stage, "inflight_state", None)
        if inflight is not None:
            out.add(inflight.value)
    return out
