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

from dance.core.database import StemFile, Track, TrackState
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
    #: Caveats the operator should see before running (printed by the CLI).
    notes: list[str] = field(default_factory=list)

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

#: Stems a track needs on disk for ``separate`` to skip Demucs
#: (``StemSeparationStage.process``'s idempotency check).
_STEMS_PER_TRACK = 4


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
        row = ReprocessRow(
            int(track.id), str(track.title or track.file_name or "-"), str(track.state)
        )
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

    if stage.name == "separate":
        kept = tracks_with_stems_on_disk(session, plan.track_ids)
        if kept:
            plan.notes.append(
                f"{len(kept)} of {len(plan.selected)} selected track(s) already have all "
                f"{_STEMS_PER_TRACK} stems on disk. `separate` keeps existing stems and does "
                "not re-run Demucs for them, so for those tracks this only re-runs the stages "
                "after it. To re-separate a track, delete its stem files first."
            )
    return plan


def tracks_with_stems_on_disk(session: Session, track_ids: Iterable[int]) -> list[int]:
    """Ids among ``track_ids`` whose four stems all exist on disk — the
    tracks ``separate`` would short-circuit instead of running Demucs."""
    ids = sorted({int(t) for t in track_ids})
    if not ids:
        return []
    paths: dict[int, list[str]] = {}
    for tid, path in session.query(StemFile.track_id, StemFile.path).filter(
        StemFile.track_id.in_(ids)
    ):
        paths.setdefault(int(tid), []).append(str(path))
    return [
        tid
        for tid in ids
        if len(paths.get(tid, ())) == _STEMS_PER_TRACK and all(Path(p).exists() for p in paths[tid])
    ]


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
    """Run the dispatcher on exactly ``track_ids`` until each is complete/error.

    Every pass is scoped to the selection (``Dispatcher.run(track_id=<ids>)``),
    so tracks the reprocess didn't select — new downloads waiting in
    ``pending``, tracks parked in ``regions_detected``, anything a crashed run
    left mid-stage — are neither processed nor auto-advanced past a skipped
    stage. They stay where they are for the next ``dance process``.

    One parallel ``Dispatcher.run`` pass can end with tracks still
    mid-pipeline: a downstream stage's workers give up after an idle grace
    period that can be shorter than the upstream work on a track (analysis +
    stem analysis + region detection on a long track). ``dance process``
    leaves that to a manual re-run; here we re-run while passes still make
    progress, bounded by the number of stages (every pass advances each
    unsettled track by at least one stage, since it starts in some stage's
    input state). Selected tracks waiting at a skipped stage are advanced
    past it here, before and after every pass (the dispatcher does it only
    at the start of a pass, and doesn't count it). Returns per-stage totals
    across passes.
    """
    scope = frozenset(int(t) for t in track_ids)
    if not scope:
        return {}
    skip_set = set(skip or ())
    totals: dict[str, dict[str, int]] = {}

    def _tally(name: str, key: str, value: int) -> None:
        agg = totals.setdefault(name, {"processed": 0, "errors": 0, "skipped": 0})
        agg[key] = agg.get(key, 0) + int(value)

    def _settle_skipped() -> tuple[int, int]:
        """Advance the selection past skipped stages (counted here — the
        dispatcher's own parallel auto-advance isn't). Returns
        ``(advanced, still_unsettled)``."""
        session = (
            dispatcher.session if dispatcher.session is not None else dispatcher.session_factory()
        )
        try:
            advanced = advance_skipped(session, dispatcher.stages, skip_set, scope)
            for name, n in advanced.items():
                _tally(name, "skipped", n)
            return sum(advanced.values()), unsettled_count(session, scope)
        finally:
            if dispatcher.session is None:
                session.close()

    _, remaining = _settle_skipped()
    for _ in range(len(dispatcher.stages) + 1):
        if not remaining:
            break
        result = dispatcher.run(skip=skip_set, track_id=scope)
        moved = 0
        for name, counts in result.items():
            for key, value in counts.items():
                _tally(name, key, value)
                moved += int(value)
        advanced, remaining = _settle_skipped()
        moved += advanced
        if not moved or not remaining:
            break
        logger.info("Reprocess: %d track(s) still mid-pipeline; running another pass", remaining)
    return totals


def advance_skipped(
    session: Session,
    stages: Sequence[Stage],
    skip: Iterable[str],
    track_ids: Iterable[int],
) -> dict[str, int]:
    """Move ``track_ids`` waiting at a skipped stage straight to its output
    state — and only those ids. Stages are walked in pipeline order, so
    consecutive skipped stages chain. Returns ``{stage_name: n_advanced}``."""
    ids = sorted({int(t) for t in track_ids})
    skip_set = set(skip)
    out: dict[str, int] = {}
    if not ids or not skip_set:
        return out
    for stage in stages:
        if stage.name not in skip_set:
            continue
        n = (
            session.query(Track)
            .filter(Track.id.in_(ids), Track.state == stage.input_state.value)
            .update({Track.state: stage.output_state.value}, synchronize_session=False)
        )
        if n:
            out[stage.name] = int(n)
    if out:
        session.commit()
    return out


def inflight_states(stages: Sequence[Stage]) -> set[str]:
    """Values of every stage's ``inflight_state`` (the ``*-ing`` states)."""
    out: set[str] = set()
    for stage in stages:
        inflight = getattr(stage, "inflight_state", None)
        if inflight is not None:
            out.add(inflight.value)
    return out
