"""Unit tests for per-role plan-queue helpers (:mod:`dance.core.set_queues`)."""

from __future__ import annotations

from dance.core import set_queues as sq
from dance.core.database import StemFile


def test_parse_plan_roundtrip_and_defaults():
    plan = {"drums": [1, 2], "vocals": [3]}
    encoded = sq.encode_plan(plan)
    parsed = sq.parse_plan(encoded)
    assert parsed["drums"] == [1, 2]
    assert parsed["vocals"] == [3]
    # every role present, empty when unqueued
    assert parsed["bass"] == []
    assert set(parsed.keys()) == set(sq.PLAN_ROLES)


def test_parse_plan_bad_inputs():
    assert sq.parse_plan(None)["drums"] == []
    assert sq.parse_plan("not json")["drums"] == []
    assert sq.parse_plan('{"banana": [1]}')["drums"] == []
    assert sq.parse_plan('{"drums": [1, "x", 2]}')["drums"] == [1, 2]


def test_encode_plan_empty_is_none():
    assert sq.encode_plan({r: [] for r in sq.PLAN_ROLES}) is None


def test_role_to_column():
    assert sq.role_to_column("song") == "mix"
    assert sq.role_to_column("drums") == "drums"


def test_plan_sequence_interleaves_by_depth():
    queues = {"drums": [10, 20], "bass": [10, 30], "vocals": [], "other": [], "song": []}
    # depth 0: drums10, bass10(dup) ; depth 1: drums20, bass30
    assert sq.plan_sequence(queues) == [10, 20, 30]


def test_context_combo_stem_ids_uses_other_role_tails(session, make_track):
    a = make_track(title="a")
    b = make_track(title="b")
    stems = {}
    for t in (a, b):
        for kind in ("drums", "bass", "vocals", "other"):
            st = StemFile(track_id=t.id, kind=kind, path=f"/tmp/{t.id}-{kind}.wav")
            session.add(st)
            session.flush()
            stems[(t.id, kind)] = st.id
    session.flush()

    queues = {
        "drums": [a.id, b.id],   # tail = b
        "bass": [a.id],          # tail = a
        "vocals": [],
        "other": [],
        "song": [a.id],          # song skipped (anchor, no stem)
    }
    # Filling vocals → combo = tails of other stem roles: drums(b), bass(a).
    combo = sq.context_combo_stem_ids(session, queues, exclude_role="vocals")
    assert set(combo) == {stems[(b.id, "drums")], stems[(a.id, "bass")]}
