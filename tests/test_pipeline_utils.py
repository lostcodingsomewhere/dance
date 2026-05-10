"""Tests for shared pipeline helpers (db.upsert, device.pick_device)."""

from __future__ import annotations

from unittest.mock import patch

from dance.core.database import StemFile, Tag, TagKind, normalize_tag_value, now_utc
from dance.pipeline.utils.db import get_stems_for_track, upsert
from dance.pipeline.utils.device import pick_device


def test_upsert_inserts_when_missing(session):
    tag = upsert(
        session,
        Tag,
        where={"kind": TagKind.MOOD.value, "normalized_value": "dark"},
        value="dark",
        created_at=now_utc(),
    )
    session.commit()
    assert tag.id is not None
    assert session.query(Tag).count() == 1


def test_upsert_updates_when_present(session):
    upsert(
        session,
        Tag,
        where={"kind": TagKind.MOOD.value, "normalized_value": "groovy"},
        value="groovy",
        created_at=now_utc(),
    )
    session.commit()

    # Same `where`, different value — should UPDATE the existing row.
    updated = upsert(
        session,
        Tag,
        where={"kind": TagKind.MOOD.value, "normalized_value": "groovy"},
        value="Groovy",
        created_at=now_utc(),
    )
    session.commit()

    assert session.query(Tag).count() == 1
    assert updated.value == "Groovy"


def test_upsert_with_none_value_in_where(session, make_track):
    """``stem_file_id=None`` is a common where-clause and must match correctly."""
    from dance.core.database import AudioAnalysis

    track = make_track()
    session.commit()

    upsert(
        session,
        AudioAnalysis,
        where={"track_id": track.id, "stem_file_id": None},
        bpm=128.0,
        analyzed_at=now_utc(),
    )
    session.commit()
    assert session.query(AudioAnalysis).count() == 1

    # Same upsert again — should not create a second row.
    upsert(
        session,
        AudioAnalysis,
        where={"track_id": track.id, "stem_file_id": None},
        bpm=130.0,
        analyzed_at=now_utc(),
    )
    session.commit()
    assert session.query(AudioAnalysis).count() == 1


def test_get_stems_for_track(session, make_track):
    track = make_track()
    session.add_all(
        [
            StemFile(track_id=track.id, kind="drums", path="/tmp/d.wav"),
            StemFile(track_id=track.id, kind="bass", path="/tmp/b.wav"),
            StemFile(track_id=track.id, kind="vocals", path="/tmp/v.wav"),
            StemFile(track_id=track.id, kind="other", path="/tmp/o.wav"),
        ]
    )
    session.commit()

    stems = get_stems_for_track(session, track.id)
    assert len(stems) == 4
    # Deterministic order
    assert [s.id for s in stems] == sorted(s.id for s in stems)


def test_get_stems_for_unknown_track_returns_empty(session):
    assert get_stems_for_track(session, 99999) == []


def test_pick_device_explicit():
    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda") == "cuda"
    assert pick_device("mps") == "mps"


def test_pick_device_auto_prefers_cuda():
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.backends.mps.is_available", return_value=True
    ):
        assert pick_device("auto") == "cuda"


def test_pick_device_auto_falls_through_to_mps():
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=True
    ):
        assert pick_device("auto") == "mps"


def test_pick_device_auto_falls_through_to_cpu():
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=False
    ):
        assert pick_device("auto") == "cpu"


def test_normalize_tag_value():
    # Sanity check while we're here — confirms the helper used by tags works.
    assert normalize_tag_value("Tech House") == "tech house"
    assert normalize_tag_value("  tech  HOUSE  ") == "tech house"
    assert normalize_tag_value("TECH\tHOUSE") == "tech house"
