"""Schema-level tests for the Dance database.

These tests exercise the SQLAlchemy models directly against an in-process
SQLite engine (see ``tests/conftest.py``). The Alembic migration is exercised
separately by ``tests/test_alembic.py``.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from dance.core import database as db
from dance.core.database import (
    now_utc,
    Analysis,
    AudioAnalysis,
    Base,
    Region,
    RegionSource,
    RegionType,
    StemFile,
    Tag,
    TagKind,
    TagSource,
    Track,
    TrackEdge,
    TrackEmbedding,
    TrackTag,
    init_db,
    normalize_tag_value,
)
from dance.core.serialization import (
    decode_curve,
    decode_embedding,
    encode_curve,
    encode_embedding,
)


EXPECTED_TABLES = {
    "tracks",
    "stem_files",
    "audio_analysis",
    "tags",
    "track_tags",
    "regions",
    "track_embeddings",
    "track_edges",
    "sessions",
    "session_plays",
    "beats",
    "phrases",
}


# ---------------------------------------------------------------------------
# 1. init_db creates all tables on a fresh sqlite path
# ---------------------------------------------------------------------------


def test_init_db_creates_all_tables(tmp_path):
    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    db._reset_engine_for_tests()
    init_db(url)

    eng = db.get_engine(url)
    insp = inspect(eng)
    tables = set(insp.get_table_names())
    assert EXPECTED_TABLES.issubset(tables), tables

    # Partial unique indexes must exist on audio_analysis.
    idx_names = {idx["name"] for idx in insp.get_indexes("audio_analysis")}
    assert "uq_audio_analysis_track_fullmix" in idx_names
    assert "uq_audio_analysis_stem" in idx_names

    db._reset_engine_for_tests()


# ---------------------------------------------------------------------------
# 2. Insert track + full-mix analysis + stem + per-stem analysis succeeds
# ---------------------------------------------------------------------------


def test_insert_track_fullmix_then_stem_then_stem_analysis(session, make_track):
    track = make_track()

    full = AudioAnalysis(
        track_id=track.id,
        stem_file_id=None,
        bpm=125.0,
        analyzed_at=now_utc(),
    )
    session.add(full)
    session.flush()

    stem = StemFile(
        track_id=track.id,
        kind="drums",
        path="/tmp/drums.wav",
        created_at=now_utc(),
    )
    session.add(stem)
    session.flush()

    stem_analysis = AudioAnalysis(
        track_id=track.id,
        stem_file_id=stem.id,
        bpm=125.0,
        kick_density=2.4,
        analyzed_at=now_utc(),
    )
    session.add(stem_analysis)
    session.commit()

    rows = (
        session.query(AudioAnalysis)
        .filter(AudioAnalysis.track_id == track.id)
        .all()
    )
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# 3. Second full-mix analysis (stem_file_id NULL) for same track FAILS
# ---------------------------------------------------------------------------


def test_second_fullmix_analysis_for_same_track_fails(session, make_track):
    track = make_track()
    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=None,
            bpm=120.0,
            analyzed_at=now_utc(),
        )
    )
    session.commit()

    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=None,
            bpm=121.0,
            analyzed_at=now_utc(),
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# ---------------------------------------------------------------------------
# 4. Second analysis row for the same stem_file_id FAILS
# ---------------------------------------------------------------------------


def test_second_analysis_for_same_stem_fails(session, make_track):
    track = make_track()
    stem = StemFile(
        track_id=track.id,
        kind="bass",
        path="/tmp/bass.wav",
        created_at=now_utc(),
    )
    session.add(stem)
    session.flush()

    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=stem.id,
            bpm=125.0,
            analyzed_at=now_utc(),
        )
    )
    session.commit()

    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=stem.id,
            bpm=126.0,
            analyzed_at=now_utc(),
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# ---------------------------------------------------------------------------
# 5. RMS curve roundtrip
# ---------------------------------------------------------------------------


def test_rms_curve_roundtrip(session, make_track):
    arr = np.linspace(0, 1, 4096, dtype=np.float32)
    blob = encode_curve(arr)

    track = make_track()
    a = AudioAnalysis(
        track_id=track.id,
        stem_file_id=None,
        rms_curve=blob,
        rms_curve_hop_ms=100,
        rms_curve_length=arr.shape[0],
        analyzed_at=now_utc(),
    )
    session.add(a)
    session.commit()

    fetched = session.query(AudioAnalysis).filter_by(id=a.id).one()
    decoded = decode_curve(fetched.rms_curve, expected_length=fetched.rms_curve_length)
    np.testing.assert_array_equal(decoded, arr)
    assert decoded.dtype == np.float32


# ---------------------------------------------------------------------------
# 6. Embedding roundtrip
# ---------------------------------------------------------------------------


def test_embedding_roundtrip(session, make_track):
    rng = np.random.default_rng(seed=42)
    arr = rng.standard_normal(512).astype(np.float32)
    blob = encode_embedding(arr)

    track = make_track()
    emb = TrackEmbedding(
        track_id=track.id,
        stem_file_id=None,
        model="clap-htsat-unfused",
        model_version=None,
        dim=512,
        embedding=blob,
        created_at=now_utc(),
    )
    session.add(emb)
    session.commit()

    fetched = session.query(TrackEmbedding).filter_by(id=emb.id).one()
    decoded = decode_embedding(fetched.embedding, dim=fetched.dim)
    np.testing.assert_array_equal(decoded, arr)
    assert decoded.dtype == np.float32


# ---------------------------------------------------------------------------
# 7. Cascade delete: removing a Track removes its analysis/stems/regions/embeddings
# ---------------------------------------------------------------------------


def test_cascade_delete_track(session, make_track):
    track = make_track()
    track_id = track.id

    session.add(
        AudioAnalysis(
            track_id=track_id,
            stem_file_id=None,
            bpm=125.0,
            analyzed_at=now_utc(),
        )
    )
    stem = StemFile(
        track_id=track_id,
        kind="vocals",
        path="/tmp/vocals.wav",
        created_at=now_utc(),
    )
    session.add(stem)
    session.flush()

    session.add(
        Region(
            track_id=track_id,
            position_ms=1000,
            region_type=RegionType.CUE.value,
            source=RegionSource.AUTO.value,
            created_at=now_utc(),
        )
    )
    session.add(
        TrackEmbedding(
            track_id=track_id,
            model="clap",
            dim=4,
            embedding=np.zeros(4, dtype=np.float32).tobytes(),
            created_at=now_utc(),
        )
    )
    session.commit()

    # Pre-delete sanity checks.
    assert session.query(AudioAnalysis).filter_by(track_id=track_id).count() == 1
    assert session.query(StemFile).filter_by(track_id=track_id).count() == 1
    assert session.query(Region).filter_by(track_id=track_id).count() == 1
    assert session.query(TrackEmbedding).filter_by(track_id=track_id).count() == 1

    session.delete(track)
    session.commit()

    assert session.query(AudioAnalysis).filter_by(track_id=track_id).count() == 0
    assert session.query(StemFile).filter_by(track_id=track_id).count() == 0
    assert session.query(Region).filter_by(track_id=track_id).count() == 0
    assert session.query(TrackEmbedding).filter_by(track_id=track_id).count() == 0


# ---------------------------------------------------------------------------
# 8. Tag normalization + UNIQUE catches duplicates
# ---------------------------------------------------------------------------


def test_tag_normalization_and_unique(session):
    assert normalize_tag_value("Tech  House") == "tech house"
    assert normalize_tag_value("tech house") == "tech house"
    assert normalize_tag_value("  Tech\tHouse  ") == "tech house"

    raw_a = "Tech  House"
    raw_b = "tech house"
    a = Tag(
        kind=TagKind.SUBGENRE.value,
        value=raw_a,
        normalized_value=normalize_tag_value(raw_a),
        created_at=now_utc(),
    )
    session.add(a)
    session.commit()

    b = Tag(
        kind=TagKind.SUBGENRE.value,
        value=raw_b,
        normalized_value=normalize_tag_value(raw_b),
        created_at=now_utc(),
    )
    session.add(b)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# ---------------------------------------------------------------------------
# 9. Region with NULL stem_file_id and Region with non-NULL stem_file_id
# ---------------------------------------------------------------------------


def test_region_with_and_without_stem(session, make_track):
    track = make_track()
    stem = StemFile(
        track_id=track.id,
        kind="drums",
        path="/tmp/d.wav",
        created_at=now_utc(),
    )
    session.add(stem)
    session.flush()

    whole = Region(
        track_id=track.id,
        stem_file_id=None,
        position_ms=0,
        region_type=RegionType.SECTION.value,
        section_label="intro",
        source=RegionSource.AUTO.value,
        created_at=now_utc(),
    )
    per_stem = Region(
        track_id=track.id,
        stem_file_id=stem.id,
        position_ms=64000,
        length_ms=8000,
        region_type=RegionType.STEM_SOLO.value,
        source=RegionSource.AUTO.value,
        created_at=now_utc(),
    )
    session.add_all([whole, per_stem])
    session.commit()

    rows = session.query(Region).filter_by(track_id=track.id).all()
    assert len(rows) == 2
    assert {r.stem_file_id for r in rows} == {None, stem.id}


# ---------------------------------------------------------------------------
# 10. TrackEdge CHECK constraint: from == to raises IntegrityError
# ---------------------------------------------------------------------------


def test_track_edge_self_loop_rejected(session, make_track):
    t = make_track()
    edge = TrackEdge(
        from_track_id=t.id,
        to_track_id=t.id,
        kind="harmonic_compat",
        weight=0.5,
        computed_at=now_utc(),
    )
    session.add(edge)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# ---------------------------------------------------------------------------
# 11. Analysis alias works
# ---------------------------------------------------------------------------


def test_analysis_alias_is_audio_analysis():
    from dance.core.database import Analysis as ImportedAnalysis

    assert ImportedAnalysis is AudioAnalysis


# ---------------------------------------------------------------------------
# Bonus: TrackTag PK uniqueness allows multiple sources for same (track, tag)
# ---------------------------------------------------------------------------


def test_track_tag_multi_source_allowed(session, make_track):
    track = make_track()
    tag = Tag(
        kind=TagKind.MOOD.value,
        value="dark",
        normalized_value="dark",
        created_at=now_utc(),
    )
    session.add(tag)
    session.flush()

    session.add_all(
        [
            TrackTag(
                track_id=track.id,
                tag_id=tag.id,
                source=TagSource.LLM.value,
                confidence=0.9,
                created_at=now_utc(),
            ),
            TrackTag(
                track_id=track.id,
                tag_id=tag.id,
                source=TagSource.MANUAL.value,
                confidence=None,
                created_at=now_utc(),
            ),
        ]
    )
    session.commit()

    assert (
        session.query(TrackTag).filter_by(track_id=track.id, tag_id=tag.id).count()
        == 2
    )
