"""Verifies that the Alembic migrations produce the expected schema.

Runs ``alembic upgrade head`` against a fresh per-test SQLite DB by passing
``-x url=...`` on the alembic command line (env.py reads that override
before falling back to ``dance.config.get_settings().db_url``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


REPO_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_INI = REPO_ROOT / "alembic.ini"


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
    "alembic_version",
}


# Subset of columns we expect on each table (autogen produces all of them; we
# just spot-check the load-bearing ones rather than recapitulating the spec).
EXPECTED_COLUMNS = {
    "tracks": {
        "id",
        "file_hash",
        "spotify_id",
        "file_path",
        "file_name",
        "file_size_bytes",
        "duration_seconds",
        "title",
        "artist",
        "album",
        "year",
        "state",
        "error_message",
        "created_at",
        "updated_at",
        "analyzed_at",
    },
    "audio_analysis": {
        "id",
        "track_id",
        "stem_file_id",
        "bpm",
        "bpm_confidence",
        "key_camelot",
        "key_standard",
        "key_confidence",
        "energy_overall",
        "energy_peak",
        "floor_energy",
        "brightness",
        "warmth",
        "danceability",
        "presence_ratio",
        "rms_curve",
        "rms_curve_hop_ms",
        "rms_curve_length",
        "presence_intervals",
        "dominant_pitch_camelot",
        "dominant_pitch_confidence",
        "vocal_present",
        "kick_density",
        "analyzed_at",
        "created_at",
    },
    "stem_files": {
        "id",
        "track_id",
        "kind",
        "path",
        "model_used",
        "separation_quality",
        "created_at",
    },
    "tags": {"id", "kind", "value", "normalized_value", "created_at"},
    "track_tags": {"track_id", "tag_id", "source", "confidence", "created_at"},
    "regions": {
        "id",
        "track_id",
        "stem_file_id",
        "position_ms",
        "length_ms",
        "region_type",
        "section_label",
        "length_bars",
        "name",
        "color",
        "confidence",
        "source",
        "snapped_to",
        "created_at",
    },
    "track_embeddings": {
        "id",
        "track_id",
        "stem_file_id",
        "model",
        "model_version",
        "dim",
        "embedding",
        "created_at",
    },
    "track_edges": {
        "id",
        "from_track_id",
        "to_track_id",
        "kind",
        "weight",
        "meta",
        "computed_at",
    },
    "sessions": {"id", "name", "notes", "started_at", "ended_at", "created_at"},
    "session_plays": {
        "id",
        "session_id",
        "track_id",
        "played_at",
        "position_in_set",
        "energy_at_play",
        "transition_type",
        "duration_played_ms",
        "created_at",
    },
    "beats": {"id", "track_id", "position_ms", "beat_number", "bar_number", "downbeat"},
    "phrases": {
        "id",
        "track_id",
        "start_ms",
        "end_ms",
        "bar_count",
        "phrase_type",
        "energy_level",
    },
}


@pytest.fixture
def upgraded_db(tmp_path: Path) -> str:
    """Run ``alembic upgrade head`` against a fresh sqlite file. Yield the URL."""

    db_path = tmp_path / "alembic_test.db"
    url = f"sqlite:///{db_path}"

    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(REPO_ROOT / "src" / "dance" / "alembic"))
    cfg.cmd_opts = type("X", (), {"x": [f"url={url}"]})()  # type: ignore[attr-defined]
    command.upgrade(cfg, "head")

    return url


def test_alembic_upgrade_creates_all_tables(upgraded_db: str):
    eng = create_engine(upgraded_db)
    insp = inspect(eng)
    tables = set(insp.get_table_names())
    assert EXPECTED_TABLES.issubset(tables), tables - EXPECTED_TABLES


def test_alembic_upgrade_column_shapes(upgraded_db: str):
    eng = create_engine(upgraded_db)
    insp = inspect(eng)
    for table, expected_cols in EXPECTED_COLUMNS.items():
        actual = {c["name"] for c in insp.get_columns(table)}
        missing = expected_cols - actual
        assert not missing, f"{table} missing columns: {missing}"


def test_alembic_upgrade_partial_unique_indexes(upgraded_db: str):
    eng = create_engine(upgraded_db)
    insp = inspect(eng)
    idx_names = {idx["name"] for idx in insp.get_indexes("audio_analysis")}
    assert "uq_audio_analysis_track_fullmix" in idx_names
    assert "uq_audio_analysis_stem" in idx_names


def test_alembic_upgrade_track_edges_check_constraint(upgraded_db: str):
    """Self-loop on track_edges should be rejected at the DB level."""

    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    eng = create_engine(upgraded_db)
    with eng.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO tracks (file_hash, file_path, file_name, "
                "file_size_bytes, state, created_at, updated_at) "
                "VALUES ('h', '/tmp/x', 'x', 1, 'pending', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        track_id = conn.execute(text("SELECT id FROM tracks LIMIT 1")).scalar()

    with eng.begin() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO track_edges "
                "(from_track_id, to_track_id, kind, weight, computed_at) "
                "VALUES (:t, :t, 'manually_paired', 1.0, CURRENT_TIMESTAMP)"
            ),
            {"t": track_id},
        )
