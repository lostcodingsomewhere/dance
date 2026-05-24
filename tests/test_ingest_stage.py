"""Tests for the IngestStage's dedup logic — especially the "adopt
optimistic" path that recovers from the Spotify-ingest-then-scan race."""

from __future__ import annotations

from pathlib import Path

from dance.core.database import Track, TrackState, now_utc
from dance.pipeline.stages.ingest import IngestStage
from tests.audio_fixtures import TrackSpec, write_track


def _make_audio(tmp_path: Path, name: str = "track.wav") -> Path:
    audio = tmp_path / name
    write_track(audio, TrackSpec(bpm=124.0, bars=2))
    return audio


def test_ingest_creates_new_track_when_no_match(session, tmp_path):
    """Baseline: a fresh file with no matching DB row → new Track row."""
    audio = _make_audio(tmp_path)
    stage = IngestStage(library_dir=tmp_path)

    result = stage.ingest_file(session, audio)

    assert result.status == "new"
    track = session.get(Track, result.track_id)
    assert track is not None
    assert track.file_path == str(audio)
    assert track.state == TrackState.PENDING.value


def test_ingest_unchanged_when_same_hash_same_path(session, tmp_path):
    """Idempotent: scanning the same file twice doesn't duplicate."""
    audio = _make_audio(tmp_path)
    stage = IngestStage(library_dir=tmp_path)

    first = stage.ingest_file(session, audio)
    second = stage.ingest_file(session, audio)

    assert first.track_id == second.track_id
    assert second.status == "unchanged"


def test_ingest_adopts_optimistic_track_by_path(session, tmp_path):
    """The Spotify-ingest path: a Track row pre-exists with a placeholder
    file_hash + the audio's final path. The scanner should ADOPT that row
    (replace the placeholder hash with the real SHA256) instead of
    creating a duplicate.

    This was a real bug uncovered during the end-to-end Spotify ingest
    test on 2026-05-24 — Four Tet "Sing" created Track 198 (optimistic)
    AND Track 199 (scanned)."""
    audio = _make_audio(tmp_path)
    # Pre-create an optimistic row with a placeholder hash, mimicking
    # what /pipeline/ingest/track does.
    optimistic = Track(
        file_hash="pending:abc123spotifyid",
        spotify_id="abc123spotifyid",
        file_path=str(audio),
        file_name=audio.name,
        file_size_bytes=0,
        title="Sing",
        artist="Four Tet",
        state=TrackState.PENDING.value,
        created_at=now_utc(),
        updated_at=now_utc(),
    )
    session.add(optimistic)
    session.commit()

    stage = IngestStage(library_dir=tmp_path)
    result = stage.ingest_file(session, audio)

    assert result.status == "adopted"
    assert result.track_id == optimistic.id

    session.refresh(optimistic)
    # Placeholder gone, real hash + real size in place.
    assert not optimistic.file_hash.startswith("pending:")
    assert len(optimistic.file_hash) == 64  # SHA256 hex
    assert optimistic.file_size_bytes > 0
    # Optimistic metadata stays — Spotify's title/artist beat ID3 tags.
    assert optimistic.title == "Sing"
    assert optimistic.artist == "Four Tet"

    # And only one Track row exists for that path.
    rows = session.query(Track).filter_by(file_path=str(audio)).all()
    assert len(rows) == 1


def test_ingest_moved_file_updates_existing(session, tmp_path):
    """Existing dedup-by-hash path is preserved when the file actually
    moves (different path, same content)."""
    audio = _make_audio(tmp_path, "original.wav")
    stage = IngestStage(library_dir=tmp_path)

    first = stage.ingest_file(session, audio)
    moved = tmp_path / "moved.wav"
    audio.rename(moved)
    second = stage.ingest_file(session, moved)

    assert first.track_id == second.track_id
    assert second.status == "updated"
    session.expire_all()
    track = session.get(Track, first.track_id)
    assert track.file_path == str(moved)
