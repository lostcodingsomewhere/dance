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


def test_ingest_normalizes_nfd_path_to_nfc(session, tmp_path):
    """Unicode-normalization regression: a path handed to ingest in
    decomposed (NFD) form must be persisted in composed (NFC) form so the
    DB and the on-disk filename agree after an rsync between macOS machines.

    This bit a real 282-track library on 2026-05-30 — 65 tracks with
    accented filenames failed ``os.path.exists`` because the DB held NFD
    while the rsynced files on disk were NFC. See ``dance.core.paths``."""
    import unicodedata

    # "Beyoncé.wav": NFD = "e" + combining acute (U+0301).
    # NFC collapses it to a single precomposed "é". Same render, diff bytes.
    nfd_name = "Beyoncé.wav"
    nfc_name = "Beyoncé.wav"
    assert nfd_name != nfc_name  # genuinely distinct byte strings

    audio = tmp_path / nfd_name
    write_track(audio, TrackSpec(bpm=124.0, bars=2))
    stage = IngestStage(library_dir=tmp_path)

    # Ingest the NFD path (mimics what rglob yields when the file is NFD).
    result = stage.ingest_file(session, audio)

    assert result.status == "new"
    track = session.get(Track, result.track_id)
    assert track is not None
    # Persisted as NFC even though we ingested NFD.
    assert track.file_path == str(tmp_path / nfc_name)
    assert track.file_name == nfc_name
    assert unicodedata.is_normalized("NFC", track.file_path)
    assert unicodedata.is_normalized("NFC", track.file_name)


def test_ingest_nfc_and_nfd_paths_dedup_to_one_row(session, tmp_path):
    """The same file referenced once as NFD and once as NFC must resolve to
    a single Track row — normalization makes the path-based dedup pass agree."""
    import unicodedata

    nfd_name = "Résumé.wav"  # "Résumé.wav" decomposed
    audio = tmp_path / nfd_name
    write_track(audio, TrackSpec(bpm=124.0, bars=2))
    stage = IngestStage(library_dir=tmp_path)

    first = stage.ingest_file(session, audio)
    # Re-ingest via the precomposed (NFC) spelling of the same path.
    nfc_path_obj = Path(unicodedata.normalize("NFC", str(audio)))
    second = stage.ingest_file(session, nfc_path_obj)

    assert first.track_id == second.track_id
    assert second.status == "unchanged"
    assert session.query(Track).count() == 1
