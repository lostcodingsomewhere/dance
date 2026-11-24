"""Track read endpoints."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi import Path as PathParam
from sqlalchemy.orm import Session

from dance.als import AlsGenerator
from dance.als.generator import AlsExportError, AlsOutsideDirError
from dance.api.deps import fullmix_analysis, get_session, get_settings, track_to_out
from dance.api.schemas import (
    AlsExportRequest,
    AlsExportResult,
    PadMove,
    PadOut,
    RegionOut,
    StemFileOut,
    TrackOut,
    TrackVersionOut,
    WaveformOut,
)
from dance.audio import get_or_compute_peaks
from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    Beat,
    Phrase,
    Region,
    RegionSource,
    RegionType,
    SessionPlay,
    StemFile,
    Track,
    TrackEdge,
)
from dance.pipeline.utils.pads import (
    PAD_COUNT,
    Section,
    bar_ms,
    load_track_pads,
    pad_at,
    pad_region,
    pick_pads,
    section_containing,
    section_from_phrase,
    snap_to_bar,
    stem_ids_by_role,
    track_bpm,
)
from dance.recommender.dedup import dedupe_track_rows
from dance.recommender.versions import version_index

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tracks", tags=["tracks"])

# Per-stem endpoints live under /stems/{stem_file_id}/... because they're
# keyed on the StemFile row, not on a (track_id, kind) pair. Same module so
# the import wiring stays simple and the test fixture pattern is shared.
stems_router = APIRouter(prefix="/stems", tags=["stems"])


# Companion app's waveform renderer asks for between 50 and 1000 peaks
# depending on viewport width. Anything past 1000 is wasted bandwidth (it'll
# render to fewer pixels anyway); anything below 50 is degenerate.
_MIN_NUM_PEAKS = 50
_MAX_NUM_PEAKS = 1000


# How many extra rows to pull before collapsing duplicates, so the caller
# still gets `limit` results back. The worst real group in this library has
# 3 copies; 3x leaves headroom without needing a second query.
_DEDUPE_OVERFETCH = 3

_EVERY_ROLE = "all"


@router.get("/search", response_model=list[TrackOut])
def search_tracks(
    q: str = Query("", description="Fuzzy match against title + artist"),
    limit: int = Query(8, ge=1, le=50),
    bpm_min: float | None = Query(None),
    bpm_max: float | None = Query(None),
    key: str | None = Query(None),
    energy: int | None = Query(None),
    session: Session = Depends(get_session),
) -> list[dict]:
    """Fuzzy name/artist search — Cmd-K's "find this specific track" half.

    Case-insensitive ``LIKE`` over ``tracks.title`` and ``tracks.artist``. At
    our scale a sub-second sequential scan is fine; if we cross 50k tracks
    we'll add a trigram index. Optional BPM/key/energy chips narrow further.
    Empty ``q`` returns the most-recently-updated tracks (browse mode).
    """
    needs_analysis = any(v is not None for v in (bpm_min, bpm_max, key, energy))
    # Hide rows marked as a redundant copy of another recording — searching
    # "navi" used to return the same song three times.
    query = session.query(Track).filter(Track.duplicate_of.is_(None))
    term = q.strip()
    if term:
        pattern = f"%{term}%"
        query = query.filter((Track.title.ilike(pattern)) | (Track.artist.ilike(pattern)))
    if needs_analysis:
        query = query.join(
            AudioAnalysis,
            (AudioAnalysis.track_id == Track.id) & (AudioAnalysis.stem_file_id.is_(None)),
        )
        if bpm_min is not None:
            query = query.filter(AudioAnalysis.bpm >= bpm_min)
        if bpm_max is not None:
            query = query.filter(AudioAnalysis.bpm <= bpm_max)
        if key is not None:
            query = query.filter(AudioAnalysis.key_camelot == key)
        if energy is not None:
            query = query.filter(AudioAnalysis.floor_energy == energy)
    if term:
        # Rank exact-prefix matches above containment matches via a simple
        # CASE — DBs without trigram support still get sensible ordering.
        # (Lowercase both sides; SQLite's LOWER is portable.)
        from sqlalchemy import case, func

        prefix = term.lower() + "%"
        title_score = case((func.lower(Track.title).like(prefix), 0), else_=1)
        artist_score = case((func.lower(Track.artist).like(prefix), 0), else_=1)
        query = query.order_by(title_score, artist_score, Track.title)
    else:
        query = query.order_by(Track.updated_at.desc())
    # Over-fetch, then collapse duplicate recordings, then trim to `limit`.
    # The library carries 145 redundant copies (see
    # docs/proposals/library-duplicates.md), so searching "navi" returned the
    # same recording three times — three of the palette's eight slots on one
    # song. Over-fetching keeps the list `limit` deep after the collapse.
    rows = query.limit(limit * _DEDUPE_OVERFETCH).all()
    tracks = _versions_together(session, dedupe_track_rows(rows)[:limit])
    return [track_to_out(session, t) for t in tracks]


def _versions_together(session: Session, tracks: list[Track]) -> list[Track]:
    versions = version_index(session)
    family_of = {int(track.id): versions.family_id(int(track.id)) for track in tracks}
    first_place: dict[int, int] = {}
    for place, track in enumerate(tracks):
        family = family_of[int(track.id)]
        first_place.setdefault(family if family is not None else -int(track.id), place)

    def place_of(track: Track) -> int:
        family = family_of[int(track.id)]
        return first_place[family if family is not None else -int(track.id)]

    return sorted(tracks, key=place_of)


@router.get("", response_model=list[TrackOut])
def list_tracks(
    session: Session = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    bpm_min: float | None = Query(None),
    bpm_max: float | None = Query(None),
    key: str | None = Query(None, description="Camelot key, e.g. '8A'"),
    energy: int | None = Query(None, description="Exact floor_energy"),
    state: str | None = Query(None),
) -> list[dict]:
    needs_analysis = any(v is not None for v in (bpm_min, bpm_max, key, energy))

    q = session.query(Track)
    if state is not None:
        q = q.filter(Track.state == state)

    if needs_analysis:
        q = q.join(
            AudioAnalysis,
            (AudioAnalysis.track_id == Track.id) & (AudioAnalysis.stem_file_id.is_(None)),
        )
        if bpm_min is not None:
            q = q.filter(AudioAnalysis.bpm >= bpm_min)
        if bpm_max is not None:
            q = q.filter(AudioAnalysis.bpm <= bpm_max)
        if key is not None:
            q = q.filter(AudioAnalysis.key_camelot == key)
        if energy is not None:
            q = q.filter(AudioAnalysis.floor_energy == energy)

    tracks = q.order_by(Track.id).offset(offset).limit(limit).all()
    return [track_to_out(session, t) for t in tracks]


@router.get("/{track_id}", response_model=TrackOut)
def get_track(
    track_id: int,
    session: Session = Depends(get_session),
) -> dict:
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")
    return track_to_out(session, track)


@router.get("/{track_id}/versions", response_model=list[TrackVersionOut])
def list_versions(
    track_id: int,
    session: Session = Depends(get_session),
) -> list[dict]:
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")
    stands_for = int(track.duplicate_of) if track.duplicate_of is not None else track_id
    out: list[dict] = []
    for version in version_index(session).versions_of(stands_for):
        other = session.get(Track, version.track_id)
        analysis = fullmix_analysis(session, version.track_id)
        out.append(
            {
                "track_id": version.track_id,
                "title": other.title if other else None,
                "artist": other.artist if other else None,
                "duration_seconds": other.duration_seconds if other else None,
                "bpm": analysis.bpm if analysis else None,
                "key_camelot": analysis.key_camelot if analysis else None,
                "floor_energy": analysis.floor_energy if analysis else None,
                "label": version.label,
            }
        )
    return out


@router.get("/{track_id}/regions", response_model=list[RegionOut])
def list_regions(
    track_id: int,
    session: Session = Depends(get_session),
    region_type: str | None = Query(None),
    stem_file_id: int | None = Query(None),
) -> list[Region]:
    if session.get(Track, track_id) is None:
        raise HTTPException(status_code=404, detail="track not found")

    q = session.query(Region).filter(Region.track_id == track_id)
    if region_type is not None:
        q = q.filter(Region.region_type == region_type)
    if stem_file_id is not None:
        q = q.filter(Region.stem_file_id == stem_file_id)
    return q.order_by(Region.position_ms).all()


def _track_or_404(session: Session, track_id: int) -> Track:
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")
    return track


def _pad_targets(session: Session, track_id: int, role: str) -> list[int | None]:
    stem_ids = stem_ids_by_role(session, track_id)
    if role == _EVERY_ROLE:
        return list(stem_ids.values())
    if role not in stem_ids:
        raise HTTPException(status_code=404, detail=f"this track has no {role!r} pads")
    return [stem_ids[role]]


def _sections(session: Session, track_id: int) -> list[Section]:
    phrases = (
        session.query(
            Phrase.start_ms, Phrase.end_ms, Phrase.bar_count, Phrase.phrase_type, Phrase.energy_level
        )
        .filter(Phrase.track_id == track_id)
        .order_by(Phrase.start_ms)
    )
    return [section_from_phrase(*phrase) for phrase in phrases]


def _pad_rows(session: Session, track_id: int):
    return session.query(Region).filter(
        Region.track_id == track_id, Region.region_type == RegionType.PAD.value
    )


def _delete_pad(session: Session, track_id: int, stem_file_id: int | None, slot: int) -> None:
    same_stem = (
        Region.stem_file_id.is_(None) if stem_file_id is None else Region.stem_file_id == stem_file_id
    )
    _pad_rows(session, track_id).filter(Region.slot == slot, same_stem).delete(
        synchronize_session=False
    )


def _pads_out(session: Session, track_id: int) -> list[PadOut]:
    return [PadOut(**asdict(pad)) for pad in load_track_pads(session, track_id)]


@router.get("/{track_id}/pads", response_model=list[PadOut])
def list_pads(track_id: int, session: Session = Depends(get_session)) -> list[PadOut]:
    _track_or_404(session, track_id)
    return _pads_out(session, track_id)


@router.put("/{track_id}/pads/{role}/{slot}", response_model=PadOut)
def move_pad(
    track_id: int,
    role: str,
    body: PadMove,
    slot: int = PathParam(ge=1, le=PAD_COUNT),
    session: Session = Depends(get_session),
) -> PadOut:
    _track_or_404(session, track_id)
    targets = _pad_targets(session, track_id, role)
    bpm = track_bpm(session, track_id)
    downbeats = [
        position_ms
        for (position_ms,) in session.query(Beat.position_ms).filter(
            Beat.track_id == track_id, Beat.downbeat.is_(True)
        )
    ]
    if bpm <= 0 or not downbeats:
        raise HTTPException(status_code=409, detail="this track has no beat grid yet")
    position_ms = snap_to_bar(body.position_ms, downbeats)
    section = section_containing(position_ms, _sections(session, track_id))
    if section is None or abs(position_ms - body.position_ms) > bar_ms(bpm):
        raise HTTPException(status_code=422, detail="position is outside the song")

    pick = pad_at(slot, position_ms, section, bpm)
    for stem_file_id in targets:
        _delete_pad(session, track_id, stem_file_id, slot)
        session.add(pad_region(track_id, stem_file_id, pick, RegionSource.MANUAL))
    session.commit()
    return next(
        pad
        for pad in _pads_out(session, track_id)
        if pad.stem_file_id == targets[0] and pad.slot == slot
    )


@router.delete("/{track_id}/pads/{role}/{slot}", response_model=list[PadOut])
def reset_pad(
    track_id: int,
    role: str,
    slot: int = PathParam(ge=1, le=PAD_COUNT),
    session: Session = Depends(get_session),
) -> list[PadOut]:
    _track_or_404(session, track_id)
    targets = _pad_targets(session, track_id, role)
    picks = pick_pads(_sections(session, track_id), track_bpm(session, track_id))
    app_pick = next((pick for pick in picks if pick.slot == slot), None)
    for stem_file_id in targets:
        _delete_pad(session, track_id, stem_file_id, slot)
        if app_pick is not None:
            session.add(pad_region(track_id, stem_file_id, app_pick, RegionSource.AUTO))
    session.commit()
    return _pads_out(session, track_id)


@router.post("/{track_id}/tag", response_model=TrackOut)
def tag_track(
    track_id: int,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    deep: bool = Query(False, description="Use the Qwen2-Audio deep tagger (slow)"),
) -> dict:
    """Run the local tagger on a single track.

    Default mode (``deep=false``) uses CLAP zero-shot: ranks a controlled
    vocabulary against the track's audio embedding. Fast, no extra weights.

    Deep mode (``deep=true``) uses Qwen2-Audio: listens to the audio and
    generates free-form tags. Slow, needs ~8 GB of model weights downloaded
    the first time. Requires ``deep_tagger_enabled = true`` in settings.

    Replaces this track's existing tags from the chosen tagger's source
    (``inferred`` for CLAP, ``llm`` for Qwen). Manual tags are preserved.
    """
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")

    from dance.llm import ClapZeroShotTagger, Qwen2AudioTagger

    if deep:
        if not settings.deep_tagger_enabled:
            raise HTTPException(status_code=503, detail="deep tagger disabled in settings")
        tagger = Qwen2AudioTagger(settings)
    else:
        if not settings.tagger_enabled:
            raise HTTPException(status_code=503, detail="tagger disabled in settings")
        tagger = ClapZeroShotTagger(settings)

    try:
        tagger.tag_track(session, track)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tagger crashed on track %s", track_id)
        raise HTTPException(status_code=502, detail=f"tagger failed: {exc}") from exc

    return track_to_out(session, track)


@router.post("/{track_id}/als", response_model=AlsExportResult)
def export_als(
    track_id: int,
    body: AlsExportRequest | None = None,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AlsExportResult:
    """Generate an Ableton Live Set (.als) for a single track.

    The Set contains:
      - 4 audio tracks (drums/bass/vocals/other) pre-pointed at the stems
      - 1 reference "mix" track (muted by default) pointing at the full mix
      - Locators at every track-level Region
      - Master tempo set to the track's BPM
      - Track colors matching the kind palette

    ``out_path`` is optional. If given, it must live inside
    ``settings.als_output_dir`` (otherwise 403). If absent, the generator
    picks a default file name in that directory.
    """
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")

    gen = AlsGenerator(session, settings)

    target: Path | None = None
    if body is not None and body.out_path:
        target = Path(body.out_path)

    try:
        written = gen.write(track, target)
    except AlsOutsideDirError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AlsExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Quick stats for the response. We re-derive these instead of plumbing
    # them out of the generator, to keep the generator's signature simple.
    stem_count = session.query(StemFile).filter(StemFile.track_id == track.id).count()
    region_count = (
        session.query(Region)
        .filter(Region.track_id == track.id, Region.stem_file_id.is_(None))
        .count()
    )

    return AlsExportResult(
        ok=True,
        out_path=str(written),
        size_bytes=written.stat().st_size,
        track_count=stem_count + 1,  # stems + mix
        locator_count=region_count,
    )


@router.delete("/{track_id}")
def delete_track(
    track_id: int,
    session: Session = Depends(get_session),
) -> Response:
    """Remove a track and everything that references it.

    Cascades cover ``stem_files``, ``regions``, ``embeddings``, ``tags``,
    ``analysis``, ``beats``, ``phrases`` (declared on the Track ORM).
    SessionPlay and TrackEdge don't cascade — they get explicit deletes
    here so the FK constraints don't fire.

    Files on disk are NOT touched. The audio file in ``library_dir`` and
    any stems under ``stems_dir/<hash>/`` are left in place; if you re-
    ingest the same file later it'll come back with a fresh track_id.
    Stems orphaned by deletes can be swept manually — that's a deliberate
    trade-off so a misclick doesn't destroy audio.
    """
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")

    # Non-cascading refs first.
    session.query(SessionPlay).filter(SessionPlay.track_id == track_id).delete(
        synchronize_session=False
    )
    session.query(TrackEdge).filter(
        (TrackEdge.from_track_id == track_id) | (TrackEdge.to_track_id == track_id)
    ).delete(synchronize_session=False)

    session.delete(track)
    session.commit()
    return Response(status_code=204)


@router.get("/{track_id}/stems", response_model=list[StemFileOut])
def list_stems(
    track_id: int,
    session: Session = Depends(get_session),
) -> list[dict]:
    if session.get(Track, track_id) is None:
        raise HTTPException(status_code=404, detail="track not found")

    stems = (
        session.query(StemFile).filter(StemFile.track_id == track_id).order_by(StemFile.kind).all()
    )
    out: list[dict] = []
    for s in stems:
        # Per-stem analysis (one row max via partial unique index).
        analysis = (
            session.query(AudioAnalysis).filter(AudioAnalysis.stem_file_id == s.id).one_or_none()
        )
        out.append(
            {
                "id": int(s.id),
                "kind": s.kind,
                "path": s.path,
                "analysis": analysis,
            }
        )
    return out


def _compute_waveform_response(audio_path_str: str, num_peaks: int) -> WaveformOut:
    """Shared body for both waveform endpoints. Path string → response model.

    Returns 404 when the file isn't on disk and 503 when librosa blows up
    (corrupt audio, codec failure)."""
    path = Path(audio_path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"audio file missing: {audio_path_str}")

    try:
        peaks, duration_seconds = get_or_compute_peaks(path, num_peaks=num_peaks)
    except Exception as exc:  # noqa: BLE001 — librosa/soundfile failure surface
        logger.exception("Failed to compute waveform peaks for %s", audio_path_str)
        raise HTTPException(status_code=503, detail=f"waveform compute failed: {exc}") from exc

    return WaveformOut(peaks=peaks, duration_seconds=duration_seconds, num_peaks=num_peaks)


@router.get("/{track_id}/waveform", response_model=WaveformOut)
def get_track_waveform(
    track_id: int,
    session: Session = Depends(get_session),
    num_peaks: int = Query(
        200,
        ge=_MIN_NUM_PEAKS,
        le=_MAX_NUM_PEAKS,
        description="How many amplitude windows to bucket the file into.",
    ),
) -> WaveformOut:
    """Decimated amplitude envelope for a track's full mix.

    Reads ``Track.file_path``. Result is cached as a sidecar JSON next to
    the audio so subsequent calls are near-instant.
    """
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")
    if not track.file_path:
        raise HTTPException(status_code=404, detail="track has no file_path on disk")
    return _compute_waveform_response(track.file_path, num_peaks)


@stems_router.get("/{stem_file_id}/waveform", response_model=WaveformOut)
def get_stem_waveform(
    stem_file_id: int,
    session: Session = Depends(get_session),
    num_peaks: int = Query(
        200,
        ge=_MIN_NUM_PEAKS,
        le=_MAX_NUM_PEAKS,
        description="How many amplitude windows to bucket the file into.",
    ),
) -> WaveformOut:
    """Decimated amplitude envelope for a single stem file (drums/bass/...)."""
    stem = session.get(StemFile, stem_file_id)
    if stem is None:
        raise HTTPException(status_code=404, detail="stem not found")
    if not stem.path:
        raise HTTPException(status_code=404, detail="stem has no path on disk")
    return _compute_waveform_response(stem.path, num_peaks)
