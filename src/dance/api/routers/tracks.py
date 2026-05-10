"""Track read endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from dance.als import AlsGenerator
from dance.als.generator import AlsExportError, AlsOutsideDirError
from dance.api.deps import get_session, get_settings, track_to_out
from dance.api.schemas import (
    AlsExportRequest,
    AlsExportResult,
    RegionOut,
    StemFileOut,
    TrackOut,
)
from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    Region,
    StemFile,
    Track,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tracks", tags=["tracks"])


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
            (AudioAnalysis.track_id == Track.id)
            & (AudioAnalysis.stem_file_id.is_(None)),
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
    stem_count = (
        session.query(StemFile).filter(StemFile.track_id == track.id).count()
    )
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


@router.get("/{track_id}/stems", response_model=list[StemFileOut])
def list_stems(
    track_id: int,
    session: Session = Depends(get_session),
) -> list[dict]:
    if session.get(Track, track_id) is None:
        raise HTTPException(status_code=404, detail="track not found")

    stems = (
        session.query(StemFile)
        .filter(StemFile.track_id == track_id)
        .order_by(StemFile.kind)
        .all()
    )
    out: list[dict] = []
    for s in stems:
        # Per-stem analysis (one row max via partial unique index).
        analysis = (
            session.query(AudioAnalysis)
            .filter(AudioAnalysis.stem_file_id == s.id)
            .one_or_none()
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
