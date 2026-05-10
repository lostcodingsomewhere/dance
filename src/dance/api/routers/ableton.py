"""Ableton OSC passthrough endpoints.

Each command is fire-and-forget — we don't wait for AbletonOSC to acknowledge.
Live's actual state arrives asynchronously via the bridge listener.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from dance.api.deps import get_bridge, get_session
from dance.api.schemas import (
    AbletonStateOut,
    FireClipRequest,
    LoadTrackRequest,
    LoadTrackResult,
    TempoRequest,
    VolumeRequest,
)
from dance.core.database import StemFile, Track
from dance.osc.bridge import AbletonBridge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ableton", tags=["ableton"])


@router.post("/play")
def play(bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    bridge.client.play()
    return {"ok": True}


@router.post("/stop")
def stop(bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    bridge.client.stop()
    return {"ok": True}


@router.post("/tempo")
def tempo(
    body: TempoRequest,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    bridge.client.set_tempo(body.bpm)
    return {"ok": True}


@router.post("/fire")
def fire(
    body: FireClipRequest,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    bridge.client.fire_clip(body.track, body.scene)
    return {"ok": True}


@router.post("/volume")
def volume(
    body: VolumeRequest,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    bridge.client.set_track_volume(body.track, body.volume)
    return {"ok": True}


@router.get("/state", response_model=AbletonStateOut)
def state(bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    return bridge.state.to_dict()


@router.post("/load-track", response_model=LoadTrackResult)
def load_track(
    body: LoadTrackRequest,
    bridge: AbletonBridge = Depends(get_bridge),
    session: Session = Depends(get_session),
) -> LoadTrackResult:
    """Push a track + its stems into Live as empty audio tracks.

    AbletonOSC doesn't support loading audio files into clip slots
    programmatically (Live's Python API lacks the hook). This endpoint
    therefore does the best it can over OSC — appends one named, colored
    audio track for the full mix plus one per stem kind — and returns the
    indices so the React UI can tell the user where to drag the files.
    """
    track = session.query(Track).filter(Track.id == body.track_id).one_or_none()
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")

    stems: list[StemFile] = []
    if body.include_stems:
        stems = (
            session.query(StemFile)
            .filter(StemFile.track_id == body.track_id)
            .all()
        )

    try:
        result = bridge.push_track_to_live(
            track, stems, include_stems=body.include_stems
        )
    except OSError as exc:
        logger.warning("OSC send failed during load-track: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Ableton Live over OSC: {exc}",
        ) from exc

    title = track.title or track.file_name or f"Track {track.id}"
    n = len(result["track_indices"])
    if n == 0:
        message = (
            f"No tracks created in Live for {title!r} — check warnings."
        )
    else:
        message = (
            f"Created {n} audio track(s) in Live for {title!r}. "
            f"Drag the stems onto scene {result['scene_index'] + 1}."
        )

    return LoadTrackResult(
        ok=True,
        scene_index=result["scene_index"],
        track_indices=result["track_indices"],
        message=message,
        warnings=result.get("warnings", []),
    )
