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
    DeckMapOut,
    DeckSceneOut,
    FireClipRequest,
    LoadTrackRequest,
    LoadTrackResult,
    TempoRequest,
    VolumeRequest,
)
from dance.core.database import AudioAnalysis, StemFile, Track
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
            track,
            stems,
            include_stems=body.include_stems,
            scene_index=body.scene_index if body.scene_index is not None else 0,
        )
    except OSError as exc:
        logger.warning("OSC send failed during load-track: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Ableton Live over OSC: {exc}",
        ) from exc

    title = track.title or track.file_name or f"Track {track.id}"
    stems_loaded = result.get("stems_loaded", 0)
    if stems_loaded == 4:
        message = (
            f"Loaded {title!r} into scene {result['scene_index'] + 1}. "
            f"Fire the scene to play."
        )
    elif stems_loaded > 0:
        message = (
            f"Partially loaded {title!r}: {stems_loaded}/4 stems on "
            f"scene {result['scene_index'] + 1}."
        )
    else:
        message = f"Could not auto-load stems for {title!r} — check warnings."

    return LoadTrackResult(
        ok=True,
        scene_index=result["scene_index"],
        track_indices=result["track_indices"],
        stems_loaded=stems_loaded,
        message=message,
        warnings=result.get("warnings", []),
    )


@router.get("/decks", response_model=DeckMapOut)
def deck_map(
    bridge: AbletonBridge = Depends(get_bridge),
    session: Session = Depends(get_session),
) -> DeckMapOut:
    """Current scene→track map plus track-name/BPM/key metadata for each.

    The companion app polls this to render its Scene Map widget — a
    distilled view of "what songs are loaded where in Live" — without
    having to keep its own localStorage state in sync with reality.
    """
    state = bridge.get_deck_state()
    scenes: list[DeckSceneOut] = []
    for scene_idx, track_id in sorted(state["scenes"].items()):
        track = session.query(Track).filter(Track.id == track_id).one_or_none()
        # AudioAnalysis is one-row-per-(track, stem) and Track.analysis is a
        # *list*; the full-mix analysis is the row where stem_file_id IS NULL.
        # That's the row whose BPM/key/energy a DJ cares about.
        analysis = (
            session.query(AudioAnalysis)
            .filter(
                AudioAnalysis.track_id == track_id,
                AudioAnalysis.stem_file_id.is_(None),
            )
            .one_or_none()
            if track is not None
            else None
        )
        scenes.append(
            DeckSceneOut(
                scene_index=scene_idx,
                track_id=track_id,
                title=getattr(track, "title", None),
                artist=getattr(track, "artist", None),
                bpm=getattr(analysis, "bpm", None),
                key_camelot=getattr(analysis, "key_camelot", None),
                floor_energy=getattr(analysis, "floor_energy", None),
            )
        )
    return DeckMapOut(columns=state["columns"], scenes=scenes)


@router.post("/decks/reset")
def reset_decks(bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Forget which Ableton tracks we used as deck columns and which scenes
    are staged. Does **not** delete anything in Live — the user is in
    charge of their session view. Use this when you've manually cleaned up
    Live's session and want the bridge to start fresh on the next Load."""
    bridge.reset_deck_columns()
    return {"ok": True}


@router.post("/decks/clean")
def clean_decks(bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Delete every ``Deck *`` track in Live (covers stragglers from prior
    runs) and reset the bridge cache. Use this when Live's session shows
    deck columns the companion no longer knows about — typically after a
    backend restart that lost its in-memory cache."""
    return {"ok": True, **bridge.clean_live_decks()}
