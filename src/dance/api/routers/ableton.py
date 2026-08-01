"""Ableton OSC passthrough endpoints.

Each command is fire-and-forget — we don't wait for AbletonOSC to acknowledge.
Live's actual state arrives asynchronously via the bridge listener.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from dance.api.deps import get_bridge, get_session
from dance.api.schemas import (
    AbletonStateOut,
    DeckCellOut,
    DeckMapOut,
    FireClipRequest,
    LoadTrackRequest,
    LoadTrackResult,
    PreviewRequest,
    PreviewResult,
    CrossfaderRequest,
    TempoRequest,
    VolumeRequest,
    WarpCheckResult,
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


@router.post("/transport/fire-deck/{side}/{scene_index}")
def fire_deck(
    side: str,
    scene_index: int,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Fire all 5 cells (4 stems + mix) on Deck ``side`` (``a`` or
    ``b``) at the given scene. Per-deck play."""
    if side not in ("a", "b"):
        raise HTTPException(
            status_code=400,
            detail=f"side must be 'a' or 'b', got {side!r}",
        )
    return bridge.fire_deck(side, scene_index)


@router.post("/transport/stop-deck/{side}")
def stop_deck(side: str, bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Stop every clip on Deck ``side``. Master transport unaffected."""
    if side not in ("a", "b"):
        raise HTTPException(
            status_code=400,
            detail=f"side must be 'a' or 'b', got {side!r}",
        )
    return bridge.stop_deck(side)


@router.post("/fx/filter/{side}")
def fx_filter(side: str, bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Toggle the parallel filter (HPF) send for Deck A or B."""
    if side not in ("a", "b"):
        raise HTTPException(400, f"side must be 'a' or 'b', got {side!r}")
    return bridge.toggle_filter(side)


@router.post("/fx/reverb/{side}")
def fx_reverb(side: str, bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Toggle the parallel Reverb send for Deck A or B (throw effect)."""
    if side not in ("a", "b"):
        raise HTTPException(400, f"side must be 'a' or 'b', got {side!r}")
    return bridge.toggle_reverb(side)


@router.post("/fx/delay/{side}")
def fx_delay(side: str, bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Toggle the parallel Delay send for Deck A or B (echo throw)."""
    if side not in ("a", "b"):
        raise HTTPException(400, f"side must be 'a' or 'b', got {side!r}")
    return bridge.toggle_delay(side)


@router.post("/fx/{fx_name}/{side}/sweep")
def fx_sweep(
    fx_name: str,
    side: str,
    bars: float = 4.0,
    direction: str = "auto",
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Smooth FX send ramp over ``bars`` bars (Phase 3).

    ``direction`` ∈ {``in`` (ramp to 1), ``out`` (ramp to 0), ``auto``
    (toggle based on current state)}. ``bars`` is interpreted at the
    current master tempo.
    """
    if side not in ("a", "b"):
        raise HTTPException(400, f"side must be 'a' or 'b', got {side!r}")
    if fx_name not in ("filter", "reverb", "delay"):
        raise HTTPException(400, f"fx must be filter|reverb|delay, got {fx_name!r}")
    state = bridge._state_dict_for(fx_name) or {}
    if direction == "in":
        target = 1.0
    elif direction == "out":
        target = 0.0
    elif direction == "auto":
        target = 0.0 if state.get(side, False) else 1.0
    else:
        raise HTTPException(400, f"direction must be in|out|auto, got {direction!r}")
    return bridge.ramp_fx_send(
        side=side,
        return_name=fx_name.capitalize(),
        target_level=target,
        duration_bars=max(0.0, min(32.0, float(bars))),
    )


@router.post("/fx/{name}")
def fx_fire(name: str, bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Fire a named one-shot FX clip (e.g. ``riser``). Quantized to the
    next bar by the clip's LaunchQuantisation."""
    if not name or not name.replace("_", "").isalnum():
        raise HTTPException(400, f"FX name must be alphanumeric, got {name!r}")
    if name in ("filter", "reverb", "delay"):
        raise HTTPException(
            400,
            f"{name!r} is an FX return, not a one-shot. Use /fx/{name}/{{side}}.",
        )
    return bridge.fire_fx(name)


@router.get("/fx/state")
def fx_state(bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Current FX state — used by the UI to render toggle states."""
    state = bridge.fx_state()
    # Shape kept compatible with the previous /fx/state response so
    # existing UI code keeps working; expanded with reverb + delay.
    return {
        "filter": state["filter"],
        "reverb": state["reverb"],
        "delay": state["delay"],
        "discovered": {
            "returns": state["returns"],
            "scenes": state["scenes"],
        },
    }


@router.post("/pfl/{side}")
def set_pfl(side: str, bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Route Deck A, Deck B, or neither into the headphone cue.

    ``side`` ∈ {``a``, ``b``, ``off``}. Implemented via Live's per-track
    Solo with Solo/Cue mode = Cue (set on bridge init). See
    ``bridge.set_pfl_side`` for details.
    """
    if side not in ("a", "b", "off"):
        raise HTTPException(
            status_code=400,
            detail=f"side must be 'a', 'b', or 'off'; got {side!r}",
        )
    return bridge.set_pfl_side(None if side == "off" else side)


@router.post("/crossfader")
def set_crossfader(
    body: CrossfaderRequest,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Set Live's master crossfader. ``value`` in [-1, +1]: -1 = full A,
    0 = center, +1 = full B. Mostly driven from the APC40 hardware fader;
    we expose the setter so on-screen drag is also possible.
    """
    v = max(-1.0, min(1.0, float(body.value)))
    bridge.client.set_crossfader(v)
    return {"ok": True, "value": v}


@router.post("/record/{state}")
def record(state: str, bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Toggle Live's session record. ``state`` is ``on`` or ``off``.

    No bookkeeping beyond the OSC toggle — Live owns the actual capture.
    The FE drives elapsed-time ticking off the moment we returned ``ok``.
    """
    if state not in ("on", "off"):
        raise HTTPException(
            status_code=400,
            detail=f"record state must be 'on' or 'off', got {state!r}",
        )
    bridge.client.set_record_mode(state == "on")
    return {"ok": True, "recording": state == "on"}


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


# ---------------------------------------------------------------------------
# Transport — path-param helpers for the live-remixing UI. Each one routes to
# an existing OSC primitive on the bridge client; we keep them grouped here so
# the FE can call them with one verb + one path segment per intent.
# ---------------------------------------------------------------------------


@router.post("/transport/fire-scene/{scene_index}")
def fire_scene(
    scene_index: int,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Fire a whole scene (anchor mode — play the row's original combo)."""
    bridge.client.fire_scene(scene_index)
    return {"ok": True, "scene_index": scene_index}


@router.post("/transport/fire-clip/{track_index}/{slot_index}")
def fire_clip(
    track_index: int,
    slot_index: int,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Fire one cell (swap or layer a single stem into the active combo)."""
    bridge.client.fire_clip(track_index, slot_index)
    return {"ok": True, "track_index": track_index, "slot_index": slot_index}


@router.post("/transport/stop-track/{track_index}")
def stop_track(
    track_index: int,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Stop every clip on one track (clears one stem from the combo)."""
    bridge.client.stop_track(track_index)
    return {"ok": True, "track_index": track_index}


@router.post("/transport/stop-all")
def stop_all_clips(bridge: AbletonBridge = Depends(get_bridge)) -> dict:
    """Stop every playing clip without halting transport (combo panic)."""
    bridge.client.stop_all_clips()
    return {"ok": True}


@router.post("/transport/stop-cell/{track_index}/{slot_index}")
def stop_cell(
    track_index: int,
    slot_index: int,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Stop a single cell. Used by tap-to-stop on the SceneGrid — tapping
    a playing cell stops it, parallel to tapping a stopped cell to fire."""
    bridge.client.stop_clip(track_index, slot_index)
    return {"ok": True, "track_index": track_index, "slot_index": slot_index}


@router.post("/transport/stop-scene/{scene_index}")
def stop_scene(
    scene_index: int,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Stop every deck-column cell on a scene (anchor mode off-switch)."""
    return bridge.stop_scene(scene_index)


@router.post("/transport/seek/{track_index}/{slot_index}")
def seek_clip(
    track_index: int,
    slot_index: int,
    position: float,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Jump a clip's playback to ``position`` beats. Sets BOTH the clip's
    ``loop_start`` and ``start_marker`` to ``position``, forces the clip to
    instant launch (launch_quantization = None), and re-fires — so the jump
    is audible immediately, the clip plays from there, AND when the loop wraps
    it returns to the same point (not 0).

    Instant launch: without overriding the clip's launch quantization, the
    re-fire obeys Live's global 1-bar quantize, so the audible jump lags up to
    a bar and feels broken. Setting launch_quantization to None makes the
    seek snap immediately. Best-effort — a stock AbletonOSC fork without the
    handler just leaves the clip on global quant (≤1-bar lag).

    Why both markers: our stem clips have ``loop_start = 0, loop_end = clip
    length, loop_on = true``. Setting only ``start_marker`` lets the
    clip play once from ``position`` then wrap to 0 — visually "the
    loop resets." Bumping ``loop_start`` along with start_marker makes
    the clip loop indefinitely from the new position forward (the
    boundary moves with the click). ``loop_end`` is left at the clip's
    natural end.

    The FE sends a precise ``position`` (no section snap) — it's exactly
    where the user clicked on the Waveform of a playing cell.

    NOTE: this is persistent for the session — both markers move and
    stay there until another seek (or a reload). To "reset to top"
    you'd need to seek to 0 explicitly.
    """
    # Markers are quick property sets; they precede the fire here so the
    # re-fire reads the new loop_start / start_marker. launch_quantization is
    # set to 0 (None) BEFORE the fire so the jump is instant, not bar-quantized.
    bridge.client.set_clip_loop_start(track_index, slot_index, position)
    bridge.client.set_clip_start_marker(track_index, slot_index, position)
    bridge.client.set_clip_launch_quantization(track_index, slot_index, 0)
    bridge.client.fire_clip(track_index, slot_index)
    return {
        "ok": True,
        "track_index": track_index,
        "slot_index": slot_index,
        "position": position,
    }


@router.post("/transport/solo-track/{track_index}")
def solo_track(
    track_index: int,
    soloed: bool = True,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Toggle a track's solo state. With Live's solo mode set to Cue (PFL),
    the soloed track routes to outs 3/4 (headphones) without muting master
    — i.e. "iso listen, don't disrupt the mix." Use ``?soloed=false`` to
    clear the solo.

    Wraps ``/live/track/set/solo``."""
    bridge.client.set_track_solo(track_index, soloed)
    return {"ok": True, "track_index": track_index, "soloed": soloed}


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
    """Push a track + its stems into Live's clip grid over OSC.

    Stock AbletonOSC has no ``load_sample`` for audio clip slots, but
    we ship a patched fork that adds ``/live/clip_slot/create_audio_clip``
    (see ``docs/abletonosc_setup.md``). ``bridge.push_track_to_live``
    uses it to drop each stem's WAV directly into the matching deck
    column on a free scene — no Finder drag required.

    Returns the scene + deck-column indices so the React UI can light
    up the cells that were just populated. ``warnings`` carries any
    per-stem misses (missing file, Live unreachable, etc.) without
    failing the whole call.
    """
    track = session.query(Track).filter(Track.id == body.track_id).one_or_none()
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")

    stems: list[StemFile] = []
    if body.include_stems:
        stems = session.query(StemFile).filter(StemFile.track_id == body.track_id).all()

    if body.side is not None and body.side not in ("a", "b"):
        raise HTTPException(
            status_code=400,
            detail=f"side must be 'a' or 'b' (or omitted to auto-pick), got {body.side!r}",
        )
    # How many beats the track should occupy once warped, from OUR analysis.
    # Feeds the bridge's post-load warp check as a tie-breaker only — Live's
    # reading of the audio outranks ours (CLAUDE.md rule 3), so a missing or
    # wrong value can never manufacture a warning on its own.
    expected_beats: float | None = None
    analysis = (
        session.query(AudioAnalysis)
        .filter(
            AudioAnalysis.track_id == track.id,
            AudioAnalysis.stem_file_id.is_(None),
        )
        .one_or_none()
    )
    if analysis is not None and analysis.bpm and track.duration_seconds:
        expected_beats = float(track.duration_seconds) * float(analysis.bpm) / 60.0

    try:
        result = bridge.push_track_to_live(
            track,
            stems,
            include_stems=body.include_stems,
            scene_index=body.scene_index,
            kinds=body.kinds,
            side=body.side,
            expected_beats=expected_beats,
        )
    except OSError as exc:
        logger.warning("OSC send failed during load-track: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Ableton Live over OSC: {exc}",
        ) from exc

    title = track.title or track.file_name or f"Track {track.id}"
    stems_loaded = result.get("stems_loaded", 0)
    is_full_song = body.kinds is None and body.include_stems
    if is_full_song and stems_loaded == 4:
        message = (
            f"Loaded {title!r} into scene {result['scene_index'] + 1}. Fire the scene to play."
        )
    elif stems_loaded > 0:
        kinds_summary = ", ".join(body.kinds) if body.kinds else f"{stems_loaded}/4 stems"
        message = f"Loaded {kinds_summary} from {title!r} on scene {result['scene_index'] + 1}."
    else:
        message = f"Could not load {title!r} — check warnings."

    return LoadTrackResult(
        ok=True,
        scene_index=result["scene_index"],
        side=result.get("side"),
        track_indices=result["track_indices"],
        stems_loaded=stems_loaded,
        message=message,
        warnings=result.get("warnings", []),
    )


@router.post("/warp-check/{scene_index}", response_model=WarpCheckResult)
def warp_check(
    scene_index: int,
    bridge: AbletonBridge = Depends(get_bridge),
    session: Session = Depends(get_session),
) -> WarpCheckResult:
    """Audit a loaded scene for Live auto-warp errors.

    Verified against Live 12.4.2 on this library: of 4 tracks loaded, every
    one had at least one stem warped to the wrong tempo (bass worst — 113 BPM
    read as 73 on one track). Nothing errors; the stem just drifts. This is
    the surface that tells the DJ it's the tool, not them.

    Call it ~15s after a load — Live reports a placeholder length until its
    background analysis lands, and the placeholder makes every stem agree.
    """
    if scene_index < 0:
        raise HTTPException(status_code=400, detail="scene_index must be >= 0")

    # Expected beats per source track on this scene, from our own analysis.
    # Tie-breaker only — see AbletonBridge._warp_reference.
    expected: dict[int, float] = {}
    durations: dict[int, float] = {}
    track_ids = {
        tid
        for (scene, kind), tid in bridge.get_deck_cells().items()
        if scene == scene_index and not kind.startswith("mix")
    }
    if track_ids:
        rows = (
            session.query(Track, AudioAnalysis)
            .join(AudioAnalysis, AudioAnalysis.track_id == Track.id)
            .filter(Track.id.in_(track_ids), AudioAnalysis.stem_file_id.is_(None))
            .all()
        )
        for track, analysis in rows:
            if track.duration_seconds:
                durations[int(track.id)] = float(track.duration_seconds)
                if analysis.bpm:
                    expected[int(track.id)] = (
                        float(track.duration_seconds) * float(analysis.bpm) / 60.0
                    )

    result = bridge.check_warp_at(
        scene_index,
        expected_beats_by_track=expected,
        duration_by_track=durations,
    )
    return WarpCheckResult(
        ok=True,
        scene_index=result["scene_index"],
        checked=result["checked"],
        warnings=result["warnings"],
    )


@router.get("/decks", response_model=DeckMapOut)
def deck_map(
    bridge: AbletonBridge = Depends(get_bridge),
    session: Session = Depends(get_session),
) -> DeckMapOut:
    """Current cell-level deck map plus per-cell source-track metadata.

    The companion polls this to render the SceneGrid + ComboStrip in sync
    with the bridge's view of Live. Cells in the same scene can come from
    different tracks — that's the whole point of the live-remixing model.
    """
    state = bridge.get_deck_state()
    raw_cells = state["cells"]
    # Cache per-track metadata so we don't query N times for the same
    # source track loaded across multiple cells.
    track_cache: dict[int, tuple[Track | None, AudioAnalysis | None]] = {}

    def _lookup(track_id: int) -> tuple[Track | None, AudioAnalysis | None]:
        if track_id in track_cache:
            return track_cache[track_id]
        track = session.query(Track).filter(Track.id == track_id).one_or_none()
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
        track_cache[track_id] = (track, analysis)
        return track, analysis

    # Per-cell stem_file_id: look up StemFile by (track_id, kind). Cache
    # by (track_id, kind) so repeated cells across scenes only query once.
    stem_cache: dict[tuple[int, str], int | None] = {}

    def _stem_id(track_id: int, kind: str) -> int | None:
        key = (track_id, kind)
        if key in stem_cache:
            return stem_cache[key]
        stem = (
            session.query(StemFile)
            .filter(StemFile.track_id == track_id, StemFile.kind == kind)
            .one_or_none()
        )
        stem_cache[key] = stem.id if stem else None
        return stem_cache[key]

    cells: list[DeckCellOut] = []
    for cell in raw_cells:
        track_id = cell["track_id"]
        track, analysis = _lookup(track_id)
        cells.append(
            DeckCellOut(
                scene_index=cell["scene_index"],
                kind=cell["kind"],
                track_id=track_id,
                stem_file_id=_stem_id(track_id, cell["kind"]),
                title=getattr(track, "title", None),
                artist=getattr(track, "artist", None),
                bpm=getattr(analysis, "bpm", None),
                key_camelot=getattr(analysis, "key_camelot", None),
                floor_energy=getattr(analysis, "floor_energy", None),
                duration_seconds=getattr(track, "duration_seconds", None),
            )
        )
    return DeckMapOut(columns=state["columns"], cells=cells)


@router.delete("/decks/cell/{track_index}/{slot_index}")
def delete_cell(
    track_index: int,
    slot_index: int,
    bridge: AbletonBridge = Depends(get_bridge),
) -> dict:
    """Remove a single stem-clip from one (track, slot) in the deck columns.

    Stops the clip first (in case it's playing), then deletes it from the
    slot via OSC, then drops the cell from the bridge's ``_deck_cells``
    cache and persists. The slot itself stays — it just becomes empty,
    ready to receive the next ``Load to Live`` for that kind.
    """
    return bridge.delete_cell(track_index, slot_index)


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


@router.post("/decks/resync")
def resync_decks(
    bridge: AbletonBridge = Depends(get_bridge),
    session: Session = Depends(get_session),
) -> dict:
    """Scan Live's deck-column clip slots and rebuild ``_deck_cells`` from
    what's actually there. Useful when the in-memory cell map drifts from
    Live's session view (typical case: backend restart without persistence,
    or user manually placed clips outside our Load endpoint).

    Workflow:
      1. ``bridge.scan_live_for_cells()`` walks each deck column × scene,
         queries the clip name, returns a list of populated cells.
      2. We parse each clip name (format we set on load: ``"{title} ({kind})"``)
         and resolve the title to a dance Track id via DB lookup.
      3. ``bridge.adopt_cells(new_map)`` replaces the in-memory cell map
         and persists.

    Returns ``{ok, scanned, adopted, unmatched: [...] }`` so the UI can
    surface "we found 12 clips, matched 10 to your library" — the user
    knows which clips couldn't be resolved.
    """
    scanned = bridge.scan_live_for_cells()
    adopted: dict[tuple[int, str], int] = {}
    unmatched: list[dict[str, Any]] = []

    for entry in scanned:
        clip_name: str = entry["clip_name"]
        kind: str = entry["kind"]  # deck-column kind, e.g. "drums_a", "mix_b"
        # Reverse the load-format set in bridge.push_track_to_live:
        # stems -> "{title} ({source} {side})" e.g. "Bad Memories (drums a)";
        # mix   -> "{title} (mix {side})"      e.g. "Bad Memories (mix a)".
        # The trailing parenthetical is "{source} {side}"; reconstruct the deck
        # kind as "{source}_{side}" and require it to match this cell's column.
        # (Also accept the legacy "({kind})" form where the paren already equals
        # the deck kind, e.g. "(drums_a)".)
        title = clip_name
        if clip_name.endswith(")") and "(" in clip_name:
            opener = clip_name.rfind("(")
            paren = clip_name[opener + 1 : -1].strip().lower()
            paren_kind = paren.replace(" ", "_")  # "drums a" -> "drums_a"
            if paren_kind == kind or paren == kind:
                title = clip_name[:opener].strip()
        track = session.query(Track).filter(Track.title == title).one_or_none()
        if track is None:
            unmatched.append({**entry, "tried_title": title})
            continue
        adopted[(entry["scene_index"], kind)] = int(track.id)

    bridge.adopt_cells(adopted)
    return {
        "ok": True,
        "scanned": len(scanned),
        "adopted": len(adopted),
        "unmatched": unmatched,
    }


# ---------------------------------------------------------------------------
# Cue / preview — audition a candidate in headphones without leaking to the
# master speakers. Routes through a dedicated "Cue" track in Live whose
# output is set to the 4i4's outs 3/4.
# ---------------------------------------------------------------------------


_STEM_COLUMNS = {"drums", "bass", "vocals", "other"}
_MIX_COLUMN = "mix"


def _resolve_preview_path(session: Session, track_id: int, column: str) -> str | None:
    """Look up the audio file on disk for a given preview request.

    For stem columns we return the matching StemFile.path; for the mix
    column the original Track.file_path (which is the full-mix audio).
    """
    if column == _MIX_COLUMN:
        track = session.query(Track).filter(Track.id == track_id).one_or_none()
        return track.file_path if track and track.file_path else None
    if column not in _STEM_COLUMNS:
        return None
    stem = (
        session.query(StemFile)
        .filter(StemFile.track_id == track_id, StemFile.kind == column)
        .one_or_none()
    )
    return stem.path if stem and stem.path else None


@router.post("/preview", response_model=PreviewResult)
def preview(
    body: PreviewRequest,
    bridge: AbletonBridge = Depends(get_bridge),
    session: Session = Depends(get_session),
) -> PreviewResult:
    """Start auditioning a candidate through the Cue track (headphones only).

    Resolves the audio file from the track + column, then drops it into
    Live's Cue track and fires it. Replaces any in-progress preview. The
    master speakers are untouched — the Cue track is permanently routed to
    outs 3/4 on the Scarlett 4i4.
    """
    if body.column != _MIX_COLUMN and body.column not in _STEM_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown column {body.column!r}; must be mix or one of {sorted(_STEM_COLUMNS)}",
        )

    audio_path = _resolve_preview_path(session, body.track_id, body.column)
    if audio_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No audio file for track {body.track_id} / column {body.column!r}",
        )

    track = session.query(Track).filter(Track.id == body.track_id).one_or_none()
    label = f"PREVIEW · {(track.title if track else None) or f'#{body.track_id}'} ({body.column})"

    try:
        result = bridge.preview_audio(audio_path, label=label)
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Ableton Live over OSC: {exc}",
        ) from exc

    return PreviewResult(**result)


@router.post("/preview/stop", response_model=PreviewResult)
def preview_stop(bridge: AbletonBridge = Depends(get_bridge)) -> PreviewResult:
    """Stop the in-progress preview and clear the Cue track's slot.

    Idempotent: safe to call when nothing is previewing.
    """
    result = bridge.stop_preview()
    return PreviewResult(ok=result.get("ok", True), warnings=[])
