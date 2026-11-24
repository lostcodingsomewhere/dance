from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from dance.core.database import StemKind
from dance.pipeline.utils.pads import (
    BEATS_PER_BAR,
    PAD_COUNT,
    TrackPad,
    load_track_pads,
    pad_start_beats,
)

PADS_PER_PAGE = PAD_COUNT
LOOPING_ROLES = frozenset({StemKind.DRUMS.value, StemKind.BASS.value, StemKind.OTHER.value})

# Indices into Live's 70-colour clip palette. Not yet checked on the APC40's pads.
SECTION_COLOR_INDEX = {"intro": 24, "outro": 24, "buildup": 17, "drop": 14, "breakdown": 22}
OTHER_SECTION_COLOR_INDEX = 27
SILENT_PAD_COLOR_OFFSET = 42


@dataclass(frozen=True)
class CueClip:
    slot: int
    start_beats: float
    loop_beats: float | None
    color_index: int | None


WHOLE_SONG_FROM_THE_TOP = CueClip(slot=1, start_beats=0.0, loop_beats=None, color_index=None)


def pad_color_index(pad: TrackPad) -> int:
    color = SECTION_COLOR_INDEX.get(pad.section_label or "", OTHER_SECTION_COLOR_INDEX)
    return color + SILENT_PAD_COLOR_OFFSET if pad.silent else color


def cue_clip(pad: TrackPad, bpm: float) -> CueClip:
    loop_beats = float((pad.length_bars or 0) * BEATS_PER_BAR) if pad.role in LOOPING_ROLES else 0.0
    return CueClip(
        slot=pad.slot,
        start_beats=pad_start_beats(pad.position_ms, bpm),
        loop_beats=loop_beats or None,
        color_index=pad_color_index(pad),
    )


def cue_clips_by_role(session: Session, track_id: int, bpm: float) -> dict[str, list[CueClip]]:
    by_role: dict[str, list[CueClip]] = {}
    for pad in load_track_pads(session, track_id):
        by_role.setdefault(pad.role, []).append(cue_clip(pad, bpm))
    return by_role
