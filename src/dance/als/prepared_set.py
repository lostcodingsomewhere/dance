from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from dance.als.generator import AlsExportError, AlsGenerator, _safe_filename
from dance.als.writer import PlacedClip, PreparedSetSpec, StemEntry, build_prepared_set_xml
from dance.config import Settings
from dance.core.database import Set, StemFile, StemKind, Track, TrackState
from dance.core.set_queues import parse_plan
from dance.pipeline.utils.pads import (
    BEATS_PER_BAR,
    MIX_ROLE,
    PAD_COUNT,
    TrackPad,
    load_track_pads,
    pad_start_beats,
    track_bpm,
)

logger = logging.getLogger(__name__)

PADS_PER_PAGE = PAD_COUNT
SONG_ROLE = "song"
STEM_ROLES = tuple(kind.value for kind in StemKind)
SIDES = ("a", "b")
LOOPING_ROLES = frozenset({StemKind.DRUMS.value, StemKind.BASS.value, StemKind.OTHER.value})

# Indices into Live's 70-colour clip palette. Not yet checked on the APC40's pads.
SECTION_COLOR_INDEX = {"intro": 24, "outro": 24, "buildup": 17, "drop": 14, "breakdown": 22}
OTHER_SECTION_COLOR_INDEX = 27
SILENT_PAD_COLOR_OFFSET = 42


@dataclass(frozen=True)
class DeckLoad:
    page: int
    side: str
    track_id: int
    roles: tuple[str, ...]
    plan_role: str


@dataclass(frozen=True)
class SkippedPick:
    track_id: int
    role: str
    reason: str


@dataclass(frozen=True)
class SetExport:
    path: Path
    size_bytes: int
    page_count: int
    scene_count: int
    clip_count: int
    skipped: list[SkippedPick]


def plan_loads(queues: dict[str, list[int]]) -> list[DeckLoad]:
    songs = queues.get(SONG_ROLE, [])
    loads = [
        DeckLoad(
            page=index // len(SIDES),
            side=SIDES[index % len(SIDES)],
            track_id=track_id,
            roles=(*STEM_ROLES, MIX_ROLE),
            plan_role=SONG_ROLE,
        )
        for index, track_id in enumerate(songs)
    ]
    first_spare_page = (len(songs) + 1) // len(SIDES)
    for role in STEM_ROLES:
        for index, track_id in enumerate(queues.get(role, [])):
            loads.append(
                DeckLoad(
                    page=first_spare_page + index // len(SIDES),
                    side=SIDES[index % len(SIDES)],
                    track_id=track_id,
                    roles=(role,),
                    plan_role=role,
                )
            )
    return loads


def pad_color_index(pad: TrackPad) -> int:
    color = SECTION_COLOR_INDEX.get(pad.section_label or "", OTHER_SECTION_COLOR_INDEX)
    return color + SILENT_PAD_COLOR_OFFSET if pad.silent else color


def _audio_paths_by_role(session: Session, track: Track) -> dict[str, Path]:
    paths = {MIX_ROLE: Path(track.file_path)}
    for kind, path in session.query(StemFile.kind, StemFile.path).filter(
        StemFile.track_id == track.id
    ):
        paths[kind] = Path(path)
    return paths


def _clips_for_load(
    session: Session, load: DeckLoad, track: Track, bpm: float
) -> tuple[list[PlacedClip], list[SkippedPick]]:
    paths = _audio_paths_by_role(session, track)
    pads_by_role: dict[str, list[TrackPad]] = {}
    for pad in load_track_pads(session, track.id):
        pads_by_role.setdefault(pad.role, []).append(pad)

    clips: list[PlacedClip] = []
    skipped: list[SkippedPick] = []
    title = track.title or track.file_name or f"Track {track.id}"
    for role in load.roles:
        if role not in paths:
            skipped.append(SkippedPick(track.id, role, "no stem file for this role"))
            continue
        if not pads_by_role.get(role):
            skipped.append(
                SkippedPick(track.id, role, "no pads; run `dance reprocess --from detect_regions`")
            )
            continue
        entry = StemEntry(kind=role, path=paths[role], duration_seconds=track.duration_seconds)
        for pad in pads_by_role[role]:
            loop_beats = (
                float((pad.length_bars or 0) * BEATS_PER_BAR) if role in LOOPING_ROLES else None
            )
            clips.append(
                PlacedClip(
                    deck_kind=f"{role}_{load.side}",
                    scene=load.page * PADS_PER_PAGE + pad.slot - 1,
                    entry=entry,
                    bpm=bpm,
                    name=f"{title} ({role} {load.side.upper()})",
                    start_beats=pad_start_beats(pad.position_ms, bpm),
                    loop_beats=loop_beats or None,
                    color_index=pad_color_index(pad),
                    muted=role == MIX_ROLE,
                )
            )
    return clips, skipped


def _scene_names(loads: list[DeckLoad], titles: dict[int, str], page_count: int) -> list[str]:
    names = [""] * (page_count * PADS_PER_PAGE)
    spare_number = 0
    for page in range(page_count):
        on_page = [load for load in loads if load.page == page]
        songs = [load for load in on_page if load.plan_role == SONG_ROLE]
        if songs:
            names[page * PADS_PER_PAGE] = " | ".join(
                f"{load.side.upper()}: {titles[load.track_id]}" for load in songs
            )
        elif on_page:
            spare_number += 1
            names[page * PADS_PER_PAGE] = f"Spare stems {spare_number}"
    return names


def _default_path(directory: Path, set_name: str, new_file: bool) -> Path:
    stem = _safe_filename(set_name)
    if not new_file:
        return directory / f"{stem}.als"
    # Live saves an open Set over its own file when asked, so that file is never rewritten.
    stamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    unused = directory / f"{stem} {stamp}.als"
    copy = 2
    while unused.exists():
        unused = directory / f"{stem} {stamp} ({copy}).als"
        copy += 1
    return unused


def export_set(
    session: Session,
    settings: Settings,
    set_row: Set,
    out_path: Path | None = None,
    new_file: bool = False,
) -> SetExport:
    loads = plan_loads(parse_plan(set_row.plan))
    if not loads:
        raise AlsExportError(f"set {set_row.id} has an empty plan")
    target = AlsGenerator(session, settings)._validate_in_output_dir(
        out_path or _default_path(settings.als_output_dir, set_row.name, new_file)
    )

    clips: list[PlacedClip] = []
    skipped: list[SkippedPick] = []
    placed: list[DeckLoad] = []
    titles: dict[int, str] = {}
    tempo: float | None = None
    for load in loads:
        track = session.get(Track, load.track_id)
        if track is None:
            skipped.append(SkippedPick(load.track_id, load.plan_role, "track not found"))
            continue
        bpm = track_bpm(session, track.id)
        if track.state != TrackState.COMPLETE.value or bpm <= 0:
            skipped.append(
                SkippedPick(track.id, load.plan_role, f"track is {track.state!r}, need 'complete'")
            )
            continue
        load_clips, load_skipped = _clips_for_load(session, load, track, bpm)
        skipped.extend(load_skipped)
        if not load_clips:
            continue
        clips.extend(load_clips)
        placed.append(load)
        titles[track.id] = track.title or track.file_name or f"Track {track.id}"
        tempo = tempo or bpm
    if not clips or tempo is None:
        raise AlsExportError(f"nothing in set {set_row.id} can be exported yet")

    page_count = max(load.page for load in placed) + 1
    spec = PreparedSetSpec(
        name=set_row.name,
        bpm=tempo,
        scene_names=_scene_names(placed, titles, page_count),
        clips=clips,
        set_dir=target.parent,
    )
    gz_bytes = gzip.compress(build_prepared_set_xml(spec), compresslevel=6)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gz_bytes)
    logger.info("wrote prepared set %s to %s (%d clips)", set_row.id, target, len(clips))
    return SetExport(
        path=target,
        size_bytes=len(gz_bytes),
        page_count=page_count,
        scene_count=page_count * PADS_PER_PAGE,
        clip_count=len(clips),
        skipped=skipped,
    )
