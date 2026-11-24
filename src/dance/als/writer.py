"""Template-based Ableton Live Set XML builder.

The ``.als`` format is gzipped, schemaless, and undocumented. Earlier
attempts to build a Set from scratch always failed: Live's loader checks
hundreds of required elements (Transport, GroovePool, MainTrack with its
Mixer/Tempo, AutomationLanes, ContentLanes, ScaleInformation, etc.) and
also enforces invariants like "MainTrack ClipSlotList size == Scenes
count". Reverse-engineering the full schema is not realistic.

Instead: we ship a real blank Live 12 Set as a template
(``templates/blank_live12.xml``, decompressed from a user-saved
``Untitled.als``), clone its sole ``AudioTrack`` as a DOM template, and
surgically inject our 5 stem tracks + clips + locators + tempo. All the
boilerplate Live demands stays intact because it's verbatim from a Set
Live itself produced.

What the writer changes:
- ``MainTrack/.../Tempo/Manual`` (and its automation FloatEvent) → BPM.
- ``Tracks``: remove default Midi/Audio tracks, insert 5 stem
  AudioTracks (drums/bass/vocals/other/mix), keep the two ReturnTracks.
- Each stem AudioTrack: set ``Id``, ``Name``, ``Color``, and inject an
  ``<AudioClip>`` into the first session ClipSlot pointing at the stem
  file.
- ``Locators``: replace inner content with one ``<Locator>`` per entry.

Color palette (Live built-in indices 0-69):
- drums  →  1  (red)
- bass   →  7  (orange)
- vocals → 13  (yellow)
- other  → 39  (blue)
- mix    → 25  (grey)
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

from lxml import etree

from dance.als.markers import LocatorEntry


# ---------------------------------------------------------------------------
# Public schema constants
# ---------------------------------------------------------------------------


# Captured from the bundled blank_live12.xml template.
ABLETON_ROOT_ATTRS = {
    "MajorVersion": "5",
    "MinorVersion": "12.0_12402",
    "SchemaChangeCount": "2",
    "Creator": "Ableton Live 12.4",
    "Revision": "d85a94ab5eb1299391a7ff9400bdb8c0dcd7cc75",
}


# Per-kind color index in Live's 70-entry palette. The palette layout
# is 7 cols x 10 rows (deduced empirically — Live 12.4). Edit to taste.
# Verified by opening a generated .als in Live:
#   0=pink/salmon  1=orange-red  2=brown  4=lime  10=light-blue
#   13=white-ish   25=magenta   39=light-purple   64=dark-blue
#
# Indexed by SOURCE stem kind. A and B deck channels share the same
# color so the eye groups them as "the same role" — tracks already
# distinguish A/B by name + crossfader assignment.
STEM_COLOR_INDEX: dict[str, int] = {
    "drums": 1,    # orange-red — strongest warm hue we observed
    "bass": 2,     # brown/mustard
    "vocals": 4,   # lime
    "other": 10,   # light blue
    "mix": 13,     # white — distinct from the 4 stems, signals "reference"
}


# Canonical track order — 8 stem decks grouped by deck (deck A's four
# roles, then deck B's) then per-deck mix references. Stems come first so
# APC40's default 8-strip view maps 1:1 to them, deck A on faders 1-4 and
# deck B on 5-8; mix_a and mix_b live at indices 8-9, off the default
# view. Mirror of dance.osc.bridge.AbletonBridge._DECK_KINDS — kept in
# sync by hand; the bridge tests catch drift on the runtime side.
DECK_ORDER: tuple[str, ...] = (
    "drums_a", "bass_a", "vocals_a", "other_a",
    "drums_b", "bass_b", "vocals_b", "other_b",
    "mix_a", "mix_b",
)

# Back-compat alias — old callers (and docs) referenced STEM_ORDER.
# Points at the new 9-entry shape; consumers that need the pre-pair
# source kinds should use SOURCE_STEM_KINDS instead.
STEM_ORDER = DECK_ORDER
SOURCE_STEM_KINDS: tuple[str, ...] = ("drums", "bass", "vocals", "other")


# Crossfader assignment per deck side. Live's CrossFadeState/Manual:
# 0 = A (left), 1 = None (always audible), 2 = B (right). A-side decks
# get A, B-side decks get B, mix gets None so it's audible regardless
# of crossfader position when the user chooses to unmute it.
_CROSSFADE_A = "0"
_CROSSFADE_NONE = "1"
_CROSSFADE_B = "2"


def _source_kind_of(deck_kind: str) -> str:
    """drums_a → drums; mix → mix."""
    if deck_kind.endswith(("_a", "_b")):
        return deck_kind[:-2]
    return deck_kind


def _side_of(deck_kind: str) -> str | None:
    """drums_a → 'a'; mix → None."""
    if deck_kind.endswith("_a"):
        return "a"
    if deck_kind.endswith("_b"):
        return "b"
    return None


def _crossfade_value_for(deck_kind: str) -> str:
    side = _side_of(deck_kind)
    if side == "a":
        return _CROSSFADE_A
    if side == "b":
        return _CROSSFADE_B
    return _CROSSFADE_NONE


def _display_for(deck_kind: str) -> str:
    """Title-case display name: 'drums_a' → 'Drums A', 'mix' → 'Mix'."""
    side = _side_of(deck_kind)
    src = _source_kind_of(deck_kind).title()
    return f"{src} {side.upper()}" if side else src


# Starting ID for our injected AudioTracks. Template uses 8 and 14; we go
# well above to avoid any collision. LiveSet's NextPointeeId is already
# 22192 in the template so anything under that is safe.
_FIRST_TRACK_ID = 100


# Starting ID used when renumbering Pointee elements inside cloned
# AudioTracks. Every deepcopy duplicates ~55 inner Pointee Ids
# (AutomationTarget, ModulationTarget, ...). Live rejects the Set with
# "non-unique Pointee IDs" if we don't make them globally unique. We
# start well above the template's NextPointeeId (22192) so we don't
# collide with anything Live wrote.
_POINTEE_ID_START = 30000


# Element tags whose ``Id`` attribute participates in Live's Pointee
# (pointer-target) system and must be globally unique across the Set.
# Anything not in this set has an Id that is positional/local (e.g.
# ClipSlot's scene index, WarpMarker's per-clip index) and must stay
# the same after cloning.
_POINTEE_TAGS = frozenset(
    {
        "AutomationTarget",
        "AutomationLane",
        "ComplexProEnvelopeModulationTarget",
        "ComplexProFormantsModulationTarget",
        "FluxModulationTarget",
        "GrainSizeModulationTarget",
        "ModulationTarget",
        "Pointee",
        "SampleOffsetModulationTarget",
        "TrackSendHolder",
        "TransientEnvelopeModulationTarget",
        "TranspositionModulationTarget",
        "VolumeModulationTarget",
    }
)


# ---------------------------------------------------------------------------
# Inputs to the writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StemEntry:
    """One audio file destined for an AudioTrack slot in the Set."""

    kind: str
    path: Path
    duration_seconds: float | None = None

    def display_name(self) -> str:
        return self.kind.capitalize()


@dataclass(frozen=True)
class LiveSetSpec:
    """Everything the writer needs to produce a Live Set XML tree."""

    name: str
    bpm: float
    stems: Sequence[StemEntry]
    locators: Sequence[LocatorEntry]
    set_dir: Path | None = None


@dataclass(frozen=True)
class PlacedClip:
    deck_kind: str
    scene: int
    entry: StemEntry
    bpm: float
    name: str
    start_beats: float
    loop_beats: float | None
    color_index: int | None
    muted: bool = False


@dataclass(frozen=True)
class PreparedSetSpec:
    name: str
    bpm: float
    scene_names: Sequence[str]
    clips: Sequence[PlacedClip]
    set_dir: Path | None = None


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def _load_template() -> etree._ElementTree:
    """Parse the bundled blank Live 12 Set into an lxml tree."""
    tpl_path = resources.files("dance.als") / "templates" / "blank_live12.xml"
    with resources.as_file(tpl_path) as p:
        return etree.parse(str(p))


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_live_set_xml(spec: LiveSetSpec) -> bytes:
    """Build a Live Set XML document (as bytes, not yet gzipped)."""
    if spec.bpm <= 0:
        raise ValueError(f"bpm must be positive, got {spec.bpm}")
    if not spec.stems:
        raise ValueError("at least one stem required")

    tree = _load_template()
    root = tree.getroot()
    liveset = root.find("LiveSet")
    if liveset is None:
        raise RuntimeError("template missing <LiveSet>")

    _set_tempo(liveset, spec.bpm)
    _replace_tracks(liveset, spec.stems, spec.bpm, spec.set_dir)
    _set_locators(liveset, spec.locators)

    return etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=False,
    )


def build_prepared_set_xml(spec: PreparedSetSpec) -> bytes:
    if spec.bpm <= 0:
        raise ValueError(f"bpm must be positive, got {spec.bpm}")
    if not spec.clips:
        raise ValueError("at least one clip required")
    for placed in spec.clips:
        if placed.deck_kind not in DECK_ORDER:
            raise ValueError(f"unknown deck {placed.deck_kind!r}")
        if not 0 <= placed.scene < len(spec.scene_names):
            raise ValueError(f"clip scene {placed.scene} is outside the set's scenes")
        if placed.bpm <= 0:
            raise ValueError(f"clip bpm must be positive, got {placed.bpm}")

    tree = _load_template()
    root = tree.getroot()
    liveset = root.find("LiveSet")
    if liveset is None:
        raise RuntimeError("template missing <LiveSet>")

    _set_tempo(liveset, spec.bpm)
    _set_scenes(liveset, spec.scene_names)
    deck_tracks = dict(_insert_deck_tracks(liveset))
    _resize_clip_slot_lists(liveset, len(spec.scene_names))
    for clip_id, placed in enumerate(spec.clips):
        _build_audio_clip(
            _clip_slot_value(deck_tracks[placed.deck_kind], placed.scene),
            clip_id=clip_id,
            entry=placed.entry,
            bpm=placed.bpm,
            muted=placed.muted,
            set_dir=spec.set_dir,
            name=placed.name,
            color_index=placed.color_index,
            start_beats=placed.start_beats,
            loop_beats=placed.loop_beats,
            loops=placed.loop_beats is not None,
        )
    _set_locators(liveset, [])

    return etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=False,
    )


def _set_scenes(liveset: etree._Element, names: Sequence[str]) -> None:
    scenes = liveset.find("Scenes")
    if scenes is None or len(scenes) == 0:
        raise RuntimeError("template missing <Scenes>")
    blank = deepcopy(scenes[0])
    for scene in list(scenes):
        scenes.remove(scene)
    for index, name in enumerate(names):
        scene = deepcopy(blank)
        scene.set("Id", str(index))
        scene_name = scene.find("Name")
        if scene_name is not None:
            scene_name.set("Value", name)
        scenes.append(scene)


def _resize_clip_slot_lists(liveset: etree._Element, scene_count: int) -> None:
    for slot_list in list(liveset.iter("ClipSlotList")):
        if len(slot_list) == 0:
            continue
        blank = deepcopy(slot_list[0])
        value = blank.find("ClipSlot/Value")
        if value is not None:
            for child in list(value):
                value.remove(child)
        for slot in list(slot_list):
            slot_list.remove(slot)
        for index in range(scene_count):
            slot = deepcopy(blank)
            slot.set("Id", str(index))
            slot_list.append(slot)


def _clip_slot_value(deck_track: etree._Element, scene: int) -> etree._Element:
    slot_list = deck_track.find("DeviceChain/MainSequencer/ClipSlotList")
    if slot_list is None or len(slot_list) <= scene:
        raise RuntimeError("deck track has no clip slot for this scene")
    inner = slot_list[scene].find("ClipSlot")
    if inner is None:
        raise RuntimeError("template ClipSlot missing inner <ClipSlot>")
    value = inner.find("Value")
    if value is None:
        value = etree.SubElement(inner, "Value")
    return value


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------


def _set_tempo(liveset: etree._Element, bpm: float) -> None:
    """Update MainTrack tempo.

    Live stores the master tempo in two places that have to agree:

    1. ``MainTrack/DeviceChain/Mixer/Tempo/Manual`` — the fallback value
       shown when no automation exists.
    2. ``MainTrack/AutomationEnvelopes/Envelopes/AutomationEnvelope`` whose
       ``EnvelopeTarget/PointeeId`` matches the Tempo's
       ``AutomationTarget Id``. This envelope holds a single anchor
       ``FloatEvent`` at Time=-63072000 ("before time begins"). When the
       envelope exists Live reads this anchor, NOT Manual, for the
       transport tempo. Leaving it at the template's 120 produced a Set
       that played all clips at 120 even though we wrote 128 to Manual.
    """
    main = liveset.find("MainTrack")
    if main is None:
        raise RuntimeError("template missing <MainTrack>")
    bpm_str = f"{bpm:.6f}"

    tempo_target_id: str | None = None
    for tempo in main.iter("Tempo"):
        manual = tempo.find("Manual")
        if manual is not None:
            manual.set("Value", bpm_str)
        at = tempo.find("AutomationTarget")
        if at is not None and at.get("Id") is not None:
            tempo_target_id = at.get("Id")
        # Some Sets also carry FloatEvent automation inside <Tempo>; cover it.
        for fe in tempo.iter("FloatEvent"):
            fe.set("Value", bpm_str)

    if tempo_target_id is None:
        return

    # Find the AutomationEnvelope whose target Pointee Id == tempo_target_id
    # and overwrite its FloatEvent anchors.
    envs = main.find("AutomationEnvelopes/Envelopes")
    if envs is None:
        return
    for env in envs.findall("AutomationEnvelope"):
        target = env.find("EnvelopeTarget/PointeeId")
        if target is not None and target.get("Value") == tempo_target_id:
            for fe in env.iter("FloatEvent"):
                fe.set("Value", bpm_str)


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


def _replace_tracks(
    liveset: etree._Element,
    stems: Sequence[StemEntry],
    bpm: float,
    set_dir: Path | None = None,
) -> None:
    """Swap the template's tracks for the deck tracks and put one song's
    stems on deck A.

    The offline per-track path is a single-song snapshot, so only the A
    side and its mix get audio. The B side ships with empty clip slots,
    ready for an incoming track's stems.
    """
    by_source = {s.kind: s for s in stems}
    for deck_kind, deck_track in _insert_deck_tracks(liveset):
        entry = by_source.get(_source_kind_of(deck_kind)) if _side_of(deck_kind) != "b" else None
        value = _clip_slot_value(deck_track, 0)
        for child in list(value):
            value.remove(child)
        if entry is None:
            continue
        _build_audio_clip(
            value,
            clip_id=int(deck_track.get("Id")),
            entry=entry,
            bpm=bpm,
            muted=deck_kind in ("mix_a", "mix_b"),
            set_dir=set_dir,
        )


def _insert_deck_tracks(liveset: etree._Element) -> list[tuple[str, etree._Element]]:
    tracks_el = liveset.find("Tracks")
    if tracks_el is None:
        raise RuntimeError("template missing <Tracks>")

    template_at: etree._Element | None = None
    for child in tracks_el:
        if child.tag == "AudioTrack":
            template_at = deepcopy(child)
            break
    if template_at is None:
        raise RuntimeError("template has no <AudioTrack> to clone")

    for child in list(tracks_el):
        if child.tag in ("MidiTrack", "AudioTrack"):
            tracks_el.remove(child)

    # Live keeps regular tracks before the return tracks.
    insert_idx = len(tracks_el)
    for i, child in enumerate(tracks_el):
        if child.tag == "ReturnTrack":
            insert_idx = i
            break

    deck_tracks: list[tuple[str, etree._Element]] = []
    next_pointee_id = _POINTEE_ID_START
    for i, deck_kind in enumerate(DECK_ORDER):
        clone = deepcopy(template_at)
        next_pointee_id = _renumber_pointees(clone, next_pointee_id)
        _style_deck_track(clone, track_id=_FIRST_TRACK_ID + i, deck_kind=deck_kind)
        tracks_el.insert(insert_idx + i, clone)
        deck_tracks.append((deck_kind, clone))

    # Live's own allocator has to start above the highest Pointee Id we used.
    npi = liveset.find("NextPointeeId")
    if npi is not None:
        try:
            existing = int(npi.get("Value") or "0")
        except ValueError:
            existing = 0
        npi.set("Value", str(max(existing, next_pointee_id + 100)))
    return deck_tracks


def _renumber_pointees(subtree: etree._Element, start_id: int) -> int:
    """Assign fresh unique ``Id`` attrs to every Pointee element in ``subtree``.

    Returns the next free Id (caller passes it to the next clone).
    """
    cur = start_id
    for el in subtree.iter():
        if el.tag in _POINTEE_TAGS and el.get("Id") is not None:
            el.set("Id", str(cur))
            cur += 1
    return cur


def _style_deck_track(at: etree._Element, *, track_id: int, deck_kind: str) -> None:
    at.set("Id", str(track_id))

    display = _display_for(deck_kind)
    name = at.find("Name")
    if name is not None:
        eff = name.find("EffectiveName")
        if eff is not None:
            eff.set("Value", display)
        usr = name.find("UserName")
        if usr is not None:
            usr.set("Value", display)

    color = at.find("Color")
    if color is not None:
        color.set("Value", str(STEM_COLOR_INDEX.get(_source_kind_of(deck_kind), 0)))

    # Crossfader assignment lives in the template's CrossFadeState/Manual leaf.
    cf_manual = at.find("DeviceChain/Mixer/CrossFadeState/Manual")
    if cf_manual is not None:
        cf_manual.set("Value", _crossfade_value_for(deck_kind))

    if at.find("DeviceChain/MainSequencer/ClipSlotList") is None:
        raise RuntimeError("template AudioTrack missing ClipSlotList")


# ---------------------------------------------------------------------------
# AudioClip
# ---------------------------------------------------------------------------


_WARP_GRID_BEATS = 1 / 32
WARP_MODE_BEATS = 0
WARP_MODE_COMPLEX = 4
WARP_MODE_COMPLEX_PRO = 6
WARP_MODE_BY_KIND: dict[str, int] = {
    "drums": WARP_MODE_BEATS,
    "bass": WARP_MODE_BEATS,
    "vocals": WARP_MODE_COMPLEX_PRO,
    "other": WARP_MODE_COMPLEX,
    "mix": WARP_MODE_COMPLEX,
}
_RELATIVE_PATH_NONE = "0"
_RELATIVE_PATH_FROM_SET = "1"
_FILE_REF_EXTERNAL = "2"


def _val(parent: etree._Element, tag: str, value: object) -> etree._Element:
    el = etree.SubElement(parent, tag)
    el.set("Value", str(value))
    return el


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _build_audio_clip(
    parent: etree._Element,
    *,
    clip_id: int,
    entry: StemEntry,
    bpm: float,
    muted: bool,
    set_dir: Path | None = None,
    name: str | None = None,
    color_index: int | None = None,
    start_beats: float = 0.0,
    loop_beats: float | None = None,
    loops: bool = True,
) -> None:
    duration = entry.duration_seconds or 0.0
    file_end_beats = (duration * bpm / 60.0) if duration > 0 else 16.0
    end_beats = start_beats + loop_beats if loop_beats else file_end_beats

    clip = etree.SubElement(parent, "AudioClip")
    clip.set("Id", str(clip_id))
    clip.set("Time", "0")

    _val(clip, "LomId", "0")
    _val(clip, "LomIdView", "0")
    _val(clip, "CurrentStart", _beats(start_beats))
    _val(clip, "CurrentEnd", _beats(end_beats))
    _build_loop_block(clip, start_beats, end_beats, loops)
    _val(clip, "Name", name or entry.display_name())
    _val(clip, "Annotation", "")
    _val(
        clip,
        "Color",
        STEM_COLOR_INDEX.get(entry.kind, 0) if color_index is None else color_index,
    )
    _val(clip, "LaunchMode", "0")
    # 0 is Global in the clip enum, as on live-loaded clips (bridge._LAUNCH_QUANT_GLOBAL).
    _val(clip, "LaunchQuantisation", "0")
    # TimeSignature is a CLASS element in Live's clip schema (carries
    # Numerator/Denominator children), not a leaf with Value attr. Omit
    # and let the clip inherit the project's 4/4.
    _val(clip, "Disabled", "true" if muted else "false")
    _val(clip, "IsWarped", "true")

    sample_ref = etree.SubElement(clip, "SampleRef")
    _add_file_ref(sample_ref, entry.path, set_dir)
    _val(sample_ref, "LastModDate", "0")
    etree.SubElement(sample_ref, "SourceContext")
    _val(sample_ref, "SampleUsageHint", "0")
    _val(sample_ref, "DefaultDuration", str(int(duration * 44100)) if duration else "0")
    _val(sample_ref, "DefaultSampleRate", "44100")
    _val(sample_ref, "SamplesToAutoWarp", "0")

    _val(clip, "WarpMode", WARP_MODE_BY_KIND.get(entry.kind, WARP_MODE_BEATS))
    _val(clip, "PitchCoarse", "0")
    _val(clip, "PitchFine", "0")
    _val(clip, "SampleVolume", "1")

    warp_markers = etree.SubElement(clip, "WarpMarkers")
    start = etree.SubElement(warp_markers, "WarpMarker")
    start.set("Id", "0")
    start.set("SecTime", "0")
    start.set("BeatTime", "0")
    slope = etree.SubElement(warp_markers, "WarpMarker")
    slope.set("Id", "1")
    slope.set("SecTime", repr(_WARP_GRID_BEATS * 60.0 / bpm))
    slope.set("BeatTime", repr(_WARP_GRID_BEATS))

    _val(clip, "MarkersGenerated", "true")


def _beats(value: float) -> str:
    return "0" if value == 0 else f"{value:.6f}"


def _build_loop_block(clip: etree._Element, start_beats: float, end_beats: float, loops: bool) -> None:
    loop = etree.SubElement(clip, "Loop")
    _val(loop, "LoopStart", _beats(start_beats))
    _val(loop, "LoopEnd", _beats(end_beats))
    _val(loop, "StartRelative", "0")
    _val(loop, "LoopOn", "true" if loops else "false")
    _val(loop, "OutMarker", _beats(end_beats))
    _val(loop, "HiddenLoopStart", _beats(start_beats))
    _val(loop, "HiddenLoopEnd", _beats(end_beats))


def _add_file_ref(parent: etree._Element, path: Path, set_dir: Path | None = None) -> None:
    """Insert a ``<FileRef>`` with the nine children Live 12.4 writes.

    With ``set_dir`` the reference carries a path relative to the Set's
    folder, as Live writes for an external file. Without it only the
    absolute ``Path`` is given.
    """
    file_ref = etree.SubElement(parent, "FileRef")
    absolute = path.resolve()
    if set_dir is None:
        _val(file_ref, "RelativePathType", _RELATIVE_PATH_NONE)
        _val(file_ref, "RelativePath", "")
    else:
        _val(file_ref, "RelativePathType", _RELATIVE_PATH_FROM_SET)
        _val(file_ref, "RelativePath", os.path.relpath(absolute, set_dir.resolve()))
    _val(file_ref, "Path", str(absolute))
    _val(file_ref, "Type", _FILE_REF_EXTERNAL)
    _val(file_ref, "LivePackName", "")
    _val(file_ref, "LivePackId", "")
    _val(file_ref, "OriginalFileSize", str(_safe_size(path)))
    _val(file_ref, "OriginalCrc", "0")
    _val(file_ref, "SourceHint", "")


# ---------------------------------------------------------------------------
# Locators
# ---------------------------------------------------------------------------


def _set_locators(
    liveset: etree._Element, entries: Iterable[LocatorEntry]
) -> None:
    """Replace the inner <Locators/> element's children with our entries."""
    outer = liveset.find("Locators")
    if outer is None:
        raise RuntimeError("template missing <Locators>")
    inner = outer.find("Locators")
    if inner is None:
        inner = etree.SubElement(outer, "Locators")
    for child in list(inner):
        inner.remove(child)
    for i, e in enumerate(entries):
        loc = etree.SubElement(inner, "Locator")
        loc.set("Id", str(i))
        _val(loc, "Time", f"{e.time_beats:.6f}")
        _val(loc, "Name", e.name)
        _val(loc, "Annotation", "")
        _val(loc, "IsSongStart", "false")


__all__ = [
    "ABLETON_ROOT_ATTRS",
    "LiveSetSpec",
    "STEM_COLOR_INDEX",
    "STEM_ORDER",
    "StemEntry",
    "build_live_set_xml",
]
