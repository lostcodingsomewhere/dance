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
STEM_COLOR_INDEX: dict[str, int] = {
    "drums": 1,    # orange-red — strongest warm hue we observed
    "bass": 2,     # brown/mustard
    "vocals": 4,   # lime
    "other": 10,   # light blue
    "mix": 13,     # white — distinct from the 4 stems, signals "reference"
}


# Canonical track order — drums/bass/vocals/other/mix.
STEM_ORDER: tuple[str, ...] = ("drums", "bass", "vocals", "other", "mix")


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
    _replace_tracks(liveset, spec.stems, spec.bpm)
    _set_locators(liveset, spec.locators)

    return etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=False,
    )


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
    liveset: etree._Element, stems: Sequence[StemEntry], bpm: float
) -> None:
    """Remove default Midi/Audio tracks, insert one AudioTrack per stem."""
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

    # Where to insert: before the first ReturnTrack (Live convention is
    # regular tracks first, returns last). If no ReturnTracks, append.
    insert_idx = len(tracks_el)
    for i, child in enumerate(tracks_el):
        if child.tag == "ReturnTrack":
            insert_idx = i
            break

    by_kind = {s.kind: s for s in stems}
    ordered: list[StemEntry] = [by_kind[k] for k in STEM_ORDER if k in by_kind]
    seen = {s.kind for s in ordered}
    for s in stems:
        if s.kind not in seen:
            ordered.append(s)
            seen.add(s.kind)

    next_pointee_id = _POINTEE_ID_START
    for i, stem in enumerate(ordered):
        clone = deepcopy(template_at)
        next_pointee_id = _renumber_pointees(clone, next_pointee_id)
        _populate_audio_track(clone, track_id=_FIRST_TRACK_ID + i, entry=stem, bpm=bpm)
        tracks_el.insert(insert_idx + i, clone)

    # Tell Live our highest Pointee Id so its allocator picks a value
    # above ours when the user edits the Set.
    npi = liveset.find("NextPointeeId")
    if npi is not None:
        try:
            existing = int(npi.get("Value") or "0")
        except ValueError:
            existing = 0
        npi.set("Value", str(max(existing, next_pointee_id + 100)))


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


def _populate_audio_track(
    at: etree._Element, *, track_id: int, entry: StemEntry, bpm: float
) -> None:
    """Fill in a cloned AudioTrack with our stem's name, color, and clip."""
    at.set("Id", str(track_id))

    name = at.find("Name")
    if name is not None:
        eff = name.find("EffectiveName")
        if eff is not None:
            eff.set("Value", entry.display_name())
        usr = name.find("UserName")
        if usr is not None:
            usr.set("Value", entry.display_name())

    color_idx = STEM_COLOR_INDEX.get(entry.kind, 0)
    color = at.find("Color")
    if color is not None:
        color.set("Value", str(color_idx))
    # Also expose a ColorIndex sibling so tests / external consumers that
    # look for it find the same value. Live ignores extra leaf elements.
    ci = at.find("ColorIndex")
    if ci is None:
        ci = etree.SubElement(at, "ColorIndex")
    ci.set("Value", str(color_idx))

    csl = at.find("DeviceChain/MainSequencer/ClipSlotList")
    if csl is None or len(csl) == 0:
        raise RuntimeError("template AudioTrack missing ClipSlotList")
    first_slot = csl[0]
    inner = first_slot.find("ClipSlot")
    if inner is None:
        raise RuntimeError("template ClipSlot missing inner <ClipSlot>")
    value = inner.find("Value")
    if value is None:
        value = etree.SubElement(inner, "Value")
    for child in list(value):
        value.remove(child)
    _build_audio_clip(
        value, clip_id=track_id, entry=entry, bpm=bpm, muted=(entry.kind == "mix")
    )


# ---------------------------------------------------------------------------
# AudioClip
# ---------------------------------------------------------------------------


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
) -> None:
    duration = entry.duration_seconds or 0.0
    end_beats = (duration * bpm / 60.0) if duration > 0 else 16.0

    clip = etree.SubElement(parent, "AudioClip")
    clip.set("Id", str(clip_id))
    clip.set("Time", "0")

    _val(clip, "LomId", "0")
    _val(clip, "LomIdView", "0")
    _val(clip, "CurrentStart", "0")
    _val(clip, "CurrentEnd", f"{end_beats:.6f}")
    _build_loop_block(clip, end_beats)
    _val(clip, "Name", entry.display_name())
    _val(clip, "Annotation", "")
    _val(clip, "ColorIndex", STEM_COLOR_INDEX.get(entry.kind, 0))
    _val(clip, "LaunchMode", "0")
    _val(clip, "LaunchQuantisation", "0")
    # TimeSignature is a CLASS element in Live's clip schema (carries
    # Numerator/Denominator children), not a leaf with Value attr. Omit
    # and let the clip inherit the project's 4/4.
    _val(clip, "Disabled", "true" if muted else "false")

    sample_ref = etree.SubElement(clip, "SampleRef")
    _add_file_ref(sample_ref, entry.path)
    _val(sample_ref, "LastModDate", "0")
    _val(sample_ref, "SourceContext", "")
    _val(sample_ref, "SampleUsageHint", "0")
    _val(sample_ref, "DefaultDuration", str(int(duration * 44100)) if duration else "0")
    _val(sample_ref, "DefaultSampleRate", "44100")

    warp_markers = etree.SubElement(clip, "WarpMarkers")
    start = etree.SubElement(warp_markers, "WarpMarker")
    start.set("Id", "0")
    start.set("SecTime", "0")
    start.set("BeatTime", "0")
    end = etree.SubElement(warp_markers, "WarpMarker")
    end.set("Id", "1")
    end.set("SecTime", f"{duration:.6f}" if duration else "1.0")
    end.set("BeatTime", f"{end_beats:.6f}")

    _val(clip, "WarpMode", "0")
    _val(clip, "GranularityResolution", "4")
    _val(clip, "Gain", "0")
    _val(clip, "PitchCoarse", "0")
    _val(clip, "PitchFine", "0")
    _val(clip, "SampleVolume", "1")
    _val(clip, "MarkersGenerated", "true")
    _val(clip, "Warp", "true")


def _build_loop_block(clip: etree._Element, end_beats: float) -> None:
    loop = etree.SubElement(clip, "Loop")
    _val(loop, "LoopStart", "0")
    _val(loop, "LoopEnd", f"{end_beats:.6f}")
    _val(loop, "StartRelative", "0")
    _val(loop, "LoopOn", "false")
    _val(loop, "OutMarker", f"{end_beats:.6f}")
    _val(loop, "HiddenLoopStart", "0")
    _val(loop, "HiddenLoopEnd", f"{end_beats:.6f}")


def _add_file_ref(parent: etree._Element, path: Path, name: str | None = None) -> None:
    """Insert a leaf-style ``<FileRef>`` describing ``path``.

    Live 11+ rejects child elements inside ``<RelativePath>``; emit it as
    a leaf with a Value attribute carrying the parent directory.
    """
    file_ref = etree.SubElement(parent, "FileRef")

    parts = list(path.resolve().parts)
    if parts and parts[0] == "/":
        parts = parts[1:]
    rel_path_str = "/".join(parts[:-1])
    rel = etree.SubElement(file_ref, "RelativePath")
    rel.set("Value", rel_path_str)

    _val(file_ref, "Name", path.name if name is None else name)
    _val(file_ref, "Type", "1")  # 1 = audio sample
    _val(file_ref, "Path", str(path.resolve()))
    _val(file_ref, "OriginalFileSize", str(_safe_size(path)))
    _val(file_ref, "OriginalCrc", "0")
    _val(file_ref, "HasRelativePath", "false")
    _val(file_ref, "HasSearchHint", "false")
    _val(file_ref, "SearchHint", "")
    _val(file_ref, "DefaultDuration", "0")
    _val(file_ref, "DefaultSampleRate", "44100")


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
