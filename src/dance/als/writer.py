"""Programmatic construction of a minimal Ableton Live Set XML tree.

The XML schema for ``.als`` files is proprietary and undocumented by
Ableton. The structure here was reverse-engineered from publicly archived
samples (see e.g. ``kiddikai/ableton-parser`` and discussion on the
Ableton forum). It targets Live 11/12 — both versions share a compatible
top-level schema, and Live happily fills in the many missing fields when
loading a sparse Set.

What's included:
- Root ``<Ableton>`` element with the four core attributes.
- ``<LiveSet>`` wrapper.
- ``<Tracks>`` with one ``<AudioTrack>`` per (stem + mix) — each has a
  ``<Name>``, ``<ColorIndex>``, and a ``<DeviceChain>`` whose
  ``<MainSequencer><Sample><SampleRef><FileRef>`` points at an absolute
  path on disk.
- ``<MasterTrack>`` with a Manual Tempo at the track's BPM.
- ``<Locators>`` with one ``<Locator>`` per marker.

What's intentionally omitted:
- Mixer state, sends, routing, automation, view state, freeze/warp data.
  Live regenerates / defaults all of these on first load.

The colors are Live's built-in palette indices (0-69):
- drums  →  1  (red)
- bass   →  7  (orange)
- vocals → 13  (yellow)
- other  → 39  (blue)
- mix    → 25  (grey)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from lxml import etree

from dance.als.markers import LocatorEntry


# ---------------------------------------------------------------------------
# Public schema constants
# ---------------------------------------------------------------------------


# Live 11.x schema attributes. Live 12 reads 11's files transparently, so
# we target the older schema for max compatibility.
ABLETON_ROOT_ATTRS = {
    "MajorVersion": "5",
    "MinorVersion": "11.0_11300",
    "SchemaChangeCount": "3",
    "Creator": "Ableton Live 11.0",
    "Revision": "f7eb4c8e7a5d5b1f7eb4c8e7a5d5b1f7eb4c8e7a",
}


# Per-kind color index in Live's palette (0-69).
STEM_COLOR_INDEX: dict[str, int] = {
    "drums": 1,
    "bass": 7,
    "vocals": 13,
    "other": 39,
    "mix": 25,
}


# The order Tracks appear in Live — mix last so the four stems sit on top.
STEM_ORDER: tuple[str, ...] = ("drums", "bass", "vocals", "other", "mix")


# ---------------------------------------------------------------------------
# Inputs to the writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StemEntry:
    """One audio file destined for an AudioTrack slot in the Set.

    ``kind`` is one of ``drums|bass|vocals|other|mix`` and drives both the
    track name and the color.
    """

    kind: str
    path: Path
    duration_seconds: float | None = None

    def display_name(self) -> str:
        return self.kind.capitalize()


@dataclass(frozen=True)
class LiveSetSpec:
    """Everything the writer needs to produce a Live Set XML tree."""

    name: str  # appears in window title; also used for FileRef Name on mix
    bpm: float
    stems: Sequence[StemEntry]
    locators: Sequence[LocatorEntry]


# ---------------------------------------------------------------------------
# Helpers — every Live XML "Value" element is ``<Tag Value="..." />``
# ---------------------------------------------------------------------------


def _val(parent: etree._Element, tag: str, value: object) -> etree._Element:
    """Append ``<Tag Value="..."/>``, returning the new element."""
    el = etree.SubElement(parent, tag)
    el.set("Value", str(value))
    return el


# Live uses absolute paths on macOS like ``/Users/foo/track.wav``; on
# Windows they look like ``C:\Users\foo\track.wav``. We always pass the
# absolute path; Live will fall back to the SearchHint if it can't find
# it. We split into ``RelativePathElement`` entries too because Live looks
# for those (older Sets had relative-to-project paths).
def _add_file_ref(parent: etree._Element, path: Path, name: str | None = None) -> None:
    """Insert a ``<FileRef>`` describing ``path`` for a Sample element."""
    file_ref = etree.SubElement(parent, "FileRef")

    # Live requires <RelativePath Value="..."/> with the relative path string;
    # the per-element <RelativePathElement Dir=".."/> children are how Live
    # records each path component for portability search.
    parts = list(path.resolve().parts)
    if parts and parts[0] == "/":  # POSIX root
        parts = parts[1:]
    rel_path_str = "/".join(parts[:-1])
    rel_path = etree.SubElement(file_ref, "RelativePath")
    rel_path.set("Value", rel_path_str)
    for part in parts[:-1]:
        rpe = etree.SubElement(rel_path, "RelativePathElement")
        rpe.set("Dir", part)

    _val(file_ref, "Name", path.name if name is None else name)
    _val(file_ref, "Type", "1")  # 1 = audio sample (per public reverse-eng)
    # Live also stores an absolute path for portability; we put the same
    # one we already used in the relative-path elements above.
    _val(file_ref, "Path", str(path.resolve()))
    _val(file_ref, "OriginalFileSize", str(_safe_size(path)))
    _val(file_ref, "OriginalCrc", "0")
    _val(file_ref, "HasRelativePath", "false")
    _val(file_ref, "HasSearchHint", "false")
    _val(file_ref, "SearchHint", "")
    _val(file_ref, "DefaultDuration", "0")
    _val(file_ref, "DefaultSampleRate", "44100")


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        # The file may not exist at write-time (e.g. when generating a Set
        # for a track whose stems live on an offline drive). Live will
        # still try to relink at load-time.
        return 0


# ---------------------------------------------------------------------------
# AudioTrack — one per stem (plus one for the mix)
# ---------------------------------------------------------------------------


def _build_audio_track(
    parent: etree._Element,
    track_id: int,
    entry: StemEntry,
    bpm: float,
) -> etree._Element:
    """Append an ``<AudioTrack>`` for ``entry`` to ``parent``."""
    at = etree.SubElement(parent, "AudioTrack")
    at.set("Id", str(track_id))

    # Track name
    name = etree.SubElement(at, "Name")
    _val(name, "EffectiveName", entry.display_name())
    _val(name, "UserName", entry.display_name())
    _val(name, "Annotation", "")
    _val(name, "MemorizedFirstClipName", "")

    # Color
    color_idx = STEM_COLOR_INDEX.get(entry.kind, 0)
    _val(at, "Color", color_idx)
    _val(at, "ColorIndex", color_idx)

    # Mute the mix track by default — it's a reference, not for performance.
    if entry.kind == "mix":
        _val(at, "TrackUnfolded", "true")
        # Mixer/track mute is buried deep; the most reliable place that
        # opens muted on load is the AudioClip's "Disabled" flag, set below.

    device_chain = etree.SubElement(at, "DeviceChain")

    # AutomationLanes (empty)
    al = etree.SubElement(device_chain, "AutomationLanes")
    etree.SubElement(al, "AutomationLanes")

    # ClipEnvelopeChooserViewState (empty)
    etree.SubElement(device_chain, "ClipEnvelopeChooserViewState")

    main = etree.SubElement(device_chain, "MainSequencer")
    sample = etree.SubElement(main, "Sample")
    _build_arranger_clip(sample, entry, bpm, muted=(entry.kind == "mix"))

    # Empty devices section
    devices = etree.SubElement(device_chain, "Devices")
    # No devices — Live populates a default mixer on load.
    del devices  # noqa: F841 — kept for explicit shape

    return at


def _build_arranger_clip(
    sample_el: etree._Element,
    entry: StemEntry,
    bpm: float,
    *,
    muted: bool,
) -> None:
    """Inside a ``<Sample>``, build the ``<ArrangerAutomation>`` + clip(s)."""
    # Outermost wrap: an ArrangerAutomation containing Events containing
    # one AudioClip with our sample referenced.
    arr = etree.SubElement(sample_el, "ArrangerAutomation")
    events = etree.SubElement(arr, "Events")

    duration = entry.duration_seconds or 0.0
    # Length in beats. If duration is unknown, use a placeholder bar count.
    end_beats = (duration * bpm / 60.0) if duration > 0 else 16.0

    clip = etree.SubElement(events, "AudioClip")
    clip.set("Id", "0")
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
    _val(clip, "TimeSignature", "")  # use master
    _val(clip, "Disabled", "true" if muted else "false")

    sample_ref = etree.SubElement(clip, "SampleRef")
    _add_file_ref(sample_ref, entry.path)
    _val(sample_ref, "LastModDate", "0")
    _val(sample_ref, "SourceContext", "")
    _val(sample_ref, "SampleUsageHint", "0")
    _val(sample_ref, "DefaultDuration", str(int(duration * 44100)) if duration else "0")
    _val(sample_ref, "DefaultSampleRate", "44100")

    # WarpMode = 0 (beats). MarkersGenerated = false so Live regenerates
    # warp markers on load (we don't try to compute sample-accurate ones).
    _val(clip, "WarpMode", "0")
    _val(clip, "GranularityResolution", "4")
    _val(clip, "Gain", "0")
    _val(clip, "PitchCoarse", "0")
    _val(clip, "PitchFine", "0")
    _val(clip, "SampleVolume", "1")
    _val(clip, "MarkersGenerated", "false")
    _val(clip, "Warp", "true")


def _build_loop_block(clip: etree._Element, end_beats: float) -> None:
    """Append a ``<Loop>`` element describing the clip's playable range."""
    loop = etree.SubElement(clip, "Loop")
    _val(loop, "LoopStart", "0")
    _val(loop, "LoopEnd", f"{end_beats:.6f}")
    _val(loop, "StartRelative", "0")
    _val(loop, "LoopOn", "false")
    _val(loop, "OutMarker", f"{end_beats:.6f}")
    _val(loop, "HiddenLoopStart", "0")
    _val(loop, "HiddenLoopEnd", f"{end_beats:.6f}")


# ---------------------------------------------------------------------------
# MasterTrack with Tempo
# ---------------------------------------------------------------------------


def _build_master_track(parent: etree._Element, bpm: float) -> etree._Element:
    """Append a ``<MasterTrack>`` carrying the project tempo."""
    mt = etree.SubElement(parent, "MasterTrack")

    device_chain = etree.SubElement(mt, "DeviceChain")
    mixer = etree.SubElement(device_chain, "Mixer")

    # Tempo lives on the master mixer.
    tempo = etree.SubElement(mixer, "Tempo")
    _val(tempo, "LomId", "0")
    arr = etree.SubElement(tempo, "ArrangerAutomation")
    events = etree.SubElement(arr, "Events")
    fe = etree.SubElement(events, "FloatEvent")
    fe.set("Id", "0")
    fe.set("Time", "-63072000")  # Live's "before time begins" sentinel
    fe.set("Value", f"{bpm:.6f}")
    _val(tempo, "Manual", f"{bpm:.6f}")
    mcr = etree.SubElement(tempo, "MidiControllerRange")
    _val(mcr, "Min", "60")
    _val(mcr, "Max", "200")
    at_target = etree.SubElement(tempo, "AutomationTarget")
    at_target.set("Id", "8")
    _val(at_target, "LockEnvelope", "0")

    # Time signature: 4/4 fixed.
    ts = etree.SubElement(mixer, "TimeSignature")
    arr2 = etree.SubElement(ts, "ArrangerAutomation")
    events2 = etree.SubElement(arr2, "Events")
    ee = etree.SubElement(events2, "EnumEvent")
    ee.set("Id", "0")
    ee.set("Time", "-63072000")
    ee.set("Value", "201")  # 201 = 4/4 (per public reverse-eng tables)

    return mt


# ---------------------------------------------------------------------------
# Locators
# ---------------------------------------------------------------------------


def _build_locators(parent: etree._Element, entries: Iterable[LocatorEntry]) -> None:
    """Append a ``<Locators>`` element with one ``<Locator>`` per entry."""
    locators = etree.SubElement(parent, "Locators")
    # Inner wrapper is also named Locators in Live's schema.
    inner = etree.SubElement(locators, "Locators")
    for i, e in enumerate(entries):
        loc = etree.SubElement(inner, "Locator")
        loc.set("Id", str(i))
        _val(loc, "Time", f"{e.time_beats:.6f}")
        _val(loc, "Name", e.name)
        _val(loc, "Annotation", "")
        _val(loc, "IsSongStart", "false")


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_live_set_xml(spec: LiveSetSpec) -> bytes:
    """Build a Live Set XML document (as bytes, not yet gzipped).

    The output starts with the standard XML declaration and uses LF line
    endings — exactly what Live writes when it saves an uncompressed Set
    via the ``--als-as-xml`` debug flag.
    """
    if spec.bpm <= 0:
        raise ValueError(f"bpm must be positive, got {spec.bpm}")
    if not spec.stems:
        raise ValueError("at least one stem required")

    ableton = etree.Element("Ableton")
    for k, v in ABLETON_ROOT_ATTRS.items():
        ableton.set(k, v)

    live_set = etree.SubElement(ableton, "LiveSet")
    _val(live_set, "OverwriteProtectionNumber", "2305")
    _val(live_set, "LomId", "0")
    _val(live_set, "LomIdView", "0")

    tracks_el = etree.SubElement(live_set, "Tracks")
    # Sort stems into the canonical STEM_ORDER (drums, bass, vocals, other, mix)
    # so the Set always shows tracks in the same vertical order.
    by_kind = {s.kind: s for s in spec.stems}
    ordered: list[StemEntry] = [by_kind[k] for k in STEM_ORDER if k in by_kind]
    # Append any extra kinds we didn't expect (defensive).
    seen = {s.kind for s in ordered}
    for s in spec.stems:
        if s.kind not in seen:
            ordered.append(s)
    for i, stem in enumerate(ordered):
        # Track Ids start at 10 to leave room for the master and arrangement
        # bookkeeping IDs Live writes implicitly.
        _build_audio_track(tracks_el, track_id=10 + i, entry=stem, bpm=spec.bpm)

    _build_master_track(live_set, spec.bpm)
    _build_locators(live_set, spec.locators)

    _val(live_set, "ScaleInformation", "")
    _val(live_set, "InKey", "false")
    _val(live_set, "SongMasterValues", "")

    # Serialize with the standard XML declaration and pretty-print so the
    # output is readable when manually un-gzipped (helps debugging Live's
    # complaints about a Set).
    return etree.tostring(
        ableton,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
        standalone=False,
    )


__all__ = [
    "ABLETON_ROOT_ATTRS",
    "LiveSetSpec",
    "STEM_COLOR_INDEX",
    "STEM_ORDER",
    "StemEntry",
    "build_live_set_xml",
]
