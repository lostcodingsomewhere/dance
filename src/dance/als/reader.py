"""Read AudioClips — and their warp markers — out of an Ableton ``.als``.

This is the read side of the Ableton-first beat-grid reference: a Live Set
the user saved after correcting warps by ear is the closest thing to ground
truth we have (CLAUDE.md rule 3). It also reads dance's own exports, which is
how the plumbing is sanity-checked (an export's grid comes from the DB, so
comparing it back against the DB should agree on tempo).

Only what a grid comparison needs is extracted: the sample path, the warp
markers (``SecTime`` = seconds into the sample, ``BeatTime`` = clip beats),
whether the clip is warped at all, and where it sits (track name, session vs
arrangement). Nothing is written; the writer lives in :mod:`dance.als.writer`.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from dance.pipeline.utils.grid_metrics import WarpMarker

_GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class AlsClip:
    """One AudioClip as found in a Live Set."""

    sample_path: str | None  # absolute path Live recorded for the sample
    sample_name: str | None  # FileRef Name (basename), survives a moved library
    clip_name: str | None
    track_name: str | None
    location: str  # "session" | "arrangement"
    warped: bool
    markers: list[WarpMarker] = field(default_factory=list)
    sample_duration_s: float | None = None  # from SampleRef DefaultDuration/Rate


def _value(el: etree._Element | None, path: str) -> str | None:
    if el is None:
        return None
    found = el.find(path)
    return None if found is None else found.get("Value")


def _bool(v: str | None) -> bool | None:
    if v is None:
        return None
    return v.strip().lower() == "true"


def _float(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_als_xml(path: Path) -> etree._Element:
    """Parse a ``.als`` (gzipped XML) — or plain XML — into its root element."""
    raw = Path(path).read_bytes()
    if raw[:2] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)
    parser = etree.XMLParser(resolve_entities=False, no_network=True, remove_blank_text=True)
    return etree.fromstring(raw, parser)


def _clip_from_element(clip: etree._Element) -> AlsClip:
    file_ref = clip.find("SampleRef/FileRef")
    sample_path = _value(file_ref, "Path")
    sample_name = _value(file_ref, "Name")
    if sample_name is None and sample_path:
        sample_name = Path(sample_path).name

    # Live 10+ writes <IsWarped>; dance's offline writer emits <Warp>.
    warped = _bool(_value(clip, "IsWarped"))
    if warped is None:
        warped = _bool(_value(clip, "Warp"))
    if warped is None:
        warped = True

    markers: list[WarpMarker] = []
    for wm in clip.findall("WarpMarkers/WarpMarker"):
        sec = _float(wm.get("SecTime"))
        beat = _float(wm.get("BeatTime"))
        if sec is not None and beat is not None:
            markers.append(WarpMarker(sec=sec, beat=beat))

    sample_ref = clip.find("SampleRef")
    frames = _float(_value(sample_ref, "DefaultDuration"))
    rate = _float(_value(sample_ref, "DefaultSampleRate"))
    duration = frames / rate if frames and rate and frames > 0 and rate > 0 else None

    track_name: str | None = None
    location = "arrangement"
    for anc in clip.iterancestors():
        if anc.tag == "ClipSlotList":
            location = "session"
        if anc.tag in ("AudioTrack", "MidiTrack", "ReturnTrack", "GroupTrack"):
            track_name = _value(anc, "Name/EffectiveName") or _value(anc, "Name/UserName")
            break

    return AlsClip(
        sample_path=sample_path,
        sample_name=sample_name,
        clip_name=_value(clip, "Name"),
        track_name=track_name,
        location=location,
        warped=warped,
        markers=markers,
        sample_duration_s=duration,
    )


def read_als_clips(path: Path) -> list[AlsClip]:
    """Every AudioClip in a Live Set, in document order."""
    root = load_als_xml(path)
    return [_clip_from_element(c) for c in root.iter("AudioClip")]


__all__ = ["AlsClip", "load_als_xml", "read_als_clips"]
