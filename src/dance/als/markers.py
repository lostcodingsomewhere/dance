"""Conversion of ``Region`` rows into Ableton Live ``Locator`` entries.

A Locator in Live is a labeled position on the master timeline expressed in
**beats** (1 beat = 1 quarter note, given the project tempo). Our regions
store ``position_ms`` and a track-level BPM lives on the ``AudioAnalysis``
row — so converting is just ``beats = position_ms / 1000 * (bpm / 60)``.

A Region may carry a length (loops, fades, sections) but Live's master
Locator is a point only; if we want a region's *end* we add it as a second
Locator with a synthesized name (``<name> end``). Cue regions contribute
exactly one Locator each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dance.core.database import Region, RegionType


@dataclass(frozen=True)
class LocatorEntry:
    """One Locator in the final Live Set: a labeled point on the timeline."""

    time_beats: float
    name: str

    def __post_init__(self) -> None:  # pragma: no cover - dataclass guard
        if self.time_beats < 0:
            raise ValueError(f"time_beats must be >= 0, got {self.time_beats}")


def position_ms_to_beats(position_ms: int, bpm: float) -> float:
    """Convert a millisecond timeline position to beats at ``bpm``.

    Live treats Locator times as 4/4 quarter-note beats relative to the
    Set's start. ``bpm`` is beats-per-minute → 60 seconds per BPM beats →
    ``beats = (position_ms / 1000) * (bpm / 60)``.
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return (position_ms / 1000.0) * (bpm / 60.0)


def _default_name(region: Region, index: int) -> str:
    """Best-effort human-readable name when the Region has no ``name``."""
    if region.name:
        return region.name
    rt = region.region_type
    if rt == RegionType.SECTION.value:
        label = region.section_label or "section"
        return label.capitalize()
    if rt == RegionType.CUE.value:
        return f"Cue {index + 1}"
    if rt == RegionType.LOOP.value:
        return f"Loop {index + 1}"
    if rt == RegionType.FADE_IN.value:
        return "Fade In"
    if rt == RegionType.FADE_OUT.value:
        return "Fade Out"
    if rt == RegionType.STEM_SOLO.value:
        return f"Stem Solo {index + 1}"
    return f"Marker {index + 1}"


def regions_to_locators(
    regions: Iterable[Region],
    bpm: float,
    *,
    include_section_ends: bool = False,
) -> list[LocatorEntry]:
    """Convert an iterable of Regions to Locator entries, sorted by time.

    Only **track-level** regions (``stem_file_id is None``) are emitted —
    per-stem regions are dropped because they have no place on Live's
    master timeline.

    If ``include_section_ends`` is True, each SECTION region with a
    ``length_ms`` also emits an "end" Locator at the section's end.
    """
    locators: list[LocatorEntry] = []
    rs = [r for r in regions if r.stem_file_id is None]
    rs.sort(key=lambda r: (r.position_ms, r.id or 0))

    # Per-type counters so auto-names ("Cue 1", "Cue 2", ...) reflect the
    # sorted timeline order, not the input order.
    type_counts: dict[str, int] = {}
    for region in rs:
        idx_for_type = type_counts.get(region.region_type, 0)
        type_counts[region.region_type] = idx_for_type + 1

        start_beats = position_ms_to_beats(region.position_ms, bpm)
        name = _default_name(region, idx_for_type)
        locators.append(LocatorEntry(time_beats=start_beats, name=name))

        if (
            include_section_ends
            and region.region_type == RegionType.SECTION.value
            and region.length_ms
        ):
            end_ms = region.position_ms + region.length_ms
            end_beats = position_ms_to_beats(end_ms, bpm)
            locators.append(
                LocatorEntry(time_beats=end_beats, name=f"{name} end")
            )

    locators.sort(key=lambda e: e.time_beats)
    return locators


__all__ = ["LocatorEntry", "position_ms_to_beats", "regions_to_locators"]
