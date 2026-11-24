from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from dance.core.database import Region, RegionSource, RegionType, SectionLabel

PAD_COUNT = 5
BEATS_PER_BAR = 4
LOOP_BAR_CHOICES = (16, 8, 4)
SILENT_WINDOW_BARS = 8
SILENT_BELOW_PRESENT_FRACTION = 0.2

PHRASE_TYPE_TO_SECTION: dict[str, SectionLabel] = {
    "intro": SectionLabel.INTRO,
    "buildup": SectionLabel.BUILDUP,
    "drop": SectionLabel.DROP,
    "breakdown": SectionLabel.BREAKDOWN,
    "outro": SectionLabel.OUTRO,
}


@dataclass(frozen=True)
class Section:
    start_ms: int
    end_ms: int
    bar_count: int
    label: str
    energy: float


@dataclass(frozen=True)
class PadPick:
    slot: int
    position_ms: int
    length_ms: int
    length_bars: int
    section_label: str


def section_from_phrase(
    start_ms: int,
    end_ms: int,
    bar_count: int | None,
    phrase_type: str | None,
    energy_level: float | None,
) -> Section:
    label = PHRASE_TYPE_TO_SECTION.get(phrase_type or "", SectionLabel.OTHER)
    return Section(
        start_ms=start_ms,
        end_ms=end_ms,
        bar_count=bar_count or 0,
        label=label.value,
        energy=energy_level or 0.0,
    )


def bar_ms(bpm: float) -> float:
    return BEATS_PER_BAR * 60_000 / bpm


def loop_bars(available_bars: int) -> int:
    for choice in LOOP_BAR_CHOICES:
        if choice <= available_bars:
            return choice
    return max(1, available_bars)


def pick_pad_sections(sections: Sequence[Section]) -> list[Section]:
    ordered = sorted(sections, key=lambda s: s.start_ms)
    if not ordered:
        return []
    chosen: list[Section] = []

    def take(section: Section | None) -> None:
        if section is not None and section not in chosen and len(chosen) < PAD_COUNT:
            chosen.append(section)

    def earliest_with_most(candidates: Iterable[Section], amount) -> Section | None:
        return max(candidates, key=lambda s: (amount(s), -s.start_ms), default=None)

    intro, outro = ordered[0], ordered[-1]
    drops = [s for s in ordered if s.label == SectionLabel.DROP.value]
    loudest_drop = earliest_with_most(drops, lambda s: s.energy) or earliest_with_most(
        ordered[1:-1], lambda s: s.energy
    )
    drop_start = loudest_drop.start_ms if loudest_drop else outro.end_ms

    builds_before_drop = [
        s for s in ordered if s.label == SectionLabel.BUILDUP.value and s.start_ms < drop_start
    ]
    section_before_drop = next(
        (s for s in reversed(ordered) if s.start_ms < drop_start and s is not intro), None
    )
    first_build = builds_before_drop[0] if builds_before_drop else section_before_drop

    breakdowns_after_drop = [
        s for s in ordered if s.label == SectionLabel.BREAKDOWN.value and s.start_ms > drop_start
    ]
    longest_breakdown = earliest_with_most(breakdowns_after_drop, lambda s: s.bar_count)

    other_drops = [s for s in drops if s is not loudest_drop]
    last_drop = other_drops[-1] if other_drops else outro

    for section in (intro, first_build, loudest_drop, longest_breakdown, last_drop):
        take(section)
    unused = [s for s in ordered if s not in chosen]
    for section in sorted(unused, key=lambda s: (-s.bar_count, -s.energy, s.start_ms)):
        take(section)
    return sorted(chosen, key=lambda s: s.start_ms)


def pad_at(slot: int, position_ms: int, section: Section, bpm: float) -> PadPick:
    one_bar = bar_ms(bpm)
    available_bars = int(round((section.end_ms - position_ms) / one_bar))
    length_bars = loop_bars(available_bars)
    return PadPick(
        slot=slot,
        position_ms=position_ms,
        length_ms=int(round(length_bars * one_bar)),
        length_bars=length_bars,
        section_label=section.label,
    )


def pick_pads(sections: Sequence[Section], bpm: float) -> list[PadPick]:
    if bpm <= 0:
        return []
    return [
        pad_at(slot, section.start_ms, section, bpm)
        for slot, section in enumerate(pick_pad_sections(sections), start=1)
    ]


def snap_to_bar(position_ms: int, downbeats_ms: Sequence[int]) -> int:
    return min(downbeats_ms, key=lambda downbeat: abs(downbeat - position_ms))


def section_containing(position_ms: int, sections: Sequence[Section]) -> Section | None:
    for section in sections:
        if section.start_ms <= position_ms < section.end_ms:
            return section
    return None


def is_silent(presence_intervals: Sequence[Sequence[int]], position_ms: int, bpm: float) -> bool:
    window_ms = SILENT_WINDOW_BARS * bar_ms(bpm)
    window_end = position_ms + window_ms
    present_ms = sum(
        max(0.0, min(end, window_end) - max(start, position_ms))
        for start, end in presence_intervals
    )
    return present_ms / window_ms < SILENT_BELOW_PRESENT_FRACTION


def pad_region(
    track_id: int, stem_file_id: int | None, pick: PadPick, source: RegionSource
) -> Region:
    return Region(
        track_id=track_id,
        stem_file_id=stem_file_id,
        position_ms=pick.position_ms,
        length_ms=pick.length_ms,
        region_type=RegionType.PAD.value,
        section_label=pick.section_label,
        length_bars=pick.length_bars,
        slot=pick.slot,
        name=f"Pad {pick.slot}",
        snapped_to="bar",
        source=source.value,
    )
