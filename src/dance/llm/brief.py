"""Compact analytical fingerprint of a track — used as text context for
generative taggers (Qwen2-Audio) and as a debug log line.

The CLAP zero-shot tagger doesn't use this — it reads the audio embedding
directly. But Qwen-style models benefit from seeing the BPM / key / energy
context alongside the audio.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from dance.core.database import (
    AudioAnalysis,
    Region,
    RegionType,
    StemFile,
    Track,
)


def build_track_brief(session: Session, track: Track) -> str:
    """Render an analytical fingerprint as plain text."""
    lines: list[str] = []
    if track.title or track.artist:
        lines.append(f"Title: {track.title or '?'}  —  Artist: {track.artist or '?'}")
    if track.duration_seconds:
        lines.append(f"Duration: {track.duration_seconds:.1f} s")

    mix = (
        session.query(AudioAnalysis)
        .filter_by(track_id=track.id, stem_file_id=None)
        .first()
    )
    if mix:
        bits: list[str] = []
        if mix.bpm:
            bits.append(f"BPM {mix.bpm:.1f}")
        if mix.key_camelot:
            bits.append(f"key {mix.key_camelot} ({mix.key_standard or '?'})")
        if mix.floor_energy:
            bits.append(f"energy {mix.floor_energy}/10")
        if mix.brightness is not None:
            bits.append(f"brightness {mix.brightness:.2f}")
        if mix.warmth is not None:
            bits.append(f"warmth {mix.warmth:.2f}")
        if mix.danceability is not None:
            bits.append(f"danceability {mix.danceability:.2f}")
        if bits:
            lines.append("Full mix: " + ", ".join(bits))

    stems = (
        session.query(StemFile, AudioAnalysis)
        .join(AudioAnalysis, AudioAnalysis.stem_file_id == StemFile.id)
        .filter(StemFile.track_id == track.id)
        .all()
    )
    for stem, sa in stems:
        parts: list[str] = []
        if sa.presence_ratio is not None:
            parts.append(f"presence {sa.presence_ratio:.0%}")
        if sa.kick_density is not None:
            parts.append(f"kicks {sa.kick_density:.1f}/s")
        if sa.vocal_present is not None:
            parts.append("vocals present" if sa.vocal_present else "no vocals")
        if sa.dominant_pitch_camelot:
            parts.append(f"pitch {sa.dominant_pitch_camelot}")
        if sa.floor_energy:
            parts.append(f"energy {sa.floor_energy}/10")
        if parts:
            lines.append(f"  {stem.kind}: " + ", ".join(parts))

    sections = (
        session.query(Region)
        .filter(Region.track_id == track.id, Region.region_type == RegionType.SECTION.value)
        .order_by(Region.position_ms)
        .all()
    )
    if sections:
        labels = [
            f"{(r.section_label or 'section')}@{r.position_ms // 1000}s"
            for r in sections
        ]
        lines.append("Sections: " + " → ".join(labels))

    return "\n".join(lines) if lines else "(no analytical data available)"
