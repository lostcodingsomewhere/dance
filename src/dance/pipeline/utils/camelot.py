"""
Camelot wheel key conversion utilities.

The Camelot wheel is a tool that helps DJs mix harmonically.
It maps musical keys to a number-letter notation (1A-12B).

- Numbers 1-12 represent positions on the wheel
- 'A' = minor keys, 'B' = major keys
- Adjacent numbers are harmonically compatible
- Same number, different letter (A/B) are relative major/minor
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Standard key names to Camelot notation
# Format: (key, mode) -> Camelot code
KEY_TO_CAMELOT = {
    # Minor keys (A)
    ("A", "minor"): "8A",
    ("A#", "minor"): "3A",
    ("Bb", "minor"): "3A",
    ("B", "minor"): "10A",
    ("C", "minor"): "5A",
    ("C#", "minor"): "12A",
    ("Db", "minor"): "12A",
    ("D", "minor"): "7A",
    ("D#", "minor"): "2A",
    ("Eb", "minor"): "2A",
    ("E", "minor"): "9A",
    ("F", "minor"): "4A",
    ("F#", "minor"): "11A",
    ("Gb", "minor"): "11A",
    ("G", "minor"): "6A",
    ("G#", "minor"): "1A",
    ("Ab", "minor"): "1A",
    # Major keys (B)
    ("A", "major"): "11B",
    ("A#", "major"): "6B",
    ("Bb", "major"): "6B",
    ("B", "major"): "1B",
    ("C", "major"): "8B",
    ("C#", "major"): "3B",
    ("Db", "major"): "3B",
    ("D", "major"): "10B",
    ("D#", "major"): "5B",
    ("Eb", "major"): "5B",
    ("E", "major"): "12B",
    ("F", "major"): "7B",
    ("F#", "major"): "2B",
    ("Gb", "major"): "2B",
    ("G", "major"): "9B",
    ("G#", "major"): "4B",
    ("Ab", "major"): "4B",
}

# Camelot to standard key names
CAMELOT_TO_KEY = {
    "1A": "Ab minor",
    "2A": "Eb minor",
    "3A": "Bb minor",
    "4A": "F minor",
    "5A": "C minor",
    "6A": "G minor",
    "7A": "D minor",
    "8A": "A minor",
    "9A": "E minor",
    "10A": "B minor",
    "11A": "F# minor",
    "12A": "C# minor",
    "1B": "B major",
    "2B": "F# major",
    "3B": "Db major",
    "4B": "Ab major",
    "5B": "Eb major",
    "6B": "Bb major",
    "7B": "F major",
    "8B": "C major",
    "9B": "G major",
    "10B": "D major",
    "11B": "A major",
    "12B": "E major",
}


# Enharmonic spellings that are not in KEY_TO_CAMELOT under their own name.
# (Every common sharp and flat already is.)
_ENHARMONIC = {"E#": "F", "B#": "C", "Fb": "E", "Cb": "B"}


def _normalize_mode(mode: str | None) -> str | None:
    """``"major"``/``"minor"`` for the spellings we accept, else ``None``."""
    if mode is None:
        return None
    m = str(mode).strip().lower()
    if m in ("minor", "min", "m"):
        return "minor"
    if m in ("major", "maj", ""):
        return "major"
    return None


def parse_key(key: str | None, mode: str | None) -> tuple[str, str] | None:
    """Normalise a ``(key, mode)`` pair to a spelling present in
    :data:`KEY_TO_CAMELOT`, or ``None`` if it can't be understood."""
    if key is None:
        return None
    k = str(key).strip().capitalize()
    k = _ENHARMONIC.get(k, k)
    m = _normalize_mode(mode)
    if m is None or (k, m) not in KEY_TO_CAMELOT:
        return None
    return k, m


def key_to_camelot(key: str | None, mode: str | None) -> str | None:
    """
    Convert a musical key to Camelot notation.

    Args:
        key: Key name (e.g., "A", "Bb", "F#")
        mode: "major" or "minor" (also "maj"/"min"/"m")

    Returns:
        Camelot code (e.g., "8A", "11B"), or ``None`` when the input can't be
        parsed. There is deliberately no default: an unknown key stored as a
        real-looking "8A" would be scored as a harmonic match.
    """
    parsed = parse_key(key, mode)
    if parsed is None:
        logger.warning("Unparseable key %r / mode %r — no Camelot code", key, mode)
        return None
    return KEY_TO_CAMELOT[parsed]


def format_standard_key(key: str | None, mode: str | None) -> str | None:
    """Short standard key name: ``"C"`` for C major, ``"Am"`` for A minor.

    Returns ``None`` (and logs) for unparseable input, mirroring
    :func:`key_to_camelot` so the two columns are always both set or both
    empty.
    """
    parsed = parse_key(key, mode)
    if parsed is None:
        logger.warning("Unparseable key %r / mode %r — no standard key", key, mode)
        return None
    pitch, m = parsed
    return f"{pitch}m" if m == "minor" else pitch


def camelot_to_key(camelot: str) -> str:
    """
    Convert Camelot notation to standard key name.

    Args:
        camelot: Camelot code (e.g., "8A", "11B")

    Returns:
        Standard key name (e.g., "A minor", "A major")
    """
    return CAMELOT_TO_KEY.get(camelot.upper(), "C major")


def get_compatible_keys(camelot: str) -> list[str]:
    """
    Get harmonically compatible Camelot keys.

    Compatible keys are:
    - Same key (obviously)
    - Adjacent numbers (±1), same letter
    - Same number, different letter (relative major/minor)

    Args:
        camelot: Camelot code (e.g., "8A")

    Returns:
        List of compatible Camelot codes
    """
    camelot = camelot.upper()
    if len(camelot) < 2:
        return [camelot]

    try:
        number = int(camelot[:-1])
        letter = camelot[-1]
    except ValueError:
        return [camelot]

    compatible = [camelot]

    # Same number, different letter (relative major/minor)
    other_letter = "B" if letter == "A" else "A"
    compatible.append(f"{number}{other_letter}")

    # Adjacent numbers, same letter
    prev_num = 12 if number == 1 else number - 1
    next_num = 1 if number == 12 else number + 1
    compatible.append(f"{prev_num}{letter}")
    compatible.append(f"{next_num}{letter}")

    return compatible


def format_key_display(camelot: str, standard: str) -> str:
    """
    Format key for display (e.g., "8A (Am)").

    Args:
        camelot: Camelot code
        standard: Standard key name

    Returns:
        Formatted string for display
    """
    # Convert standard format to short form
    short = standard.replace(" minor", "m").replace(" major", "")
    return f"{camelot} ({short})"
