"""Generate Ableton Live Set (.als) files for a (track + 4 stems + regions) bundle.

The Live Set format is gzipped XML with a proprietary, undocumented schema.
We build a *minimal* file programmatically using lxml: the bare elements
Live needs to parse a Set (root ``Ableton`` element, ``LiveSet`` with
``Tracks`` / ``MasterTrack``, an AudioTrack per stem with a Sample
reference, a ``Locators`` collection for cues/sections, and master Tempo).
This is verified by:

1. The output decompresses with ``gzip`` and parses as well-formed XML.
2. The element tree matches a hand-crafted reference schema and our spec
   (5 audio tracks, the right tempo, the right locator count, etc.).

What we deliberately do NOT do is try to fake every device chain, mixer
parameter, automation envelope, etc. Live tolerates many missing fields
when opening older Sets — and worst case the user gets a Set that opens
with one tempo + five clips on five named, colored tracks, which is the
goal.

The single public entry point is :class:`AlsGenerator`.
"""

from __future__ import annotations

from dance.als.generator import AlsGenerator

__all__ = ["AlsGenerator"]
