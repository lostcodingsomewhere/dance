"""Local track tagging — no API keys, runs entirely on-device.

Two modes:

- :class:`ClapZeroShotTagger` — fast (~50 ms / track), no extra weights.
  Re-uses the already-loaded CLAP model to rank a controlled vocabulary of
  candidate labels against each track's audio embedding. Limited to the
  vocabulary it knows about. ``source = inferred`` in ``track_tags``.

- :class:`Qwen2AudioTagger` — slow (~10-30 s / track), generative.
  Listens to the audio directly via Qwen2-Audio and produces free-form
  descriptions for ``dj_notes`` + unusual elements. Opt-in.
  ``source = llm`` in ``track_tags``.

Both write to the same ``tags`` / ``track_tags`` tables. Both leave manual
tags untouched. Either can be run via the CLI (``dance tag``) or the
``/api/v1/tracks/{id}/tag`` endpoint.
"""

from dance.llm.brief import build_track_brief
from dance.llm.qwen_audio import Qwen2AudioTagger
from dance.llm.tagger import ClapZeroShotTagger, TaggerResponse

# Default "Tagger" alias for back-compat with the CLI/API hooks.
Tagger = ClapZeroShotTagger


__all__ = [
    "build_track_brief",
    "ClapZeroShotTagger",
    "Qwen2AudioTagger",
    "Tagger",
    "TaggerResponse",
]
