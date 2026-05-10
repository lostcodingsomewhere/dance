"""Qwen2-Audio deep tagger — listens to the audio directly and produces
free-form tags (dj_notes + unusual elements).

Slower (~10-30 s / track) and heavier (~8 GB weights, or ~4 GB at 4-bit)
than the CLAP zero-shot tagger. Use this for the long-tail descriptions
that don't fit into a fixed vocabulary.

The previous incarnation of this repo had a Qwen2-Audio integration that
never worked — wrong model class, missing chat template. This module avoids
those mistakes:

- Uses ``Qwen2AudioForConditionalGeneration`` (the correct class).
- Applies the chat template via the processor.
- Loads audio at the rate the processor expects (16 kHz).
- 4-bit quantization is opt-in via settings, not forced.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import (
    Tag,
    TagKind,
    TagSource,
    Track,
    normalize_tag_value,
    now_utc,
)
from dance.core.database import TrackTag
from dance.llm.brief import build_track_brief
from dance.pipeline.utils.db import upsert
from dance.pipeline.utils.device import pick_device

logger = logging.getLogger(__name__)


try:
    import librosa
    import torch
    from transformers import (
        AutoProcessor,
        Qwen2AudioForConditionalGeneration,
    )

    _QWEN_OK = True
except ImportError:  # pragma: no cover
    _QWEN_OK = False
    logger.info("Qwen2-Audio dependencies unavailable — deep tagger disabled")


_QWEN_SAMPLE_RATE = 16_000


_SYSTEM_PROMPT = (
    "You are a DJ's assistant tagging electronic dance music tracks. "
    "Listen to the audio. Use the analytical fingerprint as additional "
    "context (BPM, key, energy, stem presence). Reply with JSON of shape "
    '{"subgenre": "<one>", "mood_tags": [...2-4...], "element_tags": [...], '
    '"dj_notes": [...peak-time, long-intro, etc...]}'
    "\n\nReply with ONLY the JSON object, no prose, no markdown fences."
)


@dataclass
class TaggerResponse:
    subgenre: str | None = None
    mood_tags: list[str] = field(default_factory=list)
    element_tags: list[str] = field(default_factory=list)
    dj_notes: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    model: str | None = None
    raw_text: str | None = None

    def all_tags(self) -> list[tuple[TagKind, str]]:
        out: list[tuple[TagKind, str]] = []
        if self.subgenre:
            out.append((TagKind.SUBGENRE, self.subgenre))
        for v in self.mood_tags:
            out.append((TagKind.MOOD, v))
        for v in self.element_tags:
            out.append((TagKind.ELEMENT, v))
        for v in self.dj_notes:
            out.append((TagKind.DJ_NOTE, v))
        return out


# ---------------------------------------------------------------------------


class Qwen2AudioTagger:
    """Generative audio→tags via Qwen2-Audio (local)."""

    source = TagSource.LLM

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._processor = None
        self._device: str | None = None

    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not _QWEN_OK:
            raise RuntimeError(
                "Qwen2-Audio dependencies not installed — pip install transformers"
            )

        device = pick_device(self.settings.deep_tagger_device)
        name = self.settings.deep_tagger_model
        logger.info("Loading %s on %s (this is slow — ~8 GB)…", name, device)

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        quant = self.settings.deep_tagger_quantize
        if quant in {"4bit", "8bit"} and device != "mps":
            try:
                from transformers import BitsAndBytesConfig

                if quant == "4bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                else:
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("bitsandbytes unavailable — falling back to float16")
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16

        # MPS doesn't support device_map="auto"; load to CPU then move.
        if device == "mps":
            kwargs["device_map"] = None
        else:
            kwargs["device_map"] = "auto"

        self._processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(name, **kwargs)
        if device == "mps" and "quantization_config" not in kwargs:
            try:
                self._model = self._model.to("mps")
            except (RuntimeError, NotImplementedError) as exc:
                logger.warning("MPS load failed (%s); falling back to CPU", exc)
                device = "cpu"
                self._model = self._model.to("cpu")
        self._model.eval()
        self._device = device

    # ------------------------------------------------------------------

    def tag_track(self, session: Session, track: Track) -> TaggerResponse:
        if not self.settings.deep_tagger_enabled:
            raise RuntimeError("deep tagger disabled in settings")

        self._ensure_model()

        path = Path(track.file_path)
        if not path.exists():
            raise FileNotFoundError(f"audio not found: {path}")

        # Limit to ~60 s of audio; Qwen2-Audio handles long clips but cost
        # scales with length and 60 s is plenty to characterize a track.
        audio, _ = librosa.load(
            str(path), sr=_QWEN_SAMPLE_RATE, mono=True, duration=60.0
        )
        brief = build_track_brief(session, track)

        # Chat template format — list of content blocks per message.
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(path)},
                    {
                        "type": "text",
                        "text": (
                            "Analytical fingerprint:\n"
                            + brief
                            + "\n\nListen to the audio and emit JSON tags."
                        ),
                    },
                ],
            },
        ]

        # The processor expects the chat template applied to messages.
        text_prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = self._processor(
            text=text_prompt,
            audios=[audio],
            sampling_rate=_QWEN_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        # Move tensors to device.
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Strip the prompt prefix; decode only the new tokens.
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[:, input_len:]
        raw = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]
        logger.debug("Qwen2-Audio raw output: %s", raw)

        parsed = self._parse_json(raw)
        parsed.model = self.settings.deep_tagger_model
        parsed.raw_text = raw

        self._write_tags(session, track, parsed)
        return parsed

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> TaggerResponse:
        """Pull the first JSON object out of the model's text output."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("Qwen2-Audio reply had no JSON: %r", raw[:200])
            return TaggerResponse()
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("Qwen2-Audio JSON parse failed (%s): %r", exc, raw[:200])
            return TaggerResponse()

        return TaggerResponse(
            subgenre=data.get("subgenre"),
            mood_tags=[s for s in (data.get("mood_tags") or []) if isinstance(s, str)],
            element_tags=[s for s in (data.get("element_tags") or []) if isinstance(s, str)],
            dj_notes=[s for s in (data.get("dj_notes") or []) if isinstance(s, str)],
        )

    # ------------------------------------------------------------------

    def _write_tags(
        self,
        session: Session,
        track: Track,
        parsed: TaggerResponse,
    ) -> None:
        """Replace this track's tags from THIS source; keep other sources."""
        (
            session.query(TrackTag)
            .filter(
                TrackTag.track_id == track.id,
                TrackTag.source == self.source.value,
            )
            .delete(synchronize_session=False)
        )

        now = now_utc()
        seen: set[tuple[str, str]] = set()
        for kind, value in parsed.all_tags():
            value = value.strip()
            if not value:
                continue
            normalized = normalize_tag_value(value)
            key = (kind.value, normalized)
            if key in seen:
                continue
            seen.add(key)

            tag = upsert(
                session,
                Tag,
                where={"kind": kind.value, "normalized_value": normalized},
                value=value,
                created_at=now,
            )
            session.flush()

            upsert(
                session,
                TrackTag,
                where={
                    "track_id": track.id,
                    "tag_id": tag.id,
                    "source": self.source.value,
                },
                confidence=None,
                created_at=now,
            )
        session.commit()
