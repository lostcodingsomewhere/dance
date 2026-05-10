"""Tests for the CLAP zero-shot tagger and the Qwen2-Audio deep tagger.

Both taggers are exercised with a fake text encoder so no real CLAP or Qwen
weights are loaded. The shape and DB writes are what's validated.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    Tag,
    TagKind,
    TagSource,
    Track,
    TrackEmbedding,
    TrackTag,
    now_utc,
)
from dance.core.serialization import encode_embedding
from dance.llm.brief import build_track_brief
from dance.llm.tagger import (
    DJ_NOTE_LABELS,
    ELEMENT_LABELS,
    MOOD_LABELS,
    SUBGENRE_LABELS,
    ClapZeroShotTagger,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        clap_model="test-clap",
        tagger_zeroshot_threshold=0.1,
        tagger_zeroshot_top_k={
            "subgenre": 1,
            "mood": 2,
            "element": 2,
            "dj_note": 2,
        },
    )


def _add_fullmix_embedding(session: Session, track_id: int, vec: np.ndarray, model: str) -> None:
    session.add(
        TrackEmbedding(
            track_id=track_id,
            stem_file_id=None,
            model=model,
            model_version=None,
            dim=int(vec.shape[0]),
            embedding=encode_embedding(vec.astype(np.float32)),
            created_at=now_utc(),
        )
    )


# ---------------------------------------------------------------------------
# CLAP zero-shot tagger
# ---------------------------------------------------------------------------


def test_zeroshot_picks_highest_similarity_label(session, make_track, settings):
    """The label whose embedding aligns with the track's gets picked."""
    track = make_track()
    # Track embedding points "forward" — only one specific label is built
    # to point the same way; others point sideways. Whatever that label
    # turns out to be (depends on vocab list order), the tagger must pick it.
    audio_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _add_fullmix_embedding(session, track.id, audio_vec, settings.clap_model)
    session.commit()

    # Build a fake text encoder that returns a label-specific vector.
    # For "tech house" we return [1,0,0] (matches audio). Everything else
    # returns [0,1,0]. This guarantees "tech house" wins.
    def fake_encoder(label: str) -> np.ndarray:
        if label == "tech house":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    tagger = ClapZeroShotTagger(settings, text_encoder=fake_encoder)
    result = tagger.tag_track(session, track)

    assert result.subgenre == "tech house"


def test_zeroshot_threshold_drops_low_scores(session, make_track, settings):
    """If every score is below the threshold, no tags are emitted."""
    track = make_track()
    audio_vec = np.array([1.0, 0.0], dtype=np.float32)
    _add_fullmix_embedding(session, track.id, audio_vec, settings.clap_model)
    session.commit()

    # Every label gets a perpendicular vector → cosine = 0, below threshold.
    fake_encoder = lambda _label: np.array([0.0, 1.0], dtype=np.float32)

    settings.tagger_zeroshot_threshold = 0.5
    tagger = ClapZeroShotTagger(settings, text_encoder=fake_encoder)
    result = tagger.tag_track(session, track)

    assert result.subgenre is None
    assert result.mood_tags == []
    assert result.element_tags == []
    assert result.dj_notes == []


def test_zeroshot_writes_tags_with_inferred_source(session, make_track, settings):
    track = make_track()
    audio_vec = np.array([1.0, 0.0], dtype=np.float32)
    _add_fullmix_embedding(session, track.id, audio_vec, settings.clap_model)
    session.commit()

    fake_encoder = lambda _label: np.array([1.0, 0.0], dtype=np.float32)
    ClapZeroShotTagger(settings, text_encoder=fake_encoder).tag_track(session, track)

    rows = (
        session.query(TrackTag)
        .filter(TrackTag.track_id == track.id)
        .all()
    )
    assert len(rows) > 0
    assert all(r.source == TagSource.INFERRED.value for r in rows)
    # Confidence is the cosine score — should be in [0, 1].
    for r in rows:
        if r.confidence is not None:
            assert 0.0 <= r.confidence <= 1.0


def test_zeroshot_replaces_existing_inferred_tags_keeps_manual(session, make_track, settings):
    track = make_track()
    audio_vec = np.array([1.0, 0.0], dtype=np.float32)
    _add_fullmix_embedding(session, track.id, audio_vec, settings.clap_model)

    # Pre-seed an old inferred tag + a manual tag.
    old_tag = Tag(
        kind=TagKind.MOOD.value, value="ancient", normalized_value="ancient",
        created_at=now_utc(),
    )
    manual_tag = Tag(
        kind=TagKind.MOOD.value, value="kept", normalized_value="kept",
        created_at=now_utc(),
    )
    session.add_all([old_tag, manual_tag])
    session.flush()
    session.add_all(
        [
            TrackTag(
                track_id=track.id, tag_id=old_tag.id,
                source=TagSource.INFERRED.value, created_at=now_utc(),
            ),
            TrackTag(
                track_id=track.id, tag_id=manual_tag.id,
                source=TagSource.MANUAL.value, created_at=now_utc(),
            ),
        ]
    )
    session.commit()

    fake_encoder = lambda _label: np.array([1.0, 0.0], dtype=np.float32)
    ClapZeroShotTagger(settings, text_encoder=fake_encoder).tag_track(session, track)

    # The old inferred tag for "ancient" should be gone.
    rows = (
        session.query(TrackTag)
        .filter(TrackTag.track_id == track.id, TrackTag.source == TagSource.INFERRED.value)
        .all()
    )
    inferred_tag_ids = {r.tag_id for r in rows}
    assert old_tag.id not in inferred_tag_ids

    # The manual tag should be preserved.
    manual_still = (
        session.query(TrackTag)
        .filter(
            TrackTag.track_id == track.id,
            TrackTag.source == TagSource.MANUAL.value,
            TrackTag.tag_id == manual_tag.id,
        )
        .one_or_none()
    )
    assert manual_still is not None


def test_zeroshot_errors_when_no_embedding(session, make_track, settings):
    """Without a CLAP embedding for the track, the tagger raises."""
    track = make_track()  # no embedding inserted
    session.commit()

    fake_encoder = lambda _label: np.array([1.0, 0.0], dtype=np.float32)
    tagger = ClapZeroShotTagger(settings, text_encoder=fake_encoder)
    with pytest.raises(RuntimeError, match="no full-mix CLAP embedding"):
        tagger.tag_track(session, track)


def test_zeroshot_label_matrix_cached(session, make_track, settings):
    """The text encoder should be called once per label, not per track."""
    t1 = make_track()
    t2 = make_track()
    for t in (t1, t2):
        _add_fullmix_embedding(
            session, t.id, np.array([1.0, 0.0], dtype=np.float32), settings.clap_model
        )
    session.commit()

    calls: list[str] = []

    def counting_encoder(label: str) -> np.ndarray:
        calls.append(label)
        return np.array([1.0, 0.0], dtype=np.float32)

    tagger = ClapZeroShotTagger(settings, text_encoder=counting_encoder)
    tagger.tag_track(session, t1)
    calls_after_first = len(calls)
    tagger.tag_track(session, t2)

    # Second call must not re-encode any label.
    assert len(calls) == calls_after_first
    # And the first call should have encoded the full vocabulary exactly once.
    expected_count = len(SUBGENRE_LABELS) + len(MOOD_LABELS) + len(ELEMENT_LABELS) + len(DJ_NOTE_LABELS)
    assert calls_after_first == expected_count


def test_zeroshot_disabled_raises(session, make_track, settings):
    settings.tagger_enabled = False
    fake_encoder = lambda _label: np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match="disabled"):
        ClapZeroShotTagger(settings, text_encoder=fake_encoder).tag_track(session, make_track())


def test_zeroshot_respects_clap_model_filter(session, make_track, settings):
    """Only embeddings produced by settings.clap_model are considered."""
    track = make_track()
    # Embedding under a DIFFERENT model name.
    _add_fullmix_embedding(
        session, track.id, np.array([1.0, 0.0], dtype=np.float32),
        model="some-other-model",
    )
    session.commit()

    fake_encoder = lambda _label: np.array([1.0, 0.0], dtype=np.float32)
    tagger = ClapZeroShotTagger(settings, text_encoder=fake_encoder)
    with pytest.raises(RuntimeError, match="no full-mix CLAP embedding"):
        tagger.tag_track(session, track)


# ---------------------------------------------------------------------------
# Brief builder
# ---------------------------------------------------------------------------


def test_brief_handles_empty_track(session, make_track):
    """No analysis, no stems, no regions → still returns a string."""
    track = make_track()
    session.commit()
    brief = build_track_brief(session, track)
    assert isinstance(brief, str)


def test_brief_includes_bpm_and_key(session, make_track):
    track = make_track()
    session.add(
        AudioAnalysis(
            track_id=track.id, stem_file_id=None,
            bpm=128.0, key_camelot="8A", key_standard="Am",
            floor_energy=7, analyzed_at=now_utc(),
        )
    )
    session.commit()
    brief = build_track_brief(session, track)
    assert "128.0" in brief
    assert "8A" in brief
    assert "energy 7/10" in brief


# ---------------------------------------------------------------------------
# Qwen2-Audio JSON parser
# ---------------------------------------------------------------------------


def test_qwen_parses_clean_json():
    from dance.llm.qwen_audio import Qwen2AudioTagger

    raw = '{"subgenre": "tech house", "mood_tags": ["dark", "driving"], "element_tags": ["rolling bassline"], "dj_notes": ["peak-time"]}'
    parsed = Qwen2AudioTagger._parse_json(raw)
    assert parsed.subgenre == "tech house"
    assert parsed.mood_tags == ["dark", "driving"]
    assert parsed.element_tags == ["rolling bassline"]
    assert parsed.dj_notes == ["peak-time"]


def test_qwen_parses_json_inside_prose():
    """The model sometimes prepends explanation; we should still extract."""
    from dance.llm.qwen_audio import Qwen2AudioTagger

    raw = 'Here are the tags:\n\n{"subgenre": "afro house", "mood_tags": ["groovy"]}\n\nLet me know if you need more.'
    parsed = Qwen2AudioTagger._parse_json(raw)
    assert parsed.subgenre == "afro house"
    assert parsed.mood_tags == ["groovy"]


def test_qwen_handles_missing_json():
    from dance.llm.qwen_audio import Qwen2AudioTagger

    parsed = Qwen2AudioTagger._parse_json("no json here at all")
    assert parsed.subgenre is None
    assert parsed.mood_tags == []


def test_qwen_drops_non_string_entries():
    from dance.llm.qwen_audio import Qwen2AudioTagger

    raw = '{"mood_tags": ["dark", 123, null, "driving"]}'
    parsed = Qwen2AudioTagger._parse_json(raw)
    assert parsed.mood_tags == ["dark", "driving"]
