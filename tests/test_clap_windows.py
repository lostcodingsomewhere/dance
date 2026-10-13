"""Deterministic, windowed CLAP embeddings.

The old stage handed whole tracks to a feature extractor configured with
``truncation="rand_trunc"``, which embeds ONE RANDOM 10 s crop — a different
one on every run. These tests pin the replacement down:

* window selection is deterministic, covers the audio evenly and skips
  near-silent windows (pure numpy, no model);
* the stage writes the current scheme version and leaves exactly one row per
  (track, stem, model), replacing older schemes;
* every reader ignores rows from other schemes;
* opt-in: the real laion/clap-htsat-unfused model, if it is already in the
  local HF cache (never downloaded), gives identical vectors run to run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from dance import cli
from dance.config import Settings, get_settings
from dance.core.database import (
    AudioAnalysis,
    StemFile,
    StemKind,
    Track,
    TrackEmbedding,
    TrackState,
    get_session,
    init_db,
    now_utc,
)
from dance.core.embedding_scheme import CLAP_EMBED_VERSION
from dance.core.serialization import decode_embedding, encode_embedding
from dance.pipeline.stages import embed as embed_mod
from dance.pipeline.stages.embed import EmbeddingStage, stale_track_ids
from dance.pipeline.utils.clap_windows import (
    extract_windows,
    pool_embeddings,
    select_windows,
    spread,
    window_rms_db,
    window_starts,
)
from tests.audio_fixtures import TrackSpec, synth_track, write_track

MODEL = "laion/clap-htsat-unfused"
SR = 48000
# Tiny windows keep the fake-model stage tests fast; the logic is length-agnostic.
WIN = SR  # 1 s
DIM = 16


# ---------------------------------------------------------------------------
# Pure window selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 999, 1000, 1001, 2500, 10_000, 10_001, 123_457])
def test_window_starts_cover_every_sample_evenly(n):
    w = 1000
    starts = window_starts(n, w)
    assert starts == window_starts(n, w)  # deterministic
    assert starts[0] == 0
    if n <= w:
        assert starts == [0]
        return
    assert len(starts) == math.ceil(n / w)
    assert starts[-1] + w == n  # tail handled by an end-aligned window
    assert starts == sorted(starts)
    gaps = np.diff(starts)
    assert gaps.max() <= w  # no uncovered sample
    assert gaps.max() - gaps.min() <= 1  # overlap spread evenly


def test_window_starts_empty_audio():
    assert window_starts(0, 1000) == []


def test_spread_takes_segment_centres_in_order():
    items = list(range(36))
    picked = spread(items, 12)
    assert picked == [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
    assert spread(items, 1) == [18]  # a cap of 1 takes the middle, not the intro
    assert spread([5, 9], 12) == [5, 9]
    for m in range(1, 40):
        for k in range(1, m + 1):
            got = spread(list(range(m)), k)
            assert len(got) == k and len(set(got)) == k and got == sorted(got)


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sparse_vocal_stem() -> np.ndarray:
    """30 s: phrases at 4-7 s and 21-25 s, otherwise digital silence."""
    audio = np.zeros(30 * SR, dtype=np.float32)
    audio[4 * SR : 7 * SR] = _tone(3)
    audio[21 * SR : 25 * SR] = _tone(4, 330.0)
    return audio


def test_select_windows_skips_silent_windows():
    audio = _sparse_vocal_stem()
    starts = select_windows(audio, WIN, max_windows=100, silence_db=-40.0)
    assert starts, "the phrases must be found"
    for s in starts:
        window = audio[s : s + WIN]
        assert np.sqrt(np.mean(window.astype(np.float64) ** 2)) > 0.01
    # Every phrase second is covered by some chosen window.
    covered = np.zeros(audio.size, dtype=bool)
    for s in starts:
        covered[s : s + WIN] = True
    assert covered[4 * SR : 7 * SR].all() and covered[21 * SR : 25 * SR].all()


def test_select_windows_all_silent_returns_nothing():
    quiet = (1e-4 * np.random.default_rng(0).standard_normal(20 * SR)).astype(np.float32)
    assert select_windows(np.zeros(20 * SR, np.float32), WIN, max_windows=12, silence_db=-40) == []
    assert select_windows(quiet, WIN, max_windows=12, silence_db=-40) == []  # -80 dBFS hiss
    assert select_windows(np.zeros(0, np.float32), WIN, max_windows=12, silence_db=-40) == []


def test_select_windows_cap_spreads_over_non_silent_content():
    # 60 s: first 20 s silent (long intro), then 40 s of content.
    audio = np.concatenate([np.zeros(20 * SR, np.float32), _tone(40)])
    starts = select_windows(audio, WIN, max_windows=8, silence_db=-40.0)
    assert len(starts) == 8
    assert starts == select_windows(audio, WIN, max_windows=8, silence_db=-40.0)
    assert min(starts) >= 19 * SR  # nothing from the silent intro
    secs = np.array(starts) / SR
    # Evenly spread across the 40 s of content: one window per 5 s segment.
    assert np.allclose(np.diff(secs), 5.0, atol=0.1)
    assert secs[0] < 25 and secs[-1] > 54


def test_window_rms_db_levels():
    audio = np.concatenate([np.zeros(WIN, np.float32), np.full(WIN, 0.1, np.float32)])
    levels = window_rms_db(audio, [0, WIN], WIN)
    assert levels[0] == -math.inf
    assert levels[1] == pytest.approx(-20.0, abs=1e-3)


def test_extract_windows_exact_length_and_repeatpad():
    audio = np.arange(1, 2501, dtype=np.float32)
    wins = extract_windows(audio, window_starts(audio.size, 1000), 1000)
    assert [w.shape[0] for w in wins] == [1000, 1000, 1000]
    assert wins[-1][-1] == 2500  # tail reached without padding
    short = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    (padded,) = extract_windows(short, [0], 8)
    # CLAP repeatpad: whole copies, then zeros.
    assert padded.tolist() == [1, 2, 3, 1, 2, 3, 0, 0]


def test_pool_embeddings_normalised_mean_of_unit_rows():
    rows = np.array([[3.0, 0.0], [0.0, 0.5], [0.0, 10.0]], dtype=np.float32)
    pooled = pool_embeddings(rows)
    assert pooled.dtype == np.float32
    assert np.linalg.norm(pooled) == pytest.approx(1.0, abs=1e-6)
    # Per-row scale is irrelevant: unit rows are [1,0], [0,1], [0,1].
    expected = np.array([1.0, 2.0]) / np.sqrt(5.0)
    assert np.allclose(pooled, expected, atol=1e-6)
    with pytest.raises(ValueError):
        pool_embeddings(np.zeros((0, 4), np.float32))


def test_extractor_never_crops_our_windows():
    """The bug and the fix, on HF's own extractor (no weights needed).

    ``rand_trunc`` takes a random crop of anything longer than max_length;
    windows of exactly max_length go through untouched, so features repeat.
    """
    pytest.importorskip("transformers")
    from transformers import ClapFeatureExtractor

    fe = ClapFeatureExtractor(truncation="rand_trunc", padding="repeatpad")
    max_len = fe.nb_max_samples
    audio = synth_track(TrackSpec(bars=14, sample_rate=SR))  # ~26 s
    assert audio.size > max_len

    np.random.seed(0)
    a = fe(audio, sampling_rate=SR, return_tensors="np")["input_features"]
    b = fe(audio, sampling_rate=SR, return_tensors="np")["input_features"]
    assert not np.allclose(a, b), "whole-track input is randomly cropped"

    starts = select_windows(audio, max_len, max_windows=12, silence_db=-40.0)
    wins = extract_windows(audio, starts, max_len)
    c = fe(wins, sampling_rate=SR, return_tensors="np")["input_features"]
    d = fe(wins, sampling_rate=SR, return_tensors="np")["input_features"]
    assert np.array_equal(c, d)


# ---------------------------------------------------------------------------
# Stage with a fake, deterministic, content-sensitive "model"
# ---------------------------------------------------------------------------


def _fake_encode_windows(stage: EmbeddingStage, calls: list[list[int]]):
    def enc(windows: list[np.ndarray]) -> np.ndarray:
        calls.append([int(w.shape[0]) for w in windows])
        assert all(w.shape[0] == stage._window for w in windows)
        feats = [
            np.array([np.abs(c).mean() for c in np.array_split(w, DIM)]) + 1e-3 for w in windows
        ]
        return np.stack(feats).astype(np.float32)

    return enc


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        clap_max_windows=4,
    )


@pytest.fixture
def window_calls() -> list[list[int]]:
    return []


@pytest.fixture
def stage(monkeypatch, window_calls) -> EmbeddingStage:
    st = EmbeddingStage()

    def _ensure(settings, *, local_files_only=False):
        st._model = object()
        st._processor = object()
        st._device = "cpu"
        st._model_name = settings.clap_model
        st._window = WIN

    monkeypatch.setattr(st, "_ensure_model", _ensure)
    monkeypatch.setattr(st, "_encode_windows", _fake_encode_windows(st, window_calls))
    return st


def test_encode_is_deterministic_and_batched(stage, settings, window_calls, monkeypatch):
    monkeypatch.setattr(embed_mod, "_BATCH", 3)
    stage._ensure_model(settings)
    stage._configure(settings.model_copy(update={"clap_max_windows": 7}))
    audio = synth_track(TrackSpec(bars=8, sample_rate=SR))  # 15 s -> 15 candidates
    a = stage._encode(audio)
    b = stage._encode(audio)
    assert a is not None and np.array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-6)
    # 7 windows per call, in batches of <= 3, each exactly one window long.
    assert [len(c) for c in window_calls] == [3, 3, 1, 3, 3, 1]
    assert {n for c in window_calls for n in c} == {WIN}


def test_encode_silent_audio_returns_none(stage, settings, window_calls):
    stage._ensure_model(settings)
    stage._configure(settings)
    assert stage._encode(np.zeros(5 * SR, np.float32)) is None
    assert window_calls == []


def _emb(session, track_id, stem_id, version, vec, model=MODEL) -> TrackEmbedding:
    v = np.asarray(vec, dtype=np.float32)
    row = TrackEmbedding(
        track_id=track_id,
        stem_file_id=stem_id,
        model=model,
        model_version=version,
        dim=int(v.shape[0]),
        embedding=encode_embedding(v),
    )
    session.add(row)
    return row


@pytest.fixture
def stem_track(tmp_path, session, make_track):
    """A REGIONS_DETECTED track with 4 stems; vocals is digital silence."""
    spec = TrackSpec(bpm=128.0, bars=4)
    mix = write_track(tmp_path / "mix.wav", spec)
    track = make_track(
        file_path=str(mix), file_name=mix.name, state=TrackState.REGIONS_DETECTED.value
    )
    for kind in (StemKind.DRUMS, StemKind.BASS, StemKind.VOCALS, StemKind.OTHER):
        path = tmp_path / f"{kind.value}.wav"
        if kind == StemKind.VOCALS:
            sf.write(str(path), np.zeros(spec.total_samples, np.float32), spec.sample_rate)
        else:
            write_track(path, TrackSpec(bpm=128.0, bars=4, seed=len(kind.value)))
        session.add(StemFile(track_id=track.id, kind=kind.value, path=str(path)))
    session.commit()
    return track


def _seed_legacy_rows(session, track) -> None:
    """What the old stage left behind: NULL-version rows, plus a stray older
    scheme, plus another model's row that must survive."""
    stems = session.query(StemFile).filter_by(track_id=track.id).all()
    for sid in [None, *[s.id for s in stems]]:
        _emb(session, track.id, sid, None, np.ones(DIM))
    _emb(session, track.id, None, "older-scheme", np.ones(DIM))
    _emb(session, track.id, None, None, np.ones(8), model="other/model")
    session.commit()


def _rows(session, track_id, model=MODEL) -> list[TrackEmbedding]:
    return session.query(TrackEmbedding).filter_by(track_id=track_id, model=model).all()


def test_process_replaces_old_scheme_rows(session, stage, settings, stem_track):
    _seed_legacy_rows(session, stem_track)
    assert len(_rows(session, stem_track.id)) == 6

    assert stage.process(session, stem_track, settings) is True
    session.expire_all()

    rows = _rows(session, stem_track.id)
    # Mix + drums/bass/other. Vocals is silent: no row, and its legacy row is gone.
    assert len(rows) == 4
    assert {r.model_version for r in rows} == {CLAP_EMBED_VERSION}
    assert len({r.stem_file_id for r in rows}) == 4  # one per (track, stem, model)
    vocals = session.query(StemFile).filter_by(track_id=stem_track.id, kind="vocals").one()
    assert vocals.id not in {r.stem_file_id for r in rows}
    for r in rows:
        v = decode_embedding(r.embedding, r.dim)
        assert r.dim == DIM and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
    # Another model's rows are not ours to delete.
    assert len(_rows(session, stem_track.id, model="other/model")) == 1
    assert session.get(Track, stem_track.id).state == TrackState.COMPLETE.value


def test_reprocess_is_idempotent_and_identical(session, stage, settings, stem_track):
    assert stage.process(session, stem_track, settings) is True
    first = {r.stem_file_id: bytes(r.embedding) for r in _rows(session, stem_track.id)}
    stem_track.state = TrackState.REGIONS_DETECTED.value
    session.commit()
    assert stage.process(session, stem_track, settings) is True
    session.expire_all()
    second = {r.stem_file_id: bytes(r.embedding) for r in _rows(session, stem_track.id)}
    assert first == second


def test_embed_track_on_complete_track_keeps_state(session, stage, settings, stem_track):
    stem_track.state = TrackState.COMPLETE.value
    session.commit()
    _seed_legacy_rows(session, stem_track)
    written, silent = stage.embed_track(session, stem_track, settings)
    session.commit()
    assert (written, silent) == (4, 1)
    assert stem_track.state == TrackState.COMPLETE.value
    assert {r.model_version for r in _rows(session, stem_track.id)} == {CLAP_EMBED_VERSION}


def test_failure_mid_track_leaves_rows_untouched(session, stage, settings, stem_track, monkeypatch):
    _seed_legacy_rows(session, stem_track)
    before = sorted((r.id, r.model_version) for r in _rows(session, stem_track.id))
    real_encode = stage._encode
    seen = {"n": 0}

    def flaky(audio):
        seen["n"] += 1
        if seen["n"] == 3:
            raise RuntimeError("MPS fell over")
        return real_encode(audio)

    monkeypatch.setattr(stage, "_encode", flaky)
    assert stage.process(session, stem_track, settings) is False
    session.expire_all()
    after = sorted((r.id, r.model_version) for r in _rows(session, stem_track.id))
    assert after == before
    assert session.get(Track, stem_track.id).state == TrackState.ERROR.value


def test_stale_track_ids(session, make_track):
    def track(state=TrackState.COMPLETE.value):
        return make_track(state=state)

    legacy = track()
    _emb(session, legacy.id, None, None, np.ones(4))
    fresh = track()
    _emb(session, fresh.id, None, CLAP_EMBED_VERSION, np.ones(4))
    half = track()
    stem = StemFile(track_id=half.id, kind="drums", path="/tmp/x.wav")
    session.add(stem)
    session.flush()
    _emb(session, half.id, None, CLAP_EMBED_VERSION, np.ones(4))
    _emb(session, half.id, stem.id, "older-scheme", np.ones(4))
    missing = track()
    in_flight = track(TrackState.REGIONS_DETECTED.value)
    _emb(session, in_flight.id, None, None, np.ones(4))
    other_model = track()
    _emb(session, other_model.id, None, CLAP_EMBED_VERSION, np.ones(4))
    _emb(session, other_model.id, None, None, np.ones(4), model="other/model")
    session.commit()

    assert stale_track_ids(session, MODEL) == sorted([legacy.id, half.id, missing.id])


# ---------------------------------------------------------------------------
# Readers ignore other schemes
# ---------------------------------------------------------------------------

STALE = [0.0, 1.0]
CURRENT = [1.0, 0.0]


@pytest.fixture
def mixed_library(session, make_track):
    """Track A: stale + current rows (vectors differ). Track B: stale only."""
    a = make_track(state=TrackState.COMPLETE.value)
    b = make_track(state=TrackState.COMPLETE.value)
    stems = {}
    for t in (a, b):
        s = StemFile(track_id=t.id, kind="drums", path=f"/tmp/{t.id}-drums.wav")
        session.add(s)
        session.flush()
        stems[t.id] = s.id
        now = now_utc()
        session.add(AudioAnalysis(track_id=t.id, bpm=128.0, key_camelot="8A", analyzed_at=now))
        session.add(AudioAnalysis(track_id=t.id, stem_file_id=s.id, bpm=128.0, analyzed_at=now))
        for sid in (None, s.id):
            _emb(session, t.id, sid, None, STALE)
            _emb(session, t.id, sid, "older-scheme", STALE)
    for sid in (None, stems[a.id]):
        _emb(session, a.id, sid, CLAP_EMBED_VERSION, CURRENT)
    session.commit()
    return a, b, stems


def test_tagger_reads_current_scheme_only(session, mixed_library, settings):
    from dance.llm.tagger import ClapZeroShotTagger

    a, b, _ = mixed_library
    tagger = ClapZeroShotTagger(settings, text_encoder=lambda s: np.ones(2, np.float32))
    assert np.allclose(tagger._track_embedding(session, a), CURRENT)
    assert tagger._track_embedding(session, b) is None


def test_journey_reads_current_scheme_only(session, mixed_library):
    from dance.recommender import journey as jn

    a, b, stems = mixed_library
    vecs = jn._ordered_fullmix_vecs(session, [a.id, b.id])
    assert len(vecs) == 1 and np.allclose(vecs[0], CURRENT)
    combo = jn._combo_embedding(session, [stems[a.id], stems[b.id]])
    assert np.allclose(combo, CURRENT)
    assert jn._combo_embedding(session, [stems[b.id]]) is None


def test_graph_builder_ignores_stale_rows(session, mixed_library, make_track, settings):
    from dance.recommender.graph_builder import GraphBuilder

    a, b, _ = mixed_library
    gb = GraphBuilder(session, settings)
    assert gb._build_embedding(None) == []  # only one current track
    c = make_track(state=TrackState.COMPLETE.value)
    _emb(session, c.id, None, CLAP_EMBED_VERSION, [0.9, 0.1])
    session.commit()
    edges = gb._build_embedding(None)
    pairs = {(e.from_track_id, e.to_track_id) for e in edges}
    assert pairs == {(a.id, c.id), (c.id, a.id)}


def test_text_search_ignores_stale_rows(session, mixed_library):
    from dance.recommender.recommender import Recommender

    a, b, _ = mixed_library
    res = Recommender(session).recommend_by_text("x", lambda q: np.array(STALE, np.float32))
    assert [r.track_id for r in res] == [a.id]
    assert res[0].score == pytest.approx(0.0, abs=1e-6)  # scored on CURRENT, not STALE


def test_column_and_tail_candidates_ignore_stale_rows(session, mixed_library):
    from dance.recommender import tail
    from dance.recommender.recommender import _ColumnRecommenderImpl

    a, b, _ = mixed_library
    impl = _ColumnRecommenderImpl(session)
    for cands in (
        impl._mix_candidates(set()),
        impl._stem_candidates("drums", set()),
        tail._candidates(session, set()),
    ):
        by_track = {}
        for c in cands:
            assert c.track_id not in by_track, "stale rows must not duplicate candidates"
            by_track[c.track_id] = c
        assert np.allclose(by_track[a.id].embedding, CURRENT)
        assert by_track[b.id].embedding is None


# ---------------------------------------------------------------------------
# `dance embed`
# ---------------------------------------------------------------------------


def test_cli_embed_reembeds_stale_complete_tracks(tmp_path, monkeypatch):
    calls: list[list[int]] = []

    def _ensure(self, settings, *, local_files_only=False):
        self._model = object()
        self._processor = object()
        self._device = "cpu"
        self._model_name = settings.clap_model
        self._window = WIN

    def _enc(self, windows):
        return _fake_encode_windows(self, calls)(windows)

    monkeypatch.setattr(EmbeddingStage, "_ensure_model", _ensure)
    monkeypatch.setattr(EmbeddingStage, "_encode_windows", _enc)

    settings = get_settings()  # isolated tmp home (conftest)
    settings.ensure_directories()
    init_db(settings.db_url)
    session = get_session(settings.db_url)
    good_audio = write_track(tmp_path / "good.wav", TrackSpec(bars=4))
    good = Track(
        file_hash="a" * 64,
        file_path=str(good_audio),
        file_name="good.wav",
        file_size_bytes=1,
        title="Good",
        state=TrackState.COMPLETE.value,
    )
    gone = Track(
        file_hash="b" * 64,
        file_path=str(tmp_path / "gone.wav"),
        file_name="gone.wav",
        file_size_bytes=1,
        title="Gone",
        state=TrackState.COMPLETE.value,
    )
    session.add_all([good, gone])
    session.flush()
    for t in (good, gone):
        _emb(session, t.id, None, None, np.ones(DIM))
    session.commit()
    good_id, gone_id = good.id, gone.id
    session.close()

    runner = CliRunner()
    result = runner.invoke(cli.main, ["embed"])
    assert result.exit_code == 0, result.output
    assert "Embedding 2 track(s)" in result.output
    assert "Embedded: 1" in result.output and "Errors: 1" in result.output

    session = get_session(settings.db_url)
    try:
        good_rows = _rows(session, good_id)
        assert [r.model_version for r in good_rows] == [CLAP_EMBED_VERSION]
        gone_rows = _rows(session, gone_id)
        assert [r.model_version for r in gone_rows] == [None]  # failure kept old rows
        assert {t.state for t in session.query(Track).all()} == {TrackState.COMPLETE.value}
    finally:
        session.close()

    again = runner.invoke(cli.main, ["embed", "--track-id", str(good_id)])
    assert again.exit_code == 0, again.output
    assert "Embedding 1 track(s)" in again.output


# ---------------------------------------------------------------------------
# Opt-in: the real model, only if already cached locally (never downloads)
# ---------------------------------------------------------------------------

_REAL: dict[str, EmbeddingStage] = {}


@pytest.fixture
def real_stage(monkeypatch, tmp_path) -> EmbeddingStage:
    pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    if MODEL not in _REAL:
        st = EmbeddingStage()
        s = Settings(data_dir=tmp_path / "d", clap_device="cpu")
        try:
            st._ensure_model(s, local_files_only=True)
        except OSError as exc:  # transformers' "not found locally"; code bugs still fail
            pytest.skip(f"{MODEL} not in the local HF cache ({type(exc).__name__})")
        st._configure(s)
        _REAL[MODEL] = st
    return _REAL[MODEL]


def _pad_chord(seconds: float) -> np.ndarray:
    """Unrelated material: a slow, drumless sustained chord with tremolo."""
    t = np.arange(int(seconds * SR)) / SR
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0, 493.9))
    return (0.2 * chord * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))).astype(np.float32)


def test_real_clap_is_deterministic_and_representative(real_stage):
    signal = synth_track(TrackSpec(bpm=128.0, bars=24, sample_rate=SR))  # 45 s
    assert signal.size == 45 * SR

    a = real_stage._encode(signal)
    b = real_stage._encode(signal)
    assert a is not None and b is not None
    assert float(a @ b) > 0.99999

    half = signal.size // 2
    first, second = real_stage._encode(signal[:half]), real_stage._encode(signal[half:])
    other = real_stage._encode(_pad_chord(45))
    same = float(first @ second)
    assert same > float(first @ other)
    assert same > float(second @ other)
