"""Analysis fixes: precise BPM, honest key names, per-stem tempo, decoded
duration.

The BPM tests run the REAL :class:`AnalysisStage` with REAL essentia on
synthetic accented click tracks at tempos that are NOT librosa lag bins
(124.0 and 131.0 sit between librosa's 123.05 / 126.05 and 129.20 / 132.51
bins at 44.1 kHz, hop 512), so a quantised detector cannot pass them.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    StemFile,
    StemKind,
    Track,
    TrackState,
    now_utc,
)
from dance.pipeline.stages.analyze import AnalysisStage
from dance.pipeline.stages.analyze_stems import StemAnalysisStage
from dance.pipeline.utils import tempo
from dance.pipeline.utils.camelot import (
    KEY_TO_CAMELOT,
    format_standard_key,
    key_to_camelot,
)
from tests.audio_fixtures import (
    TrackSpec,
    bass_layer,
    click_track,
    drum_layer,
    pad_layer,
    vocal_layer,
    write_click_track,
    write_layer,
    write_track,
)

# librosa.beat.beat_track's only reachable tempos near house range at
# sr=44100 / hop=512 — the values the whole library collapsed onto.
LIBROSA_LAG_BINS = (
    120.18531976744185,
    123.046875,
    126.04801829268293,
    129.19921875,
    132.51201923076923,
)

TEMPO_LOGGER = "dance.pipeline.utils.tempo"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
    )


def _fullmix(session, track_id: int) -> AudioAnalysis:
    return session.query(AudioAnalysis).filter_by(track_id=track_id, stem_file_id=None).one()


# ---------------------------------------------------------------------------
# (a) BPM — real essentia, end to end through the stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [124.0, 131.0])
def test_analyze_stage_bpm_is_precise_with_real_essentia(
    tmp_path, session, make_track, settings, target
):
    assert tempo.ESSENTIA_AVAILABLE, "essentia must be installed for this test"
    path = write_click_track(tmp_path / f"click_{target:g}.wav", target, seconds=30)
    track = make_track(file_path=str(path), file_name=path.name)
    session.commit()

    assert AnalysisStage().process(session, track, settings) is True

    row = _fullmix(session, track.id)
    assert abs(row.bpm - target) < 0.15, row.bpm
    assert all(abs(row.bpm - b) > 0.5 for b in LIBROSA_LAG_BINS), row.bpm
    assert row.bpm_confidence != pytest.approx(0.7)
    assert 0.0 <= row.bpm_confidence <= 1.0


def test_bpm_comes_from_essentia_even_when_librosa_beat_tracker_is_broken(
    tmp_path, session, make_track, settings, monkeypatch
):
    import librosa

    def _boom(*_a, **_k):
        raise AssertionError("librosa.beat.beat_track must not be needed")

    monkeypatch.setattr(librosa.beat, "beat_track", _boom)
    path = write_click_track(tmp_path / "click124.wav", 124.0, seconds=30)
    track = make_track(file_path=str(path), file_name=path.name)
    session.commit()

    assert AnalysisStage().process(session, track, settings) is True
    row = _fullmix(session, track.id)
    assert abs(row.bpm - 124.0) < 0.15
    # Essentia's 0..5.32 confidence normalised: a clean click is "excellent".
    assert row.bpm_confidence > 3.5 / tempo.ESSENTIA_MAX_CONFIDENCE


class _BrokenRhythmExtractor:
    def __init__(self, *_a, **_k):
        pass

    def __call__(self, *_a, **_k):
        raise RuntimeError("simulated essentia failure")


@pytest.mark.parametrize("target", [124.0, 131.0])
def test_librosa_fallback_fits_beats_instead_of_quantised_tempo(monkeypatch, caplog, target):
    monkeypatch.setattr(tempo.es, "RhythmExtractor2013", _BrokenRhythmExtractor)
    with caplog.at_level(logging.WARNING, logger=TEMPO_LOGGER):
        bpm, confidence = tempo.estimate_bpm(click_track(target, 30.0), 44100)

    assert "simulated essentia failure" in caplog.text
    assert abs(bpm - target) < 0.15, bpm
    assert all(abs(bpm - b) > 0.5 for b in LIBROSA_LAG_BINS)
    assert 0.0 < confidence <= tempo.LIBROSA_FIT_MAX_CONFIDENCE
    assert confidence != pytest.approx(0.7)


def test_librosa_fallback_through_the_stage(tmp_path, session, make_track, settings, monkeypatch):
    monkeypatch.setattr(tempo.es, "RhythmExtractor2013", _BrokenRhythmExtractor)
    path = write_click_track(tmp_path / "click131.wav", 131.0, seconds=30)
    track = make_track(file_path=str(path), file_name=path.name)
    session.commit()

    assert AnalysisStage().process(session, track, settings) is True
    row = _fullmix(session, track.id)
    assert abs(row.bpm - 131.0) < 0.15
    assert row.bpm_confidence <= tempo.LIBROSA_FIT_MAX_CONFIDENCE


def test_silence_has_no_tempo_rather_than_a_fake_one():
    # Essentia alone reports ~369 BPM at 0.88 confidence on digital silence.
    with pytest.raises(RuntimeError, match="silent"):
        tempo.estimate_bpm(np.zeros(44100 * 10, dtype=np.float32), 44100)


def test_essentia_confidence_normalisation():
    assert tempo.essentia_confidence(0.0) == 0.0
    assert tempo.essentia_confidence(tempo.ESSENTIA_MAX_CONFIDENCE) == 1.0
    assert tempo.essentia_confidence(2.66) == pytest.approx(0.5)
    assert tempo.essentia_confidence(9.0) == 1.0
    assert tempo.essentia_confidence(float("nan")) == 0.0


# ---------------------------------------------------------------------------
# Beat-grid fit
# ---------------------------------------------------------------------------


def _grid(bpm: float, n: int, offset: float = 0.37) -> np.ndarray:
    return offset + np.arange(n) * 60.0 / bpm


def test_fit_beat_grid_is_precise_with_jitter():
    rng = np.random.default_rng(7)
    beats = _grid(124.0, 300) + rng.normal(0.0, 0.004, 300)
    fit = tempo.fit_beat_grid(beats)
    assert fit is not None
    assert abs(fit.bpm - 124.0) < 0.01
    assert fit.inlier_ratio > 0.95


def test_fit_beat_grid_survives_skips_spurious_ticks_and_breakdowns():
    beats = list(_grid(131.0, 400))
    del beats[200:232]  # a 32-beat breakdown with no ticks at all
    beats = [b for i, b in enumerate(beats) if i % 17 != 5]  # skipped beats
    period = 60.0 / 131.0
    beats += [beats[i] + 0.4 * period for i in range(3, len(beats), 23)]  # extras
    fit = tempo.fit_beat_grid(beats)
    assert fit is not None
    assert abs(fit.bpm - 131.0) < 0.02, fit


def test_fit_beat_grid_ignores_a_section_where_the_tracker_wandered():
    """The failure seen on real library tracks: Essentia's ticks ran at
    ~146 BPM through a breakdown, so a line through all ticks (or through
    tick *indices*) came out at 136 BPM and Essentia's own histogram BPM was
    0.11-0.17 off. Intro and outro are on the true 125 BPM grid."""
    hop = 512 / 44100
    period = 60.0 / 125.0
    intro = 0.5 + np.arange(250) * period
    wander_start = intro[-1] + period
    wander = wander_start + np.arange(260) * (60.0 / 146.0)
    outro_start = intro[0] + np.ceil((wander[-1] - intro[0]) / period + 1) * period
    outro = outro_start + np.arange(250) * period
    ticks = np.round(np.concatenate([intro, wander, outro]) / hop) * hop
    fit = tempo.fit_beat_grid(ticks, approx_bpm=124.89)
    assert fit is not None
    assert abs(fit.bpm - 125.0) < 0.01, fit
    assert 0.5 < fit.inlier_ratio < 0.8


def test_fit_beat_grid_averages_out_hop_quantisation():
    # Ticks snapped to a 512-sample hop: a median-of-slopes estimator alone
    # locks onto a rational number of hops per beat (124.03 here).
    hop = 512 / 44100
    beats = np.round(_grid(124.0, 64) / hop) * hop
    fit = tempo.fit_beat_grid(beats)
    assert fit is not None
    assert abs(fit.bpm - 124.0) < 0.01, fit.bpm


def test_fit_beat_grid_needs_enough_beats():
    assert tempo.fit_beat_grid(_grid(128.0, 5)) is None
    assert tempo.fit_beat_grid([]) is None


# ---------------------------------------------------------------------------
# (b) Key formatting + Camelot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, mode, expected",
    [
        ("C", "major", "C"),
        ("A", "minor", "Am"),
        ("F#", "minor", "F#m"),
        ("Bb", "maj", "Bb"),
        ("c", "m", "Cm"),
        ("Eb", "Minor", "Ebm"),
        ("E#", "major", "F"),
    ],
)
def test_format_standard_key(key, mode, expected):
    assert format_standard_key(key, mode) == expected


def test_major_key_is_no_longer_stored_as_minor():
    # The old f"{key}{mode[0]}" turned "C major" into "Cm" ('major'[0] == 'm').
    assert format_standard_key("C", "major") == "C"
    assert key_to_camelot("C", "major") == "8B"


@pytest.mark.parametrize(
    "key, mode", [("H", "major"), ("C", "dorian"), (None, "minor"), ("A", None), ("", "")]
)
def test_unparseable_key_returns_none_and_logs(caplog, key, mode):
    with caplog.at_level(logging.WARNING, logger="dance.pipeline.utils.camelot"):
        assert key_to_camelot(key, mode) is None
        assert format_standard_key(key, mode) is None
    assert "Unparseable key" in caplog.text


def test_enharmonic_spellings_map_correctly():
    assert key_to_camelot("A#", "minor") == key_to_camelot("Bb", "minor") == "3A"
    assert key_to_camelot("B#", "major") == key_to_camelot("C", "major") == "8B"


@pytest.mark.parametrize("key, mode", sorted(KEY_TO_CAMELOT))
def test_standard_key_minor_iff_camelot_a(key, mode):
    standard = format_standard_key(key, mode)
    camelot = key_to_camelot(key, mode)
    assert standard is not None and camelot is not None
    assert standard.endswith("m") == camelot.endswith("A")


@pytest.mark.parametrize(
    "detected, expected",
    [
        (("C", "major", 0.9), ("C", "8B")),
        (("A", "minor", 0.9), ("Am", "8A")),
        (("Eb", "major", 0.8), ("Eb", "5B")),
        (("H", "major", 0.9), (None, None)),
    ],
)
def test_analyze_stage_stores_consistent_key_columns(
    tmp_path, session, make_track, settings, monkeypatch, detected, expected
):
    monkeypatch.setattr(AnalysisStage, "_analyze_key", lambda self, a, sr: detected)
    path = write_click_track(tmp_path / "k.wav", 124.0, seconds=12)
    track = make_track(file_path=str(path), file_name=path.name)
    session.commit()

    assert AnalysisStage().process(session, track, settings) is True
    row = _fullmix(session, track.id)
    assert (row.key_standard, row.key_camelot) == expected
    if row.key_camelot is None:
        assert row.key_confidence is None
    else:
        assert row.key_standard.endswith("m") == row.key_camelot.endswith("A")


# ---------------------------------------------------------------------------
# (c) Per-stem tempo: inherited BPM + coherence
# ---------------------------------------------------------------------------


STEM_LAYERS = {
    StemKind.DRUMS: drum_layer,
    StemKind.BASS: bass_layer,
    StemKind.VOCALS: vocal_layer,
    StemKind.OTHER: pad_layer,  # a sustained pad: no pulse at all
}


def _separated_track(tmp_path, session, make_track, *, bpm=124.0, fullmix_bpm=124.0):
    spec = TrackSpec(bpm=bpm, bars=16)
    mix = write_track(tmp_path / "mix.wav", spec)
    track = make_track(file_path=str(mix), file_name=mix.name, state=TrackState.SEPARATED.value)
    for kind, layer in STEM_LAYERS.items():
        path = write_layer(tmp_path / f"{kind.value}.wav", layer(spec))
        session.add(StemFile(track_id=track.id, kind=kind.value, path=str(path), model_used="test"))
    if fullmix_bpm is not None:
        session.add(
            AudioAnalysis(
                track_id=track.id, stem_file_id=None, bpm=fullmix_bpm, analyzed_at=now_utc()
            )
        )
    session.commit()
    return track


def _stem_rows(session, track_id: int) -> dict[str, AudioAnalysis]:
    rows = (
        session.query(StemFile.kind, AudioAnalysis)
        .join(AudioAnalysis, AudioAnalysis.stem_file_id == StemFile.id)
        .filter(StemFile.track_id == track_id)
        .all()
    )
    return {kind: row for kind, row in rows}


def test_stem_rows_inherit_track_bpm_and_score_coherence(tmp_path, session, make_track, settings):
    track = _separated_track(tmp_path, session, make_track)
    assert StemAnalysisStage().process(session, track, settings) is True

    rows = _stem_rows(session, track.id)
    assert set(rows) == {k.value for k in STEM_LAYERS}
    for kind, row in rows.items():
        assert row.bpm == 124.0, kind
        assert 0.0 <= row.bpm_confidence <= 1.0, kind

    drums = rows[StemKind.DRUMS.value].bpm_confidence
    pad = rows[StemKind.OTHER.value].bpm_confidence
    assert drums > 0.6, drums
    assert drums - pad > 0.3, (drums, pad)


def test_tempo_coherence_edges():
    assert tempo.tempo_coherence(np.zeros(22050 * 5, dtype=np.float32), 22050, 128.0) == 0.0
    assert tempo.tempo_coherence(np.ones(10, dtype=np.float32), 22050, None) == 0.0
    assert tempo.tempo_coherence(np.ones(10, dtype=np.float32), 22050, 0.0) == 0.0


def test_missing_fullmix_row_falls_back_to_estimating_the_track_tempo(
    tmp_path, session, make_track, settings, caplog
):
    track = _separated_track(tmp_path, session, make_track, fullmix_bpm=None)
    with caplog.at_level(logging.WARNING, logger="dance.pipeline.stages.analyze_stems"):
        assert StemAnalysisStage().process(session, track, settings) is True
    assert "no full-mix BPM" in caplog.text

    bpms = {row.bpm for row in _stem_rows(session, track.id).values()}
    assert len(bpms) == 1
    (bpm,) = bpms
    assert abs(bpm - 124.0) < 0.15


def test_no_tempo_anywhere_leaves_stem_bpm_empty(tmp_path, session, make_track, settings):
    track = make_track(file_path=str(tmp_path / "gone.wav"), state=TrackState.SEPARATED.value)
    silent = np.zeros(44100 * 5, dtype=np.float32)
    for kind in StemKind:
        path = tmp_path / f"{kind.value}.wav"
        write_layer(path, silent)
        session.add(StemFile(track_id=track.id, kind=kind.value, path=str(path), model_used="test"))
    session.commit()

    assert StemAnalysisStage().process(session, track, settings) is True
    for row in _stem_rows(session, track.id).values():
        assert row.bpm is None
        assert row.bpm_confidence is None


# ---------------------------------------------------------------------------
# (d) Duration from the decoded audio
# ---------------------------------------------------------------------------


def test_duration_is_overwritten_and_a_big_mismatch_warns(
    tmp_path, session, make_track, settings, caplog
):
    path = write_click_track(tmp_path / "short.wav", 124.0, seconds=20)
    track = make_track(
        file_path=str(path), file_name=path.name, title="Wrong Version", duration_seconds=300.0
    )
    session.commit()

    with caplog.at_level(logging.WARNING, logger="dance.pipeline.stages.analyze"):
        assert AnalysisStage().process(session, track, settings) is True

    session.refresh(track)
    assert track.duration_seconds == pytest.approx(20.0, abs=0.05)
    assert "Duration mismatch" in caplog.text
    assert "Wrong Version" in caplog.text


def test_small_duration_difference_is_corrected_silently(
    tmp_path, session, make_track, settings, caplog
):
    path = write_click_track(tmp_path / "close.wav", 124.0, seconds=20)
    track = make_track(file_path=str(path), file_name=path.name, duration_seconds=21.5)
    session.commit()

    with caplog.at_level(logging.WARNING, logger="dance.pipeline.stages.analyze"):
        assert AnalysisStage().process(session, track, settings) is True

    session.refresh(track)
    assert track.duration_seconds == pytest.approx(20.0, abs=0.05)
    assert "Duration mismatch" not in caplog.text


@pytest.mark.parametrize(
    "stored, decoded, warns",
    [
        (412.0, 400.0, False),  # 12 s < 3.5 % of 400 s (14 s)
        (415.0, 400.0, True),
        (189.0, 200.0, True),  # 11 s > max(10 s, 7 s)
        (191.0, 200.0, False),
        (None, 200.0, False),
    ],
)
def test_duration_mismatch_threshold(caplog, stored, decoded, warns):
    track = Track(id=1, title="T", artist="A", duration_seconds=stored)
    with caplog.at_level(logging.WARNING, logger="dance.pipeline.stages.analyze"):
        AnalysisStage._reconcile_duration(track, decoded)
    assert track.duration_seconds == decoded
    assert ("Duration mismatch" in caplog.text) is warns
