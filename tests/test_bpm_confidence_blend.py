"""Full-mix BPM confidence: Essentia's agreement score, raised to the share
of ticks on the fitted beat grid when that grid confirms the tempo.

Essentia's multifeature confidence grades how well its beat trackers agree
tick by tick, so breakdowns pull it down even when the tempo is exact: three
real house tracks whose fitted BPMs were right to 0.002 got 0.20, 0.35 and
0.48. The recommender multiplies BPM scores by ``bpm_confidence``, so exact
BPMs were counting for less than the old flat 0.7. The grid fit's inlier
share (0.56, 0.76, 0.85 on those tracks) is evidence about the BPM itself.

These tests drive ``estimate_bpm`` with a stub extractor so the tick layout,
and therefore the inlier share, is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dance.pipeline.utils import tempo

BPM = 124.0
PERIOD = 60.0 / BPM


class _StubExtractor:
    """Stands in for ``RhythmExtractor2013``: fixed ticks + raw confidence."""

    def __init__(self, ticks: np.ndarray, raw_confidence: float, bpm: float = BPM) -> None:
        self.ticks = np.asarray(ticks, dtype=np.float32)
        self.raw_confidence = raw_confidence
        self.bpm = bpm

    def __call__(self, *_a, **_k):
        return self.bpm, self.ticks, self.raw_confidence, np.array([self.bpm]), np.diff(self.ticks)

    def factory(self, *_a, **_k) -> _StubExtractor:
        return self


def _estimate(monkeypatch, stub: _StubExtractor) -> tuple[float, float]:
    assert tempo.ESSENTIA_AVAILABLE, "essentia must be installed for this test"
    monkeypatch.setattr(tempo.es, "RhythmExtractor2013", stub.factory)
    audio = np.random.default_rng(0).standard_normal(44100).astype(np.float32) * 0.1
    return tempo.estimate_bpm(audio, 44100)


def _grid(n: int) -> np.ndarray:
    return 0.37 + PERIOD * np.arange(n)


def test_clean_grid_lifts_a_low_essentia_score(monkeypatch):
    bpm, confidence = _estimate(monkeypatch, _StubExtractor(_grid(400), raw_confidence=1.0))

    assert bpm == pytest.approx(BPM, abs=1e-3)
    assert tempo.essentia_confidence(1.0) == pytest.approx(0.188, abs=1e-3)
    assert confidence == pytest.approx(1.0)


def test_confidence_is_the_share_of_ticks_on_the_grid(monkeypatch):
    on = _grid(300)
    off = on[:300] + 0.5 * PERIOD  # off-beat ticks: a tracker that slipped half a beat
    ticks = np.sort(np.concatenate([on, off]))

    bpm, confidence = _estimate(monkeypatch, _StubExtractor(ticks, raw_confidence=1.0))

    assert bpm == pytest.approx(BPM, abs=1e-3)
    assert confidence == pytest.approx(0.5, abs=0.01)


def test_essentia_score_wins_when_it_is_higher(monkeypatch):
    on = _grid(300)
    off = on + 0.5 * PERIOD
    ticks = np.sort(np.concatenate([on, off]))

    _, confidence = _estimate(monkeypatch, _StubExtractor(ticks, raw_confidence=4.79))

    assert confidence == pytest.approx(tempo.essentia_confidence(4.79))
    assert confidence == pytest.approx(0.9, abs=0.01)


def test_no_grid_means_no_lift(monkeypatch):
    """Ticks with no constant grid: the fit is rejected, Essentia's tempo and
    its own (low) confidence stand."""
    ticks = np.sort(np.random.default_rng(7).uniform(0.0, 200.0, 400))

    bpm, confidence = _estimate(monkeypatch, _StubExtractor(ticks, raw_confidence=1.0))

    assert bpm == pytest.approx(BPM)
    assert confidence == pytest.approx(tempo.essentia_confidence(1.0))


def test_a_disagreeing_grid_does_not_lift(monkeypatch):
    """A clean grid at a tempo more than 1 % away from Essentia's own BPM is
    not accepted as a refinement, so it can't vouch for that BPM either."""
    ticks = 0.37 + (60.0 / 127.0) * np.arange(400)

    bpm, confidence = _estimate(monkeypatch, _StubExtractor(ticks, raw_confidence=1.0))

    assert bpm == pytest.approx(BPM)
    assert confidence == pytest.approx(tempo.essentia_confidence(1.0))


def test_librosa_fallback_stays_capped(monkeypatch):
    class _Broken:
        def __init__(self, *_a, **_k):
            raise RuntimeError("simulated essentia failure")

    from tests.audio_fixtures import click_track

    monkeypatch.setattr(tempo.es, "RhythmExtractor2013", _Broken)
    bpm, confidence = tempo.estimate_bpm(click_track(BPM, 30.0), 44100)

    assert bpm == pytest.approx(BPM, abs=0.15)
    assert 0.0 < confidence <= tempo.LIBROSA_FIT_MAX_CONFIDENCE
