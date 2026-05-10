"""Smoke tests for the synthetic audio fixtures themselves.

We don't test perceptual quality — we just make sure:
- BPM detection on the synthetic mix lands near the configured BPM
- The file roundtrips to/from disk
- Determinism: same seed → same audio
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.audio_fixtures import SR, TrackSpec, synth_track, write_track


def test_synth_track_shape():
    spec = TrackSpec(bars=4)
    audio = synth_track(spec)
    assert audio.dtype == np.float32
    assert len(audio) == int(spec.duration_seconds * SR)
    assert audio.max() <= 1.0
    assert audio.min() >= -1.0


def test_synth_track_deterministic():
    a = synth_track(TrackSpec(bars=2, seed=7))
    b = synth_track(TrackSpec(bars=2, seed=7))
    assert np.array_equal(a, b)


def test_synth_track_seed_changes_audio():
    a = synth_track(TrackSpec(bars=2, seed=7))
    b = synth_track(TrackSpec(bars=2, seed=11))
    # Different seeds change the noise-based hats and kick clicks.
    assert not np.array_equal(a, b)


def test_write_track_roundtrip(tmp_path):
    out = write_track(tmp_path / "synth.wav", TrackSpec(bars=2))
    assert out.exists()
    assert out.stat().st_size > 0

    import soundfile as sf

    data, sr = sf.read(str(out))
    assert sr == SR
    assert len(data) > 0


@pytest.mark.skipif(
    not pytest.importorskip("librosa", reason="librosa not installed"),
    reason="librosa required",
)
def test_synth_bpm_near_target():
    """librosa.beat.beat_track should land within ~3 BPM of the target."""
    import librosa

    spec = TrackSpec(bpm=128.0, bars=8)
    audio = synth_track(spec)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=SR)
    tempo = float(tempo) if not hasattr(tempo, "__len__") else float(tempo[0])
    # Allow half/double-time matches too.
    diffs = [abs(tempo - 128.0), abs(tempo - 64.0), abs(tempo - 256.0)]
    assert min(diffs) < 4.0, f"BPM detection too far off: {tempo}"
