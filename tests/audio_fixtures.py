"""Synthetic audio generators for tests.

Designed to give the pipeline real-feeling input without committing audio
files or requiring downloads. The track has clearly separable layers so that
even simple analyzers (BPM detection, RMS-based energy, stem separation)
produce sane results in tests.

Defaults: 128 BPM, 4/4, 16-bar drop with a breakdown — short enough to keep
tests fast (~10 sec at 44.1kHz mono = ~440k samples = ~1.7 MB float32).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


SR = 44100


@dataclass(frozen=True)
class TrackSpec:
    bpm: float = 128.0
    bars: int = 8                  # ~15s at 128 BPM
    sample_rate: int = SR
    seed: int = 42

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_seconds(self) -> float:
        return 4 * self.beat_seconds

    @property
    def duration_seconds(self) -> float:
        return self.bars * self.bar_seconds

    @property
    def total_samples(self) -> int:
        return int(self.duration_seconds * self.sample_rate)


# ---------------------------------------------------------------------------
# Layer generators — each returns a float32 array of length total_samples
# ---------------------------------------------------------------------------


def _envelope(sr: int, attack_s: float, decay_s: float) -> np.ndarray:
    """Simple percussive AD envelope."""
    n_attack = max(1, int(attack_s * sr))
    n_decay = max(1, int(decay_s * sr))
    attack = np.linspace(0.0, 1.0, n_attack, dtype=np.float32)
    decay = np.exp(-np.linspace(0.0, 4.0, n_decay, dtype=np.float32))
    return np.concatenate([attack, decay])


def kick_layer(spec: TrackSpec) -> np.ndarray:
    """Four-on-the-floor kick drum: one hit per beat."""
    rng = np.random.default_rng(spec.seed)
    out = np.zeros(spec.total_samples, dtype=np.float32)
    env = _envelope(spec.sample_rate, 0.002, 0.18)
    n_beats = int(spec.duration_seconds / spec.beat_seconds)
    for i in range(n_beats):
        t_start = int(i * spec.beat_seconds * spec.sample_rate)
        t = np.arange(len(env)) / spec.sample_rate
        # Pitched-down sine (kick fundamental) with click
        freq = 60.0 * np.exp(-t * 12.0)  # pitch drop
        sample = (np.sin(2 * np.pi * np.cumsum(freq) / spec.sample_rate) * env).astype(np.float32)
        sample += 0.02 * rng.standard_normal(len(env)).astype(np.float32) * env  # click
        end = min(t_start + len(sample), len(out))
        out[t_start:end] += sample[: end - t_start]
    return out * 0.7


def bass_layer(spec: TrackSpec) -> np.ndarray:
    """Pulsing bass on every offbeat, root note A2 (~110 Hz)."""
    out = np.zeros(spec.total_samples, dtype=np.float32)
    env = _envelope(spec.sample_rate, 0.005, 0.12)
    eighth = spec.beat_seconds / 2.0
    n_eighths = int(spec.duration_seconds / eighth)
    freq = 110.0  # A2
    for i in range(n_eighths):
        if i % 2 == 0:
            continue  # only on the offbeats
        t_start = int(i * eighth * spec.sample_rate)
        t = np.arange(len(env)) / spec.sample_rate
        sample = (np.sin(2 * np.pi * freq * t) * env).astype(np.float32)
        end = min(t_start + len(sample), len(out))
        out[t_start:end] += sample[: end - t_start]
    return out * 0.5


def hat_layer(spec: TrackSpec) -> np.ndarray:
    """16th-note closed hi-hats — burst of high-freq noise."""
    rng = np.random.default_rng(spec.seed + 1)
    out = np.zeros(spec.total_samples, dtype=np.float32)
    env = _envelope(spec.sample_rate, 0.001, 0.04)
    sixteenth = spec.beat_seconds / 4.0
    n_steps = int(spec.duration_seconds / sixteenth)
    for i in range(n_steps):
        t_start = int(i * sixteenth * spec.sample_rate)
        noise = rng.standard_normal(len(env)).astype(np.float32)
        # High-pass via differentiator
        noise = np.diff(noise, prepend=0.0)
        sample = noise * env
        end = min(t_start + len(sample), len(out))
        out[t_start:end] += sample[: end - t_start]
    return out * 0.25


def vocal_layer(spec: TrackSpec) -> np.ndarray:
    """Mid-frequency sustained tone in the second half of the track."""
    out = np.zeros(spec.total_samples, dtype=np.float32)
    start_sample = int(spec.total_samples * 0.5)
    n = spec.total_samples - start_sample
    t = np.arange(n) / spec.sample_rate
    # Vibrato around 440 Hz (A4)
    freq = 440.0 + 5.0 * np.sin(2 * np.pi * 5.0 * t)
    sample = np.sin(2 * np.pi * np.cumsum(freq) / spec.sample_rate).astype(np.float32)
    # Fade in
    fade = np.minimum(t * 4.0, 1.0).astype(np.float32)
    out[start_sample:] = sample * fade * 0.35
    return out


def _arrangement_envelope(spec: TrackSpec) -> np.ndarray:
    """Drop/breakdown gain envelope at the bar level.

    Bars 0-1 = intro (low), 2-3 = build, 4-5 = drop (full), 6 = breakdown (low), 7 = outro (mid).
    """
    out = np.ones(spec.total_samples, dtype=np.float32)
    sr = spec.sample_rate
    levels = [0.4, 0.5, 0.7, 0.9, 1.0, 1.0, 0.3, 0.6]
    for bar in range(min(spec.bars, len(levels))):
        start = int(bar * spec.bar_seconds * sr)
        end = int((bar + 1) * spec.bar_seconds * sr)
        out[start:end] = levels[bar]
    return out


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def synth_track(spec: TrackSpec | None = None) -> np.ndarray:
    """Render a full mix as a single float32 mono array in [-1, 1]."""
    spec = spec or TrackSpec()
    kick = kick_layer(spec)
    bass = bass_layer(spec)
    hat = hat_layer(spec)
    vocal = vocal_layer(spec)
    arrangement = _arrangement_envelope(spec)

    mix = (kick + bass + hat + vocal) * arrangement
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.9  # normalize, leave a tiny headroom
    return mix.astype(np.float32)


def write_track(path: Path, spec: TrackSpec | None = None) -> Path:
    """Render and write a track to ``path``. Returns the path."""
    spec = spec or TrackSpec()
    audio = synth_track(spec)
    sf.write(str(path), audio, spec.sample_rate, subtype="PCM_16")
    return path
