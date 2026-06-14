"""StemSeparationStage — guard that the quality knobs (shifts, bit depth) flow
from Settings into Demucs + the WAV writer, so tuning config can't silently
no-op. Real Demucs is heavy (and downloads weights), so it's mocked here; the
actual separation math is exercised by real runs / the audio-quality A/B.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from dance.config import Settings
from dance.core.database import StemFile, TrackState
from dance.pipeline.stages import separate as sep


def test_separate_feeds_shifts_and_bit_depth_from_settings(tmp_path, session, make_track):
    settings = Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        demucs_shifts=3,
        demucs_bit_depth="PCM_24",
    )
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"RIFF0000WAVE")  # existence only — librosa.load is mocked
    track = make_track(
        file_path=str(audio),
        file_name=audio.name,
        state=TrackState.ANALYZED.value,
    )

    fake_model = MagicMock()
    fake_model.sources = ["drums", "bass", "vocals", "other"]
    fake_model.samplerate = 44100

    captured: dict = {"subtypes": []}

    def fake_apply_model(model, wav, **kwargs):
        captured["shifts"] = kwargs.get("shifts")
        srcs = []
        for _ in range(4):
            s = MagicMock()
            s.cpu.return_value.numpy.return_value = np.zeros((2, 64), dtype="float32")
            srcs.append(s)
        return [srcs]  # the stage indexes [0][i]

    def fake_sf_write(path, data, sr, subtype=None):
        captured["subtypes"].append(subtype)

    stage = sep.StemSeparationStage()

    def fake_ensure_model(_settings):
        stage._model = fake_model
        stage._model_name = "fake"

    with (
        patch.object(sep, "_DEMUCS_OK", True),
        patch.object(stage, "_ensure_model", fake_ensure_model),
        patch.object(sep, "apply_model", fake_apply_model),
        patch.object(sep, "sf") as msf,
        patch.object(sep, "torch") as mtorch,
        patch.object(
            sep.librosa,
            "load",
            lambda *a, **k: (np.zeros((2, 128), dtype="float32"), 44100),
        ),
    ):
        msf.write.side_effect = fake_sf_write
        mtorch.from_numpy.return_value.to.return_value = MagicMock()
        ok = stage.process(session, track, settings)

    assert ok is True
    # shifts came from settings (not the hardcoded 0), and every stem was
    # written at the configured bit depth (not the hardcoded PCM_16).
    assert captured["shifts"] == 3
    assert captured["subtypes"] == ["PCM_24"] * 4
    assert len(session.query(StemFile).filter_by(track_id=track.id).all()) == 4


def test_demucs_quality_defaults():
    """Defaults are the forward-looking quality settings we shipped."""
    s = Settings()
    assert s.demucs_shifts == 2
    assert s.demucs_bit_depth == "PCM_24"
