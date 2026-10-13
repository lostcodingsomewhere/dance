"""Stem-separation hygiene: atomic stem writes, reconstruction-residual QC,
provenance strings, the MPS->CPU retry, and waveform-sidecar invalidation
when a stem is rewritten at the same path.

Demucs itself is faked (no weights in CI); the WAV writes, the residual math
and the waveform cache run for real on synthetic audio.
"""

from __future__ import annotations

import contextlib
import json
import os
from importlib.metadata import version as dist_version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from dance.audio import waveform as wf
from dance.config import Settings
from dance.core.database import StemFile, TrackState
from dance.pipeline.stages import separate as sep

SR = 44100
KINDS = ["drums", "bass", "vocals", "other"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Just enough of a torch tensor for ``StemSeparationStage._apply``."""

    def __init__(self, arr: np.ndarray, device: str = "cpu") -> None:
        self.arr = arr
        self.device = device

    def to(self, device: str) -> _FakeTensor:
        return _FakeTensor(self.arr, device)

    def __getitem__(self, idx):  # wav[None]
        return _FakeTensor(self.arr[idx], self.device)

    def cpu(self) -> _FakeTensor:
        return _FakeTensor(self.arr, "cpu")

    def numpy(self) -> np.ndarray:
        return self.arr


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        from_numpy=lambda a: _FakeTensor(np.asarray(a)),
        no_grad=contextlib.nullcontext,
        mps=SimpleNamespace(empty_cache=lambda: None),
    )


class _FakeModel:
    sources = KINDS
    samplerate = SR

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.moved_to: list[str] = []

    def to(self, device: str) -> _FakeModel:
        self.moved_to.append(device)
        return self

    def eval(self) -> _FakeModel:
        return self


# Each stem is a fixed fraction of the mix; the fractions sum to 0.95, so the
# reconstruction residual is exactly 0.05 * mix -> 20*log10(0.05) dB.
_FRACTIONS = [0.25, 0.25, 0.25, 0.20]
_EXPECTED_RESIDUAL_DB = 20 * np.log10(0.05)


def _mix(seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    left = 0.4 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    return np.stack([left, right]).astype("float32")


def _sources_for(mix: np.ndarray) -> _FakeTensor:
    stacked = np.stack([f * mix for f in _FRACTIONS]).astype("float32")
    return _FakeTensor(stacked[None])  # batch dim; the stage indexes [0]


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        **kw,
    )


def _track(tmp_path: Path, make_track):
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"RIFF0000WAVE")  # existence only — librosa.load is patched
    return make_track(file_path=str(audio), file_name=audio.name, state=TrackState.ANALYZED.value)


def _stage(model: _FakeModel, device: str | None) -> sep.StemSeparationStage:
    stage = sep.StemSeparationStage()

    def fake_ensure(_settings):
        stage._model = model
        stage._model_name = "htdemucs_ft"
        stage._device = device

    stage._ensure_model = fake_ensure  # type: ignore[method-assign]
    return stage


def _run(stage, session, track, settings, mix, apply_fn, get_model_fn=None, sf_write=None):
    patches = [
        patch.object(sep, "_DEMUCS_OK", True),
        patch.object(sep, "torch", _fake_torch()),
        patch.object(sep, "apply_model", apply_fn),
        patch.object(sep.librosa, "load", lambda *a, **k: (mix, SR)),
    ]
    if get_model_fn is not None:
        patches.append(patch.object(sep, "get_model", get_model_fn))
    if sf_write is not None:
        patches.append(patch.object(sep.sf, "write", sf_write))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return stage.process(session, track, settings)


def _no_partials(d: Path) -> bool:
    return not list(d.glob(".*.partial.wav")) and not list(d.glob("*.partial.wav"))


# ---------------------------------------------------------------------------
# a) Atomic writes
# ---------------------------------------------------------------------------


def _stem_arrays(scale: float = 0.5) -> list[tuple[str, np.ndarray]]:
    mix = _mix(0.1)
    return [(k, (scale * mix).T.copy()) for k in KINDS]


def test_write_stems_atomic_writes_all_and_leaves_no_temps(tmp_path):
    out = tmp_path / "abcd1234"
    out.mkdir()
    paths = sep.write_stems_atomic(out, _stem_arrays(), SR, "PCM_24")
    assert sorted(paths) == sorted(KINDS)
    for k in KINDS:
        info = sf.info(str(out / f"{k}.wav"))
        assert info.samplerate == SR and info.channels == 2 and info.subtype == "PCM_24"
        assert oct((out / f"{k}.wav").stat().st_mode & 0o777) == oct(0o644)
    assert _no_partials(out)


def test_write_stems_atomic_failure_midway_leaves_previous_stems_intact(tmp_path):
    out = tmp_path / "abcd1234"
    out.mkdir()
    sep.write_stems_atomic(out, _stem_arrays(0.5), SR, "PCM_24")
    before = {k: (out / f"{k}.wav").read_bytes() for k in KINDS}

    real_write = sf.write
    calls = {"n": 0}

    def flaky_write(path, data, sr, subtype=None):
        calls["n"] += 1
        if calls["n"] == 3:
            # Simulate dying mid-file: some bytes hit disk, then an error.
            Path(path).write_bytes(b"RIFF\x00\x00trunc")
            raise OSError("disk full")
        return real_write(path, data, sr, subtype=subtype)

    with patch.object(sep.sf, "write", flaky_write), pytest.raises(OSError, match="disk full"):
        sep.write_stems_atomic(out, _stem_arrays(0.9), SR, "PCM_24")

    # Every final stem is byte-identical to the previous complete set, and no
    # temp (complete or truncated) was left behind.
    assert {k: (out / f"{k}.wav").read_bytes() for k in KINDS} == before
    assert _no_partials(out)
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{k}.wav" for k in KINDS)


def test_sweep_partials_removes_crash_leftovers_only(tmp_path):
    out = tmp_path / "abcd1234"
    out.mkdir()
    (out / ".drums.x1y2.partial.wav").write_bytes(b"trunc")
    (out / ".bass.z9.partial.wav").write_bytes(b"trunc")
    (out / "drums.wav").write_bytes(b"keep")
    (out / "drums.wav.waveform.json").write_text("{}")
    assert sep.sweep_partials(out) == 2
    assert sorted(p.name for p in out.iterdir()) == ["drums.wav", "drums.wav.waveform.json"]


def test_stage_write_failure_leaves_no_stem_files_or_rows(tmp_path, session, make_track):
    settings = _settings(tmp_path)
    track = _track(tmp_path, make_track)
    mix = _mix()
    stage = _stage(_FakeModel(), "cpu")
    out_dir = settings.stems_dir / track.file_hash[:8]
    out_dir.mkdir(parents=True)
    (out_dir / ".vocals.stale.partial.wav").write_bytes(b"from a SIGBUS")

    real_write = sf.write
    calls = {"n": 0}

    def flaky_write(path, data, sr, subtype=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom mid-separation-write")
        return real_write(path, data, sr, subtype=subtype)

    with pytest.raises(RuntimeError, match="boom"):
        _run(
            stage,
            session,
            track,
            settings,
            mix,
            lambda model, wav, **kw: _sources_for(mix),
            sf_write=flaky_write,
        )
    session.rollback()
    assert list(out_dir.iterdir()) == []  # no <kind>.wav, no temps, stale temp swept
    assert session.query(StemFile).filter_by(track_id=track.id).count() == 0


# ---------------------------------------------------------------------------
# b) Residual QC
# ---------------------------------------------------------------------------


def test_residual_db_known_ratio():
    mix = _mix()
    stems = [f * mix for f in _FRACTIONS]
    got = sep.reconstruction_residual_db(mix, stems)
    assert got == pytest.approx(_EXPECTED_RESIDUAL_DB, abs=1e-3)  # -26.02 dB


def test_residual_db_perfect_reconstruction_is_floored_not_inf():
    mix = _mix().astype(np.float64)
    stems = [0.25 * mix] * 4
    assert sep.reconstruction_residual_db(mix, stems) == -200.0


def test_residual_db_silent_mix_is_none():
    z = np.zeros((2, 1000), dtype="float32")
    assert sep.reconstruction_residual_db(z, [z, z]) is None
    assert sep.reconstruction_residual_db(np.zeros((2, 0)), []) is None


def test_residual_db_reflects_write_clipping():
    # A stem that exceeds full scale is hard-clipped by the PCM writer, so
    # the stems Live plays no longer sum to the mix. clip=None ignores that.
    mix = np.full((2, 1000), 0.8, dtype="float32")
    stems = [np.full_like(mix, 1.6), np.full_like(mix, -0.8)]
    assert sep.reconstruction_residual_db(mix, stems, clip=None) == -200.0
    # clipped: 1.0 - 0.8 = 0.2 vs mix 0.8 -> residual 0.6/0.8
    assert sep.reconstruction_residual_db(mix, stems) == pytest.approx(
        20 * np.log10(0.6 / 0.8), abs=1e-6
    )


def test_residual_db_chunking_and_length_mismatch():
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((2, 10_001)).astype("float32") * 0.1
    stems = [0.5 * mix, 0.4 * mix[:, :9_000]]  # shorter stem -> compare overlap
    a = sep.reconstruction_residual_db(mix, stems, chunk=1 << 20)
    b = sep.reconstruction_residual_db(mix, stems, chunk=333)
    assert a == pytest.approx(b, abs=1e-9)
    assert a == pytest.approx(20 * np.log10(0.1), abs=1e-4)


# ---------------------------------------------------------------------------
# c) Provenance
# ---------------------------------------------------------------------------


def test_provenance_string_shape():
    assert (
        sep.stem_provenance("htdemucs_ft", 2, "PCM_24", "4.0.1")
        == "htdemucs_ft|s2|pcm24|demucs4.0.1"
    )
    assert sep.stem_provenance("htdemucs_ft", 0, "PCM_16", "4.0.1") == (
        "htdemucs_ft|s0|pcm16|demucs4.0.1"
    )


def test_provenance_capped_at_64_keeps_the_knobs():
    got = sep.stem_provenance("m" * 100, 10, "PCM_24", "4.0.1")
    assert len(got) == 64
    assert got.endswith("|s10|pcm24|demucs4.0.1")


def test_provenance_defaults_to_installed_demucs_version():
    assert sep.demucs_version() == dist_version("demucs")
    assert sep.stem_provenance("htdemucs_ft", 2, "PCM_24").endswith(
        f"|demucs{dist_version('demucs')}"
    )


def test_stage_records_provenance_and_residual_on_every_row(tmp_path, session, make_track):
    settings = _settings(tmp_path, demucs_shifts=2, demucs_bit_depth="PCM_24")
    track = _track(tmp_path, make_track)
    mix = _mix()
    stage = _stage(_FakeModel(), "cpu")

    ok = _run(stage, session, track, settings, mix, lambda model, wav, **kw: _sources_for(mix))
    assert ok is True

    rows = session.query(StemFile).filter_by(track_id=track.id).all()
    assert sorted(r.kind for r in rows) == sorted(KINDS)
    want = f"htdemucs_ft|s2|pcm24|demucs{dist_version('demucs')}"
    for r in rows:
        assert r.model_used == want
        # residual is measured on the PCM_24 file contents' source floats
        assert r.separation_quality == pytest.approx(_EXPECTED_RESIDUAL_DB, abs=1e-3)
        assert Path(r.path).exists()
    assert _no_partials(settings.stems_dir / track.file_hash[:8])
    assert track.state == TrackState.SEPARATED.value


# ---------------------------------------------------------------------------
# d) MPS -> CPU fallback
# ---------------------------------------------------------------------------


def test_mps_runtime_error_retries_once_on_cpu(tmp_path, session, make_track):
    settings = _settings(tmp_path)
    track = _track(tmp_path, make_track)
    mix = _mix()
    shared = _FakeModel("shared-mps")
    stage = _stage(shared, "mps")
    cpu_copy = _FakeModel("cpu-copy")
    seen: list[tuple[str, str, str]] = []

    def apply_fn(model, wav, **kw):
        seen.append((model.name, kw["device"], wav.device))
        if kw["device"] == "mps":
            raise RuntimeError("MPS backend out of memory")
        return _sources_for(mix)

    loads: list[str] = []

    def get_model_fn(name):
        loads.append(name)
        return cpu_copy

    # Spy on the logger directly: other tests reconfigure logging (CLI
    # dictConfig), which can leave caplog blind to this module's records.
    with patch.object(sep.logger, "warning") as warn:
        ok = _run(stage, session, track, settings, mix, apply_fn, get_model_fn)

    assert ok is True
    assert seen == [("shared-mps", "mps", "mps"), ("cpu-copy", "cpu", "cpu")]
    assert stage.last_device == "cpu"
    assert loads == [settings.demucs_model]
    assert cpu_copy.moved_to == ["cpu"]
    assert shared.moved_to == []  # the shared MPS model is never re-homed
    assert any("retrying once on CPU" in c.args[0] for c in warn.call_args_list)
    assert session.query(StemFile).filter_by(track_id=track.id).count() == 4


def test_cpu_failure_after_mps_failure_propagates(tmp_path, session, make_track):
    settings = _settings(tmp_path)
    track = _track(tmp_path, make_track)
    mix = _mix()
    stage = _stage(_FakeModel(), "mps")
    calls: list[str] = []

    def apply_fn(model, wav, **kw):
        calls.append(kw["device"])
        raise RuntimeError(f"fail on {kw['device']}")

    with pytest.raises(RuntimeError, match="fail on cpu"):
        _run(stage, session, track, settings, mix, apply_fn, lambda n: _FakeModel())
    assert calls == ["mps", "cpu"]  # exactly one retry


def test_non_mps_runtime_error_is_not_retried(tmp_path, session, make_track):
    settings = _settings(tmp_path)
    track = _track(tmp_path, make_track)
    mix = _mix()
    stage = _stage(_FakeModel(), "cpu")
    calls: list[str] = []

    def apply_fn(model, wav, **kw):
        calls.append(kw["device"])
        raise RuntimeError("cpu says no")

    def no_load(name):
        raise AssertionError("must not load a fallback model")

    with pytest.raises(RuntimeError, match="cpu says no"):
        _run(stage, session, track, settings, mix, apply_fn, no_load)
    assert calls == ["cpu"]


def test_mps_load_failure_falls_back_to_cpu(tmp_path):
    settings = _settings(tmp_path, demucs_device="mps")

    class _NoMps(_FakeModel):
        def to(self, device):
            if device == "mps":
                raise NotImplementedError("op not implemented for mps")
            return super().to(device)

    m = _NoMps()
    stage = sep.StemSeparationStage()
    with patch.object(sep, "_DEMUCS_OK", True), patch.object(sep, "get_model", lambda n: m):
        stage._ensure_model(settings)
    assert stage._device == "cpu"
    assert m.moved_to == ["cpu"]


# ---------------------------------------------------------------------------
# e) Waveform sidecar invalidation
# ---------------------------------------------------------------------------


def _write_wav(path: Path, data: np.ndarray) -> None:
    sf.write(str(path), data, SR, subtype="PCM_24")


def test_waveform_sidecar_recomputed_after_same_size_rewrite(tmp_path):
    audio = tmp_path / "drums.wav"
    n = SR // 2
    flat = np.full(n, 0.5, dtype="float32")
    ramp = np.linspace(0.0, 0.5, n, dtype="float32")
    _write_wav(audio, flat)
    size_before = audio.stat().st_size

    peaks1, _ = wf.get_or_compute_peaks(audio, num_peaks=50)
    assert peaks1 == pytest.approx([1.0] * 50, abs=1e-3)
    sidecar = audio.with_name(audio.name + ".waveform.json")
    payload = json.loads(sidecar.read_text())
    assert payload["source"]["size"] == size_before

    # Same path, same length, same subtype -> same byte size. Push mtime
    # forward explicitly so the test doesn't depend on fs timestamp grain.
    st = audio.stat()
    _write_wav(audio, ramp)
    assert audio.stat().st_size == size_before
    os.utime(audio, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    peaks2, _ = wf.get_or_compute_peaks(audio, num_peaks=50)
    assert peaks2 != pytest.approx(peaks1, abs=1e-3)
    assert peaks2[0] < 0.05 and peaks2[-1] == pytest.approx(1.0, abs=1e-3)
    assert json.loads(sidecar.read_text())["source"]["mtime_ns"] == audio.stat().st_mtime_ns


def test_waveform_unchanged_file_hits_cache(tmp_path):
    audio = tmp_path / "bass.wav"
    _write_wav(audio, np.full(SR // 4, 0.3, dtype="float32"))
    first = wf.get_or_compute_peaks(audio, num_peaks=20)
    with patch.object(wf, "compute_peaks", side_effect=AssertionError("cache miss")):
        assert wf.get_or_compute_peaks(audio, num_peaks=20) == first


def test_waveform_v1_sidecar_without_fingerprint_is_ignored(tmp_path):
    audio = tmp_path / "other.wav"
    _write_wav(audio, np.full(SR // 4, 0.3, dtype="float32"))
    sidecar = audio.with_name(audio.name + ".waveform.json")
    sidecar.write_text(
        json.dumps({"peaks": [0.0] * 20, "duration_seconds": 9.9, "num_peaks": 20, "version": 1})
    )
    peaks, dur = wf.get_or_compute_peaks(audio, num_peaks=20)
    assert dur == pytest.approx(0.25, abs=1e-3)
    assert peaks == pytest.approx([1.0] * 20, abs=1e-3)


def test_reseparated_stem_invalidates_its_waveform(tmp_path):
    """End-to-end within the repo: the atomic stem writer + the waveform
    cache. A re-separation rewrites ``drums.wav`` at the same path and size;
    the stem waveform endpoint's cache must not serve the old envelope."""
    out = tmp_path / "abcd1234"
    out.mkdir()
    n = SR // 2
    first = [(k, np.stack([np.full(n, 0.5)] * 2, 1).astype("float32")) for k in KINDS]
    ramp = np.linspace(0.0, 0.5, n)
    second = [(k, np.stack([ramp] * 2, 1).astype("float32")) for k in KINDS]

    sep.write_stems_atomic(out, first, SR, "PCM_24")
    drums = out / "drums.wav"
    old_stat = drums.stat()
    peaks_old, _ = wf.get_or_compute_peaks(drums, num_peaks=40)

    sep.write_stems_atomic(out, second, SR, "PCM_24")
    new_stat = drums.stat()
    assert new_stat.st_size == old_stat.st_size
    if new_stat.st_mtime_ns == old_stat.st_mtime_ns:
        pytest.skip("filesystem mtime too coarse to distinguish back-to-back writes")
    peaks_new, _ = wf.get_or_compute_peaks(drums, num_peaks=40)
    assert peaks_new != pytest.approx(peaks_old, abs=1e-3)
    assert peaks_new == pytest.approx(wf.compute_peaks(drums, num_peaks=40)[0])
