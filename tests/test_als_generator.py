"""Tests for the Ableton Live Set generator.

The ``.als`` format is gzipped, undocumented XML — we can validate
*structure* exhaustively, but not "Live actually opens it" without
running Live. So this file is rigorous about:

* The output is a valid gzipped XML doc.
* Top-level shape: Ableton root, LiveSet, 5 AudioTrack, 1 MasterTrack.
* Stem → AudioTrack mapping (drums clip lands on the drums track, etc.).
* Locators match the Regions' positions and BPM-derived beats.
* Track colors match the palette spec.
* The clip's FileRef path matches the stem's stored path.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from lxml import etree

from dance.als import AlsGenerator
from dance.als.generator import AlsExportError, AlsOutsideDirError
from dance.als.markers import LocatorEntry, regions_to_locators
from dance.als.writer import (
    ABLETON_ROOT_ATTRS,
    STEM_COLOR_INDEX,
    LiveSetSpec,
    StemEntry,
    build_live_set_xml,
)
from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    Region,
    RegionSource,
    RegionType,
    SectionLabel,
    StemFile,
    StemKind,
    TrackState,
    now_utc,
)
from tests.audio_fixtures import TrackSpec, write_track


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def als_settings(tmp_path: Path) -> Settings:
    """Settings rooted under tmp_path so tests never touch the user's home."""
    s = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
    )
    s.ensure_directories()
    return s


@pytest.fixture
def complete_track_with_stems(session, make_track, tmp_path: Path, als_settings):
    """Build a fully-analyzed COMPLETE track with 4 stems on disk + regions.

    Returns ``(track, regions, stem_paths)``.
    """
    spec = TrackSpec(bpm=128.0, bars=4)  # short — keeps test fast

    # Write real (small) audio so SampleRef sizes are non-zero.
    full_mix = tmp_path / "library" / "test.wav"
    full_mix.parent.mkdir(parents=True, exist_ok=True)
    write_track(full_mix, spec)
    stem_dir = tmp_path / "stems" / "test"
    stem_dir.mkdir(parents=True, exist_ok=True)
    stem_paths: dict[str, Path] = {}
    for kind in ("drums", "bass", "vocals", "other"):
        p = stem_dir / f"{kind}.wav"
        write_track(p, spec)
        stem_paths[kind] = p

    track = make_track(
        title="Test Track",
        artist="Tester",
        file_path=str(full_mix),
        duration_seconds=spec.duration_seconds,
        state=TrackState.COMPLETE.value,
    )

    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=None,
            bpm=spec.bpm,
            key_camelot="8A",
            floor_energy=7,
            energy_overall=0.6,
            analyzed_at=now_utc(),
        )
    )
    for kind in ("drums", "bass", "vocals", "other"):
        session.add(
            StemFile(track_id=track.id, kind=kind, path=str(stem_paths[kind]))
        )
    regions = [
        Region(
            track_id=track.id,
            position_ms=0,
            region_type=RegionType.SECTION.value,
            section_label=SectionLabel.INTRO.value,
            source=RegionSource.AUTO.value,
            name="Intro",
        ),
        Region(
            track_id=track.id,
            position_ms=5_000,  # 5s @ 128 BPM = 10.667 beats
            region_type=RegionType.CUE.value,
            source=RegionSource.MANUAL.value,
            name="Drop",
        ),
        Region(
            track_id=track.id,
            position_ms=10_000,
            length_ms=1_000,
            region_type=RegionType.LOOP.value,
            length_bars=4,
            source=RegionSource.AUTO.value,
        ),
    ]
    session.add_all(regions)
    session.commit()

    return track, regions, stem_paths


# ---------------------------------------------------------------------------
# Markers — pure-data, no I/O
# ---------------------------------------------------------------------------


def test_position_to_beats_basic():
    from dance.als.markers import position_ms_to_beats

    # 1 second at 60 BPM = 1 beat
    assert position_ms_to_beats(1000, 60.0) == pytest.approx(1.0)
    # 1 second at 120 BPM = 2 beats
    assert position_ms_to_beats(1000, 120.0) == pytest.approx(2.0)
    # 500ms at 128 BPM = 1.0667 beats
    assert position_ms_to_beats(500, 128.0) == pytest.approx(1.0667, rel=1e-3)


def test_position_to_beats_rejects_zero_bpm():
    from dance.als.markers import position_ms_to_beats

    with pytest.raises(ValueError):
        position_ms_to_beats(1000, 0.0)


def test_regions_to_locators_sorts_and_names(session, make_track):
    track = make_track()
    regions = [
        Region(
            track_id=track.id,
            position_ms=2000,
            region_type=RegionType.CUE.value,
            source=RegionSource.AUTO.value,
        ),
        Region(
            track_id=track.id,
            position_ms=0,
            region_type=RegionType.SECTION.value,
            section_label=SectionLabel.INTRO.value,
            source=RegionSource.AUTO.value,
        ),
    ]
    session.add_all(regions)
    session.commit()

    out = regions_to_locators(regions, bpm=120.0)
    assert [e.name for e in out] == ["Intro", "Cue 1"]
    assert out[0].time_beats == 0.0
    assert out[1].time_beats == pytest.approx(4.0)  # 2s at 120 BPM = 4 beats


def test_regions_to_locators_drops_per_stem_regions(session, make_track):
    """Regions with stem_file_id should not show up in master Locators."""
    track = make_track()
    stem = StemFile(track_id=track.id, kind="drums", path="/tmp/d.wav")
    session.add(stem)
    session.flush()
    regions = [
        Region(
            track_id=track.id,
            position_ms=1000,
            region_type=RegionType.CUE.value,
            source=RegionSource.AUTO.value,
            name="Track cue",
        ),
        Region(
            track_id=track.id,
            stem_file_id=stem.id,
            position_ms=2000,
            region_type=RegionType.STEM_SOLO.value,
            source=RegionSource.AUTO.value,
            name="Stem cue",
        ),
    ]
    session.add_all(regions)
    session.commit()

    out = regions_to_locators(regions, bpm=120.0)
    assert [e.name for e in out] == ["Track cue"]


# ---------------------------------------------------------------------------
# Writer (build_live_set_xml) — pure-data, no DB
# ---------------------------------------------------------------------------


def _build_minimal_spec(tmp_path: Path, bpm: float = 128.0) -> LiveSetSpec:
    """Make a LiveSetSpec with 5 fake on-disk files."""
    entries = []
    for kind in ("drums", "bass", "vocals", "other", "mix"):
        p = tmp_path / f"{kind}.wav"
        p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")  # not a real WAV, but exists
        entries.append(StemEntry(kind=kind, path=p, duration_seconds=10.0))
    return LiveSetSpec(
        name="Demo",
        bpm=bpm,
        stems=entries,
        locators=[LocatorEntry(time_beats=0.0, name="Start")],
    )


def test_writer_emits_root_ableton_with_known_attrs(tmp_path: Path):
    xml = build_live_set_xml(_build_minimal_spec(tmp_path))
    root = etree.fromstring(xml)
    assert root.tag == "Ableton"
    for k, v in ABLETON_ROOT_ATTRS.items():
        assert root.get(k) == v, f"attr {k} mismatch"


def test_writer_emits_five_audio_tracks(tmp_path: Path):
    xml = build_live_set_xml(_build_minimal_spec(tmp_path))
    root = etree.fromstring(xml)
    audio_tracks = root.findall("./LiveSet/Tracks/AudioTrack")
    assert len(audio_tracks) == 5


def test_writer_emits_one_master_track_with_correct_tempo(tmp_path: Path):
    xml = build_live_set_xml(_build_minimal_spec(tmp_path, bpm=132.0))
    root = etree.fromstring(xml)
    masters = root.findall("./LiveSet/MasterTrack")
    assert len(masters) == 1
    tempo_manual = root.find("./LiveSet/MasterTrack/DeviceChain/Mixer/Tempo/Manual")
    assert tempo_manual is not None
    assert float(tempo_manual.get("Value")) == pytest.approx(132.0)


def test_writer_track_color_palette(tmp_path: Path):
    xml = build_live_set_xml(_build_minimal_spec(tmp_path))
    root = etree.fromstring(xml)
    audio_tracks = root.findall("./LiveSet/Tracks/AudioTrack")
    # The writer orders tracks as drums/bass/vocals/other/mix (STEM_ORDER).
    names = [t.find("./Name/EffectiveName").get("Value") for t in audio_tracks]
    assert names == ["Drums", "Bass", "Vocals", "Other", "Mix"]

    for t, expected_kind in zip(audio_tracks, ("drums", "bass", "vocals", "other", "mix")):
        color_idx = int(t.find("./ColorIndex").get("Value"))
        assert color_idx == STEM_COLOR_INDEX[expected_kind], (
            f"color for {expected_kind!r} should be {STEM_COLOR_INDEX[expected_kind]}, "
            f"got {color_idx}"
        )


def test_writer_emits_one_locator_per_entry(tmp_path: Path):
    spec = _build_minimal_spec(tmp_path)
    spec = LiveSetSpec(
        name=spec.name,
        bpm=spec.bpm,
        stems=spec.stems,
        locators=[
            LocatorEntry(time_beats=0.0, name="Intro"),
            LocatorEntry(time_beats=10.667, name="Drop"),
            LocatorEntry(time_beats=21.333, name="Outro"),
        ],
    )
    xml = build_live_set_xml(spec)
    root = etree.fromstring(xml)
    locs = root.findall("./LiveSet/Locators/Locators/Locator")
    assert len(locs) == 3
    assert [loc.find("Name").get("Value") for loc in locs] == ["Intro", "Drop", "Outro"]
    assert float(locs[1].find("Time").get("Value")) == pytest.approx(10.667, rel=1e-3)


def test_writer_clip_references_actual_file_path(tmp_path: Path):
    spec = _build_minimal_spec(tmp_path)
    xml = build_live_set_xml(spec)
    root = etree.fromstring(xml)

    # First AudioTrack is drums (per STEM_ORDER). Its SampleRef's FileRef
    # Path should equal the resolved drums.wav we wrote above.
    drums = root.find("./LiveSet/Tracks/AudioTrack")
    path_node = drums.find(
        "./DeviceChain/MainSequencer/Sample/ArrangerAutomation/Events/AudioClip/SampleRef/FileRef/Path"
    )
    assert path_node is not None
    assert path_node.get("Value") == str((tmp_path / "drums.wav").resolve())


def test_writer_rejects_zero_bpm(tmp_path: Path):
    with pytest.raises(ValueError):
        build_live_set_xml(
            LiveSetSpec(name="x", bpm=0.0, stems=_build_minimal_spec(tmp_path).stems, locators=[])
        )


def test_writer_rejects_empty_stems(tmp_path: Path):
    with pytest.raises(ValueError):
        build_live_set_xml(LiveSetSpec(name="x", bpm=120.0, stems=[], locators=[]))


def test_writer_mix_clip_is_muted_by_default(tmp_path: Path):
    """The mix track's clip should be muted so playback uses the stems."""
    xml = build_live_set_xml(_build_minimal_spec(tmp_path))
    root = etree.fromstring(xml)
    audio_tracks = root.findall("./LiveSet/Tracks/AudioTrack")
    # Mix is last per STEM_ORDER.
    mix = audio_tracks[-1]
    disabled = mix.find(
        "./DeviceChain/MainSequencer/Sample/ArrangerAutomation/Events/AudioClip/Disabled"
    )
    assert disabled is not None
    assert disabled.get("Value") == "true"

    # The other stems should NOT be disabled.
    for t in audio_tracks[:-1]:
        d = t.find(
            "./DeviceChain/MainSequencer/Sample/ArrangerAutomation/Events/AudioClip/Disabled"
        )
        assert d.get("Value") == "false"


# ---------------------------------------------------------------------------
# Generator — end-to-end with a real DB row
# ---------------------------------------------------------------------------


def test_generator_smoke_writes_a_file(als_settings, session, complete_track_with_stems):
    track, _regions, _paths = complete_track_with_stems
    gen = AlsGenerator(session, als_settings)
    out = gen.write(track)
    assert out.exists()
    assert out.suffix == ".als"
    assert out.stat().st_size > 100  # non-trivial


def test_generator_output_is_valid_gzipped_xml(
    als_settings, session, complete_track_with_stems
):
    track, _, _ = complete_track_with_stems
    out = AlsGenerator(session, als_settings).write(track)
    raw = out.read_bytes()
    # GZip magic bytes 1f 8b
    assert raw[:2] == b"\x1f\x8b"
    xml = gzip.decompress(raw)
    root = etree.fromstring(xml)
    assert root.tag == "Ableton"


def test_generator_structure_matches_spec(
    als_settings, session, complete_track_with_stems
):
    track, regions, _ = complete_track_with_stems
    out = AlsGenerator(session, als_settings).write(track)
    root = etree.fromstring(gzip.decompress(out.read_bytes()))

    # 5 AudioTrack
    audio_tracks = root.findall("./LiveSet/Tracks/AudioTrack")
    assert len(audio_tracks) == 5

    # 1 MasterTrack with tempo == 128.0
    tempo = root.find("./LiveSet/MasterTrack/DeviceChain/Mixer/Tempo/Manual")
    assert tempo is not None
    assert float(tempo.get("Value")) == pytest.approx(128.0)

    # 3 Locators (one per Region)
    locs = root.findall("./LiveSet/Locators/Locators/Locator")
    assert len(locs) == len(regions)


def test_generator_stem_to_track_mapping(
    als_settings, session, complete_track_with_stems
):
    """The drums clip must reference the drums file, etc."""
    track, _, stem_paths = complete_track_with_stems
    out = AlsGenerator(session, als_settings).write(track)
    root = etree.fromstring(gzip.decompress(out.read_bytes()))

    audio_tracks = root.findall("./LiveSet/Tracks/AudioTrack")
    by_name = {
        t.find("./Name/EffectiveName").get("Value"): t for t in audio_tracks
    }
    for kind in ("drums", "bass", "vocals", "other"):
        track_el = by_name[kind.capitalize()]
        path_el = track_el.find(
            "./DeviceChain/MainSequencer/Sample/ArrangerAutomation/Events/AudioClip/SampleRef/FileRef/Path"
        )
        assert path_el is not None
        assert path_el.get("Value") == str(stem_paths[kind].resolve()), (
            f"track {kind} should reference {stem_paths[kind]}"
        )

    # And mix references the original file.
    mix_path = by_name["Mix"].find(
        "./DeviceChain/MainSequencer/Sample/ArrangerAutomation/Events/AudioClip/SampleRef/FileRef/Path"
    )
    assert mix_path.get("Value") == str(Path(track.file_path).resolve())


def test_generator_locator_positions_in_beats(
    als_settings, session, complete_track_with_stems
):
    """Region at 5s @ 128 BPM should yield a Locator at 10.667 beats."""
    track, _regions, _ = complete_track_with_stems
    out = AlsGenerator(session, als_settings).write(track)
    root = etree.fromstring(gzip.decompress(out.read_bytes()))

    locs = root.findall("./LiveSet/Locators/Locators/Locator")
    names = [loc.find("Name").get("Value") for loc in locs]
    times = {
        loc.find("Name").get("Value"): float(loc.find("Time").get("Value"))
        for loc in locs
    }
    assert "Intro" in names
    assert times["Intro"] == pytest.approx(0.0)
    assert times["Drop"] == pytest.approx(5.0 * 128.0 / 60.0, rel=1e-4)  # ≈ 10.667
    # The auto-named loop locator should be at 10s → 21.333 beats
    loop_name = [n for n in names if n.startswith("Loop")][0]
    assert times[loop_name] == pytest.approx(10.0 * 128.0 / 60.0, rel=1e-4)


def test_generator_track_colors(als_settings, session, complete_track_with_stems):
    track, _, _ = complete_track_with_stems
    out = AlsGenerator(session, als_settings).write(track)
    root = etree.fromstring(gzip.decompress(out.read_bytes()))

    by_name = {
        t.find("./Name/EffectiveName").get("Value"): int(
            t.find("./ColorIndex").get("Value")
        )
        for t in root.findall("./LiveSet/Tracks/AudioTrack")
    }
    assert by_name["Drums"] == STEM_COLOR_INDEX["drums"]
    assert by_name["Bass"] == STEM_COLOR_INDEX["bass"]
    assert by_name["Vocals"] == STEM_COLOR_INDEX["vocals"]
    assert by_name["Other"] == STEM_COLOR_INDEX["other"]
    assert by_name["Mix"] == STEM_COLOR_INDEX["mix"]


def test_generator_default_path_inside_output_dir(
    als_settings, session, complete_track_with_stems
):
    track, _, _ = complete_track_with_stems
    out = AlsGenerator(session, als_settings).write(track)
    # Resolves under als_output_dir
    assert (
        als_settings.als_output_dir.resolve() in out.resolve().parents
    )
    assert out.name.endswith(".als")


def test_generator_rejects_out_path_outside_dir(
    als_settings, session, complete_track_with_stems, tmp_path: Path
):
    track, _, _ = complete_track_with_stems
    gen = AlsGenerator(session, als_settings)
    bad = tmp_path / "escape.als"  # outside als_output_dir
    with pytest.raises(AlsOutsideDirError):
        gen.write(track, bad)


def test_generator_404s_when_track_not_complete(
    als_settings, session, make_track
):
    track = make_track(state=TrackState.PENDING.value)
    session.commit()
    gen = AlsGenerator(session, als_settings)
    with pytest.raises(AlsExportError):
        gen.write(track)


def test_generator_404s_when_no_analysis(als_settings, session, make_track):
    track = make_track(state=TrackState.COMPLETE.value)
    # No AudioAnalysis row
    session.commit()
    gen = AlsGenerator(session, als_settings)
    with pytest.raises(AlsExportError, match="no full-mix analysis"):
        gen.write(track)


def test_generator_404s_when_no_stems(als_settings, session, make_track):
    track = make_track(state=TrackState.COMPLETE.value)
    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=None,
            bpm=128.0,
            analyzed_at=now_utc(),
        )
    )
    session.commit()
    gen = AlsGenerator(session, als_settings)
    with pytest.raises(AlsExportError, match="no stems"):
        gen.write(track)


def test_generator_handles_missing_stem_kinds_gracefully(
    als_settings, session, make_track, tmp_path
):
    """Only 2 of 4 stems should still yield a Set with 3 AudioTracks (2 stems + mix)."""
    full_mix = tmp_path / "library" / "x.wav"
    full_mix.parent.mkdir(parents=True, exist_ok=True)
    full_mix.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    track = make_track(
        state=TrackState.COMPLETE.value,
        file_path=str(full_mix),
        duration_seconds=4.0,
    )
    session.add(
        AudioAnalysis(track_id=track.id, stem_file_id=None, bpm=120.0, analyzed_at=now_utc())
    )
    for kind in ("drums", "vocals"):
        p = tmp_path / f"{kind}.wav"
        p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        session.add(StemFile(track_id=track.id, kind=kind, path=str(p)))
    session.commit()

    out = AlsGenerator(session, als_settings).write(track)
    root = etree.fromstring(gzip.decompress(out.read_bytes()))
    audio_tracks = root.findall("./LiveSet/Tracks/AudioTrack")
    names = sorted(t.find("./Name/EffectiveName").get("Value") for t in audio_tracks)
    assert names == ["Drums", "Mix", "Vocals"]
