from __future__ import annotations

import gzip
from pathlib import Path

import pytest
import soundfile as sf
from click.testing import CliRunner
from fastapi.testclient import TestClient
from lxml import etree

from dance import cli
from dance.api import create_app
from dance.config import get_settings
from dance.core.database import (
    Track,
    TrackState,
    _reset_engine_for_tests,
    get_session,
    get_session_factory,
    init_db,
)
from dance.pipeline import dispatcher as dispatcher_mod
from dance.pipeline import preflight
from dance.pipeline.dispatcher import Dispatcher
from tests.test_api import FakeAbletonBridge
from tests.test_cue_pads_e2e import API, _heavy_models_faked, _song_with_six_sections

SONGS = {"first_song": 128.0, "second_song": 126.0, "third_song": 130.0}
PADS_PER_PAGE = 5
BEATS_PER_BAR = 4
WARP_BEATS, WARP_COMPLEX, WARP_COMPLEX_PRO = 0, 4, 6


@pytest.fixture
def three_song_library(monkeypatch):
    monkeypatch.setattr(dispatcher_mod, "_IDLE_GRACE_S", 0.3)
    monkeypatch.setattr(dispatcher_mod, "_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    settings = get_settings()
    settings.ensure_directories()
    init_db(settings.db_url)
    for seed, (name, bpm) in enumerate(SONGS.items(), start=1):
        audio, spec = _song_with_six_sections(bpm=bpm, seed=seed)
        sf.write(
            str(Path(settings.library_dir) / f"{name}.wav"),
            audio,
            spec.sample_rate,
            subtype="PCM_16",
        )

    session = get_session(settings.db_url)
    separate, ensure_model, encode = _heavy_models_faked()
    with separate, ensure_model, encode:
        dispatcher = Dispatcher(settings, session)
        assert dispatcher.ingest()["new"] == len(SONGS)
        dispatcher.run()
    track_ids = {}
    for track in session.query(Track):
        assert track.state == TrackState.COMPLETE.value, track.error_message
        track_ids[Path(track.file_path).stem] = track.id
    session.close()

    app = create_app(
        settings=settings,
        bridge=FakeAbletonBridge(),
        session_factory=get_session_factory(settings.db_url),
    )
    with TestClient(app) as client:
        yield client, track_ids
    _reset_engine_for_tests()


def _planned_set(client: TestClient, queues: dict[str, list[int]]) -> int:
    created = client.post(f"{API}/sets", json={"name": "Friday night"})
    assert created.status_code == 200, created.text
    set_id = created.json()["id"]
    planned = client.put(f"{API}/sets/{set_id}/plan", json={"queues": queues})
    assert planned.status_code == 200, planned.text
    return set_id


def _live_set(path: str | Path) -> etree._Element:
    return etree.fromstring(gzip.decompress(Path(path).read_bytes()))


def _clips_by_row(live_set: etree._Element, track_name: str) -> dict[int, etree._Element]:
    for track in live_set.iter("AudioTrack"):
        if track.find("Name/EffectiveName").get("Value") == track_name:
            slots = track.findall("DeviceChain/MainSequencer/ClipSlotList/ClipSlot")
            clips = {row: slot.find("ClipSlot/Value/AudioClip") for row, slot in enumerate(slots)}
            return {row: clip for row, clip in clips.items() if clip is not None}
    raise AssertionError(f"no track named {track_name}")


def _number(clip: etree._Element, path: str) -> float:
    return float(clip.find(path).get("Value"))


def _text(clip: etree._Element, path: str) -> str:
    return clip.find(path).get("Value")


def _clip_tempo(clip: etree._Element) -> float:
    first, second = clip.findall("WarpMarkers/WarpMarker")
    beats = float(second.get("BeatTime")) - float(first.get("BeatTime"))
    seconds = float(second.get("SecTime")) - float(first.get("SecTime"))
    return beats / seconds * 60


def _pads(client: TestClient, track_id: int, role: str) -> list[dict]:
    pads = client.get(f"{API}/tracks/{track_id}/pads").json()
    return sorted((p for p in pads if p["role"] == role), key=lambda p: p["slot"])


def _stem_path(client: TestClient, track_id: int, kind: str) -> str:
    stems = client.get(f"{API}/tracks/{track_id}/stems").json()
    return next(stem["path"] for stem in stems if stem["kind"] == kind)


def test_a_planned_set_exports_as_one_live_set_laid_out_in_pages(three_song_library):
    client, ids = three_song_library
    first, second, third = ids["first_song"], ids["second_song"], ids["third_song"]
    set_id = _planned_set(client, {"song": [first, second, third], "drums": [third]})

    exported = client.post(f"{API}/sets/{set_id}/als", json={})
    assert exported.status_code == 200, exported.text
    result = exported.json()
    assert result["skipped"] == []
    assert (result["page_count"], result["scene_count"]) == (3, 3 * PADS_PER_PAGE)
    assert result["clip_count"] == 3 * 5 * PADS_PER_PAGE + PADS_PER_PAGE

    live_set = _live_set(result["out_path"])
    first_bpm = client.get(f"{API}/tracks/{first}").json()["analysis"]["bpm"]
    assert _number(live_set, ".//Tempo/Manual") == pytest.approx(first_bpm, abs=1e-6)

    scenes = live_set.findall(".//Scenes/Scene")
    assert len(scenes) == 3 * PADS_PER_PAGE
    titles = {name: client.get(f"{API}/tracks/{tid}").json()["title"] for name, tid in ids.items()}
    page_names = [scenes[page * PADS_PER_PAGE].find("Name").get("Value") for page in range(3)]
    assert titles["first_song"] in page_names[0] and titles["second_song"] in page_names[0]
    assert titles["third_song"] in page_names[1]
    assert titles["first_song"] not in page_names[1]

    for track in live_set.iter("AudioTrack"):
        for slot_list in track.iter("ClipSlotList"):
            assert len(slot_list.findall("ClipSlot")) == 3 * PADS_PER_PAGE

    assert sorted(_clips_by_row(live_set, "Drums A")) == list(range(15))
    assert sorted(_clips_by_row(live_set, "Drums B")) == [0, 1, 2, 3, 4]
    assert sorted(_clips_by_row(live_set, "Vocals A")) == list(range(10))
    assert sorted(_clips_by_row(live_set, "Mix B")) == [0, 1, 2, 3, 4]

    page_of = {first: ("A", 0), second: ("B", 0), third: ("A", 1)}
    for track_id, (side, page) in page_of.items():
        song_bpm = client.get(f"{API}/tracks/{track_id}").json()["analysis"]["bpm"]
        for kind, label in (("drums", "Drums"), ("vocals", "Vocals"), ("other", "Other")):
            clips = _clips_by_row(live_set, f"{label} {side}")
            for pad in _pads(client, track_id, kind):
                clip = clips[page * PADS_PER_PAGE + pad["slot"] - 1]
                assert _text(clip, "SampleRef/FileRef/Path") == str(
                    Path(_stem_path(client, track_id, kind)).resolve()
                )
                assert _clip_tempo(clip) == pytest.approx(song_bpm, abs=1e-6)
                start_beats = pad["position_ms"] / 60_000 * song_bpm
                assert _number(clip, "Loop/LoopStart") == pytest.approx(start_beats, abs=1e-3)

    spare_drums = _clips_by_row(live_set, "Drums A")
    for pad in _pads(client, third, "drums"):
        clip = spare_drums[2 * PADS_PER_PAGE + pad["slot"] - 1]
        assert _text(clip, "SampleRef/FileRef/Path") == str(
            Path(_stem_path(client, third, "drums")).resolve()
        )


def test_rhythm_pads_loop_their_section_and_vocals_and_the_song_play_on(three_song_library):
    client, ids = three_song_library
    first = ids["first_song"]
    set_id = _planned_set(client, {"song": [first]})
    live_set = _live_set(client.post(f"{API}/sets/{set_id}/als", json={}).json()["out_path"])

    for kind, label, warp_mode in (
        ("drums", "Drums A", WARP_BEATS),
        ("bass", "Bass A", WARP_BEATS),
        ("other", "Other A", WARP_COMPLEX),
    ):
        clips = _clips_by_row(live_set, label)
        for pad in _pads(client, first, kind):
            clip = clips[pad["slot"] - 1]
            assert _text(clip, "Loop/LoopOn") == "true"
            loop_beats = _number(clip, "Loop/LoopEnd") - _number(clip, "Loop/LoopStart")
            assert loop_beats == pytest.approx(pad["length_bars"] * BEATS_PER_BAR, abs=1e-6)
            assert int(_number(clip, "WarpMode")) == warp_mode
            assert _text(clip, "IsWarped") == "true"

    song_beats = max(
        _number(clip, "Loop/LoopEnd") for clip in _clips_by_row(live_set, "Vocals A").values()
    )
    for label, warp_mode in (("Vocals A", WARP_COMPLEX_PRO), ("Mix A", WARP_COMPLEX)):
        for clip in _clips_by_row(live_set, label).values():
            assert _text(clip, "Loop/LoopOn") == "false"
            assert _number(clip, "Loop/LoopEnd") == pytest.approx(song_beats, abs=1e-6)
            assert _number(clip, "Loop/LoopEnd") > _number(clip, "Loop/LoopStart")
            assert int(_number(clip, "WarpMode")) == warp_mode
    assert {_text(clip, "Disabled") for clip in _clips_by_row(live_set, "Mix A").values()} == {
        "true"
    }
    assert {_text(clip, "Disabled") for clip in _clips_by_row(live_set, "Drums A").values()} == {
        "false"
    }


def test_pad_colour_shows_the_section_and_dims_where_the_stem_is_silent(three_song_library):
    client, ids = three_song_library
    first = ids["first_song"]
    set_id = _planned_set(client, {"song": [first]})
    live_set = _live_set(client.post(f"{API}/sets/{set_id}/als", json={}).json()["out_path"])

    drum_pads = _pads(client, first, "drums")
    drum_clips = _clips_by_row(live_set, "Drums A")
    colour_of_section: dict[str, set[str]] = {}
    for pad in drum_pads:
        colour = _text(drum_clips[pad["slot"] - 1], "Color")
        colour_of_section.setdefault(pad["section_label"], set()).add(colour)
    assert all(len(colours) == 1 for colours in colour_of_section.values())
    distinct = {next(iter(colours)) for colours in colour_of_section.values()}
    assert len(distinct) == len(colour_of_section) >= 4

    vocal_pads = _pads(client, first, "vocals")
    vocal_clips = _clips_by_row(live_set, "Vocals A")
    assert [pad["silent"] for pad in vocal_pads] == [True, False, False, False, False]
    for pad in vocal_pads:
        same_section_on_drums = next(iter(colour_of_section[pad["section_label"]]))
        colour = _text(vocal_clips[pad["slot"] - 1], "Color")
        assert (colour != same_section_on_drums) == pad["silent"]


def test_the_command_writes_the_same_set_and_bad_requests_are_refused(three_song_library):
    client, ids = three_song_library
    set_id = _planned_set(client, {"song": [ids["first_song"], ids["second_song"]]})
    from_api = client.post(f"{API}/sets/{set_id}/als", json={}).json()

    target = Path(get_settings().als_output_dir) / "from-the-command.als"
    ran = CliRunner().invoke(cli.main, ["export-set", str(set_id), "--out", str(target)])
    assert ran.exit_code == 0, ran.output
    assert len(_live_set(target).findall(".//AudioClip")) == from_api["clip_count"]

    outside = client.post(
        f"{API}/sets/{set_id}/als", json={"out_path": str(target.parent.parent / "x.als")}
    )
    assert outside.status_code == 403

    assert client.post(f"{API}/sets/999999/als", json={}).status_code == 404
    empty_set = _planned_set(client, {})
    assert client.post(f"{API}/sets/{empty_set}/als", json={}).status_code == 400


def test_a_new_file_each_time_leaves_what_live_saved_in_the_earlier_one(three_song_library):
    client, ids = three_song_library
    set_id = _planned_set(client, {"song": [ids["first_song"]]})
    first_path = Path(
        client.post(f"{API}/sets/{set_id}/als", json={"new_file": True}).json()["out_path"]
    )
    saved_by_live = b"the open Set, saved by Live over its own file"
    first_path.write_bytes(saved_by_live)

    replanned = client.put(
        f"{API}/sets/{set_id}/plan",
        json={"queues": {"song": [ids["first_song"], ids["second_song"]]}},
    )
    assert replanned.status_code == 200, replanned.text
    second_path = Path(
        client.post(f"{API}/sets/{set_id}/als", json={"new_file": True}).json()["out_path"]
    )

    assert second_path != first_path
    assert first_path.read_bytes() == saved_by_live
    output_dir = Path(get_settings().als_output_dir).resolve()
    for path in (first_path, second_path):
        assert path.resolve().parent == output_dir
        assert path.name.startswith("Friday night ") and path.suffix == ".als"
    assert sorted(_clips_by_row(_live_set(second_path), "Drums B")) == [0, 1, 2, 3, 4]

    one_name = client.post(f"{API}/sets/{set_id}/als", json={}).json()["out_path"]
    assert Path(one_name).name == "Friday night.als"
    assert client.post(f"{API}/sets/{set_id}/als", json={}).json()["out_path"] == one_name
