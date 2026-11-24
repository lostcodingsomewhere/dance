from __future__ import annotations

import gzip
from pathlib import Path

from fastapi.testclient import TestClient
from lxml import etree

from dance.als.writer import STEM_COLOR_INDEX
from dance.osc.bridge import AbletonBridge
from tests.test_cue_pads_e2e import API, library_with_one_song  # noqa: F401

# Child order of an AudioClip as Live 12.4.3 saved it on 2026-11-23 (docs/live-compat.md).
LIVE_AUDIO_CLIP_ORDER = [
    "LomId", "LomIdView", "CurrentStart", "CurrentEnd", "Loop", "Name", "Annotation", "Color",
    "LaunchMode", "LaunchQuantisation", "TimeSignature", "Envelopes", "ScrollerTimePreserver",
    "TimeSelection", "Legato", "Ram", "GrooveSettings", "Disabled", "VelocityAmount",
    "FollowAction", "Grid", "FreezeStart", "FreezeEnd", "IsWarped", "TakeId", "IsInKey",
    "ScaleInformation", "AutomationEnvelopesListWrapper", "SampleRef", "Onsets", "WarpMode",
    "GranularityTones", "GranularityTexture", "FluctuationTexture", "TransientResolution",
    "TransientLoopMode", "TransientEnvelope", "ComplexProFormants", "ComplexProEnvelope", "Sync",
    "HiQ", "Fade", "Fades", "PitchCoarse", "PitchFine", "SampleVolume", "WarpMarkers",
    "SavedWarpMarkersForStretched", "MarkersGenerated", "IsSongTempoLeader",
]  # fmt: skip
LIVE_SAMPLE_REF_CHILDREN = [
    "FileRef", "LastModDate", "SourceContext", "SampleUsageHint", "DefaultDuration",
    "DefaultSampleRate", "SamplesToAutoWarp",
]  # fmt: skip
LIVE_FILE_REF_CHILDREN = [
    "RelativePathType", "RelativePath", "Path", "Type", "LivePackName", "LivePackId",
    "OriginalFileSize", "OriginalCrc", "SourceHint",
]  # fmt: skip
NAMES_LIVE_DOES_NOT_READ = {"Warp", "Gain", "GranularityResolution", "ColorIndex"}
RELATIVE_TO_THE_SET_FILE = "1"
EXTERNAL_FILE = "2"
WARP_GRID_BEATS = 1 / 32


def _exported_set(client: TestClient, track_id: int) -> tuple[etree._Element, Path]:
    exported = client.post(f"{API}/tracks/{track_id}/als", json={})
    assert exported.status_code == 200, exported.text
    set_path = Path(exported.json()["out_path"])
    return etree.fromstring(gzip.decompress(set_path.read_bytes())), set_path


def _tracks_with_a_clip(live_set: etree._Element) -> list[tuple[etree._Element, etree._Element]]:
    pairs = [(track, track.find(".//AudioClip")) for track in live_set.iter("AudioTrack")]
    return [(track, clip) for track, clip in pairs if clip is not None]


def _in_order_within(names: list[str], order: list[str]) -> bool:
    remaining = iter(order)
    return all(name in remaining for name in names)


def test_exported_clips_use_the_names_and_order_live_writes(library_with_one_song):  # noqa: F811
    client, track_id = library_with_one_song
    live_set, _ = _exported_set(client, track_id)
    clips = _tracks_with_a_clip(live_set)
    assert len(clips) == 5

    for track, clip in clips:
        assert _in_order_within([child.tag for child in clip], LIVE_AUDIO_CLIP_ORDER)
        assert [child.tag for child in clip.find("SampleRef")] == LIVE_SAMPLE_REF_CHILDREN
        assert [child.tag for child in clip.find("SampleRef/FileRef")] == LIVE_FILE_REF_CHILDREN
        assert not {element.tag for element in clip.iter()} & NAMES_LIVE_DOES_NOT_READ
        assert track.find("ColorIndex") is None


def test_exported_clips_follow_the_song_tempo_and_carry_their_stem_colour(
    library_with_one_song,  # noqa: F811
):
    client, track_id = library_with_one_song
    live_set, _ = _exported_set(client, track_id)
    song_tempo = float(live_set.find(".//Tempo/Manual").get("Value"))

    for track, clip in _tracks_with_a_clip(live_set):
        stem_kind = track.find("Name/EffectiveName").get("Value").split()[0].lower()
        assert track.find("Color").get("Value") == str(STEM_COLOR_INDEX[stem_kind])
        assert clip.find("Color").get("Value") == track.find("Color").get("Value")

        assert clip.find("IsWarped").get("Value") == "true"
        assert int(clip.find("WarpMode").get("Value")) == AbletonBridge._WARP_MODE_BEATS
        assert (
            int(clip.find("LaunchQuantisation").get("Value")) == AbletonBridge._LAUNCH_QUANT_GLOBAL
        )
        assert clip.find("MarkersGenerated").get("Value") == "true"
        assert clip.find("SampleRef/SamplesToAutoWarp").get("Value") == "0"

        first, second = clip.findall("WarpMarkers/WarpMarker")
        assert (float(first.get("SecTime")), float(first.get("BeatTime"))) == (0.0, 0.0)
        assert float(second.get("BeatTime")) == WARP_GRID_BEATS
        marker_tempo = float(second.get("BeatTime")) / float(second.get("SecTime")) * 60
        assert abs(marker_tempo - song_tempo) < 1e-6


def test_exported_clips_point_at_their_stem_files_the_way_live_does(
    library_with_one_song,  # noqa: F811
):
    client, track_id = library_with_one_song
    live_set, set_path = _exported_set(client, track_id)

    for _, clip in _tracks_with_a_clip(live_set):
        file_ref = clip.find("SampleRef/FileRef")
        audio_file = Path(file_ref.find("Path").get("Value"))
        assert audio_file.is_file()
        assert file_ref.find("Type").get("Value") == EXTERNAL_FILE
        assert file_ref.find("RelativePathType").get("Value") == RELATIVE_TO_THE_SET_FILE
        from_the_set_folder = set_path.parent / file_ref.find("RelativePath").get("Value")
        assert from_the_set_folder.resolve() == audio_file.resolve()
