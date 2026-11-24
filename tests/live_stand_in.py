from __future__ import annotations

import copy
import gzip
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi.testclient import TestClient
from lxml import etree
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server, udp_client

from dance.api import create_app
from dance.config import get_settings
from dance.core.database import _reset_engine_for_tests, get_session_factory
from dance.osc.bridge import AbletonBridge
from tests.test_export_set_e2e import three_songs_through_the_pipeline

GRID_GUARD_DELAYS = (0.2, 0.6, 1.2)

SHADOW_BEATS = 1 / 32
Marker = tuple[float, float]
Slot = tuple[int, int]
PROPERTIES_LIVE_ANSWERS = (
    "name",
    "file_path",
    "start_marker",
    "end_marker",
    "loop_start",
    "loop_end",
    "looping",
    "warping",
    "warp_mode",
    "color_index",
    "muted",
)


def free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def straight_grid(bpm: float) -> list[Marker]:
    return [(0.0, 0.0), (SHADOW_BEATS, SHADOW_BEATS * 60.0 / bpm)]


def clip_tempo(clip: dict[str, Any]) -> float:
    (first_beat, first_sec), (next_beat, next_sec) = clip["markers"][:2]
    return (next_beat - first_beat) / (next_sec - first_sec) * 60.0


def _clip_in_file(clip: etree._Element) -> dict[str, Any]:
    def value(path: str) -> str:
        return clip.find(path).get("Value")

    loop_start = float(value("Loop/LoopStart"))
    return {
        "name": value("Name"),
        "file_path": value("SampleRef/FileRef/Path"),
        "start_marker": loop_start + float(value("Loop/StartRelative")),
        "end_marker": float(value("CurrentEnd")),
        "loop_start": loop_start,
        "loop_end": float(value("Loop/LoopEnd")),
        "looping": value("Loop/LoopOn") == "true",
        "warping": value("IsWarped") == "true",
        "warp_mode": int(value("WarpMode")),
        "color_index": int(value("Color")),
        "muted": value("Disabled") == "true",
        "markers": [
            (float(marker.get("BeatTime")), float(marker.get("SecTime")))
            for marker in clip.findall("WarpMarkers/WarpMarker")
        ],
    }


class SetFile:
    def __init__(self, set_path: str | Path) -> None:
        live_set = etree.fromstring(gzip.decompress(Path(set_path).read_bytes()))
        tracks = list(live_set.find(".//Tracks"))
        self.track_names = [track.find("Name/EffectiveName").get("Value") for track in tracks]
        self.scene_count = len(live_set.findall(".//Scenes/Scene"))
        self.tempo = float(live_set.find(".//Tempo/Manual").get("Value"))
        self.clips: dict[Slot, dict[str, Any]] = {}
        for track_index, track in enumerate(tracks):
            slots = track.findall("DeviceChain/MainSequencer/ClipSlotList/ClipSlot")
            for row, slot in enumerate(slots):
                clip = slot.find("ClipSlot/Value/AudioClip")
                if clip is not None:
                    self.clips[(track_index, row)] = _clip_in_file(clip)

    def clip(self, track_name: str, row: int) -> dict[str, Any]:
        return self.clips[(self.track_names.index(track_name), row)]


class LiveStandIn:
    def __init__(self) -> None:
        self.receive_port = free_port()
        self.reply_port = free_port()
        self.received: list[tuple[str, tuple[Any, ...]]] = []
        self._track_names: list[str] = []
        self._clips: dict[Slot, dict[str, Any]] = {}
        self._scene_count = 0
        self._tempo = 120.0
        self._tempo_listener_registered = False
        self._auto_warps: dict[str, tuple[list[Marker], float]] = {}
        self._answering = True
        self._timers: list[threading.Timer] = []
        self._lock = threading.RLock()
        in_the_order_sent = osc_dispatcher.Dispatcher()
        in_the_order_sent.set_default_handler(
            lambda address, *args: self._answer(address, tuple(args))
        )
        self._server = osc_server.BlockingOSCUDPServer(
            ("127.0.0.1", self.receive_port), in_the_order_sent
        )
        self._reply = udp_client.SimpleUDPClient("127.0.0.1", self.reply_port)

    def __enter__(self) -> LiveStandIn:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: Any) -> None:
        for timer in self._timers:
            timer.cancel()
        self._server.shutdown()
        self._server.server_close()

    def open(self, set_path: str | Path) -> None:
        opened = SetFile(set_path)
        with self._lock:
            self._track_names = opened.track_names
            self._clips = opened.clips
            self._scene_count = opened.scene_count
            self._tempo = opened.tempo
            self._tempo_listener_registered = False
        self._reply.send_message("/live/startup", [])

    @property
    def scene_count(self) -> int:
        return self._scene_count

    def track_index(self, track_name: str) -> int:
        return self._track_names.index(track_name)

    def clip(self, track_name: str, row: int) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._clips[(self.track_index(track_name), row)])

    def has_clip(self, track_name: str, row: int) -> bool:
        with self._lock:
            return (self.track_index(track_name), row) in self._clips

    def clips(self) -> dict[Slot, dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._clips)

    def put_clip(self, track_name: str, row: int, **clip: Any) -> None:
        with self._lock:
            self._clips[(self.track_index(track_name), row)] = clip

    def auto_warps(self, file_path: str | Path, markers: list[Marker], after: float = 0.0) -> None:
        with self._lock:
            self._auto_warps[str(Path(file_path).resolve())] = (markers, after)

    def rewarp(self, track_name: str, row: int, markers: list[Marker]) -> None:
        with self._lock:
            self._clips[(self.track_index(track_name), row)]["markers"] = list(markers)

    def stop_answering(self) -> None:
        self._answering = False

    def change_tempo(self, bpm: float) -> None:
        if self._tempo_listener_registered:
            self._reply.send_message("/live/song/get/tempo", [bpm])

    def sent_to(self, address: str) -> list[tuple[Any, ...]]:
        with self._lock:
            return [args for seen, args in self.received if seen == address]

    def fired(self) -> list[Slot]:
        return [(int(args[0]), int(args[1])) for args in self.sent_to("/live/clip_slot/fire")]

    def _answer(self, address: str, args: tuple[Any, ...]) -> None:
        if not self._answering:
            return
        with self._lock:
            self.received.append((address, args))
            if address == "/live/song/start_listen/tempo":
                self._tempo_listener_registered = True
            elif address == "/live/song/get/track_names":
                self._reply.send_message(address, list(self._track_names))
            elif address == "/live/song/get/num_tracks":
                self._reply.send_message(address, [len(self._track_names)])
            elif address == "/live/song/get/num_scenes":
                self._reply.send_message(address, [self._scene_count])
            elif address == "/live/song/create_scene":
                self._scene_count += 1
            elif address == "/live/track/get/clips/name":
                track = int(args[0])
                names = [
                    self._clips.get((track, row), {}).get("name")
                    for row in range(self._scene_count)
                ]
                self._reply.send_message(address, [track, *names])
            elif address.startswith("/live/clip_slot/"):
                self._clip_slot(address.rsplit("/", 1)[1], address, args)
            elif address.startswith("/live/clip/"):
                self._clip(address, args)

    def _clip_slot(self, action: str, address: str, args: tuple[Any, ...]) -> None:
        slot = (int(args[0]), int(args[1]))
        in_range = slot[0] < len(self._track_names) and slot[1] < self._scene_count
        if action == "has_clip":
            self._reply.send_message(address, [*slot, slot in self._clips])
        elif action == "create_audio_clip" and in_range and slot not in self._clips:
            self._clips[slot] = self._new_clip(str(args[2]))
        elif action == "duplicate_clip_to" and slot in self._clips:
            self._clips[(int(args[2]), int(args[3]))] = copy.deepcopy(self._clips[slot])
        elif action == "delete_clip":
            self._clips.pop(slot, None)

    def _new_clip(self, file_path: str) -> dict[str, Any]:
        seconds = sf.info(file_path).duration
        beats = seconds * self._tempo / 60.0
        clip = {
            "name": Path(file_path).stem,
            "file_path": file_path,
            "seconds": seconds,
            "start_marker": 0.0,
            "end_marker": beats,
            "loop_start": 0.0,
            "loop_end": beats,
            "looping": False,
            "warping": True,
            "warp_mode": 0,
            "color_index": 1,
            "muted": False,
            "markers": straight_grid(self._tempo),
        }
        markers, after = self._auto_warps.get(str(Path(file_path).resolve()), (None, 0.0))
        if markers is not None and after <= 0:
            clip["markers"] = list(markers)
        elif markers is not None:
            timer = threading.Timer(after, self._auto_warp_lands, (file_path, markers))
            timer.daemon = True
            self._timers.append(timer)
            timer.start()
        return clip

    def _auto_warp_lands(self, file_path: str, markers: list[Marker]) -> None:
        with self._lock:
            for clip in self._clips.values():
                if clip["file_path"] == file_path:
                    clip["markers"] = list(markers)

    def _clip(self, address: str, args: tuple[Any, ...]) -> None:
        slot = (int(args[0]), int(args[1]))
        clip = self._clips.get(slot)
        if clip is None:
            return
        parts = address.split("/")[3:]
        if parts == ["get", "warp_marker_times"]:
            flat = [value for marker in clip["markers"] for value in marker]
            self._reply.send_message(address, [*slot, *flat])
        elif parts[0] == "get" and parts[1] in PROPERTIES_LIVE_ANSWERS:
            self._reply.send_message(address, [*slot, _read(clip, parts[1])])
        elif parts[0] == "set":
            _write(clip, parts[1], args[2])
        elif parts == ["remove_warp_marker"]:
            _remove_marker(clip, float(args[2]))
        elif parts == ["add_warp_marker"]:
            _add_marker(clip, float(args[2]), float(args[3]))


def _edge(clip: dict[str, Any], edge: str) -> str:
    if clip.get("looping", True):
        return edge
    return {"loop_start": "start_marker", "loop_end": "end_marker"}.get(edge, edge)


def _read(clip: dict[str, Any], prop: str) -> Any:
    return clip[_edge(clip, prop)]


def _write(clip: dict[str, Any], prop: str, value: Any) -> None:
    key = _edge(clip, prop)
    if key in ("start_marker", "loop_start"):
        end = clip["end_marker" if key == "start_marker" else "loop_end"]
        if float(value) < end:
            clip[key] = float(value)
    elif key in ("end_marker", "loop_end"):
        start = clip["start_marker" if key == "end_marker" else "loop_start"]
        if float(value) > start:
            clip[key] = float(value)
    elif key in ("looping", "warping", "muted"):
        clip[key] = bool(value)
    elif key in ("warp_mode", "color_index"):
        clip[key] = int(value)
    elif key == "name":
        clip[key] = str(value)


def _remove_marker(clip: dict[str, Any], beat: float) -> None:
    markers: list[Marker] = clip["markers"]
    index = next((i for i, (at, _) in enumerate(markers) if abs(at - beat) < 1e-6), None)
    last_real = len(markers) - 2
    if index is None or index > last_real or len(markers) <= 2:
        return
    if index == last_real:
        (kept_beat, kept_sec), (gone_beat, gone_sec) = markers[index - 1], markers[index]
        seconds_per_beat = (gone_sec - kept_sec) / (gone_beat - kept_beat)
        shadow = (kept_beat + SHADOW_BEATS, kept_sec + SHADOW_BEATS * seconds_per_beat)
        clip["markers"] = [*markers[:index], shadow]
    else:
        clip["markers"] = [*markers[:index], *markers[index + 1 :]]


def _add_marker(clip: dict[str, Any], beat: float, sec: float) -> None:
    markers: list[Marker] = clip["markers"]
    real, shadow = markers[:-1], markers[-1]
    earlier = [marker for marker in real if marker[0] < beat]
    later = [marker for marker in real if marker[0] > beat]
    if len(earlier) + len(later) != len(real) or not 0.0 <= sec <= clip.get("seconds", sec):
        return
    if (earlier and earlier[-1][1] >= sec) or (later and later[0][1] <= sec):
        return
    tail_seconds_per_beat = (shadow[1] - real[-1][1]) / (shadow[0] - real[-1][0])
    real = [*earlier, (beat, sec), *later]
    last_beat, last_sec = real[-1]
    clip["markers"] = [
        *real,
        (last_beat + SHADOW_BEATS, last_sec + SHADOW_BEATS * tail_seconds_per_beat),
    ]


def booth_with_live(monkeypatch) -> Iterator[tuple[TestClient, dict[str, int], LiveStandIn]]:
    track_ids = three_songs_through_the_pipeline(monkeypatch)
    settings = get_settings()
    with LiveStandIn() as live:
        bridge = AbletonBridge(
            send_port=live.receive_port,
            listen_port=live.reply_port,
            state_file=settings.data_dir / "deck_state.json",
            grid_guard_delays=GRID_GUARD_DELAYS,
        )
        app = create_app(
            settings=settings,
            bridge=bridge,
            session_factory=get_session_factory(settings.db_url),
        )
        bridge.start()
        try:
            with TestClient(app) as client:
                yield client, track_ids, live
        finally:
            bridge.stop()
    _reset_engine_for_tests()
