"""Picking, downloading and verifying the right recording.

The old downloader took ``ytsearch1`` — YouTube's top hit — and when the
±30 s filter rejected it, retried with no filter at all. On the real library
that produced a 414 s ``Cycles`` for a 186 s Spotify track (and 21 more files
off by >10 s), each re-tagged with the Spotify identity so it *looked* right.

These tests pin the replacement: score every candidate, gate hard on duration
(never relaxed), verify the downloaded file, and move on to the next candidate
when one fails. yt-dlp is faked (no network); ffprobe/ffmpeg run for real on
synthetic audio so the verification is genuinely exercised.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dance.spotify import csv_importer
from dance.spotify.acquire import (
    POLITE_FLAGS,
    TagValues,
    Tools,
    acquire_track,
    audio_format_spec,
)
from dance.spotify.matching import (
    MIN_SCORE,
    Candidate,
    Target,
    duration_gate_s,
    eligible,
    no_candidate_message,
    parse_search_results,
    passes_duration_gate,
    rank_candidates,
    score_candidate,
    search_queries,
    split_artists,
    unwanted_markers,
    verify_tolerance_s,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe not installed"
)

_REAL_RUN = subprocess.run
_FAKE_YT_DLP = "/fake/bin/yt-dlp"


# ---------------------------------------------------------------------------
# Synthetic audio
# ---------------------------------------------------------------------------


def _encode(path: Path, seconds: float, codec: str) -> Path:
    _REAL_RUN(
        [
            FFMPEG or "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}:sample_rate=22050",
            "-ac",
            "1",
            "-c:a",
            codec,
            "-b:a",
            "64k",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def audio(tmp_path_factory) -> dict[str, Path]:
    """mp3s of known lengths + deliberately broken variants."""
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("audio")
    out = {
        "mp3_60": _encode(d / "s60.mp3", 60, "libmp3lame"),
        "mp3_20": _encode(d / "s20.mp3", 20, "libmp3lame"),
    }
    good = out["mp3_60"].read_bytes()
    # Truncated download: the Xing header still says 60 s, so ffprobe
    # reports full length — only a real decode notices.
    out["mp3_trunc"] = d / "trunc.mp3"
    out["mp3_trunc"].write_bytes(good[: len(good) // 2])
    # Bit-rot in the middle: probes fine, decodes with "Header missing".
    damaged = bytearray(good)
    for i in range(len(damaged) // 3, len(damaged) // 3 + 20_000):
        damaged[i] = (i * 7919) % 256
    out["mp3_corrupt"] = d / "corrupt.mp3"
    out["mp3_corrupt"].write_bytes(bytes(damaged))
    return out


@pytest.fixture(scope="module")
def m4a_audio(tmp_path_factory) -> dict[str, Path]:
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("m4a")
    try:
        aac = _encode(d / "aac60.m4a", 60, "aac")
        alac = _encode(d / "alac60.m4a", 60, "alac")
    except subprocess.CalledProcessError:
        pytest.skip("this ffmpeg cannot write m4a")
    return {"aac_60": aac, "alac_60": alac}


# ---------------------------------------------------------------------------
# Fake yt-dlp
# ---------------------------------------------------------------------------


def _entry(vid: str, title: str, uploader: str, duration: float | None, **extra) -> dict:
    """One ``--flat-playlist -J`` entry, shaped like yt-dlp 2026's."""
    return {
        "_type": "url",
        "ie_key": "Youtube",
        "id": vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": title,
        "uploader": uploader,
        "channel": uploader,
        "duration": duration,
        **extra,
    }


class FakeYtDlp:
    """Stands in for the yt-dlp binary; every other command runs for real.

    ``results`` maps the search query (text after ``ytsearch10:``) to entries;
    ``media`` maps a video id to the file its "download" produces (None =
    the upload is unavailable).
    """

    def __init__(
        self,
        results: dict[str, list[dict]],
        media: dict[str, Path | None] | None = None,
    ) -> None:
        self.results = results
        self.media = media or {}
        self.search_cmds: list[list[str]] = []
        self.download_cmds: list[list[str]] = []

    @property
    def queries(self) -> list[str]:
        return [c[-1].split(":", 1)[1] for c in self.search_cmds]

    @property
    def downloads(self) -> list[str]:
        return [c[-1].rsplit("v=", 1)[1] for c in self.download_cmds]

    def __call__(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        if cmd[0] != _FAKE_YT_DLP:
            return _REAL_RUN(cmd, **kwargs)
        if "--flat-playlist" in cmd:
            self.search_cmds.append(cmd)
            query = cmd[-1].split(":", 1)[1]
            payload = {"_type": "playlist", "entries": self.results.get(query, [])}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")
        self.download_cmds.append(cmd)
        vid = cmd[-1].rsplit("v=", 1)[1]
        src = self.media.get(vid)
        if src is None:
            return subprocess.CompletedProcess(
                cmd, 1, "", f"ERROR: [youtube] {vid}: Video unavailable"
            )
        template = cmd[cmd.index("--output") + 1]
        ext = "m4a" if "bestaudio[ext=m4a]" in cmd else "mp3"
        shutil.copyfile(src, template.replace("%(ext)s", ext))
        return subprocess.CompletedProcess(cmd, 0, "", "")


def _tools() -> Tools:
    return Tools(yt_dlp=_FAKE_YT_DLP, ffmpeg=FFMPEG or "ffmpeg", ffprobe=FFPROBE or "ffprobe")


def _acquire(
    fake: FakeYtDlp,
    target: Target,
    dest: Path,
    fmt: str = "mp3",
    work_root: Path | None = None,
):
    return acquire_track(
        target,
        TagValues(title=target.title, artist=", ".join(target.artists), album="LP"),
        dest,
        audio_format_spec(fmt),
        _tools(),
        runner=fake,
        work_root=work_root,
    )


def _ffprobe(path: Path) -> dict:
    proc = _REAL_RUN(
        [
            FFPROBE or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:format_tags=comment:stream=codec_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Scoring table (pure)
# ---------------------------------------------------------------------------

CYCLES = Target(title="Cycles", artists=("CamelPhat", "Arodes"), duration_s=185.8)


def test_scoring_table_realistic_titles() -> None:
    """Every case from the brief, scored against the real ``Cycles`` identity."""
    rows = {
        "official": Candidate(
            "OFFICIALaud", "CamelPhat & Arodes - Cycles (Official Audio)", "CamelPhat", 186.0
        ),
        "topic": Candidate("TOPICchanne", "Cycles", "CamelPhat - Topic", 186.0),
        "extended": Candidate(
            "EXTENDEDmix", "CamelPhat & Arodes - Cycles (Extended Mix)", "Spinnin' Records", 414.0
        ),
        "live": Candidate(
            "LIVEatPrint", "CamelPhat & Arodes - Cycles (Live at Printworks)", "CamelPhat", 200.0
        ),
        "slowed": Candidate("SLOWEDreverb", "Cycles (slowed + reverb)", "sad edits", 205.0),
        "nightcore": Candidate("NIGHTCOREcy", "Nightcore - Cycles", "NightcoreX", 170.0),
        "hour_mix": Candidate(
            "HOURLONGmix", "CamelPhat - Live @ Printworks London (Full Set)", "Mixmag", 3600.0
        ),
    }
    ranked = rank_candidates(CYCLES, list(rows.values()))
    by_id = {s.candidate.video_id: s for s in ranked}

    # Only the two genuine uploads qualify, official audio first.
    assert [s.candidate.video_id for s in eligible(ranked)] == ["OFFICIALaud", "TOPICchanne"]
    assert by_id["OFFICIALaud"].score > by_id["TOPICchanne"].score >= MIN_SCORE

    # Wrong-length versions die at the hard gate, whatever their score.
    assert "duration 414s" in by_id["EXTENDEDmix"].rejection
    assert "duration 3600s" in by_id["HOURLONGmix"].rejection
    # Right-length but wrong *version* uploads die on the version penalty.
    assert by_id["LIVEatPrint"].duration_ok and by_id["LIVEatPrint"].score < MIN_SCORE
    assert by_id["NIGHTCOREcy"].duration_ok and by_id["NIGHTCOREcy"].score < MIN_SCORE
    assert by_id["SLOWEDreverb"].duration_ok and by_id["SLOWEDreverb"].score < MIN_SCORE
    assert by_id["LIVEatPrint"].unwanted_markers == ("live",)
    assert by_id["EXTENDEDmix"].unwanted_markers == ("extended",)
    assert by_id["NIGHTCOREcy"].unwanted_markers == ("nightcore",)


def test_topic_channel_gets_official_bonus() -> None:
    plain = Candidate("PLAINupload", "Cycles", "CamelPhat", 186.0)
    topic = Candidate("TOPICupload", "Cycles", "CamelPhat - Topic", 186.0)
    assert score_candidate(CYCLES, topic) == pytest.approx(score_candidate(CYCLES, plain) + 0.04)


def test_version_words_free_when_the_request_names_them() -> None:
    """Remix requested → the remix upload is not penalised; original requested
    → the same upload is, and loses to the original."""
    remix_req = Target("Surrender - TH;EN Remix", ("Sian", "KASIA", "Burko", "TH;EN"), 210.2)
    orig_req = Target("Surrender", ("Sian", "KASIA", "Burko"), 200.2)
    remix_up = Candidate(
        "REMIXupload", "Sian, KASIA, Burko - Surrender (TH;EN Remix)", "Armada Music", 210.0
    )
    orig_up = Candidate("ORIGupload1", "Sian, KASIA & Burko - Surrender", "Armada Music", 200.0)

    assert unwanted_markers(remix_req, remix_up.title) == ()
    assert score_candidate(remix_req, remix_up) >= MIN_SCORE
    assert unwanted_markers(orig_req, remix_up.title) == ("remix",)
    ranked = rank_candidates(orig_req, [remix_up, orig_up])
    assert ranked[0].candidate.video_id == "ORIGupload1"

    extended_req = Target("Cycles - Extended Mix", ("CamelPhat", "Arodes"), 414.0)
    ext_up = Candidate(
        "EXTENDEDmix", "CamelPhat & Arodes - Cycles (Extended Mix)", "Spinnin' Records", 414.0
    )
    assert eligible(rank_candidates(extended_req, [ext_up]))


def test_markers_match_whole_words_and_ignore_the_artist_name() -> None:
    alive = Target("Alive", ("RÜFÜS DU SOL",), 240.0)
    assert unwanted_markers(alive, "RÜFÜS DU SOL - Alive (Official Video)") == ()
    band = Target("Lightning Crashes", ("Live",), 325.0)
    assert unwanted_markers(band, "Live - Lightning Crashes") == ()
    assert unwanted_markers(band, "Live - Lightning Crashes (Live at Woodstock)") == ()
    assert unwanted_markers(CYCLES, "Cycles (Karaoke Version)") == ("karaoke",)


def test_duration_gate_bounds_and_unknowns() -> None:
    assert duration_gate_s(185.8) == 30.0
    assert duration_gate_s(600.0) == pytest.approx(60.0)
    assert passes_duration_gate(185.8, 215.0)
    assert not passes_duration_gate(185.8, 216.0)
    # An unmeasurable upload never passes when we know what we want…
    assert not passes_duration_gate(185.8, None)
    # …and with nothing to gate on, nothing is gated.
    assert passes_duration_gate(None, 3600.0)
    assert verify_tolerance_s(185.8) == 10.0
    assert verify_tolerance_s(600.0) == pytest.approx(21.0)


def test_live_streams_are_never_eligible() -> None:
    stream = Candidate(
        "STREAMnow01", "CamelPhat & Arodes - Cycles", "CamelPhat", 186.0, live_status="is_live"
    )
    (only,) = rank_candidates(CYCLES, [stream])
    assert not only.eligible and "live stream" in only.rejection


def test_parse_search_results_keeps_only_downloadable_ids() -> None:
    payload = {
        "entries": [
            _entry("abcdefghijk", "Good", "Chan", 200),
            _entry("short", "Bad id", "Chan", 200),
            _entry("ABCDEFGHIJK", "", "Chan", 200),
            {"id": "zyxwvutsrqp", "title": "No uploader", "channel": "C", "duration": "201.5"},
            "not a dict",
        ]
    }
    got = parse_search_results(payload)
    assert [c.video_id for c in got] == ["abcdefghijk", "zyxwvutsrqp"]
    assert got[1].uploader == "C" and got[1].duration_s == 201.5
    assert [c.rank for c in got] == [0, 3]
    assert [c.rank for c in parse_search_results(payload, rank_offset=10)] == [10, 13]
    assert parse_search_results([]) == []


def test_search_queries_all_artists_then_primary() -> None:
    assert search_queries(CYCLES) == ["CamelPhat Arodes Cycles", "CamelPhat Cycles"]
    assert search_queries(Target("Strobe", ("deadmau5",), 600)) == ["deadmau5 Strobe"]
    assert split_artists("Faul & Wad, MESTIZA, Martina Camargo") == (
        "Faul & Wad",
        "MESTIZA",
        "Martina Camargo",
    )
    assert split_artists("Argy;Son of Son") == ("Argy", "Son of Son")
    # Seen live: a name containing ";" survives a comma-joined credit line.
    assert split_artists("Sian, KASIA, Burko, TH;EN") == ("Sian", "KASIA", "Burko", "TH;EN")


def test_score_ties_keep_youtube_relevance_order() -> None:
    a = Candidate("zzzzzzzzzzz", "CamelPhat, Arodes - Cycles", "Reuploads", 186.0, rank=0)
    b = Candidate("aaaaaaaaaaa", "CamelPhat, Arodes - Cycles", "Reuploads", 186.0, rank=1)
    ranked = rank_candidates(CYCLES, [b, a])
    assert ranked[0].score == ranked[1].score
    assert [s.candidate.video_id for s in ranked] == ["zzzzzzzzzzz", "aaaaaaaaaaa"]


def test_no_candidate_message_names_the_best_near_miss() -> None:
    ranked = rank_candidates(
        CYCLES,
        [Candidate("EXTENDEDmix", "CamelPhat & Arodes - Cycles (Extended Mix)", "X", 414.0)],
    )
    msg = no_candidate_message(ranked)
    assert msg.startswith("no candidate passed (best 0.")
    assert "Cycles (Extended Mix) 414s" in msg


# ---------------------------------------------------------------------------
# Acquisition loop (fake yt-dlp, real ffprobe/ffmpeg)
# ---------------------------------------------------------------------------

T60 = Target(title="Sixty", artists=("Synth Artist", "Guest"), duration_s=60.0)
Q_FULL, Q_PRIMARY = search_queries(T60)


@needs_ffmpeg
def test_gate_never_relaxed_nothing_downloaded(tmp_path, audio) -> None:
    """Only wrong-length uploads exist → fail, and yt-dlp never downloads.

    The old code's second attempt dropped the filter and took the top hit."""
    wrong = [
        _entry("EXTENDED001", "Synth Artist - Sixty (Extended Mix)", "Label", 240),
        _entry("HOURSET0001", "Synth Artist - Sixty (1 Hour Loop)", "Loops", 3600),
    ]
    fake = FakeYtDlp(
        {Q_FULL: wrong, Q_PRIMARY: wrong},
        media={"EXTENDED001": audio["mp3_60"], "HOURSET0001": audio["mp3_60"]},
    )
    library = tmp_path / "library"
    dest = library / "Synth Artist - Sixty.mp3"

    result = _acquire(fake, T60, dest)

    assert result.status == "fail"
    assert result.message.startswith("no candidate passed (best ")
    assert fake.downloads == []
    # Primary-artist fallback was tried before giving up — same gate.
    assert fake.queries == [Q_FULL, Q_PRIMARY]
    assert not dest.exists()


@needs_ffmpeg
def test_primary_artist_query_only_when_needed(tmp_path, audio) -> None:
    fake = FakeYtDlp(
        {Q_FULL: [_entry("GOODupload1", "Synth Artist, Guest - Sixty", "Synth Artist", 60)]},
        media={"GOODupload1": audio["mp3_60"]},
    )
    result = _acquire(fake, T60, tmp_path / "Synth Artist - Sixty.mp3")
    assert result.status == "ok", result.message
    assert fake.queries == [Q_FULL]


@needs_ffmpeg
def test_primary_artist_fallback_finds_single_credit_upload(tmp_path, audio) -> None:
    fake = FakeYtDlp(
        {Q_FULL: [], Q_PRIMARY: [_entry("PRIMARYonly", "Sixty", "Synth Artist - Topic", 61)]},
        media={"PRIMARYonly": audio["mp3_60"]},
    )
    result = _acquire(fake, T60, tmp_path / "Synth Artist - Sixty.mp3")
    assert result.status == "ok", result.message
    assert fake.downloads == ["PRIMARYonly"]


@needs_ffmpeg
def test_wrong_length_file_rejected_then_next_candidate(tmp_path, audio) -> None:
    """Search metadata said 60 s, the file is 20 s: verification catches it,
    deletes only that attempt's file, and the next candidate succeeds."""
    fake = FakeYtDlp(
        {
            Q_FULL: [
                _entry(
                    "LIESABOUTit",
                    "Synth Artist, Guest - Sixty (Official Audio)",
                    "Synth Artist",
                    60,
                ),
                _entry("TOPICgood01", "Sixty", "Synth Artist - Topic", 60),
            ]
        },
        media={"LIESABOUTit": audio["mp3_20"], "TOPICgood01": audio["mp3_60"]},
    )
    library = tmp_path / "library"
    library.mkdir()
    (library / "Someone Else - Keep Me.mp3").write_bytes(b"x" * 1000)
    work = tmp_path / "work"
    work.mkdir()
    dest = library / "Synth Artist - Sixty.mp3"

    result = _acquire(fake, T60, dest, work_root=work)

    assert result.status == "ok", result.message
    assert fake.downloads == ["LIESABOUTit", "TOPICgood01"]
    assert result.chosen is not None and result.chosen.candidate.video_id == "TOPICgood01"
    assert "yt:TOPICgood01 score" in result.message
    assert abs(result.measured_s - 60.0) < 1.0
    # The rejected 20 s file is gone; nothing else in the library was touched.
    assert sorted(p.name for p in library.iterdir()) == [
        "Someone Else - Keep Me.mp3",
        "Synth Artist - Sixty.mp3",
    ]
    assert list(work.iterdir()) == []
    probe = _ffprobe(dest)
    assert abs(float(probe["format"]["duration"]) - 60.0) < 1.0


@needs_ffmpeg
def test_truncated_and_corrupt_downloads_rejected(tmp_path, audio) -> None:
    fake = FakeYtDlp(
        {
            Q_FULL: [
                _entry(
                    "TRUNCATED01",
                    "Synth Artist, Guest - Sixty (Official Audio)",
                    "Synth Artist",
                    60,
                ),
                _entry("CORRUPTED01", "Synth Artist, Guest - Sixty", "Synth Artist Official", 60),
                _entry("GOODupload1", "Sixty", "Synth Artist - Topic", 60),
            ]
        },
        media={
            "TRUNCATED01": audio["mp3_trunc"],
            "CORRUPTED01": audio["mp3_corrupt"],
            "GOODupload1": audio["mp3_60"],
        },
    )
    result = _acquire(fake, T60, tmp_path / "Synth Artist - Sixty.mp3")
    assert result.status == "ok", result.message
    # Both broken uploads outrank the good one, so both get tried first.
    assert sorted(fake.downloads[:2]) == ["CORRUPTED01", "TRUNCATED01"]
    assert fake.downloads[2] == "GOODupload1"
    assert result.chosen is not None and result.chosen.candidate.video_id == "GOODupload1"


@needs_ffmpeg
def test_every_eligible_failure_is_reported(tmp_path, audio) -> None:
    fake = FakeYtDlp(
        {
            Q_FULL: [
                _entry(
                    "GONEvideo01",
                    "Synth Artist, Guest - Sixty (Official Audio)",
                    "Synth Artist",
                    60,
                ),
                _entry("WRONGlength", "Sixty", "Synth Artist - Topic", 60),
            ]
        },
        media={"GONEvideo01": None, "WRONGlength": audio["mp3_20"]},
    )
    library = tmp_path / "library"
    result = _acquire(fake, T60, library / "Synth Artist - Sixty.mp3")
    assert result.status == "fail"
    assert "2 eligible candidate(s) failed" in result.message
    assert "yt:GONEvideo01 yt-dlp rc=1" in result.message
    assert "yt:WRONGlength duration 20.0s vs 60.0s expected" in result.message
    assert not library.exists() or list(library.iterdir()) == []


@needs_ffmpeg
def test_downloads_by_video_id_with_polite_flags(tmp_path, audio) -> None:
    fake = FakeYtDlp(
        {Q_FULL: [_entry("GOODupload1", "Synth Artist, Guest - Sixty", "Synth Artist", 60)]},
        media={"GOODupload1": audio["mp3_60"]},
    )
    assert _acquire(fake, T60, tmp_path / "x.mp3").status == "ok"
    (search,) = fake.search_cmds
    assert search[-1] == f"ytsearch10:{Q_FULL}"
    assert "--flat-playlist" in search and "-J" in search
    (download,) = fake.download_cmds
    assert download[-1] == "https://www.youtube.com/watch?v=GOODupload1"
    assert "--no-playlist" in download
    assert not any("ytsearch" in arg for arg in download)
    assert "--match-filter" not in download
    for cmd in (search, download):
        joined = " ".join(cmd)
        assert " ".join(POLITE_FLAGS) in joined
    assert "--sleep-requests 1 --sleep-interval 2 --max-sleep-interval 5" in " ".join(download)
    assert "--retries 3 --retry-sleep exp=1:20" in " ".join(download)


@needs_ffmpeg
def test_mp3_tags_carry_identity_and_provenance(tmp_path, audio) -> None:
    from mutagen.id3 import ID3

    fake = FakeYtDlp(
        {Q_FULL: [_entry("GOODupload1", "Synth Artist, Guest - Sixty", "Synth Artist", 60)]},
        media={"GOODupload1": audio["mp3_60"]},
    )
    dest = tmp_path / "Synth Artist, Guest - Sixty.mp3"
    result = _acquire(fake, T60, dest)
    assert result.status == "ok", result.message

    tags = ID3(dest)
    assert str(tags["TIT2"]) == "Sixty"
    assert str(tags["TPE1"]) == "Synth Artist, Guest"
    assert str(tags["TALB"]) == "LP"
    (comment,) = tags.getall("COMM")
    assert "https://www.youtube.com/watch?v=GOODupload1" in str(comment)
    assert f"match score {result.chosen.score:.2f}" in str(comment)
    # Readable by the tool the rest of the pipeline uses, too.
    assert "GOODupload1" in _ffprobe(dest)["format"]["tags"]["comment"]


@needs_ffmpeg
def test_m4a_keeps_aac_stream_and_tags_mp4(tmp_path, m4a_audio) -> None:
    from mutagen.mp4 import MP4

    fake = FakeYtDlp(
        {Q_FULL: [_entry("AACupload01", "Synth Artist, Guest - Sixty", "Synth Artist", 60)]},
        media={"AACupload01": m4a_audio["aac_60"]},
    )
    dest = tmp_path / "Synth Artist - Sixty.m4a"
    result = _acquire(fake, T60, dest, fmt="m4a")
    assert result.status == "ok", result.message

    (download,) = fake.download_cmds
    assert download[download.index("-f") + 1] == "bestaudio[ext=m4a]"
    assert "--extract-audio" not in download and "--audio-format" not in download
    probe = _ffprobe(dest)
    assert probe["streams"][0]["codec_name"] == "aac"
    tags = MP4(dest).tags
    assert tags["\xa9nam"] == ["Sixty"]
    assert tags["\xa9ART"] == ["Synth Artist, Guest"]
    assert "watch?v=AACupload01" in tags["\xa9cmt"][0]


@needs_ffmpeg
def test_m4a_rejects_a_non_aac_stream(tmp_path, m4a_audio) -> None:
    fake = FakeYtDlp(
        {Q_FULL: [_entry("ALACupload1", "Synth Artist, Guest - Sixty", "Synth Artist", 60)]},
        media={"ALACupload1": m4a_audio["alac_60"]},
    )
    result = _acquire(fake, T60, tmp_path / "x.m4a", fmt="m4a")
    assert result.status == "fail"
    assert "codec alac (expected aac)" in result.message


# ---------------------------------------------------------------------------
# download_track — the entry point the API and the CLI script share
# ---------------------------------------------------------------------------


def _csv_row(**kw) -> csv_importer.CsvRow:
    base = dict(artist="Synth Artist, Guest", title="Sixty", album="LP", duration_s=60)
    base.update(kw)
    return csv_importer.CsvRow(**base)


def test_expected_target_extension_follows_format(tmp_path) -> None:
    row = _csv_row()
    assert csv_importer.expected_target(tmp_path, row).suffix == ".mp3"
    assert csv_importer.expected_target(tmp_path, row, "m4a").suffix == ".m4a"
    assert (
        csv_importer.expected_target(tmp_path, row, "mp3").name == "Synth Artist, Guest - Sixty.mp3"
    )


def test_audio_format_spec() -> None:
    mp3 = audio_format_spec("mp3")
    assert mp3.ext == "mp3" and "ffmpeg:-b:a 320k" in mp3.yt_dlp_args
    m4a = audio_format_spec("M4A")
    assert m4a.ext == "m4a" and m4a.codec == "aac"
    assert "--extract-audio" not in m4a.yt_dlp_args
    with pytest.raises(ValueError):
        audio_format_spec("wma")


@needs_ffmpeg
@pytest.mark.parametrize("fmt", ["mp3", "m4a"])
def test_download_track_end_to_end_with_fake_yt_dlp(
    tmp_path, monkeypatch, audio, m4a_audio, fmt
) -> None:
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    src = audio["mp3_60"] if fmt == "mp3" else m4a_audio["aac_60"]
    fake = FakeYtDlp(
        {
            "Synth Artist Guest Sixty": [
                _entry("GOODupload1", "Synth Artist, Guest - Sixty", "Synth Artist", 60)
            ]
        },
        media={"GOODupload1": src},
    )
    library = tmp_path / "library"
    status, msg = csv_importer.download_track(
        _csv_row(),
        library,
        cookies_file=tmp_path / "cookies.txt",
        audio_format=fmt,
        runner=fake,
    )
    assert status == "ok", msg
    assert "yt:GOODupload1 score" in msg and "60s vs 60s" in msg
    assert (library / f"Synth Artist, Guest - Sixty.{fmt}").exists()
    # --cookies is forwarded to every yt-dlp call.
    assert all("--cookies" in c for c in fake.search_cmds + fake.download_cmds)

    # Second call is a no-op skip — nothing searched again.
    status, msg = csv_importer.download_track(
        _csv_row(),
        library,
        cookies_file=tmp_path / "cookies.txt",
        audio_format=fmt,
        runner=fake,
    )
    assert (status, msg) == ("skip", "already downloaded")
    assert len(fake.search_cmds) == 1


def test_download_track_unsupported_format_fails_cleanly(tmp_path) -> None:
    status, msg = csv_importer.download_track(
        _csv_row(), tmp_path, cookies_file=tmp_path / "c.txt", audio_format="wma"
    )
    assert status == "fail" and "unsupported audio_format" in msg


@needs_ffmpeg
def test_download_track_without_duration_still_verifies_decode(
    tmp_path, monkeypatch, audio
) -> None:
    """A CSV row with no duration skips the length gates — but a truncated
    file is still caught by decoded-vs-header length."""
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    fake = FakeYtDlp(
        {
            "Synth Artist Sixty": [
                _entry("TRUNCATED01", "Synth Artist - Sixty", "Synth Artist - Topic", 60),
                _entry("GOODupload1", "Synth Artist - Sixty", "Synth Artist", 60),
            ]
        },
        media={"TRUNCATED01": audio["mp3_trunc"], "GOODupload1": audio["mp3_60"]},
    )
    status, msg = csv_importer.download_track(
        _csv_row(artist="Synth Artist", duration_s=0),
        tmp_path,
        cookies_file=tmp_path / "c.txt",
        runner=fake,
    )
    assert status == "ok", msg
    assert fake.downloads == ["TRUNCATED01", "GOODupload1"]


def _never(*_a, **_k) -> subprocess.CompletedProcess[str]:  # pragma: no cover - guard
    raise AssertionError("yt-dlp must not run")


def test_download_track_skips_existing_without_running_anything(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    row = _csv_row()
    target = csv_importer.expected_target(tmp_path, row)
    target.write_bytes(b"\x00" * 200_000)
    assert csv_importer.download_track(
        row, tmp_path, cookies_file=tmp_path / "c.txt", runner=_never
    ) == ("skip", "already downloaded")
