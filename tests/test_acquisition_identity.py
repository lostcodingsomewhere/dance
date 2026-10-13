"""Hard identity gates: right artist, right version, sane length.

The duration gate cannot tell a live take, a cover, a slowed edit or a VIP
mix from the studio cut when they run about as long — and an exact title on
a "- Topic" channel scored 0.79 with *no* artist match at all. These tests
pin the gates that close that hole (``HARD_VERSION_MARKERS`` and
``artist_credited``), the softer penalties around them, and the 12-minute cap
used when there is no Spotify duration. Pure: no subprocess, no network.
"""

from __future__ import annotations

import pytest

from dance.spotify.matching import (
    HARD_VERSION_MARKERS,
    MIN_SCORE,
    UNKNOWN_TARGET_MAX_S,
    Candidate,
    Target,
    artist_credited,
    eligible,
    no_candidate_message,
    normalize_match_text,
    rank_candidates,
    score_candidate,
    unwanted_markers,
)

OMT = Target("One More Time", ("Daft Punk",), 320.0)
STUDIO = Candidate("STUDIOaudio", "Daft Punk - One More Time (Official Audio)", "Daft Punk", 320.0)


def _only(target: Target, cand: Candidate):
    (scored,) = rank_candidates(target, [cand])
    return scored


# ---------------------------------------------------------------------------
# Different-audio markers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "duration", "marker"),
    [
        ("(Live)", 322.0, "live"),
        ("(Live at Coachella 2006)", 318.0, "live"),
        ("(Cover)", 321.0, "cover"),
        ("(Slowed)", 330.0, "slowed"),
        ("(Sped Up)", 300.0, "sped up"),
        ("(Nightcore)", 300.0, "nightcore"),
        ("(Karaoke Version)", 320.0, "karaoke"),
        ("(Instrumental)", 320.0, "instrumental"),
        ("(Acapella)", 320.0, "acapella"),
        ("(8D Audio)", 320.0, "8d"),
        ("(Someone Remix)", 325.0, "remix"),
        ("(VIP Mix)", 315.0, "vip"),
        ("(Bootleg)", 318.0, "bootleg"),
        ("(Someone Rework)", 322.0, "rework"),
        ("(Someone Flip)", 317.0, "flip"),
        ("(Dub Mix)", 315.0, "dub mix"),
        ("(Preview)", 330.0, "preview"),
        ("[Snippet]", 300.0, "snippet"),
    ],
)
def test_unrequested_version_is_rejected_whatever_its_score(
    suffix: str, duration: float, marker: str
) -> None:
    """Right artist, right title, inside the duration gate — still not the
    track. Several of these used to clear MIN_SCORE (Live 0.80, Cover 0.80,
    Slowed 0.79, VIP Mix 0.98) and were downloaded whenever the studio upload
    was not in the top ten."""
    cand = Candidate("VERSIONupld", f"Daft Punk - One More Time {suffix}", "Daft Punk", duration)
    scored = _only(OMT, cand)
    assert scored.duration_ok
    assert not scored.eligible
    assert scored.rejection.startswith("different version (")
    assert marker in scored.rejection
    assert marker in HARD_VERSION_MARKERS


def test_studio_upload_is_the_only_eligible_one_among_versions() -> None:
    versions = [
        Candidate("LIVEversion", "Daft Punk - One More Time (Live)", "Daft Punk", 322.0, rank=0),
        Candidate("COVERversio", "One More Time (Cover)", "Daft Punk Covers", 321.0, rank=1),
        Candidate("VIPversion1", "Daft Punk - One More Time (VIP Mix)", "Daft Punk", 315.0, rank=2),
        Candidate("SLOWEDversi", "Daft Punk - One More Time (Slowed)", "slowed", 330.0, rank=3),
    ]
    ranked = rank_candidates(OMT, [*versions, STUDIO])
    assert [s.candidate.video_id for s in eligible(ranked)] == ["STUDIOaudio"]
    # With the studio upload missing, nothing is downloaded at all.
    ranked = rank_candidates(OMT, versions)
    assert eligible(ranked) == []
    assert no_candidate_message(ranked).startswith("no candidate passed (best 0.")


def test_requested_versions_are_not_rejected() -> None:
    live_req = Target("One More Time - Live", ("Daft Punk",), 322.0)
    assert _only(
        live_req, Candidate("LIVEversion", "Daft Punk - One More Time (Live)", "Daft Punk", 322.0)
    ).eligible
    vip_req = Target("One More Time - VIP Mix", ("Daft Punk",), 315.0)
    assert _only(
        vip_req, Candidate("VIPversion1", "Daft Punk - One More Time (VIP Mix)", "Daft Punk", 315.0)
    ).eligible
    # A "- Dub" request accepts a "(Dub Mix)" upload: the phrase marker
    # counts as requested through its marker word.
    dub_req = Target("One More Time - Dub", ("Daft Punk",), 315.0)
    assert unwanted_markers(dub_req, "Daft Punk - One More Time (Dub Mix)") == ()
    # A remix request still rejects a *different* unrequested version.
    remix_req = Target("One More Time - Someone Remix", ("Daft Punk",), 325.0)
    assert _only(
        remix_req,
        Candidate(
            "REMIXlive01", "Daft Punk - One More Time (Someone Remix) (Live)", "Daft Punk", 325.0
        ),
    ).rejection == ("different version (live)")


def test_benign_phrases_and_genre_tags() -> None:
    # "(Cover Art)" labels the still image of an official audio upload.
    art = Candidate("COVERart001", "Daft Punk - One More Time (Cover Art)", "Daft Punk", 320.0)
    assert unwanted_markers(OMT, art.title) == ()
    assert _only(OMT, art).eligible
    assert unwanted_markers(OMT, "Daft Punk - One More Time [Album Cover]") == ()
    # A bare "dub" is usually a genre tag: penalised, not rejected.
    tagged = Candidate("DUBtechno01", "Daft Punk - One More Time [Dub Techno]", "Daft Punk", 320.0)
    scored = _only(OMT, tagged)
    assert scored.unwanted_markers == ("dub",)
    assert scored.eligible
    assert scored.score == pytest.approx(score_candidate(OMT, STUDIO) - 0.18, abs=0.002)


# ---------------------------------------------------------------------------
# Artist credit
# ---------------------------------------------------------------------------

SURRENDER = Target("Surrender", ("Sian", "KASIA", "Burko"), 200.2)
VISION = Target("Vision Blurred", ("Kaskade", "CID", "Anabel Englund"), 185.6)


def test_different_artist_on_a_topic_channel_is_rejected() -> None:
    """Exact title + right length + official-looking channel = 0.79 with no
    artist match: the case that made the gate necessary."""
    sian = Target("Surrender", ("Sian",), 200.0)
    other = Candidate("NATALIEtayl", "Surrender", "Natalie Taylor - Topic", 198.0)
    assert score_candidate(sian, other) >= MIN_SCORE
    scored = _only(sian, other)
    assert scored.duration_ok and not scored.eligible
    assert scored.rejection == "artist not credited in title or channel"
    assert scored.artist_ok is False


@pytest.mark.parametrize(
    ("target", "title", "uploader", "credited"),
    [
        # 1. more than half of the credited artists' words in the title
        (SURRENDER, "Sian, KASIA & Burko - Surrender", "Octopus Recordings", True),
        (SURRENDER, "Burko & KASIA (ofc) Feat. Sian – Surrender", "Power Station", True),
        # 2. the part before " - " names only credited artists
        (VISION, "Kaskade - Vision Blurred", "EDM Uploads", True),
        (VISION, "Kaskade – Vision Blurred", "EDM Uploads", True),
        (VISION, "Kaskade & Deadmau5 - Vision Blurred", "EDM Uploads", False),
        (SURRENDER, "Natalie Taylor - Surrender", "Lyrics Hub", False),
        # 3. a channel named only for credited artists
        (VISION, "Vision Blurred", "Kaskade - Topic", True),
        (VISION, "Vision Blurred", "KaskadeVEVO", True),
        (VISION, "Vision Blurred", "Anabel Englund Official", True),
        (SURRENDER, "Surrender", "Sian Music", True),
        # ...not one that merely contains an artist's word
        (SURRENDER, "Surrender", "Dr Tammy Sian", False),
        (SURRENDER, "Surrender", "Asian Music", False),
        (SURRENDER, "Surrender", None, False),
        # exactly half of the words is not "more than half"
        (Target("Surrender", ("Sian", "KASIA"), 200.0), "Surrender (KASIA edit)", "x", False),
    ],
)
def test_artist_credited(target: Target, title: str, uploader: str | None, credited: bool) -> None:
    cand = Candidate("CANDIDATE01", title, uploader, target.duration_s)
    assert artist_credited(target, cand) is credited


def test_artist_gate_skips_credits_with_nothing_identifying() -> None:
    # Single Latin letters carry no identity: nothing to check, so pass.
    assert artist_credited(Target("Song", ("X",), 200.0), Candidate("CANDIDATE01", "Song", "Y"))


def test_non_latin_names_normalise_to_themselves() -> None:
    """The old ASCII-only normaliser turned Cyrillic / CJK / Hangul into ""
    — every artist then "matched" (coverage 0.5 for an empty credit)."""
    assert normalize_match_text("Кино") == "кино"
    assert normalize_match_text("米津玄師 - Lemon") == "米津玄師 lemon"
    assert normalize_match_text("아이유") == "아이유"  # syllables, not loose jamo
    assert normalize_match_text("RÜFÜS DU SOL") == "rufus du sol"  # accents still fold
    kino = Target("Группа крови", ("Кино",), 285.0)
    assert artist_credited(kino, Candidate("KINOupload1", "Кино - Группа крови", "Fan"))
    assert not artist_credited(kino, Candidate("COVERband01", "Группа крови", "Другая группа"))
    # One CJK character can be a whole name.
    arashi = Target("Love so sweet", ("嵐",), 260.0)
    assert artist_credited(arashi, Candidate("ARASHIupld1", "嵐 - Love so sweet", "Fan"))
    assert not artist_credited(arashi, Candidate("SOMEONEelse", "Love so sweet", "Fan"))


# ---------------------------------------------------------------------------
# No Spotify duration
# ---------------------------------------------------------------------------


def test_unknown_target_duration_caps_length() -> None:
    """Without a Spotify duration, "Artist - Title" at an hour scored 0.85
    on name alone and was eligible."""
    no_dur = Target("One More Time", ("Daft Punk",), None)
    hour = Candidate("FULLSET0001", "Daft Punk - One More Time", "Daft Punk", 3600.0)
    radio = Candidate("RADIOcut001", "Daft Punk - One More Time", "Daft Punk", 320.0)
    unknown = Candidate("NOduration1", "Daft Punk - One More Time", "Daft Punk", None)
    assert score_candidate(no_dur, hour) >= MIN_SCORE
    by_id = {s.candidate.video_id: s for s in rank_candidates(no_dur, [hour, radio, unknown])}
    assert by_id["FULLSET0001"].rejection == (
        f"duration 3600s > {UNKNOWN_TARGET_MAX_S:.0f}s cap (no expected duration)"
    )
    assert by_id["NOduration1"].rejection == "duration unknown"
    assert by_id["RADIOcut001"].eligible


# ---------------------------------------------------------------------------
# Real search results (cached `yt-dlp --flat-playlist -J ytsearch10:` output)
# ---------------------------------------------------------------------------


def test_real_results_surrender_uploads_without_artist_credit_are_rejected() -> None:
    """#206 in the real library: the only right-length uploads come from a
    channel called "Dr Tammy Sian" with no artist in the title. One "Sian" in
    a channel name is not a credit — the track fails with a near-miss
    message rather than taking a guess."""
    results = [
        Candidate(
            "FX8c79jKsj4",
            "Sian, KASIA, Burko - Surrender (TH;EN Remix) - Octopus Recordings",
            "Octopus Recordings",
            376.0,
            rank=0,
        ),
        Candidate(
            "_ijVYNWwzYI",
            "Sian, KASIA, Burko - Surrender - Octopus Recordings",
            "Octopus Recordings",
            373.0,
            rank=1,
        ),
        Candidate("eBrFZrAuhME", "Surrender (TH;EN Remix)", "Dr Tammy Sian ", 211.0, rank=3),
        Candidate("4QKkwtfj4VY", "Surrender", "Dr Tammy Sian ", 201.0, rank=4),
        # The primary-artist fallback ("Sian Surrender") adds other acts.
        Candidate("e6pD_cDk3V0", "Sian WaXX - Surrender", "SianWaXX", 116.0, rank=16),
        Candidate("-dGu8UY2gY0", "Surrender", "Release - Topic", 232.0, rank=17),
    ]
    ranked = rank_candidates(SURRENDER, results)
    assert eligible(ranked) == []
    by_id = {s.candidate.video_id: s for s in ranked}
    assert by_id["4QKkwtfj4VY"].rejection == "artist not credited in title or channel"
    assert by_id["eBrFZrAuhME"].rejection == "different version (remix)"
    # Wrong length here — but not credited either, had the length matched.
    sian_waxx = Candidate("e6pD_cDk3V0", "Sian WaXX - Surrender", "SianWaXX", 200.0)
    release = Candidate("-dGu8UY2gY0", "Surrender", "Release - Topic", 200.0)
    assert not artist_credited(SURRENDER, sian_waxx)
    assert not artist_credited(SURRENDER, release)
    assert no_candidate_message(ranked).startswith("no candidate passed (best 0.83: Surrender 201s")


def test_real_results_cycles_and_vision_blurred_pick_the_radio_cut() -> None:
    """#300 / #337: YouTube's top hit is the extended mix the library has
    today (414 s / 340 s); the scorer picks a credited upload at Spotify's
    length."""
    cycles = Target("Cycles", ("CamelPhat", "Arodes"), 185.8)
    ranked = rank_candidates(
        cycles,
        [
            Candidate(
                "D3J3cMf9sMg", "CamelPhat & Arodes - Cycles", "Obsessive Progressive", 414.0, rank=0
            ),
            Candidate(
                "R1vNAwTTQP4", "CamelPhat & Arodes - Cycles", "THE VOID - [JOURNEY]", 186.0, rank=1
            ),
            Candidate(
                "YocfdTiF0XI",
                "CamelPhat & Arodes - Cycles (Extended Mix)",
                "Stranger Network",
                413.0,
                rank=2,
            ),
        ],
    )
    assert ranked[0].candidate.video_id == "R1vNAwTTQP4" and ranked[0].eligible
    assert [s.candidate.video_id for s in eligible(ranked)] == ["R1vNAwTTQP4"]

    ranked = rank_candidates(
        VISION,
        [
            Candidate(
                "sWXs5LwEVkE",
                "Kaskade x CID x Anabel Englund - Vision Blurred [Extended Mix]",
                "Experts Only and 3 more",
                340.0,
                rank=0,
            ),
            Candidate(
                "E2ScHHGuwpY",
                "Kaskade, CID & Anabel Englund - Vision Blurred (Preview)",
                "AnabelEnglundTV",
                202.0,
                rank=6,
            ),
            Candidate(
                "Uqh_ueRuI9U",
                "Kaskade, CID & Anabel Englund – Vision Blurred",
                "ML Music",
                187.0,
                rank=4,
            ),
        ],
    )
    assert ranked[0].candidate.video_id == "Uqh_ueRuI9U"
    by_id = {s.candidate.video_id: s for s in ranked}
    assert "duration 340s" in by_id["sWXs5LwEVkE"].rejection
    # 202 s is inside the ±30 s search gate; the "(Preview)" marker rejects it.
    assert by_id["E2ScHHGuwpY"].rejection == "different version (preview)"
