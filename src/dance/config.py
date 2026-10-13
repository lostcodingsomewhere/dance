"""
Configuration management for Dance.

Uses pydantic-settings to load from environment variables and ~/.dance/.env.
"""

import json
from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Local API origin guard (``dance.api.origin_guard``): the loopback hostnames
# the API answers to, and the browser origins allowed to call it — the
# companion's Vite dev server (default port and ``vite --port 5174``) under
# both loopback spellings. The origin list doubles as the CORS allowlist.
DEFAULT_API_ALLOWED_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
DEFAULT_API_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
)


def _get_env_file() -> Path:
    """Get the path to the .env file in data_dir."""
    return Path.home() / ".dance" / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="DANCE_",
        env_file=_get_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Spotify ingest
    spotify_playlist_url: Optional[str] = Field(default=None)
    # Spotify Web API credentials for the search endpoint (Client Credentials
    # flow — no user OAuth). Register an app at developer.spotify.com →
    # paste Client ID + Secret into ~/.dance/.env. Separate from spotDL's
    # bundled credentials (which are for downloads only).
    spotify_client_id: Optional[str] = Field(default=None)
    spotify_client_secret: Optional[str] = Field(default=None)
    # Optional Spotify USER OAuth token (Authorization Code flow). Required
    # for reading arbitrary public playlists' track lists since Spotify's
    # Nov 2024 API policy: Client Credentials can search the catalog and
    # fetch single tracks, but ``GET /playlists/{id}/tracks`` returns 403.
    # A user-auth token (from any OAuth-flow Spotify app — e.g. paste from
    # exportify.net's DevTools, or wire up our own OAuth flow later) does
    # work. Lives in ~/.dance/.env as ``DANCE_SPOTIFY_USER_TOKEN``.
    # Tokens last ~1 hour — when this one expires, the playlist endpoint
    # falls back to Client Credentials (which 403s on /tracks) and the
    # user will need to refresh.
    spotify_user_token: Optional[str] = Field(default=None)

    # YouTube cookies for yt-dlp ingest. Default: ~/.dance/cookies.txt if
    # present, otherwise None (yt-dlp will run unauthenticated and likely
    # hit bot detection). Export via the "Get cookies.txt LOCALLY" Chrome
    # extension — portable across machines, unlike Chrome-profile auth.
    youtube_cookies_file: Optional[Path] = Field(
        default_factory=lambda: (
            (Path.home() / ".dance" / "cookies.txt")
            if (Path.home() / ".dance" / "cookies.txt").exists()
            else None
        )
    )

    # Directory paths
    library_dir: Path = Field(default=Path.home() / "Music" / "DJ" / "library")
    stems_dir: Path = Field(default=Path.home() / "Music" / "DJ" / "stems")
    data_dir: Path = Field(default=Path.home() / ".dance")
    # Where generated Ableton Live Sets (.als) are written. Kept separate
    # from library_dir so a user can sweep it without touching audio.
    als_output_dir: Path = Field(default=Path.home() / "Music" / "Dance" / "Sets")

    # Database
    database_url: Optional[str] = Field(default=None)

    # Processing toggles
    skip_stems: bool = Field(default=False)
    skip_embeddings: bool = Field(default=False)
    audio_format: str = Field(default="mp3")
    audio_quality: str = Field(default="320k")
    # Simultaneous yt-dlp downloads across ALL ingest paths (CSV commit,
    # single-track, playlist) — one shared pool, so overlapping jobs can't
    # stack up and hammer YouTube. Read once, when the pool first starts.
    download_workers: int = Field(default=3, ge=1, le=16)

    # CLAP embeddings
    clap_model: str = Field(default="laion/clap-htsat-unfused")
    clap_device: str = Field(default="auto")  # auto, mps, cuda, cpu
    # Each audio (mix or stem) is embedded as the mean of up to this many of
    # the model's native 10 s windows, evenly spread over its non-silent
    # content. Cost scales linearly; 12 covers a 6-min track every ~30 s.
    clap_max_windows: int = Field(default=12, ge=1)
    # Windows whose RMS is below this (dBFS) are skipped — silent intros,
    # vocal stems between phrases, drums in a breakdown. An audio with no
    # window above it gets no embedding at all (readers treat that as a
    # missing feature). Changing either knob needs `dance embed --all`.
    clap_silence_db: float = Field(default=-40.0, le=0.0)

    # Demucs
    demucs_model: str = Field(default="htdemucs_ft")
    demucs_device: str = Field(default="auto")
    # Separation quality knobs. ``shifts`` > 0 averages N randomly time-shifted
    # separations to cut artifacts — ≈ N× compute for a modest gain (on a lossy
    # source the win is small; see the audio-quality A/B). ``bit_depth`` is the
    # written stem's PCM subtype: PCM_24 preserves Demucs's float headroom,
    # PCM_16 is smaller. Forward-looking — only NEW separations pick these up.
    demucs_shifts: int = Field(default=2)
    demucs_bit_depth: str = Field(default="PCM_24")

    # Recommender
    recommender_top_k: int = Field(default=20)

    # Tagging — runs locally, no API keys.
    # The default tagger is CLAP zero-shot (re-uses the already-loaded CLAP
    # model to rank a controlled vocabulary of labels against each track's
    # audio embedding — ~50 ms per track, no extra weights).
    tagger_enabled: bool = Field(default=True)
    tagger_zeroshot_threshold: float = Field(default=0.30)
    tagger_zeroshot_top_k: dict[str, int] = Field(
        default_factory=lambda: {
            "subgenre": 1,
            "mood": 3,
            "element": 4,
            "dj_note": 3,
        }
    )

    # Deep tagger (Qwen2-Audio) — opt-in. Generates free-form dj_notes.
    # Heavy: ~8 GB weights, 10-30 s/track inference on M1.
    deep_tagger_enabled: bool = Field(default=False)
    deep_tagger_model: str = Field(default="Qwen/Qwen2-Audio-7B-Instruct")
    deep_tagger_device: str = Field(default="auto")
    deep_tagger_quantize: str | None = Field(default=None)  # "4bit" | "8bit" | None

    # Daemon
    sync_interval_minutes: int = Field(default=30)

    # Logging
    log_level: str = Field(default="INFO")

    # Local API origin guard. Env values are comma-separated (or a JSON list)
    # and REPLACE the defaults, e.g. a second companion on port 5175:
    # DANCE_API_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5175
    # "*" turns the respective check off.
    api_allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_API_ALLOWED_HOSTS)
    )
    api_allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_API_ALLOWED_ORIGINS)
    )

    @field_validator("api_allowed_hosts", "api_allowed_origins", mode="before")
    @classmethod
    def split_csv(cls, v: Any) -> Any:
        if isinstance(v, str):
            text = v.strip()
            if text.startswith("["):
                return json.loads(text)
            return [part.strip() for part in text.split(",") if part.strip()]
        return v

    @field_validator(
        "library_dir", "stems_dir", "data_dir", "als_output_dir", mode="before"
    )
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v.expanduser()

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.data_dir / 'dance.db'}"

    def ensure_directories(self) -> None:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.stems_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.als_output_dir.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
