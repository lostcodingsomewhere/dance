"""
Configuration management for Dance.

Uses pydantic-settings to load from environment variables and ~/.dance/.env.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Directory paths
    library_dir: Path = Field(default=Path.home() / "Music" / "DJ" / "library")
    stems_dir: Path = Field(default=Path.home() / "Music" / "DJ" / "stems")
    data_dir: Path = Field(default=Path.home() / ".dance")

    # Database
    database_url: Optional[str] = Field(default=None)

    # Processing toggles
    skip_stems: bool = Field(default=False)
    skip_embeddings: bool = Field(default=False)
    audio_format: str = Field(default="mp3")
    audio_quality: str = Field(default="320k")

    # CLAP embeddings
    clap_model: str = Field(default="laion/clap-htsat-unfused")
    clap_device: str = Field(default="auto")  # auto, mps, cuda, cpu

    # Demucs
    demucs_model: str = Field(default="htdemucs_ft")
    demucs_device: str = Field(default="auto")

    # Recommender
    recommender_top_k: int = Field(default=20)

    # Daemon
    sync_interval_minutes: int = Field(default=30)

    # Logging
    log_level: str = Field(default="INFO")

    @field_validator("library_dir", "stems_dir", "data_dir", mode="before")
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
