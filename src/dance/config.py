"""
Configuration management for Dance DJ Pipeline.

Uses pydantic-settings to load from environment variables and .env files.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_traktor_path() -> Optional[Path]:
    """Find Traktor collection.nml in common locations."""
    traktor_paths = [
        Path.home() / "Documents" / "Native Instruments" / "Traktor Pro 4",
        Path.home() / "Documents" / "Native Instruments" / "Traktor Pro 3",
        Path.home() / "Documents" / "Native Instruments" / "Traktor 3.11.1",
    ]

    for base_path in traktor_paths:
        collection = base_path / "collection.nml"
        if collection.exists():
            return collection

    return None


def _get_env_file() -> Path:
    """Get the path to the .env file in data_dir."""
    data_dir = Path.home() / ".dance"
    return data_dir / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="DANCE_",
        env_file=_get_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Spotify configuration
    spotify_playlist_url: Optional[str] = Field(
        default=None,
        description="Spotify playlist URL to sync tracks from",
    )

    # Directory paths
    library_dir: Path = Field(
        default=Path.home() / "Music" / "DJ" / "library",
        description="Directory for downloaded and processed tracks",
    )
    stems_dir: Path = Field(
        default=Path.home() / "Music" / "DJ" / "stems",
        description="Directory for Demucs stem separation output",
    )
    data_dir: Path = Field(
        default=Path.home() / ".dance",
        description="Directory for database and application data",
    )

    # Traktor integration
    traktor_collection_path: Optional[Path] = Field(
        default=None,
        description="Path to Traktor collection.nml file",
    )

    # Database
    database_url: Optional[str] = Field(
        default=None,
        description="SQLAlchemy database URL (defaults to SQLite in data_dir)",
    )

    # Processing options
    skip_stems: bool = Field(
        default=False,
        description="Skip stem separation (faster but less accurate cue detection)",
    )
    skip_llm: bool = Field(
        default=False,
        description="Skip LLM augmentation (Qwen2-Audio tagging and validation)",
    )
    audio_format: str = Field(
        default="mp3",
        description="Audio format for downloads (mp3, flac, wav)",
    )
    audio_quality: str = Field(
        default="320k",
        description="Audio quality/bitrate for downloads",
    )

    # LLM configuration (Qwen2-Audio)
    llm_model: str = Field(
        default="Qwen/Qwen2-Audio-7B-Instruct",
        description="HuggingFace model ID for audio understanding",
    )
    llm_device: str = Field(
        default="auto",
        description="Device for LLM inference (auto, mps, cuda, cpu)",
    )
    llm_quantize: Optional[str] = Field(
        default="4bit",
        description="Quantization level (4bit, 8bit, or none for full precision)",
    )
    llm_cache_responses: bool = Field(
        default=True,
        description="Cache LLM responses to avoid re-processing",
    )

    # Daemon settings
    sync_interval_minutes: int = Field(
        default=30,
        description="How often to check Spotify playlist for new tracks (daemon mode)",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    @field_validator("library_dir", "stems_dir", "data_dir", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand ~ and convert to Path."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v.expanduser()

    @field_validator("traktor_collection_path", mode="before")
    @classmethod
    def expand_traktor_path(cls, v: str | Path | None) -> Optional[Path]:
        """Expand ~ and convert to Path, or auto-detect."""
        if v is None:
            return get_default_traktor_path()
        if isinstance(v, str):
            return Path(v).expanduser()
        return v.expanduser()

    @property
    def db_url(self) -> str:
        """Get database URL, defaulting to SQLite in data_dir."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.data_dir / 'dance.db'}"

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.stems_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_to_env_file(self, path: Optional[Path] = None) -> None:
        """Save current settings to .env file."""
        if path is None:
            path = self.data_dir / ".env"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        lines = []
        if self.spotify_playlist_url:
            lines.append(f"DANCE_SPOTIFY_PLAYLIST_URL={self.spotify_playlist_url}")
        lines.append(f"DANCE_LIBRARY_DIR={self.library_dir}")
        lines.append(f"DANCE_STEMS_DIR={self.stems_dir}")
        lines.append(f"DANCE_DATA_DIR={self.data_dir}")
        if self.traktor_collection_path:
            lines.append(f"DANCE_TRAKTOR_COLLECTION_PATH={self.traktor_collection_path}")
        lines.append(f"DANCE_SKIP_STEMS={str(self.skip_stems).lower()}")
        lines.append(f"DANCE_SKIP_LLM={str(self.skip_llm).lower()}")
        lines.append(f"DANCE_AUDIO_FORMAT={self.audio_format}")
        lines.append(f"DANCE_AUDIO_QUALITY={self.audio_quality}")
        lines.append(f"DANCE_LLM_MODEL={self.llm_model}")
        lines.append(f"DANCE_LLM_DEVICE={self.llm_device}")
        if self.llm_quantize:
            lines.append(f"DANCE_LLM_QUANTIZE={self.llm_quantize}")
        lines.append(f"DANCE_LLM_CACHE_RESPONSES={str(self.llm_cache_responses).lower()}")
        lines.append(f"DANCE_SYNC_INTERVAL_MINUTES={self.sync_interval_minutes}")
        lines.append(f"DANCE_LOG_LEVEL={self.log_level}")

        path.write_text("\n".join(lines) + "\n")


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
