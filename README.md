# Dance - DJ Track Analysis Pipeline

Automated cue points, energy tagging, and Traktor integration for house/techno DJs.

**Spotify playlist → downloaded → analyzed → Traktor-ready in one pipeline**

## Quick Start

```bash
# Install
pip install -e .

# Configure Spotify playlist
dance config --spotify-playlist "https://open.spotify.com/playlist/YOUR_PLAYLIST"

# Run full pipeline
dance run --once
```

## Features

- **Spotify-first workflow**: Add tracks to playlist → they appear in Traktor
- **Energy scoring (1-10)**: Objective metric for set building
- **Auto cue points**: Color-coded at phrase boundaries (intro, drop, breakdown, outro)
- **Camelot key detection**: For harmonic mixing
- **MPS-accelerated stems**: Fast Demucs separation on Apple Silicon
- **LLM audio understanding**: Rich tagging via Qwen2-Audio (optional)

## Pipeline Stages

```
INGEST → ANALYZE → SEPARATE → LLM AUGMENT → DETECT CUES → EXPORT
           ↓          ↓            ↓              ↓           ↓
        Essentia   Demucs    Qwen2-Audio    Phrase-snap   Traktor
        BPM/Key    Stems     Genre/Mood     Cue points    NML
```

## Commands

```bash
# Show current configuration
dance config --show

# Sync from Spotify (download new tracks)
dance sync

# Process tracks (analyze, separate, detect cues)
dance process

# Process without stems or LLM (faster)
dance process --skip-stems --skip-llm

# Export to Traktor
dance export

# Full pipeline in daemon mode (continuous sync)
dance run

# Show pipeline status
dance status

# List tracks with filters
dance list --energy 8 --bpm-range 125-130 --key 8A
```

## LLM Augmentation (Optional)

Dance can use [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) to add rich metadata:

- **Subgenre classification**: tech house, melodic techno, progressive house, etc.
- **Mood tags**: dark, driving, hypnotic, uplifting, groovy
- **Notable elements**: acid line, vocal chops, rolling bassline
- **DJ notes**: "Good peak-time track", "Long intro for mixing"
- **Contextual cue names**: "Drop 1 - Big acid synth" instead of just "Drop 1"
- **Quality validation**: Verify BPM/key detection accuracy

### Install LLM Dependencies

```bash
pip install -e ".[llm]"
```

### Hardware Requirements

| Mac | Memory | Quantization | Time/Track |
|-----|--------|--------------|------------|
| M1 Pro | 16GB | 4-bit | ~12-15 sec |
| M1/M2 Max | 32GB | 8-bit or none | ~8-10 sec |
| M3 Max/Ultra | 48GB+ | none | ~3-5 sec |

### LLM Commands

```bash
# Check LLM status and GPU info
dance llm-status

# Analyze specific track with LLM
dance llm-analyze --track-id 123

# Re-analyze completed tracks
dance llm-analyze --reanalyze --limit 10

# Disable LLM (environment variable)
export DANCE_SKIP_LLM=true
```

### Traktor Export with LLM

When LLM is enabled, tracks export with enhanced metadata:

**Comments field:**
```
[Tech House] dark, driving | Peak-time banger | E8 128bpm 8A
```

**Cue point names:**
- `Intro - Minimal kick pattern`
- `Drop 1 - Big acid synth`
- `Breakdown - Vocal chop loop`
- `Outro - Filter sweep out`

## Configuration

Settings are loaded from environment variables or `~/.dance/.env`:

```bash
# Directories
DANCE_LIBRARY_DIR=~/Music/DJ/library
DANCE_STEMS_DIR=~/Music/DJ/stems
DANCE_DATA_DIR=~/.dance

# Traktor (auto-detected if not set)
DANCE_TRAKTOR_COLLECTION_PATH=~/Documents/Native Instruments/Traktor Pro 4/collection.nml

# Processing
DANCE_SKIP_STEMS=false
DANCE_SKIP_LLM=false

# LLM Configuration
DANCE_LLM_MODEL=Qwen/Qwen2-Audio-7B-Instruct
DANCE_LLM_DEVICE=auto  # auto, mps, cuda, cpu
DANCE_LLM_QUANTIZE=4bit  # 4bit, 8bit, or none

# Daemon mode
DANCE_SYNC_INTERVAL_MINUTES=30
```

## Database

Tracks are stored in SQLite with content-based deduplication (SHA256 hash). The database includes:

- Track metadata (artist, title, file path)
- Analysis results (BPM, key, energy, mood)
- LLM augmentation (subgenre, tags, cue contexts)
- Beat grid and phrase boundaries
- Cue points with colors and names
- Stem file paths

## Architecture

```
src/dance/
├── cli.py              # Click CLI commands
├── config.py           # Pydantic settings
├── core/
│   └── database.py     # SQLAlchemy models
├── pipeline/
│   ├── orchestrator.py # Pipeline coordinator
│   └── stages/
│       ├── ingest.py   # File scanning
│       ├── analyze.py  # Essentia analysis
│       ├── separate.py # Demucs stems
│       ├── llm_augment.py  # Qwen2-Audio
│       └── detect_cues.py  # Cue placement
├── llm/
│   ├── qwen_audio.py   # Model wrapper
│   └── prompts.py      # Prompt templates
├── spotify/
│   └── downloader.py   # spotDL wrapper
└── export/
    └── traktor.py      # NML file export
```

## Spotify Setup

By default, spotDL uses shared API credentials that can get rate-limited. For reliable downloads, set up your own credentials:

### Option 1: Your Own API Credentials (Recommended)

1. Go to https://developer.spotify.com/dashboard
2. Create an app to get your `client_id` and `client_secret`
3. Update `~/.spotdl/config.json`:

```json
{
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET"
}
```

### Option 2: OAuth User Auth

```bash
spotdl --user-auth
```

This opens a browser to log in with your Spotify account and uses your personal auth token.

## Dependencies

Core:
- `essentia` - Audio analysis (BPM, key, energy)
- `demucs` - Stem separation
- `librosa` - Audio loading (fallback)
- `spotdl` - Spotify downloads
- `traktor-nml-utils` - Traktor integration

Optional (LLM):
- `transformers` - Hugging Face models
- `accelerate` - Model optimization
- `bitsandbytes` - Quantization

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/dance

# Linting
ruff check src/dance
```

## License

MIT
