# Tagging

Local-only tagging. No API keys, no cloud calls. Two modes; pick per invocation.

## Modes

| Mode      | Class                      | Weights      | Latency       | Source value in DB | Notes |
|-----------|----------------------------|--------------|---------------|--------------------|-------|
| CLAP zero-shot (default) | `ClapZeroShotTagger` (`src/dance/llm/tagger.py:167`) | none extra (re-uses the embedded CLAP model) | ~50 ms / track | `inferred` | Constrained vocabulary; reads the stored `track_embeddings` row |
| Qwen2-Audio deep         | `Qwen2AudioTagger` (`src/dance/llm/qwen_audio.py:99`) | ~8 GB (or ~4 GB at 4-bit) | ~10-30 s / track on M1 | `llm` | Free-form; listens to audio + reads the analytical brief |

Both write to the same `tags` / `track_tags` schema. Both leave `manual` tags alone.

## CLAP zero-shot — how it works

`src/dance/llm/tagger.py`. CLAP is a joint audio-text embedding. We:

1. Encode each candidate label in the controlled vocabulary as a CLAP text vector (cached in-process).
2. Load the track's already-stored full-mix audio embedding from `track_embeddings`.
3. Cosine-rank labels per kind; emit top-K above `tagger_zeroshot_threshold`.

Track must be in state `complete` and have a full-mix CLAP embedding. If you ran with `--skip-embeddings`, this tagger errors out.

## Controlled vocabulary

`src/dance/llm/tagger.py:52`. Four `TagKind`s:

| Kind         | Default top-K | Vocabulary list             |
|--------------|---------------|-----------------------------|
| `SUBGENRE`   | 1             | `SUBGENRE_LABELS` (18 items) |
| `MOOD`       | 3             | `MOOD_LABELS` (18 items)     |
| `ELEMENT`    | 4             | `ELEMENT_LABELS` (19 items)  |
| `DJ_NOTE`    | 3             | `DJ_NOTE_LABELS` (11 items)  |

To extend: append to the relevant list. The label is encoded by CLAP on first use (one `~10 ms` text forward pass) and cached for the process lifetime. No retraining, no rebuild step. The next `dance tag` run will consider the new label.

```python
# src/dance/llm/tagger.py
SUBGENRE_LABELS: list[str] = [
    "tech house",
    "deep house",
    ...
    "afrobeats",   # <-- append here
]
```

Caveat: each new label dilutes the relative scores. Keep lists DJ-useful, not exhaustive.

## Tuning

`src/dance/config.py:64`:

```python
tagger_enabled: bool = True
tagger_zeroshot_threshold: float = 0.30
tagger_zeroshot_top_k: dict[str, int] = {
    "subgenre": 1,
    "mood": 3,
    "element": 4,
    "dj_note": 3,
}
```

- `tagger_zeroshot_threshold` — labels with cosine below this are dropped, even if they're in the top-K. Lower it (e.g. `0.25`) for more recall, raise it for precision.
- `tagger_zeroshot_top_k` — per-kind cap. `subgenre: 1` is intentional (one genre per track).

Override via env: `DANCE_TAGGER_ZEROSHOT_THRESHOLD=0.25`.

## Tag sources and re-tag semantics

`src/dance/core/database.py:105`:

```python
class TagSource(str, Enum):
    LLM = "llm"          # Qwen2-Audio output
    MANUAL = "manual"    # user-applied
    INFERRED = "inferred"  # CLAP zero-shot
```

Both taggers' `_write_tags` (`tagger.py:287`, `qwen_audio.py:263`) do:

```sql
DELETE FROM track_tags WHERE track_id = ? AND source = '<this source>'
-- ...then re-insert
```

So re-running the CLAP tagger replaces only `inferred` rows. Manual tags survive. Re-running Qwen replaces only `llm` rows.

CLI: `dance tag` skips tracks that already have rows from the chosen source (`cli.py:336`). Pass `--retag` to re-process them. `--track-id` always re-tags the named track.

## Output

```python
@dataclass
class TaggerResponse:                # tagger.py:142, qwen_audio.py:73
    subgenre: str | None
    mood_tags: list[str]
    element_tags: list[str]
    dj_notes: list[str]
    scores: dict[str, float]         # "{kind}:{label}" -> cosine
    model: str | None
```

CLI prints the first 6 tags per track:

```
Tagged 1 track(s) via CLAP zero-shot (laion/clap-htsat-unfused)...
  ✓ Untitled: minimal techno, minimal, groovy, peak-time banger

Tagged: 1  Errors: 0
```

(Output from a dry-run on synthetic 4-on-the-floor audio — the labels are vocabulary picks, not a guarantee of musicological accuracy on real tracks.)

## Enabling Qwen2-Audio

Off by default. To enable:

```bash
export DANCE_DEEP_TAGGER_ENABLED=true
dance tag --deep --track-id 42
```

Or per-track via API:

```bash
curl -X POST 'http://127.0.0.1:8000/api/v1/tracks/42/tag?deep=true'
```

First call downloads `Qwen/Qwen2-Audio-7B-Instruct` (~8 GB) to `~/.cache/huggingface`. Subsequent calls reuse the cached model.

System prompt + JSON contract: `src/dance/llm/qwen_audio.py:63`. The model sees both the audio (first 60 s, downsampled to 16 kHz) and an analytical brief built by `src/dance/llm/brief.py` (BPM, key, energy, per-stem presence, sections).

## Quantization (low-memory machines)

`src/dance/config.py:80`:

```python
deep_tagger_quantize: str | None = None  # "4bit" | "8bit" | None
```

- `None` — full float16. Best quality, ~8 GB resident.
- `"4bit"` — `BitsAndBytesConfig` nf4, ~4 GB. Requires `bitsandbytes` (installed on Linux/Windows via `pyproject.toml`; **not** on macOS — see the platform marker at `pyproject.toml:45`).
- `"8bit"` — `BitsAndBytesConfig` 8-bit, ~5 GB.

On macOS / MPS, `4bit`/`8bit` is silently downgraded to float16 (`qwen_audio.py:126`). MPS itself is supported but the model is loaded onto CPU first then moved — if that move fails it falls back to CPU and warns (`qwen_audio.py:152`).

## When to use which

- Use CLAP zero-shot as the default for the whole library. It's effectively free once the embedding stage has run.
- Use Qwen2-Audio sparingly for tracks where you need free-form `dj_notes` ("peak-time, female vocal sits in the mid-range, long instrumental tail"). Run it once per track and let it sit.
- Combine: CLAP gives you subgenre/mood; Qwen fills in nuance. Both sources coexist in `track_tags` so the UI can render the union.
