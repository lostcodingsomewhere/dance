# Stem quality: put a vocal specialist in front of htdemucs_ft

**Status:** PROPOSAL. Awaiting a decision. No code has been written yet.
**Date:** 2026-10-13 (facts re-checked the same day; sources linked inline)
**Touches:** `pipeline/stages/separate.py` (internals, plus the skip condition), `config.py`
(new keys), one new optional dependency. **No** model/Alembic, Stage-protocol, `.als`, OSC or
API-schema change. Rule 1 still applies because this is the user's call on the model, the
dependency and roughly a day of compute.

---

## Where we are

`separate` runs `htdemucs_ft` with `shifts=2` and writes PCM_24 (`config.py:95-104`,
`separate.py:323-337`). `htdemucs_ft` is a bag of four HTDemucs nets, one per source. Demucs
4.0.1's `remote/htdemucs_ft.yaml` gives the bag diagonal weights, but `apply_model`
(`demucs/apply.py:179-200`) still runs every sub-model over the whole track, and each sub-model
`shifts` times (`:206-218`). That is **8 full forward passes per track**, and 3 of every 4
outputs are multiplied by 0.

- **Cost, measured:** a 30 s excerpt took 31.7 s of separation on MPS at s2 (commit `994f886`).
  That is ≈1.06 s per audio-second, ≈0.13 s per pass.
- **Library (per the brief, not re-measured here; this task does not read `~/.dance` or
  `~/Music`):** 278 tracks, 1,112 stems, ~52 GB (≈187 MB per track). At 44.1 kHz stereo that
  is ≈47 MB per stem, i.e. ≈3 min of PCM_24 or ≈4.4 min of PCM_16, so the **average track is
  probably 4–5 min, not 6**. Time estimates below give a 4–6 min range.
  - Full re-separation today: 278 × 4–6 min × 1.06 ≈ **20–29 h**. `f14e99d` estimated "37h"
    for a full library re-run at s2 (no track count or basis given).
- **Weights:** 4 × 84 MB (84,141,271 bytes each) in `~/.cache/torch/hub/checkpoints`.
- **Weak stem: vocals.** On the like-for-like table below, the htdemucs_ft vocals specialist
  scores 8.38 dB; public vocal specialists score ~11 dB. Our drums and bass are level with the
  best clearly licensed public 4-stem model.

Vocals matter more here than average SDR suggests. The vocals fader is the main performance
move (acapella over the other deck, or muting vocals to dodge a clash), and two errors hurt it:

- **Vocal bleed into `other`:** muting the vocals fader still leaves ghost vocals.
- **Synth leakage into `vocals`:** this flips `vocal_present` (`presence_ratio > 0.15`,
  `analyze_stems.py:197`), the vocal regions (`_PRESENCE_THRESHOLD = 0.3`,
  `detect_regions.py:59`), and skews the per-stem vocals CLAP embedding (`embed.py:170-172`)
  behind vocals-column recs.

## The benchmark numbers, read carefully

**Like-for-like: Multisong, whole-track SDR (dB).** All rows except BS-RoFormer SW come from
one table, [MSST `pretrained_models.md`](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md).

| model | vocals | drums | bass | other | checkpoint | licence |
|---|---|---|---|---|---|---|
| **htdemucs_ft** (per-source specialists, ours)¹ | 8.38 | 11.13 | 11.96 | 5.85 | 4 × 84 MB | Demucs repo MIT; weights not separately licensed |
| Kim Mel-Band RoFormer (vocals/other) | **10.98** | – | – | – | [913 MB](https://huggingface.co/KimberleyJSN/melbandroformer) | MIT (HF licence tag; the card has no text, training data undisclosed) |
| viperx BS-RoFormer ep317 (vocals/other) | 10.87 | – | – | – | 639 MB (UVR `model_repo` release) | none stated |
| SCNet XL IHF (4-stem) | 9.68 | 11.58 | 11.94 | 6.48 | [214 MB](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.15) | MSST repo MIT; trained on MUSDB18-HQ only (100 songs) |
| BS-RoFormer SW (6-stem) | 11.30 | 14.11 | 14.62 | 8.71² | not released | **unknown** ([MVSEP, 2025-09-15](https://mvsep.com/news/64)) |
| MelBandRoformer4StemFTLarge (djbooth) | no Multisong figure | | | | 3.76 GB | Apache-2.0 |

¹ The single-stem table that lists the four HTDemucs4 FT specialists does not name its
dataset. Multisong is inferred: the vocal-model table's note says Multisong, and the plain
HTDemucs4 Multisong row (vocals 8.24, drums 10.88, bass 11.76, other 5.74) is consistent.
² A 6-stem `other` excludes guitar and piano, so it is not comparable with the 4-stem rows.

**MUSDB18 test: two different metrics, not rankable against each other.**

- **HT Demucs fine-tuned:** 9.00 avg (drums 10.08, bass 10.39, other 6.32, vocals 9.20),
  [paper](https://arxiv.org/abs/2211.08553) Table. Metric: SiSEC18, median over 1-second
  chunks per song, then median over songs.
- **djbooth's MelBandRoformer4StemFTLarge:** 8.99 avg (vocals 10.70, drums 10.43, bass 7.74,
  other 7.10), [card](https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge). Metric:
  MSST whole-track SDR, mean over songs
  ([`utils/metrics.py` `sdr()`](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/utils/metrics.py)).
  Its `ver2.ckpt` scores lower (8.87).
- **On djbooth's own yardstick:** SCNet XL IHF scores 10.08 (bass 9.23) and MSST's 4-stem
  BS-RoFormer 9.65 (bass 8.48). djbooth's model trails SCNet XL IHF by ~1.1 dB, and its bass
  is the lowest of the three.

**Caveats:** training sets differ (htdemucs_ft adds 800 songs; SCNet XL IHF saw only
MUSDB18-HQ). Nobody publishes overlap checks against Multisong. None of these sets is club
music, and our vocals are sparse, chopped and heavily effected. The numbers rank candidates;
listening decides.

## Options

All options keep exactly `drums/bass/vocals/other`. That is a hard requirement: `StemKind`
(`database.py:86-92`), `SOURCE_STEM_KINDS` (`als/writer.py:97`), the bridge's
`_SOURCE_STEM_KINDS` (`osc/bridge.py:882`) and `_STEM_COLUMNS` (`api/routers/ableton.py:808`)
all assume those four. A 6-stem model would fit only by summing guitar+piano+other.

### 0. Status quo

- **Pros:** no work, no risk.
- **Cons:** vocals stay ~2.6 dB behind the specialists, and we keep paying for 8 passes where
  the hybrid needs 3 plus one RoFormer pass.

### 1. djbooth's 4-stem RoFormer (MelBandRoformer4StemFTLarge)

- **Quality:** not shown to beat htdemucs_ft (above). Its weakest stem is bass, the stem that
  clashes in a blend, so it is a regression risk exactly where we are strong.
- **Compute:** djbooth reports "faster than real time" on a 12-core M2 Max / 32 GB via MLX
  (djbooth `packages/rekordbox-ts/PORTING.md:329-331`; a 10.88 s demo in 1.49 s including model
  load, `:351-352`). Torch on MPS on this M2 Pro (16-core GPU, checked with `system_profiler`)
  is unmeasured; an M2 Max has 30 or 38 GPU cores.
- **Memory:** 3.76 GB checkpoint; djbooth measured 3.96 GB max RSS (MLX).
- **Packaging / risk signal:**
  - djbooth's MLX path needed a monkeypatch to restore the checkpoint's `skip_connection`
    schedule, because `mlx-audio-separator` 0.1.5 "omits that config flag and adds an untrained
    terminal normalization" (djbooth `python/stems-mlx/djbooth_stems_mlx.py:128-172`,
    `PORTING.md:343-346`). The weights load either way, so the unpatched output is presumably
    wrong without raising — a red flag for an unattended pipeline. (Inference: djbooth only
    reports agreeing with the PyTorch reference *after* patching.)
  - djbooth's wrapper pins Python `>=3.12,<3.13`; `mlx-audio-separator` itself allows `>=3.10`
    and lists 3.13. Via torch (MSST code), the skip-connection issue does not arise.
- **Verdict:** a 3.76 GB download with no demonstrated quality gain. Reject.

### 2. Hybrid: RoFormer vocals, then htdemucs_ft drums/bass/other on (mix − vocals) ← **recommended**

**How it works:**
1. `V` = Kim Mel-RoFormer vocals of the mix.
2. `I = mix − V`.
3. Run only htdemucs_ft sub-models 0–2 (drums, bass, other) on `I`, `shifts=0`, keeping each
   one's own source. Call `apply_model` per sub-model: a 3-member `BagOfModels` with sliced
   weights leaves the vocals total at 0 and divides 0/0 (`apply.py:197-198`).
4. Drop the htdemucs vocals specialist.

This is the MDX23 Leaderboard C 3rd-place recipe: vocals first, then Demucs "to the
instrumental track only", with the instrumental obtained "by subtracting vocals from the
mixture" ([Solovyev et al. 2023, §6.2.1](https://arxiv.org/abs/2305.07489)).

- **Expected quality (Multisong):**
  - vocals ≈ 10.98 (+2.6 dB);
  - drums/bass ≈ 11.13/11.96, assumed unchanged. MDX23's version gained here too
    (11.69/12.50 vs HT Demucs 4's 11.30/12.10,
    [repo table](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model)), but it
    ensembled four Demucs variants, so we don't count on it;
  - `other` ≥ 5.85, since vocals are removed first (MDX23 reached 6.54 with the ensemble).
- **Compute (per audio-second):**
  - htdemucs half: 3 passes ≈ **0.40 s/s** (measured per-pass rate × 3).
  - Kim pass: **unmeasured**. Back-of-envelope from its
    [config](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml)
    (`dim 384`, `depth 6` × time+freq transformers, `num_bands 60`, hop 441 → 6,000
    tokens/s, 8 s chunks, `num_overlap 2`): ≈0.7 TFLOP per audio-second, counting matmuls only
    (could be off 2×). This machine's 16-core M2 Pro GPU peaks at ≈5.7 TFLOPS fp32; at 30–50%
    utilisation that is **≈0.25–0.4 s/s**.
  - Total ≈ **0.65–0.8 s/s** vs today's 1.06 — likely *cheaper* than now. Stage 0 gates on it.
  - Dropping shifts on a lossy spotDL source costs little: `config.py:98-100` records the s2
    win as small.
- **Memory:** 913 MB + 3 × 84 MB of checkpoints plus per-chunk activations; estimate 2–4 GB
  per worker (`DANCE_GPU_CONCURRENCY` defaults to 1, `pipeline/_gpu.py:20`). Full-track fp32
  buffers (mix, V, I, D, B, O) are ≈0.8 GB for a 6-min track.
- **Licence:** Kim weights MIT (see caveat in the table). htdemucs_ft unchanged.
- **Packaging** (PyPI JSON checked 2026-10-13):
  - **[`msst`](https://pypi.org/project/msst/) 0.1.0** (Roman Solovyev/ZFTurbo, MIT, one
    release, 2026-09-09, "Beta"): `requires-python >=3.10,<3.14`, `torch>=2.0.1,<2.12` — fits
    our 3.13.3 + torch 2.9.1. Base deps add `ml-collections`, `omegaconf`, `matplotlib`; the
    `mel-band-roformer` extra adds `beartype, einops, packaging, rotary-embedding-torch,
    PoPE-pytorch`. API: `msst.Separator(model_type=..., config_path=..., checkpoint_path=...)`.
    The repo's `inference.py:197-204` falls back cuda → mps → cpu; whether `Separator` does the
    same is **unverified**.
  - **[`audio-separator`](https://pypi.org/project/audio-separator/):** 0.46.0 (2026-08-27) and
    0.47.0 require **`torch>=2.13` on macOS arm64** — a forced torch upgrade under Demucs, CLAP
    and Qwen. 0.45.0 allows `torch>=2.3` but pins `beartype<0.19` and
    `rotary-embedding-torch<0.7`, and brings a model-download manager we don't need.
  - **Vendoring** MSST's MIT `mel_band_roformer.py`: smallest footprint, but we own the code.
- **Layout:** unchanged. Each stem row keeps its own `model_used` (`String(64)`,
  `database.py:280`), e.g. `melrof-kim|pcm24|msst0.1.0` on vocals and
  `htdemucs_ft-novox|s0|pcm24|demucs4.0.1` on the others. `stem_provenance`
  (`separate.py:124-137`) hard-codes a `|demucs<ver>` suffix, so the vocals row needs a sibling
  helper.
- **Open choice: how `other` is defined** (decision 3):
  - **2a:** the htdemucs other-specialist's output. The residual QC keeps its meaning (−20.7 dB
    on the one excerpt measured in `994f886`).
  - **2b:** `I − D − B`. Stems sum to the mix exactly, so the residual QC falls to the
    PCM_24/clipping floor and stops flagging anything; every drums/bass miss lands in `other`.
  - Both come out of one run, so the A/B renders both.

### 3. One stronger 4-stem model

- **SCNet XL IHF:** +1.30 vocals, +0.45 drums, −0.02 bass, +0.63 other vs htdemucs_ft. One
  214 MB network instead of four; MPS speed unmeasured. Vocals still 1.3 dB below Kim, and
  trained on 100 MUSDB18 songs, so club-music behaviour is open. Packaging: `msst[scnet]`.
  **Verdict:** a reasonable optional third arm in Stage 0, or a later swap for the
  drums/bass/other half.
- **BS-RoFormer SW:** best public numbers by far, but no weights release or licence on MVSEP's
  announcement, and the HF rehost `jarredou/BS-ROFO-SW-Fixed` returned HTTP 401 on 2026-10-13.
  **Verdict:** excluded.

### 4. An MLX backend

A runtime choice, not a model choice; it could apply to any option above.

- **Pros:** possibly faster.
- **Cons:**
  - a second ML runtime next to torch;
  - [`mlx-audio-separator`](https://pypi.org/project/mlx-audio-separator/) is "Alpha" and shipped
    21 releases from 2026-02-27 to 2026-10-02 (now 0.1.20, `mlx>=0.32.3,<0.33`); djbooth pinned
    0.1.5 and needed the patch under option 1;
  - checkpoint conversion still needs torch (its `convert` extra);
  - our MPS→CPU retry (`separate.py:339-358`) is torch-specific and would need an MLX twin.
    (The dispatcher's heal ledger is backend-agnostic and would still apply.)
- **Verdict:** revisit only if Stage 0 measures Kim on torch MPS above ~1 s/s.

| | vocals | drums / bass | other | s per audio-s | new weights | dep risk |
|---|---|---|---|---|---|---|
| 0 status quo | 8.38 | 11.13 / 11.96 | 5.85 | 1.06 (measured) | – | – |
| 1 djbooth 4-stem | ? (MUSDB 10.70) | ? (MUSDB bass 7.74) | ? | unmeasured | 3.76 GB | high (MLX) / medium (torch) |
| **2 hybrid** | **10.98** | 11.13 / 11.96 | ≥5.85 | 0.65–0.8 (est.) | 913 MB | medium |
| 3 SCNet XL IHF | 9.68 | 11.58 / 11.94 | 6.48 | unmeasured | 214 MB | medium |
| 4 MLX runtime | – | – | – | maybe lower | – | high |

## Recommendation: option 2, in four stages

**Stage 0: measure (nothing in the repo). Needs your OK for the install and the 913 MB
download.**

1. Scratch venv and weights:
   ```bash
   S=<scratchpad>/sep-ab
   /Users/arya/git/dance/.venv/bin/python -m venv $S/venv   # same base 3.13.3 interpreter
   $S/venv/bin/pip install "torch==2.9.1" "torchaudio==2.9.1" "demucs==4.0.1" \
       "msst[mel_band_roformer]==0.1.0" soundfile librosa
   curl -L --create-dirs -o $S/w/MelBandRoformer.ckpt https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt
   curl -L --create-dirs -o $S/w/kj.yaml https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml
   ```
2. A throwaway `$S/ab.py` using `msst.Separator(model_type="mel_band_roformer", ...)` and
   demucs `apply_model` on each of `get_model("htdemucs_ft").models[:3]`. For each of 5 tracks
   it writes:
   - **A:** a *fresh* `htdemucs_ft` run at s2/PCM_24 (older library stems may be s0/PCM_16, so
     they are not a fair baseline);
   - **B-2a** and **B-2b**, as defined under option 2.

   Outputs go to `$S/out/<track>/{X,Y,Z}/` with a hidden shuffle key. Each run is wrapped in
   `/usr/bin/time -l` for peak RSS; it logs s per audio-second, the device actually used, and
   `reconstruction_residual_db` (copied from `separate.py:140`; importing `dance` would pull
   its whole dependency set into the scratch venv).
3. **Gate** (all must hold): B ≤ 1.06 s/s on MPS; peak RSS ≤ 6 GB; no MPS error on any track.
   If speed fails, re-time Kim with `inference.num_overlap: 1` (halves its pass) and listen for
   seams; only then consider option 4.

**Stage 1: implement behind a setting, off by default.**
- New keys: `separation_vocal_model` (`None` = today's behaviour), `separation_vocal_config`,
  `separation_other` (`"model"` | `"residual"`).
- Hybrid path in `separate.py`, with a CPU-retry twin for the RoFormer mirroring
  `_cpu_fallback_model`.
- Provenance-aware skip condition (`separate.py:369-375`; decision 5).
- Unit tests with fakes in a new `tests/test_separate_hybrid.py`.
- Real e2e the way `994f886` did it: temp DB and stems dir, one real track, then open in Live.

**Stage 2: A/B in Live** (protocol below). Gate: B wins or ties on ≥4 of 5 tracks for
listening tests 1–3, and has *no* loss on test 4.

**Stage 3: flip the default; migrate forward-only.** New tracks get the hybrid; existing tracks
are re-separated in explicit batches (next section).

## Migration (forward-only)

- **Trigger.** Today `separate` skips any track with four stem rows whose files exist
  (`separate.py:369-375`), so `dance reprocess --from separate` does **not** re-run separation
  (`cli.py:454-455`, `pipeline/reprocess.py:5-8`).
  - Proposed: skip only when every row's `model_used` equals the configured provenance.
    `dance process` never rewinds a complete track, so nothing re-separates on its own.
  - Alternative: a `--reseparate` flag on `reprocess` (decision 5).
- **Batching.** Use explicit ids; `--limit` always takes the lowest ids first
  (`reprocess.py:103`), so repeating `--all --limit N` re-selects the same tracks. List
  candidates read-only:
  ```bash
  sqlite3 'file:/Users/arya/.dance/dance.db?immutable=1' \
    "SELECT track_id FROM stem_files WHERE kind='vocals' AND model_used NOT LIKE 'melrof%' ORDER BY track_id"
  dance reprocess --from separate -t 12 -t 15 ...   # ~20 per overnight batch; never --skip-embeddings
  dance build-graph
  ```
- **Cascade.** The rewind re-runs analyze_stems → detect_regions → embed (per-stem CLAP), then
  `build-graph`. `duplicate_of` tracks are skipped unless `--include-duplicates`
  (`reprocess.py:120`). Until a migration finishes, recs compare old and new stems.
- **Unchanged:**
  - **`.als`:** stem paths stay `stems_dir/<hash8>/<kind>.wav` (`separate.py:431`); no
    regeneration.
  - **Waveform sidecars:** keyed on size + `st_mtime_ns`, so they self-invalidate
    (`audio/waveform.py:8-13`).
- **Cleanup.** Close any open Live set first, and delete each re-separated stem's `.asd`: the
  `.asd` records `OriginalFileSize` (`docs/proposals/warp-guard.md:105`), and a same-length
  PCM_24 rewrite has the same size, so Live may keep using stale analysis (inferred, not tested).
- **Disk:** rewritten in place at the same size; ≈0 net change. No library-wide backup;
  rollback means re-separating with the old setting (≈20–29 h).
- **Time:** 278 × 4–6 min × 0.65–0.8 s/s ≈ **12–22 h** of separation, plus downstream stages.
  - Cheap path (2b only): keep the on-disk D/B, compute a new V, set `O = mix − D − B − V`.
    Only the Kim pass runs: ≈ **5–11 h**, and drums/bass stay byte-identical. Caveat: here D/B
    were estimated on the full mix (and may be s0/PCM_16), so this is a fourth arm, not the
    same output as option 2; A/B it before choosing it.

## A/B protocol (5 real tracks)

**Tracks:** 2 full-vocal house tracks; 1 with chopped vocal samples; 1 instrumental (false
positives); 1 with a lead synth in the vocal range (leakage).

**Setup.** Blank Live 12.4 set; arms X, Y, Z as groups. Solo → Cue mode, Cue Out on Scarlett
outs 3/4 to the headphones. Score blind; open the shuffle key only after scoring.

**Listening tests:**
1. **Vocals solo:** hats or synth leaking in, watery artifacts.
2. **Mix with vocals muted:** ghost vocals left in `other` or `drums`.
3. **Acapella swap:** this track's vocals over another track's D+B+O at matched tempo — the
   actual move.
4. **Drums solo and bass solo:** transient smear, sub stability. Any regression vetoes B.
5. **Sum of 4 vs the mix:** must be indistinguishable.

**Objective proxies** (none is ground truth; logged per arm):
- `reconstruction_residual_db`.
- **Vocal bleed:** run *both* Kim and the htdemucs vocals specialist over each arm's `other`
  and `drums`; report extracted vocal energy in dB relative to the stem. Two probes because Kim
  alone favours B by construction.
- **Vocals-stem floor:** 20th-percentile 100 ms RMS in dBFS, plus whole-track vocals RMS on the
  instrumental track.
- **Downstream flips:** whether `presence_ratio`, `vocal_present` (>0.15) or detect_regions'
  0.3 threshold flips between arms, using analyze_stems' code, writing nothing to the DB.
- **Cost:** s per audio-second and peak RSS.

## Decisions for you

1. **Stage 0:** may I create a scratch venv (`msst==0.1.0` + deps) and download Kim's 913 MB
   checkpoint? Add SCNet XL IHF (214 MB) as a third arm?
2. **Shifts on the htdemucs half:** s0 (recommended: deterministic, pays for the RoFormer
   pass) or s2 (3 more passes, ≈ +0.4 s/s).
3. **`other`:** model output (2a) or residual (2b). Recommend: decide by the A/B.
4. **Packaging:** optional `dance[roformer]` extra pinning `msst[mel_band_roformer]==0.1.0`
   (recommended), vendor the MIT module, or `audio-separator==0.45.0`. Avoid ≥0.46 (torch ≥2.13
   on Apple Silicon).
5. **Re-separation trigger:** provenance-aware skip (recommended) or `reprocess --reseparate`.
6. **Migration scope:** whole library (12–22 h), vocal-bearing tracks only, or the 2b
   vocals-only shortcut (5–11 h, different output — needs its own A/B arm).
7. **Vocal model:** Kim (MIT tag; recommended) or viperx BS-RoFormer (no stated licence, 0.11 dB
   lower). MSST's BS PolarFormer scores 11.00 (+0.02 dB) but has no `msst` extra; not worth the
   extra integration.

## Not verified here

- No RoFormer, SCNet or MLX code was run on this machine (no installs or downloads allowed).
  Every RoFormer timing and memory figure is an estimate or a third-party number; Stage 0
  replaces them.
- Library size and average track length are from the brief / inferred from bytes on disk.
- The htdemucs_ft rate is from the real MPS run in `994f886` (one 30 s excerpt).
- Isolating the separator in a worker process (branch
  `claude/proposal-separator-worker-process`) is orthogonal; if it lands first, the hybrid runs
  inside it.
