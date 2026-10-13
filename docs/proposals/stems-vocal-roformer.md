# Stem quality: put a vocal specialist in front of htdemucs_ft

**Status:** proposal, not implemented. Needs an explicit "go ahead" (CLAUDE.md rule 1).
**Date:** 2026-10-13 (fact-checked the same day; sources re-fetched, code refs checked against `main`)
**Touches:** the internals of `pipeline/stages/separate.py`, new `Settings` fields, one optional
dependency. It does **not** touch `StemKind`, the `stem_files` schema, the `.als` writer, OSC, or
API shapes.
**Builds on:** `claude/stems-hygiene` (provenance strings, residual QC, atomic writes, sidecar
invalidation) and `claude/analysis-fixes` (`dance reprocess`). Neither is merged yet.

---

## What we actually have

| | |
|---|---|
| Configured | `htdemucs_ft`, `shifts=2`, `PCM_24` ([`config.py:83-91`](../../src/dance/config.py), [`separate.py:119-129`](../../src/dance/pipeline/stages/separate.py)) |
| **On disk** | all 1,112 stems (278 tracks × 4) have `model_used='htdemucs_ft'` and were created 2026-05-17 → 05-26, before `f14e99d` (2026-06-14) wired in `shifts` and `bit_depth`. Before that commit the code passed no `shifts`, so Demucs used `apply_model`'s default **shifts=1** (one randomly offset pass, no averaging, same cost as 0), and wrote `PCM_16`. The configured shifts=2 / PCM_24 has never run on the library. |
| Size | 21.32 h of audio × 4 stems × 176.4 kB/s = **54 GB** (50 GiB) at PCM_16, consistent with the ~52 GB reported on disk. At PCM_24 it is **81 GB**. |
| Free disk | **17 GiB** of 460 GiB on the data volume (97 % used, `df` on 2026-10-13). Rewriting the library at PCM_24 does not fit. |
| QC | `separation_quality` is NULL on every row. `stems-hygiene` starts writing it. |

The DB-derived counts in this document (1,112 stems, creation dates, 221/33 vocal stems below,
23,739 regions, 252 orphan rows) come from a read-only snapshot taken while drafting; the
fact-check did not re-query `~/.dance`.

Separation quality reaches the product in two ways:

- **What you hear.** With the vocal fader down, ghost vocals stay in OTHER; a solo vocal carries
  hats and synth stabs. Both break the main stem move: one track's acapella over another's
  instrumental.
- **What the analysis believes.** `vocal_present` means vocal presence > 0.15
  ([`analyze_stems.py:133`](../../src/dance/pipeline/stages/analyze_stems.py)). Per-stem LOOP
  regions need presence > 0.3 ([`detect_regions.py:59`](../../src/dance/pipeline/stages/detect_regions.py)).
  Combo re-scoring uses per-stem CLAP embeddings
  ([`journey.py:232`](../../src/dance/recommender/journey.py)). Today 221 of 278 vocal stems claim
  vocals; 33 sit in the 0.05–0.15 band, where bleed and real vocal chops look the same.

## What the benchmarks say, and what they don't

Only numbers from one evaluation harness are like-for-like. The table uses ZFTurbo's MSST tables
([pretrained_models.md](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)):

| Multisong SDR (dB) | drums | bass | other | vocals | avg |
|---|---|---|---|---|---|
| htdemucs_ft (one specialist per stem; what we run) | 11.13 | 11.96 | 5.85 | 8.38 | 9.33¹ |
| SCNet XL IHF (one 4-stem model) | 11.58 | 11.94 | 6.48 | 9.68 | 9.92 |
| Mel-RoFormer, Kim (vocals only) | | | | **10.98** | |
| BS-RoFormer, viperx (vocals only) | | | | 10.87 | |

¹ Mean of the four rows; MSST gives no average for the fine-tuned bag. **Caveat:** the
vocal-model table says "Metrics measured on Multisong Dataset", but the single-stem table holding
all four htdemucs_ft rows names no dataset at all. They sit just above plain htdemucs's Multisong
row (10.88 / 11.76 / 5.74 / 8.24), so I read them as Multisong, but that is an inference. The only
vocal model in MSST's table above Kim is BS PolarFormer at 11.00.

djbooth uses **MelBandRoformer4StemFTLarge**. Its [model card](https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge)
(Apache-2.0) reports musdb18 test drums 10.43, bass **7.74**, other 7.10, vocals 10.70, avg 8.99.
The card labels the metric only "sdr", with a per-track std, so it is probably a mean of per-track
SDRs, but it does not say. It was fine-tuned from MSST's `model_mel_band_roformer_ep_5_sdr_8.9443`
(release v1.0.11), so the closest comparison is MSST's own MUSDB-test column:

- SCNet XL IHF: 11.81 / 9.23 / 7.88 / 11.42 = 10.08.
- BS-RoFormer 4-stem: 11.61 / 8.48 / 7.44 / 11.08 = 9.65.

Both beat FT Large on every stem. The Demucs README's "9.0" for htdemucs_ft uses a different
metric: SiSEC18's median over songs of per-second median SDR, giving 10.08 / 10.39 / 6.32 / 9.20
([HT Demucs, Table 1](https://arxiv.org/abs/2211.08553)). It cannot be compared with the card.

Three conclusions:

1. **djbooth's model is not a demonstrated upgrade.** Among the 4-stem numbers we can compare it
   is the weakest, and its bass is the worst, which matters because bass swaps are a core stem move.
2. **The large gap is vocals.** A vocal specialist scores **+2.6 dB** over htdemucs_ft's vocals
   (subject to footnote 1). htdemucs_ft's bass already matches anything listed.
3. **Vocals first, then Demucs on the rest is established practice.** MVSEP-MDX23 estimates
   vocals, sets `instrum = mixed_sound_array - vocals`, runs an ensemble of four Demucs models on
   `instrum`, and finally sets `other = mix − vocals − bass − drums`, so its four stems sum to the
   mix ([inference.py](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model/blob/main/inference.py),
   lines ~407 and ~443–464). On Multisong, as **bass / drums / other / vocals**, it scores
   12.50 / 11.69 / 6.54 / 9.51 against Demucs HT4's 12.10 / 11.30 / 5.77 / 8.36
   ([README](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model)). Those HT4 numbers
   differ from MSST's, so this is a different Multisong snapshot. The gain is also confounded by
   ensembling on both halves, but it points the same way.

## What it costs on this Mac (measured)

**Method.** No weights were available offline, so I built random-weight replicas and timed them on
the M2 Pro 16 GB (16 GPU cores; torch 2.9.1, MPS, fp32, `PYTORCH_ENABLE_MPS_FALLBACK` unset, so
every op ran natively). The replica follows lucidrains' Mel-Band-RoFormer, which MSST adapts. These
timings were not re-run during the fact-check; the arithmetic below was re-derived from them.

**Fidelity checks:**

- The Kim replica has **228.2 M** parameters. Published: ~228 M
  ([CoreAI port card](https://huggingface.co/mlboydaisuke/MelBandRoformer-Vocal-CoreAI)); the
  913 MB fp32 checkpoint implies the same.
- The FT Large replica has 840.5 M parameters against ~940 M implied by its 3.76 GB checkpoint
  if that is pure fp32 weights. Treat its timings as ±15 %.
- The `HTDemucs(bottom_channels=512)` replica has 41.5 M parameters. The four cached checkpoints
  are 84.1 MB each (fp16), which is consistent.
- **Calibration.** The method predicts 30.2 s for htdemucs_ft shifts=2 on a 30 s excerpt.
  `stems-hygiene` (commit `994f886`) measured **31.7 s** with real weights on this machine.

| per forward (best of 3, two runs) | chunk | forward | MPS memory² |
|---|---|---|---|
| Kim Mel-RoFormer vocals | 8.0 s | 2.47–2.54 s | 6.8 GB |
| FT Large 4-stem (`skip_connection`, hop 882) | 11.0 s | 2.88–3.02 s | 7.6 GB |
| one htdemucs_ft specialist | 7.8 s | 0.63 s | 3.1 GB |

² `torch.mps.driver_allocated_memory()` after the forward pass.

Per average track (276 s) and for the whole library (21.3 h), one GPU job at a time
(`DANCE_GPU_CONCURRENCY` defaults to 1, [`_gpu.py:20`](../../src/dance/pipeline/_gpu.py)):

| pipeline | per track | library |
|---|---|---|
| htdemucs_ft shifts=1 (what made the library) | ~121 s | ~9.3 h |
| htdemucs_ft shifts=2 (configured default) | ~242 s | ~18.7 h |
| Kim vocals (num_overlap 2) + 3 htdemucs_ft specialists at shifts=1 | **~266 s** | **~20.5 h** |
| FT Large, num_overlap 4 (its config) / 2 | ~298 s / ~150 s | ~23 h / ~11.6 h |
| SCNet XL IHF | not measured | |

In a separate run, fp16 autocast on MPS gave 3.07 s vs 3.27 s per Kim chunk: within noise, not
worth the numerical risk. The figures exclude the downstream stages that must re-run
(analyze_stems, detect_regions, embed). Stage 1 measures those.

## Constraint every option must meet

Exactly four stems: drums, bass, vocals, other, at 44.1 kHz. That shape is `StemKind`
([`database.py:86-92`](../../src/dance/core/database.py)), `SOURCE_STEM_KINDS`
([`writer.py:97`](../../src/dance/als/writer.py)), `_SOURCE_STEM_KINDS`
([`bridge.py:882`](../../src/dance/osc/bridge.py)) and the 8-fader layout. 6-stem models
(htdemucs_6s, guitar/piano) are out. All the candidates below fit.

## Options

### 0 — Status quo

- **Quality:** unchanged. Re-running the library at the configured shifts=2 buys little: the
  Demucs README says shifts makes prediction "SHIFTS times slower", and our config comment calls
  the gain "modest" and small on lossy sources ([`config.py:85-88`](../../src/dance/config.py)).
  That is 18.7 h for a small gain.
- **Cost:** zero.
- **Risk:** `facebookresearch/demucs` was archived on 2025-01-01. Maintenance moved to the
  author's fork, [`adefossez/demucs`](https://github.com/adefossez/demucs), which released 4.1.0 on
  2026-07-11 but says to "expect slow replies and no new feature". We run 4.0.1, which works on
  torch 2.9.1 today. Low but non-zero bit-rot risk.

### 1 — djbooth's 4-stem RoFormer (MelBandRoformer4StemFTLarge)

- **Quality:** see above. No evidence it beats htdemucs_ft, and its bass is the weakest of the
  comparable 4-stem models.
- **Time and memory:** ~298 s per track (1.2× today's default), 7.6 GB on MPS.
- **Weights:** 3.76 GB, Apache-2.0. A second 3.76 GB `ver2.ckpt` scores slightly lower on SDR
  (avg 8.87); its author claims better perceived quality and adds "fullness"/"bleedless" scores.
- **Packaging:** the config sets `skip_connection: True`. djbooth's MLX path needed a monkeypatch
  that restores ZFTurbo's skip-connected transformer schedule
  (`python/stems-mlx/djbooth_stems_mlx.py:128-172`), because the MLX package "omits that config
  flag and adds an untrained terminal normalization" (`packages/rekordbox-ts/PORTING.md:343-346`).
  Without it the MLX model is not the architecture the checkpoint was trained as. The PyTorch
  reference (MSST) supports the flag natively.
- **Verdict:** **reject** as the default. Keep as optional A/B candidate E only because djbooth
  ships it.

### 2 — Hybrid: vocal specialist, then htdemucs_ft on (mix − vocals) ← **recommended**

**How it works:**

1. Run Kim Mel-RoFormer on the mix to get VOCALS.
2. Compute `inst = mix − vocals`.
3. Run htdemucs_ft's drums, bass and other specialists on `inst`; skip its vocals specialist. In
   the bag each sub-model only contributes its own stem; MSST notes the other three outputs of each
   are "dummy". mlx-audio-separator documents the same per-specialist use
   (`-m htdemucs_ft.yaml --single_stem bass`).
4. OTHER is the specialist's output, or the residual `mix − vocals − drums − bass` (MDX23's final
   step). This is decision 2.

**Assessment:**

- **Quality:** vocals +2.6 dB on Multisong (footnote 1). Drums and bass should be at least
  htdemucs_ft's, which is already the best bass listed, but the specialists were trained on full
  mixes, not on `mix − vocals`. MDX23 does the same thing successfully, but stage 1 has to confirm
  it by ear. OTHER should lose most vocal bleed, which is the audible defect.
- **Time:** ~266 s per track, **+10 % over the configured default**, with the Demucs half at
  shifts=1 (one pass). At shifts=2 it would be ~355 s. It is 2.2× what made the library.
- **Memory:** peak 6.8 GB on MPS, one model at a time.
- **Weights:** Kim is 913 MB, MIT ([HF card](https://huggingface.co/KimberleyJSN/melbandroformer)).
  htdemucs_ft is already cached (4 × 84 MB; Demucs code is MIT, and the README states no separate
  weights licence). viperx BS-RoFormer is the alternative specialist, but its weights ship from
  `TRvlvr/model_repo`, which has no licence, so treat them as unlicensed.
- **Layout:** VOCALS comes from Kim; DRUMS, BASS and OTHER come from Demucs. `model_used` is per
  row, so each stem records its own model without a schema change.
- **Risk:** a second model family and an optional dependency (see Packaging). One more
  full-track pass, roughly paid for by not running shifts=2.

### 3 — A single stronger 4-stem model (SCNet XL IHF)

- **Quality:** on Multisong, +1.30 vocals, +0.45 drums, +0.63 other, −0.02 bass against
  htdemucs_ft. That is half option 2's vocal gain, with the same bass.
- **Time:** not measured. Base SCNet takes 48 % of HT Demucs' CPU inference time per its
  [paper](https://arxiv.org/abs/2401.13276); XL is larger. Its config
  (`config_musdb18_scnet_xl_more_wide_v5.yaml`, MSST release v1.0.15) uses 11 s chunks with
  num_overlap 4, so expect several full-track-equivalents of compute.
- **Weights and packaging:** 214 MB checkpoint in the MSST GitHub release v1.0.15. The repo is
  MIT; no separate weights licence is stated. MSST's SCNet code imports only torch.
- **Verdict:** **A/B candidate D**. It is simpler than option 2 (one model, one pass) if it is
  close on vocals and faster.

### 4 — An MLX backend (mlx-audio-separator)

- **Speed:**
  - djbooth measured 10.88 s of audio in 1.49 s (model load + inference + write) with 3.96 GB
    max RSS, on an M2 Max (30–38 GPU cores vs this machine's 16) and one short clip
    (`PORTING.md:350-352`).
  - The package claims 1.40× over audio-separator for htdemucs_ft and up to 2.50× for RoFormers
    (median 1.85×) on an M4 mini ([README](https://github.com/ssmall256/mlx-audio-separator)).
- **Packaging:** v0.1.20 (2026-10-02), MIT, Python ≥3.10 (classifiers to 3.13),
  `mlx>=0.32.3,<0.33`. 17 stars, two contributors, 21 PyPI releases since 2026-02-27.
- **Correctness:** its parity smoke test against audio-separator passes at rel-L2 ≤ 5e-2, which
  is loose. djbooth needed the patch above, and pins the older 0.1.5 with Python `>=3.12,<3.13`
  (`python/stems-mlx/pyproject.toml:5-6`).
- **Memory:** a second ML runtime beside torch, CLAP and Live on 16 GB.
- **Verdict:** **defer.** It is a backend choice, not a quality choice. Revisit only if stage 1
  shows the library re-run is unacceptable (more than 2 nights), and then only after a
  checkpoint-exact parity test against PyTorch. It belongs with the separator-worker-process
  proposal (`claude/proposal-separator-worker-process`).

## Packaging for options 2 and 3 (PyPI metadata, 2026-10-13)

| package | fits py3.13 + torch 2.9.1? | notes |
|---|---|---|
| [`msst`](https://pypi.org/project/msst/) 0.1.0 (2026-09-09) | yes: `<3.14`, `torch<2.12,>=2.0.1` | ZFTurbo's reference code, published by him. The `[mel-band-roformer]` extra adds beartype, einops (already installed), packaging, rotary-embedding-torch and PoPE-pytorch. Core also pulls matplotlib, omegaconf and ml-collections. **Only release so far.** |
| [`audio-separator`](https://pypi.org/project/audio-separator/) | **only ≤0.45.0** | 0.46.0+ (2026-08-27) requires `torch>=2.13` on arm64 macOS, which would force a torch upgrade under CLAP and Demucs. Heavy tree (onnx-weekly, onnx2torch-py313, diffq, `samplerate==0.1.0`, `beartype<0.19`). `normalization_threshold` defaults to 0.9: any stem whose peak exceeds 0.9 is scaled down on its own, which breaks the stem sum on loud masters unless set to 1.0. |
| vendor `mel_band_roformer.py` (MIT) | yes | About 300 lines plus rotary-embedding-torch and beartype. We would own the ~50-line chunked overlap-add, and drift from upstream fixes. |

## Migration: forward-only, opt-in, resumable

- **Nothing re-separates today, by design.** `separate.py:88-94` skips any track that already
  has 4 stems on disk, so `dance reprocess --from separate` (from `analysis-fixes`) would not
  re-run Demucs at all.
  - Re-separation needs a provenance check: compare `model_used` with the configured pipeline.
  - That check must only fire when asked (e.g. `--reseparate`). Otherwise a routine
    `reprocess --from analyze` silently becomes a 20-hour job.
- **Cascade.** `separate.py:131-135` deletes the old `StemFile` rows; the ORM cascade then
  deletes the per-stem `AudioAnalysis`, `Region` and `TrackEmbedding` rows (`database.py:283-291`),
  and the track re-runs analyze_stems → detect_regions → embed.
  - Today that loses nothing authored: all 23,739 regions are `auto`.
  - Graph edges use full-mix embeddings only
    ([`graph_builder.py:74-82`](../../src/dance/recommender/graph_builder.py)), so `build-graph`
    is not needed.
  - `set_tracks.stem_kinds` stores kind strings, so sets are unaffected.
- **Paths and `.als` files.** Paths stay `stems_dir/<hash8>/<kind>.wav`, so `.als` references
  survive.
  - `writer.py:568` records `OriginalFileSize`, so run `dance export-als --all` afterwards.
  - Delete each stem's sibling `.asd` so Live re-analyses the new audio. No warp repair is
    persisted yet ([`warp-guard.md`](warp-guard.md)), so nothing is lost.
- **Waveform sidecars.** On `main` the cache is validated only by version and `num_peaks`
  ([`waveform.py:82-102`](../../src/dance/audio/waveform.py)), so an in-place rewrite keeps
  serving the old envelope. `stems-hygiene` keys it on size and mtime, one reason it lands first.
- **Provenance** is per row, ≤64 chars, in `stems-hygiene`'s format
  (`htdemucs_ft|s2|pcm24|demucs4.0.1`), e.g. `melrof-kim|ov2|pcm16|msst0.1.0` and
  `htdemucs_ft:bass@inst|s1|pcm16|demucs4.0.1`. The old bare `htdemucs_ft` rows stay valid and
  are re-separated only on request.
- **Pre-existing debris.** 252 per-stem `audio_analysis` rows (63 tracks) point at `stem_files`
  that no longer exist. Sweep them first so post-migration counts check out exactly.
- **Cost:**
  - ~20.5 GPU-hours for 278 tracks, plus downstream stages. Run in `--limit 40` batches
    overnight; each track commits independently.
  - Never run it during a session: `stem_file_id`s change under the live recs.
  - Disk: PCM_16 is +0 GB. PCM_24 is +27 GB and needs space freed first (decision 3).

## Recommendation (staged)

0. **Land `stems-hygiene` and `analysis-fixes`.** Without them a re-separated stem cannot be told
   apart from an old one, and its waveform is stale.
1. **A/B before any code** (protocol below). Scratch venv, no repo changes, nothing written under
   `~/Music/DJ` or `~/.dance`.
2. **If the hybrid wins**, add `separation_backend: "demucs" | "vocal_first"` to `Settings`, with
   `demucs` still the default.
   - Add the provenance-aware skip behind an explicit flag.
   - Tests in a new `tests/test_separate_vocal_first.py` with fake models.
   - Verify on one real track in Live (rule 2): 5 tracks load, audio plays, tempo is correct.
3. **Flip the default for new tracks**, then re-separate the library in batches, tracks in sets
   first.

The A/B decides between options 2 and 3. Options 1 and 4 are not pursued.

## Decisions for the user

1. **Approve stage 1**, which downloads Kim's weights (913 MB, MIT). Optional extra candidates:
   SCNet XL IHF (214 MB, MSST release) and FT Large (3.76 GB, Apache-2.0).
2. **How OTHER is built in the hybrid.**
   - The model's OTHER: cleaner, but all four faders up ≠ the record. For reference, htdemucs_ft
     measured −20.7 dB residual on the one 30 s excerpt `stems-hygiene` tested.
   - The residual `mix − V − D − B` (MDX23's choice): exact reconstruction, but every artifact
     lands in OTHER.
   - Decide by ear in stage 1.
3. **Bit depth for re-separated stems:** PCM_16 fits (54 GB). PCM_24 needs ~27 GB more than is
   free today (17 GiB).
4. **Packaging:** pin `msst==0.1.0` (first and only release), or vendor the MIT model file.
5. **Scope:** new tracks only, tracks in sets, or the whole library (~20.5 h).

## A/B protocol

**Tracks.** Pick 5: 2 vocal-led, 1 instrumental, 1 ambiguous, 1 bass-led. This read-only query
covers the vocal axis; choose the bass-led one by ear.

```bash
sqlite3 'file:/Users/arya/.dance/dance.db?immutable=1' "
  select t.id, round(a.presence_ratio,2) vox, t.artist||' - '||t.title
  from tracks t join stem_files s on s.track_id=t.id and s.kind='vocals'
  join audio_analysis a on a.stem_file_id=s.id
  where t.duplicate_of is null order by vox;"   # take from <0.05, 0.05-0.15 and >=0.5
```

**Candidates.** Write each one's four WAVs to `$TMPDIR/dance-ab/<track>/<letter>/`, letters
shuffled and the key sealed until scoring is done:

- A: the existing library stems, copied, never touched in place.
- B: htdemucs_ft at shifts=2, PCM_24.
- C: the hybrid, OTHER from the model.
- C′: the hybrid, OTHER as the residual.
- D: SCNet XL IHF (optional).
- E: FT Large (optional).

Scratch environment (`huggingface_hub` provides the `hf` CLI; nothing else here installs it):

```bash
python3.13 -m venv "$TMPDIR/dance-ab/venv"
"$TMPDIR/dance-ab/venv/bin/pip" install torch==2.9.1 demucs==4.0.1 "msst[mel-band-roformer]==0.1.0" huggingface_hub soundfile librosa
"$TMPDIR/dance-ab/venv/bin/hf" download KimberleyJSN/melbandroformer MelBandRoformer.ckpt --local-dir "$TMPDIR/dance-ab/weights"
```

Use Kim's config, `configs/config_vocals_mel_band_roformer.yaml` from
[KimberleyJensen/Mel-Band-Roformer-Vocal-Model](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)
(chunk 352800, num_overlap 2). Record wall time per track and per stage, plus the
analyze_stems / detect_regions / embed time on a temp DB.

**Objective proxies** (no ground truth, so these are screens, not verdicts):

- **Reconstruction residual:** `20·log10(rms(mix − Σstems)/rms(mix))`, the `stems-hygiene`
  definition. Compute it before any residual folding, because C′ is −∞ by construction.
- **Bleed into vocals:** RMS of VOCALS relative to the mix on the instrumental track, and in
  sections every candidate marks vocal-free. Lower is better.
- **Ghost vocals in OTHER:** correlate the 300 Hz–4 kHz envelopes of VOCALS and OTHER over
  vocal-present frames. Lower is better. Model-free on purpose: a RoFormer as judge would favour C.

**Listening** in Live, on the Scarlett's cue bus (outs 3/4 → headphones). Score 1–5 per check,
blind:

1. Solo VOCALS: instrument bleed, "underwater" phasing.
2. Mute VOCALS (D+B+O): ghost vocals. This is the check that matters.
3. Solo BASS: kick bleed.
4. Solo DRUMS: sub bleed.
5. All four up against the original mix on the master: does it sound like the record?

**Pass for C (or D):**

- Wins checks 1 and 2 on at least 4 of the 5 tracks.
- No worse on checks 3–4.
- Ghost-vocal correlation drops clearly.
- Costs ≤1.2× B per track.

Otherwise stop at option 0 and record the result here.
