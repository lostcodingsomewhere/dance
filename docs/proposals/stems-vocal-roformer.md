# Stem quality: put a vocal specialist in front of htdemucs_ft

**Status:** proposal, not implemented. Needs an explicit "go ahead" (CLAUDE.md rule 1).
**Date:** 2026-10-13
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
| **On disk** | all 1,112 stems (278 tracks × 4) have `model_used='htdemucs_ft'` and were created between 2026-05-17 and 05-26. That is before `f14e99d` (2026-06-14) turned on shifts=2 and PCM_24. So the library is **shifts=0, PCM_16**, and the configured default has never run on it. |
| Size | 21.32 h of audio × 4 stems × 176.4 kB/s = **54 GB** at PCM_16. That fits the ~52 GB on disk. The same library at PCM_24 is **81 GB**. |
| Free disk | **19 GiB** of 460 GiB (96 % used). Rewriting the library at PCM_24 does not fit. |
| QC | `separation_quality` is NULL on every row. `stems-hygiene` starts writing it. |

Separation quality reaches the product in two ways:

- **What you hear.** With the vocal fader down, ghost vocals stay in OTHER. A solo vocal carries
  hats and synth stabs. Both break the main stem move: one track's acapella over another track's
  instrumental.
- **What the analysis believes.** `vocal_present` means vocal presence > 0.15
  ([`analyze_stems.py:133`](../../src/dance/pipeline/stages/analyze_stems.py)). Per-stem LOOP
  regions need presence > 0.3 ([`detect_regions.py:59`](../../src/dance/pipeline/stages/detect_regions.py)).
  Column recs score per-stem CLAP embeddings. Today 221 of 278 vocal stems claim vocals. 33 sit in
  the ambiguous 0.05–0.15 band, where bleed and real vocal chops look the same.

## What the benchmarks say, and what they don't

The only comparison that is like-for-like is within one evaluation harness. The table uses ZFTurbo's
MSST tables on MVSEP's Multisong set ([pretrained_models.md](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)):

| Multisong SDR (dB) | drums | bass | other | vocals | avg |
|---|---|---|---|---|---|
| htdemucs_ft (one specialist per stem; what we run) | 11.13 | 11.96 | 5.85 | 8.38 | 9.33¹ |
| SCNet XL IHF (one 4-stem model) | 11.58 | 11.94 | 6.48 | 9.68 | 9.92 |
| Mel-RoFormer, Kim (vocals only) | | | | **10.98** | |
| BS-RoFormer, viperx (vocals only) | | | | 10.87 | |

¹ This is the mean of the four rows. MSST lists no 4-stem average for the fine-tuned bag. Its
single-stem table does not name a dataset for the drums, bass and other rows. They line up with
plain htdemucs's Multisong row (10.88 / 11.76 / 5.74 / 8.24), so I read them as Multisong.

djbooth uses **MelBandRoformer4StemFTLarge**. Its [model card](https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge)
reports MUSDB18 test drums 10.43, bass **7.74**, other 7.10, vocals 10.70, avg 8.99. The card does
not state the metric. The model was fine-tuned from an MSST checkpoint, so the closest comparison
is MSST's own MUSDB-test column:

- SCNet XL IHF: 11.81 / 9.23 / 7.88 / 11.42 = 10.08.
- BS-RoFormer 4-stem: 11.61 / 8.48 / 7.44 / 11.08 = 9.65.

Both beat FT Large on every stem. The Demucs README's "9.0 dB" for htdemucs_ft is a different
metric: SiSEC18 median of per-second medians, which gives 10.08 / 10.39 / 6.32 / 9.20
([HT Demucs, Table 1](https://arxiv.org/abs/2211.08553)). That number cannot be set against the
card.

Three conclusions:

1. **djbooth's model is not a demonstrated upgrade.** Of the 4-stem numbers we can compare, it is
   the weakest, and its bass is the worst. Bass is the stem a DJ swaps most.
2. **The large, robust gap is vocals.** On the same harness a vocal specialist scores **+2.6 dB**
   over htdemucs_ft's vocals. htdemucs_ft's bass is already as good as anything listed.
3. **Vocals first, then Demucs on the rest is established practice.** MVSEP-MDX23 estimates
   vocals, sets `instrum = mixed_sound_array - vocals`, then runs the Demucs models on `instrum`
   ([inference.py](https://raw.githubusercontent.com/ZFTurbo/MVSEP-MDX23-music-separation-model/main/inference.py)).
   On Multisong it scores 12.50 / 11.69 / 6.54 / 9.51 against Demucs HT4's 12.10 / 11.30 / 5.77 /
   8.36 ([README](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model)). That gain is
   confounded by its ensembling, but it points the same way.

## What it costs on this Mac (measured)

**Method.** No weights were available offline, so I built random-weight replicas and timed them on
the M2 Pro 16 GB (torch 2.9.1, MPS, fp32, `PYTORCH_ENABLE_MPS_FALLBACK` unset, so every op ran
natively). The replica follows lucidrains' Mel-Band-RoFormer, which MSST uses.

**Fidelity checks:**

- The Kim replica has **228.2 M** parameters. The published figure is ~228 M
  ([CoreAI port card](https://huggingface.co/mlboydaisuke/MelBandRoformer-Vocal-CoreAI)).
- The FT Large replica has 840.5 M parameters against ~940 M implied by its 3.76 GB checkpoint.
  Treat its timings as ±15 %.
- The `HTDemucs(bottom_channels=512)` replica has 41.5 M parameters. The cached fp16 checkpoints
  are 84 MB each, which is consistent.
- **Calibration.** The same method predicts 30.2 s for htdemucs_ft shifts=2 on a 30 s excerpt.
  `stems-hygiene` measured **31.7 s** with real weights on this machine.

| per forward (best of 3, two runs) | chunk | forward | MPS memory² |
|---|---|---|---|
| Kim Mel-RoFormer vocals | 8.0 s | 2.47–2.54 s | 6.8 GB |
| FT Large 4-stem (`skip_connection`, hop 882) | 11.0 s | 2.88–3.02 s | 7.6 GB |
| one htdemucs_ft specialist | 7.8 s | 0.63 s | 3.1 GB |

² `torch.mps.driver_allocated_memory()` after the forward pass.

Per average track (276 s) and for the whole library (21.3 h of audio), with one GPU job at a time
(`DANCE_GPU_CONCURRENCY` defaults to 1, [`_gpu.py:20`](../../src/dance/pipeline/_gpu.py)):

| pipeline | per track | library |
|---|---|---|
| htdemucs_ft shifts=0 (what made the library) | ~121 s | ~9.3 h |
| htdemucs_ft shifts=2 (configured default) | ~242 s | ~18.7 h |
| Kim vocals (num_overlap 2) + 3 htdemucs_ft specialists at shifts=0 | **~266 s** | **~20.5 h** |
| FT Large, num_overlap 4 (its config) / 2 | ~298 s / ~150 s | ~23 h / ~11.6 h |
| SCNet XL IHF | not measured | |

fp16 autocast on MPS buys nothing: 3.07 s vs 3.27 s per Kim chunk. The figures above exclude the
downstream stages that must re-run (analyze_stems, detect_regions, CLAP). Stage 1 measures those.

## Constraint every option must meet

The output is exactly four stems: drums, bass, vocals, other, at 44.1 kHz. That shape is
`StemKind` ([`database.py:86-92`](../../src/dance/core/database.py)), `SOURCE_STEM_KINDS`
([`writer.py:97`](../../src/dance/als/writer.py)), `_SOURCE_STEM_KINDS`
([`bridge.py:882`](../../src/dance/osc/bridge.py)) and the 8-fader layout. 6-stem models
(htdemucs_6s, guitar/piano) are out. All the candidates below fit.

## Options

### 0 — Status quo

- **Quality:** unchanged. Even re-running the library at the configured shifts=2 buys little.
  The Demucs README says shifts makes prediction "SHIFTS times slower", and our own config
  comment calls the gain "modest" and "small" on lossy sources
  ([`config.py:85-88`](../../src/dance/config.py)). That would be 18.7 h for a small gain.
- **Cost:** zero.
- **Risk:** Demucs has been archived since 2025-01-01 and is unmaintained
  ([README](https://github.com/facebookresearch/demucs)). It still works on torch 2.9.1 today.

### 1 — djbooth's 4-stem RoFormer (MelBandRoformer4StemFTLarge)

- **Quality:** see above. No evidence it beats htdemucs_ft, and its bass is the weakest of the
  comparable 4-stem models.
- **Time and memory:** ~298 s per track (1.2× today's default), 7.6 GB on MPS.
- **Weights:** 3.76 GB, Apache-2.0. A second 3.76 GB `ver2.ckpt` scores lower: avg 8.87.
- **Packaging:** it needs `skip_connection: True`. djbooth's MLX path only produced correct
  output after a monkeypatch that restores ZFTurbo's skip-connected transformer schedule
  (`python/stems-mlx/djbooth_stems_mlx.py:128-172`). djbooth's notes say the MLX package omits
  that flag "and adds an untrained terminal normalization" (`packages/rekordbox-ts/PORTING.md:343-346`).
  The PyTorch reference (MSST) supports it natively.
- **Verdict:** **reject** as the default. Worth keeping as an A/B candidate only because djbooth
  ships it.

### 2 — Hybrid: vocal specialist, then htdemucs_ft on (mix − vocals) ← **recommended**

**How it works:**

1. Run Kim Mel-RoFormer on the mix to get VOCALS.
2. Compute `inst = mix − vocals`.
3. Run htdemucs_ft's drums, bass and other specialists on `inst`. Skip its vocals specialist. In
   the bag, each sub-model only contributes its own stem, and mlx-audio-separator's docs show the
   same per-specialist use (`--single_stem bass`).
4. OTHER is either the specialist's output or the residual `mix − vocals − drums − bass`, which
   is MDX23's choice. This is decision 2 below.

**Assessment:**

- **Quality:** vocals +2.6 dB on Multisong. Drums and bass are at least htdemucs_ft's, which is
  already the best bass listed. OTHER loses most vocal bleed, the audible defect.
- **Time:** ~266 s per track, **+10 % over the configured default**. That holds when shifts
  drops to 0 for the Demucs half; shifts=2 would make it ~355 s. It is 2.2× what made the
  library.
- **Memory:** peak 6.8 GB on MPS, one model at a time.
- **Weights:** Kim is 913 MB, MIT ([card](https://huggingface.co/KimberleyJSN/melbandroformer)).
  htdemucs_ft is already cached (4 × 84 MB, MIT code). viperx BS-RoFormer is the alternative
  specialist, but I found no licence for its weights, so treat them as unlicensed.
- **Layout:** VOCALS comes from Kim. DRUMS, BASS and OTHER come from Demucs. `model_used` is
  per row, so each stem records its own model without a schema change.
- **Risk:** it adds a second model family and an optional dependency (see Packaging). There is
  one more full-track pass, but the cost is roughly neutral because shifts drops.

### 3 — A single stronger 4-stem model (SCNet XL IHF)

- **Quality:** on Multisong, +1.30 vocals, +0.45 drums, +0.63 other, −0.02 bass. That is a
  smaller vocal gain than option 2 and the same bass.
- **Time:** not measured. Base SCNet takes 48 % of HT Demucs' CPU time per its
  [paper](https://arxiv.org/abs/2401.13276). XL is bigger, and I could not find its config.
- **Weights and packaging:** weights ship as an MSST GitHub release. The repo is MIT; no
  separate licence for the weights is stated. Runs through MSST.
- **Verdict:** a good **A/B candidate D**. It is simpler than option 2 (one model, one pass) if
  it is close on vocals and faster.

### 4 — An MLX backend (mlx-audio-separator)

- **Speed:**
  - djbooth measured 10.88 s of audio in 1.49 s with 3.96 GB RSS, on an **M2 Max** (2× this
    machine's GPU cores) and a single short clip (`PORTING.md:350-352`).
  - The package itself claims 1.4–2.5× over audio-separator on an M4 mini
    ([README](https://github.com/ssmall256/mlx-audio-separator)).
- **Packaging:** v0.1.20 (2026-10-02), MIT, py3.10–3.13, `mlx<0.33`. It has 17 stars and one
  maintainer, and has published 21 releases (0.1.1 to 0.1.20).
- **Correctness:** its parity gate against upstream is rel-L2 ≤ 5e-2, which is loose. djbooth
  needed a patch for silently wrong output, and pins an older 0.1.5 with Python < 3.13
  (`python/stems-mlx/pyproject.toml:5-6`).
- **Memory:** it puts a second ML runtime beside torch, CLAP and Live on 16 GB.
- **Verdict:** **defer.** It is a backend choice, not a quality choice. Revisit only if stage 1
  shows the library re-run is unacceptable (more than 2 nights). It would then need a
  checkpoint-exact parity test against PyTorch first. It belongs with the parallel
  separator-worker-process proposal (`claude/proposal-separator-worker-process`), not here.

## Packaging for options 2 and 3 (checked against PyPI metadata, 2026-10-13)

| package | fits py3.13 + torch 2.9.1? | notes |
|---|---|---|
| [`msst`](https://pypi.org/project/msst/) 0.1.0 (2026-09-09) | yes: `<3.14`, `torch<2.12,>=2.0.1` | ZFTurbo's reference code, packaged. The `[mel-band-roformer]` extra adds beartype, einops (already installed), rotary-embedding-torch and PoPE-pytorch. Core also pulls matplotlib, omegaconf and ml-collections. This is its **first release**. |
| [`audio-separator`](https://pypi.org/project/audio-separator/) | **only ≤0.45.0** | 0.46.0+ (2026-08-27) requires `torch>=2.13` on arm64 macOS, which would force a torch upgrade under CLAP and Demucs. It has a heavy dependency tree (onnx-weekly, onnx2torch-py313, diffq, `samplerate==0.1.0`, `beartype<0.19`). `normalization_threshold` defaults to 0.9, which peak-normalises output and breaks stem sums unless set to 1.0. |
| vendor `mel_band_roformer.py` (MIT) | yes | About 300 lines plus rotary-embedding-torch and beartype. We would own the ~50-line chunked overlap-add. It drifts from upstream fixes. |

## Migration: forward-only, opt-in, resumable

- **Nothing re-separates today, by design.** `separate.py:88-94` skips any track that already
  has 4 stems on disk. So `dance reprocess --from separate` (from `analysis-fixes`) would not
  re-run Demucs at all.
  - Re-separation needs a provenance check: compare `model_used` with the configured pipeline.
  - That check must only fire when explicitly asked (e.g. `--reseparate`). Otherwise a routine
    `reprocess --from analyze` silently becomes a 20-hour job.
- **Cascade.** `separate.py:131-135` deletes the old `StemFile` rows. The ORM then deletes the
  per-stem `AudioAnalysis`, `Region` and `TrackEmbedding` rows (`database.py:283-291`), and the
  track re-runs analyze_stems → detect_regions → embed.
  - Today that loses nothing authored: all 23,739 regions are `auto`.
  - Graph edges use full-mix embeddings only ([`graph_builder.py:79`](../../src/dance/recommender/graph_builder.py)),
    so `build-graph` is not needed.
  - `set_tracks` stores stem kinds as strings, so it is unaffected.
- **Paths and `.als` files.** Paths stay `stems_dir/<hash8>/<kind>.wav`, so `.als` references
  survive.
  - `writer.py:568` records `OriginalFileSize`, so run `dance export-als --all` afterwards.
  - Delete each stem's sibling `.asd` so Live re-analyses the new audio. No warp repair is
    persisted yet ([`warp-guard.md`](warp-guard.md)), so nothing is lost.
- **Waveform sidecars.** On main the cache is validated only by version and `num_peaks`
  ([`waveform.py:82-102`](../../src/dance/audio/waveform.py)). An in-place rewrite would keep
  serving the old envelope. `stems-hygiene` fixes this by keying on size and mtime, which is one
  reason it lands first.
- **Provenance** is per row, ≤64 chars, for example `melrof-kim|ov2|pcm16|msst0.1.0` and
  `htdemucs_ft:bass@inst|s0|pcm16|demucs4.0.1`. The old bare `htdemucs_ft` rows stay valid and
  are re-separated only on request.
- **Pre-existing debris.** 252 per-stem `audio_analysis` rows (63 tracks) point at `stem_files`
  that no longer exist. Sweep them first so post-migration counts check out exactly.
- **Cost:**
  - ~20.5 GPU-hours for 278 tracks, plus the downstream stages. Run in `--limit 40` batches
    overnight; each track commits independently.
  - Never run it during a session, because `stem_file_id`s change under the live recs.
  - Disk: PCM_16 is +0 GB. PCM_24 is +27 GB and needs space freed first (decision 3).

## Recommendation (staged)

0. **Land `stems-hygiene` and `analysis-fixes`.** Without them a re-separated stem cannot be
   told apart from an old one, and its waveform is stale.
1. **A/B before any code** (protocol below). It runs in a scratch venv, writes no repo changes,
   and stays out of `~/Music/DJ` and `~/.dance`.
2. **If the hybrid wins**, add `separation_backend: "demucs" | "vocal_first"` to `Settings`, with
   `demucs` still the default.
   - Add the provenance-aware skip behind an explicit flag.
   - Put tests in a new `tests/test_separate_vocal_first.py` with fake models.
   - Verify on one real track in Live (rule 2): 5 tracks load, audio plays, tempo is correct.
3. **Flip the default for new tracks**, then re-separate the library in batches, starting with
   tracks in sets.

The A/B decides between options 2 and 3. Options 1 and 4 are not pursued.

## Decisions for the user

1. **Approve stage 1**, which downloads Kim's weights (913 MB, MIT). Two optional candidates:
   SCNet XL IHF (MSST release) and FT Large (3.76 GB).
2. **How OTHER is built in the hybrid.**
   - The model's OTHER: cleaner, but all four faders up ≠ the record. htdemucs_ft is already
     at −20.7 dB residual on the one 30 s excerpt `stems-hygiene` measured.
   - The residual `mix − V − D − B`: exact reconstruction, but every artifact lands in OTHER.
   - Decide by ear in stage 1.
3. **Bit depth for re-separated stems:** PCM_16 fits (54 GB). PCM_24 needs ~27 GB freed first.
4. **Packaging:** pin `msst==0.1.0`, or vendor the MIT model file.
5. **Scope:** new tracks only, tracks in sets, or the whole library (~20.5 h).

## A/B protocol

**Tracks.** Pick 5 with this read-only query: 2 vocal-led, 1 instrumental, 1 ambiguous, 1 bass-led.

```bash
sqlite3 'file:/Users/arya/.dance/dance.db?immutable=1' "
  select t.id, round(a.presence_ratio,2) vox, t.artist||' - '||t.title
  from tracks t join stem_files s on s.track_id=t.id and s.kind='vocals'
  join audio_analysis a on a.stem_file_id=s.id
  where t.duplicate_of is null order by vox;"   # take from <0.05, 0.05-0.15 and >=0.5
```

**Candidates.** Write each one's four WAVs to `$TMPDIR/dance-ab/<track>/<letter>/`, with the
letters shuffled and the key sealed until scoring is done:

- A: the existing library stems, copied and never touched in place.
- B: htdemucs_ft at shifts=2 and PCM_24.
- C: the hybrid, OTHER from the model.
- C′: the hybrid, OTHER as the residual.
- D: SCNet XL IHF (optional).
- E: FT Large (optional).

Use the scratch environment:

```bash
python3.13 -m venv "$TMPDIR/dance-ab/venv"
"$TMPDIR/dance-ab/venv/bin/pip" install torch==2.9.1 demucs==4.0.1 "msst[mel-band-roformer]==0.1.0" soundfile librosa
"$TMPDIR/dance-ab/venv/bin/hf" download KimberleyJSN/melbandroformer MelBandRoformer.ckpt --local-dir "$TMPDIR/dance-ab/weights"
```

Use Kim's config, `configs/config_vocals_mel_band_roformer.yaml` from
[KimberleyJensen/Mel-Band-Roformer-Vocal-Model](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)
(chunk 352800, num_overlap 2). Record the wall time per track and per stage. Also record the
analyze_stems / detect_regions / embed time on a temp DB.

**Objective proxies** (no ground truth, so these are screens, not verdicts):

- **Reconstruction residual:** `20·log10(rms(mix − Σstems)/rms(mix))`, the `stems-hygiene`
  definition. Compute it before any residual folding, because C′ is 0 by construction.
- **Bleed into vocals:** RMS of VOCALS relative to the mix on the instrumental track, and in
  sections every candidate marks vocal-free. Lower is better.
- **Ghost vocals in OTHER:** correlate the 300 Hz–4 kHz envelopes of VOCALS and OTHER over
  vocal-present frames. Lower is better. This is model-free on purpose: using a RoFormer as the
  judge would favour C.

**Listening** happens in Live, on the Scarlett's cue bus (outs 3/4 → headphones). Score 1–5 per
check, blind:

1. Solo VOCALS: instrument bleed, "underwater" phasing.
2. Mute VOCALS (D+B+O): ghost vocals. This is the check that matters.
3. Solo BASS: kick bleed.
4. Solo DRUMS: sub bleed.
5. All four up against the original mix on the master: does it sound like the record?

**Pass for C (or D):**

- It wins checks 1 and 2 on at least 4 of the 5 tracks.
- It is no worse on checks 3–4.
- Ghost-vocal correlation drops clearly.
- It costs ≤1.2× B per track.

Otherwise stop at option 0, and record the result here.
