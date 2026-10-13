# Stem quality: put a vocal specialist in front of htdemucs_ft

**Status:** PROPOSAL. Awaiting a decision. No code has been written yet.
**Date:** 2026-10-13
**Touches:** `pipeline/stages/separate.py` (internals, plus the skip condition), `config.py`
(new keys), one new optional dependency. **No** model/Alembic, Stage-protocol, `.als`, OSC or
API-schema change. Rule 1 still applies because this is the user's call on the model, the
dependency and roughly a day of compute.

---

## Where we are

`separate` runs `htdemucs_ft` with `shifts=2` and writes PCM_24 (`config.py:95-104`,
`separate.py:323-337`). `htdemucs_ft` is a bag of four HTDemucs nets, one per source.
`demucs/remote/htdemucs_ft.yaml` gives the bag diagonal weights, but `apply_model` still runs
every sub-model over the whole track, `shifts` times. That means **8 full forward passes per
track**, and 3 of every 4 outputs are thrown away.

- **Cost, measured:** a 30 s excerpt took 31.7 s on MPS at s2 (commit `994f886`). That is
  ≈1.06 s per audio-second, ≈0.13 s per pass. A 6-minute track takes ≈6.3 min. Re-running
  all 278 tracks (6 min average assumed) takes ≈29 h, which matches the "37 h for 353 tracks"
  noted in `f14e99d`.
- **Library:** 278 tracks, 1,112 stems, ~52 GB on disk (≈187 MB per track). Weights are
  4 × 84 MB, in `~/.cache/torch/hub/checkpoints`.
- **Weak stem: vocals.** In the like-for-like table below, the htdemucs_ft vocals specialist
  scores 8.38 dB. Public vocal specialists score ~11 dB. Our drums and bass are already as good
  as anything public with a clear licence.

Vocals matter more here than the average SDR suggests. The vocals fader is the main
performance move (acapella over the other deck, or muting vocals to dodge a clash), and two
kinds of error hurt it:

- **Vocal bleed into `other`.** Muting the vocals fader still leaves ghost vocals.
- **Synth leakage into `vocals`.** This flips `vocal_present` (`analyze_stems.py:21,197`) and
  the vocal regions (`detect_regions.py:59`), and it skews the vocals-stem CLAP embedding
  behind vocals-column recs.

## The benchmark numbers, read carefully

**Like-for-like: Multisong, track-level SDR (dB).** All rows come from one leaderboard
([MSST pretrained models](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)):

| model | vocals | drums | bass | other | weights | licence |
|---|---|---|---|---|---|---|
| **htdemucs_ft** (per-source specialists, ours) | 8.38 | 11.13 | 11.96 | 5.85 | 4 × 84 MB | MIT code, weights unstated |
| Kim Mel-Band RoFormer (vocals/other) | **10.98** | – | – | – | [913 MB](https://huggingface.co/KimberleyJSN/melbandroformer) | MIT (HF card) |
| viperx BS-RoFormer ep317 (vocals/other) | 10.87 | – | – | – | ≈640 MB (UVR model_repo) | none stated |
| SCNet XL IHF (4-stem) | 9.68 | 11.58 | 11.94 | 6.48 | [204 MB](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.15) | MIT repo; trained on MUSDB18 only |
| BS-RoFormer SW (6-stem) | 11.30 | 14.11 | 14.62 | 8.71¹ | ? | **unknown** ([MVSEP](https://mvsep.com/news/64)) |
| MelBandRoformer4StemFTLarge (djbooth) | no Multisong figure | | | | 3.76 GB | Apache-2.0 |

¹ A 6-stem `other` excludes guitar and piano, so it is not comparable with the 4-stem rows.

**MUSDB18 test: two different metrics.** These two sets of numbers can't be ranked against
each other:

- **HT Demucs fine-tuned:** 9.00 average (drums 10.08, bass 10.39, other 6.32, vocals 9.20),
  from the [paper](https://arxiv.org/abs/2211.08553). The metric is the SiSEC18 median of
  1-second chunks.
- **djbooth's MelBandRoformer4StemFTLarge:** 8.99 average (vocals 10.70, drums 10.43,
  bass 7.74, other 7.10), from its [card](https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge).
  This uses MSST's whole-track SDR
  ([metrics.py](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/utils/metrics.py)).
- **Same yardstick as djbooth's model:** SCNet XL IHF scores 10.08 (bass 9.23) and MSST's
  4-stem BS-RoFormer scores 9.65. So djbooth's model trails the best clearly licensed 4-stem
  model by ~1.1 dB on its own metric. Its later "v2" upload scores lower again (8.87).

**Caveats:**
- Training sets differ. htdemucs_ft adds 800 extra songs. SCNet XL IHF saw only MUSDB18's
  100 songs.
- Nobody publishes overlap checks against Multisong.
- None of these sets is club music, and our vocals are sparse, chopped and heavily effected.

The numbers rank the candidates. Listening decides.

## Options

All options keep exactly `drums/bass/vocals/other`. That is a hard requirement:
`StemKind` (`database.py:86-92`), `SOURCE_STEM_KINDS` (`als/writer.py:97`), the bridge's
`_SOURCE_STEM_KINDS` (`osc/bridge.py:882`) and `_STEM_COLUMNS` (`api/routers/ableton.py:808`)
all assume those four. A 6-stem model would only fit by summing guitar+piano+other into
`other`.

### 0. Status quo

- **Pros:** no work and no risk.
- **Cons:** vocals stay ~2.6 dB behind the specialists, and we keep paying for 8 htdemucs
  passes. The hybrid needs 3 of them plus one RoFormer pass.

### 1. djbooth's 4-stem RoFormer (MelBandRoformer4StemFTLarge)

- **Quality:** not shown to beat htdemucs_ft (see above). Its weakest stem is bass, and bass
  is the stem that clashes in a blend. That makes it a likely regression risk exactly where we
  are strong.
- **Compute:** djbooth reports "faster than real time" on an M2 Max via MLX (`PORTING.md:331`;
  10.88 s of audio in 1.49 s including load, `:350-352`). Torch on MPS on an M2 Pro is
  unmeasured. The M2 Pro has 16/19 GPU cores against the M2 Max's 30/38
  ([Wikipedia](https://en.wikipedia.org/wiki/Apple_M2)).
- **Memory:** 3.76 GB of fp32 weights. djbooth measured 3.96 GB max RSS.
- **Packaging:**
  - The MLX port needed a monkeypatch to restore the checkpoint's `skip_connection` schedule
    (`python/stems-mlx/djbooth_stems_mlx.py:128-172`).
  - The upstream package also "adds an untrained terminal normalization" (`PORTING.md:343-347`).
  - djbooth pins Python `>=3.12,<3.13` (`pyproject.toml:5`); we run 3.13.
  - Without the patch the output is silently wrong, not an error. That is a red flag for an
    unattended pipeline.
- **Verdict:** a 3.8 GB download with an unproven quality gain. Reject.

### 2. Hybrid: RoFormer vocals, then htdemucs_ft drums/bass/other on (mix − vocals) ← **recommended**

**How it works:**
1. `V` = Kim Mel-RoFormer vocals of the mix.
2. `I = mix − V`.
3. Run only bag members 0–2 of htdemucs_ft (drums, bass, other) on `I`, with `shifts=0`.
4. Drop the htdemucs vocals specialist entirely.

This is the MDX23 3rd-place recipe: Demucs applied "to the instrumental track only"
([Solovyev et al. 2023, §6.2.1](https://arxiv.org/abs/2305.07489)).

- **Expected quality (Multisong):**
  - vocals ≈ 10.98 (+2.6 dB);
  - drums/bass ≈ 11.13/11.96, unchanged, and already level with SCNet XL IHF;
  - `other` ≥ 5.85, because vocals are removed before it is estimated.
  - The MDX23 ensemble's `other` reached 6.54, but with more models
    ([repo table](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model)).
- **Compute:**
  - htdemucs half: 3 passes ≈ **0.40 s/s**, measured rate × 3.
  - Kim pass: unmeasured. A back-of-envelope estimate from its config (`dim 384, depth 6,
    60 bands, hop 441, 8 s chunks, num_overlap 2`) is ≈0.7 TFLOP per audio-second. At 30–50%
    of the M2 Pro GPU's ~6.8 TFLOPS fp32 peak, that is **≈0.2–0.5 s/s**. This is an
    assumption.
  - Total ≈ **0.6–0.9 s/s** against today's 1.06, so likely *cheaper* than now. Stage 0 gates
    on this.
  - Dropping shifts on a lossy spotDL source costs little: `config.py:98-100` records the s2
    win as small.
- **Memory:** Kim weights are 913 MB fp32, plus 3 × 84 MB, plus per-chunk activations.
  - Estimate: 2–4 GB per worker. If peak RSS is high, run with `DANCE_GPU_CONCURRENCY=1`.
  - The full-track buffers (mix, V, I, D, B, O) are ≈0.8 GB for a 6-min track.
- **Licence:** Kim weights are MIT. htdemucs_ft is unchanged.
- **Packaging** (PyPI metadata checked 2026-10-13):
  - **[`msst`](https://pypi.org/project/msst/) 0.1.0** (ZFTurbo, MIT, 2026-09-09):
    `requires-python >=3.10,<3.14`, `torch>=2.0.1,<2.12`. That fits our 3.13.3 + torch 2.9.1.
    The `mel_band_roformer` extra adds `beartype, einops, rotary-embedding-torch, PoPE-pytorch`.
    It is a single "Beta" release; MSST's `inference.py` selects MPS when it is available.
  - **[`audio-separator`](https://pypi.org/project/audio-separator/):** 0.46.0 and later
    (from 2026-08-27) require **`torch>=2.13` on macOS arm64**, which would force a torch
    upgrade under Demucs, CLAP and Qwen. 0.45.0 still allows `torch>=2.3` but pins
    `beartype<0.19` and `rotary-embedding-torch<0.7`, and carries a model-download manager we
    don't need.
  - **Vendoring** MSST's MIT `mel_band_roformer.py` is the smallest footprint, but we would
    own the code.
- **Layout:** unchanged. Each stem row keeps its own `model_used` (String(64)), for example
  `melrof-kim|pcm24|msst0.1.0` on vocals and `htdemucs_ft-novox|s0|pcm24|demucs4.0.1` on the
  others. `stem_provenance` already trims to fit (`separate.py:124-137`).
- **Open choice: how to define `other`** (decision 3):
  - **2a:** the htdemucs other-specialist's output. The residual QC keeps its meaning (≈ −20
    dB today).
  - **2b:** `I − D − B`. The stems then sum to the mix exactly, but the residual QC drops to
    the PCM_24 floor (well below −100 dB) and stops flagging anything. Every drums/bass miss
    lands in `other`.
  - Both come out of one run, so the A/B renders both.

### 3. One stronger 4-stem model

- **SCNet XL IHF:**
  - **Gains:** +1.3 vocals, +0.45 drums, ≈0 bass, +0.63 other.
  - **Cost:** one 204 MB network instead of four; unmeasured on MPS.
  - **Downsides:** vocals are still 1.3 dB below Kim. It was trained on 100 MUSDB18 songs,
    so how it handles club music is an open question.
  - **Packaging:** MSST base package (`model_type=scnet`).
  - **Verdict:** a reasonable optional third arm in Stage 0, or a later swap for the
    drums/bass/other half.
- **BS-RoFormer SW:**
  - Has the best public numbers by a wide margin.
  - Its licence is unknown and its upstream is undisclosed. The HF rehost
    (`jarredou/BS-ROFO-SW-Fixed`) returned HTTP 401 on 2026-10-13.
  - **Verdict:** excluded.

### 4. An MLX backend

This is a runtime choice, not a model choice, and could be applied to any of the options
above.

- **Pros:** possibly faster.
- **Cons:**
  - It brings a second ML runtime next to torch.
  - [`mlx-audio-separator`](https://pypi.org/project/mlx-audio-separator/) shipped 21 releases
    between 2026-02-27 and 2026-10-02 (now 0.1.20, `mlx<0.33`). djbooth pinned 0.1.5 and
    needed the patch described under option 1.
  - Checkpoint conversion still needs torch.
  - Our MPS→CPU retry (`separate.py:339-358`) and the heal ledger are torch-path only.
- **Verdict:** revisit only if Stage 0 measures Kim on torch MPS above ~1 s/s.

| | vocals | drums/bass | other | est. s per audio-s | new weights | dep risk |
|---|---|---|---|---|---|---|
| 0 status quo | 8.38 | 11.13 / 11.96 | 5.85 | 1.06 (measured) | – | – |
| 1 djbooth 4-stem | ? (MUSDB 10.70) | ? (MUSDB bass 7.74) | ? | unmeasured | 3.76 GB | high (MLX) / medium (torch) |
| **2 hybrid** | **10.98** | 11.13 / 11.96 | ≥5.85 | 0.6–0.9 (est.) | 913 MB | medium |
| 3 SCNet XL IHF | 9.68 | 11.58 / 11.94 | 6.48 | unmeasured | 204 MB | medium |
| 4 MLX runtime | – | – | – | maybe lower | – | high |

## Recommendation: option 2, in four stages

**Stage 0: measure (nothing in the repo). Needs your OK for the install and the 913 MB
download.**

1. Set up a scratch venv and fetch the weights:
   ```bash
   S=<scratchpad>/sep-ab
   /Users/arya/git/dance/.venv/bin/python -m venv $S/venv   # same base 3.13.3 interpreter
   $S/venv/bin/pip install "torch==2.9.1" "demucs==4.0.1" "msst[mel_band_roformer]==0.1.0" soundfile librosa
   curl -L --create-dirs -o $S/w/MelBandRoformer.ckpt https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt
   curl -L --create-dirs -o $S/w/kj.yaml https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml
   ```
2. Write a throwaway `$S/ab.py`. It uses `msst.Separator(model_type="mel_band_roformer", ...)`
   and demucs `apply_model` on `BagOfModels(get_model("htdemucs_ft").models[:3], ...)`. For
   each of 5 tracks it writes:
   - **A:** a *fresh* `htdemucs_ft` run at s2/PCM_24. Older library stems may be s0/PCM_16,
     so they are not a fair baseline.
   - **B-2a** and **B-2b**, as defined under option 2.

   Outputs go to `$S/out/<track>/{X,Y,Z}/` with a hidden shuffle key. Each run is wrapped in
   `/usr/bin/time -l` for peak RSS. It logs s per audio-second and `reconstruction_residual_db`
   (imported from `dance`).
3. **Gate.** All of these must hold:
   - B ≤ 1.06 s/s on MPS;
   - peak RSS ≤ 6 GB;
   - no MPS error on any of the 5 tracks.

   If the speed gate fails, re-time Kim with `inference.num_overlap: 1`, which halves its
   pass, and listen for seams. Only then consider option 4.

**Stage 1: implement behind a setting, off by default.**
- New keys: `separation_vocal_model` (`None` means today's behaviour),
  `separation_vocal_config`, `separation_other` (`"model"` or `"residual"`).
- Add a hybrid path in `separate.py`. It needs a CPU-retry twin for the RoFormer, mirroring
  `_cpu_fallback_model`.
- Make the skip condition (`separate.py:369-375`) provenance-aware (decision 5).
- Unit tests use fakes, in a new `tests/test_separate_hybrid.py`.
- Real e2e, the same way `994f886` did it: a temp DB and stems dir, one real track, then open
  it in Live.

**Stage 2: A/B in Live** (protocol below). Gate: B wins or ties on ≥4 of 5 tracks for
listening tests 1–3, and has *no* loss on test 4.

**Stage 3: flip the default and migrate forward-only.**
- New tracks get the hybrid.
- Existing tracks are re-separated in explicit batches (next section).

## Migration (forward-only)

- **What triggers a re-separation.** Today, `separate` skips any track that already has four
  stems on disk (`separate.py:369-375`). So `dance reprocess --from separate` does **not**
  re-run separation (`cli.py:454-455`, `pipeline/reprocess.py:5-8`).
  - Proposed: skip only when every row's `model_used` equals the configured provenance.
  - `dance process` still never rewinds a complete track, so nothing re-separates on its own.
  - Alternative: a `--reseparate` flag on `reprocess` (decision 5).
- **How to batch.** Use explicit ids. Repeating `--all --limit N` re-selects the same
  lowest-id tracks. To list candidates:
  ```bash
  sqlite3 'file:/Users/arya/.dance/dance.db?immutable=1' \
    "SELECT track_id FROM stem_files WHERE kind='vocals' AND model_used NOT LIKE 'melrof%' ORDER BY track_id"
  dance reprocess --from separate -t 12 -t 15 ...   # ~20 per overnight batch; never --skip-embeddings
  dance build-graph
  ```
- **What else re-runs.** The rewind runs analyze_stems → detect_regions → embed (per-stem
  CLAP). After that, `build-graph`.
  - `duplicate_of` tracks are skipped by default (`reprocess.py:120`).
  - Until a migration finishes, recs compare old and new stems.
- **What doesn't need to change:**
  - **`.als`:** stem paths stay `stems_dir/<hash8>/<kind>.wav` (`separate.py:431`), so the
    files need no regeneration.
  - **Waveform sidecars:** they self-invalidate on mtime (`audio/waveform.py:8-13`).
- **Cleanup.** Close any open Live set first. Delete each stem's Live `.asd`, because Live
  matches on `OriginalFileSize` (`warp-guard.md:105`), and a same-length PCM_24 rewrite has
  the same size.
- **Disk:** replaced in place at the same size, so ≈0 net change against today's 52 GB. There
  is no library-wide backup. Rollback means re-separating with the old setting (≈29 h).
- **Time:** 278 × 6 min × 0.6–0.9 s/s ≈ **17–25 h** of separation, plus the downstream
  stages.
  - Cheap path with 2b: keep the on-disk D/B, compute the new V, and set `O = mix − D − B − V`.
    Only the Kim pass is needed, ≈ **6–14 h**, and drums/bass analysis stays byte-identical.

## A/B protocol (5 real tracks)

**Picking the tracks:**
- 2 full-vocal house tracks;
- 1 with chopped vocal samples;
- 1 instrumental (to test false positives);
- 1 with a lead synth in the vocal range (to test leakage).

**Setup.** In a blank Live 12.4 set, load arms X, Y and Z as groups. Put Live in Solo → Cue
mode, with Cue Out on Scarlett outs 3/4 to the headphones. Score blind; open the shuffle key
only after scoring.

**Listening tests:**
1. **Vocals solo:** hats or synth leaking in, watery artifacts.
2. **Mix with vocals muted:** ghost vocals left in `other` or `drums`.
3. **Acapella swap:** this track's vocals over another track's D+B+O at matched tempo. This is
   the actual move.
4. **Drums solo and bass solo:** transient smear, sub stability. Any regression here vetoes B.
5. **Sum of 4 against the mix:** they must be indistinguishable.

**Objective proxies.** None of these is ground truth; they are logged per arm:
- `reconstruction_residual_db`.
- **Vocal bleed:** run *both* the Kim model and the htdemucs vocals specialist over each
  arm's `other` and `drums`, and report extracted vocal energy in dB relative to the stem.
  Two probes are needed because Kim alone favours B by construction.
- **Vocals-stem floor:** the 20th-percentile 100 ms RMS in dBFS, plus the whole-track vocals
  RMS on the instrumental track.
- **Downstream flips:** whether `presence_ratio`, `vocal_present` (>0.15) or
  detect_regions' >0.3 threshold flips between arms. Use analyze_stems' constants and write
  nothing to the DB.
- **Cost:** s per audio-second and peak RSS.

## Decisions for you

1. **Stage 0:** may I create a scratch venv (`msst==0.1.0` + deps) and download Kim's
   913 MB checkpoint? Should SCNet XL IHF (204 MB) be added as a third arm?
2. **Shifts on the htdemucs half:** s0 (recommended: deterministic, and it pays for the
   RoFormer pass) or keep s2 (3 more passes, ≈ +0.4 s/s).
3. **`other`:** model output (2a) or residual (2b). Recommend: decide by the A/B.
4. **Packaging:** pin `msst[mel_band_roformer]==0.1.0` as an optional `dance[roformer]` extra
   (recommended), vendor the MIT module, or `audio-separator<=0.45.0`. Avoid ≥0.46, which
   forces torch ≥2.13.
5. **Re-separation trigger:** provenance-aware skip (recommended) or a `reprocess
   --reseparate` flag.
6. **Migration scope:** whole library (17–25 h), vocal-bearing tracks only, or the 2b
   vocals-only shortcut (6–14 h).
7. **Vocal model:** Kim (MIT; recommended) or viperx BS-RoFormer (no stated licence, 0.1 dB
   lower).

## Not verified here

- No RoFormer, SCNet or MLX code was run on this machine. This task allowed no installs or
  downloads.
- Every RoFormer timing and memory figure above is an estimate or a third-party number, and
  Stage 0 exists to replace it.
- The htdemucs_ft rate is from the real MPS run recorded in `994f886`.
- Isolating the separator in a worker process (branch
  `claude/proposal-separator-worker-process`) is orthogonal. If it lands first, the hybrid
  runs inside it.
