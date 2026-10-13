# Separator worker process: Demucs out of the `dance process` interpreter

**Status:** proposal, not implemented
**Date:** 2026-10-13
**Touches:** `pipeline/stages/separate.py` (internals only), new `src/dance/separator/`
(worker + client), three settings in `config.py`, `docs/troubleshooting.md`.
**Does not touch:** the Stage protocol, the dispatcher, models/Alembic, API
schemas, the OSC contract, or the `.als` writer.

---

## The problem, measured

Demucs runs inside the `dance process` interpreter (`separate.py:99-129`). A
native crash in torch/MPS (SIGBUS, which a shell reports as exit 138) is not an
exception. The dispatcher's `except Exception` (`dispatcher.py:445`) never sees
it, so the whole interpreter dies. Every track that is in flight in *any*
stage is left in its `*-ing` state. The next run handles those tracks in
`_heal_inflight_orphans` (`dispatcher.py:534-621`): each orphan gets a heal-ledger
count, and after 2 heals (`_MAX_HEALS_BEFORE_ERROR`, `dispatcher.py:50`) the track
is set to ERROR. The ledger was added on 2026-06-13 (`52ace3d`).

### Reproduced on today's code (Experiment A)

The harness uses this branch's real `Dispatcher` and `HealLedger` against a
throwaway SQLite DB. The stages are fakes that copy the real stages' attributes:
`separate` has concurrency 2 and `uses_gpu`, and `analyze` has concurrency 2. The
fake `separate` raises a real `os.kill(getpid(), SIGBUS)` on track 1. Each pass
runs in a fresh interpreter through `subprocess.run`, the same way the API runs
`dance process` (`api/routers/pipeline.py:944`).

| pass | returncode | after the pass |
|---|---|---|
| 1 | `-10` (SIGBUS) | tracks 1, 2 stuck `separating`; 5, 6 stuck `analyzing`; 3, 4 never reached |
| 2 | `-10` | the four orphans are healed (count 1), the same crash happens, and the same four are stuck again |
| 3 | `0` | 3, 4 `separated`. **1, 2, 5, 6 all `error`:** *"repeated native crash (likely SIGBUS/exit 138) during separate / during analyze … (try DANCE_DEMUCS_DEVICE=cpu)"* |

One bad track costs three runs and produces four ERRORs, three of them on
innocent tracks:

- Track 2 never ran. It had been claimed, then waited on the GPU semaphore
  (`dispatcher.py:440-442`).
- Tracks 5 and 6 were partway through `analyze`. Their error tells the user to
  change the *Demucs* device.

Under the API, each crashing pass also ends the job, because `rc != 0` breaks
the loop at `pipeline.py:954`. Recovery therefore takes three presses of
Process.

### What the real logs show

These figures come from `~/.dance/pipeline-logs`, read-only. The 12 job logs
date from 2026-05-24/25, before the heal ledger existed.

- **Timing.** 85 successful separations.
  - Demucs compute, measured from the `Separating:` line to `✓ separate` (so it
    excludes the semaphore queue): min 51 s, **median 93 s**, p75 145 s, max
    343 s.
  - Claim-to-done wall time: median 197 s, max 981 s.
  - Model load, from `Loading Demucs model` to the first `Separating:`: 1–3 s
    (once 7 s).
- **Runs that died without a trace.** 4 of the 8 non-empty job logs end with no
  traceback and no end-of-run summary. The exit code was not recorded; it
  lived only in the in-memory job registry.
  - `a899e3e07547` stops 17 min in, at `Separating: … Surrender` (track 206),
    with another separation still open.
  - `58aad2d89779` heals 206 and starts separating it, then its log ends within
    seconds.
  - `57c01b005f54` heals 206 and 207, runs one Demucs at a time, and
    **separates 206 cleanly in 172 s**.

  Today's ledger would have marked 206 ERROR after the second death. That
  argues against treating every crash as a property of the track.
- **Runs that hit the time cap.** `57c01…` and `8fade2c81b76` each stop about
  1 h after they start, mid-separation. Each is followed by `Healed 2 inflight
  orphans {'separate': 2}`. This is consistent with the API's 1-hour per-pass
  cap (`pipeline.py:951`), which SIGKILLs the whole pipeline.
- **Catchable errors.** The same `57c01` run shows what a *catchable* failure
  looks like. Track 222 hit a Python `RuntimeError: MPS backend out of memory
  (MPS allocated: 14.44 GiB … max allowed: 18.13 GiB. Tried to allocate 4.92
  GiB)` after 14.7 min. It was marked ✗ and the run continued. The run also
  logged 131 `kIOGPUCommandBufferCallbackErrorOutOfMemory` lines. This is the
  case the current code already handles, because the error surfaces as a
  Python exception.

### Why a warm model is a bonus, not the main argument

Starting a cold worker interpreter costs 0.86–0.95 s. That covers importing
torch, demucs and librosa plus the first MPS allocation, measured 3× with no
weights loaded. Loading the model takes another 1–3 s. Together that is about
3 % of the 93 s median for `htdemucs_ft`, whose weights are 4 × 84 MB
checkpoints in `~/.cache/torch/hub/checkpoints`.

That overhead becomes significant with the next separator. djbooth's
Mel-Band-RoFormer checkpoint is **3.76 GB** (`djbooth_stems_mlx.py:25`), about
11× larger. **The case for this change is isolation.** Keeping the model warm is
cheap to add now and will be needed later.

## Proposal

```
dance process (parent; owns the DB)              separator worker (child; no `dance` imports)
  Dispatcher ── separate thread (concurrency=1)    <separator_python> -I src/dance/separator/worker.py
    SeparateStage.process()                          load model once ─▶ {"event":"ready",…}
      SeparatorClient ── stdin: one JSON line ──▶    decode → apply_model → WAVs into tmp dir
                     ◀── stdout: one JSON line ──    {"ok":true,"stems":{…},…}
      validate → os.replace(tmp → stems_dir/<hash8>/<kind>.wav) → StemFile rows → SEPARATED
```

### Protocol (JSON lines, `v: 1`)

| direction | frame |
|---|---|
| child → parent, once | `{"v":1,"event":"ready","pid":…,"model":"htdemucs_ft","device":"mps","versions":{"torch":"2.9.1","demucs":"4.0.1"},"load_s":2.1}` |
| parent → child | `{"v":1,"op":"separate","id":"<uuid>","audio":"/abs/in.mp3","tmp_dir":"<stems_dir>/.tmp-<hash8>-<id>","shifts":2,"bit_depth":"PCM_24"}` |
| child → parent | `{"v":1,"id":…,"ok":true,"stems":{"drums":"…/drums.wav",…},"sample_rate":44100,"model":"htdemucs_ft","seconds":93.4}` |
| child → parent | `{"v":1,"id":…,"ok":false,"error":{"type":"RuntimeError","message":"MPS backend out of memory …"}}` |
| parent → child, end of run | `{"v":1,"op":"shutdown"}` → `{"v":1,"event":"bye"}`, exit 0 |

Rules:

- **One request at a time.**
- **stdout carries only protocol frames.** At startup the worker `dup`s fd 1
  for frames and points fd 1 at stderr, so tqdm, torch warnings or a stray
  `print()` cannot corrupt the channel. The prototype prints to stdout on
  purpose and still parses cleanly.
- **stderr is passed through** to the parent's log. The API already writes
  `dance process` output to `pipeline-logs/<job>.log`.
- **The parent validates every frame** with a pydantic model and checks that
  `id` matches. djbooth validates its helper's output with zod in the same way
  (`index.ts:31-57`, `:192`).
- **Model and device are fixed when the worker is spawned**
  (`--model htdemucs_ft --device auto`). Changing either means respawning, not
  sending a new request.

**Why not copy djbooth's shape as-is?** djbooth spawns one process per call and
reads a single JSON object at exit (`runPython`, `index.ts:198-261`). Its batch
mode loads the model once but reports only when the whole batch finishes
(`separate_batch`, `djbooth_stems_mlx.py:452-517`), so a crash mid-batch loses
every result in that batch. Streaming one answer per job keeps the model warm
*and* limits the damage of a failure to one track.

### Crash isolation

- **Worker death.** EOF on stdout means the worker died. The parent calls
  `proc.wait()`; a return code below 0 gives the signal name. The track gets
  `separate: separator worker died (signal SIGBUS) …`. The parent never dies.
  It marks only that track and respawns the worker lazily on the next request.
- **Retry once.** A track whose worker died gets **one more attempt in a fresh
  worker**, which also means a fresh MPS heap, before it is marked ERROR. Track
  206 above is the reason. This replaces the ledger's "2 heals across 2 runs"
  with "2 attempts within one run".
- **Python-level errors.** An `ok:false` reply, such as a catchable MPS OOM,
  marks the track ERROR at once with the worker's message. The worker stays up.
- **Timeouts.** The worker is killed and the track is marked ERROR, with no
  retry. A hang usually repeats, and each one costs a full timeout.
- **Circuit breaker.** After 3 worker deaths in a row on different tracks,
  `SeparateStage` stops spawning for the rest of the run. It marks each
  remaining claimed track ERROR with *"separator disabled for this run after 3
  consecutive worker deaths, not this track's fault"*. The stage cannot
  un-claim a track without a busy loop in `_worker_loop`, and adding a "stage
  unavailable" signal would change the protocol.
- **Heal ledger stays.** It is still needed when the *parent* is killed:
  Ctrl-C, the API's 1-hour cap, or a power cut. It just stops firing for
  separator crashes.
- **No partial output.** The worker writes into `stems_dir/.tmp-…`. The parent
  `os.replace`s each file to the unchanged final path
  `stems_dir/<hash8>/<kind>.wav` only after a valid `ok` frame. The move stays
  on one filesystem, so it is atomic. Final paths matter because Live
  references stems by absolute path. Today the code writes straight to the
  final path (`separate.py:145-154`).

### Timeouts

| timeout | default | basis |
|---|---|---|
| ready (spawn → `ready`) | 300 s | Measured 0.9 s imports plus 1–3 s load. The rest covers a first-run weight fetch (4 × 84 MB). |
| per job (`DANCE_SEPARATOR_TIMEOUT_S`) | 900 s | 2.6× the slowest compute observed (343 s), about 10× the median. Today the only bound is the API's 1-hour cap on the *whole pipeline* (`pipeline.py:951`); the CLI has none. |
| shutdown | 10 s | Then SIGKILL. Spawn with `start_new_session=True` so the kill reaches any grandchildren. |

There is no heartbeat. Demucs 4.0.1's `apply_model` exposes no progress
callback: its signature has only `progress: bool`, which drives tqdm. A
heartbeat thread would also keep beating while the main thread is stuck in a
Metal call. A fixed deadline is the honest option.

### GPU semaphore (`_gpu.py`)

- **What the semaphore actually covers.** `GPU_SEMAPHORE` is a
  `threading.Semaphore`, so it only coordinates threads within one interpreter.
- **Its use is unchanged.** The dispatcher still wraps `process()` in it
  (`dispatcher.py:440-442`). The parent thread holds a permit for the whole
  round trip, so CLAP `embed` in the parent still never overlaps Demucs in the
  child.
- **No existing coordination is lost.** The semaphore never coordinated with
  the API's CLAP text encode, despite what the `_gpu.py` docstring says,
  because the API runs `dance process` as a subprocess (`pipeline.py:865-952`).
  A cross-process GPU lock is a separate problem and out of scope.
- **`separate.concurrency` goes from 2 to 1.**
  - At the default `DANCE_GPU_CONCURRENCY=1`, the two threads already
    serialize completely: the semaphore wraps decode, compute *and* WAV
    writes. The second thread can only pre-claim a track and wait. That is
    track 2 in Experiment A, and it is the "Healed 2" pattern in the real logs.
  - At 2, two copies of the model share 16 GB of unified memory. The
    `separate.py:53` comment promising overlap is only true in that mode.
  - If parallel separation is ever wanted, it means N workers holding N model
    copies. See the decisions below.
- **Lifetime.** One worker per `dance process` run. It is spawned on the first
  `separate` claim and shut down when the dispatcher returns. Nothing stays
  resident between runs.

### Separate pinned environment

- **The worker stands alone.** It is launched by path with `-I` and imports
  nothing from `dance`: only the standard library plus torch, demucs, librosa
  and soundfile.
- **`DANCE_SEPARATOR_PYTHON` chooses the interpreter.** It defaults to
  `sys.executable`, which is today's venv, so step 1 needs no migration.
- **A future RoFormer gets its own env.** It would ship as
  `separators/<name>/pyproject.toml` plus a lockfile, launched the way djbooth
  launches its helper: `uv run --python 3.12 --project … --locked --no-dev
  python worker.py` (`index.ts:205-216`).
- **This machine:**
  - `uv` is **not installed** (`which uv` finds nothing).
  - Homebrew `python3.11`, `python3.12` and `python3.13` are all present.
  - The main venv is **3.13.3**, while `docs/troubleshooting.md:40` blames
    "Python 3.14". A pinned separator env makes the separator's torch
    independent of that drift.
- **Per-process MPS memory settings (hypothesis, untested).** A separate
  process can set its own `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, which defaults to
  1.7 ([PyTorch 2.9 docs](https://docs.pytorch.org/docs/2.9/mps_environment_variables.html)).
  The log's "max allowed: 18.13 GiB" on a 16 GB machine is that 1.7×. Lowering
  it for the separator alone might turn paging into a catchable OOM without
  affecting CLAP.

### Stage protocol: unchanged

- `StemSeparationStage` keeps its name, states, `error_state`, `uses_gpu` and
  the `process(session, track, settings) -> bool` signature. All of the work
  stays inside `process()`. Only the *value* of `concurrency` changes.
- The idempotent short-circuit (`separate.py:88-94`) and all DB writes stay in
  the parent.
- The torch and demucs imports (`separate.py:21-32`) move into the worker.
  `_DEMUCS_OK` becomes the `ready` handshake.
- **A worker that cannot start is an environment failure, not a track
  failure.** Examples are a missing demucs or a bad interpreter path. Unless
  `--skip-stems` is set, `dance process` runs `worker.py --doctor` once, next
  to the ffmpeg preflight (`preflight.py`). If it fails, `dance process` exits
  and prints the worker's stderr tail. Without this check, every track would
  ERROR one by one, as `separate.py:82-86` does today.

### Prototype (Experiment B, scratch; not committed)

The prototype is a fake worker (about 60 lines) plus a client (about 90 lines),
run through this branch's real parallel `Dispatcher`. Seven tracks:

- Track 1 SIGBUSes.
- Track 2 hangs; the job timeout is 3 s for the test.
- Track 3 replies `ok:false`.
- Tracks 4 and 5 succeed.
- Tracks 6 and 7 start `pending`: they go through a 4-second fake `analyze`,
  then separate.

The worker imports torch and demucs and initialises MPS on startup, so respawn
cost is real:

```
track 1: attempt 1: separator worker died (signal SIGBUS)
track 1: attempt 2: separator worker died (signal SIGBUS)
✗ separate — track 1: separate: separator worker died (signal SIGBUS) on 2 attempts
✗ separate — track 2: separate: no answer within 3s; worker killed
✗ separate — track 3: separate: RuntimeError: MPS backend out of memory
result: {"analyze": {"processed": 2, "errors": 0, …}, "separate": {"processed": 4, "errors": 3, …}}
worker spawns=4 ready_latency_s=[0.89, 0.71, 0.7, 0.69]
parent returncode=0 · heal_ledger.json: (none) · stems on disk: tracks 4–7 only (16 files)
```

Same failures as Experiment A: one pass instead of three, zero false ERRORs,
and a bounded hang.

## Alternative: a fresh process per track (`multiprocessing`)

This means `ProcessPoolExecutor(max_workers=1, mp_context=spawn,
max_tasks_per_child=1)`, or a bare `multiprocessing.Process` per track.

- **For:** about 40 lines and no protocol. A crash is still contained.
  Spawning measured 0.62–0.75 s per task including torch and MPS init.
- **Against, measured (scratch `alt_pool.py`):**
  - The crash surfaces as `BrokenProcessPool: A process in the process pool was
    terminated abruptly…`. The signal is lost, and the pool refuses every later
    submit until it is rebuilt.
- **Against, from the docs:**
  - `max_tasks_per_child` needs Python 3.11+, but the repo targets py310.
  - Python **3.15** fixes a deadlock where the executor hangs after a worker
    exits at its `max_tasks_per_child` limit while tasks are queued
    ([concurrent.futures docs](https://docs.python.org/3/library/concurrent.futures.html),
    gh-115634).
- **Against, by construction:**
  - A spawned child unpickles a function from `dance`, so it must run an
    interpreter where `dance` is installed. That rules out a separately pinned
    env.
  - The model reloads for every track: about 3 s today, but roughly 11× the
    checkpoint size for RoFormer.

Also considered, as the status quo: `DANCE_DEMUCS_DEVICE=cpu`
(`troubleshooting.md:44-50`). It avoids the MPS crash at several times the
cost. It remains the documented fallback, and the worker passes `--device`
through unchanged.

## Recommendation

**Build the long-lived worker in two steps.**

1. **Isolation only, in the current venv.** Add `worker.py` and `client.py`,
   shrink `separate.py` to the client call plus DB writes, set `concurrency`
   to 1, use retry-once and a 900 s timeout, and add the `--doctor` preflight.
   This needs no new environment and no new dependency.
2. **A pinned environment, only when a second separator (RoFormer) lands.**
   Until then a pinned env costs tooling (`uv`) without fixing anything.

Demucs upstream is archived ("not maintained anymore", read-only since
2025-01-01: [facebookresearch/demucs](https://github.com/facebookresearch/demucs)).
Step 2 is when the separator can move without touching the rest of the venv.

## Decisions for the user

1. **Retry once on worker death, or ERROR on the first crash?** I recommend
   retrying once because of track 206. The cost is one more attempt (about
   1.5 min) on a track that really is bad.
2. **Per-job timeout:** a fixed 900 s (recommended, configurable) or one scaled
   to track duration. Scaling needs duration-vs-compute data, which I did not
   pull from the DB for this proposal.
3. **Separation concurrency:** one worker (recommended; one model in memory) or
   N workers following `DANCE_GPU_CONCURRENCY`.
4. **Pinned-env tooling for step 2:** `uv` (djbooth's precedent, needs an
   install) or a plain `python3.12 -m venv` plus a hash-pinned `pip`
   lockfile.

## Migration cost

- No schema, Alembic, API, OSC or `.als` change. The Stage protocol and
  dispatcher are unchanged. Step 1 adds no dependency.
- Existing stems and `StemFile` rows are untouched, and final stem paths are
  identical.
- **Behaviour changes:**
  - A separator crash now costs one track instead of the run.
  - `separate` runs one track at a time. That is throughput-neutral at the
    default `DANCE_GPU_CONCURRENCY=1`. Users who set it to 2 lose the overlap,
    which is the configuration in which two separations were open together.
- **Code size:**
  - `worker.py`: about 150 lines (today's `separate.py:65-163` minus the DB).
  - `client.py`: about 120 lines (spawn, frames, deadlines, respawn, breaker).
  - `separate.py`: shrinks to about 80 lines.
  - `config.py`: 3 settings (`separator_python`, `separator_timeout_s`,
    `separator_ready_timeout_s`).
  - Roughly a day of work, including the real-data run.

## Verification plan

1. **CI, synthetic, no torch: `tests/test_separator_worker.py`.** This turns
   Experiment B into a test.
   - A fake worker script picks its behaviour from the audio file name: SIGBUS,
     hang, `ok:false`, junk `print()` to stdout, or success.
   - Assertions, through a real parallel `Dispatcher`:
     - One pass with return code 0.
     - The crasher is ERROR with `signal SIGBUS … 2 attempts`; the hanger is
       ERROR with a timeout message.
     - Bystanders in `analyze` are unaffected.
     - The heal ledger is empty, and the failed tracks have no file at the
       final path.
     - The respawn count is exact, and the breaker trips after 3 consecutive
       deaths.
   - Follows `tests/conftest.py` isolation.
2. **Contract test against the real `worker.py --doctor`** in the current venv,
   skipped when demucs is absent. It checks the `ready` frame schema and the
   versions.
3. **Real data (rule 2), before calling it done:**
   - Setup: temp `DANCE_DATA_DIR`, `DANCE_LIBRARY_DIR` and `DANCE_STEMS_DIR`,
     with 3 real tracks *copied* in.
   - Run `dance process --track-id …` on MPS.
     - **Expect:** compute in line with the 93 s median, and one `Loading
       Demucs model` per run.
     - **Expect:** per-stem RMS within 0.5 dB of the existing stems for the
       same tracks.
   - **Crash e2e with no code hook:** mid-separation, run `kill -BUS $(pgrep
     -f separator/worker.py)` from another shell.
     - **Expect:** the parent logs `separator worker died (signal SIGBUS)`,
       retries in a fresh worker, and finishes the run with exit code 0.
   - Repeat the kill on both attempts.
     - **Expect:** exactly that one track is ERROR, and the next track
       separates in the same run.
4. **No Live check is needed**, because no `.als` or stem path changes.

## Follow-ups (not in this proposal)

- `_heal_inflight_orphans` puts the Demucs hint ("try DANCE_DEMUCS_DEVICE=cpu")
  on orphans from every stage (`dispatcher.py:583-588`; Experiment A tracks 5
  and 6). The message should name the stage's own remedy.
- The API's 1-hour per-pass `subprocess.run(timeout=…)` SIGKILLs every stage.
  With per-track deadlines it could become a no-progress timeout instead.
