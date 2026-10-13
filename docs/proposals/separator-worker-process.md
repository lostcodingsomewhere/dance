# Separator worker: Demucs in a long-lived child process

**Status:** proposed (2026-10-13), not implemented
**Touches:** `pipeline/stages/separate.py` (internals of `process()` only), a new
`src/dance/separator/` package, three settings in `config.py`, tests, `docs/troubleshooting.md`.
**Does not touch:** the Stage protocol, the dispatcher, models/Alembic, API schemas, the OSC
contract, the `.als` writer, or stem paths on disk.

---

## The problem

Demucs runs inside the `dance process` interpreter (`separate.py:67-78`, `:380`, `:399`). A
native crash in torch/MPS (SIGBUS, exit 138 in a shell) is not an exception: the dispatcher
thread's `except Exception` (`dispatcher.py:460`) never runs and the whole process dies. Every track
in flight in *any* stage is left in its `*-ing` state. The next run's
`_heal_inflight_orphans` (`dispatcher.py:553-640`) requeues each one and counts the heal in
`heal_ledger.json`. On the second heal (`_MAX_HEALS_BEFORE_ERROR = 2`, `:52`) it sets
the track to ERROR with "repeated native crash … try DANCE_DEMUCS_DEVICE=cpu" (`:602-607`).

### Reproduced on today's `main` (Experiment A)

This uses the real `Dispatcher`, `HealLedger` and dispatcher lock against a scratch HOME and
SQLite DB. Only the stage bodies are fakes, and they keep the real stages' attributes:
`separate` has concurrency 2 and `uses_gpu`, and `analyze` has concurrency 2. The fake
`separate` raises a real `SIGBUS` on track 1. Each pass runs in a fresh interpreter, the
same way the API runs `dance process` (`api/routers/pipeline.py:979`).

| pass | rc | after the pass |
|---|---|---|
| 1 | `-10` | 1, 2 stuck `separating`; 5, 6 stuck `analyzing`; 3, 4 never reached |
| 2 | `-10` | the four are healed (count 1), the same crash happens, and the same four are stuck |
| 3 | `0` | 3, 4 `separated`. **1, 2, 5, 6 all `error`**: "repeated native crash … during separate / during analyze … (try DANCE_DEMUCS_DEVICE=cpu)" |

So one bad track takes three runs and leaves four ERRORs, three of them on innocent tracks.
Track 2 never ran: it was claimed and then waited on the GPU semaphore. Tracks 5 and 6 were
being analyzed, yet their message tells the user to change the Demucs device. Under the API,
`rc != 0` ends the job (`pipeline.py:990-991`), so recovery takes three presses of Process.

### What the real data says (read-only: `~/.dance/pipeline-logs`, `jobs.json`, DB via `immutable=1`)

| fact | number |
|---|---|
| Successful separations in the 12 API job logs (2026-05-24/25) | 85 |
| Demucs compute, from `Separating:` to `✓ separate` | median **93 s**, p90 188 s, max 343 s |
| Claim to done (includes the semaphore wait) | median 197 s, max 981 s |
| Compute ÷ track duration (81 tracks joined to `duration_seconds`) | median 0.44, max 0.87 |
| Model load, from `Loading Demucs model` to the first `Separating:` | 1–3 s (once 7 s) |
| The 5 pipeline jobs still in `jobs.json` | 1 done, 3 × `rc=-15` (SIGTERM), 1 × "pipeline exceeded 1h timeout", **0 × SIGBUS** |
| ERROR tracks in the DB today | 4, all catchable errors in `separate`: 2 MPS OOM, 2 `AssertionError` |

These logs were written at `shifts=1`. The default has been `shifts=2` since `f14e99d`
(`config.py:103`), which runs two passes of the model, so expect about 3 min per track
today. Two consequences:

- **SIGBUS has not appeared in the recorded runs.** The crash is documented on an off-spec
  Python 3.14 venv (`docs/troubleshooting.md:44-46`, heal ledger `52ace3d`). The venv is
  3.13.3 today. The case for this proposal is the blast radius when a crash does happen
  (Experiment A), a bound on hangs, and the second separator, not crash frequency.
- **The ledger cannot tell a native crash from a killed parent.** Track 206 was in flight
  when job `a899e3e07547` was SIGTERMed. It was healed, claimed again, and was in flight when
  `58aad2d89779` was SIGTERMed too. Under today's ledger the next run would have flagged it
  "repeated native crash (likely SIGBUS)". In fact the run that started next (`57c01b005f54`)
  separated it cleanly in 172 s. A child's own exit status is a direct signal; the ledger
  is an inference.

## Proposal

```
dance process (parent: owns DB, dispatcher, GPU permit)      separator worker (child: no `dance` imports)
  separate thread (concurrency 1)                              <separator_python> -I src/dance/separator/demucs_worker.py
    StemSeparationStage.process()                                --model htdemucs_ft --device mps
      SeparatorClient ── stdin: one JSON line per job ──▶        load model once ─▶ ready
                      ◀── stdout: one JSON line per reply ──     decode → apply_model → QC → hidden temps
      validate → delete stale rows → os.replace temps → StemFile rows → SEPARATED
```

### Protocol: JSON lines, `"v": 1`, one request in flight

| direction | frame |
|---|---|
| child → parent, once | `{"v":1,"event":"ready","pid":…,"model":"htdemucs_ft","device":"mps","versions":{"torch":"2.9.1","demucs":"4.0.1"},"load_s":2.1}` |
| parent → child | `{"v":1,"op":"separate","id":"<hex>","audio":"/abs/mix.mp3","out_dir":"<stems_dir>/<hash8>","shifts":2,"subtype":"PCM_24"}` |
| child → parent, optional | `{"v":1,"event":"retrying","id":…,"device":"cpu","reason":"RuntimeError: MPS backend out of memory …"}` |
| child → parent | `{"v":1,"id":…,"ok":true,"stems":{"drums":"…/.drums.k3j2.partial.wav",…},"sample_rate":44100,"frames":…,"device":"mps","seconds":181.2,"residual_db":-20.7,"clipped":{"bass":12},"provenance":"htdemucs_ft\|s2\|pcm24\|demucs4.0.1"}` |
| child → parent | `{"v":1,"id":…,"ok":false,"error":{"type":"AssertionError","message":""}}` |
| parent → child / child → parent | `{"v":1,"op":"shutdown"}` / `{"v":1,"event":"bye"}`, then exit 0 |

- **stdout carries frames and nothing else.** At startup the worker keeps a private `dup`
  of fd 1 for frames and `dup2`s stderr onto fd 1. A `print()`, tqdm, or a C-level write
  then lands in the log instead of the protocol. The prototype prints to stdout on every
  job and still parses cleanly.
- **stderr is inherited.** Under the API it lands in `pipeline-logs/<job>.log`, as today.
- **The parent validates every frame** with pydantic (already a dependency) and checks that
  `id` matches. An unknown `v` fails the job.
- **Model and device are fixed when the worker is spawned.** A different
  `settings.demucs_model` means a respawn. `shifts` and the PCM subtype travel with each
  request.
- **Why not copy djbooth as is?** Its `runPython` (`rekordbox-ts/src/stems/index.ts:198-261`)
  spawns one process per call, parses stdout only at exit, and has no timeout. Its batch mode
  loads the model once but answers once, at the end (`djbooth_stems_mlx.py:452-517`), so a
  crash mid-batch loses the report for every finished job. One reply per job keeps the
  model warm *and* limits a failure to one track.

### Crash isolation

| in the child | the parent sees | track | worker |
|---|---|---|---|
| SIGBUS / segfault on MPS | EOF; `wait()` gives `rc < 0`, which names the signal | partials swept, then **one retry in a fresh `--device cpu` worker**. If that also fails: ERROR "separator worker died (SIGBUS on mps, SIGBUS on cpu)" | the next track respawns on MPS |
| Python error | an `ok:false` frame | ERROR `separate: <Type>: <message>` | stays up |
| no reply by the deadline | timeout | ERROR "no answer within N s; worker killed" (no retry: a second hang would cost a second full deadline) | `killpg(SIGKILL)`, respawned lazily |
| cannot start (bad interpreter, import error) | spawn error, or EOF before `ready` | ERROR "separator worker failed to start: <stderr tail>", the same per-track behaviour as today's "Demucs not installed" (`separate.py:363-367`) | none |
| the *parent* dies (Ctrl-C, SIGTERM, the API's 1 h cap) | — | heal ledger, unchanged | a watchdog thread sees `getppid()` change and calls `_exit` |

- **Why the retry is on CPU:** CPU is the documented workaround for the crash
  (`troubleshooting.md:48-54`). In-process, the stage already retries a Python-level MPS
  error on CPU (`separate.py:339-358`). That logic moves into the worker unchanged, which is
  why the `retrying` frame exists.
- **Breaker:** after 2 *consecutive* tracks whose MPS attempt died, the client pins CPU for
  the rest of the run and logs it once. A systematically broken MPS then costs two crashes,
  not one per track.
- **No partial output.** This is today's two-phase write (`write_stems_atomic`,
  `separate.py:210-248`), split across the two processes. The worker does phase 1, writing
  hidden `.<kind>.*.partial.wav` temps into the final directory. The parent does phase 2, the
  four `os.replace`s (`:237-242`), only after a validated `ok` frame. A worker death strands
  only hidden temps, and `sweep_partials` (`:198-207`) removes them before the retry. Live
  still never sees a truncated `<kind>.wav`, and final paths are unchanged.
- **The heal ledger stays** for parent deaths. Separate-stage crashes stop reaching it.

### Timeouts

| deadline | value | basis |
|---|---|---|
| spawn → `ready` | 300 s | Spawn plus `import torch, demucs` plus MPS init measured 0.63–0.73 s (Experiment B), and the model load is 1–3 s. The rest is headroom for a first-run download of the 321 MB of weights. |
| job on MPS | `120 s + 2.0 × max(1, shifts) × duration_s` | The worst observed ratio is 0.87 × duration per model pass. At `shifts=2` this allows 4× duration, about 2.3× headroom. For a median 228 s track that is 17 min. |
| job on CPU (after `retrying`, or in a CPU worker) | the MPS value × 4 | **Unmeasured**: the docs say CPU takes "a few minutes per track instead of ~1–3". Measure it in verification step 3. |
| shutdown | 10 s | Then `killpg`. The worker gets `start_new_session=True` so the kill reaches any grandchildren. |

- **There is no heartbeat.** `apply_model` in demucs 4.0.1 takes only `progress: bool`,
  which drives tqdm, not a callback (`demucs/apply.py:132-138`). A heartbeat thread would
  also keep beating while the main thread is stuck inside a Metal call.
- **No hang has been observed.** The deadline is a defensive bound.
- **The API cap still applies.** Its 1 h `subprocess.run` cap per pass (`pipeline.py:986`)
  is still the outer bound. One 3742 s track in the library would need about an hour of
  compute on its own at `shifts=2`.
- **Idle shutdown.** The client stops the worker after 60 s with no request, and at
  interpreter exit. `dance ingest` keeps the interpreter alive for build-graph, tag and
  export-als after `process`, and the model should not sit resident through them. The worker
  also exits on stdin EOF.
- **Orphans are measured (Experiment B2).**
  - **With the watchdog:** SIGKILLing the parent mid-job, which is what the API cap does,
    leaves the worker alive for 0.25 s.
  - **Without it:** the worker was still computing 15 s later. Meanwhile a new run could
    already take the dispatcher lock, because the orphan never held it: the lock fd is not
    inheritable (`dispatcher_lock.py:292-293`). Two separators would then share the GPU.

### GPU semaphore (`_gpu.py`)

- **Nothing changes in how it is used.** The dispatcher wraps `process()` in
  `GPU_SEMAPHORE` (`dispatcher.py:455-457`), and the parent thread holds that permit for the
  whole round trip. CLAP `embed` in the parent therefore still never overlaps Demucs in the
  child.
- **It is a `threading.Semaphore`** (`_gpu.py:21`), so it only coordinates threads within
  one process.
  - It already fails to coordinate the API's text encode (`recommend.py:99-122`) with
    `dance process`, which runs as a subprocess (`pipeline.py:936-988`). The `_gpu.py`
    docstring overstates what it covers.
  - The worker makes this no worse. A cross-process lock is out of scope.
- **`concurrency` goes from 2 to 1** (`separate.py:264-266`).
  - At the default `DANCE_GPU_CONCURRENCY=1`, the second thread can only pre-claim a track
    and wait. That is why claim-to-done (197 s) is about twice the compute time (93 s). It is
    also why track 2 was a bystander in Experiment A.
  - One worker means one model in memory. Overlapping separations would take N workers and
    N model copies (decision 3).
- **Memory.**
  - **The MPS cache is not shared across processes.** The worker calls
    `torch.mps.empty_cache()` after each job so that CLAP in the parent can use the memory.
  - **Each process gets its own allocator cap.** `PYTORCH_MPS_HIGH_WATERMARK_RATIO` defaults
    to 1.7 ([PyTorch 2.9 docs](https://docs.pytorch.org/docs/2.9/mps_environment_variables.html)).
    The logged "max allowed: 18.13 GiB" on this 16 GB Mac is that cap. Giving the separator
    alone a lower ratio might turn swapping into a catchable OOM. That is a hypothesis.
  - **Measure peak memory** for the parent and the worker together in verification step 3.

### A separate, pinned environment (later)

- **The worker imports nothing from `dance`** and is launched by path with `-I`, so it runs
  on any interpreter that has its own dependencies.
- **`DANCE_SEPARATOR_PYTHON`** defaults to `sys.executable`, today's 3.13.3 venv, so step 1
  needs no new environment.
- **A RoFormer separator would bring its own locked project**, launched the way djbooth
  launches its helper: `uv run --python 3.12 --project <dir> --locked --no-dev python
  <worker>.py` (`index.ts:205-216`).
  - djbooth pins `requires-python = ">=3.12,<3.13"` and `mlx-audio-separator==0.1.5`
    (`stems-mlx/pyproject.toml:5-6`).
  - It also pins a 3.76 GB checkpoint by revision and sha256 (`djbooth_stems_mlx.py:21-27`).
  - Neither belongs in the torch venv. The frames above are the seam: a RoFormer worker
    answers the same protocol.
- **On this Mac:** `uv` is not installed. Homebrew `python3.11`, `3.12` and `3.13` are present.
- **Demucs itself is archived upstream** and has been read-only since 2025-01-01
  ([facebookresearch/demucs](https://github.com/facebookresearch/demucs)). A pinned env is
  also how it stays reproducible.

### Stage protocol: unchanged

- `StemSeparationStage` keeps its `name`, its states, `error_state`, `uses_gpu`, and
  `process(session, track, settings) -> bool` (`stage.py:65-82`). Only the *value* of
  `concurrency` changes. The dispatcher is untouched.
- **Stays in `process()` (parent):**
  - the idempotent short-circuit (`separate.py:369-375`)
  - the stale-row delete (`:422-429`)
  - `sweep_partials`, phase 2 of the write, the `StemFile` rows with provenance and
    residual, and the state change.
- **Moves to the worker:**
  - the decode (`:392-396`)
  - `_ensure_model` and the MPS→CPU retry (`:282-358`)
  - the residual and clipping QC (`:411-420`)
  - phase 1 of the write
- **Settings:**
  - `DANCE_SEPARATOR_MODE=inline` runs the same worker core in-process: today's behaviour.
    It is the rollback switch and what the existing mocked tests use.
  - Added: `separator_python`, `separator_mode` and `separator_timeout_scale`.

## Alternative: a fresh process per track (`multiprocessing`)

Experiment C, with the child importing `dance.pipeline.stages.separate` the way a real one
would:

```
ProcessPoolExecutor(max_workers=1, mp_context=spawn, max_tasks_per_child=1)
  a.wav        ok, child started 0.90 s after submit
  b-crash.wav  BrokenProcessPool: A process in the process pool was terminated abruptly …
  c.wav        BrokenProcessPool: A child process terminated abruptly, the process pool is not usable anymore
bare multiprocessing.Process per track (spawn)
  a.wav exitcode=0 1.16 s · b-crash.wav exitcode=-10 1.20 s · c.wav exitcode=0 1.09 s
```

- **Pool: rejected.**
  - It loses the signal, and one crash poisons every later submit until the pool is rebuilt.
  - `max_tasks_per_child` needs Python 3.11, but the repo targets py310.
  - 3.15 fixes a hang that can follow `max_tasks_per_child` (gh-115634,
    [docs](https://docs.python.org/3/library/concurrent.futures.html)).
- **A bare `Process` per track is a real option.**
  - **For:** about 40 lines, no framing, and the exit code survives.
  - **For:** with Demucs the overhead is small, about 1 s to spawn plus 1–3 s to load the
    model, against ~3 min of compute.
  - **Against:** a spawned child unpickles a function from `dance`, so it must run on an
    interpreter where `dance` and all its dependencies are installed. That rules out a
    pinned separator env.
  - **Against:** the model reloads on every track (djbooth added batch mode for its 3.76 GB
    checkpoint for this reason).
  - **Against:** it still needs a results channel, the same deadline and kill logic, and
    the same orphan watchdog.

## Recommendation

**Build the long-lived worker, in two steps.**

1. **Isolation, in the current venv.**
   - Add `demucs_worker.py` and `client.py`, and shrink `process()`.
   - Set `concurrency` to 1, retry once on CPU after a worker death, and use the deadlines
     above.
   - No new environment and no new dependency.
2. **A pinned environment, when a second separator lands.** Until then it costs tooling
   (`uv`) and fixes nothing.

If no second separator is planned, a bare `Process` per track buys the same isolation for
less code. Decision 1 is that trade.

## Decisions for the user

1. **Worker shape.**
   - **Long-lived worker (recommended):** keeps the model warm and allows a pinned env.
   - **Per-track `Process`:** simpler, but stays in the same venv.
2. **After a worker death.**
   - **Retry once on CPU (recommended):** costs one CPU separation for a track that really
     is bad.
   - **Retry once on MPS.**
   - **ERROR at once.**
3. **Concurrency.**
   - **One worker (recommended).**
   - **N workers when `DANCE_GPU_CONCURRENCY` ≥ 2:** N model copies in 16 GB.
4. **The `inline` mode:** keep it permanently, or remove it one release after the worker
   proves out.
5. **Tooling for the step 2 environment:** `uv` (djbooth's precedent; needs an install), or
   `python3.12 -m venv` plus a hash-pinned requirements file. This can wait for the
   RoFormer proposal.

## Migration cost

- **What does not change:**
  - no schema, Alembic, API, OSC or `.als` change, and no new dependency
  - the Stage protocol and the dispatcher
  - existing stems, `StemFile` rows and final paths
- **New `src/dance/separator/`:**
  - `protocol.py`: pydantic frames, parent side, about 60 lines
  - `client.py`: spawn, deadlines, retry, breaker, idle shutdown, watchdog-friendly launch;
    about 150 lines
  - `demucs_worker.py`: about 200 lines, mostly moved from `separate.py:104-358`
- **`separate.py`** shrinks by about 150 lines.
- **`config.py`** gets 3 settings.
- **Tests:**
  - `tests/test_separate.py` (2 tests) and the stage tests in `test_stems_hygiene.py` patch
    `sep.apply_model`, `sep.torch`, `sep.librosa` and `sep.sf`. They re-target the worker
    core in `inline` mode.
  - The pure-helper tests are unchanged.
- **Docs:** rewrite the SIGBUS section of `docs/troubleshooting.md`.
- **Behaviour changes:**
  - a native crash costs at most one track, not the run
  - `separate` runs one track at a time, which is neutral at the default
    `DANCE_GPU_CONCURRENCY=1`
  - a hang now ERRORs at the deadline
  - error texts change
- **Effort:** about one day of work, plus the real-data run.

## Verification plan

1. **CI, no torch: new `tests/test_separator_worker.py`.**
   - **Setup:** a stdlib-only fake worker that picks its behaviour from the audio file name,
     as in the scratch prototype:
     - `crash`: SIGBUS on any device
     - `mpsonly`: SIGBUS on MPS only
     - `hang`
     - `oom`: an `ok:false` reply
     - a stray stdout `print`
     - success
   - **Run:** through the real parallel `Dispatcher` with the `conftest.py` isolation.
   - **Assert:**
     - one pass
     - `crash` is ERROR naming both signals, and `mpsonly` is SEPARATED via CPU
     - `hang` is ERROR after its deadline, and `oom` is ERROR with the message
     - bystanders in `analyze` are untouched
     - no `heal_ledger.json`, and no stranded partials
     - the exact respawn count, and the breaker trips after 2 consecutive MPS deaths
     - killing the parent kills the worker within 1 s
   - **The scratch prototype already shows this** (Experiment B, real torch import, one
     pass, rc 0):

     ```
     separate: processed 5, errors 3 · spawns mps 0.73 s, cpu 0.65, mps 0.66, cpu 0.63, mps 0.70, mps 0.67
     track 1: separator worker died (signal SIGBUS on mps, signal SIGBUS on cpu)
     track 3 (mpsonly): separated on the CPU retry
     track 4: no answer within 3s; worker killed
     track 5: RuntimeError: MPS backend out of memory (fake)
     final stems for tracks 2, 3, 6, 7, 8 · stranded partials: [] · heal_ledger: (none)
     ```
2. **Contract test.** Run the real `demucs_worker.py --probe` (imports plus the `ready`
   frame, no weights) and check the frame schema. Skip it if demucs is absent. `dance
   doctor` gets the same probe as a check next to `check_demucs_weights` (`doctor.py:1243`).
3. **Real data (rule 2), before calling it done.**
   - **Setup:** temp `DANCE_DATA_DIR`, `DANCE_LIBRARY_DIR` and `DANCE_STEMS_DIR`, with 3 real
     tracks *copied* in. Run `dance process --track-id <id>` on MPS.
   - **Expect a normal run:**
     - one `Loading Demucs model` per run
     - compute within the shifts-adjusted range above
     - `residual_db` within 0.5 dB of the in-process value for the same track (`shifts`
       randomises, so the output is not bit-identical)
   - **Crash, with no code hook:** run `kill -BUS $(pgrep -f demucs_worker.py)` mid-job.
     Expect the CPU retry and `rc 0`.
   - **Crash twice:** kill both attempts. Expect only that track ERROR, and the next track
     separating in the same run.
   - **Record:**
     - the CPU/MPS time ratio, to replace the ×4 guess
     - peak memory for the parent and the worker during `embed` plus `separate`
4. **No Live check is needed:** stem paths and the `.als` output do not change.

## Follow-ups (not in this proposal)

- **The MPS→CPU retry catches only `RuntimeError`** (`separate.py:347`).
  - Tracks 235 and 236 died on `assert sum_weight.min() > 0` (`demucs/apply.py:255`), right
    after `kIOGPUCommandBufferCallbackErrorOutOfMemory`; one run logged that line 131 times.
  - They sit in ERROR with the message `"separate: "`, because `str(AssertionError())` is
    empty (`dispatcher.py:468`).
- **The heal ledger's message blames SIGBUS and Demucs** for any stage and for any parent
  death (`dispatcher.py:602-607`). See track 206.
- **The API's 1 h cap per pass SIGKILLs healthy runs.** Job `57c01…` finished 21
  separations in its hour at `shifts=1`. With per-job deadlines, the cap could become a
  no-progress timeout.

Evidence scripts (scratch, not committed): `exp_a.py`, `exp_b.py`, `exp_b2.py`, `exp_c.py`,
`parse_logs.py`, `rtf.py`. They ran against the worktree's `src` with an isolated HOME.
