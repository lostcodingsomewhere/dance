import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ingestSpotifyTrack,
  pushTrackToLive,
  recommendByText,
  revealPath,
  searchTracks,
  serverMessage,
  spotifySearch,
} from "../api";
import { useActiveSet } from "../hooks/useSets";
import { useWarpCheck } from "../hooks/useWarpCheck";
import { useAppendToPlan, useSetPlan } from "../hooks/useSetPlan";
import { formatDuration, formatDurationMs } from "../lib/format";
import { sideTrackIndices } from "../lib/roles";
import { store, useAppStore } from "../store";
import type { Recommendation, SpotifyTrackHit, Track } from "../types";
import { BpmRangePicker } from "./BpmRangePicker";
import { EnergyBar } from "./EnergyBar";
import { KeyBadge } from "./KeyBadge";

/**
 * Global Cmd-K palette — hybrid find + vibe.
 *
 * Section 1 (top): fuzzy name/artist via ``/tracks/search``. The "I know
 * what I want" half. Top 8.
 * Section 2 (bottom): CLAP vibe via ``/recommend/text``. The "find me
 * something that feels like X" half. Top 12. Auto-hidden if the query is
 * too short to be a useful vibe prompt (< 8 chars).
 *
 * Filter chips below the input narrow both sections (BPM range, key,
 * energy). Empty query shows recent tracks for casual browsing.
 *
 * Result actions: Load to Live + Add to the active set's plan.
 *
 * Keys: Down and Up move a highlight through every row in the order shown.
 * Nothing is highlighted until Down is pressed, and typing clears it; the
 * arrows then wait until the rows on screen are the ones for the new text. Enter
 * on the highlighted library row adds it to the plan in the Set view and
 * loads it to Live in every other view; Cmd+Enter adds it to the plan in
 * every view. Enter on a highlighted Spotify row presses that row's button.
 * Enter with nothing highlighted does nothing.
 */
export function CommandBar() {
  const open = useAppStore((s) => s.commandBarOpen);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  const [highlightedRow, setHighlightedRow] = useState<string | null>(null);

  const [bpmMin, setBpmMin] = useState<number | null>(null);
  const [bpmMax, setBpmMax] = useState<number | null>(null);
  const [keyFilter, setKeyFilter] = useState<string | null>(null);

  const active = useActiveSet();
  const activeSet = active.data;

  // Debounce text input — 200 ms is enough to make typing feel live but
  // avoids hammering the server while the user is mid-word.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(q), 200);
    return () => clearTimeout(t);
  }, [q]);

  // Fuzzy name/artist search — runs whenever the debounced query is
  // non-empty, OR when the filters change but the input is empty (browse
  // mode with chips active).
  const fuzzy = useQuery({
    queryKey: [
      "tracks",
      "search",
      debouncedQ.trim().toLowerCase(),
      bpmMin,
      bpmMax,
      keyFilter,
    ],
    queryFn: () =>
      searchTracks({
        q: debouncedQ.trim(),
        limit: 8,
        bpm_min: bpmMin ?? undefined,
        bpm_max: bpmMax ?? undefined,
        key: keyFilter ?? undefined,
      }),
    enabled: open,
    staleTime: 5_000,
  });

  // Vibe search — only auto-runs once the query is long enough to feel
  // like a phrase, not a fragment of an artist name.
  const vibeEligible = debouncedQ.trim().length >= 8;
  const vibe = useQuery({
    queryKey: ["recommend", "text", debouncedQ.trim().toLowerCase()],
    queryFn: () => recommendByText(debouncedQ.trim(), 12),
    enabled: open && vibeEligible,
    staleTime: 10_000,
  });

  // Spotify catalog search — surfaces the "not in your library yet?" path.
  // Same eligibility threshold as vibe so we don't spam Spotify on
  // single-character queries.
  const spotifyEligible = debouncedQ.trim().length >= 3;
  const spotify = useQuery({
    queryKey: ["spotify", "search", debouncedQ.trim().toLowerCase()],
    queryFn: () => spotifySearch(debouncedQ.trim(), 5),
    enabled: open && spotifyEligible,
    staleTime: 30_000,
    retry: false, // 503 means "no creds" — don't pound the endpoint
  });

  const rowsInOrder = useMemo(
    () => [
      ...(fuzzy.data ?? []).map(libraryRowKey),
      ...(vibeEligible ? (vibe.data ?? []).map(vibeRowKey) : []),
      ...(spotifyEligible ? (spotify.data?.hits ?? []).map(spotifyRowKey) : []),
    ],
    [fuzzy.data, vibe.data, spotify.data, vibeEligible, spotifyEligible],
  );
  const rowsAreForTheTypedText =
    q.trim().toLowerCase() === debouncedQ.trim().toLowerCase();

  // Global key listener — Cmd/Ctrl+K toggles, Escape closes, Down and Up
  // move the highlight.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (open) store.closeCommandBar();
        else store.openCommandBar();
      }
      if (open && e.key === "Escape") {
        store.closeCommandBar();
      }
      const step = open ? stepOfArrowKey(e) : null;
      if (step == null || usesArrowKeysItself(e.target)) return;
      e.preventDefault();
      inputRef.current?.focus();
      if (!rowsAreForTheTypedText) return;
      setHighlightedRow((current) => rowAfter(rowsInOrder, current, step));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, rowsInOrder, rowsAreForTheTypedText]);

  // Focus on open / reset on close.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!open) {
      setQ("");
      setHighlightedRow(null);
      setBpmMin(null);
      setBpmMax(null);
      setKeyFilter(null);
      return;
    }
    const t = setTimeout(() => inputRef.current?.focus(), 30);
    return () => clearTimeout(t);
  }, [open]);

  const fuzzyHits = fuzzy.data ?? [];
  const vibeHits = vibe.data ?? [];
  const showVibeSection = vibeEligible;
  const empty =
    !fuzzy.isLoading &&
    !vibe.isLoading &&
    fuzzyHits.length === 0 &&
    (!showVibeSection || vibeHits.length === 0);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/70 backdrop-blur-sm"
      onClick={() => store.closeCommandBar()}
      role="presentation"
    >
      <div
        className="w-full max-w-2xl rounded-xl border border-neutral-800 bg-neutral-950 shadow-2xl overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Find & vibe search"
      >
        <div className="flex items-center gap-2 px-4 py-3 border-b border-neutral-800">
          <span className="text-purple-400 text-xl" aria-hidden>
            ✦
          </span>
          <input
            ref={inputRef}
            id={SEARCH_FIELD_ID}
            type="text"
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setHighlightedRow(null);
            }}
            placeholder="Type an artist, title — or a vibe like 'deep rolling bass'"
            className="flex-1 bg-transparent text-lg text-neutral-100 placeholder:text-neutral-600 outline-none"
          />
          <kbd className="font-mono text-[10px] text-neutral-500 border border-neutral-800 rounded px-1">
            ESC
          </kbd>
        </div>

        <FilterChips
          bpmMin={bpmMin}
          bpmMax={bpmMax}
          keyFilter={keyFilter}
          onBpmMin={setBpmMin}
          onBpmMax={setBpmMax}
          onKey={setKeyFilter}
        />

        {/* Some browsers drop key presses while a button that went disabled holds the focus. */}
        <div
          className="max-h-[60vh] overflow-y-auto"
          onClick={() => takeFocusBackFromAButton(inputRef.current)}
        >
          {fuzzy.isError && (
            <SectionError msg={(fuzzy.error as Error).message} />
          )}
          <Section
            title={debouncedQ.trim() ? "Tracks" : "Recent"}
            count={fuzzyHits.length}
            loading={fuzzy.isLoading}
          >
            {fuzzyHits.map((t, place) => (
              <TrackRow
                key={libraryRowKey(t)}
                track={t}
                activeSetId={activeSet?.id ?? null}
                versionMark={versionMark(fuzzyHits, place)}
                highlighted={highlightedRow === libraryRowKey(t)}
              />
            ))}
          </Section>

          {showVibeSection && (
            <>
              {vibe.isError && (
                <SectionError msg={(vibe.error as Error).message} />
              )}
              <Section
                title="Vibe"
                count={vibeHits.length}
                loading={vibe.isLoading}
              >
                {vibeHits.map((r) => (
                  <RecRow
                    key={vibeRowKey(r)}
                    rec={r}
                    activeSetId={activeSet?.id ?? null}
                    highlighted={highlightedRow === vibeRowKey(r)}
                  />
                ))}
              </Section>
            </>
          )}

          {spotifyEligible && (
            <Section
              title="Add from Spotify"
              count={spotify.data?.hits.length ?? 0}
              loading={spotify.isLoading}
            >
              {spotify.data?.hits.map((hit) => (
                <SpotifyHitRow
                  key={spotifyRowKey(hit)}
                  hit={hit}
                  activeSetId={activeSet?.id ?? null}
                  highlighted={highlightedRow === spotifyRowKey(hit)}
                />
              ))}
            </Section>
          )}

          {empty && (
            <div className="px-4 py-6 text-sm text-neutral-500">
              No matches.{" "}
              {!vibeEligible && debouncedQ.trim() && (
                <span className="text-neutral-600">
                  Type a few more chars for vibe search.
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const SEARCH_FIELD_ID = "command-bar-search";
const HIGHLIGHT = "bg-neutral-900 ring-2 ring-inset ring-amber-300/80";

function libraryRowKey(track: Track): string {
  return `fuzzy-${track.id}`;
}

function vibeRowKey(rec: Recommendation): string {
  return `vibe-${rec.track_id}`;
}

function spotifyRowKey(hit: SpotifyTrackHit): string {
  return `spot-${hit.spotify_id}`;
}

function stepOfArrowKey(e: KeyboardEvent): 1 | -1 | null {
  if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return null;
  if (e.key === "ArrowDown") return 1;
  if (e.key === "ArrowUp") return -1;
  return null;
}

function rowAfter(rows: string[], current: string | null, step: 1 | -1): string | null {
  const place = current == null ? -1 : rows.indexOf(current);
  if (place === -1) return step === 1 ? rows[0] ?? null : null;
  return rows[Math.min(rows.length - 1, Math.max(0, place + step))];
}

function tagOf(target: EventTarget | null): string {
  return target instanceof Element ? target.tagName.toLowerCase() : "";
}

function usesArrowKeysItself(target: EventTarget | null): boolean {
  if (target instanceof Element && target.id === SEARCH_FIELD_ID) return false;
  return ["input", "select", "textarea"].includes(tagOf(target));
}

function usesEnterItself(target: EventTarget | null): boolean {
  return usesArrowKeysItself(target) || ["button", "a"].includes(tagOf(target));
}

function takeFocusBackFromAButton(searchField: HTMLInputElement | null): void {
  if (document.activeElement instanceof HTMLButtonElement) searchField?.focus();
}

interface RowKeys {
  enter: () => void;
  commandEnter: () => void;
}

function useHighlightedRow(highlighted: boolean, { enter, commandEnter }: RowKeys) {
  const row = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (highlighted) row.current?.scrollIntoView?.({ block: "nearest" });
  }, [highlighted]);

  useEffect(() => {
    if (!highlighted) return;
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Enter" || e.repeat || e.shiftKey || e.altKey) return;
      if (usesEnterItself(e.target)) return;
      e.preventDefault();
      if (e.metaKey || e.ctrlKey) commandEnter();
      else enter();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [highlighted, enter, commandEnter]);

  return row;
}

function EnterHint() {
  return (
    <kbd
      aria-hidden
      className="ml-1.5 rounded border border-current px-1 font-mono text-[10px] opacity-80"
    >
      ⏎
    </kbd>
  );
}

function FilterChips({
  bpmMin,
  bpmMax,
  keyFilter,
  onBpmMin,
  onBpmMax,
  onKey,
}: {
  bpmMin: number | null;
  bpmMax: number | null;
  keyFilter: string | null;
  onBpmMin: (v: number | null) => void;
  onBpmMax: (v: number | null) => void;
  onKey: (v: string | null) => void;
}) {
  const anyActive = bpmMin != null || bpmMax != null || keyFilter != null;
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border-b border-neutral-900 text-[11px]">
      <BpmRangePicker
        min={bpmMin}
        max={bpmMax}
        onChange={({ min, max }) => {
          onBpmMin(min);
          onBpmMax(max);
        }}
      />
      <KeyChip value={keyFilter} onChange={onKey} />
      {anyActive && (
        <button
          type="button"
          onClick={() => {
            onBpmMin(null);
            onBpmMax(null);
            onKey(null);
          }}
          className="ml-auto text-neutral-500 hover:text-neutral-200"
          title="Clear all filters"
        >
          × clear all
        </button>
      )}
    </div>
  );
}

const CAMELOT_KEYS = [
  "1A",
  "2A",
  "3A",
  "4A",
  "5A",
  "6A",
  "7A",
  "8A",
  "9A",
  "10A",
  "11A",
  "12A",
  "1B",
  "2B",
  "3B",
  "4B",
  "5B",
  "6B",
  "7B",
  "8B",
  "9B",
  "10B",
  "11B",
  "12B",
];

function KeyChip({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (v: string | null) => void;
}) {
  return (
    <label className="inline-flex items-center gap-1 text-neutral-500">
      <span>Key</span>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
        className="bg-neutral-900 border border-neutral-800 rounded px-1.5 py-0.5 text-neutral-100 outline-none focus:border-neutral-700"
      >
        <option value="">any</option>
        {CAMELOT_KEYS.map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>
    </label>
  );
}

function Section({
  title,
  count,
  loading,
  children,
}: {
  title: string;
  count: number;
  loading: boolean;
  children: React.ReactNode;
}) {
  if (!loading && count === 0) return null;
  return (
    <div className="py-1">
      <div className="px-3 py-1 flex items-center gap-2 text-[10px] uppercase tracking-wider text-neutral-500">
        <span>{title}</span>
        {loading && <span className="text-neutral-600">…</span>}
        <span className="flex-1 border-b border-neutral-900" />
        {!loading && count > 0 && (
          <span className="text-neutral-600">{count}</span>
        )}
      </div>
      <div className="px-2 py-1 flex flex-col gap-0.5">{children}</div>
    </div>
  );
}

function SectionError({ msg }: { msg: string }) {
  return <div className="m-2 text-sm text-rose-300">{msg}</div>;
}

type VersionMark = { text: string; underTheRowAbove: boolean } | null;

function versionMark(hits: Track[], place: number): VersionMark {
  const family = hits[place].version_family ?? null;
  if (family == null) return null;
  if (place > 0 && hits[place - 1].version_family === family) {
    return { text: "version", underTheRowAbove: true };
  }
  const together = hits.filter((hit) => hit.version_family === family).length;
  return together > 1 ? { text: `${together} versions`, underTheRowAbove: false } : null;
}

const READY = "complete";
const FAILED = "error";
const NOT_STARTED = "pending";

interface NotReady {
  inWords: string;
  whatToDo: string;
}

function whyItCannotBeLoaded(state: string): NotReady | null {
  if (state === READY) return null;
  if (state === FAILED) {
    return {
      inWords: "failed, see Pipeline",
      whatToDo: "The download or the processing failed. The Pipeline view says why.",
    };
  }
  if (state === NOT_STARTED) {
    return {
      inWords: "not processed yet",
      whatToDo: "Still downloading, or waiting for Process now in the Pipeline view.",
    };
  }
  return { inWords: "processing", whatToDo: "It can be loaded once processing has finished." };
}

function planHelp(activeSetId: number | null, inPlan: boolean): string {
  if (activeSetId == null) return "No active set — make one in the Set view first";
  if (inPlan) return "Already in the plan's Song column";
  return "Add to the active set's plan, Song column (⌘⏎ on the highlighted row)";
}

function TrackRow({
  track,
  activeSetId,
  versionMark,
  highlighted,
}: {
  track: Track;
  activeSetId: number | null;
  versionMark: VersionMark;
  highlighted: boolean;
}) {
  const analysis = track.analysis;
  return (
    <RowShell
      versionMark={versionMark}
      state={track.state}
      highlighted={highlighted}
      title={track.title ?? `Track #${track.id}`}
      artist={track.artist}
      bpm={analysis?.bpm ?? null}
      keyCamelot={analysis?.key_camelot ?? null}
      floorEnergy={analysis?.floor_energy ?? null}
      durationSeconds={track.duration_seconds}
      score={null}
      trackId={track.id}
      filePath={track.file_path}
      activeSetId={activeSetId}
    />
  );
}

function RecRow({
  rec,
  activeSetId,
  highlighted,
}: {
  rec: Recommendation;
  activeSetId: number | null;
  highlighted: boolean;
}) {
  return (
    <RowShell
      highlighted={highlighted}
      title={rec.title ?? `Track #${rec.track_id}`}
      artist={rec.artist}
      bpm={rec.bpm}
      keyCamelot={rec.key_camelot}
      floorEnergy={rec.floor_energy}
      durationSeconds={null}
      score={rec.score}
      trackId={rec.track_id}
      filePath={rec.file_path}
      activeSetId={activeSetId}
    />
  );
}

function RowShell({
  title,
  artist,
  bpm,
  keyCamelot,
  floorEnergy,
  durationSeconds,
  score,
  trackId,
  filePath,
  activeSetId,
  highlighted,
  state = READY,
  versionMark = null,
}: {
  title: string;
  artist: string | null;
  bpm: number | null;
  keyCamelot: string | null;
  floorEnergy: number | null;
  durationSeconds: number | null;
  score: number | null;
  trackId: number;
  filePath: string | null;
  activeSetId: number | null;
  highlighted: boolean;
  state?: string;
  versionMark?: VersionMark;
}) {
  const addToPlan = useAppendToPlan();
  const scheduleWarpCheck = useWarpCheck();
  const plan = useSetPlan(activeSetId);
  const enterAddsToPlan = useAppStore((s) => s.currentView === "set");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const inPlan = (plan.data?.queues.song ?? []).some((pick) => pick.track_id === trackId);
  const canPlan = activeSetId != null && !inPlan && !addToPlan.isPending;
  const notReady = whyItCannotBeLoaded(state);
  const canLoad = notReady == null && !loading && done == null;

  async function loadToLive() {
    if (!canLoad) return;
    try {
      setLoading(true);
      setFailure(null);
      const r = await pushTrackToLive(trackId, {
        includeStems: true,
        page: store.peekBoothPage(),
      });
      store.registerDeck({
        track_id: trackId,
        scene_index: r.scene_index,
        side: r.side ?? null,
        stem_track_indices: sideTrackIndices(r.track_indices, r.side),
        loaded_at: Date.now(),
      });
      const fully = r.stems_loaded === 4;
      setDone(fully ? `scene ${r.scene_index + 1}` : `${r.stems_loaded}/4`);
      // The palette closes itself half a second from now, so anything worth
      // reading (warp check, missing stems) has to outlive it — the banner
      // in the app shell does that.
      const label = `${title} → scene ${r.scene_index + 1}`;
      store.setLoadWarnings(label, r.warnings);
      scheduleWarpCheck(r.scene_index, label);
      store.armDeck(r.side, r.scene_index);
      if (!fully && filePath) {
        try {
          await revealPath(filePath);
        } catch {
          // best-effort
        }
      }
      setTimeout(() => store.closeCommandBar(), 500);
    } catch (e) {
      setFailure(serverMessage(e as Error));
    } finally {
      setLoading(false);
    }
  }

  function addToActiveSet() {
    if (activeSetId == null || !canPlan) return;
    setFailure(null);
    addToPlan.mutate(
      { setId: activeSetId, role: "song", trackId },
      { onError: (error) => setFailure(serverMessage(error)) },
    );
  }

  const row = useHighlightedRow(highlighted, {
    enter: enterAddsToPlan ? addToActiveSet : loadToLive,
    commandEnter: addToActiveSet,
  });

  return (
    <div
      ref={row}
      role="group"
      aria-label={`Search result: ${title}`}
      aria-current={highlighted ? "true" : undefined}
      className={`flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-neutral-900/60 ${
        highlighted ? HIGHLIGHT : ""
      } ${versionMark?.underTheRowAbove ? "ml-5 border-l border-fuchsia-500/40" : ""}`}
    >
      <KeyBadge keyCamelot={keyCamelot ?? null} size="sm" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5 text-sm font-semibold text-neutral-100">
          <span className="truncate">{title}</span>
          {versionMark && (
            <span className="shrink-0 rounded bg-fuchsia-500/25 px-1 text-[10px] font-semibold uppercase tracking-wide text-fuchsia-200/90">
              {versionMark.text}
            </span>
          )}
        </div>
        <div className="text-xs text-neutral-500 truncate">
          {artist ?? "Unknown"}
          {durationSeconds != null && (
            <span className="ml-2 font-mono text-neutral-600" title="Track duration">
              {formatDuration(durationSeconds)}
            </span>
          )}
          {bpm != null && (
            <span className="ml-2 font-mono">{bpm.toFixed(1)} BPM</span>
          )}
        </div>
      </div>
      <EnergyBar energy={floorEnergy ?? null} size="sm" />
      {score != null && (
        <span className="font-mono text-[10px] text-purple-300 w-10 text-right">
          ✦ {score.toFixed(2)}
        </span>
      )}
      {failure && (
        <span
          role="alert"
          title={failure}
          className="max-w-[14rem] text-[11px] leading-tight text-rose-300 line-clamp-2"
        >
          {failure}
        </span>
      )}
      <button
        type="button"
        onClick={addToActiveSet}
        disabled={!canPlan}
        title={planHelp(activeSetId, inPlan)}
        className="min-h-[32px] px-2 rounded-md text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200 disabled:opacity-40"
      >
        {inPlan ? "✓ in plan" : "+ Plan"}
        {highlighted && enterAddsToPlan && canPlan && <EnterHint />}
      </button>
      {notReady ? (
        <span
          title={notReady.whatToDo}
          className={`shrink-0 px-1 text-[11px] ${
            state === FAILED ? "text-rose-300" : "text-neutral-500"
          }`}
        >
          {notReady.inWords}
        </span>
      ) : (
        <button
          type="button"
          onClick={loadToLive}
          disabled={!canLoad}
          className="min-h-[32px] px-3 rounded-md text-xs font-semibold bg-purple-700 hover:bg-purple-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-white"
        >
          {loading ? "…" : done ?? "Load"}
          {highlighted && !enterAddsToPlan && canLoad && <EnterHint />}
        </button>
      )}
    </div>
  );
}

function SpotifyHitRow({
  hit,
  activeSetId,
  highlighted,
}: {
  hit: SpotifyTrackHit;
  activeSetId: number | null;
  highlighted: boolean;
}) {
  const qc = useQueryClient();
  const addToPlan = useAppendToPlan();
  const [state, setState] = useState<"idle" | "pending" | "done" | "error">(
    "idle",
  );
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const canRun = state !== "pending" && state !== "done";

  async function ingest() {
    if (!canRun) return;
    setState("pending");
    setErrMsg(null);
    try {
      const result = await ingestSpotifyTrack(hit);
      if (activeSetId) {
        await addToPlan.mutateAsync({
          setId: activeSetId,
          role: "song",
          trackId: result.track_id,
        });
      }
      // Invalidate everything that might surface the new track — active
      // set re-renders with a ⌛ chip, pipeline status counts shift.
      qc.invalidateQueries({ queryKey: ["sets"] });
      qc.invalidateQueries({ queryKey: ["pipeline"] });
      setState("done");
    } catch (e) {
      setState("error");
      setErrMsg((e as Error).message.slice(0, 80));
    }
  }

  const durMin = hit.duration_ms != null ? formatDurationMs(hit.duration_ms) : null;
  const row = useHighlightedRow(highlighted, { enter: ingest, commandEnter: ingest });

  return (
    <div
      ref={row}
      role="group"
      aria-label={`Spotify result: ${hit.title}`}
      aria-current={highlighted ? "true" : undefined}
      className={`flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-neutral-900/60 ${
        highlighted ? HIGHLIGHT : ""
      }`}
    >
      {hit.image_url ? (
        <img
          src={hit.image_url}
          alt=""
          className="w-9 h-9 rounded shrink-0 object-cover"
        />
      ) : (
        <div className="w-9 h-9 rounded shrink-0 bg-neutral-800 flex items-center justify-center text-[10px] text-neutral-600">
          ♪
        </div>
      )}
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-neutral-100 truncate flex items-center gap-1.5">
          {hit.title}
          {hit.explicit && (
            <span className="text-[8px] px-1 py-0.5 rounded bg-neutral-800 text-neutral-500">
              E
            </span>
          )}
        </div>
        <div className="text-xs text-neutral-500 truncate">
          {hit.artist}
          {hit.album && <span className="ml-1.5">· {hit.album}</span>}
          {durMin && <span className="ml-1.5 font-mono">{durMin}</span>}
        </div>
      </div>
      {state === "error" && errMsg && (
        <span className="text-[10px] text-rose-300 truncate max-w-[10rem]" title={errMsg}>
          {errMsg}
        </span>
      )}
      <button
        type="button"
        onClick={ingest}
        disabled={!canRun}
        title={
          activeSetId
            ? "Download + analyze + add to the plan (Song column)"
            : "Download + analyze (no active set to add to)"
        }
        className="min-h-[32px] px-3 rounded-md text-xs font-semibold bg-emerald-700/80 hover:bg-emerald-600 text-white disabled:bg-neutral-800 disabled:text-neutral-500"
      >
        {state === "pending" && "⌛ adding…"}
        {state === "done" && "✓ added"}
        {state === "error" && "↻ retry"}
        {state === "idle" && (activeSetId ? "+ Plan" : "+ Library")}
        {highlighted && canRun && <EnterHint />}
      </button>
    </div>
  );
}
