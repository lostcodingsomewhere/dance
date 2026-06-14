import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ingestSpotifyTrack,
  pushTrackToLive,
  recommendByText,
  revealPath,
  searchTracks,
  spotifySearch,
} from "../api";
import { useActiveSet } from "../hooks/useSets";
import { useAppendToPlan } from "../hooks/useSetPlan";
import { formatDuration, formatDurationMs } from "../lib/format";
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
 * Result actions: Load to Live (primary) + Add to active set (secondary).
 * Cmd+Enter on a row adds it to the set; Enter loads. Enter alone in the
 * input runs vibe.
 */
export function CommandBar() {
  const open = useAppStore((s) => s.commandBarOpen);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");

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

  // Global hotkey listener — Cmd/Ctrl+K toggles.
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
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Focus on open / reset on close.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!open) {
      setQ("");
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
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
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

        <div className="max-h-[60vh] overflow-y-auto">
          {fuzzy.isError && (
            <SectionError msg={(fuzzy.error as Error).message} />
          )}
          <Section
            title={debouncedQ.trim() ? "Tracks" : "Recent"}
            count={fuzzyHits.length}
            loading={fuzzy.isLoading}
          >
            {fuzzyHits.map((t) => (
              <TrackRow
                key={`fuzzy-${t.id}`}
                track={t}
                activeSetId={activeSet?.id ?? null}
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
                    key={`vibe-${r.track_id}`}
                    rec={r}
                    activeSetId={activeSet?.id ?? null}
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
                  key={`spot-${hit.spotify_id}`}
                  hit={hit}
                  activeSetId={activeSet?.id ?? null}
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

function TrackRow({
  track,
  activeSetId,
}: {
  track: Track;
  activeSetId: number | null;
}) {
  const analysis = track.analysis;
  return (
    <RowShell
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
}: {
  rec: Recommendation;
  activeSetId: number | null;
}) {
  return (
    <RowShell
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
}) {
  const addToPlan = useAppendToPlan();
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState<string | null>(null);

  const inActiveSet = useMemo(() => {
    return false;
  }, []);

  async function loadToLive() {
    try {
      setLoading(true);
      const sceneIndex = store.peekNextScene();
      const r = await pushTrackToLive(trackId, {
        includeStems: true,
        sceneIndex,
      });
      store.registerDeck({
        track_id: trackId,
        scene_index: r.scene_index,
        stem_track_indices: Object.values(r.track_indices),
        loaded_at: Date.now(),
      });
      const fully = r.stems_loaded === 4;
      setDone(fully ? `scene ${r.scene_index + 1}` : `${r.stems_loaded}/4`);
      if (!fully && filePath) {
        try {
          await revealPath(filePath);
        } catch {
          // best-effort
        }
      }
      setTimeout(() => store.closeCommandBar(), 500);
    } catch (e) {
      setDone(`! ${(e as Error).message.slice(0, 40)}`);
    } finally {
      setLoading(false);
    }
  }

  function addToActiveSet() {
    if (!activeSetId) return;
    addToPlan.mutate({ setId: activeSetId, role: "song", trackId });
  }

  return (
    <div className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-neutral-900/60">
      <KeyBadge keyCamelot={keyCamelot ?? null} size="sm" />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-semibold text-neutral-100 truncate">
          {title}
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
      <button
        type="button"
        onClick={addToActiveSet}
        disabled={!activeSetId || addToPlan.isPending || inActiveSet}
        title={
          activeSetId
            ? "Add to the active set's plan — Song column (⌘⏎)"
            : "No active set — make one in the Set view first"
        }
        className="min-h-[32px] px-2 rounded-md text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200 disabled:opacity-40"
      >
        + Plan
      </button>
      <button
        type="button"
        onClick={loadToLive}
        disabled={loading}
        className="min-h-[32px] px-3 rounded-md text-xs font-semibold bg-purple-700 hover:bg-purple-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-white"
      >
        {loading ? "…" : done ?? "Load"}
      </button>
    </div>
  );
}

function SpotifyHitRow({
  hit,
  activeSetId,
}: {
  hit: SpotifyTrackHit;
  activeSetId: number | null;
}) {
  const qc = useQueryClient();
  const addToPlan = useAppendToPlan();
  const [state, setState] = useState<"idle" | "pending" | "done" | "error">(
    "idle",
  );
  const [errMsg, setErrMsg] = useState<string | null>(null);

  async function ingest() {
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

  return (
    <div className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-neutral-900/60">
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
        disabled={state === "pending" || state === "done"}
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
      </button>
    </div>
  );
}
