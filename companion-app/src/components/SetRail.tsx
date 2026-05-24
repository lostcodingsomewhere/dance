/**
 * Set Rail — slide-out sidebar showing the active Set + tail-recs.
 *
 * Three modes:
 *   - Closed (default in Booth): a narrow tab on the right edge with the
 *     set name + count. Click or ⌘\ to open. Auto-collapses 3 s after a
 *     clip fires so the SceneGrid stays sovereign during a mix.
 *   - Drawer (open in Booth): 320 px right sidebar. Track list with
 *     up/down nudge + remove; tail-recs section at the bottom; "⊕ from
 *     ⌘K" footer to add more.
 *   - Expanded (Set editor view): the SetEditor route renders this same
 *     data full-width — see views/SetEditor.tsx.
 *
 * Interaction model (locked decisions from the proposal):
 *   - Tap a rail track → soft-pin to Mix-column recs (reversible — same
 *     as today's pinnedSongRecs ↗ gesture, just sourced from the rail).
 *   - Shift-tap → force-load into the next empty stem column (push to
 *     Live now).
 *   - Tap a tail-rec → add to active set.
 */

import { useEffect, useMemo, useRef } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useCurrentSession } from "../hooks/useSession";
import { pushTrackToLive } from "../api";
import { useMutation } from "@tanstack/react-query";
import {
  useActiveSet,
  useActivateSet,
  useAddTrackToSet,
  useCreateSet,
  useMoveTrackInSet,
  useRemoveTrackFromSet,
  useTailRecs,
} from "../hooks/useSets";
import { store, useAppStore } from "../store";
import type { ColumnRec, DanceSet, SetTrack, TailRec } from "../types";
import { KeyBadge } from "./KeyBadge";
import { SetMenu } from "./SetMenu";
import { StemKindsChip } from "./StemKindsChip";

const RAIL_WIDTH = 320;
const AUTO_COLLAPSE_MS = 3_000;

// TrackState values that mean "not playable yet." Everything else (including
// COMPLETE) is treated as ready.
const PROCESSING_STATES = new Set([
  "pending",
  "analyzing",
  "analyzed",
  "separating",
  "separated",
  "analyzing_stems",
  "stems_analyzed",
  "detecting_regions",
  "regions_detected",
  "embedding",
  "embedded",
]);

function isProcessing(t: SetTrack): boolean {
  return t.track_state != null && PROCESSING_STATES.has(t.track_state);
}

function StateChip({
  state,
  error,
}: {
  state: string | null;
  error: string | null;
}) {
  if (state == null || state === "complete") return null;
  if (state === "error") {
    return (
      <span
        className="text-[9px] px-1 py-0.5 rounded bg-rose-500/20 text-rose-200 border border-rose-500/40"
        title={error ?? "Failed"}
      >
        ⚠ failed
      </span>
    );
  }
  // Use a short label per stage; default to ⌛ for anything else.
  const label =
    state === "pending"
      ? "downloading"
      : state === "separating"
        ? "separating stems"
        : state === "embedding"
          ? "embedding"
          : state.replace(/_/g, " ");
  return (
    <span
      className="text-[9px] px-1 py-0.5 rounded bg-amber-500/15 text-amber-200 border border-amber-500/30 animate-pulse"
      title={`Pipeline: ${state}`}
    >
      ⌛ {label}
    </span>
  );
}

export function SetRail() {
  const open = useAppStore((s) => s.setRailOpen);
  const view = useAppStore((s) => s.currentView);
  const active = useActiveSet();
  const set = active.data ?? null;

  // ⌘\ global toggle. ESC closes.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
        e.preventDefault();
        store.toggleSetRail();
      }
      if (open && e.key === "Escape") {
        store.closeSetRail();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Auto-collapse 3 s after a clip fires (in Booth only). The rail is a
  // resource the DJ pulls on between mixes, not during one. Off-plan
  // behavior (below) can override this by reopening immediately.
  const playingClips = useAbletonState().playing_clips ?? {};
  const playingFingerprint = useMemo(
    () =>
      Object.entries(playingClips)
        .map(([t, s]) => `${t}:${s}`)
        .sort()
        .join("|"),
    [playingClips],
  );
  const prevFp = useRef(playingFingerprint);
  useEffect(() => {
    if (!open || view !== "booth") return;
    if (prevFp.current === playingFingerprint) return;
    prevFp.current = playingFingerprint;
    const t = setTimeout(() => store.closeSetRail(), AUTO_COLLAPSE_MS);
    return () => clearTimeout(t);
  }, [playingFingerprint, open, view]);

  // Plan-aware rail behavior. When a new play is auto-logged in the
  // session, compare it to the next-planned set track:
  //   - off-plan (DJ deviated) → openSetRail() immediately, so the plan
  //     reasserts as a reorientation cue
  //   - on-plan → if rail was open, schedule a soft collapse; the rail
  //     "breathes out" once you're back on track
  const session = useCurrentSession();
  const plays = session.data?.plays ?? [];
  const prevPlayCount = useRef(plays.length);
  useEffect(() => {
    if (view !== "booth" || !set) return;
    if (plays.length <= prevPlayCount.current) {
      prevPlayCount.current = plays.length;
      return;
    }
    prevPlayCount.current = plays.length;

    const lastPlay = plays[plays.length - 1];
    const setIds = new Set(set.tracks.map((t) => t.track_id));
    const priorInSet = plays
      .slice(0, -1)
      .filter((p) => setIds.has(p.track_id)).length;
    const expectedNextId = set.tracks[priorInSet]?.track_id ?? null;
    const onPlan = expectedNextId === lastPlay.track_id;
    if (!onPlan) {
      store.openSetRail();
    } else if (open) {
      const t = setTimeout(() => store.closeSetRail(), AUTO_COLLAPSE_MS);
      return () => clearTimeout(t);
    }
  }, [plays.length, set, view, open]);

  // The Set editor view renders the rail content full-pane; in that view we
  // don't render the drawer/tab — SetEditor handles its own layout.
  if (view === "set") return null;

  return (
    <>
      {/* Edge tab — always visible when closed; collapses into nothing when
          open (drawer covers the same space). */}
      {!open && <ClosedTab set={set} />}
      {open && <Drawer set={set} loading={active.isLoading} />}
    </>
  );
}

function ClosedTab({ set }: { set: DanceSet | null }) {
  const label = set ? `${set.name} · ${set.tracks.length}` : "No active set";
  return (
    <button
      type="button"
      onClick={() => store.openSetRail()}
      title={`Open Set Rail (⌘\\) — ${label}`}
      aria-label="open set rail"
      className="fixed right-0 top-1/2 -translate-y-1/2 z-30 h-20 w-2.5 rounded-l-md bg-violet-700/70 hover:bg-violet-600 hover:w-3 transition-all"
    />
  );
}

function Drawer({
  set,
  loading,
}: {
  set: DanceSet | null;
  loading: boolean;
}) {
  return (
    <aside
      className="fixed right-0 top-0 bottom-0 z-30 flex flex-col border-l border-neutral-800 bg-neutral-950/95 backdrop-blur-md shadow-2xl"
      style={{ width: RAIL_WIDTH }}
      aria-label="Set Rail"
    >
      <Header set={set} />
      <div className="flex-1 overflow-y-auto px-2 py-2">
        {loading && (
          <div className="text-[11px] text-neutral-500 px-1 py-2">Loading…</div>
        )}
        {!loading && !set && <EmptyState />}
        {set && <Body set={set} />}
      </div>
    </aside>
  );
}

function Header({ set }: { set: DanceSet | null }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-neutral-800 shrink-0">
      <span className="text-[10px] uppercase tracking-wider text-neutral-500 shrink-0">
        Set
      </span>
      {set ? (
        <SetMenu set={set} variant="compact" />
      ) : (
        <span className="flex-1 text-sm text-neutral-500 truncate">—</span>
      )}
      <span className="text-[10px] text-neutral-500 tabular-nums shrink-0">
        {set ? `${set.tracks.length}` : "0"}
      </span>
      <button
        type="button"
        onClick={() => store.closeSetRail()}
        title="Close (Esc / ⌘\\)"
        aria-label="close set rail"
        className="w-6 h-6 inline-flex items-center justify-center rounded text-neutral-500 hover:text-neutral-100 hover:bg-neutral-800"
      >
        ×
      </button>
    </div>
  );
}

function EmptyState() {
  const create = useCreateSet();
  const activate = useActivateSet();
  const pending = create.isPending || activate.isPending;

  function createAndActivate() {
    create.mutate(
      { name: "My set" },
      {
        onSuccess: (created) => activate.mutate(created.id),
      },
    );
  }

  return (
    <div className="text-[11px] text-neutral-500 px-1 py-4 space-y-3 leading-relaxed">
      <p>No active set yet.</p>
      <p>
        A Set is your curated plan for a gig — name it, fill it via ⌘K, save
        it, reload it. The rail surfaces tail recs as you go.
      </p>
      <button
        type="button"
        onClick={createAndActivate}
        disabled={pending}
        className="w-full h-8 rounded bg-violet-700 hover:bg-violet-600 text-white text-xs disabled:opacity-50"
      >
        {pending ? "…" : "+ create empty set"}
      </button>
    </div>
  );
}

function Body({ set }: { set: DanceSet }) {
  // Pull the current session here so the rail can mark which set track was
  // most recently played (helps reorient when the rail pops open off-plan).
  const session = useCurrentSession();
  const lastPlayId =
    session.data?.plays?.[session.data.plays.length - 1]?.track_id ?? null;
  return (
    <div className="flex flex-col gap-2">
      <TrackList set={set} lastPlayId={lastPlayId} />
      <div className="border-t border-neutral-900 my-1" />
      <TailRecs set={set} />
      <AddFromCmdK />
    </div>
  );
}

function TrackList({
  set,
  lastPlayId,
}: {
  set: DanceSet;
  lastPlayId: number | null;
}) {
  if (set.tracks.length === 0) {
    return (
      <div className="text-[11px] text-neutral-500 px-1 py-2 italic">
        Empty set — add tracks from ⌘K or pin from the rec banners.
      </div>
    );
  }
  return (
    <ol className="flex flex-col gap-1" aria-label="Set tracks">
      {set.tracks.map((t, i) => (
        <SetTrackRow
          key={t.track_id}
          setId={set.id}
          track={t}
          index={i}
          isLast={i === set.tracks.length - 1}
          isCurrentPlay={t.track_id === lastPlayId}
        />
      ))}
    </ol>
  );
}

function SetTrackRow({
  setId,
  track,
  index,
  isLast,
  isCurrentPlay = false,
}: {
  setId: number;
  track: SetTrack;
  index: number;
  isLast: boolean;
  isCurrentPlay?: boolean;
}) {
  const move = useMoveTrackInSet();
  const remove = useRemoveTrackFromSet();

  // Soft-pin → mirror today's pinnedSongRecs gesture. Build a ColumnRec
  // shaped enough for the Mix-column banner to render.
  function softPin() {
    const rec: ColumnRec = {
      track_id: track.track_id,
      stem_file_id: null,
      track_title: track.title,
      track_artist: track.artist,
      bpm: track.bpm,
      key_camelot: track.key_camelot,
      floor_energy: track.floor_energy,
      score: 0,
      score_breakdown: {},
      reasons: ["from active set"],
    };
    store.pinToSong(rec);
  }

  // Force-load respects the per-slot stem filter. null/missing = load all
  // stems (the existing default).
  const forceLoad = useMutation({
    mutationFn: () =>
      pushTrackToLive(track.track_id, {
        includeStems: true,
        kinds: track.stem_kinds ?? undefined,
      }),
  });

  function onTap(e: React.MouseEvent) {
    if (e.shiftKey) {
      forceLoad.mutate();
    } else {
      softPin();
    }
  }

  return (
    <li
      className={`group relative rounded-md border transition-colors ${
        isCurrentPlay
          ? "border-emerald-500/60 bg-emerald-500/10 hover:bg-emerald-500/15"
          : "border-neutral-800 bg-neutral-900/60 hover:bg-neutral-900"
      }`}
    >
      {isCurrentPlay && (
        <span
          className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1 text-emerald-400 text-[10px] animate-pulse"
          aria-label="now playing"
        >
          ●
        </span>
      )}
      {/* Stem-filter chip is the leftmost element of the row — first
          thing the eye lands on so the planning intent reads at a glance.
          Lives outside the main click target so taps don't soft-pin. */}
      <div
        className="absolute left-1 top-1/2 -translate-y-1/2 z-10"
        onClick={(e) => e.stopPropagation()}
      >
        <StemKindsChip
          setId={setId}
          trackId={track.track_id}
          stemKinds={track.stem_kinds}
          variant="compact"
        />
      </div>
      <button
        type="button"
        onClick={onTap}
        title="Tap: pin to Mix recs · Shift-tap: load into Live"
        className="w-full text-left pl-12 pr-2 py-1.5"
      >
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[10px] text-neutral-500 tabular-nums shrink-0 w-5 text-right">
            {index + 1}
          </span>
          <span
            className={`flex-1 text-xs truncate font-medium ${
              isProcessing(track) ? "text-neutral-500" : "text-neutral-100"
            }`}
          >
            {track.title ?? `Track #${track.track_id}`}
          </span>
          {track.note && (
            <span
              title={track.note}
              aria-label={`note: ${track.note}`}
              className="text-[10px] text-amber-300/80"
            >
              📝
            </span>
          )}
          <StateChip
            state={track.track_state}
            error={track.track_error}
          />
        </div>
        <div className="flex items-center gap-1.5 pl-6">
          <span className="text-[10px] text-neutral-500 truncate flex-1">
            {track.artist ?? "—"}
          </span>
          {track.bpm != null && (
            <span className="text-[10px] text-neutral-500 font-mono tabular-nums">
              {track.bpm.toFixed(0)}
            </span>
          )}
          {track.key_camelot && (
            <KeyBadge keyCamelot={track.key_camelot} size="sm" />
          )}
        </div>
      </button>
      {/* Hover actions — up / down / remove. Positioned absolute so the
          row's clickable area stays uninterrupted. */}
      <div className="absolute right-1 top-1 hidden group-hover:flex items-center gap-0.5">
        <button
          type="button"
          disabled={index === 0 || move.isPending}
          onClick={() =>
            move.mutate({ setId, trackId: track.track_id, position: index - 1 })
          }
          title="Move up"
          aria-label="move up"
          className="w-5 h-5 inline-flex items-center justify-center rounded text-neutral-500 hover:text-neutral-100 hover:bg-neutral-800 disabled:opacity-30"
        >
          ▲
        </button>
        <button
          type="button"
          disabled={isLast || move.isPending}
          onClick={() =>
            move.mutate({ setId, trackId: track.track_id, position: index + 1 })
          }
          title="Move down"
          aria-label="move down"
          className="w-5 h-5 inline-flex items-center justify-center rounded text-neutral-500 hover:text-neutral-100 hover:bg-neutral-800 disabled:opacity-30"
        >
          ▼
        </button>
        <button
          type="button"
          disabled={remove.isPending}
          onClick={() => remove.mutate({ setId, trackId: track.track_id })}
          title="Remove from set"
          aria-label="remove from set"
          className="w-5 h-5 inline-flex items-center justify-center rounded text-neutral-500 hover:text-rose-200 hover:bg-rose-500/20 disabled:opacity-30"
        >
          ×
        </button>
      </div>
    </li>
  );
}

function TailRecs({ set }: { set: DanceSet }) {
  const recs = useTailRecs(set.id, { k: 5 });
  const add = useAddTrackToSet();

  if (set.tracks.length === 0) return null;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2 px-1 pt-1">
        <span className="text-[10px] uppercase tracking-wider text-neutral-500">
          Tail recs
        </span>
        <span className="flex-1 border-b border-neutral-900" />
      </div>
      {recs.isLoading && (
        <div className="text-[10px] text-neutral-600 px-1 py-1">…</div>
      )}
      {recs.isError && (
        <div className="text-[10px] text-rose-300 px-1 py-1">
          Couldn't fetch recs
        </div>
      )}
      {recs.data?.recs.map((r) => (
        <TailRecRow
          key={r.track_id}
          rec={r}
          onAdd={() => add.mutate({ setId: set.id, trackId: r.track_id })}
        />
      ))}
      {recs.data && recs.data.recs.length === 0 && (
        <div className="text-[10px] text-neutral-600 px-1 py-1 italic">
          No tail-rec candidates yet
        </div>
      )}
    </div>
  );
}

function TailRecRow({
  rec,
  onAdd,
}: {
  rec: TailRec;
  onAdd: () => void;
}) {
  const tooltip = [
    `Score: ${Math.round(rec.score * 100)}`,
    rec.reasons.length ? `Why: ${rec.reasons.join(" · ")}` : null,
  ]
    .filter(Boolean)
    .join("\n");
  return (
    <div
      className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5 text-xs"
      title={tooltip}
    >
      <div className="flex items-baseline gap-1.5">
        <span className="font-mono text-[10px] text-neutral-500 tabular-nums w-7 shrink-0">
          {Math.round(rec.score * 100)}
        </span>
        <span className="flex-1 truncate text-neutral-100 text-xs font-medium">
          {rec.track_title ?? `Track #${rec.track_id}`}
        </span>
        <button
          type="button"
          onClick={onAdd}
          title="Add to set"
          aria-label="add to set"
          className="shrink-0 w-5 h-5 inline-flex items-center justify-center rounded bg-violet-700/70 hover:bg-violet-700 text-white text-[10px]"
        >
          +
        </button>
      </div>
      <div className="text-[10px] text-neutral-500 truncate pl-9">
        {rec.track_artist ?? "—"}
        {rec.bpm != null && (
          <span className="ml-1.5 font-mono">{rec.bpm.toFixed(0)}</span>
        )}
        {rec.key_camelot && <span className="ml-1.5">· {rec.key_camelot}</span>}
        {rec.floor_energy != null && (
          <span className="ml-1.5 text-neutral-600">· E{rec.floor_energy}</span>
        )}
      </div>
    </div>
  );
}

function AddFromCmdK() {
  return (
    <button
      type="button"
      onClick={() => store.openCommandBar()}
      title="Add tracks via ⌘K"
      className="mt-1 w-full h-8 rounded border border-dashed border-neutral-800 hover:border-violet-700/60 text-xs text-neutral-500 hover:text-neutral-200 transition-colors"
    >
      ⊕ from ⌘K
    </button>
  );
}
