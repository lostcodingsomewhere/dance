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
  DndContext,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  useSortable,
  arrayMove,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  useActiveSet,
  useActivateSet,
  useAddTrackToSet,
  useCreateSet,
  useMoveTrackInSet,
  useRemoveTrackFromSet,
  useTailRecs,
} from "../hooks/useSets";
import {
  usePreviewState,
  useStartPreview,
  useStopPreview,
} from "../hooks/usePreview";
import { formatDuration } from "../lib/format";
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

// Exported so SetEditor's row can reuse the same definition — keeps the
// "should the → Live button be disabled?" gate in lockstep across surfaces.
export function isProcessing(t: SetTrack): boolean {
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

/**
 * ``inline`` is passed by App.tsx and means: "render me as a flex sibling
 * of the main column, not as a fixed drawer overlaying it." That's how
 * the rail's open-state in Booth pushes the top nav over instead of
 * floating above it. When false (Pipeline view, Booth-unpinned), the
 * component falls back to the legacy fixed drawer / pill-tab UI.
 */
export function SetRail({ inline = false }: { inline?: boolean }) {
  const open = useAppStore((s) => s.setRailOpen);
  const pinned = useAppStore((s) => s.setRailPinned);
  const view = useAppStore((s) => s.currentView);
  const active = useActiveSet();
  const set = active.data ?? null;

  // In Booth, pinned mode is the source of truth: the drawer shows
  // regardless of ``open``. In other views the rail behaves as before.
  const isShownInBooth = view === "booth" && pinned;
  const drawerVisible = open || isShownInBooth;

  // ⌘\ global toggle. ESC closes — but never in pinned-Booth, where the
  // rail is intentionally always present.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "\\") {
        e.preventDefault();
        if (view === "booth") {
          // In Booth ⌘\ toggles the pin (since pinned is the truth).
          store.toggleSetRailPinned();
        } else {
          store.toggleSetRail();
        }
      }
      if (open && !isShownInBooth && e.key === "Escape") {
        store.closeSetRail();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, view, isShownInBooth]);

  // Auto-collapse 3 s after a clip fires (Booth, unpinned only). When
  // pinned, the rail stays put — the user explicitly chose "always on,"
  // which would conflict with auto-collapse.
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
    if (!open || view !== "booth" || pinned) return;
    if (prevFp.current === playingFingerprint) return;
    prevFp.current = playingFingerprint;
    const t = setTimeout(() => store.closeSetRail(), AUTO_COLLAPSE_MS);
    return () => clearTimeout(t);
  }, [playingFingerprint, open, view, pinned]);

  // Plan-aware rail behavior. When a new play is auto-logged in the
  // session, compare it to the next-planned set track:
  //   - off-plan (DJ deviated) → openSetRail() immediately, so the plan
  //     reasserts as a reorientation cue (only meaningful when unpinned —
  //     pinned mode already shows the rail)
  //   - on-plan → if rail was open & not pinned, schedule a soft collapse
  const session = useCurrentSession();
  const plays = session.data?.plays ?? [];
  const prevPlayCount = useRef(plays.length);
  useEffect(() => {
    if (view !== "booth" || !set || pinned) return;
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
  }, [plays.length, set, view, open, pinned]);

  // The Set editor view renders the rail content full-pane; in that view we
  // don't render the drawer/tab — SetEditor handles its own layout.
  if (view === "set") return null;

  // Inline mode: render as a flex sibling. App.tsx is responsible for
  // placing us in the right side of the flex row, so we just need to set
  // an explicit width and full-height styles. The closed-pill tab is
  // never shown in inline mode (there's nothing to peek behind).
  if (inline) {
    return (
      <aside
        className="shrink-0 flex flex-col border-l border-neutral-800 bg-neutral-950/95"
        style={{ width: RAIL_WIDTH }}
        aria-label="Set Rail"
      >
        <Header set={set} showCloseButton={!isShownInBooth} />
        <div className="flex-1 overflow-y-auto px-2 py-2 min-h-0">
          {active.isLoading && (
            <div className="text-[11px] text-neutral-500 px-1 py-2">Loading…</div>
          )}
          {!active.isLoading && !set && <EmptyState />}
          {set && <Body set={set} />}
        </div>
      </aside>
    );
  }

  // Non-inline fallback (Pipeline view): closed pill ↔ fixed drawer.
  return (
    <>
      {!drawerVisible && <ClosedTab set={set} />}
      {drawerVisible && <Drawer set={set} loading={active.isLoading} />}
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
      <Header set={set} showCloseButton />
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

function Header({
  set,
  showCloseButton,
}: {
  set: DanceSet | null;
  /** Hide the × close button in Booth-pinned mode where there is nothing
   *  to close — the rail is meant to stay. The pin toggle stays available
   *  regardless so the user can unpin without keyboard. */
  showCloseButton: boolean;
}) {
  const view = useAppStore((s) => s.currentView);
  const pinned = useAppStore((s) => s.setRailPinned);
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
      {view === "booth" && (
        <button
          type="button"
          onClick={() => store.toggleSetRailPinned()}
          title={
            pinned
              ? "Unpin rail (⌘\\) — auto-collapse during mixes"
              : "Pin rail open (⌘\\) — always visible in Booth"
          }
          aria-label={pinned ? "unpin set rail" : "pin set rail"}
          className={`w-6 h-6 inline-flex items-center justify-center rounded ${
            pinned
              ? "text-violet-300 hover:text-violet-100 hover:bg-violet-500/20"
              : "text-neutral-500 hover:text-neutral-100 hover:bg-neutral-800"
          }`}
        >
          {pinned ? "📌" : "📍"}
        </button>
      )}
      {showCloseButton && (
        <button
          type="button"
          onClick={() => store.closeSetRail()}
          title="Close (Esc / ⌘\\)"
          aria-label="close set rail"
          className="w-6 h-6 inline-flex items-center justify-center rounded text-neutral-500 hover:text-neutral-100 hover:bg-neutral-800"
        >
          ×
        </button>
      )}
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
  const move = useMoveTrackInSet();
  // ``PointerSensor`` with a small activation distance keeps a quick *click*
  // on the row from being mistaken for a drag (so the soft-pin tap and the
  // ▶ Cue / → Live buttons still work). 6px is the dnd-kit default; we
  // raise to 8px because the rail rows are tightly packed.
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
  );

  if (set.tracks.length === 0) {
    return (
      <div className="text-[11px] text-neutral-500 px-1 py-2 italic">
        Empty set — add tracks from ⌘K or pin from the rec banners.
      </div>
    );
  }

  function onDragEnd(e: DragEndEvent) {
    const { active, over } = e;
    if (!over || active.id === over.id) return;
    const oldIdx = set.tracks.findIndex((t) => t.track_id === Number(active.id));
    const newIdx = set.tracks.findIndex((t) => t.track_id === Number(over.id));
    if (oldIdx < 0 || newIdx < 0) return;
    // Optimistic: arrayMove gives us the next order. We don't update local
    // state — the move mutation invalidates ["sets","active"] which triggers
    // a refetch and re-render with the new order from the server. dnd-kit's
    // animation handles the visual transition during the few ms of latency.
    void arrayMove; // (imported so callers know we COULD use it for optimism)
    move.mutate({
      setId: set.id,
      trackId: Number(active.id),
      position: newIdx,
    });
  }

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragEnd={onDragEnd}
    >
      <SortableContext
        items={set.tracks.map((t) => t.track_id)}
        strategy={verticalListSortingStrategy}
      >
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
      </SortableContext>
    </DndContext>
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
  /** Kept so the prop signature stays stable for future "last row only"
   *  affordances; no longer used now that up/down arrows are gone. */
  isLast: boolean;
  isCurrentPlay?: boolean;
}) {
  // Suppress unused-prop warning while keeping signature stable.
  void isLast;
  // ``useSortable`` wires up everything: drag-source attrs, drop-target
  // collision detection, smooth translate transition. We attach
  // ``setNodeRef`` + ``attributes`` to the <li>, and ``listeners`` to
  // the ⋮⋮ drag handle (so clicks on the rest of the row still soft-pin).
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: track.track_id });
  const sortableStyle: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
  };
  const remove = useRemoveTrackFromSet();
  const startPreview = useStartPreview();
  const stopPreview = useStopPreview();
  const previewing = usePreviewState();
  const isPreviewingThis = previewing?.trackId === track.track_id;

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
      ref={setNodeRef}
      style={sortableStyle}
      {...attributes}
      className={`group relative rounded-md border transition-colors ${
        isCurrentPlay
          ? "border-emerald-500/60 bg-emerald-500/10 hover:bg-emerald-500/15"
          : "border-neutral-800 bg-neutral-900/60 hover:bg-neutral-900"
      } ${isDragging ? "opacity-40 z-30" : ""}`}
    >
      {isCurrentPlay && (
        <span
          className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1 text-emerald-400 text-[10px] animate-pulse"
          aria-label="now playing"
        >
          ●
        </span>
      )}
      {/* Drag handle — only THIS element receives the drag listeners
          (via {...listeners}). Clicks on the rest of the row still do
          soft-pin / Cue / etc. The handle is wider + brighter now so
          users can find it; tooltip explains the gesture. */}
      <div
        {...listeners}
        title="Drag to reorder"
        aria-label="drag handle"
        className="absolute left-0 top-0 bottom-0 w-4 flex items-center justify-center cursor-grab active:cursor-grabbing text-neutral-600 hover:text-neutral-300 z-10 select-none touch-none"
      >
        ⋮⋮
      </div>
      {/* Stem-filter chip — second leftmost element, after the drag handle. */}
      <div
        className="absolute left-4 top-1/2 -translate-y-1/2 z-10"
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
        title="Tap: pin to Mix recs · Shift-tap (or hover → Live): load into Live"
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
          <StateChip
            state={track.track_state}
            error={track.track_error}
          />
        </div>
        {/* Bottom metadata row — hidden when hovering or when this row's
            preview is active. The action toolbar (below) occupies the same
            vertical slot so the card height stays stable. ``invisible``
            preserves layout; ``hidden`` would collapse the row and make
            the card jump on hover. */}
        <div
          className={`flex items-center gap-1.5 pl-6 ${
            isPreviewingThis ? "invisible" : "group-hover:invisible"
          }`}
        >
          <span className="text-[10px] text-neutral-500 truncate flex-1">
            {track.artist ?? "—"}
          </span>
          {track.duration_seconds != null && (
            <span
              className="text-[10px] text-neutral-600 font-mono tabular-nums"
              title="Track duration"
            >
              {formatDuration(track.duration_seconds)}
            </span>
          )}
          {track.bpm != null && (
            <span className="text-[10px] text-neutral-500 font-mono tabular-nums">
              {track.bpm.toFixed(0)}
            </span>
          )}
          {track.key_camelot && (
            <KeyBadge keyCamelot={track.key_camelot} size="sm" />
          )}
        </div>
        {/* Inline note — surfaces the user's plan annotation directly so
            it doesn't need a hover or extra click to read. Hidden on
            hover/preview so the action toolbar has clean space. */}
        {track.note && (
          <div
            className={`pl-6 mt-0.5 text-[10px] text-amber-200/80 italic truncate flex items-baseline gap-1 ${
              isPreviewingThis ? "invisible" : "group-hover:invisible"
            }`}
            title={track.note}
          >
            <span aria-hidden="true">📝</span>
            <span className="truncate">{track.note}</span>
          </div>
        )}
      </button>
      {/* Hover actions — Cue preview / load-into-Live / up / down / remove.
          Anchored to bottom-left of the card (matching the click button's
          ``pl-12 py-1.5`` so the toolbar sits in the metadata row's slot
          rather than overlaying the title). ``isPreviewingThis`` keeps the
          Cue button visible (not hover-only) while this row's clip is
          auditioning so the DJ can hit Stop without re-finding the row. */}
      <div
        className={`absolute bottom-1.5 left-12 items-center gap-0.5 ${
          isPreviewingThis ? "flex" : "hidden group-hover:flex"
        }`}
      >
        <button
          type="button"
          disabled={
            isProcessing(track) ||
            startPreview.isPending ||
            stopPreview.isPending
          }
          onClick={(e) => {
            e.stopPropagation();
            if (isPreviewingThis) {
              stopPreview.mutate();
            } else {
              startPreview.mutate({ trackId: track.track_id, column: "mix" });
            }
          }}
          title={
            isProcessing(track)
              ? "Pipeline still running — preview enabled when complete"
              : isPreviewingThis
                ? "Stop preview (Cue Out → headphones)"
                : "Cue preview in headphones (Scarlett outs 3/4) — master untouched"
          }
          aria-label={isPreviewingThis ? "stop cue preview" : "cue preview"}
          className={`px-1.5 h-5 inline-flex items-center justify-center rounded text-[10px] font-medium disabled:opacity-30 disabled:cursor-not-allowed ${
            isPreviewingThis
              ? "text-cyan-200 bg-cyan-500/30 hover:bg-cyan-500/40"
              : "text-cyan-300 hover:text-cyan-100 hover:bg-cyan-500/20"
          }`}
        >
          {isPreviewingThis ? "■ Cue" : "▶ Cue"}
        </button>
        <button
          type="button"
          disabled={isProcessing(track) || forceLoad.isPending}
          onClick={(e) => {
            e.stopPropagation();
            forceLoad.mutate();
          }}
          title={
            isProcessing(track)
              ? "Pipeline still running — load enabled when complete"
              : "Load stems into Live now"
          }
          aria-label="load into Live"
          className="px-1.5 h-5 inline-flex items-center justify-center rounded text-[10px] font-medium text-emerald-300 hover:text-emerald-100 hover:bg-emerald-500/20 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          → Live
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
  const startPreview = useStartPreview();
  const stopPreview = useStopPreview();
  const previewing = usePreviewState();
  const isPreviewingThis = previewing?.trackId === rec.track_id;
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
          onClick={() => {
            if (isPreviewingThis) {
              stopPreview.mutate();
            } else {
              startPreview.mutate({ trackId: rec.track_id, column: "mix" });
            }
          }}
          disabled={startPreview.isPending || stopPreview.isPending}
          title={
            isPreviewingThis
              ? "Stop preview"
              : "Cue preview in headphones — master untouched"
          }
          aria-label={isPreviewingThis ? "stop cue preview" : "cue preview"}
          className={`shrink-0 px-1 h-5 inline-flex items-center justify-center rounded text-[10px] font-medium disabled:opacity-30 ${
            isPreviewingThis
              ? "text-cyan-200 bg-cyan-500/30 hover:bg-cyan-500/40"
              : "text-cyan-300 hover:text-cyan-100 hover:bg-cyan-500/20"
          }`}
        >
          {isPreviewingThis ? "■" : "▶"}
        </button>
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
