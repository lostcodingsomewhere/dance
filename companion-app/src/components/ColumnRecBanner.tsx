import { useColumnRecs } from "../hooks/useColumnRecs";
import { pushTrackToLive } from "../api";
import { useMutation } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import {
  useStartPreview,
  useStopPreview,
  usePreviewState,
} from "../hooks/usePreview";
import { roleLabel } from "../lib/roles";
import type { ColumnRec } from "../types";
import { store } from "../store";
import { RoleIcon } from "./RoleIcon";

const ROLE_ACCENT: Record<string, string> = {
  drums: "bg-red-500/15 border-red-500/30 text-red-300",
  bass: "bg-amber-500/15 border-amber-500/30 text-amber-300",
  vocals: "bg-lime-500/15 border-lime-500/30 text-lime-300",
  other: "bg-sky-500/15 border-sky-500/30 text-sky-300",
  mix: "bg-neutral-700/30 border-neutral-500/40 text-neutral-200",
};

/**
 * One column's live-rescoring rec stream. Renders as a vertical stack of
 * candidate cards above the scene grid for the matching column. Each card
 * answers "this candidate fits the current combo because: ...".
 *
 * The hook automatically re-queries when the active combo changes (driven by
 * playing_clips from the WebSocket).
 */
export function ColumnRecBanner({ column, k = 4 }: { column: string; k?: number }) {
  const q = useColumnRecs(column, { k });
  const accent = ROLE_ACCENT[column] ?? ROLE_ACCENT.mix;

  return (
    <div className="flex flex-col gap-1" data-testid={`rec-banner-${column}`}>
      <div
        className={`flex items-center justify-between px-2 py-1 rounded-md border text-[10px] uppercase tracking-widest ${accent}`}
      >
        <span className="flex items-center gap-1.5 font-semibold">
          <RoleIcon role={column} size={14} />
          {roleLabel(column)}
        </span>
        <span className="opacity-60">{q.data?.recs.length ?? 0} recs</span>
      </div>
      {q.isLoading && (
        <div className="text-[10px] text-neutral-600 px-2 py-2">…</div>
      )}
      {q.isError && (
        <div className="text-[10px] text-rose-300 px-2 py-2">
          Couldn't fetch recs
        </div>
      )}
      {q.data?.recs.map((rec) => (
        <RecCard
          key={`${rec.track_id}-${rec.stem_file_id ?? "mix"}`}
          rec={rec}
          column={column}
        />
      ))}
      {q.data && q.data.recs.length === 0 && !q.isLoading && (
        <div className="text-[10px] text-neutral-600 px-2 py-2 italic">
          No candidates yet
        </div>
      )}
    </div>
  );
}

function RecCard({ rec, column }: { rec: ColumnRec; column: string }) {
  const qc = useQueryClient();
  const startPreview = useStartPreview();
  const stopPreview = useStopPreview();
  const previewing = usePreviewState();
  const isPreviewing =
    previewing?.trackId === rec.track_id && previewing?.column === column;
  // Whole-song cue from a stem card: same track, but auditioned via the
  // mix path (full original audio file). Lets the user follow "I like
  // these drums — does the whole song hold up?" without leaving the card.
  const isPreviewingSong =
    previewing?.trackId === rec.track_id && previewing?.column === "mix";

  // Stem cards load only their own stem; the song-column card loads the
  // whole 4-stem combo into a fresh row.
  const isSongCard = column === "mix";

  function onLoadSuccess(result: { scene_index: number; track_indices: Record<string, number> }, isFullSong: boolean) {
    // Auto-stop any preview when committing — the candidate is now on
    // master, so cue should go silent.
    if (previewing) stopPreview.mutate();
    // Register the deck locally only for whole-song commits. Single-stem
    // loads don't form a "deck" — their row may have cells from other
    // tracks and the backend is the source of truth.
    if (isFullSong) {
      store.registerDeck({
        track_id: rec.track_id,
        scene_index: result.scene_index,
        stem_track_indices: Object.values(result.track_indices),
        loaded_at: Date.now(),
      });
    }
    qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
    qc.invalidateQueries({ queryKey: ["recommend", "by-column"] });
  }

  const load = useMutation({
    mutationFn: () =>
      pushTrackToLive(rec.track_id, {
        includeStems: true,
        kinds: isSongCard ? undefined : [column],
      }),
    onSuccess: (result) => onLoadSuccess(result, isSongCard),
  });

  // "Load whole song" — only available on stem cards. Same target track as
  // this rec, but commits all 4 stems into a fresh row (anchor-ready).
  const loadSong = useMutation({
    mutationFn: () =>
      pushTrackToLive(rec.track_id, {
        includeStems: true,
        kinds: undefined,
      }),
    onSuccess: (result) => onLoadSuccess(result, true),
  });

  function onPreview() {
    if (isPreviewing) {
      stopPreview.mutate();
    } else {
      startPreview.mutate({ trackId: rec.track_id, column });
    }
  }

  function onPreviewSong() {
    if (isPreviewingSong) {
      stopPreview.mutate();
    } else {
      // Replaces any in-flight preview (single-preview constraint on
      // backend). The CueStrip auto-flips to song mode because the new
      // preview state has column="mix".
      startPreview.mutate({ trackId: rec.track_id, column: "mix" });
    }
  }

  const energy = rec.floor_energy;
  return (
    <div
      className={`rounded-md border px-2 py-1.5 text-xs transition-colors ${
        isPreviewing || isPreviewingSong
          ? "border-cyan-400/60 bg-cyan-500/10 shadow-[0_0_12px_rgba(34,211,238,0.25)]"
          : "border-neutral-800/70 bg-neutral-900/40"
      }`}
    >
      <div className="flex items-baseline gap-1.5">
        <span className="font-mono text-[10px] text-neutral-500 tabular-nums shrink-0">
          {Math.round(rec.score * 100)}
        </span>
        <span className="truncate flex-1 text-neutral-50 text-sm font-medium leading-tight">
          {rec.track_title ?? `Track #${rec.track_id}`}
        </span>
      </div>
      <div className="text-xs text-neutral-300 truncate mt-0.5">
        {rec.track_artist ?? "—"}
      </div>
      <div className="text-[10px] text-neutral-500 truncate font-mono mt-0.5">
        {rec.bpm != null && <span>{rec.bpm.toFixed(1)} BPM</span>}
        {rec.key_camelot && (
          <span className="ml-1.5">· {rec.key_camelot}</span>
        )}
        {energy != null && (
          <span className="ml-1.5 text-neutral-600">· E{energy}</span>
        )}
      </div>
      {rec.reasons.length > 0 && (
        <div className="text-[9px] text-neutral-600 uppercase tracking-wide mt-0.5 truncate">
          {rec.reasons.join(" · ")}
        </div>
      )}
      <div className="mt-1 grid grid-cols-[auto_1fr] gap-1">
        {/* Row 1: stem actions (or song actions on the song card). */}
        <button
          type="button"
          onClick={onPreview}
          disabled={startPreview.isPending || stopPreview.isPending}
          title={
            isPreviewing
              ? "Stop preview (Cue track → headphones)"
              : isSongCard
              ? "Preview the song in headphones (Scarlett outs 3/4)"
              : `Preview just the ${roleLabel(column).toLowerCase()} stem in headphones`
          }
          className={`shrink-0 w-7 text-[10px] rounded py-1 transition-colors disabled:opacity-50 ${
            isPreviewing
              ? "bg-cyan-500/30 hover:bg-cyan-500/40 text-cyan-200 border border-cyan-400/40"
              : "bg-neutral-800 hover:bg-neutral-700 text-neutral-300"
          }`}
          aria-label={isPreviewing ? "stop preview" : "preview"}
        >
          {isPreviewing ? "⏹" : "▶"}
        </button>
        <button
          type="button"
          onClick={() => load.mutate()}
          disabled={load.isPending}
          title={
            isSongCard
              ? "Load all 4 stems into a fresh row (anchor-ready)"
              : `Load only the ${roleLabel(column).toLowerCase()} stem into the next free ${roleLabel(column).toLowerCase()} slot`
          }
          className="text-[10px] rounded bg-violet-700/70 hover:bg-violet-700 text-white py-1 transition-colors disabled:opacity-50"
        >
          {load.isPending
            ? "loading…"
            : isSongCard
            ? "Load song"
            : `Load ${roleLabel(column).toLowerCase()}`}
        </button>
        {/* Row 2: song escape hatch — only on stem cards. ♪ previews the
            whole song through cue; Load song commits all 4 stems to a
            fresh row regardless of which stem column we're sitting in. */}
        {!isSongCard && (
          <>
            <button
              type="button"
              onClick={onPreviewSong}
              disabled={startPreview.isPending || stopPreview.isPending}
              title={
                isPreviewingSong
                  ? "Stop preview (whole song in headphones)"
                  : "Preview the WHOLE SONG of this rec in headphones"
              }
              className={`shrink-0 w-7 text-[10px] rounded py-1 transition-colors disabled:opacity-50 ${
                isPreviewingSong
                  ? "bg-cyan-500/30 hover:bg-cyan-500/40 text-cyan-200 border border-cyan-400/40"
                  : "bg-neutral-800 hover:bg-neutral-700 text-neutral-400"
              }`}
              aria-label={isPreviewingSong ? "stop song preview" : "preview song"}
            >
              {isPreviewingSong ? "⏹" : "♪"}
            </button>
            <button
              type="button"
              onClick={() => loadSong.mutate()}
              disabled={loadSong.isPending}
              title="Load the WHOLE SONG (all 4 stems) of this rec into a fresh row"
              className="text-[10px] rounded bg-neutral-800 hover:bg-violet-700/70 text-neutral-300 hover:text-white py-1 transition-colors disabled:opacity-50 border border-neutral-700/60 hover:border-violet-500/0"
              aria-label="load whole song"
            >
              {loadSong.isPending ? "loading…" : "Load song"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
