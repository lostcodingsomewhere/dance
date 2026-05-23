import { useColumnRecs } from "../hooks/useColumnRecs";
import { pushTrackToLive } from "../api";
import { useMutation } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import {
  useStartPreview,
  useStopPreview,
  usePreviewState,
} from "../hooks/usePreview";
import { ROLE_STYLES, roleLabel } from "../lib/roles";
import type { ColumnRec } from "../types";
import { store } from "../store";

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

  return (
    <div className="flex flex-col gap-1" data-testid={`rec-banner-${column}`}>
      {/* No header chip — the role label + count are redundant with the
          unified column headers up top. The per-card tint inherited
          from ROLE_STYLES carries the column identity down. */}
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
  const styles = ROLE_STYLES[column as keyof typeof ROLE_STYLES] ?? ROLE_STYLES.mix;
  // Reasons + load tooltips collapsed into one rich hover-title on the
  // card so the visible card stays minimal.
  const scoreTooltip = [
    `Score: ${Math.round(rec.score * 100)}`,
    rec.reasons.length > 0 ? `Why: ${rec.reasons.join(" · ")}` : null,
  ]
    .filter(Boolean)
    .join("\n");
  return (
    <div
      className={`rounded-md border px-2 py-1.5 text-xs transition-colors ${
        isPreviewing || isPreviewingSong
          ? "border-cyan-400/60 bg-cyan-500/10 shadow-[0_0_12px_rgba(34,211,238,0.25)]"
          : `${styles.border} ${styles.bg}`
      }`}
      title={scoreTooltip}
    >
      <div className="flex items-baseline gap-1.5">
        <span className="font-mono text-[10px] text-neutral-500 tabular-nums shrink-0">
          {Math.round(rec.score * 100)}
        </span>
        <span className="truncate flex-1 text-neutral-50 text-sm font-medium leading-tight">
          {rec.track_title ?? `Track #${rec.track_id}`}
        </span>
      </div>
      <div className="text-[11px] text-neutral-400 truncate leading-tight">
        {rec.track_artist ?? "—"}
      </div>
      <div className="text-[10px] text-neutral-500 truncate font-mono leading-tight">
        {rec.bpm != null && <span>{rec.bpm.toFixed(0)}</span>}
        {rec.key_camelot && <span className="ml-1.5">· {rec.key_camelot}</span>}
        {energy != null && (
          <span className="ml-1.5 text-neutral-600">· E{energy}</span>
        )}
      </div>
      {/* Action row — all icons, no text labels. The column header up
          top already tells you what role you're loading; repeating
          "Load vocals" inside every card was redundant. Stem cards get
          4 buttons (preview stem · load stem · preview song · load
          song); song cards get 2 (preview · load). Primary action (load
          stem) is the WIDE button — same visual weight that "Load
          vocals" had before, but now an icon-only canvas. */}
      <div
        className={`mt-1 grid gap-1 ${
          isSongCard
            ? "grid-cols-[auto_1fr]"
            : "grid-cols-[auto_1fr_auto_auto]"
        }`}
      >
        <button
          type="button"
          onClick={onPreview}
          disabled={startPreview.isPending || stopPreview.isPending}
          title={
            isPreviewing
              ? "Stop preview"
              : isSongCard
              ? "Preview the song in headphones"
              : `Preview the ${roleLabel(column).toLowerCase()} stem in headphones`
          }
          className={`shrink-0 w-7 h-7 text-xs rounded transition-colors disabled:opacity-50 inline-flex items-center justify-center ${
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
              : `Load the ${roleLabel(column).toLowerCase()} stem into the next free slot`
          }
          className="h-7 text-sm rounded bg-violet-700/70 hover:bg-violet-700 text-white transition-colors disabled:opacity-50 inline-flex items-center justify-center"
          aria-label={
            isSongCard
              ? "load whole song"
              : `load ${roleLabel(column).toLowerCase()}`
          }
        >
          {load.isPending ? "…" : "⤓"}
        </button>
        {/* Song escape hatch — only on stem cards. Smaller buttons so
            the stem-load (primary) stays visually dominant. Music note
            paired with the action icon: ♪▶ = preview song, ♪⤓ = load
            song. */}
        {!isSongCard && (
          <>
            <button
              type="button"
              onClick={onPreviewSong}
              disabled={startPreview.isPending || stopPreview.isPending}
              title={
                isPreviewingSong
                  ? "Stop song preview"
                  : "Preview the WHOLE SONG in headphones"
              }
              className={`shrink-0 w-8 h-7 text-[11px] rounded transition-colors disabled:opacity-50 inline-flex items-center justify-center gap-0.5 ${
                isPreviewingSong
                  ? "bg-cyan-500/30 hover:bg-cyan-500/40 text-cyan-200 border border-cyan-400/40"
                  : "bg-neutral-800 hover:bg-neutral-700 text-neutral-400"
              }`}
              aria-label={isPreviewingSong ? "stop song preview" : "preview song"}
            >
              <span className="opacity-70">♪</span>
              <span>{isPreviewingSong ? "⏹" : "▶"}</span>
            </button>
            <button
              type="button"
              onClick={() => loadSong.mutate()}
              disabled={loadSong.isPending}
              title="Load the WHOLE SONG (all 4 stems) into a fresh row"
              className="h-7 w-8 text-[11px] rounded bg-neutral-800 hover:bg-violet-700/70 text-neutral-300 hover:text-white transition-colors disabled:opacity-50 border border-neutral-700/60 inline-flex items-center justify-center gap-0.5"
              aria-label="load whole song"
            >
              {loadSong.isPending ? (
                "…"
              ) : (
                <>
                  <span className="opacity-70">♪</span>
                  <span>⤓</span>
                </>
              )}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
