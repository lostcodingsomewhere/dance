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
import { store, useAppStore } from "../store";

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
  // User-pinned tracks (sent from stem cards via the ↗ button). Only
  // surfaced in the SONG column — pinning IS "queue this as a whole-
  // song candidate." Dedupe against backend recs so a pinned track
  // that's also recommended by the backend appears once (pinned wins).
  const pinned = useAppStore((s) => s.pinnedSongRecs);
  const recs = column === "mix"
    ? [
        ...pinned,
        ...(q.data?.recs ?? []).filter(
          (r) => !pinned.some((p) => p.track_id === r.track_id),
        ),
      ]
    : (q.data?.recs ?? []);
  const pinnedIds = column === "mix" ? new Set(pinned.map((p) => p.track_id)) : new Set();

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
      {recs.map((rec) => (
        <RecCard
          key={`${rec.track_id}-${rec.stem_file_id ?? "mix"}`}
          rec={rec}
          column={column}
          isPinned={pinnedIds.has(rec.track_id)}
        />
      ))}
      {!q.isLoading && recs.length === 0 && (
        <div className="text-[10px] text-neutral-600 px-2 py-2 italic">
          No candidates yet
        </div>
      )}
    </div>
  );
}

function RecCard({
  rec,
  column,
  isPinned = false,
}: {
  rec: ColumnRec;
  column: string;
  isPinned?: boolean;
}) {
  const qc = useQueryClient();
  const startPreview = useStartPreview();
  const stopPreview = useStopPreview();
  const previewing = usePreviewState();
  const isPreviewing =
    previewing?.trackId === rec.track_id && previewing?.column === column;

  // Stem cards load only their own stem; the song-column card loads the
  // whole 4-stem combo into a fresh row.
  const isSongCard = column === "mix";

  const load = useMutation({
    mutationFn: () =>
      pushTrackToLive(rec.track_id, {
        includeStems: true,
        kinds: isSongCard ? undefined : [column],
      }),
    onSuccess: (result) => {
      // Auto-stop any preview when committing — the candidate is now
      // on master, so cue should go silent.
      if (previewing) stopPreview.mutate();
      // Register the deck locally only for whole-song commits. Single-
      // stem loads don't form a "deck" — their row may have cells from
      // other tracks and the backend is the source of truth.
      if (isSongCard) {
        store.registerDeck({
          track_id: rec.track_id,
          scene_index: result.scene_index,
          stem_track_indices: Object.values(result.track_indices),
          loaded_at: Date.now(),
        });
        // Clean up: if this song was pinned, drop it now that it's
        // committed to a real deck.
        store.unpinFromSong(rec.track_id);
      }
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
      qc.invalidateQueries({ queryKey: ["recommend", "by-column"] });
    },
  });

  function onPreview() {
    if (isPreviewing) {
      stopPreview.mutate();
    } else {
      startPreview.mutate({ trackId: rec.track_id, column });
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
      className={`relative rounded-md border px-2 py-1.5 text-xs transition-colors ${
        isPreviewing
          ? "border-cyan-400/60 bg-cyan-500/10 shadow-[0_0_12px_rgba(34,211,238,0.25)]"
          : isPinned
          ? `${styles.border} ${styles.bg} border-l-2 border-l-emerald-400/60`
          : `${styles.border} ${styles.bg}`
      }`}
      title={scoreTooltip}
    >
      {/* Pinned: emerald left-edge accent + an × button in the corner
          to unpin. The accent makes "this is a pinned card" obvious at
          a glance even before hovering; the × is the universal "remove
          this" affordance. Drops the track from the local pinned list;
          backend recs take it from there. */}
      {isPinned && (
        <button
          type="button"
          onClick={() => store.unpinFromSong(rec.track_id)}
          title="Unpin from Song recs"
          aria-label="unpin from song recs"
          className="absolute top-1 right-1 w-4 h-4 rounded-full text-[10px] leading-none inline-flex items-center justify-center text-emerald-200/80 bg-emerald-500/15 border border-emerald-400/40 hover:bg-rose-500/30 hover:text-rose-100 hover:border-rose-400/60 transition-colors"
        >
          ×
        </button>
      )}
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
      {/* Two icon actions per card: preview + load. Both act on the
          card's own column (stem cards load that stem, song card loads
          the whole song). Stem cards get one extra tiny ↗ that sends
          the track to the SONG column's rec list — unified action
          model: every card just previews / loads its own column, and
          ↗ is the "consider this for song mode too" gesture. */}
      <div
        className={`mt-1 grid gap-1 ${
          isSongCard ? "grid-cols-[auto_1fr]" : "grid-cols-[auto_1fr_auto]"
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
        {!isSongCard && (
          <button
            type="button"
            onClick={() => store.pinToSong(rec)}
            title="Send to Song recs (pin this track at the top of the SONG column)"
            aria-label="send to song recs"
            className="shrink-0 w-7 h-7 text-xs rounded bg-neutral-800/70 hover:bg-neutral-700 text-neutral-400 hover:text-neutral-200 border border-neutral-700/60 transition-colors inline-flex items-center justify-center"
          >
            ↗
          </button>
        )}
      </div>
    </div>
  );
}
