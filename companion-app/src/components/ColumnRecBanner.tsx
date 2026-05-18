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
        className={`flex items-baseline justify-between px-2 py-1 rounded-md border text-[10px] uppercase tracking-widest ${accent}`}
      >
        <span className="font-semibold">{roleLabel(column)}</span>
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
      // Auto-stop any preview when committing — the candidate is now on
      // master, so cue should go silent.
      if (previewing) stopPreview.mutate();
      // Register the deck locally only for whole-song commits. Single-stem
      // loads don't form a "deck" — their row may have cells from other
      // tracks and the backend is the source of truth.
      if (isSongCard) {
        store.registerDeck({
          track_id: rec.track_id,
          scene_index: result.scene_index,
          stem_track_indices: Object.values(result.track_indices),
          loaded_at: Date.now(),
        });
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
  return (
    <div
      className={`rounded-md border px-2 py-1.5 text-xs transition-colors ${
        isPreviewing
          ? "border-cyan-400/60 bg-cyan-500/10 shadow-[0_0_12px_rgba(34,211,238,0.25)]"
          : "border-neutral-800/70 bg-neutral-900/40"
      }`}
    >
      <div className="flex items-baseline gap-1.5">
        <span className="font-mono text-[10px] text-neutral-500 tabular-nums">
          {Math.round(rec.score * 100)}
        </span>
        <span className="truncate flex-1 text-neutral-100">
          {rec.track_title ?? `Track #${rec.track_id}`}
        </span>
      </div>
      <div className="text-[10px] text-neutral-500 truncate">
        {rec.track_artist ?? "—"}
        {rec.bpm != null && (
          <span className="font-mono ml-1.5">· {rec.bpm.toFixed(1)} BPM</span>
        )}
        {rec.key_camelot && (
          <span className="font-mono ml-1.5">· {rec.key_camelot}</span>
        )}
        {energy != null && (
          <span className="font-mono ml-1.5 text-neutral-600">· E{energy}</span>
        )}
      </div>
      {rec.reasons.length > 0 && (
        <div className="text-[9px] text-neutral-600 uppercase tracking-wide mt-0.5 truncate">
          {rec.reasons.join(" · ")}
        </div>
      )}
      <div className="mt-1 flex gap-1">
        <button
          type="button"
          onClick={onPreview}
          disabled={startPreview.isPending || stopPreview.isPending}
          title={
            isPreviewing
              ? "Stop preview (Cue track → headphones)"
              : "Preview in headphones only (Scarlett outs 3/4)"
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
          className="flex-1 text-[10px] rounded bg-violet-700/70 hover:bg-violet-700 text-white py-1 transition-colors disabled:opacity-50"
        >
          {load.isPending
            ? "loading…"
            : isSongCard
            ? "Load song"
            : `Load ${roleLabel(column).toLowerCase()}`}
        </button>
      </div>
    </div>
  );
}
