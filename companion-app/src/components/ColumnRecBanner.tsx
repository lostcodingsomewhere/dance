import { useColumnRecs } from "../hooks/useColumnRecs";
import { pushTrackToLive } from "../api";
import { useMutation } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import type { ColumnRec } from "../types";
import { store } from "../store";

const ROLE_LABEL: Record<string, string> = {
  drums: "Drums",
  bass: "Bass",
  vocals: "Vocals",
  other: "Other",
  mix: "Mix",
};

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
        <span className="font-semibold">{ROLE_LABEL[column] ?? column}</span>
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
        <RecCard key={`${rec.track_id}-${rec.stem_file_id ?? "mix"}`} rec={rec} />
      ))}
      {q.data && q.data.recs.length === 0 && !q.isLoading && (
        <div className="text-[10px] text-neutral-600 px-2 py-2 italic">
          No candidates yet
        </div>
      )}
    </div>
  );
}

function RecCard({ rec }: { rec: ColumnRec }) {
  const qc = useQueryClient();
  const load = useMutation({
    mutationFn: () => pushTrackToLive(rec.track_id, { includeStems: true }),
    onSuccess: (result) => {
      // Register the deck in local state so SceneMap + NowPlaying see it.
      store.registerDeck({
        track_id: rec.track_id,
        scene_index: result.scene_index,
        stem_track_indices: Object.values(result.track_indices),
        loaded_at: Date.now(),
      });
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
      qc.invalidateQueries({ queryKey: ["recommend", "by-column"] });
    },
  });

  const energy = rec.floor_energy;
  return (
    <div className="rounded-md border border-neutral-800/70 bg-neutral-900/40 px-2 py-1.5 text-xs">
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
      <button
        type="button"
        onClick={() => load.mutate()}
        disabled={load.isPending}
        className="mt-1 w-full text-[10px] rounded bg-violet-700/70 hover:bg-violet-700 text-white py-1 transition-colors disabled:opacity-50"
      >
        {load.isPending ? "loading…" : "Load → Live"}
      </button>
    </div>
  );
}
