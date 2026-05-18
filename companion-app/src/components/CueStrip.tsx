import { useMutation, useQueryClient } from "@tanstack/react-query";
import { pushTrackToLive } from "../api";
import { useAbletonState } from "../hooks/useAbletonState";
import { useStopPreview, usePreviewState } from "../hooks/usePreview";
import { useTrack } from "../hooks/useTracks";
import { useTrackWaveform } from "../hooks/useWaveform";
import { roleLabel } from "../lib/roles";
import { store } from "../store";
import { RoleIcon } from "./RoleIcon";
import { Waveform } from "./Waveform";

/**
 * Cue strip — the parallel surface to ComboStrip for *what's in headphones*.
 *
 * Materializes only when something is being auditioned through the Cue
 * track (Scarlett 4i4 outs 3/4 → headphones). Shows the source-track
 * metadata + two actions:
 *
 *   - ⏹  stop preview — silences the Cue, leaves master untouched.
 *   - →  commit to master — loads the previewed stem (or whole song if
 *        previewing from the Song column) into a real deck row.
 *
 * Visual language: cyan accents, matching the cyan ring used on rec
 * cards while previewing. Master content stays neutral; cue stands out.
 */
export function CueStrip() {
  const previewing = usePreviewState();
  const stopPreview = useStopPreview();
  const trackQuery = useTrack(previewing?.trackId);
  const waveform = useTrackWaveform(previewing?.trackId);
  const ableton = useAbletonState();
  const qc = useQueryClient();

  const commit = useMutation({
    mutationFn: () => {
      if (!previewing) throw new Error("nothing to commit");
      const kinds =
        previewing.column === "mix" ? undefined : [previewing.column];
      return pushTrackToLive(previewing.trackId, {
        includeStems: true,
        kinds,
      });
    },
    onSuccess: (result) => {
      if (!previewing) return;
      // Whole-song commits register a deck locally; single-stem commits
      // don't (the row may already have cells from other tracks).
      if (previewing.column === "mix") {
        store.registerDeck({
          track_id: previewing.trackId,
          scene_index: result.scene_index,
          stem_track_indices: Object.values(result.track_indices),
          loaded_at: Date.now(),
        });
      }
      stopPreview.mutate();
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
      qc.invalidateQueries({ queryKey: ["recommend", "by-column"] });
    },
  });

  if (!previewing) return null;

  const track = trackQuery.data;
  const analysis = track?.analysis ?? null;
  const isSong = previewing.column === "mix";
  const commitLabel = isSong ? "→ Load song to master" : `→ Load ${roleLabel(previewing.column).toLowerCase()} to master`;

  // Playhead position 0-1. Derive from Live's master beat clock + the audio's
  // own duration (waveform endpoint reports it). Wraps because preview clips
  // loop by default. Without tempo/beat we just hide the playhead.
  const tempo = ableton.tempo;
  const beat = ableton.beat;
  let playheadPosition: number | undefined = undefined;
  if (
    tempo != null &&
    beat != null &&
    waveform.data?.duration_seconds &&
    waveform.data.duration_seconds > 0
  ) {
    const elapsedSec = (beat / tempo) * 60;
    playheadPosition = (elapsedSec % waveform.data.duration_seconds) /
      waveform.data.duration_seconds;
  }

  return (
    <div
      className="rounded-md border border-cyan-400/40 bg-cyan-500/[0.06] px-3 py-2 flex items-center gap-3 shadow-[0_0_16px_rgba(34,211,238,0.12)]"
      data-testid="cue-strip"
    >
      <div className="flex items-center gap-2 shrink-0 min-w-[9rem]">
        <span className="text-[10px] uppercase tracking-widest text-cyan-300/80 font-semibold">
          In Cue
        </span>
        <span className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-cyan-300">
          <RoleIcon role={previewing.column} size={12} />
          {isSong ? "song" : roleLabel(previewing.column)}
        </span>
      </div>
      <div className="flex-1 min-w-0 flex flex-col gap-1">
        <div className="flex items-baseline gap-2">
          <div className="text-sm text-neutral-100 truncate font-medium flex-1 min-w-0">
            {track?.title ?? `Track #${previewing.trackId}`}
          </div>
          <div className="text-[11px] text-neutral-500 truncate font-mono shrink-0">
            {track?.artist ? track.artist : "—"}
            {analysis?.bpm != null && (
              <span className="ml-1.5">· {analysis.bpm.toFixed(1)} BPM</span>
            )}
            {analysis?.key_camelot && (
              <span className="ml-1.5">· {analysis.key_camelot}</span>
            )}
            {analysis?.floor_energy != null && (
              <span className="ml-1.5 text-neutral-600">
                · E{analysis.floor_energy}
              </span>
            )}
          </div>
        </div>
        <Waveform
          peaks={waveform.data?.peaks ?? []}
          position={playheadPosition}
          height={24}
          className="text-cyan-400/70"
          playheadColor="rgb(34, 211, 238)"
        />
      </div>
      <div className="shrink-0 flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => stopPreview.mutate()}
          disabled={stopPreview.isPending}
          className="h-8 px-2.5 rounded text-[11px] bg-neutral-900 hover:bg-neutral-800 text-neutral-300 border border-neutral-800 transition-colors disabled:opacity-50"
          title="Stop preview — silences the Cue track, leaves master untouched"
        >
          ⏹ stop
        </button>
        <button
          type="button"
          onClick={() => commit.mutate()}
          disabled={commit.isPending || !track}
          className="h-8 px-3 rounded text-[11px] bg-cyan-500/80 hover:bg-cyan-500 text-neutral-950 font-semibold transition-colors disabled:opacity-50"
          title="Promote this candidate from cue to master"
        >
          {commit.isPending ? "loading…" : commitLabel}
        </button>
      </div>
    </div>
  );
}
