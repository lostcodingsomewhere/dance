import { EnergyBar } from "./EnergyBar";
import { KeyBadge } from "./KeyBadge";
import { StructureTimeline } from "./StructureTimeline";
import { useAbletonState } from "../hooks/useAbletonState";
import { useRegions } from "../hooks/useRegions";
import { useStems, useTrack } from "../hooks/useTracks";

const STEM_COLORS: Record<string, string> = {
  drums: "bg-rose-500",
  bass: "bg-sky-500",
  vocals: "bg-amber-400",
  other: "bg-violet-500",
};

interface Props {
  trackId: number | null;
  /** True if we derived the track from Ableton state (vs. last-session-play). */
  liveLinked: boolean;
}

export function NowCard({ trackId, liveLinked }: Props) {
  const ableton = useAbletonState();
  const track = useTrack(trackId);
  const stems = useStems(trackId);
  const regions = useRegions(trackId);

  if (trackId == null) {
    return (
      <div className="flex-1 flex items-center justify-center text-center px-8">
        <div>
          <div className="text-3xl text-neutral-300 font-semibold">
            Nothing playing
          </div>
          <div className="mt-2 text-neutral-500 max-w-md">
            Load a track to Live from the Up Next rail or the Crate, then fire
            a clip on your APC40. A session starts automatically.
          </div>
        </div>
      </div>
    );
  }

  const t = track.data;
  const analysis = t?.analysis ?? null;

  return (
    <div className="flex-1 flex flex-col gap-5 p-6 overflow-y-auto min-h-0">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-neutral-500 uppercase tracking-widest text-[11px]">
            Now playing
          </span>
          <span
            className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
              liveLinked
                ? "bg-emerald-500/15 text-emerald-300"
                : "bg-neutral-800 text-neutral-500"
            }`}
            title={
              liveLinked
                ? "Inferred from Ableton's playing clip"
                : "Inferred from the most recent logged play — not from Ableton state"
            }
          >
            {liveLinked ? "● LIVE" : "○ HISTORY"}
          </span>
        </div>
        <div className="text-4xl font-bold text-neutral-50 leading-tight">
          {t?.title ?? `Track #${trackId}`}
        </div>
        <div className="text-xl text-neutral-400 mt-0.5">
          {t?.artist ?? "Unknown artist"}
        </div>
      </div>

      <div className="flex items-end gap-8 flex-wrap">
        <div>
          <div className="text-neutral-500 uppercase text-[11px] tracking-widest">
            BPM
          </div>
          <div className="font-mono text-6xl text-neutral-50 tabular-nums leading-none">
            {analysis?.bpm != null ? analysis.bpm.toFixed(1) : "--"}
          </div>
        </div>
        <div>
          <div className="text-neutral-500 uppercase text-[11px] tracking-widest mb-1">
            Key
          </div>
          <KeyBadge keyCamelot={analysis?.key_camelot ?? null} size="lg" />
        </div>
        <div>
          <div className="text-neutral-500 uppercase text-[11px] tracking-widest mb-1">
            Energy
          </div>
          <EnergyBar energy={analysis?.floor_energy ?? null} size="lg" />
        </div>
      </div>

      <StructureTimeline
        regions={regions.data}
        durationSeconds={t?.duration_seconds ?? null}
        bpm={analysis?.bpm ?? ableton.tempo ?? null}
        beat={ableton.beat}
      />

      <div>
        <div className="text-neutral-500 uppercase text-[11px] tracking-widest mb-2">
          Stems
        </div>
        <div className="grid grid-cols-4 gap-3">
          {["drums", "bass", "vocals", "other"].map((kind) => {
            const stem = stems.data?.find((s) => s.kind === kind);
            const presence = stem?.analysis?.presence_ratio ?? null;
            const pct = presence != null ? Math.round(presence * 100) : null;
            return (
              <div
                key={kind}
                className="bg-neutral-900 border border-neutral-800 rounded-lg p-3"
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2.5 h-2.5 rounded-full ${
                      stem
                        ? STEM_COLORS[kind] ?? "bg-neutral-500"
                        : "bg-neutral-800"
                    }`}
                  />
                  <div className="capitalize text-neutral-200 font-semibold text-sm">
                    {kind}
                  </div>
                </div>
                <div className="mt-1.5 font-mono text-xl text-neutral-100">
                  {pct == null ? "--" : `${pct}%`}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {t?.tags && t.tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {t.tags.slice(0, 12).map((tag) => (
            <span
              key={tag}
              className="text-[11px] uppercase tracking-wide px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-300"
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
