import { useMemo } from "react";
import { EnergyBar } from "../components/EnergyBar";
import { KeyBadge } from "../components/KeyBadge";
import { useAbletonState } from "../hooks/useAbletonState";
import { useCurrentSession } from "../hooks/useSession";
import { useStems, useTrack } from "../hooks/useTracks";

const STEM_COLORS: Record<string, string> = {
  drums: "bg-rose-500",
  bass: "bg-sky-500",
  vocals: "bg-amber-400",
  other: "bg-violet-500",
};

export function NowPlaying() {
  const ableton = useAbletonState();
  const session = useCurrentSession();

  // Current track: most recent SessionPlay, otherwise null.
  // (Mapping a playing Ableton clip back to a track requires a clip-to-track
  // map that the backend doesn't yet expose; that's a Phase-2.4+ concern.)
  const lastPlay = session.data?.plays.at(-1);
  const trackId = lastPlay?.track_id ?? null;
  const track = useTrack(trackId);
  const stems = useStems(trackId);

  const beatLabel = useMemo(() => {
    const beat = ableton.beat;
    if (beat == null) return "--";
    const bar = Math.floor(beat / 4) + 1;
    const within = (beat % 4) + 1;
    return `${bar} . ${within.toFixed(0)}`;
  }, [ableton.beat]);

  if (trackId == null) {
    return (
      <div className="flex-1 flex items-center justify-center text-center px-8">
        <div>
          <div className="text-3xl text-neutral-300 font-semibold">
            Nothing playing
          </div>
          <div className="mt-2 text-neutral-500">
            Pin a track in Up Next, or start a session and add a play.
          </div>
        </div>
      </div>
    );
  }

  const t = track.data;
  const analysis = t?.analysis ?? null;

  return (
    <div className="flex-1 flex flex-col gap-6 p-8 overflow-auto">
      <div>
        <div className="text-neutral-500 uppercase tracking-widest text-xs mb-2">
          Now playing
        </div>
        <div className="text-5xl font-bold text-neutral-50">
          {t?.title ?? `Track #${trackId}`}
        </div>
        <div className="text-2xl text-neutral-400 mt-1">
          {t?.artist ?? "Unknown artist"}
        </div>
      </div>

      <div className="flex items-end gap-10">
        <div>
          <div className="text-neutral-500 uppercase text-xs tracking-widest">
            BPM
          </div>
          <div className="font-mono text-7xl text-neutral-50 tabular-nums leading-none">
            {analysis?.bpm != null ? analysis.bpm.toFixed(1) : "--"}
          </div>
        </div>
        <div>
          <div className="text-neutral-500 uppercase text-xs tracking-widest mb-1">
            Key
          </div>
          <KeyBadge keyCamelot={analysis?.key_camelot ?? null} size="lg" />
        </div>
        <div>
          <div className="text-neutral-500 uppercase text-xs tracking-widest mb-1">
            Energy
          </div>
          <EnergyBar energy={analysis?.floor_energy ?? null} size="lg" />
        </div>
        <div>
          <div className="text-neutral-500 uppercase text-xs tracking-widest">
            Beat
          </div>
          <div className="font-mono text-5xl text-neutral-50 tabular-nums leading-none">
            {beatLabel}
          </div>
        </div>
      </div>

      <div>
        <div className="text-neutral-500 uppercase text-xs tracking-widest mb-2">
          Stems
        </div>
        <div className="grid grid-cols-4 gap-3 max-w-3xl">
          {["drums", "bass", "vocals", "other"].map((kind) => {
            const stem = stems.data?.find((s) => s.kind === kind);
            const presence = stem?.analysis?.presence_ratio ?? null;
            const pct = presence != null ? Math.round(presence * 100) : null;
            return (
              <div
                key={kind}
                className="bg-neutral-900 border border-neutral-800 rounded-xl p-4"
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`w-3 h-3 rounded-full ${
                      stem ? STEM_COLORS[kind] ?? "bg-neutral-500" : "bg-neutral-800"
                    }`}
                  />
                  <div className="capitalize text-neutral-200 font-semibold">
                    {kind}
                  </div>
                </div>
                <div className="mt-2 font-mono text-2xl text-neutral-100">
                  {pct == null ? "--" : `${pct}%`}
                </div>
                <div className="text-xs text-neutral-500">
                  {stem ? "present" : "no stem"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <BeatStrip beat={ableton.beat} />
    </div>
  );
}

function BeatStrip({ beat }: { beat: number | null }) {
  // Simple 16-beat strip (4 bars) with the playhead position.
  const totalBeats = 16;
  const pos = beat == null ? 0 : ((beat % totalBeats) / totalBeats) * 100;
  return (
    <div className="mt-2">
      <div className="text-neutral-500 uppercase text-xs tracking-widest mb-2">
        Beat position
      </div>
      <div className="relative h-3 rounded-full bg-neutral-900 border border-neutral-800 overflow-hidden">
        {Array.from({ length: totalBeats }).map((_, i) => (
          <div
            key={i}
            className="absolute top-0 bottom-0 w-px bg-neutral-800"
            style={{ left: `${(i / totalBeats) * 100}%` }}
          />
        ))}
        {beat != null && (
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-amber-400 shadow"
            style={{ left: `calc(${pos}% - 6px)` }}
          />
        )}
      </div>
    </div>
  );
}
