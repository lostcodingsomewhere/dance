import { useMemo } from "react";
import { SceneMap } from "./SceneMap";
import { useCurrentSession, useEndSession } from "../hooks/useSession";
import type { SessionPlay } from "../types";

/**
 * Left rail of The Booth: the set arc so far. Energy curve + recent plays.
 * Replaces the old standalone "Session" tab. Session creation is automatic
 * (see useAutoSession), but the rail also exposes an explicit Start button
 * for the case where Ableton hasn't been touched yet.
 */
export function SetRail() {
  const session = useCurrentSession();
  const endSession = useEndSession();

  if (!session.data) {
    return (
      <aside className="w-72 shrink-0 border-r border-neutral-800 flex flex-col min-h-0">
        <div className="p-4 pb-2 flex flex-col gap-1">
          <h2 className="text-xs uppercase tracking-widest text-neutral-500">
            Set
          </h2>
          <div className="text-neutral-500 text-sm">
            Press play in Ableton to start a set.
          </div>
        </div>
        <div className="border-t border-neutral-800/60">
          <SceneMap />
        </div>
      </aside>
    );
  }

  const dj = session.data;
  const plays = dj.plays;
  return (
    <aside className="w-72 shrink-0 border-r border-neutral-800 flex flex-col min-h-0">
      <div className="px-4 pt-4 pb-2">
        <div className="flex items-baseline justify-between">
          <h2 className="text-xs uppercase tracking-widest text-neutral-500">
            {dj.name ?? "Set"}
          </h2>
          <button
            type="button"
            onClick={() => endSession.mutate(dj.id)}
            className="text-[11px] text-neutral-500 hover:text-rose-400"
            title="End this set — next press-play in Ableton starts a new one"
          >
            end set
          </button>
        </div>
        <div className="text-xs text-neutral-500 mt-1">
          {plays.length} {plays.length === 1 ? "play" : "plays"} · started{" "}
          {formatTime(dj.started_at)}
        </div>
      </div>

      <SceneMap />

      <div className="border-t border-neutral-800/60 mt-1 pt-2 px-4">
        <div className="text-[10px] uppercase tracking-widest text-neutral-500 mb-1">
          Played
        </div>
      </div>

      <EnergyArc plays={plays} />

      <div className="flex-1 overflow-y-auto px-3 pb-4 flex flex-col gap-1.5">
        {plays.length === 0 && (
          <div className="px-2 py-3 text-xs text-neutral-600 italic">
            Plays appear here automatically as Ableton fires clips.
          </div>
        )}
        {plays.map((p) => (
          <div
            key={`${p.position_in_set}-${p.track_id}-${p.played_at}`}
            className="rounded-md border border-neutral-800/60 bg-neutral-900/40 px-2 py-1.5"
          >
            <div className="flex items-baseline gap-2">
              <span className="text-[10px] font-mono text-neutral-500 w-5 text-right shrink-0">
                {p.position_in_set}
              </span>
              <div className="text-sm text-neutral-100 truncate flex-1">
                {p.title ?? `Track #${p.track_id}`}
              </div>
              <span className="text-[10px] font-mono text-neutral-500">
                {p.energy_at_play != null ? `E${p.energy_at_play}` : "--"}
              </span>
            </div>
            <div className="ml-7 text-[11px] text-neutral-500 truncate">
              {p.artist ?? "—"}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "--";
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function EnergyArc({ plays }: { plays: SessionPlay[] }) {
  const W = 240;
  const H = 60;
  const points = useMemo(() => {
    const pts: { x: number; y: number; e: number }[] = [];
    plays.forEach((p, i) => {
      if (p.energy_at_play == null) return;
      const x =
        plays.length === 1 ? W / 2 : (i / Math.max(1, plays.length - 1)) * W;
      const y = H - (p.energy_at_play / 10) * H;
      pts.push({ x, y, e: p.energy_at_play });
    });
    return pts;
  }, [plays]);

  if (points.length === 0) {
    return (
      <div className="mx-3 my-2 h-14 rounded border border-dashed border-neutral-800/60 flex items-center justify-center text-[10px] text-neutral-600">
        Energy arc appears as plays come in
      </div>
    );
  }

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(" ");

  return (
    <div className="mx-3 my-2 rounded border border-neutral-800/60 bg-neutral-900/40 p-2">
      <div className="text-[10px] text-neutral-500 uppercase tracking-widest mb-1">
        Energy arc
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-14" preserveAspectRatio="none">
        <path
          d={path}
          fill="none"
          stroke="#fbbf24"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={2} fill="#fbbf24" />
        ))}
      </svg>
    </div>
  );
}
