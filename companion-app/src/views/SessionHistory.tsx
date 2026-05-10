import { useMemo } from "react";
import {
  useCreateSession,
  useCurrentSession,
  useEndSession,
} from "../hooks/useSession";
import type { SessionPlay } from "../types";

export function SessionHistory() {
  const session = useCurrentSession();
  const createSession = useCreateSession();
  const endSession = useEndSession();

  if (session.isLoading) {
    return (
      <div className="flex-1 p-8 text-neutral-500">Loading session…</div>
    );
  }

  if (!session.data) {
    return (
      <div className="flex-1 flex items-center justify-center text-center px-8">
        <div>
          <div className="text-3xl text-neutral-200 font-semibold">
            No active session
          </div>
          <div className="mt-2 text-neutral-500">
            Start one to track plays, energy arc, and transitions.
          </div>
          <button
            type="button"
            onClick={() => createSession.mutate(undefined)}
            className="mt-6 min-h-[56px] px-6 rounded-lg bg-amber-400 text-neutral-950 font-semibold text-lg hover:bg-amber-300"
          >
            Start Session
          </button>
        </div>
      </div>
    );
  }

  const dj = session.data;
  return (
    <div className="flex-1 flex flex-col gap-5 p-6 overflow-auto">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold text-neutral-50">
            {dj.name ?? `Session #${dj.id}`}
          </h1>
          <div className="text-sm text-neutral-500">
            Started {formatTime(dj.started_at)} · {dj.plays.length} plays
          </div>
        </div>
        <button
          type="button"
          onClick={() => endSession.mutate(dj.id)}
          className="min-h-[48px] px-5 rounded-lg bg-red-500 text-neutral-950 font-semibold hover:bg-red-400"
        >
          End Session
        </button>
      </div>

      <EnergyArc plays={dj.plays} />

      <div className="flex flex-col gap-2">
        {dj.plays.length === 0 && (
          <div className="text-neutral-500">No plays yet.</div>
        )}
        {dj.plays.map((p) => (
          <div
            key={`${p.position_in_set}-${p.track_id}`}
            className="flex items-center gap-4 p-3 bg-neutral-900 border border-neutral-800 rounded-lg"
          >
            <div className="w-10 text-center text-neutral-500 font-mono">
              {p.position_in_set}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-neutral-100 truncate">
                {p.title ?? `Track #${p.track_id}`}
              </div>
              <div className="text-neutral-500 text-sm truncate">
                {p.artist ?? "—"}
                {p.transition_type ? ` · ${p.transition_type}` : ""}
              </div>
            </div>
            <div className="font-mono text-neutral-300 text-sm">
              E {p.energy_at_play ?? "--"}
            </div>
            <div className="font-mono text-neutral-500 text-xs">
              {formatTime(p.played_at)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "--";
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function EnergyArc({ plays }: { plays: SessionPlay[] }) {
  const W = 600;
  const H = 100;
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
      <div className="h-24 rounded-lg border border-dashed border-neutral-800 flex items-center justify-center text-neutral-600 text-sm">
        Energy arc appears once plays have analysis data.
      </div>
    );
  }

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(" ");

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-3">
      <div className="text-xs text-neutral-500 uppercase tracking-widest mb-2">
        Energy arc
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-24"
        preserveAspectRatio="none"
      >
        <path
          d={path}
          fill="none"
          stroke="#fbbf24"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={3} fill="#fbbf24" />
        ))}
      </svg>
    </div>
  );
}
