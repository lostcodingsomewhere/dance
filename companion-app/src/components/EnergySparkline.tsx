import { useMemo } from "react";
import { useCurrentSession } from "../hooks/useSession";

const W = 110;
const H = 20;

/**
 * Tiny inline energy arc — designed to sit in the MasterStrip beside BPM /
 * KEY / heartbeat as a glance-anywhere set-arc cue. One point per logged
 * play; missing energy values are skipped.
 *
 * Empty / single-point sets render as a faint baseline placeholder so the
 * strip layout stays stable from session start.
 */
export function EnergySparkline() {
  const session = useCurrentSession();
  const plays = session.data?.plays ?? [];

  const path = useMemo(() => {
    const points: { x: number; y: number }[] = [];
    const valid = plays.filter((p) => p.energy_at_play != null);
    if (valid.length === 0) return null;
    valid.forEach((p, i) => {
      const x =
        valid.length === 1 ? W / 2 : (i / Math.max(1, valid.length - 1)) * W;
      const y = H - (p.energy_at_play! / 10) * H;
      points.push({ x, y });
    });
    return points
      .map(
        (pt, i) => `${i === 0 ? "M" : "L"} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`,
      )
      .join(" ");
  }, [plays]);

  const playCount = plays.length;

  return (
    <div
      className="flex items-center gap-2 px-2 border-l border-neutral-800 h-10"
      title={
        playCount === 0
          ? "Energy arc — populates as the session plays tracks"
          : `Energy arc — ${playCount} play${playCount === 1 ? "" : "s"} so far`
      }
    >
      <span className="text-neutral-500 text-[10px] uppercase tracking-wider">
        Arc
      </span>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        preserveAspectRatio="none"
        aria-hidden
      >
        {/* baseline at y=H/2 — visual anchor when the arc is empty */}
        <line
          x1={0}
          y1={H / 2}
          x2={W}
          y2={H / 2}
          stroke="#262626"
          strokeWidth={1}
          strokeDasharray="2 3"
        />
        {path && (
          <path
            d={path}
            fill="none"
            stroke="#fbbf24"
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
      </svg>
      <span className="text-[10px] font-mono text-neutral-500 tabular-nums">
        {playCount}
      </span>
    </div>
  );
}
