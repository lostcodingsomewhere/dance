import type React from "react";
import type { Region } from "../types";

export interface WaveformProps {
  peaks: number[]; // normalized 0-1 amplitudes
  position?: number; // playhead position 0-1; null/undefined → no playhead
  className?: string; // wrapper className (Tailwind from caller)
  barColor?: string; // CSS color for bars; default "currentColor"
  playheadColor?: string; // CSS color for playhead; default "white"
  height?: number; // px; default 32
  /** Click handler — receives the click's position as a 0-1 ratio along
   * the waveform's horizontal axis. When set, the cursor becomes a
   * pointer to signal interactivity. */
  onSeek?: (ratio: number) => void;
  /** Region overlay: section bands + cue tick marks rendered behind the
   * peaks. Sections show as faint colored regions by section_label; cues
   * show as thin vertical ticks. ``durationSeconds`` is required to map
   * region.position_ms / length_ms into the waveform's 0-100 viewBox. */
  regions?: Region[];
  durationSeconds?: number;
}

/** Section_label → muted background color. Matches the StructureTimeline
 * convention from the pre-redesign component for visual consistency. */
const SECTION_FILL: Record<string, string> = {
  intro: "rgba(82, 82, 91, 0.25)",
  buildup: "rgba(234, 179, 8, 0.18)",
  drop: "rgba(239, 68, 68, 0.22)",
  breakdown: "rgba(6, 182, 212, 0.18)",
  bridge: "rgba(168, 85, 247, 0.20)",
  outro: "rgba(82, 82, 91, 0.25)",
  verse: "rgba(59, 130, 246, 0.18)",
  chorus: "rgba(245, 158, 11, 0.20)",
  other: "rgba(115, 115, 115, 0.15)",
};

/**
 * Pure presentational waveform. Renders an inline SVG of normalized peak
 * amplitudes that stretches to its container's width via
 * `preserveAspectRatio="none"`. Color defaults to `currentColor` so a Tailwind
 * `text-*` class on the wrapper inherits down to the bars.
 *
 * Used by ComboStrip (playing stems) and CueStrip (cueing stems) — same
 * primitive, different colors per context.
 */
export function Waveform({
  peaks,
  position,
  className,
  barColor = "currentColor",
  playheadColor = "white",
  height = 32,
  onSeek,
  regions,
  durationSeconds,
}: WaveformProps) {
  const H = height;
  const hasPeaks = peaks && peaks.length > 0;

  // Build region overlays once. Sections render as faint colored bands
  // *behind* the peaks (so the bars stay legible); cues render as thin
  // ticks at the section boundary. Both need durationSeconds to map ms →
  // viewBox x-coords (0-100). If duration is missing, skip overlays —
  // better than guessing.
  const totalMs = durationSeconds ? durationSeconds * 1000 : 0;
  const sectionBands =
    regions && totalMs > 0
      ? regions
          .filter((r) => r.region_type === "section" && r.length_ms != null)
          .map((r) => {
            const x = Math.max(0, Math.min(100, (r.position_ms / totalMs) * 100));
            const w = Math.max(
              0,
              Math.min(100 - x, ((r.length_ms ?? 0) / totalMs) * 100),
            );
            const label = r.section_label ?? "other";
            const fill = SECTION_FILL[label] ?? SECTION_FILL.other;
            return { id: r.id, x, w, fill };
          })
      : [];
  const cueTicks =
    regions && totalMs > 0
      ? regions
          .filter((r) => r.region_type === "cue" || r.region_type === "phrase")
          .map((r) => {
            const x = Math.max(0, Math.min(100, (r.position_ms / totalMs) * 100));
            return { id: r.id, x };
          })
      : [];

  function handleClick(e: React.MouseEvent<SVGSVGElement>) {
    if (!onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(ratio);
  }

  if (!hasPeaks) {
    return (
      <div
        className={className}
        data-testid="waveform-empty"
        style={{ height: H, display: "flex", alignItems: "center" }}
      >
        <svg
          viewBox={`0 0 100 ${H}`}
          preserveAspectRatio="none"
          width="100%"
          height={H}
          style={{ display: "block", color: barColor }}
        >
          <line
            x1="0"
            x2="100"
            y1={H / 2}
            y2={H / 2}
            stroke="currentColor"
            strokeWidth="0.5"
            strokeDasharray="2 2"
            opacity="0.3"
          />
        </svg>
      </div>
    );
  }

  const barWidth = 100 / peaks.length;
  const showPlayhead = typeof position === "number";
  const playX = showPlayhead ? Math.max(0, Math.min(1, position!)) * 100 : 0;

  return (
    <svg
      className={className}
      viewBox={`0 0 100 ${H}`}
      preserveAspectRatio="none"
      width="100%"
      height={H}
      style={{
        display: "block",
        color: barColor,
        cursor: onSeek ? "pointer" : undefined,
      }}
      onClick={onSeek ? handleClick : undefined}
    >
      {/* Section bands — rendered first so peaks draw on top. Each band is
          a faint colored rect spanning the section's time-range. */}
      {sectionBands.map((b) => (
        <rect
          key={`section-${b.id}`}
          data-testid="waveform-section"
          x={b.x}
          y={0}
          width={b.w}
          height={H}
          fill={b.fill}
        />
      ))}
      {/* Cue / phrase ticks — thin vertical lines for hard boundaries. */}
      {cueTicks.map((t) => (
        <line
          key={`cue-${t.id}`}
          data-testid="waveform-cue"
          x1={t.x}
          x2={t.x}
          y1={0}
          y2={H}
          stroke="rgba(255,255,255,0.35)"
          strokeWidth="0.25"
          strokeDasharray="1 1"
        />
      ))}
      {peaks.map((p, i) => {
        const clamped = Math.max(0, Math.min(1, p));
        const barH = Math.max(clamped * H, 2);
        const y = (H - barH) / 2;
        return (
          <rect
            key={i}
            data-testid="waveform-bar"
            x={i * barWidth}
            y={y}
            width={barWidth}
            height={barH}
            fill={barColor}
          />
        );
      })}
      {showPlayhead && (
        <line
          data-testid="waveform-playhead"
          x1={playX}
          x2={playX}
          y1={0}
          y2={H}
          stroke={playheadColor}
          strokeWidth="0.4"
        />
      )}
    </svg>
  );
}
