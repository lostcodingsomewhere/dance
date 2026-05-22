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

/** Section_label → solid color for the start-marker line + label text.
 * Punchier than SECTION_FILL so the user can read "this is a DROP coming
 * up" at a glance against the muted band. */
const SECTION_ACCENT: Record<string, { stroke: string; label: string }> = {
  intro:     { stroke: "rgba(212, 212, 216, 0.85)", label: "rgba(228, 228, 231, 0.95)" },
  buildup:   { stroke: "rgba(250, 204, 21, 0.85)",  label: "rgba(253, 224, 71, 0.95)" },
  drop:      { stroke: "rgba(248, 113, 113, 0.95)", label: "rgba(252, 165, 165, 0.95)" },
  breakdown: { stroke: "rgba(34, 211, 238, 0.85)",  label: "rgba(103, 232, 249, 0.95)" },
  bridge:    { stroke: "rgba(192, 132, 252, 0.85)", label: "rgba(216, 180, 254, 0.95)" },
  outro:     { stroke: "rgba(212, 212, 216, 0.85)", label: "rgba(228, 228, 231, 0.95)" },
  verse:     { stroke: "rgba(96, 165, 250, 0.85)",  label: "rgba(147, 197, 253, 0.95)" },
  chorus:    { stroke: "rgba(251, 191, 36, 0.85)",  label: "rgba(253, 224, 71, 0.95)" },
  other:     { stroke: "rgba(163, 163, 163, 0.7)",  label: "rgba(212, 212, 212, 0.9)" },
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
            const accent = SECTION_ACCENT[label] ?? SECTION_ACCENT.other;
            return {
              id: r.id,
              x,
              w,
              fill,
              label,
              markerStroke: accent.stroke,
              labelFill: accent.label,
            };
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
      {/* Section start-line + label badge at the top of each band so
          DJs can see WHAT section is coming up, not just a colored blob.
          Label uses the section_label (drop / buildup / breakdown / ...)
          truncated to first 4 chars to fit narrow combo cards. */}
      {sectionBands.map((b) => (
        <g key={`section-marker-${b.id}`} data-testid="waveform-section-marker">
          <line
            x1={b.x}
            x2={b.x}
            y1={0}
            y2={H}
            stroke={b.markerStroke}
            strokeWidth="0.4"
          />
          {b.w > 6 && (
            <text
              x={b.x + 0.6}
              y={H * 0.32}
              fill={b.labelFill}
              fontSize={H * 0.36}
              fontFamily="ui-monospace, monospace"
              fontWeight="600"
              style={{ textTransform: "uppercase", letterSpacing: "0.5px" }}
            >
              {b.label.slice(0, 4)}
            </text>
          )}
        </g>
      ))}
      {/* Cue / phrase ticks — solid lines for hard boundaries, with a
          small downward-pointing triangle at the top so they read as
          "markers" rather than just stripes. */}
      {cueTicks.map((t) => (
        <g key={`cue-${t.id}`} data-testid="waveform-cue">
          <line
            x1={t.x}
            x2={t.x}
            y1={H * 0.15}
            y2={H}
            stroke="rgba(255,255,255,0.55)"
            strokeWidth="0.35"
          />
          <polygon
            points={`${t.x - 0.7},0 ${t.x + 0.7},0 ${t.x},${H * 0.18}`}
            fill="rgba(255,255,255,0.8)"
          />
        </g>
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
