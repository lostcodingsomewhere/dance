export interface WaveformProps {
  peaks: number[]; // normalized 0-1 amplitudes
  position?: number; // playhead position 0-1; null/undefined → no playhead
  className?: string; // wrapper className (Tailwind from caller)
  barColor?: string; // CSS color for bars; default "currentColor"
  playheadColor?: string; // CSS color for playhead; default "white"
  height?: number; // px; default 32
}

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
}: WaveformProps) {
  const H = height;
  const hasPeaks = peaks && peaks.length > 0;

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
      style={{ display: "block", color: barColor }}
    >
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
