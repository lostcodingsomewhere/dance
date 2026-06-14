import { SCORE_FEATURES } from "../types";

/**
 * Renders a rec's ``score_breakdown`` as a row of compact per-feature chips,
 * in the canonical feature order (vibe · key · bpm · energy · kick · presence ·
 * timbre · mix). One source of truth shared by the Booth column banners, the
 * Set Rail tail-recs, and the deck-move next-move recs — so every surface
 * explains a score the same way.
 *
 * Only features actually present in the breakdown render (the scorer drops
 * features it can't compute), so the chip set tells you *which signals fired*
 * for this candidate, not just the blended number.
 */
export function ScoreBreakdown({
  breakdown,
  size = "md",
  className = "",
}: {
  breakdown: Record<string, number>;
  size?: "sm" | "md";
  className?: string;
}) {
  const present = SCORE_FEATURES.filter((f) => breakdown[f.key] != null);
  if (present.length === 0) return null;

  const pad = size === "sm" ? "px-1 py-[1px] text-[9px]" : "px-1.5 py-0.5 text-[10px]";

  return (
    <div className={`flex flex-wrap gap-1 ${className}`}>
      {present.map((f) => {
        const v = breakdown[f.key];
        const pct = Math.round(Math.max(0, Math.min(1, v)) * 100);
        return (
          <span
            key={f.key}
            title={`${f.label}: ${pct}`}
            className={`inline-flex items-center gap-1 rounded ring-1 font-mono ${pad} ${f.color}`}
          >
            <span className="opacity-80">{f.short}</span>
            <span className="tabular-nums">{pct}</span>
          </span>
        );
      })}
    </div>
  );
}
