import type { Region } from "../types";

/**
 * Snap a click-ratio (0-1) along a waveform to the *nearest* section start,
 * within ``toleranceRatio``. Outside tolerance → return the click ratio
 * verbatim so the user can still seek into the gap past the last section.
 *
 * The pre-snap rule ("nearest section ≤ ratio") made clicks past the last
 * detected section snap backward to ~50%, which felt like "seek doesn't
 * work past halfway" on tracks with sparse section detection.
 */
export function snapRatioToSection(
  ratio: number,
  regions: Region[] | undefined,
  durationSec: number,
  toleranceRatio = 0.08,
): number {
  if (!regions || !durationSec || durationSec <= 0) return ratio;
  let best: number | null = null;
  let bestDist = Infinity;
  for (const r of regions) {
    if (r.region_type !== "section") continue;
    const s = r.position_ms / 1000 / durationSec;
    const d = Math.abs(s - ratio);
    if (d < bestDist) {
      bestDist = d;
      best = s;
    }
  }
  return best != null && bestDist <= toleranceRatio ? best : ratio;
}

/** Convert a 0-1 ratio along a clip to beats, given clip duration in seconds
 * and the BPM the clip is warped at. */
export function ratioToBeats(
  ratio: number,
  durationSec: number,
  bpm: number,
): number {
  return ratio * durationSec * (bpm / 60);
}
