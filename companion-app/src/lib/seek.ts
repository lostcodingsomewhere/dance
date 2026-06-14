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

export interface ClipPlayheadValues {
  /** 0-1 position of the playhead along the clip's waveform. */
  position: number;
  /** Seconds elapsed into the (looped) clip, wrapped to [0, durationSec). */
  elapsedSec: number;
  /** Seconds remaining until the clip loops/ends. */
  remaining: number;
}

/**
 * Map a clip's live ``playing_position`` (in beats, at the clip's warp ``bpm``)
 * onto a 0-1 ratio along its ORIGINAL-audio waveform, plus the matching
 * elapsed/remaining seconds. The inverse of {@link ratioToBeats}, so the
 * playhead lands exactly where a click of the same ratio would seek to.
 *
 * This is THE single conversion behind every playhead in the app (deck
 * waveforms + cue/preview strip). Using one function means "where's it
 * playing" can only ever be right or wrong in one place.
 *
 * Returns ``null`` when the inputs can't place a playhead (no duration / bpm).
 * Wraps defensively so a looped clip's position stays in-range.
 */
export function clipPlayhead(
  positionBeats: number,
  durationSec: number,
  bpm: number,
): ClipPlayheadValues | null {
  if (!(durationSec > 0) || !(bpm > 0)) return null;
  const elapsedRaw = (positionBeats / bpm) * 60;
  // Modulo can go negative for negative inputs; normalise into [0, dur).
  const elapsedSec = ((elapsedRaw % durationSec) + durationSec) % durationSec;
  return {
    position: elapsedSec / durationSec,
    elapsedSec,
    remaining: durationSec - elapsedSec,
  };
}
