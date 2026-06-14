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
 * and the BPM the clip is warped at. Used to seek WARPED clips (deck columns),
 * whose loop/start markers are in beats. */
export function ratioToBeats(
  ratio: number,
  durationSec: number,
  bpm: number,
): number {
  return ratio * durationSec * (bpm / 60);
}

/** Convert a 0-1 ratio along a clip to seconds. Used to seek UNWARPED clips
 * (the cue/preview clip), whose loop/start markers are in seconds. */
export function ratioToSeconds(ratio: number, durationSec: number): number {
  return ratio * durationSec;
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
 * THE single conversion behind every playhead in the app. Given the elapsed
 * seconds into a clip, returns the 0-1 ratio along its waveform + the wrapped
 * elapsed/remaining seconds. Wraps defensively so a looped clip stays in-range.
 *
 * Callers differ only in how they get ``elapsedSecRaw`` from Live's
 * ``playing_position``: WARPED clips (deck columns) report beats → use
 * {@link clipPlayhead}; UNWARPED clips (the cue/preview clip) already report
 * seconds → pass them straight in. One wrap/ratio impl, so "where's it playing"
 * can only ever be right or wrong in one place.
 */
export function clipPlayheadFromElapsed(
  elapsedSecRaw: number,
  durationSec: number,
): ClipPlayheadValues | null {
  if (!(durationSec > 0)) return null;
  // Modulo can go negative for negative inputs; normalise into [0, dur).
  const elapsedSec = ((elapsedSecRaw % durationSec) + durationSec) % durationSec;
  return {
    position: elapsedSec / durationSec,
    elapsedSec,
    remaining: durationSec - elapsedSec,
  };
}

/**
 * Map a WARPED clip's live ``playing_position`` (in beats, at the clip's warp
 * ``bpm``) onto a 0-1 ratio + elapsed/remaining. The inverse of
 * {@link ratioToBeats}, so the playhead lands exactly where a click of the same
 * ratio would seek to. Returns ``null`` when inputs can't place a playhead.
 */
export function clipPlayhead(
  positionBeats: number,
  durationSec: number,
  bpm: number,
): ClipPlayheadValues | null {
  if (!(bpm > 0)) return null;
  return clipPlayheadFromElapsed((positionBeats / bpm) * 60, durationSec);
}
