import { clipPlayhead, clipPlayheadFromElapsed } from "../lib/seek";
import { useAbletonState } from "./useAbletonState";

export interface ClipPlayhead {
  /** 0-1 position along the clip; ``undefined`` when the clip isn't playing
   * (so the caller renders no playhead rather than a fabricated one). */
  position: number | undefined;
  /** Seconds into the (looped) clip; ``undefined`` when not playing. */
  elapsedSec: number | undefined;
  /** Seconds until the clip loops/ends; ``null`` when not playing. */
  remaining: number | null;
  /** True when the position came from Live's real per-clip position. */
  live: boolean;
}

/**
 * THE single source of truth for "where is this clip playing." Reads Live's
 * per-clip ``playing_position`` (track_index → beats, pushed over the WS while
 * the clip fires) and converts it onto the clip's original-audio timeline via
 * the warp BPM + duration — the same conversion a click-to-seek uses, so the
 * playhead and the seek target agree to the pixel.
 *
 * Used by BOTH the Booth deck waveforms and the cue/preview strip. Centralising
 * it here means a bug in playhead placement is fixed in exactly one spot.
 *
 * Deliberately renders NO playhead when there's no live clip position. The old
 * per-surface code fabricated a playhead from the master transport beat when it
 * lacked the clip's own position — but the master beat has no relationship to
 * an independently-fired (cue) or warped clip, so that "playhead" sat frozen or
 * wrong. Honest absence beats a confident lie.
 */
export function useClipPlayhead({
  trackIdx,
  durationSec,
  clipBpm,
  positionUnit = "beats",
}: {
  /** The Live track index whose firing clip drives the playhead. */
  trackIdx: number | null | undefined;
  /** The displayed audio's duration in seconds. */
  durationSec: number | null | undefined;
  /** The BPM the clip is warped at (the track's analyzed BPM); falls back to
   * the project tempo. Only used when ``positionUnit === "beats"``. */
  clipBpm: number | null | undefined;
  /** Unit Live reports ``playing_position`` in: WARPED clips (deck columns)
   * report "beats"; UNWARPED clips (the cue/preview clip) report "seconds". */
  positionUnit?: "beats" | "seconds";
}): ClipPlayhead {
  const ableton = useAbletonState();
  const livePos =
    trackIdx != null ? ableton.playing_positions[String(trackIdx)] : undefined;
  const bpm = clipBpm ?? ableton.tempo;

  let v = null;
  if (livePos != null && durationSec != null) {
    v =
      positionUnit === "seconds"
        ? clipPlayheadFromElapsed(livePos, durationSec)
        : bpm != null
          ? clipPlayhead(livePos, durationSec, bpm)
          : null;
  }

  return {
    position: v?.position,
    elapsedSec: v?.elapsedSec,
    remaining: v?.remaining ?? null,
    live: v != null,
  };
}
