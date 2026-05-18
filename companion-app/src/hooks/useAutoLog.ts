import { useEffect, useRef } from "react";
import { useNowPlayingTrack } from "./useNowPlayingTrack";
import { useAddPlay, useCurrentSession } from "./useSession";

/**
 * Log a play in the current session each time the playing track changes.
 * Idempotent: if the same track keeps playing, we only log it once. If the
 * same track plays again later in the set, we *do* log it again (DJs replay
 * tracks intentionally — but only after a 90s cooldown to absorb flicker).
 */
const REPLAY_COOLDOWN_MS = 90_000;

export function useAutoLog(): void {
  const { trackId, source } = useNowPlayingTrack();
  const session = useCurrentSession();
  const addPlay = useAddPlay(session.data?.id ?? null);
  const lastLoggedRef = useRef<{ trackId: number; at: number } | null>(null);

  useEffect(() => {
    if (trackId == null) return;
    // We only log inferred-from-Ableton plays. Session-derived "now playing"
    // already came from a logged play, so re-logging it would double-count.
    if (source !== "ableton") return;
    if (session.data == null) return;

    const now = Date.now();
    const last = lastLoggedRef.current;
    if (last && last.trackId === trackId && now - last.at < REPLAY_COOLDOWN_MS) {
      return;
    }

    lastLoggedRef.current = { trackId, at: now };
    addPlay.mutate({ track_id: trackId });
  }, [trackId, source, session.data, addPlay]);
}
