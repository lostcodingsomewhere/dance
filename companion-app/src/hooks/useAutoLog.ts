import { useEffect, useRef } from "react";
import { useNowPlayingTrack } from "./useNowPlayingTrack";
import { useAddPlay, useCurrentSession } from "./useSession";

const REPLAY_COOLDOWN_MS = 90_000;

interface HeardInSession {
  sessionId: number;
  lastHeardAt: Map<number, number>;
}

export function useAutoLog(): void {
  const { trackId, source } = useNowPlayingTrack();
  const session = useCurrentSession().data;
  const sessionId = session?.id ?? null;
  const lastLoggedTrackId = session?.plays.at(-1)?.track_id ?? null;
  const { mutate: logPlay } = useAddPlay(sessionId);
  const heardRef = useRef<HeardInSession | null>(null);

  useEffect(() => {
    if (trackId == null || source !== "ableton" || sessionId == null) return;
    if (heardRef.current?.sessionId !== sessionId) {
      heardRef.current = { sessionId, lastHeardAt: new Map() };
    }
    const { lastHeardAt } = heardRef.current;
    const heardAt = lastHeardAt.get(trackId);
    const isSamePlay =
      heardAt == null
        ? lastLoggedTrackId === trackId
        : Date.now() - heardAt < REPLAY_COOLDOWN_MS;
    if (!isSamePlay) logPlay({ track_id: trackId });
    return () => {
      lastHeardAt.set(trackId, Date.now());
    };
  }, [trackId, source, sessionId, lastLoggedTrackId, logPlay]);
}
