import { useCallback, useEffect, useRef, useState } from "react";
import { useNowPlayingTrack } from "./useNowPlayingTrack";
import { useAddPlay, useCurrentSession } from "./useSession";

const REPLAY_COOLDOWN_MS = 90_000;
const RETRY_AFTER_MS = 5_000;

interface PlaysInSession {
  sessionId: number;
  lastHeardAt: Map<number, number>;
  logged: Set<number>;
  beingLogged: Set<number>;
}

function noPlaysYet(sessionId: number): PlaysInSession {
  return { sessionId, lastHeardAt: new Map(), logged: new Set(), beingLogged: new Set() };
}

function settleWhichPlaysAreLogged(
  plays: PlaysInSession,
  playingTrackIds: number[],
  lastLoggedTrackId: number | null,
): void {
  for (const trackId of playingTrackIds) {
    const heardAt = plays.lastHeardAt.get(trackId);
    if (heardAt == null) {
      if (trackId === lastLoggedTrackId) plays.logged.add(trackId);
    } else if (Date.now() - heardAt >= REPLAY_COOLDOWN_MS) {
      plays.logged.delete(trackId);
    }
  }
}

function useRetry(): { retries: number; retryLater: () => void } {
  const [retries, setRetries] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const retryLater = useCallback(() => {
    if (timerRef.current != null) return;
    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      setRetries((count) => count + 1);
    }, RETRY_AFTER_MS);
  }, []);

  useEffect(
    () => () => {
      if (timerRef.current != null) clearTimeout(timerRef.current);
      timerRef.current = null;
    },
    [],
  );

  return { retries, retryLater };
}

export function useAutoLog(sessionId: number | null): void {
  const { trackId, source, playingTrackIds } = useNowPlayingTrack();
  const leadingTrackId = source === "ableton" ? trackId : null;
  const session = useCurrentSession().data;
  const lastLoggedTrackId = session?.plays.at(-1)?.track_id ?? null;
  const { mutateAsync: logPlay } = useAddPlay(sessionId);
  const { retries, retryLater } = useRetry();
  const playsRef = useRef<PlaysInSession | null>(null);

  useEffect(() => {
    if (sessionId == null) return;
    if (playsRef.current?.sessionId !== sessionId) playsRef.current = noPlaysYet(sessionId);
    const plays = playsRef.current;
    settleWhichPlaysAreLogged(plays, playingTrackIds, lastLoggedTrackId);

    const leaderNeedsLogging =
      leadingTrackId != null &&
      !plays.logged.has(leadingTrackId) &&
      !plays.beingLogged.has(leadingTrackId);
    if (leaderNeedsLogging) {
      plays.beingLogged.add(leadingTrackId);
      logPlay({ track_id: leadingTrackId })
        .then(() => plays.logged.add(leadingTrackId), retryLater)
        .finally(() => plays.beingLogged.delete(leadingTrackId));
    }
    return () => {
      const heardAt = Date.now();
      for (const playingTrackId of playingTrackIds) plays.lastHeardAt.set(playingTrackId, heardAt);
    };
  }, [playingTrackIds, leadingTrackId, sessionId, lastLoggedTrackId, logPlay, retries, retryLater]);
}
