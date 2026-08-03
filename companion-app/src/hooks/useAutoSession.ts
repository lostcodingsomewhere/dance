import { useEffect, useRef } from "react";
import { useAbletonState } from "./useAbletonState";
import {
  useCreateSession,
  useCurrentSession,
  useEndSession,
} from "./useSession";

/** Treat a session whose last play was longer ago than this as stale. */
const STALE_AFTER_LAST_PLAY_MS = 4 * 60 * 60 * 1000; // 4 hours
/** Treat an empty session this old as stale too (someone hit Start, walked away). */
const STALE_AFTER_START_NO_PLAYS_MS = 60 * 60 * 1000; // 1 hour

/**
 * Auto-start a session the first time Ableton plays in a tab. Also auto-ends
 * stale active sessions so coming back the next day starts a fresh set
 * instead of gluing tonight onto last night.
 *
 * Runs once per component mount (attemptedRef) so we don't loop on session
 * data updates.
 */
export function useAutoSession(): void {
  const ableton = useAbletonState();
  const session = useCurrentSession();
  const createSession = useCreateSession();
  const endSession = useEndSession();
  const attemptedRef = useRef(false);

  useEffect(() => {
    if (attemptedRef.current) return;
    if (session.isLoading) return;

    if (session.data) {
      const last = session.data.plays.at(-1)?.played_at;
      const started = session.data.started_at;
      const now = Date.now();
      const isStale = last
        ? now - new Date(last).getTime() > STALE_AFTER_LAST_PLAY_MS
        : now - new Date(started).getTime() > STALE_AFTER_START_NO_PLAYS_MS;
      if (isStale) {
        // Guard against re-entering while the end is in flight, then RELEASE
        // the guard so the normal "playing -> create" branch below can run.
        //
        // This used to latch `attemptedRef` permanently and create the
        // replacement only `if (ableton.is_playing)` — evaluated against the
        // value captured when the effect ran. Come back the next day, open
        // the app with the transport stopped (which is how you always open
        // it), and the stale session was ended, no replacement was made, and
        // the guard blocked every retry for the lifetime of the mount. The
        // whole night went unlogged and only a page reload fixed it.
        attemptedRef.current = true;
        endSession.mutate(session.data.id, {
          onSuccess: () => {
            attemptedRef.current = false;
          },
          onError: () => {
            attemptedRef.current = false;
          },
        });
        return;
      }
      // Recent + active — keep using it.
      attemptedRef.current = true;
      return;
    }

    if (ableton.is_playing) {
      attemptedRef.current = true;
      createSession.mutate(undefined);
    }
  }, [
    ableton.is_playing,
    session.data,
    session.isLoading,
    createSession,
    endSession,
  ]);
}
