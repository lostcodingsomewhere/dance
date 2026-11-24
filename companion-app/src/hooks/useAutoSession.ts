import { useEffect, useRef } from "react";
import { serverTimeMs } from "../api";
import type { DjSession } from "../types";
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

const TRANSPORT = "transport";

function isStale(session: DjSession): boolean {
  const lastPlayedAt = session.plays.at(-1)?.played_at;
  return lastPlayedAt
    ? Date.now() - serverTimeMs(lastPlayedAt) > STALE_AFTER_LAST_PLAY_MS
    : Date.now() - serverTimeMs(session.started_at) > STALE_AFTER_START_NO_PLAYS_MS;
}

function whatLivePlays(
  isPlaying: boolean | null,
  playingClips: Record<string, number>,
): Set<string> {
  if (!isPlaying) return new Set();
  const clips = Object.entries(playingClips).map(([track, scene]) => `${track}:${scene}`);
  return new Set([TRANSPORT, ...clips]);
}

export function useAutoSession(): void {
  const { is_playing: isPlaying, playing_clips: playingClips } = useAbletonState();
  const session = useCurrentSession();
  const current = session.data;
  const sessionIsSettled = !session.isFetching;
  const { mutate: createSession } = useCreateSession();
  const { mutate: endSession } = useEndSession();
  const lastSeenPlayingRef = useRef(new Set<string>());
  const startAwaitsSessionRef = useRef(false);
  const staleCheckedRef = useRef<number | null>(null);
  const busyRef = useRef(false);

  useEffect(() => {
    const playing = whatLivePlays(isPlaying, playingClips ?? {});
    const somethingStarted = [...playing].some((part) => !lastSeenPlayingRef.current.has(part));
    lastSeenPlayingRef.current = playing;
    if (somethingStarted) startAwaitsSessionRef.current = true;
    if (busyRef.current || !sessionIsSettled) return;

    const release = () => {
      busyRef.current = false;
    };
    if (current) {
      const isFirstLook = staleCheckedRef.current !== current.id;
      staleCheckedRef.current = current.id;
      if (isFirstLook && isStale(current)) {
        busyRef.current = true;
        endSession(current.id, { onSettled: release });
        return;
      }
      startAwaitsSessionRef.current = false;
      return;
    }
    if (!startAwaitsSessionRef.current) return;
    startAwaitsSessionRef.current = false;
    busyRef.current = true;
    createSession(undefined, { onSettled: release });
  }, [isPlaying, playingClips, current, sessionIsSettled, createSession, endSession]);
}
