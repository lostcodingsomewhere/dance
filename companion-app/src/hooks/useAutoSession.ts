import { useEffect, useRef, useState } from "react";
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

export function useAutoSession(): number | null {
  const { is_playing: isPlaying, playing_clips: playingClips } = useAbletonState();
  const session = useCurrentSession();
  const current = session.data;
  const sessionIsSettled = !session.isFetching;
  const { mutate: createSession } = useCreateSession();
  const { mutate: endSession } = useEndSession();
  const lastSeenPlayingRef = useRef(new Set<string>());
  const startAwaitsSessionRef = useRef(false);
  const firstLookedAtRef = useRef<number | null>(null);
  const busyRef = useRef(false);
  const [askedToEnd, setAskedToEnd] = useState<number | null>(null);
  const [ending, setEnding] = useState(false);

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
      const isFirstLook = firstLookedAtRef.current !== current.id;
      firstLookedAtRef.current = current.id;
      const timeToLook = isFirstLook || startAwaitsSessionRef.current;
      if (timeToLook && askedToEnd !== current.id && isStale(current)) {
        busyRef.current = true;
        setAskedToEnd(current.id);
        setEnding(true);
        endSession(current.id, {
          onSettled: () => {
            release();
            setEnding(false);
          },
        });
        return;
      }
      startAwaitsSessionRef.current = false;
      return;
    }
    if (!startAwaitsSessionRef.current) return;
    startAwaitsSessionRef.current = false;
    busyRef.current = true;
    createSession(undefined, { onSettled: release });
  }, [isPlaying, playingClips, current, sessionIsSettled, askedToEnd, createSession, endSession]);

  if (!current || ending) return null;
  const aboutToBeEnded = askedToEnd !== current.id && isStale(current);
  return aboutToBeEnded ? null : current.id;
}
