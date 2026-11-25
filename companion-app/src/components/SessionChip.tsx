/**
 * Compact session indicator — lives in MasterStrip.
 *
 * Replaces the old PlayedStrip footer's "set name · plays · end set" block
 * with the bare-minimum affordance set: tiny play-count chip that opens the
 * last few plays + an "end" link. The full play history lives in the
 * DjSession record (queryable via the API); the Set Rail covers the planning
 * surface. A persistent footer earned its keep before the rail existed — now
 * it's noise.
 */

import { useEffect, useReducer, useRef, useState } from "react";
import { serverTimeMs } from "../api";
import { useCurrentSession, useEndSession } from "../hooks/useSession";
import type { DjSession, SessionPlay } from "../types";

const RECENT_PLAYS_SHOWN = 8;
const MINUTE_MS = 60_000;

export function SessionChip() {
  const session = useCurrentSession().data;

  if (!session) {
    // Pre-session state — tiny dot, intentionally easy to miss.
    return (
      <span
        className="text-[10px] text-neutral-700 px-1"
        title="No active session — auto-starts on first clip fire"
      >
        ○
      </span>
    );
  }

  return <ActiveSessionChip key={session.id} session={session} />;
}

function ActiveSessionChip({ session }: { session: DjSession }) {
  const endSession = useEndSession();
  const [recentOpen, setRecentOpen] = useState(false);
  const chipRef = useRef<HTMLDivElement | null>(null);
  const plays = session.plays;

  useEffect(() => {
    if (!recentOpen) return;
    function closeOnOutsidePress(event: MouseEvent) {
      if (!chipRef.current?.contains(event.target as Node)) setRecentOpen(false);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setRecentOpen(false);
    }
    window.addEventListener("mousedown", closeOnOutsidePress);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("mousedown", closeOnOutsidePress);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [recentOpen]);

  return (
    <div
      ref={chipRef}
      className="relative flex items-center gap-1 px-2 py-1 rounded border border-neutral-800/60 bg-neutral-900/40 text-[10px]"
    >
      <span className="text-emerald-400">●</span>
      <button
        type="button"
        onClick={() => setRecentOpen((open) => !open)}
        aria-expanded={recentOpen}
        className="text-neutral-400 hover:text-neutral-100 tabular-nums"
        title="Show recent plays"
      >
        {plays.length} {plays.length === 1 ? "play" : "plays"}
      </button>
      <button
        type="button"
        onClick={() => endSession.mutate(session.id)}
        disabled={endSession.isPending}
        className="text-neutral-600 hover:text-rose-400 uppercase tracking-wider ml-1 disabled:opacity-50"
        title="End this session — next clip fire starts a new one"
      >
        end
      </button>
      {recentOpen && (
        <RecentPlays
          sessionName={session.name}
          plays={plays.slice(-RECENT_PLAYS_SHOWN).reverse()}
        />
      )}
    </div>
  );
}

function RecentPlays({
  sessionName,
  plays,
}: {
  sessionName: string | null;
  plays: SessionPlay[];
}) {
  const [, refreshTimes] = useReducer((ticks: number) => ticks + 1, 0);

  useEffect(() => {
    const everyHalfMinute = window.setInterval(refreshTimes, MINUTE_MS / 2);
    return () => window.clearInterval(everyHalfMinute);
  }, []);

  return (
    <div className="absolute right-0 top-full mt-1 z-50 w-80 rounded-md border border-neutral-700 bg-neutral-950 shadow-2xl">
      <div className="flex items-baseline gap-2 px-3 py-2 border-b border-neutral-900 text-[10px] uppercase tracking-wider text-neutral-500">
        <span className="shrink-0">Recent plays</span>
        {sessionName && (
          <span className="min-w-0 flex-1 truncate text-right normal-case tracking-normal">
            {sessionName}
          </span>
        )}
      </div>
      {plays.length === 0 ? (
        <p className="px-3 py-2 text-xs text-neutral-500">No plays yet.</p>
      ) : (
        <ul aria-label="Recent plays" className="py-1">
          {plays.map((play) => {
            const title = play.title ?? `Track #${play.track_id}`;
            return (
              <li
                key={play.position_in_set}
                className="flex items-baseline gap-3 px-3 py-1 text-xs"
              >
                <span className="min-w-0 flex-1 truncate text-neutral-100" title={title}>
                  {title}
                </span>
                <span className="shrink-0 tabular-nums text-neutral-500">
                  {playedAgo(play.played_at)}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function playedAgo(playedAt: string): string {
  const minutes = Math.floor((Date.now() - serverTimeMs(playedAt)) / MINUTE_MS);
  return minutes < 1 ? "just now" : `${minutes} min ago`;
}
