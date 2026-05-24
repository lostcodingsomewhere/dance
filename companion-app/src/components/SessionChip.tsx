/**
 * Compact session indicator — lives in MasterStrip.
 *
 * Replaces the old PlayedStrip footer's "set name · plays · end set" block
 * with the bare-minimum affordance set: tiny play-count chip with a hover
 * tooltip listing the last few plays + an "end" link. The full play
 * history lives in the DjSession record (queryable via the API); the
 * Set Rail covers the planning surface. A persistent footer earned its
 * keep before the rail existed — now it's noise.
 */

import { useCurrentSession, useEndSession } from "../hooks/useSession";

export function SessionChip() {
  const session = useCurrentSession();
  const endSession = useEndSession();
  const data = session.data;

  if (!data) {
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

  const plays = data.plays;
  const recentTitles = plays
    .slice(-5)
    .map((p) => `${p.position_in_set}. ${p.title ?? `#${p.track_id}`}`)
    .join("\n");
  const tooltip =
    `${data.name ?? "Session"} — ${plays.length} ${
      plays.length === 1 ? "play" : "plays"
    }\n` +
    (recentTitles ? `\nRecent:\n${recentTitles}` : "No plays yet.");

  return (
    <div
      className="flex items-center gap-1 px-2 py-1 rounded border border-neutral-800/60 bg-neutral-900/40 text-[10px]"
      title={tooltip}
    >
      <span className="text-emerald-400">●</span>
      <span className="text-neutral-400 tabular-nums">
        {plays.length} {plays.length === 1 ? "play" : "plays"}
      </span>
      <button
        type="button"
        onClick={() => endSession.mutate(data.id)}
        disabled={endSession.isPending}
        className="text-neutral-600 hover:text-rose-400 uppercase tracking-wider ml-1 disabled:opacity-50"
        title="End this session — next clip fire starts a new one"
      >
        end
      </button>
    </div>
  );
}
