import { useCurrentSession, useEndSession } from "../hooks/useSession";

/**
 * Bottom-strip session history. Horizontal scroll of recent plays so the
 * user can read the set's trajectory at a glance during lulls. Lives at
 * the bottom of the Booth (low priority during transitions; ambient).
 *
 * Empty session → tiny start-prompt. Active session → end-set button +
 * up to ~30 most recent plays (newest right).
 */
export function PlayedStrip() {
  const session = useCurrentSession();
  const endSession = useEndSession();
  const data = session.data;

  if (!data) {
    return (
      <div className="px-4 py-2 text-[11px] text-neutral-600 italic border-t border-neutral-800">
        Press play in Ableton to start a set.
      </div>
    );
  }

  const plays = data.plays;
  return (
    <footer
      className="border-t border-neutral-800 bg-neutral-950/80 px-4 py-2 flex items-center gap-3 shrink-0"
      data-testid="played-strip"
    >
      <div className="flex flex-col shrink-0 min-w-[7rem]">
        <span className="text-[10px] uppercase tracking-widest text-neutral-500">
          {data.name ?? "Set"}
        </span>
        <span className="text-[10px] text-neutral-500">
          {plays.length} {plays.length === 1 ? "play" : "plays"} ·{" "}
          {formatTime(data.started_at)}
        </span>
      </div>
      <div className="flex-1 overflow-x-auto">
        <ol className="flex gap-1.5 items-center">
          {plays.length === 0 ? (
            <li className="text-[11px] text-neutral-700 italic">
              No plays yet — plays auto-log as Ableton fires clips.
            </li>
          ) : (
            plays.slice(-30).map((p) => (
              <li
                key={`${p.position_in_set}-${p.track_id}-${p.played_at}`}
                className="shrink-0 rounded border border-neutral-800/60 bg-neutral-900/40 px-2 py-1 text-[11px] flex items-center gap-2 min-w-0"
                title={p.title ?? `Track #${p.track_id}`}
              >
                <span className="text-neutral-600 font-mono">
                  {p.position_in_set}
                </span>
                <span className="text-neutral-200 truncate max-w-[14rem]">
                  {p.title ?? `Track #${p.track_id}`}
                </span>
                {p.energy_at_play != null && (
                  <span className="font-mono text-neutral-500 text-[10px]">
                    E{p.energy_at_play}
                  </span>
                )}
              </li>
            ))
          )}
        </ol>
      </div>
      <button
        type="button"
        onClick={() => endSession.mutate(data.id)}
        className="shrink-0 text-[10px] text-neutral-500 hover:text-rose-400 uppercase tracking-widest"
        title="End this set — next play in Ableton starts a new one"
      >
        end set
      </button>
    </footer>
  );
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "--";
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
