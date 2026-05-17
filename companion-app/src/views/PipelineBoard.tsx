import { useQueries } from "@tanstack/react-query";
import * as api from "../api";
import {
  STAGE_GROUPS,
  TERMINAL_GROUPS,
  type PipelineRecentTrack,
  type StageGroup,
} from "../types";

const PER_COLUMN = 8;
const POLL_MS = 2000;

/** Renders the pipeline as a Kanban board: one column per stage group, each
 * showing the queue (waiting) and the active (in-flight) tracks. Done /
 * Error get terminal columns. */
export function PipelineBoard({
  onTrackClick,
}: {
  onTrackClick?: (track: PipelineRecentTrack) => void;
}) {
  // Fan-out: one query per relevant state. React Query dedupes and batches
  // refetch nicely. Each column reads from up to 2 of these queries.
  // Order in the array mirrors STAGE_GROUPS so the indices line up.
  const stateQueries = useQueries({
    queries: STAGE_GROUPS.flatMap((g) =>
      [g.waiting_state, g.active_state].filter((s): s is string => !!s).map((state) => ({
        queryKey: ["board-state", state],
        queryFn: () => api.getPipelineRecent(PER_COLUMN, state),
        refetchInterval: POLL_MS,
        refetchIntervalInBackground: true,
      })),
    ).concat(
      TERMINAL_GROUPS.flatMap((g) =>
        g.states.map((state) => ({
          queryKey: ["board-state", state],
          queryFn: () => api.getPipelineRecent(PER_COLUMN, state),
          refetchInterval: POLL_MS,
          refetchIntervalInBackground: true,
        })),
      ),
    ),
  });

  function tracksForState(state: string): PipelineRecentTrack[] {
    const idx = stateQueries.findIndex(
      (_, i) => allStates[i] === state,
    );
    return stateQueries[idx]?.data ?? [];
  }

  // Build a parallel array of state keys so the lookup above is O(1)-ish.
  const allStates: string[] = [
    ...STAGE_GROUPS.flatMap((g) =>
      [g.waiting_state, g.active_state].filter((s): s is string => !!s),
    ),
    ...TERMINAL_GROUPS.flatMap((g) => g.states),
  ];

  return (
    <div className="flex gap-3 overflow-x-auto pb-2">
      {STAGE_GROUPS.map((group) => (
        <StageColumn
          key={group.key}
          group={group}
          waitingTracks={
            group.waiting_state ? tracksForState(group.waiting_state) : []
          }
          activeTracks={
            group.active_state ? tracksForState(group.active_state) : []
          }
          onTrackClick={onTrackClick}
        />
      ))}
      {TERMINAL_GROUPS.map((group) => (
        <TerminalColumn
          key={group.key}
          label={group.label}
          color={group.color}
          tracks={group.states.flatMap((s) => tracksForState(s))}
          onTrackClick={onTrackClick}
        />
      ))}
    </div>
  );
}

function StageColumn({
  group,
  waitingTracks,
  activeTracks,
  onTrackClick,
}: {
  group: StageGroup;
  waitingTracks: PipelineRecentTrack[];
  activeTracks: PipelineRecentTrack[];
  onTrackClick?: (track: PipelineRecentTrack) => void;
}) {
  return (
    <div
      className={`flex flex-col w-64 shrink-0 rounded-lg ${group.color} p-2`}
    >
      <div className="flex items-baseline justify-between mb-2 px-1">
        <h3 className="text-sm font-semibold uppercase tracking-wider">
          {group.label}
        </h3>
        <div className="text-xs tabular-nums text-neutral-400">
          {activeTracks.length > 0 && (
            <span className="text-emerald-300 mr-2">
              ● {activeTracks.length} active
            </span>
          )}
          <span className="text-neutral-400">{waitingTracks.length} waiting</span>
        </div>
      </div>

      {/* Active tracks (highlighted, pulsing) */}
      {activeTracks.length > 0 && (
        <ul className="space-y-1 mb-2">
          {activeTracks.map((t) => (
            <TrackCardLite
              key={`a-${t.id}`}
              track={t}
              active
              onClick={onTrackClick}
            />
          ))}
        </ul>
      )}

      {/* Waiting queue */}
      {waitingTracks.length === 0 && activeTracks.length === 0 ? (
        <div className="text-xs text-neutral-500 italic px-1 py-2">
          (empty)
        </div>
      ) : (
        <ul className="space-y-1 flex-1 overflow-y-auto">
          {waitingTracks.map((t) => (
            <TrackCardLite
              key={`w-${t.id}`}
              track={t}
              onClick={onTrackClick}
            />
          ))}
          {waitingTracks.length === PER_COLUMN && (
            <li className="text-xs text-neutral-500 px-1">… more</li>
          )}
        </ul>
      )}
    </div>
  );
}

function TerminalColumn({
  label,
  color,
  tracks,
  onTrackClick,
}: {
  label: string;
  color: string;
  tracks: PipelineRecentTrack[];
  onTrackClick?: (track: PipelineRecentTrack) => void;
}) {
  return (
    <div className={`flex flex-col w-64 shrink-0 rounded-lg ${color} p-2`}>
      <div className="flex items-baseline justify-between mb-2 px-1">
        <h3 className="text-sm font-semibold uppercase tracking-wider">
          {label}
        </h3>
        <div className="text-xs tabular-nums text-neutral-400">
          {tracks.length} {tracks.length === PER_COLUMN ? "+ shown" : ""}
        </div>
      </div>
      {tracks.length === 0 ? (
        <div className="text-xs text-neutral-500 italic px-1 py-2">
          (empty)
        </div>
      ) : (
        <ul className="space-y-1 flex-1 overflow-y-auto">
          {tracks.map((t) => (
            <TrackCardLite
              key={`t-${t.id}`}
              track={t}
              onClick={onTrackClick}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function TrackCardLite({
  track,
  active = false,
  onClick,
}: {
  track: PipelineRecentTrack;
  active?: boolean;
  onClick?: (track: PipelineRecentTrack) => void;
}) {
  return (
    <li
      className={`px-2 py-1 rounded text-xs cursor-pointer transition-colors ${
        active
          ? "bg-neutral-900/80 ring-1 ring-emerald-400/50 animate-pulse"
          : "bg-neutral-900/40 hover:bg-neutral-900/70"
      }`}
      onClick={() => onClick?.(track)}
    >
      <div className="truncate text-neutral-100 font-medium">
        {track.title ?? "(untitled)"}
      </div>
      <div className="truncate text-neutral-500">{track.artist ?? "?"}</div>
      {track.error_message && (
        <div className="text-rose-400 text-[10px] truncate mt-0.5">
          {track.error_message}
        </div>
      )}
    </li>
  );
}
