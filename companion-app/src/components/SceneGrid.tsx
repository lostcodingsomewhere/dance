import { useMemo, useState } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import {
  useDeleteCell,
  useFireCell,
  useFireScene,
  useSoloTrack,
  useStopCell,
  useStopScene,
} from "../hooks/useTransport";
import { STEM_COLUMNS, roleLabel, type StemRole } from "../lib/roles";
import type { DeckCell } from "../types";
import { RoleIcon } from "./RoleIcon";

const COLLAPSED_ROWS = 4;
const EXPANDED_ROWS = 8;

const ROLE_COLOR: Record<StemRole, { dot: string; text: string; border: string; bg: string }> = {
  drums:  { dot: "bg-red-500",    text: "text-red-300",    border: "border-red-500/30",    bg: "bg-red-500/10" },
  bass:   { dot: "bg-amber-500",  text: "text-amber-300",  border: "border-amber-500/30",  bg: "bg-amber-500/10" },
  vocals: { dot: "bg-lime-400",   text: "text-lime-300",   border: "border-lime-500/30",   bg: "bg-lime-500/10" },
  other:  { dot: "bg-sky-400",    text: "text-sky-300",    border: "border-sky-500/30",    bg: "bg-sky-500/10" },
  mix:    { dot: "bg-neutral-200", text: "text-neutral-200", border: "border-neutral-500/30", bg: "bg-neutral-500/10" },
};

/**
 * The 8×5 scene grid — the canonical visual representation of what the APC40
 * is touching. Columns are stem roles (drums/bass/vocals/melody/song); rows
 * are scenes. Cells in the same row can come from different source tracks
 * (the live-remixing model); when all 4 stem cells in a row point at the
 * same track, that row is the song-as-recorded (anchor mode).
 *
 * Interactions:
 * - Tap a cell → fire that one stem clip.
 * - Tap a row label → fire the whole scene (anchor mode).
 * - Hover a loaded cell → see truncated track title + metadata.
 */
export function SceneGrid() {
  const deckMap = useDeckMap();
  const ableton = useAbletonState();
  const fireScene = useFireScene();
  const fireCell = useFireCell();
  const stopScene = useStopScene();
  const stopCell = useStopCell();
  const deleteCell = useDeleteCell();
  const soloTrack = useSoloTrack();
  const [expanded, setExpanded] = useState(false);
  // Locally-tracked solo state per role. The bridge doesn't push solo
  // status back, so this is "what the FE asked for" — works for the
  // common case where solo toggles happen via this UI. Resets if the
  // SceneGrid unmounts (e.g., tab switch).
  const [soloedRoles, setSoloedRoles] = useState<Set<string>>(new Set());

  function toggleSolo(role: StemRole, trackIdx: number) {
    const wasSoloed = soloedRoles.has(role);
    const next = new Set(soloedRoles);
    if (wasSoloed) next.delete(role);
    else next.add(role);
    setSoloedRoles(next);
    soloTrack.mutate({ track: trackIdx, soloed: !wasSoloed });
  }

  const columns = deckMap.data?.columns ?? null;
  const cells = deckMap.data?.cells ?? [];
  const playing = ableton.playing_clips ?? {};
  const tempo = ableton.tempo ?? 120;

  // (scene_index, kind) → DeckCell, for O(1) lookup during render.
  const cellAt = useMemo(() => {
    const m = new Map<string, DeckCell>();
    for (const c of cells) m.set(`${c.scene_index}|${c.kind}`, c);
    return m;
  }, [cells]);

  // Default to 4 visible rows; expand to 8 if user wants more headroom.
  // If they've loaded cells in higher rows, auto-bump the visible count so
  // loaded scenes are never hidden behind the collapse.
  const highestLoadedIdx = useMemo(() => {
    if (cells.length === 0) return -1;
    return Math.max(...cells.map((c) => c.scene_index));
  }, [cells]);
  const visibleRows = expanded
    ? EXPANDED_ROWS
    : Math.max(COLLAPSED_ROWS, highestLoadedIdx + 1);
  const rows = Array.from({ length: visibleRows }, (_, i) => i);
  const hiddenRows = EXPANDED_ROWS - visibleRows;

  // Beat-pulse animation duration in ms. One pulse per beat.
  const beatMs = Math.max(200, Math.round(60_000 / Math.max(40, tempo)));

  if (!columns) {
    return (
      <div className="rounded-lg border border-dashed border-neutral-800 p-6 text-xs text-neutral-600">
        Waiting for Ableton deck columns. Open Live and load a track to
        populate the grid.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 select-none" data-testid="scene-grid">
      {/* Column header */}
      <div className="grid grid-cols-[2.5rem_repeat(5,minmax(0,1fr))] gap-1 mb-1">
        <div /> {/* spacer over the row labels */}
        {STEM_COLUMNS.map((role) => {
          const trackIdx = columns[role];
          const isSoloed = soloedRoles.has(role);
          return (
            <div
              key={role}
              className={`flex items-center gap-1.5 px-2 text-[10px] uppercase tracking-widest ${ROLE_COLOR[role].text}`}
            >
              <RoleIcon role={role} size={12} />
              <span className="flex-1">{roleLabel(role)}</span>
              {trackIdx != null && (
                <button
                  type="button"
                  onClick={() => toggleSolo(role, trackIdx)}
                  title={
                    isSoloed
                      ? `Unsolo ${roleLabel(role).toLowerCase()} — return to normal routing`
                      : `Solo ${roleLabel(role).toLowerCase()} → cue out (Live's solo button must be in Cue/PFL mode)`
                  }
                  className={`text-[10px] leading-none rounded px-1 transition-colors ${
                    isSoloed
                      ? "bg-amber-400/30 text-amber-100 border border-amber-300/60"
                      : "text-neutral-600 hover:text-amber-200 border border-transparent"
                  }`}
                  aria-label={isSoloed ? `unsolo ${role}` : `solo ${role}`}
                  aria-pressed={isSoloed}
                >
                  S
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Rows */}
      {rows.map((sceneIdx) => {
        // Anchor mode: all 4 stem cells in this row point at the same track.
        const rowCells = STEM_COLUMNS.filter((r) => r !== "mix").map((r) =>
          cellAt.get(`${sceneIdx}|${r}`),
        );
        const rowTrackIds = rowCells.map((c) => c?.track_id ?? null);
        const isAnchorReady =
          rowTrackIds.every((t) => t != null) &&
          new Set(rowTrackIds).size === 1;
        const anyPlaying = Object.values(columns).some(
          (trackIdx) => playing[trackIdx] === sceneIdx,
        );
        const anyLoaded = rowCells.some((c) => c != null);

        return (
          <div
            key={sceneIdx}
            className="grid grid-cols-[2.5rem_repeat(5,minmax(0,1fr))] gap-1 items-stretch"
          >
            <RowLabel
              sceneIdx={sceneIdx}
              loaded={anyLoaded}
              anchorReady={isAnchorReady}
              playing={anyPlaying}
              onTap={() =>
                anyPlaying
                  ? stopScene.mutate(sceneIdx)
                  : fireScene.mutate(sceneIdx)
              }
              pending={fireScene.isPending || stopScene.isPending}
            />
            {STEM_COLUMNS.map((role) => {
              const trackIdx = columns[role];
              const cell = cellAt.get(`${sceneIdx}|${role}`);
              const isPlaying =
                trackIdx != null && playing[trackIdx] === sceneIdx;
              return (
                <Cell
                  key={role}
                  role={role}
                  cell={cell}
                  loaded={cell != null}
                  playing={isPlaying}
                  beatMs={beatMs}
                  onTap={
                    cell != null && trackIdx != null
                      ? () =>
                          isPlaying
                            ? stopCell.mutate({ track: trackIdx, slot: sceneIdx })
                            : fireCell.mutate({ track: trackIdx, slot: sceneIdx })
                      : undefined
                  }
                  onRemove={
                    cell != null && trackIdx != null
                      ? () =>
                          deleteCell.mutate({ track: trackIdx, slot: sceneIdx })
                      : undefined
                  }
                />
              );
            })}
          </div>
        );
      })}

      {/* Expand / collapse — saves vertical space for the recs banners
          when you only have a few rows in play. Loaded rows always show
          (highestLoadedIdx+1 floor), so collapse never hides content. */}
      {(hiddenRows > 0 || expanded) && (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="mt-1 mx-auto text-[10px] uppercase tracking-widest text-neutral-500 hover:text-neutral-300 transition-colors"
          title={
            expanded
              ? "Show fewer rows"
              : `Show all ${EXPANDED_ROWS} rows (currently showing ${visibleRows})`
          }
        >
          {expanded
            ? "▴ show fewer"
            : `▾ show all ${EXPANDED_ROWS} rows`}
        </button>
      )}
    </div>
  );
}

function RowLabel({
  sceneIdx,
  loaded,
  anchorReady,
  playing,
  onTap,
  pending,
}: {
  sceneIdx: number;
  loaded: boolean;
  anchorReady: boolean;
  playing: boolean;
  onTap: () => void;
  pending: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onTap}
      disabled={pending || !loaded}
      title={
        playing
          ? `Stop scene ${sceneIdx + 1}`
          : anchorReady
          ? `Fire scene ${sceneIdx + 1} — play the original combo (anchor mode)`
          : loaded
          ? `Fire scene ${sceneIdx + 1} — plays whatever cells are loaded`
          : `Scene ${sceneIdx + 1} (empty)`
      }
      className={`flex items-center justify-center rounded-md text-xs font-mono font-semibold transition-colors ${
        playing
          ? "bg-emerald-500/40 text-emerald-50 border border-emerald-300 shadow-[0_0_12px_rgba(16,185,129,0.5)] hover:bg-emerald-500/60 cursor-pointer"
          : anchorReady
          ? "bg-neutral-900/70 text-emerald-300/70 border border-emerald-500/30 hover:border-emerald-500/60 hover:text-emerald-300 cursor-pointer"
          : loaded
          ? "bg-neutral-900/70 text-neutral-400 border border-neutral-800 hover:border-neutral-700 cursor-pointer"
          : "bg-neutral-950 text-neutral-700 border border-neutral-900"
      }`}
    >
      {playing ? "⏹" : sceneIdx + 1}
    </button>
  );
}

function Cell({
  role,
  cell,
  loaded,
  playing,
  beatMs,
  onTap,
  onRemove,
}: {
  role: StemRole;
  cell: DeckCell | undefined;
  loaded: boolean;
  playing: boolean;
  beatMs: number;
  onTap: (() => void) | undefined;
  /** Optional: remove the clip from this slot. When set, an X button
   * appears on hover. The X stops + deletes via the bridge — the slot
   * stays, ready to receive the next ``Load to Live``. */
  onRemove?: () => void;
}) {
  const color = ROLE_COLOR[role];

  if (!loaded) {
    return (
      <div
        className="rounded-md border border-neutral-900 bg-neutral-950 h-14"
        aria-label={`${roleLabel(role)} (empty)`}
      />
    );
  }

  // Wrap in a relative div so the X can absolute-position over the
  // <button>. Using a sibling instead of a nested button keeps HTML
  // valid (no button-in-button).
  return (
    <div className="relative group">
      <button
        type="button"
        onClick={onTap}
        disabled={!onTap}
        title={
          playing
            ? `Stop ${roleLabel(role).toLowerCase()} (${cell?.title ?? "loaded"})`
            : cell
            ? `${roleLabel(role)}: ${cell.title ?? `Track #${cell.track_id}`} — tap to fire`
            : `${roleLabel(role)} (loaded)`
        }
        className={`w-full rounded-md border h-14 px-2 py-1 text-left overflow-hidden transition-all duration-100 ease-out cursor-pointer focus:outline-none ${
          playing
            ? "border-emerald-300 bg-emerald-500/25 shadow-[0_0_18px_rgba(16,185,129,0.45)] hover:bg-emerald-500/35"
            : `${color.border} bg-neutral-900/40 hover:bg-neutral-900/80 hover:border-neutral-700`
        }`}
        style={
          playing
            ? {
                animation: `dance-beat-pulse ${beatMs}ms ease-in-out infinite`,
              }
            : undefined
        }
      >
        <div className={`text-[10px] uppercase tracking-wider font-semibold flex items-center gap-1 ${playing ? "text-emerald-100" : color.text}`}>
          {playing && <span className="text-emerald-200">⏹</span>}
          {playing ? "playing" : roleLabel(role)}
          {playing && <span className="ml-auto text-[9px] text-emerald-200/80" title="Clip loops by default">🔁</span>}
        </div>
        {cell && (
          <div className={`text-xs truncate font-medium leading-tight mt-0.5 ${playing ? "text-emerald-50" : "text-neutral-200"}`}>
            {cell.title ?? `Track #${cell.track_id}`}
          </div>
        )}
      </button>
      {onRemove && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          title={`Remove ${roleLabel(role).toLowerCase()} from this scene`}
          aria-label={`Remove ${roleLabel(role)} clip from scene`}
          className="absolute top-0.5 right-0.5 w-4 h-4 rounded-full flex items-center justify-center text-[10px] leading-none text-neutral-400 bg-neutral-950/80 border border-neutral-800 opacity-0 group-hover:opacity-100 hover:text-rose-300 hover:border-rose-500/40 transition-opacity focus:opacity-100 focus:outline-none"
        >
          ×
        </button>
      )}
    </div>
  );
}
