import { useMemo } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import { useFireCell, useFireScene } from "../hooks/useTransport";
import { STEM_COLUMNS, roleLabel, type StemRole } from "../lib/roles";
import type { DeckCell } from "../types";
import { RoleIcon } from "./RoleIcon";

const GRID_ROWS = 8;

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

  const rows = Array.from({ length: GRID_ROWS }, (_, i) => i);

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
        {STEM_COLUMNS.map((role) => (
          <div
            key={role}
            className={`flex items-center gap-1.5 px-2 text-[10px] uppercase tracking-widest ${ROLE_COLOR[role].text}`}
          >
            <RoleIcon role={role} size={12} />
            {roleLabel(role)}
          </div>
        ))}
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
              onFire={() => fireScene.mutate(sceneIdx)}
              pending={fireScene.isPending}
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
                  onFire={
                    cell != null && trackIdx != null
                      ? () =>
                          fireCell.mutate({ track: trackIdx, slot: sceneIdx })
                      : undefined
                  }
                />
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

function RowLabel({
  sceneIdx,
  loaded,
  anchorReady,
  playing,
  onFire,
  pending,
}: {
  sceneIdx: number;
  loaded: boolean;
  anchorReady: boolean;
  playing: boolean;
  onFire: () => void;
  pending: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onFire}
      disabled={pending || !loaded}
      title={
        anchorReady
          ? `Fire scene ${sceneIdx + 1} — play the original combo (anchor mode)`
          : loaded
          ? `Fire scene ${sceneIdx + 1} — plays whatever cells are loaded in this row`
          : `Scene ${sceneIdx + 1} (empty)`
      }
      className={`flex items-center justify-center rounded-md text-xs font-mono font-semibold transition-colors ${
        playing
          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
          : anchorReady
          ? "bg-neutral-900/70 text-emerald-300/70 border border-emerald-500/30 hover:border-emerald-500/60 hover:text-emerald-300 cursor-pointer"
          : loaded
          ? "bg-neutral-900/70 text-neutral-400 border border-neutral-800 hover:border-neutral-700 cursor-pointer"
          : "bg-neutral-950 text-neutral-700 border border-neutral-900"
      }`}
    >
      {sceneIdx + 1}
    </button>
  );
}

function Cell({
  role,
  cell,
  loaded,
  playing,
  beatMs,
  onFire,
}: {
  role: StemRole;
  cell: DeckCell | undefined;
  loaded: boolean;
  playing: boolean;
  beatMs: number;
  onFire: (() => void) | undefined;
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

  return (
    <button
      type="button"
      onClick={onFire}
      disabled={!onFire}
      title={
        cell
          ? `${roleLabel(role)}: ${cell.title ?? `Track #${cell.track_id}`} — tap to fire`
          : `${roleLabel(role)} (loaded)`
      }
      className={`rounded-md border h-14 px-2 py-1 text-left overflow-hidden transition-all duration-100 ease-out cursor-pointer focus:outline-none ${
        playing
          ? `border-emerald-500/60 ${color.bg} shadow-[0_0_12px_rgba(16,185,129,0.25)]`
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
      <div className={`text-[10px] uppercase tracking-wider ${playing ? "text-emerald-300" : color.text}`}>
        {playing ? "▶ playing" : roleLabel(role)}
      </div>
      {cell && (
        <div className="text-xs text-neutral-200 truncate font-medium leading-tight mt-0.5">
          {cell.title ?? `Track #${cell.track_id}`}
        </div>
      )}
    </button>
  );
}
