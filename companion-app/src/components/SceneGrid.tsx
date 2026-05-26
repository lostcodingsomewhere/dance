import { useMemo, useState } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import {
  useDeleteCell,
  useFireCell,
  useFireScene,
  useStopCell,
  useStopScene,
} from "../hooks/useTransport";
import { formatDuration } from "../lib/format";
import { ROLE_STYLES, roleLabel, type StemRole } from "../lib/roles";
import { useAppStore } from "../store";
import type { DeckCell } from "../types";

const COLLAPSED_ROWS = 3;
const EXPANDED_ROWS = 8;

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
  const [expanded, setExpanded] = useState(false);
  // User-customizable column order — reorder via drag-and-drop on the
  // BoothColumnHeaders chips. We render cells in this order; the
  // underlying Ableton track index stays the same (looked up via
  // ``columns[role]``).
  const stemColumns = useAppStore((s) => s.stemColumnOrder);

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
      {/* Rows — column headers + solo buttons live in BoothColumnHeaders
          one level up, so the same 5 column labels can sit above
          ComboStrip + SceneGrid + the rec banner row instead of being
          repeated inside each. */}
      {rows.map((sceneIdx) => {
        // Anchor mode: all 4 stem cells in this row point at the same track.
        const rowCells = stemColumns
          .filter((r) => r !== "mix")
          .map((r) => cellAt.get(`${sceneIdx}|${r}`));
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
            {stemColumns.map((role) => {
              const trackIdx = columns[role];
              const cell = cellAt.get(`${sceneIdx}|${role}`);
              const isPlaying =
                trackIdx != null && playing[trackIdx] === sceneIdx;
              // Shadow cell: when the SONG slot is empty but the 4 stems
              // all come from the same track, surface that track as a
              // ghost cell so the row visibly belongs to a song. UI-only
              // — Live's SONG slot stays empty (no audio implications);
              // this just makes the row legible at a glance.
              const showShadow =
                role === "mix" && cell == null && isAnchorReady;
              const shadowCell: DeckCell | undefined = showShadow
                ? rowCells.find((c) => c != null) ?? undefined
                : undefined;
              // Shadow X: clear every cell in this row (4 stems + a real
              // mix cell if one happens to be there). One click to undo
              // the whole anchor instead of removing four cells by hand.
              const onClearRow = showShadow
                ? () => {
                    for (const r of stemColumns) {
                      const c = cellAt.get(`${sceneIdx}|${r}`);
                      const tIdx = columns[r];
                      if (c != null && tIdx != null) {
                        deleteCell.mutate({ track: tIdx, slot: sceneIdx });
                      }
                    }
                  }
                : undefined;
              return (
                <Cell
                  key={role}
                  role={role as StemRole}
                  cell={shadowCell ?? cell}
                  loaded={cell != null}
                  shadow={showShadow}
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
                    showShadow
                      ? onClearRow
                      : cell != null && trackIdx != null
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
  shadow = false,
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
  /** UI-only ghost render: track is *inferred* from the row's stems but
   * not actually loaded in Live. Used in the SONG column when all 4
   * stems point at the same track. No interaction — read-only label. */
  shadow?: boolean;
}) {
  const styles = ROLE_STYLES[role];

  if (shadow && cell) {
    return (
      <div
        className="relative group"
        data-testid="scene-cell-shadow"
      >
        <div
          className={`rounded-md border-2 border-dashed h-14 px-2 py-1 overflow-hidden bg-neutral-800/60 border-neutral-500`}
          title={`Inferred from stems: ${cell.title ?? `Track #${cell.track_id}`}${cell.artist ? ` — ${cell.artist}` : ""}`}
          aria-label={`${roleLabel(role)} (inferred from stems)`}
        >
          <div className="text-xs truncate font-medium leading-tight text-neutral-100">
            {cell.title ?? `Track #${cell.track_id}`}
          </div>
          {cell.artist && (
            <div className="text-[10px] truncate leading-tight text-neutral-300">
              {cell.artist}
            </div>
          )}
          <div className="text-[9px] truncate font-mono leading-none mt-0.5 text-neutral-400 uppercase tracking-wider">
            ◇ anchor
          </div>
        </div>
        {onRemove && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
            title={`Clear all stems for ${cell.title ?? `Track #${cell.track_id}`} from this scene`}
            aria-label="Clear anchor row"
            className="absolute top-0.5 right-0.5 w-4 h-4 rounded-full flex items-center justify-center text-[10px] leading-none text-neutral-400 bg-neutral-950/80 border border-neutral-700 opacity-0 group-hover:opacity-100 hover:text-rose-300 hover:border-rose-500/40 transition-opacity focus:opacity-100 focus:outline-none"
          >
            ×
          </button>
        )}
      </div>
    );
  }

  if (!loaded) {
    return (
      <div
        className={`rounded-md border h-14 ${styles.border} ${styles.bg} opacity-50`}
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
            ? `${roleLabel(role)}: ${cell.title ?? `Track #${cell.track_id}`}${cell.artist ? ` — ${cell.artist}` : ""} — tap to fire`
            : `${roleLabel(role)} (loaded)`
        }
        className={`w-full rounded-md border h-14 px-2 py-1 text-left overflow-hidden transition-all duration-100 ease-out cursor-pointer focus:outline-none ${
          playing
            ? "border-emerald-300 bg-emerald-500/25 shadow-[0_0_18px_rgba(16,185,129,0.45)] hover:bg-emerald-500/35"
            : `${styles.border} ${styles.bg} hover:bg-neutral-900/60 hover:border-neutral-700`
        }`}
        style={
          playing
            ? {
                animation: `dance-beat-pulse ${beatMs}ms ease-in-out infinite`,
              }
            : undefined
        }
      >
        {cell && (
          <>
            <div className={`text-xs truncate font-medium leading-tight ${playing ? "text-emerald-50" : "text-neutral-100"}`}>
              {cell.title ?? `Track #${cell.track_id}`}
            </div>
            {cell.artist && (
              <div className={`text-[10px] truncate leading-tight ${playing ? "text-emerald-100/80" : "text-neutral-400"}`}>
                {cell.artist}
              </div>
            )}
            {(cell.bpm != null || cell.key_camelot || cell.floor_energy != null || cell.duration_seconds != null) && (
              <div className={`text-[9px] truncate font-mono leading-none mt-0.5 ${playing ? "text-emerald-200/70" : "text-neutral-500"}`}>
                {cell.bpm != null && <span>{cell.bpm.toFixed(1)}</span>}
                {cell.key_camelot && (
                  <span className="ml-1">· {cell.key_camelot}</span>
                )}
                {cell.floor_energy != null && (
                  <span className="ml-1">· E{cell.floor_energy}</span>
                )}
                {cell.duration_seconds != null && (
                  <span className="ml-1">· {formatDuration(cell.duration_seconds)}</span>
                )}
              </div>
            )}
          </>
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
