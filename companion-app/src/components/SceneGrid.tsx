import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { pushTrackToLive } from "../api";
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
import {
  ROLE_STYLES,
  deckKindOf,
  roleLabel,
  type DeckSide,
  type StemRole,
} from "../lib/roles";
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
  const qc = useQueryClient();
  // Anchor-fill: load all 4 stems from a single-stem row's track into the
  // same row. Backend already supports it — pass scene_index=this row,
  // kinds=undefined (= full song). Lone stem gets overwritten with the
  // same audio, the 3 missing stems join it = instant anchor.
  const anchorFill = useMutation({
    mutationFn: (vars: { trackId: number; sceneIdx: number }) =>
      pushTrackToLive(vars.trackId, {
        includeStems: true,
        sceneIndex: vars.sceneIdx,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
    },
  });
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
        // Per-side anchor detection: a side counts as "anchor-ready" when
        // all 4 of its source stems are loaded AND point at the same track.
        // Each side can have its own anchor — that's the whole point of
        // deck pairs (A holding the current song while B holds the next).
        const sourceRoles: StemRole[] = ["drums", "bass", "vocals", "other"];
        const sideAnchor = (side: DeckSide): number | null => {
          const tids = sourceRoles.map(
            (r) => cellAt.get(`${sceneIdx}|${deckKindOf(r, side)}`)?.track_id,
          );
          if (tids.some((t) => t == null)) return null;
          return new Set(tids).size === 1 ? (tids[0] as number) : null;
        };
        const aAnchor = sideAnchor("a");
        const bAnchor = sideAnchor("b");
        // Used for the SONG-column shadow + the row-clear behavior.
        const isAnchorReady = aAnchor != null || bAnchor != null;
        // Lone-stem detection: exactly one stem-deck cell loaded across
        // BOTH sides of the row. We use this to offer the ◇ anchor-fill
        // gesture (one click loads the rest of that track's stems).
        const allStemCells: DeckCell[] = [];
        for (const r of sourceRoles) {
          for (const side of ["a", "b"] as const) {
            const c = cellAt.get(`${sceneIdx}|${deckKindOf(r, side)}`);
            if (c) allStemCells.push(c);
          }
        }
        const loneStemTrackId =
          allStemCells.length === 1 ? allStemCells[0].track_id : null;
        const loneStemSide: DeckSide | null =
          allStemCells.length === 1
            ? (allStemCells[0].kind.endsWith("_b") ? "b" : "a")
            : null;
        const anyPlaying = Object.values(columns).some(
          (trackIdx) => playing[trackIdx] === sceneIdx,
        );
        const mixCell = cellAt.get(`${sceneIdx}|mix`);
        const anyLoaded = allStemCells.length > 0 || mixCell != null;

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
              // MIX is single-deck: no A/B split. SONG cell may show a
              // shadow (anchor inferred from either side's 4 stems).
              if (role === "mix") {
                const showShadow = mixCell == null && isAnchorReady;
                // Prefer the most-recently-completed anchor for the
                // shadow display. Tie-break to A.
                const anchorTid = aAnchor ?? bAnchor;
                const shadowCell: DeckCell | undefined =
                  showShadow && anchorTid != null
                    ? allStemCells.find((c) => c.track_id === anchorTid)
                    : undefined;
                const onClearRow = showShadow
                  ? () => {
                      // Clear every cell in this row across both sides.
                      for (const r of sourceRoles) {
                        for (const side of ["a", "b"] as const) {
                          const dk = deckKindOf(r, side);
                          const c = cellAt.get(`${sceneIdx}|${dk}`);
                          const tIdx = columns[dk];
                          if (c != null && tIdx != null) {
                            deleteCell.mutate({ track: tIdx, slot: sceneIdx });
                          }
                        }
                      }
                      if (mixCell != null && columns["mix"] != null) {
                        deleteCell.mutate({
                          track: columns["mix"],
                          slot: sceneIdx,
                        });
                      }
                    }
                  : undefined;
                const mixTrackIdx = columns["mix"];
                const mixPlaying =
                  mixTrackIdx != null && playing[mixTrackIdx] === sceneIdx;
                return (
                  <Cell
                    key={role}
                    role={role as StemRole}
                    cell={shadowCell ?? mixCell}
                    loaded={mixCell != null}
                    shadow={showShadow}
                    playing={mixPlaying}
                    beatMs={beatMs}
                    onTap={
                      mixCell != null && mixTrackIdx != null
                        ? () =>
                            mixPlaying
                              ? stopCell.mutate({
                                  track: mixTrackIdx,
                                  slot: sceneIdx,
                                })
                              : fireCell.mutate({
                                  track: mixTrackIdx,
                                  slot: sceneIdx,
                                })
                        : undefined
                    }
                    onRemove={
                      showShadow
                        ? onClearRow
                        : mixCell != null && mixTrackIdx != null
                        ? () =>
                            deleteCell.mutate({
                              track: mixTrackIdx,
                              slot: sceneIdx,
                            })
                        : undefined
                    }
                  />
                );
              }
              // Stem column: render A/B half-cells stacked vertically.
              return (
                <SplitCell
                  key={role}
                  role={role as StemRole}
                  sceneIdx={sceneIdx}
                  beatMs={beatMs}
                  cellAt={cellAt}
                  columns={columns}
                  playing={playing}
                  loneStemTrackId={loneStemTrackId}
                  loneStemSide={loneStemSide}
                  anchorFillPending={anchorFill.isPending}
                  onFireCell={(track, slot) => fireCell.mutate({ track, slot })}
                  onStopCell={(track, slot) => stopCell.mutate({ track, slot })}
                  onDeleteCell={(track, slot) =>
                    deleteCell.mutate({ track, slot })
                  }
                  onAnchorFill={(trackId) =>
                    anchorFill.mutate({ trackId, sceneIdx })
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

/** A/B stacked half-cells for one stem role's column. Layout follows
 * Option C from docs/proposals/stem-deck-pair.md: same horizontal real
 * estate as a single cell, but split horizontally into a top half (A)
 * and a bottom half (B). Each half is independently fireable, stoppable,
 * and removable. Anchor-fill (◇) only appears on the single lone-stem
 * half-cell when the rest of the row is empty.
 */
function SplitCell({
  role,
  sceneIdx,
  beatMs,
  cellAt,
  columns,
  playing,
  loneStemTrackId,
  loneStemSide,
  anchorFillPending,
  onFireCell,
  onStopCell,
  onDeleteCell,
  onAnchorFill,
}: {
  role: StemRole;
  sceneIdx: number;
  beatMs: number;
  cellAt: Map<string, DeckCell>;
  columns: Record<string, number>;
  playing: Record<number, number>;
  loneStemTrackId: number | null;
  loneStemSide: DeckSide | null;
  anchorFillPending: boolean;
  onFireCell: (track: number, slot: number) => void;
  onStopCell: (track: number, slot: number) => void;
  onDeleteCell: (track: number, slot: number) => void;
  onAnchorFill: (trackId: number) => void;
}) {
  const sides: DeckSide[] = ["a", "b"];
  return (
    <div className="flex flex-col gap-0.5 h-14">
      {sides.map((side) => {
        const deckKind = deckKindOf(role, side);
        const trackIdx = columns[deckKind];
        const cell = cellAt.get(`${sceneIdx}|${deckKind}`);
        const isPlaying =
          trackIdx != null && playing[trackIdx] === sceneIdx;
        const isLoneStem =
          cell != null
          && loneStemTrackId === cell.track_id
          && loneStemSide === side;
        return (
          <HalfCell
            key={side}
            role={role}
            side={side}
            cell={cell}
            loaded={cell != null}
            playing={isPlaying}
            beatMs={beatMs}
            onTap={
              cell != null && trackIdx != null
                ? () =>
                    isPlaying
                      ? onStopCell(trackIdx, sceneIdx)
                      : onFireCell(trackIdx, sceneIdx)
                : undefined
            }
            onRemove={
              cell != null && trackIdx != null
                ? () => onDeleteCell(trackIdx, sceneIdx)
                : undefined
            }
            onAnchorFill={
              isLoneStem && cell != null && !anchorFillPending
                ? () => onAnchorFill(cell.track_id)
                : undefined
            }
          />
        );
      })}
    </div>
  );
}

/** Half-height cell variant for the A/B split layout. Same affordances
 * as Cell (tap to fire/stop, × to remove, ◇ for anchor-fill), just at
 * h-7 instead of h-14 and with a tiny A/B side badge in the corner.
 */
function HalfCell({
  role,
  side,
  cell,
  loaded,
  playing,
  beatMs,
  onTap,
  onRemove,
  onAnchorFill,
}: {
  role: StemRole;
  side: DeckSide;
  cell: DeckCell | undefined;
  loaded: boolean;
  playing: boolean;
  beatMs: number;
  onTap: (() => void) | undefined;
  onRemove?: () => void;
  onAnchorFill?: () => void;
}) {
  const styles = ROLE_STYLES[role];
  const badge = side.toUpperCase();
  if (!loaded) {
    return (
      <div
        className={`rounded border h-[1.625rem] ${styles.border} ${styles.bg} opacity-40 px-1.5 py-0.5 text-[9px] font-mono text-neutral-600 flex items-center`}
        aria-label={`${roleLabel(role)} ${badge} (empty)`}
      >
        <span className="opacity-60">{badge}</span>
      </div>
    );
  }
  return (
    <div className="relative group">
      <button
        type="button"
        onClick={onTap}
        disabled={!onTap}
        title={
          playing
            ? `Stop ${roleLabel(role).toLowerCase()} ${badge} (${cell?.title ?? "loaded"})`
            : cell
            ? `${roleLabel(role)} ${badge}: ${cell.title ?? `Track #${cell.track_id}`}${cell.artist ? ` — ${cell.artist}` : ""} — tap to fire`
            : `${roleLabel(role)} ${badge} (loaded)`
        }
        className={`w-full rounded border h-[1.625rem] px-1.5 text-left overflow-hidden transition-all duration-100 ease-out cursor-pointer focus:outline-none flex items-center gap-1.5 ${
          playing
            ? "border-emerald-300 bg-emerald-500/25 shadow-[0_0_12px_rgba(16,185,129,0.4)] hover:bg-emerald-500/35"
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
        <span
          className={`shrink-0 text-[9px] font-mono ${playing ? "text-emerald-100/80" : "text-neutral-500"}`}
        >
          {badge}
        </span>
        {cell && (
          <span
            className={`truncate text-[11px] font-medium leading-none ${
              playing ? "text-emerald-50" : "text-neutral-100"
            }`}
          >
            {cell.title ?? `Track #${cell.track_id}`}
          </span>
        )}
      </button>
      {onRemove && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          title={`Remove ${roleLabel(role).toLowerCase()} ${badge} from this scene`}
          aria-label={`Remove ${roleLabel(role)} ${badge} clip from scene`}
          className="absolute top-0 right-0 w-3.5 h-3.5 rounded-full flex items-center justify-center text-[9px] leading-none text-neutral-400 bg-neutral-950/80 border border-neutral-800 opacity-0 group-hover:opacity-100 hover:text-rose-300 hover:border-rose-500/40 transition-opacity focus:opacity-100 focus:outline-none"
        >
          ×
        </button>
      )}
      {onAnchorFill && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onAnchorFill();
          }}
          title={`Fill the rest of ${cell?.title ?? "this track"}'s stems into side ${badge}`}
          aria-label="Fill row with rest of stems (anchor)"
          data-testid="cell-anchor-fill"
          className="absolute top-0 left-0 w-3.5 h-3.5 rounded-full flex items-center justify-center text-[9px] leading-none text-neutral-400 bg-neutral-950/80 border border-neutral-800 opacity-0 group-hover:opacity-100 hover:text-sky-300 hover:border-sky-500/40 transition-opacity focus:opacity-100 focus:outline-none"
        >
          ◇
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
  onAnchorFill,
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
  /** Optional: load the rest of this track's stems into the same row,
   * turning a lone stem into an anchor. When set, an ◇ button appears
   * on hover (top-left, opposite the × in the top-right). */
  onAnchorFill?: () => void;
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
      {onAnchorFill && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onAnchorFill();
          }}
          title={`Fill the rest of the stems from ${cell?.title ?? "this track"} into this scene`}
          aria-label="Fill row with rest of stems (anchor)"
          data-testid="cell-anchor-fill"
          className="absolute top-0.5 left-0.5 w-4 h-4 rounded-full flex items-center justify-center text-[10px] leading-none text-neutral-400 bg-neutral-950/80 border border-neutral-800 opacity-0 group-hover:opacity-100 hover:text-sky-300 hover:border-sky-500/40 transition-opacity focus:opacity-100 focus:outline-none"
        >
          ◇
        </button>
      )}
    </div>
  );
}
