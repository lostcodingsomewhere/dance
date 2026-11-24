import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { pushTrackToLive } from "../api";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import { useWarpCheck } from "../hooks/useWarpCheck";
import {
  useDeleteCell,
  useFireCell,
  useStopCell,
} from "../hooks/useTransport";
import { formatDuration } from "../lib/format";
import {
  cueText,
  pageCount,
  pageOfScene,
  pagesPlaying,
  scenesOnPage,
  songTitlesOnPage,
} from "../lib/pages";
import { sectionTone } from "../lib/sectionTones";
import { store, useAppStore } from "../store";
import {
  ROLE_STYLES,
  TWO_DECK_COLUMN_ORDER,
  deckColumnLabel,
  roleLabel,
  sideOf,
  sourceKindOf,
  type StemRole,
} from "../lib/roles";
import type { DeckCell } from "../types";

/**
 * The 8-column scene grid — the canonical mirror of what the APC40 is
 * touching. Columns are the 8 stem decks in fader order (drums A · bass
 * A · vocals A · other A · drums B · bass B · vocals B · other B); rows
 * are scenes. The mix/anchor reference no longer has a column here — it
 * lives as a per-deck chip in the TwoDeckStrip header. Cells in the same
 * row can come from different source tracks (the live-remixing model);
 * when all 4 stem cells on a side point at the same track, that side is
 * the song-as-recorded (anchor mode).
 *
 * Interactions:
 * - Tap a cell → fire that one stem clip (tap again → stop).
 * - Tap a row label → fire the whole scene (anchor mode).
 * - Hover a loaded cell → see truncated track title + metadata.
 * - Fire a 2nd bass / lead-vocal while one is already playing → a
 *   non-blocking "swap instead?" chip warns about mud (never blocks).
 */
export function SceneGrid() {
  const deckMap = useDeckMap();
  const ableton = useAbletonState();
  const fireCell = useFireCell();
  const stopCell = useStopCell();
  const deleteCell = useDeleteCell();
  const qc = useQueryClient();
  const scheduleWarpCheck = useWarpCheck();
  const anchorFill = useMutation({
    mutationFn: (vars: { trackId: number; sceneIdx: number; side: "a" | "b" }) =>
      pushTrackToLive(vars.trackId, {
        includeStems: true,
        side: vars.side,
        page: pageOfScene(vars.sceneIdx),
      }),
    onSuccess: (result, vars) => {
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
      const label = `Anchor → scene ${vars.sceneIdx + 1}`;
      store.setLoadWarnings(label, result.warnings);
      // An anchor-fill is a whole-song load, so it gets audited too.
      scheduleWarpCheck(result.scene_index, label);
      store.armDeck(result.side, result.scene_index);
    },
  });
  const boothPage = useAppStore((s) => s.boothPage);
  // Soft "mud" warning — a transient, non-blocking chip shown when the DJ
  // fires a 2nd bass (or 2nd lead-vocal) while one is already playing.
  // Stacked basslines / lead vocals fight each other; the product
  // philosophy is "the user decides, not autopilot", so we only nudge.
  const [mudWarning, setMudWarning] = useState<StemRole | null>(null);
  const mudTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Two-deck column order: 8 columns in APC40 fader order (drums A · bass
  // A · vocals A · other A · drums B · …). Drag-to-reorder is removed for
  // now (the deck semantics break if columns are reordered freely —
  // drums_a doesn't map cleanly to the bass spot). The mix/anchor
  // reference is not a grid column; it lives in the TwoDeckStrip header.
  const stemColumns = TWO_DECK_COLUMN_ORDER;

  const columns = deckMap.data?.columns ?? null;
  const cells = deckMap.data?.cells ?? [];
  const playing = ableton.playing_clips ?? {};
  const tempo = ableton.tempo ?? 120;

  // True when a stem of ``role`` (other than the cell we're about to fire)
  // is already the firing clip somewhere — i.e. firing this one stacks a
  // 2nd of the same role on top. Only "thick" roles get the nudge: two
  // basslines = mud, two lead vocals = clash. Drums/melody layer fine.
  function wouldStack(role: StemRole, deckKind: string, sceneIdx: number): boolean {
    if (role !== "bass" && role !== "vocals") return false;
    if (!columns) return false;
    for (const side of ["a", "b"] as const) {
      const dk = `${role}_${side}`;
      if (dk === deckKind) continue; // the cell being fired
      const tIdx = columns[dk];
      if (tIdx != null && playing[tIdx] != null) return true;
    }
    return false;
  }

  function flashMud(role: StemRole) {
    setMudWarning(role);
    if (mudTimer.current) clearTimeout(mudTimer.current);
    mudTimer.current = setTimeout(() => setMudWarning(null), 4000);
  }

  // (scene_index, kind) → DeckCell, for O(1) lookup during render.
  const cellAt = useMemo(() => {
    const m = new Map<string, DeckCell>();
    for (const c of cells) m.set(`${c.scene_index}|${c.kind}`, c);
    return m;
  }, [cells]);

  const pages = pageCount(cells);
  const shownPage = Math.min(boothPage, pages - 1);
  const rows = scenesOnPage(shownPage);
  const playingPages = columns ? pagesPlaying(columns, playing) : new Set<number>();

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
    <div
      className="relative flex flex-col gap-1 select-none"
      data-testid="scene-grid"
    >
      {/* Soft mud warning — floats over the grid, auto-dismisses, never
          blocks the action that triggered it. */}
      {mudWarning && (
        <div
          className="pointer-events-none absolute left-1/2 -top-1 -translate-x-1/2 -translate-y-full z-20 rounded-md border border-amber-400/50 bg-amber-500/15 px-2.5 py-1 text-[11px] font-medium text-amber-100 shadow-lg"
          role="status"
          data-testid="mud-warning"
        >
          2 {mudWarning === "bass" ? "basslines" : "lead vocals"} = mud — swap
          instead?
        </div>
      )}
      <nav aria-label="Pages" className="flex items-center gap-1">
        <span className="mr-1 text-[10px] uppercase tracking-widest text-neutral-500">
          Page
        </span>
        {Array.from({ length: pages }, (_, page) => {
          const isShown = page === shownPage;
          const isPlaying = playingPages.has(page);
          return (
            <button
              key={page}
              type="button"
              onClick={() => store.setBoothPage(page)}
              aria-pressed={isShown}
              aria-label={`Page ${page + 1}${isPlaying ? ", playing" : ""}`}
              title={songTitlesOnPage(cells, page) || `Page ${page + 1}`}
              className={`relative h-6 min-w-6 rounded border px-1.5 text-[11px] font-mono tabular-nums transition-colors ${
                isShown
                  ? "border-violet-400 bg-violet-500/25 text-violet-50"
                  : "border-neutral-800 text-neutral-400 hover:border-neutral-600 hover:text-neutral-100"
              }`}
            >
              {page + 1}
              {isPlaying && (
                <span className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-emerald-400" />
              )}
            </button>
          );
        })}
      </nav>
      {/* Rows — column headers + solo buttons live in BoothColumnHeaders
          one level up, so the same 8 column labels can sit above the
          TwoDeckStrip + SceneGrid + the rec banner row instead of being
          repeated inside each. */}
      {rows.map((sceneIdx) => {
        // Stem cells across both sides — for lone-stem detection, which
        // drives the per-cell anchor-fill (◇) affordance.
        const allStemCells: DeckCell[] = [];
        for (const dk of stemColumns) {
          if (dk.startsWith("mix")) continue;
          const c = cellAt.get(`${sceneIdx}|${dk}`);
          if (c) allStemCells.push(c);
        }
        const loneStemTrackId =
          allStemCells.length === 1 ? allStemCells[0].track_id : null;

        return (
          <div
            key={sceneIdx}
            className="grid grid-cols-8 gap-1 items-stretch"
          >
            {stemColumns.map((deckKind) => {
              // The 8 grid columns are all stem decks (drums/bass/vocals/
              // other × A/B) — the mix/anchor reference moved to the
              // TwoDeckStrip header, so there's no mix-column shadow here.
              const role = sourceKindOf(deckKind);
              const trackIdx = columns[deckKind];
              const cell = cellAt.get(`${sceneIdx}|${deckKind}`);
              const isPlaying =
                trackIdx != null && playing[trackIdx] === sceneIdx;
              return (
                <Cell
                  key={deckKind}
                  role={role}
                  deckKind={deckKind}
                  cell={cell}
                  loaded={cell != null}
                  playing={isPlaying}
                  beatMs={beatMs}
                  onTap={
                    cell != null && trackIdx != null
                      ? () => {
                          if (isPlaying) {
                            stopCell.mutate({ track: trackIdx, slot: sceneIdx });
                            return;
                          }
                          // Soft mud nudge — fire anyway, just warn when a
                          // 2nd bass / lead-vocal would stack. Never blocks.
                          if (wouldStack(role, deckKind, sceneIdx)) {
                            flashMud(role);
                          }
                          fireCell.mutate({ track: trackIdx, slot: sceneIdx });
                        }
                      : undefined
                  }
                  onRemove={
                    cell != null && trackIdx != null
                      ? () =>
                          deleteCell.mutate({ track: trackIdx, slot: sceneIdx })
                      : undefined
                  }
                  onAnchorFill={
                    cell != null
                    && loneStemTrackId === cell.track_id
                    && !anchorFill.isPending
                      ? () =>
                          anchorFill.mutate({
                            trackId: cell.track_id,
                            sceneIdx,
                            side: deckKind.endsWith("_b") ? "b" : "a",
                          })
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

function Cell({
  role,
  deckKind,
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
  /** Backend deck-channel id (drums_a, drums_b, ..., mix_a, mix_b). Used
   * to render the A/B side badge inside the cell. */
  deckKind: string;
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
   * not actually loaded in Live. Used in a SONG cell when its side's
   * 4 stems all point at the same track. No interaction — read-only. */
  shadow?: boolean;
}) {
  const styles = ROLE_STYLES[role];
  const sideBadge = sideOf(deckKind);
  const badgeText = sideBadge?.toUpperCase() ?? "";
  const cue = cell ? cueText(cell) : null;
  const cueInTooltip =
    cell && cell.cue != null
      ? ` — cue ${[cell.cue, cell.cue_label, cell.cue_silent ? "silent here" : null]
          .filter((part) => part != null)
          .join(", ")}`
      : "";

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
        className={`relative rounded-md border h-14 ${styles.border} ${styles.bg} opacity-50`}
        aria-label={`${deckColumnLabel(deckKind)} (empty)`}
      >
        {badgeText && (
          <span className="absolute bottom-0.5 right-1 text-[8px] font-mono text-neutral-600 opacity-70">
            {badgeText}
          </span>
        )}
      </div>
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
            ? `Stop ${deckColumnLabel(deckKind).toLowerCase()} (${cell?.title ?? "loaded"})`
            : cell
            ? `${deckColumnLabel(deckKind)}: ${cell.title ?? `Track #${cell.track_id}`}${cell.artist ? ` — ${cell.artist}` : ""}${cueInTooltip} — tap to fire`
            : `${deckColumnLabel(deckKind)} (loaded)`
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
            {(cue || cell.artist) && (
              <div className={`flex items-center gap-1 text-[10px] leading-tight ${playing ? "text-emerald-100/80" : "text-neutral-400"}`}>
                {cue && (
                  <span
                    className={`shrink-0 rounded px-1 text-[9px] font-semibold uppercase leading-[14px] tracking-wide ${sectionTone(cell.cue_label).head} ${cell.cue_silent ? "opacity-40" : ""}`}
                  >
                    {cue}
                  </span>
                )}
                {cell.artist && <span className="truncate">{cell.artist}</span>}
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
      {badgeText && (
        <span
          className={`pointer-events-none absolute bottom-0.5 right-1 text-[8px] font-mono leading-none ${
            playing ? "text-emerald-200/60" : "text-neutral-500"
          }`}
        >
          {badgeText}
        </span>
      )}
      {onRemove && <RemoveClipButton role={role} onRemove={onRemove} />}
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

const REMOVE_STAYS_ARMED_MS = 2500;

function RemoveClipButton({ role, onRemove }: { role: StemRole; onRemove: () => void }) {
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!armed) return;
    const disarm = setTimeout(() => setArmed(false), REMOVE_STAYS_ARMED_MS);
    return () => clearTimeout(disarm);
  }, [armed]);

  const clip = `${roleLabel(role)} clip`;
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        setArmed(!armed);
        if (armed) onRemove();
      }}
      title={
        armed
          ? "Press again to remove this clip from Live"
          : `Remove ${roleLabel(role).toLowerCase()} from this scene`
      }
      aria-label={armed ? `Press again to remove ${clip} from scene` : `Remove ${clip} from scene`}
      className={`absolute top-0.5 right-0.5 h-4 rounded-full flex items-center justify-center leading-none border focus:outline-none ${
        armed
          ? "px-1.5 text-[9px] font-semibold text-white bg-rose-600 border-rose-400"
          : "w-4 text-[10px] text-neutral-400 bg-neutral-950/80 border-neutral-800 opacity-0 group-hover:opacity-100 hover:text-rose-300 hover:border-rose-500/40 transition-opacity focus:opacity-100"
      }`}
    >
      {armed ? "remove?" : "×"}
    </button>
  );
}
