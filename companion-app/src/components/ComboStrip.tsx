import { useMemo } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import { STEM_COLUMNS, roleLabel, type StemRole } from "../lib/roles";
import type { DeckCell } from "../types";

const ROLE_ACCENT: Record<StemRole, { dot: string; chip: string }> = {
  drums:  { dot: "bg-red-500",    chip: "text-red-300" },
  bass:   { dot: "bg-amber-500",  chip: "text-amber-300" },
  vocals: { dot: "bg-lime-400",   chip: "text-lime-300" },
  other:  { dot: "bg-sky-400",    chip: "text-sky-300" },
  mix:    { dot: "bg-neutral-200", chip: "text-neutral-200" },
};

/**
 * Horizontal 5-card row showing the *current active combo* — one card per
 * stem role with the source-track metadata of whatever's playing in that
 * role. In live-remixing there's no single "now playing" track; this strip
 * makes that honest by showing the source of each stem independently.
 *
 * Anchor mode: when all non-empty cells in a single scene point at the
 * same source track, the user has fired a whole row. We surface that
 * explicitly so they know the combo is the original song-as-recorded.
 */
export function ComboStrip() {
  const ableton = useAbletonState();
  const deckMap = useDeckMap();

  const columns = deckMap.data?.columns ?? null;
  const cells = deckMap.data?.cells ?? [];
  const playing = ableton.playing_clips ?? {};

  // (scene_index, kind) → DeckCell
  const cellAt = useMemo(() => {
    const m = new Map<string, DeckCell>();
    for (const c of cells) m.set(`${c.scene_index}|${c.kind}`, c);
    return m;
  }, [cells]);

  // Per-role: { sceneIdx, cell } for whatever's currently playing in that
  // role's column. Drives one card.
  const cards = useMemo(() => {
    if (!columns) return null;
    return STEM_COLUMNS.map((role) => {
      const trackIdx = columns[role];
      const sceneIdx = trackIdx != null ? playing[trackIdx] : undefined;
      const cell =
        sceneIdx != null ? cellAt.get(`${sceneIdx}|${role}`) : undefined;
      return { role, sceneIdx, cell };
    });
  }, [columns, playing, cellAt]);

  // Anchor detection — every non-empty card pointing at the same scene AND
  // that scene's stem cells all sourced from the same track.
  const anchor = useMemo(() => {
    if (!cards) return null;
    const occupied = cards.filter((c) => c.sceneIdx != null && c.cell != null);
    if (occupied.length === 0) return null;
    const sceneIdx = occupied[0].sceneIdx;
    if (!occupied.every((c) => c.sceneIdx === sceneIdx)) return null;
    const trackIds = occupied.map((c) => c.cell?.track_id);
    if (new Set(trackIds).size !== 1) return null;
    const sample = occupied[0].cell!;
    return {
      sceneIdx,
      title: sample.title,
      track_id: sample.track_id,
    };
  }, [cards]);

  if (!columns) {
    return (
      <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-3 text-xs text-neutral-600">
        Waiting for Ableton — load a track from the recs banner to begin a combo.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1" data-testid="combo-strip">
      <div className="flex items-baseline justify-between px-1">
        <div className="text-[10px] uppercase tracking-widest text-neutral-500">
          Current combo
        </div>
        {anchor ? (
          <div className="text-[10px] text-emerald-300/90 uppercase tracking-widest">
            ⚓ anchored to scene {anchor.sceneIdx! + 1} ·{" "}
            <span className="text-neutral-200 normal-case tracking-normal">
              {anchor.title ?? `Track #${anchor.track_id}`}
            </span>
          </div>
        ) : (
          <div className="text-[10px] text-neutral-600 uppercase tracking-widest">
            live remix
          </div>
        )}
      </div>
      <div className="grid grid-cols-5 gap-1.5">
        {cards?.map((c) => (
          <ComboCard
            key={c.role}
            role={c.role}
            cell={c.cell}
            isAnchorPart={anchor != null && c.sceneIdx === anchor.sceneIdx}
          />
        ))}
      </div>
    </div>
  );
}

function ComboCard({
  role,
  cell,
  isAnchorPart,
}: {
  role: StemRole;
  cell: DeckCell | undefined;
  isAnchorPart: boolean;
}) {
  const accent = ROLE_ACCENT[role];
  if (!cell) {
    return (
      <div className="rounded-md border border-neutral-900 bg-neutral-950/60 px-2 py-2 h-16">
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-neutral-700">
          <span className={`w-1.5 h-1.5 rounded-full ${accent.dot} opacity-40`} />
          {roleLabel(role)}
        </div>
        <div className="text-[11px] text-neutral-700 italic mt-1">silent</div>
      </div>
    );
  }
  return (
    <div
      className={`rounded-md border px-2 py-2 h-16 ${
        isAnchorPart
          ? "border-emerald-500/30 bg-emerald-500/5"
          : "border-neutral-800 bg-neutral-900/40"
      }`}
    >
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider">
        <span className={`w-1.5 h-1.5 rounded-full ${accent.dot}`} />
        <span className={accent.chip}>{roleLabel(role)}</span>
        {cell.key_camelot && (
          <span className="ml-auto font-mono text-neutral-400">
            {cell.key_camelot}
          </span>
        )}
      </div>
      <div className="text-xs text-neutral-100 truncate font-medium leading-tight mt-1">
        {cell.title ?? `Track #${cell.track_id}`}
      </div>
      <div className="text-[10px] text-neutral-500 truncate font-mono">
        {cell.bpm != null ? `${cell.bpm.toFixed(1)} BPM` : "—"}
        {cell.artist && (
          <span className="text-neutral-600 normal-case font-sans">
            {" · "}
            {cell.artist}
          </span>
        )}
      </div>
    </div>
  );
}
