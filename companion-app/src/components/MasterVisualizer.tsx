import { useMemo } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import { useStemWaveform } from "../hooks/useWaveform";
import { STEM_ONLY_COLUMNS, roleLabel, type StemRole } from "../lib/roles";
import type { DeckCell } from "../types";
import { RoleIcon } from "./RoleIcon";
import { Waveform } from "./Waveform";

/**
 * Stacked stem-waveform visualizer — one horizontal row per stem role
 * (drums/bass/vocals/melody). Each row shows:
 *
 *   [icon · role · ⏵ if playing]  [stem waveform with playhead · 🔁]
 *
 * The waveform is the *actual* stem audio (not the full track), so when
 * combos mix stems from different source tracks you see each stem's own
 * loop. Playhead is derived from Live's master beat clock + the stem's
 * own duration (wraps for continuously looping clips — the typical
 * live-remixing mode).
 *
 * Song column is excluded — songs surface in ComboStrip's "anchor" hint
 * and the SceneGrid; stacking 5 waveforms would crowd the strip.
 */
export function MasterVisualizer() {
  const ableton = useAbletonState();
  const deckMap = useDeckMap();

  const columns = deckMap.data?.columns ?? null;
  const cells = deckMap.data?.cells ?? [];
  const playing = ableton.playing_clips ?? {};
  const tempo = ableton.tempo;
  const beat = ableton.beat;

  // (scene_index, kind) → DeckCell for fast lookup
  const cellAt = useMemo(() => {
    const m = new Map<string, DeckCell>();
    for (const c of cells) m.set(`${c.scene_index}|${c.kind}`, c);
    return m;
  }, [cells]);

  // Per-role: the cell currently playing in that role's column (or null).
  const playingByRole = useMemo(() => {
    if (!columns) return null;
    const m: Partial<Record<StemRole, DeckCell>> = {};
    for (const role of STEM_ONLY_COLUMNS) {
      const trackIdx = columns[role];
      if (trackIdx == null) continue;
      const sceneIdx = playing[trackIdx];
      if (sceneIdx == null) continue;
      const cell = cellAt.get(`${sceneIdx}|${role}`);
      if (cell) m[role] = cell;
    }
    return m;
  }, [columns, playing, cellAt]);

  if (!columns) return null; // nothing to visualize yet

  return (
    <div
      className="flex flex-col gap-1 rounded-md border border-neutral-800/60 bg-neutral-950/40 p-2"
      data-testid="master-visualizer"
    >
      <div className="text-[10px] uppercase tracking-widest text-neutral-500 px-1">
        Master · stacked stems
      </div>
      {STEM_ONLY_COLUMNS.map((role) => (
        <StemRow
          key={role}
          role={role}
          cell={playingByRole?.[role]}
          tempo={tempo}
          beat={beat}
        />
      ))}
    </div>
  );
}

const ROLE_TINT: Record<StemRole, string> = {
  drums: "text-red-300",
  bass: "text-amber-300",
  vocals: "text-lime-300",
  other: "text-sky-300",
  mix: "text-neutral-200",
};

function StemRow({
  role,
  cell,
  tempo,
  beat,
}: {
  role: StemRole;
  cell: DeckCell | undefined;
  tempo: number | null;
  beat: number | null;
}) {
  const waveform = useStemWaveform(cell?.stem_file_id);
  const tint = ROLE_TINT[role];

  // Wrapping playhead: clips loop by default (LoopOn=true in our writer),
  // so position = (elapsed seconds) mod (clip duration).
  let position: number | undefined = undefined;
  if (
    cell &&
    tempo != null &&
    beat != null &&
    waveform.data?.duration_seconds &&
    waveform.data.duration_seconds > 0
  ) {
    const elapsedSec = (beat / tempo) * 60;
    position =
      (elapsedSec % waveform.data.duration_seconds) /
      waveform.data.duration_seconds;
  }

  return (
    <div className="grid grid-cols-[6rem_1fr] gap-2 items-center min-h-[18px]">
      <div className={`flex items-center gap-1.5 text-[10px] uppercase tracking-wider px-1 ${cell ? tint : "text-neutral-700"}`}>
        <RoleIcon role={role} size={11} />
        <span className="font-semibold">{roleLabel(role)}</span>
        {cell && <span className="text-[9px] text-neutral-500">🔁</span>}
      </div>
      {cell ? (
        <Waveform
          peaks={waveform.data?.peaks ?? []}
          position={position}
          height={18}
          className={`${tint} opacity-90`}
          playheadColor="rgba(255,255,255,0.85)"
        />
      ) : (
        <div className="text-[10px] text-neutral-700 italic px-1">silent</div>
      )}
    </div>
  );
}
