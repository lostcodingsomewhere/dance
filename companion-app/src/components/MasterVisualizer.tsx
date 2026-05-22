import { useMemo } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import { useRegions } from "../hooks/useRegions";
import { useSeekClip } from "../hooks/useTransport";
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
  const seek = useSeekClip();

  const columns = deckMap.data?.columns ?? null;
  const cells = deckMap.data?.cells ?? [];
  const playing = ableton.playing_clips ?? {};
  const positions = ableton.playing_positions ?? {};
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
      {STEM_ONLY_COLUMNS.map((role) => {
        const cell = playingByRole?.[role];
        const trackIdx = columns?.[role];
        const livePosBeats =
          trackIdx != null ? positions[String(trackIdx)] : undefined;
        return (
          <StemRow
            key={role}
            role={role}
            cell={cell}
            trackIdx={trackIdx}
            tempo={tempo}
            beat={beat}
            livePosBeats={livePosBeats}
            onSeek={(beats) => {
              if (trackIdx == null || cell == null) return;
              seek.mutate({
                track: trackIdx,
                slot: cell.scene_index,
                positionBeats: beats,
              });
            }}
          />
        );
      })}
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
  trackIdx,
  tempo,
  beat,
  livePosBeats,
  onSeek,
}: {
  role: StemRole;
  cell: DeckCell | undefined;
  trackIdx: number | undefined;
  tempo: number | null;
  beat: number | null;
  livePosBeats: number | undefined;
  onSeek: (beats: number) => void;
}) {
  const waveform = useStemWaveform(cell?.stem_file_id);
  // Region overlays come from the *source track* (not the stem itself) —
  // sections + cues are detected on the mix, and the stems share that
  // structure. ``useRegions`` no-ops when track_id is null so silent rows
  // don't fire spurious requests.
  const regions = useRegions(cell?.track_id ?? null);
  const tint = ROLE_TINT[role];

  // Playhead position 0-1. Prefer Live's per-clip playing_position (beat-
  // accurate, respects loop wraps automatically) over our master-beat
  // estimate. Fall back to the estimate when subscription data hasn't
  // landed yet — keeps the playhead from flickering on cold start.
  const duration = waveform.data?.duration_seconds;
  let position: number | undefined = undefined;
  if (cell && duration && duration > 0 && tempo != null) {
    if (livePosBeats != null) {
      const elapsedSec = (livePosBeats / tempo) * 60;
      position = (elapsedSec % duration) / duration;
    } else if (beat != null) {
      const elapsedSec = (beat / tempo) * 60;
      position = (elapsedSec % duration) / duration;
    }
  }

  // Click-to-jump: convert the ratio (0-1) back into beats and POST to
  // /transport/seek/{track}/{slot}. Sets start_marker + re-fires.
  function handleSeek(ratio: number) {
    if (!duration || !tempo) return;
    const beats = ratio * duration * (tempo / 60);
    onSeek(beats);
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
          onSeek={trackIdx != null ? handleSeek : undefined}
          regions={regions.data ?? undefined}
          durationSeconds={waveform.data?.duration_seconds}
        />
      ) : (
        <div className="text-[10px] text-neutral-700 italic px-1">silent</div>
      )}
    </div>
  );
}
