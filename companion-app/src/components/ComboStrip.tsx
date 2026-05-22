import { useMemo } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import { useRegions } from "../hooks/useRegions";
import { useSeekClip } from "../hooks/useTransport";
import { useStemWaveform, useTrackWaveform } from "../hooks/useWaveform";
import { STEM_COLUMNS, roleLabel, type StemRole } from "../lib/roles";
import type { DeckCell } from "../types";
import { RoleIcon } from "./RoleIcon";
import { Waveform } from "./Waveform";

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
 * role plus a full-featured interactive waveform (playhead, section bands,
 * cue ticks, click-to-jump). In live-remixing there's no single "now
 * playing" track; this strip makes that honest by showing the source of
 * each stem independently AND lets the DJ scrub each one.
 *
 * Anchor mode: when all non-empty cells in a single scene point at the
 * same source track, the user has fired a whole row. We surface that
 * explicitly so they know the combo is the original song-as-recorded.
 *
 * Merged the old MasterVisualizer's rich waveform features into the cards
 * (sections + cues + click-to-jump + Live playing_position playhead) so
 * the Booth has one canonical "what's playing" surface instead of two.
 */
export function ComboStrip() {
  const ableton = useAbletonState();
  const deckMap = useDeckMap();
  const seek = useSeekClip();

  const columns = deckMap.data?.columns ?? null;
  const cells = deckMap.data?.cells ?? [];
  const playing = ableton.playing_clips ?? {};
  const positions = ableton.playing_positions ?? {};
  const tempo = ableton.tempo;
  const beat = ableton.beat;

  // (scene_index, kind) → DeckCell
  const cellAt = useMemo(() => {
    const m = new Map<string, DeckCell>();
    for (const c of cells) m.set(`${c.scene_index}|${c.kind}`, c);
    return m;
  }, [cells]);

  // Per-role: { sceneIdx, cell, trackIdx, livePosBeats } for whatever's
  // currently playing in that role's column. Drives one card.
  const cards = useMemo(() => {
    if (!columns) return null;
    return STEM_COLUMNS.map((role) => {
      const trackIdx = columns[role];
      const sceneIdx = trackIdx != null ? playing[trackIdx] : undefined;
      const cell =
        sceneIdx != null ? cellAt.get(`${sceneIdx}|${role}`) : undefined;
      const livePosBeats =
        trackIdx != null ? positions[String(trackIdx)] : undefined;
      return { role, sceneIdx, cell, trackIdx, livePosBeats };
    });
  }, [columns, playing, positions, cellAt]);

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
          Current combo · click waveforms to scrub
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
      <div className="grid grid-cols-[2.5rem_repeat(5,minmax(0,1fr))] gap-1">
        {/* Empty leading cell — matches SceneGrid's row-label column so
            the 5 stem cards line up vertically with the 5 stem cols in
            the grid above. */}
        <div aria-hidden="true" />
        {cards?.map((c) => (
          <ComboCard
            key={c.role}
            role={c.role}
            cell={c.cell}
            trackIdx={c.trackIdx}
            livePosBeats={c.livePosBeats}
            isAnchorPart={anchor != null && c.sceneIdx === anchor.sceneIdx}
            tempo={tempo}
            beat={beat}
            onSeek={(beats) => {
              if (c.trackIdx == null || c.cell == null) return;
              seek.mutate({
                track: c.trackIdx,
                slot: c.cell.scene_index,
                positionBeats: beats,
              });
            }}
          />
        ))}
      </div>
    </div>
  );
}

function ComboCard({
  role,
  cell,
  trackIdx,
  livePosBeats,
  isAnchorPart,
  tempo,
  beat,
  onSeek,
}: {
  role: StemRole;
  cell: DeckCell | undefined;
  trackIdx: number | undefined;
  livePosBeats: number | undefined;
  isAnchorPart: boolean;
  tempo: number | null;
  beat: number | null;
  onSeek: (beats: number) => void;
}) {
  const accent = ROLE_ACCENT[role];
  // Stem waveform when this cell has a stem (drums/bass/vocals/other);
  // fall back to the full-track waveform for the mix/song column. Both
  // hooks no-op via ``enabled`` when their id is null/undefined, so it's
  // safe to call them unconditionally on every render.
  const stemWf = useStemWaveform(cell?.stem_file_id);
  const trackWf = useTrackWaveform(
    cell != null && cell.stem_file_id == null ? cell.track_id : null,
  );
  const waveform = cell?.stem_file_id != null ? stemWf : trackWf;
  // Regions come from the *source track* — sections + cues are detected
  // on the mix, and the stems share that structure.
  const regions = useRegions(cell?.track_id ?? null);

  // Playhead position 0-1. Prefer Live's per-clip playing_position (beat-
  // accurate, respects loop wraps automatically) over our master-beat
  // estimate. Fall back to the estimate when subscription data hasn't
  // landed yet — keeps the playhead from flickering on cold start.
  //
  // IMPORTANT: convert via the *clip's* nominal BPM (cell.bpm = the
  // source track's analyzed BPM), NOT Live's project tempo. Our .als
  // writer sets ``end_beats = duration * analyzed_bpm / 60`` and warps
  // the clip — so ``playing_position`` is in clip-beats indexed against
  // analyzed_bpm. Using Live's tempo would scale the playhead by
  // ``live_tempo / analyzed_bpm`` when the user tempo'd up or down.
  const duration = waveform.data?.duration_seconds;
  const clipBpm = cell?.bpm ?? tempo;
  let position: number | undefined = undefined;
  if (cell && duration && duration > 0 && clipBpm != null && clipBpm > 0) {
    if (livePosBeats != null) {
      const elapsedSec = (livePosBeats / clipBpm) * 60;
      position = (elapsedSec % duration) / duration;
    } else if (beat != null && tempo != null) {
      // Master-beat fallback uses project tempo since beat is in
      // project-time, not clip-time.
      const elapsedSec = (beat / tempo) * 60;
      position = (elapsedSec % duration) / duration;
    }
  }

  // Click-to-jump: SNAPS the click to the start of the section it
  // landed in, then converts that section-start to clip beats and
  // POSTs /transport/seek. Snap-to-section is more useful for live
  // performance than pixel-precise seeking — DJs think in "drop,
  // breakdown, drop" sections, not "180.4 seconds in." Falls back to
  // ratio=0 (clip start) when the click is before any section.
  function handleSeek(ratio: number) {
    if (!duration || !clipBpm || clipBpm <= 0) return;
    const sectionStarts = (regions.data ?? [])
      .filter((r) => r.region_type === "section")
      .map((r) => r.position_ms / 1000 / duration)
      .filter((s) => s <= ratio + 0.001) // small epsilon for click on icon
      .sort((a, b) => b - a);
    const snapped = sectionStarts.length > 0 ? sectionStarts[0] : 0;
    const beats = snapped * duration * (clipBpm / 60);
    onSeek(beats);
  }

  if (!cell) {
    return (
      <div className="rounded-md border border-neutral-900 bg-neutral-950/60 px-2 py-2 h-24 flex flex-col">
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-neutral-700">
          <RoleIcon role={role} size={12} className="opacity-40" />
          {roleLabel(role)}
        </div>
        <div className="text-[11px] text-neutral-700 italic mt-1">silent</div>
      </div>
    );
  }
  return (
    <div
      className={`rounded-md border px-2 py-2 h-24 flex flex-col ${
        isAnchorPart
          ? "border-emerald-500/30 bg-emerald-500/5"
          : "border-neutral-800 bg-neutral-900/40"
      }`}
    >
      <div className={`flex items-center gap-1.5 text-[10px] uppercase tracking-wider ${accent.chip}`}>
        <RoleIcon role={role} size={12} />
        <span>{roleLabel(role)}</span>
        {cell.key_camelot && (
          <span className="ml-auto font-mono text-neutral-400">
            {cell.key_camelot}
          </span>
        )}
      </div>
      <div className="text-xs text-neutral-100 truncate font-medium leading-tight mt-0.5">
        {cell.title ?? `Track #${cell.track_id}`}
      </div>
      <Waveform
        peaks={waveform.data?.peaks ?? []}
        position={position}
        height={32}
        className={`${accent.chip} opacity-90 mt-auto`}
        playheadColor="rgba(255,255,255,0.9)"
        regions={regions.data ?? undefined}
        durationSeconds={duration}
        onSeek={trackIdx != null ? handleSeek : undefined}
      />
    </div>
  );
}
