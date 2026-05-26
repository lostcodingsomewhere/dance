import { useMemo } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import { useRegions } from "../hooks/useRegions";
import { useSeekClip } from "../hooks/useTransport";
import { useStemWaveform, useTrackWaveform } from "../hooks/useWaveform";
import { formatDuration, formatRemaining } from "../lib/format";
import { ratioToBeats, snapRatioToSection } from "../lib/seek";
import { ROLE_STYLES, roleLabel, type StemRole } from "../lib/roles";
import { useAppStore } from "../store";
import type { DeckCell } from "../types";
import { Waveform } from "./Waveform";

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
  // Render combo cards in the user's chosen column order. Underlying
  // Ableton track indices come from ``columns[role]`` which is identity-
  // mapped, so reordering is purely cosmetic.
  const stemColumns = useAppStore((s) => s.stemColumnOrder);

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
  //
  // Deck-pair: each role now has two Live tracks (A and B). Per role we
  // pick the side that's actively playing — prefer A on ties, fall back
  // to A's track when neither is playing so the card still has somewhere
  // to anchor its seek callbacks.
  const cards = useMemo(() => {
    if (!columns) return null;
    return stemColumns.map((role) => {
      if (role === "mix") {
        const trackIdx = columns["mix"];
        const sceneIdx = trackIdx != null ? playing[trackIdx] : undefined;
        const cell =
          sceneIdx != null ? cellAt.get(`${sceneIdx}|mix`) : undefined;
        const livePosBeats =
          trackIdx != null ? positions[String(trackIdx)] : undefined;
        return { role: role as StemRole, sceneIdx, cell, trackIdx, livePosBeats };
      }
      const aIdx = columns[`${role}_a`];
      const bIdx = columns[`${role}_b`];
      const aPlayingScene = aIdx != null ? playing[aIdx] : undefined;
      const bPlayingScene = bIdx != null ? playing[bIdx] : undefined;
      const chosenSide: "a" | "b" =
        aPlayingScene != null ? "a" : bPlayingScene != null ? "b" : "a";
      const trackIdx = chosenSide === "a" ? aIdx : bIdx;
      const sceneIdx =
        chosenSide === "a" ? aPlayingScene : bPlayingScene;
      const cell =
        sceneIdx != null && trackIdx != null
          ? cellAt.get(`${sceneIdx}|${role}_${chosenSide}`)
          : undefined;
      const livePosBeats =
        trackIdx != null ? positions[String(trackIdx)] : undefined;
      return { role: role as StemRole, sceneIdx, cell, trackIdx, livePosBeats };
    });
  }, [columns, playing, positions, cellAt, stemColumns]);

  if (!columns) {
    return (
      <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-3 text-xs text-neutral-600">
        Waiting for Ableton — load a track from the recs banner to begin a combo.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1" data-testid="combo-strip">
      <div className="grid grid-cols-[2.5rem_repeat(5,minmax(0,1fr))] gap-1">
        {/* Empty leading cell — matches SceneGrid's row-label column so
            the 5 stem cards line up vertically with the 5 stem cols in
            the grid above. Header text + anchor hint stripped per the
            unified-column-color redesign: the colored column headers
            up top now carry the section's identity. */}
        <div aria-hidden="true" />
        {cards?.map((c) => (
          <ComboCard
            key={c.role}
            role={c.role}
            cell={c.cell}
            trackIdx={c.trackIdx}
            livePosBeats={c.livePosBeats}
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
  tempo,
  beat,
  onSeek,
}: {
  role: StemRole;
  cell: DeckCell | undefined;
  trackIdx: number | undefined;
  livePosBeats: number | undefined;
  tempo: number | null;
  beat: number | null;
  onSeek: (beats: number) => void;
}) {
  const styles = ROLE_STYLES[role];
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
  // Prefer the cell's reported duration (comes from Track.duration_seconds
  // — set during the analyze stage, doesn't require a waveform fetch)
  // and fall back to the waveform's measured duration. Waveform load can
  // lag the deck cell during a fresh load.
  const duration = cell?.duration_seconds ?? waveform.data?.duration_seconds;
  const clipBpm = cell?.bpm ?? tempo;
  let position: number | undefined = undefined;
  let elapsedSec: number | undefined = undefined;
  if (cell && duration && duration > 0 && clipBpm != null && clipBpm > 0) {
    if (livePosBeats != null) {
      elapsedSec = (livePosBeats / clipBpm) * 60;
      position = (elapsedSec % duration) / duration;
    } else if (beat != null && tempo != null) {
      // Master-beat fallback uses project tempo since beat is in
      // project-time, not clip-time.
      elapsedSec = (beat / tempo) * 60;
      position = (elapsedSec % duration) / duration;
    }
  }
  // Remaining = duration - (elapsed mod duration). Wraps with the clip
  // so a 4-minute loop continually shows -4:00 down to 0:00.
  const remaining =
    duration != null && elapsedSec != null
      ? duration - (elapsedSec % duration)
      : null;

  // Click-to-jump: SNAPS the click to the start of the section it
  // landed in, then converts that section-start to clip beats and
  // POSTs /transport/seek. Snap-to-section is more useful for live
  // performance than pixel-precise seeking — DJs think in "drop,
  // breakdown, drop" sections, not "180.4 seconds in." Falls back to
  // ratio=0 (clip start) when the click is before any section.
  function handleSeek(ratio: number) {
    if (!duration || !clipBpm || clipBpm <= 0) return;
    const snapped = snapRatioToSection(ratio, regions.data, duration);
    onSeek(ratioToBeats(snapped, duration, clipBpm));
  }

  if (!cell) {
    return (
      <div
        className={`rounded-md border px-2 py-2 h-24 flex items-center justify-center ${styles.border} ${styles.bg}`}
        aria-label={`${roleLabel(role)} (silent)`}
      >
        <div className="text-[11px] text-neutral-700 italic">silent</div>
      </div>
    );
  }
  return (
    <div
      className={`rounded-md border px-2 py-2 h-24 flex flex-col ${styles.border} ${styles.bg}`}
    >
      <div className="flex items-baseline gap-1.5">
        <div className="text-xs text-neutral-100 truncate font-medium leading-tight flex-1">
          {cell.title ?? `Track #${cell.track_id}`}
        </div>
        {cell.key_camelot && (
          <span className={`text-[10px] font-mono ${styles.text}`}>
            {cell.key_camelot}
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-1.5 leading-tight">
        {cell.artist && (
          <div className="text-[10px] text-neutral-400 truncate flex-1">
            {cell.artist}
          </div>
        )}
        {/* Time pair — elapsed / remaining when actively playing, else
            just the total duration. Tabular-nums so the digit slots don't
            jitter when the playhead advances. */}
        {duration != null && (
          <div
            className="text-[10px] font-mono tabular-nums text-neutral-500 shrink-0"
            title={
              elapsedSec != null
                ? "elapsed / remaining"
                : "track duration"
            }
          >
            {elapsedSec != null
              ? `${formatDuration(elapsedSec % duration)} ${formatRemaining(remaining)}`
              : formatDuration(duration)}
          </div>
        )}
      </div>
      <Waveform
        peaks={waveform.data?.peaks ?? []}
        position={position}
        height={32}
        className={`${styles.text} opacity-90 mt-auto`}
        playheadColor="rgba(255,255,255,0.9)"
        regions={regions.data ?? undefined}
        durationSeconds={duration}
        onSeek={trackIdx != null ? handleSeek : undefined}
      />
    </div>
  );
}
