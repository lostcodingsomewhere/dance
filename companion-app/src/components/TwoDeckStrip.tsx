import { useMemo } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import { useRegions } from "../hooks/useRegions";
import { useSeekClip } from "../hooks/useTransport";
import { useStemWaveform, useTrackWaveform } from "../hooks/useWaveform";
import { formatDuration, formatRemaining } from "../lib/format";
import { ratioToBeats, snapRatioToSection } from "../lib/seek";
import type { DeckCell } from "../types";
import { Waveform } from "./Waveform";

/**
 * Two-deck "now playing" strip — replaces the old per-role ComboStrip
 * with a Traktor-style layout: one big deck panel per side (A on top,
 * B below), each showing the anchored song's metadata header plus a
 * stacked stem waveform.
 *
 * See docs/proposals/two-deck-ui-rethink.md.
 *
 * Anchor detection: a side is "anchored" when its 4 source stems all
 * point at the same track AT THE SAME SCENE that's currently firing.
 * If no scene is firing yet, we fall back to the LOWEST scene where
 * the 4 stems share a track (the prepared but unfired anchor).
 *
 * Mid-mashup (cells from different tracks on the same side): the
 * waveform falls back to the dominant track and the header notes
 * "mixed" so the user knows it's not a single-song anchor.
 */
export function TwoDeckStrip() {
  const deckMap = useDeckMap();
  const ableton = useAbletonState();
  const seek = useSeekClip();

  const columns = deckMap.data?.columns ?? null;
  const cells = deckMap.data?.cells ?? [];
  const playing = ableton.playing_clips ?? {};
  const positions = ableton.playing_positions ?? {};
  const tempo = ableton.tempo;
  const beat = ableton.beat;

  // (scene_index, deck_kind) → DeckCell, for O(1) lookup.
  const cellAt = useMemo(() => {
    const m = new Map<string, DeckCell>();
    for (const c of cells) m.set(`${c.scene_index}|${c.kind}`, c);
    return m;
  }, [cells]);

  if (!columns) {
    return (
      <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-3 text-xs text-neutral-600">
        Waiting for Ableton — load a track from the recs banner to begin a combo.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-2" data-testid="two-deck-strip">
      <DeckPanel
        side="a"
        columns={columns}
        cellAt={cellAt}
        playing={playing}
        positions={positions}
        tempo={tempo}
        beat={beat}
        onSeek={(track, slot, beats) =>
          seek.mutate({ track, slot, positionBeats: beats })
        }
      />
      <DeckPanel
        side="b"
        columns={columns}
        cellAt={cellAt}
        playing={playing}
        positions={positions}
        tempo={tempo}
        beat={beat}
        onSeek={(track, slot, beats) =>
          seek.mutate({ track, slot, positionBeats: beats })
        }
      />
    </div>
  );
}

/** One deck — header + stacked-stem waveform + transport. The "side"
 * is the deck identity; the panel's content auto-resolves from anchor
 * detection on that side's 4 stem cells. */
function DeckPanel({
  side,
  columns,
  cellAt,
  playing,
  positions,
  tempo,
  beat,
  onSeek,
}: {
  side: "a" | "b";
  columns: Record<string, number>;
  cellAt: Map<string, DeckCell>;
  playing: Record<number, number>;
  positions: Record<string, number>;
  tempo: number | null;
  beat: number | null;
  onSeek: (track: number, slot: number, beats: number) => void;
}) {
  const SOURCE_ROLES = ["drums", "bass", "vocals", "other"] as const;
  // Pick the scene that's anchoring this side. Priority:
  //   1. A scene where ANY stem on this side is currently firing — use
  //      that scene's anchor track if 4-of-4 match, else the firing
  //      cell's track as the dominant.
  //   2. The lowest scene where all 4 stems share a track (prepared
  //      anchor that hasn't fired yet).
  //   3. Whichever scene has the most populated stems on this side.
  const anchor = useMemo(() => {
    // (a) firing scene
    let firingScene: number | null = null;
    for (const r of SOURCE_ROLES) {
      const tIdx = columns[`${r}_${side}`];
      if (tIdx != null && playing[tIdx] != null) {
        firingScene = playing[tIdx];
        break;
      }
    }
    const candidate = firingScene;
    if (candidate != null) {
      const tids = SOURCE_ROLES.map(
        (r) => cellAt.get(`${candidate}|${r}_${side}`)?.track_id,
      );
      const present = tids.filter((t): t is number => t != null);
      if (present.length === 0) return null;
      // "Pure anchor" = 4-of-4 same track. "Mixed" = 2+ different
      // tracks present (real mashup). Single-stem loads are partial,
      // not mixed — they get isPureAnchor=true because the only
      // track present IS the anchor.
      const distinct = new Set(present).size;
      const isPureAnchor = distinct === 1;
      return {
        sceneIdx: candidate,
        trackId: present[0],
        isPureAnchor,
        firing: true,
      };
    }
    // (b) any scene where 4-of-4 match
    for (let s = 0; s < 16; s++) {
      const tids = SOURCE_ROLES.map(
        (r) => cellAt.get(`${s}|${r}_${side}`)?.track_id,
      );
      if (tids.every((t) => t != null) && new Set(tids).size === 1) {
        return {
          sceneIdx: s,
          trackId: tids[0] as number,
          isPureAnchor: true,
          firing: false,
        };
      }
    }
    // (c) any cell at all — sweep the lowest scene that has stuff,
    // pick the most common track id (if it's a single-track row,
    // that's our anchor; if it's a real mashup, we surface "mixed").
    for (let s = 0; s < 16; s++) {
      const tids = SOURCE_ROLES
        .map((r) => cellAt.get(`${s}|${r}_${side}`)?.track_id)
        .filter((t): t is number => t != null);
      if (tids.length === 0) continue;
      // Most common track id across the present cells.
      const counts = new Map<number, number>();
      for (const t of tids) counts.set(t, (counts.get(t) ?? 0) + 1);
      const [dominant] = [...counts.entries()].sort((a, b) => b[1] - a[1]);
      return {
        sceneIdx: s,
        trackId: dominant[0],
        // "Mixed" only when there are 2+ distinct tracks. Single-stem
        // and partial-load rows aren't mixed — they're just incomplete.
        isPureAnchor: counts.size === 1,
        firing: false,
      };
    }
    return null;
  }, [columns, cellAt, playing, side]);

  // Source-track metadata — read from any cell that matches anchor.
  const anchorCell: DeckCell | undefined = useMemo(() => {
    if (!anchor) return undefined;
    for (const r of SOURCE_ROLES) {
      const c = cellAt.get(`${anchor.sceneIdx}|${r}_${side}`);
      if (c && c.track_id === anchor.trackId) return c;
    }
    // Mix cell as fallback (post-full-song load with no surviving stems)
    return cellAt.get(`${anchor.sceneIdx}|mix_${side}`);
  }, [anchor, cellAt, side]);

  const sideLabel = side.toUpperCase();
  const accentBg = side === "a"
    ? "bg-gradient-to-br from-violet-950/30 via-neutral-950/60 to-neutral-950"
    : "bg-gradient-to-br from-indigo-950/30 via-neutral-950/60 to-neutral-950";
  const accentBorder = side === "a"
    ? "border-violet-700/40"
    : "border-indigo-700/40";

  return (
    <section
      className={`rounded-lg border ${accentBorder} ${accentBg} p-2 flex flex-col gap-1.5`}
      data-testid={`deck-panel-${side}`}
      aria-label={`Deck ${sideLabel}`}
    >
      <DeckHeader
        side={side}
        sideLabel={sideLabel}
        anchorCell={anchorCell}
        isPureAnchor={anchor?.isPureAnchor ?? false}
        firing={anchor?.firing ?? false}
      />
      <DeckWaveform
        side={side}
        sceneIdx={anchor?.sceneIdx}
        trackId={anchor?.trackId}
        anchorCell={anchorCell}
        columns={columns}
        cellAt={cellAt}
        playing={playing}
        positions={positions}
        tempo={tempo}
        beat={beat}
        onSeek={onSeek}
      />
    </section>
  );
}

function DeckHeader({
  side,
  sideLabel,
  anchorCell,
  isPureAnchor,
  firing,
}: {
  side: "a" | "b";
  sideLabel: string;
  anchorCell: DeckCell | undefined;
  isPureAnchor: boolean;
  firing: boolean;
}) {
  const accentText = side === "a" ? "text-violet-200" : "text-indigo-200";
  return (
    <div className="flex items-baseline gap-2">
      <div className={`text-base font-bold tracking-wider ${accentText}`}>
        DECK {sideLabel}
      </div>
      {firing && (
        <span className="text-[10px] text-emerald-300 uppercase tracking-widest animate-pulse">
          ▶ live
        </span>
      )}
      {anchorCell ? (
        <>
          <div className="text-sm text-neutral-100 truncate font-medium min-w-0 flex-1">
            {anchorCell.title ?? `Track #${anchorCell.track_id}`}
          </div>
          {anchorCell.artist && (
            <div className="text-[11px] text-neutral-400 truncate shrink-[2] min-w-0">
              {anchorCell.artist}
            </div>
          )}
          <div className="text-[10px] font-mono tabular-nums text-neutral-500 shrink-0">
            {anchorCell.bpm != null && (
              <span>{anchorCell.bpm.toFixed(0)}</span>
            )}
            {anchorCell.key_camelot && (
              <span className="ml-1.5">· {anchorCell.key_camelot}</span>
            )}
            {anchorCell.floor_energy != null && (
              <span className="ml-1.5 text-neutral-600">
                · E{anchorCell.floor_energy}
              </span>
            )}
          </div>
          {!isPureAnchor && (
            <span
              className="text-[10px] uppercase tracking-wider text-amber-400/70"
              title="Stems from multiple source tracks — not a single-song anchor"
            >
              mixed
            </span>
          )}
        </>
      ) : (
        <div className="text-xs text-neutral-600 italic">empty</div>
      )}
    </div>
  );
}

function DeckWaveform({
  side,
  sceneIdx,
  trackId,
  anchorCell,
  columns,
  cellAt,
  playing,
  positions,
  tempo,
  beat,
  onSeek,
}: {
  side: "a" | "b";
  sceneIdx: number | undefined;
  trackId: number | undefined;
  anchorCell: DeckCell | undefined;
  columns: Record<string, number>;
  cellAt: Map<string, DeckCell>;
  playing: Record<number, number>;
  positions: Record<string, number>;
  tempo: number | null;
  beat: number | null;
  onSeek: (track: number, slot: number, beats: number) => void;
}) {
  // Pull the mix's full-track waveform if available; falls back to a
  // single stem's waveform when no mix cell exists yet (partial loads).
  // Reuses the same hooks as ComboStrip — both are stable across renders
  // and no-op when their ids are null.
  const mixCell = sceneIdx != null
    ? cellAt.get(`${sceneIdx}|mix_${side}`)
    : undefined;
  const trackWf = useTrackWaveform(trackId ?? null);
  const drumsCell = sceneIdx != null
    ? cellAt.get(`${sceneIdx}|drums_${side}`)
    : undefined;
  const drumsWf = useStemWaveform(drumsCell?.stem_file_id);

  const waveform = mixCell || trackId ? trackWf : drumsWf;
  const duration = anchorCell?.duration_seconds
    ?? waveform.data?.duration_seconds;
  const clipBpm = anchorCell?.bpm ?? tempo;
  const regions = useRegions(trackId ?? null);

  // Pick a representative track index to drive the playhead — prefer a
  // currently-firing cell, else mix, else drums.
  const playheadTrackIdx = useMemo(() => {
    for (const r of ["drums", "bass", "vocals", "other"] as const) {
      const tIdx = columns[`${r}_${side}`];
      if (tIdx != null && playing[tIdx] === sceneIdx) return tIdx;
    }
    return columns[`mix_${side}`] ?? columns[`drums_${side}`];
  }, [columns, playing, sceneIdx, side]);
  const livePosBeats = playheadTrackIdx != null
    ? positions[String(playheadTrackIdx)]
    : undefined;

  // Convert to 0-1 ratio for the playhead.
  let position: number | undefined = undefined;
  let elapsedSec: number | undefined = undefined;
  if (duration && duration > 0 && clipBpm != null && clipBpm > 0) {
    if (livePosBeats != null) {
      elapsedSec = (livePosBeats / clipBpm) * 60;
      position = (elapsedSec % duration) / duration;
    } else if (beat != null && tempo != null) {
      elapsedSec = (beat / tempo) * 60;
      position = (elapsedSec % duration) / duration;
    }
  }
  const remaining = duration != null && elapsedSec != null
    ? duration - (elapsedSec % duration)
    : null;

  function handleSeek(ratio: number) {
    if (
      sceneIdx == null
      || playheadTrackIdx == null
      || !duration
      || !clipBpm
      || clipBpm <= 0
    ) return;
    const snapped = snapRatioToSection(ratio, regions.data, duration);
    onSeek(
      playheadTrackIdx,
      sceneIdx,
      ratioToBeats(snapped, duration, clipBpm),
    );
  }

  return (
    <div className="flex flex-col gap-0.5">
      <Waveform
        peaks={waveform.data?.peaks ?? []}
        position={position}
        height={56}
        className={
          side === "a"
            ? "text-violet-300/70"
            : "text-indigo-300/70"
        }
        playheadColor={side === "a" ? "rgb(196,181,253)" : "rgb(165,180,252)"}
        regions={regions.data ?? undefined}
        durationSeconds={duration ?? undefined}
        onSeek={anchorCell ? handleSeek : undefined}
      />
      <div className="flex items-baseline justify-between text-[10px] font-mono tabular-nums text-neutral-500">
        <span>
          {elapsedSec != null && duration != null
            ? formatDuration(elapsedSec % duration)
            : "—:—"}
        </span>
        <span>
          {remaining != null
            ? formatRemaining(remaining)
            : duration != null
            ? formatDuration(duration)
            : ""}
        </span>
      </div>
    </div>
  );
}
