import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import * as api from "../api";
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
  // Which side (if any) is currently PFL'd to headphones. PFL on a side
  // sends Solo on every stem track of that side; the SINGLE source of
  // truth for "is it lit" is the contract's ``soloed_kinds`` (Live's real
  // solo bus), so this toggle and the header "S" buttons can't disagree.
  // A side counts as PFL-active when any of its 4 stem-kinds are soloed.
  const soloedKinds = ableton.soloed_kinds;
  const pflSide: "a" | "b" | null =
    soloedKinds.some((k) => k.endsWith("_a"))
      ? "a"
      : soloedKinds.some((k) => k.endsWith("_b"))
      ? "b"
      : null;
  const setPfl = useMutation({
    mutationFn: (side: "a" | "b" | "off") => api.abletonSetPfl(side),
  });
  function togglePfl(side: "a" | "b") {
    // Optimistic command only — lit state is recomputed from soloed_kinds
    // on the next WS frame, so we don't keep a local mirror.
    const next = pflSide === side ? "off" : side;
    setPfl.mutate(next);
  }
  const setTempo = useMutation({ mutationFn: api.abletonSetTempo });
  // Bring the original full-song (mix) clip in on a side — the "fall back
  // to the record" parachute. The mix clip sits loaded-but-silent until
  // fired; firing it unmutes the anchor reference. (No mute OSC endpoint
  // exists, so firing the mix clip is how we surface the original.)
  const unmuteMix = useMutation({
    mutationFn: (vars: { trackIdx: number; sceneIdx: number }) =>
      api.abletonFireClip(vars.trackIdx, vars.sceneIdx),
  });
  const fireDeck = useMutation({
    mutationFn: (vars: { side: "a" | "b"; sceneIdx: number }) =>
      api.abletonFireDeck(vars.side, vars.sceneIdx),
  });
  const stopDeck = useMutation({
    mutationFn: (side: "a" | "b") => api.abletonStopDeck(side),
  });
  // FX state — light read every 2s so the HPF toggle stays in sync if
  // the user filters from elsewhere (Live, OSC). Cheap, single endpoint.
  const fxState = useQuery({
    queryKey: ["ableton", "fx", "state"],
    queryFn: api.abletonFxState,
    refetchInterval: 2000,
  });
  const fxToggle = useMutation({
    mutationFn: (vars: { fx: "filter" | "reverb" | "delay"; side: "a" | "b" }) =>
      api.abletonFxToggle(vars.fx, vars.side),
    onSuccess: () => fxState.refetch(),
  });
  const fxSweep = useMutation({
    mutationFn: (vars: {
      fx: "filter" | "reverb" | "delay";
      side: "a" | "b";
      bars: number;
      direction?: "in" | "out" | "auto";
    }) =>
      api.abletonFxSweep(vars.fx, vars.side, {
        bars: vars.bars,
        direction: vars.direction ?? "auto",
      }),
    onSuccess: () => fxState.refetch(),
  });
  const fxFire = useMutation({
    mutationFn: (name: string) => api.abletonFxFire(name),
  });
  const filterActive = fxState.data?.filter ?? { a: false, b: false };
  const reverbActive = fxState.data?.reverb ?? { a: false, b: false };
  const delayActive = fxState.data?.delay ?? { a: false, b: false };
  const returns = fxState.data?.discovered?.returns ?? {};
  const filterAvailable = "filter" in returns;
  const reverbAvailable = "reverb" in returns;
  const delayAvailable = "delay" in returns;
  const riserAvailable =
    fxState.data?.discovered?.scenes
    && "riser" in fxState.data.discovered.scenes;

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
    <div className="relative" data-testid="two-deck-strip">
      {/* One-shot FX cluster — sits ON TOP of the deck panels, centered
          between them. Riser fires the named FX clip; quantized to the
          next bar by the .als writer's LaunchQuantisation = 1. Disabled
          (with a tooltip explaining why) when the .als doesn't have a
          riser clip authored yet. */}
      <div
        className="absolute left-1/2 top-0 -translate-x-1/2 z-10 flex items-center gap-1 pt-1"
        data-testid="fx-cluster"
      >
        <button
          type="button"
          onClick={() => fxFire.mutate("riser")}
          disabled={!riserAvailable || fxFire.isPending}
          title={
            riserAvailable
              ? "Fire the Riser one-shot on the next bar"
              : "No 'Riser' clip in this .als — author one per docs/proposals/fx-phase-1-runbook.md"
          }
          aria-label="Fire riser"
          className={`text-[10px] font-bold uppercase tracking-widest leading-none rounded px-2.5 py-1.5 border transition-colors ${
            riserAvailable
              ? "bg-amber-500/20 hover:bg-amber-500/40 text-amber-100 border-amber-400/50 hover:border-amber-300"
              : "bg-neutral-900 text-neutral-700 border-neutral-800 cursor-not-allowed"
          }`}
        >
          ▲ Riser
        </button>
      </div>
      <div className="grid grid-cols-2 gap-2">
      <DeckPanel
        side="a"
        columns={columns}
        cellAt={cellAt}
        playing={playing}
        positions={positions}
        tempo={tempo}
        beat={beat}
        pflActive={pflSide === "a"}
        onTogglePfl={() => togglePfl("a")}
        onSyncTempo={(bpm) => setTempo.mutate(bpm)}
        onFireDeck={(sceneIdx) => fireDeck.mutate({ side: "a", sceneIdx })}
        onStopDeck={() => stopDeck.mutate("a")}
        onUnmuteMix={(sceneIdx) => {
          const t = columns["mix_a"];
          if (t != null) unmuteMix.mutate({ trackIdx: t, sceneIdx });
        }}
        filterActive={filterActive.a}
        filterAvailable={!!filterAvailable}
        onToggleFilter={() =>
          fxToggle.mutate({ fx: "filter", side: "a" })
        }
        onSweepFilter={(bars) =>
          fxSweep.mutate({ fx: "filter", side: "a", bars })
        }
        reverbActive={reverbActive.a}
        reverbAvailable={reverbAvailable}
        onToggleReverb={() => fxToggle.mutate({ fx: "reverb", side: "a" })}
        delayActive={delayActive.a}
        delayAvailable={delayAvailable}
        onToggleDelay={() => fxToggle.mutate({ fx: "delay", side: "a" })}
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
        pflActive={pflSide === "b"}
        onTogglePfl={() => togglePfl("b")}
        onSyncTempo={(bpm) => setTempo.mutate(bpm)}
        onFireDeck={(sceneIdx) => fireDeck.mutate({ side: "b", sceneIdx })}
        onStopDeck={() => stopDeck.mutate("b")}
        onUnmuteMix={(sceneIdx) => {
          const t = columns["mix_b"];
          if (t != null) unmuteMix.mutate({ trackIdx: t, sceneIdx });
        }}
        filterActive={filterActive.b}
        filterAvailable={!!filterAvailable}
        onToggleFilter={() =>
          fxToggle.mutate({ fx: "filter", side: "b" })
        }
        onSweepFilter={(bars) =>
          fxSweep.mutate({ fx: "filter", side: "b", bars })
        }
        reverbActive={reverbActive.b}
        reverbAvailable={reverbAvailable}
        onToggleReverb={() => fxToggle.mutate({ fx: "reverb", side: "b" })}
        delayActive={delayActive.b}
        delayAvailable={delayAvailable}
        onToggleDelay={() => fxToggle.mutate({ fx: "delay", side: "b" })}
        onSeek={(track, slot, beats) =>
          seek.mutate({ track, slot, positionBeats: beats })
        }
      />
      </div>
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
  pflActive,
  onTogglePfl,
  onSyncTempo,
  onFireDeck,
  onStopDeck,
  onUnmuteMix,
  filterActive,
  filterAvailable,
  onToggleFilter,
  onSweepFilter,
  reverbActive,
  reverbAvailable,
  onToggleReverb,
  delayActive,
  delayAvailable,
  onToggleDelay,
  onSeek,
}: {
  side: "a" | "b";
  columns: Record<string, number>;
  cellAt: Map<string, DeckCell>;
  playing: Record<number, number>;
  positions: Record<string, number>;
  tempo: number | null;
  beat: number | null;
  pflActive: boolean;
  onTogglePfl: () => void;
  onSyncTempo: (bpm: number) => void;
  onFireDeck: (sceneIdx: number) => void;
  onStopDeck: () => void;
  onUnmuteMix: (sceneIdx: number) => void;
  filterActive: boolean;
  filterAvailable: boolean;
  onToggleFilter: () => void;
  onSweepFilter: (bars: number) => void;
  reverbActive: boolean;
  reverbAvailable: boolean;
  onToggleReverb: () => void;
  delayActive: boolean;
  delayAvailable: boolean;
  onToggleDelay: () => void;
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

  // Mix/anchor reference for this side — the original full song. Shown as
  // a muted ◇ chip in the header (the proposal's "fall back to the
  // record" parachute). Click fires the mix clip so the original is
  // audible. Lit when that mix clip is currently the firing clip.
  const mixCell: DeckCell | undefined =
    anchor != null ? cellAt.get(`${anchor.sceneIdx}|mix_${side}`) : undefined;
  const mixTrackIdx = columns[`mix_${side}`];
  const mixFiring =
    mixTrackIdx != null && anchor != null
      ? playing[mixTrackIdx] === anchor.sceneIdx
      : false;

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
        anchorSceneIdx={anchor?.sceneIdx}
        isPureAnchor={anchor?.isPureAnchor ?? false}
        firing={anchor?.firing ?? false}
        pflActive={pflActive}
        onTogglePfl={onTogglePfl}
        mixCell={mixCell}
        mixFiring={mixFiring}
        onUnmuteMix={() =>
          anchor != null && onUnmuteMix(anchor.sceneIdx)
        }
        projectTempo={tempo}
        onSyncTempo={onSyncTempo}
        onFireDeck={onFireDeck}
        onStopDeck={onStopDeck}
        filterActive={filterActive}
        filterAvailable={filterAvailable}
        onToggleFilter={onToggleFilter}
        onSweepFilter={onSweepFilter}
        reverbActive={reverbActive}
        reverbAvailable={reverbAvailable}
        onToggleReverb={onToggleReverb}
        delayActive={delayActive}
        delayAvailable={delayAvailable}
        onToggleDelay={onToggleDelay}
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
  anchorSceneIdx,
  isPureAnchor,
  firing,
  pflActive,
  onTogglePfl,
  mixCell,
  mixFiring,
  onUnmuteMix,
  projectTempo,
  onSyncTempo,
  onFireDeck,
  onStopDeck,
  filterActive,
  filterAvailable,
  onToggleFilter,
  onSweepFilter,
  reverbActive,
  reverbAvailable,
  onToggleReverb,
  delayActive,
  delayAvailable,
  onToggleDelay,
}: {
  side: "a" | "b";
  sideLabel: string;
  anchorCell: DeckCell | undefined;
  anchorSceneIdx: number | undefined;
  isPureAnchor: boolean;
  firing: boolean;
  pflActive: boolean;
  onTogglePfl: () => void;
  /** The original full-song (mix) clip loaded on this side, if any —
   * surfaced as a muted ◇ chip the DJ can fire to fall back to the
   * record. ``undefined`` when no mix clip is loaded for the side. */
  mixCell: DeckCell | undefined;
  /** Whether that mix clip is the clip currently firing on this side. */
  mixFiring: boolean;
  onUnmuteMix: () => void;
  projectTempo: number | null;
  onSyncTempo: (bpm: number) => void;
  onFireDeck: (sceneIdx: number) => void;
  onStopDeck: () => void;
  filterActive: boolean;
  filterAvailable: boolean;
  onToggleFilter: () => void;
  onSweepFilter: (bars: number) => void;
  reverbActive: boolean;
  reverbAvailable: boolean;
  onToggleReverb: () => void;
  delayActive: boolean;
  delayAvailable: boolean;
  onToggleDelay: () => void;
}) {
  const accentText = side === "a" ? "text-violet-200" : "text-indigo-200";
  // Sync gesture: snap the project's master tempo to this deck's
  // anchored track BPM. Only shows when there's a delta > 0.5 BPM.
  const sourceBpm = anchorCell?.bpm ?? null;
  const showSync =
    sourceBpm != null
    && projectTempo != null
    && Math.abs(projectTempo - sourceBpm) > 0.5;
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
      {/* Per-deck play / stop. Play fires all 5 of this side's cells
          (4 stems + mix) at the anchor scene; stop halts every clip
          on this side's tracks. Disabled when no anchor scene exists
          yet (nothing to fire). */}
      <button
        type="button"
        onClick={() => anchorSceneIdx != null && onFireDeck(anchorSceneIdx)}
        disabled={anchorSceneIdx == null}
        title={
          anchorSceneIdx == null
            ? `Deck ${sideLabel} is empty`
            : `Play Deck ${sideLabel} (fire scene ${anchorSceneIdx + 1})`
        }
        aria-label={`Play Deck ${sideLabel}`}
        className={`text-[10px] leading-none rounded px-1.5 py-1 border transition-colors ${
          firing
            ? "bg-emerald-500/40 text-emerald-50 border-emerald-300/60"
            : "bg-neutral-900 text-neutral-300 hover:bg-emerald-900/40 hover:text-emerald-200 border-neutral-700 disabled:opacity-40 disabled:hover:bg-neutral-900 disabled:hover:text-neutral-300"
        }`}
      >
        ▶
      </button>
      <button
        type="button"
        onClick={onStopDeck}
        title={`Stop Deck ${sideLabel} (halt all of this side's clips)`}
        aria-label={`Stop Deck ${sideLabel}`}
        className="text-[10px] leading-none rounded px-1.5 py-1 border bg-neutral-900 text-neutral-400 hover:bg-rose-950/40 hover:text-rose-200 border-neutral-700 transition-colors"
      >
        ■
      </button>
      <button
        type="button"
        onClick={onTogglePfl}
        title={
          pflActive
            ? `Stop monitoring Deck ${sideLabel} in headphones`
            : `Monitor Deck ${sideLabel} in headphones (PFL via Live's Solo with Solo/Cue=Cue)`
            + "\nCaveat: Live's Cue is additive — a playing deck stays on Master too,"
            + " so PFL adds it to headphones rather than isolating it."
        }
        aria-pressed={pflActive}
        aria-label={`PFL Deck ${sideLabel}`}
        className={`text-[9px] font-mono uppercase tracking-widest leading-none rounded px-1.5 py-1 border transition-colors ${
          pflActive
            ? "bg-amber-400/30 text-amber-100 border-amber-300/60"
            : "text-neutral-500 hover:text-amber-200 border-neutral-700 hover:border-amber-400/40"
        }`}
      >
        PFL
      </button>
      <button
        type="button"
        onClick={(e) => {
          // Shift-click = continuous 4-bar sweep. Plain click = snap toggle.
          // Gives DJs both gestures on one button: tap for instant, hold
          // shift for analog-style ramp.
          if (e.shiftKey) onSweepFilter(4);
          else onToggleFilter();
        }}
        disabled={!filterAvailable}
        title={
          !filterAvailable
            ? "No 'Filter' return in this .als — author one per docs/proposals/fx-phase-1-runbook.md"
            : filterActive
            ? `Filter off — Deck ${sideLabel} returns to full range\nShift-click = 4-bar sweep`
            : `Filter on — HPF Deck ${sideLabel} (cuts lows)\nShift-click = 4-bar sweep`
        }
        aria-pressed={filterActive}
        aria-label={`HPF Deck ${sideLabel}`}
        className={`text-[9px] font-mono uppercase tracking-widest leading-none rounded px-1.5 py-1 border transition-colors ${
          !filterAvailable
            ? "text-neutral-700 border-neutral-800 cursor-not-allowed"
            : filterActive
            ? "bg-rose-500/30 text-rose-100 border-rose-400/60"
            : "text-neutral-500 hover:text-rose-200 border-neutral-700 hover:border-rose-400/40"
        }`}
      >
        HPF
      </button>
      <button
        type="button"
        onClick={onToggleReverb}
        disabled={!reverbAvailable}
        title={
          !reverbAvailable
            ? "No 'Reverb' return in this .als — see fx-phase-1-runbook.md (same pattern as Filter)"
            : reverbActive
            ? `Reverb off — Deck ${sideLabel} dry`
            : `Reverb throw — send Deck ${sideLabel} into the reverb tail`
        }
        aria-pressed={reverbActive}
        aria-label={`Reverb throw Deck ${sideLabel}`}
        className={`text-[9px] font-mono uppercase tracking-widest leading-none rounded px-1.5 py-1 border transition-colors ${
          !reverbAvailable
            ? "text-neutral-700 border-neutral-800 cursor-not-allowed"
            : reverbActive
            ? "bg-cyan-500/30 text-cyan-100 border-cyan-400/60"
            : "text-neutral-500 hover:text-cyan-200 border-neutral-700 hover:border-cyan-400/40"
        }`}
      >
        REV
      </button>
      <button
        type="button"
        onClick={onToggleDelay}
        disabled={!delayAvailable}
        title={
          !delayAvailable
            ? "No 'Delay' return in this .als — see fx-phase-1-runbook.md (same pattern as Filter)"
            : delayActive
            ? `Delay off — Deck ${sideLabel} dry`
            : `Delay throw — echo Deck ${sideLabel} into the next bars`
        }
        aria-pressed={delayActive}
        aria-label={`Delay throw Deck ${sideLabel}`}
        className={`text-[9px] font-mono uppercase tracking-widest leading-none rounded px-1.5 py-1 border transition-colors ${
          !delayAvailable
            ? "text-neutral-700 border-neutral-800 cursor-not-allowed"
            : delayActive
            ? "bg-fuchsia-500/30 text-fuchsia-100 border-fuchsia-400/60"
            : "text-neutral-500 hover:text-fuchsia-200 border-neutral-700 hover:border-fuchsia-400/40"
        }`}
      >
        DLY
      </button>
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
          <div className="text-[10px] font-mono tabular-nums text-neutral-500 shrink-0 flex items-baseline gap-1.5">
            {sourceBpm != null && <span>{sourceBpm.toFixed(0)}</span>}
            {showSync && (
              <button
                type="button"
                onClick={() => onSyncTempo(Number(sourceBpm!.toFixed(2)))}
                title={`Set master tempo to ${sourceBpm!.toFixed(1)} BPM (currently ${projectTempo!.toFixed(1)})`}
                className="text-[9px] font-bold leading-none rounded px-1 py-0.5 bg-emerald-700/40 hover:bg-emerald-600/60 text-emerald-100 transition-colors"
              >
                SYNC
              </button>
            )}
            {anchorCell.key_camelot && (
              <span>· {anchorCell.key_camelot}</span>
            )}
            {anchorCell.floor_energy != null && (
              <span className="text-neutral-600">
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
          {/* Mix / anchor reference — the original full song, replacing the
              old SONG grid column. Muted ◇ chip; click fires the mix clip
              so the record itself comes back in (the "fall back" parachute).
              Lit emerald while the mix clip is the firing clip. */}
          {mixCell && (
            <button
              type="button"
              onClick={onUnmuteMix}
              title={
                mixFiring
                  ? `Original (${mixCell.title ?? `Track #${mixCell.track_id}`}) is playing on Deck ${sideLabel}`
                  : `Fall back to the original record (${mixCell.title ?? `Track #${mixCell.track_id}`}) on Deck ${sideLabel}`
              }
              aria-label={`Play original mix on Deck ${sideLabel}`}
              aria-pressed={mixFiring}
              className={`shrink-0 text-[9px] font-mono uppercase tracking-wider leading-none rounded px-1.5 py-1 border transition-colors inline-flex items-center gap-1 max-w-[8rem] ${
                mixFiring
                  ? "bg-emerald-500/30 text-emerald-100 border-emerald-400/60"
                  : "text-neutral-400 hover:text-neutral-100 border-neutral-700 hover:border-neutral-500"
              }`}
            >
              <span>◇</span>
              <span className="truncate">
                {mixCell.title ?? `Track #${mixCell.track_id}`}
              </span>
            </button>
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
