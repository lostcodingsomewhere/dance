import { useMutation } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import * as api from "../api";
import { useAbletonState } from "../hooks/useAbletonState";
import { useBridgeHeartbeat } from "../hooks/useBridgeHeartbeat";
import { useDeckMap } from "../hooks/useDeckMap";
import { store, useAppStore } from "../store";
import type { ViewName } from "../types";
import { BpmSlider } from "./BpmSlider";
import { EnergySparkline } from "./EnergySparkline";

const VIEWS: { id: ViewName; label: string; hint: string }[] = [
  { id: "booth", label: "Booth", hint: "Live performance — now / next / set" },
  { id: "crate", label: "Crate", hint: "Pre-set planning, library, vibe search" },
  { id: "pipeline", label: "Pipeline", hint: "Ingest & processing status" },
];

/**
 * Persistent top strip. Replaces the old TopBar. Shows live BPM (huge), a
 * full transport (play / stop / panic / halve / double), the view nav, and
 * a command-bar trigger.
 */
export function MasterStrip() {
  const state = useAbletonState();
  const view = useAppStore((s) => s.currentView);
  const deckMap = useDeckMap();
  const heartbeat = useBridgeHeartbeat();

  // Camelot key of the dominant playing cell — the harmonic anchor for compat
  // math. We scan playing_clips against the deck columns and pick the first
  // cell that's actually playing (drums first, so the kick's root note
  // anchors when present). Falls back to null when nothing is firing.
  const currentKey = useMemo<string | null>(() => {
    const columns = deckMap.data?.columns;
    const cells = deckMap.data?.cells ?? [];
    const playing = state.playing_clips ?? {};
    if (!columns) return null;
    for (const role of ["vocals", "bass", "drums", "other", "mix"]) {
      const trackIdx = columns[role];
      if (trackIdx == null) continue;
      const sIdx = playing[trackIdx];
      if (sIdx == null) continue;
      const cell = cells.find(
        (c) => c.scene_index === sIdx && c.kind === role,
      );
      if (cell?.key_camelot) return cell.key_camelot;
    }
    return null;
  }, [deckMap.data, state.playing_clips]);

  const play = useMutation({ mutationFn: api.abletonPlay });
  const stop = useMutation({ mutationFn: api.abletonStop });
  const setTempo = useMutation({ mutationFn: api.abletonSetTempo });

  // Click-to-edit BPM: opens a slider popover with genre BPM anchors.
  // Commit fires on slider release (or Enter); Escape / outside-click cancels.
  const [editingBpm, setEditingBpm] = useState(false);
  function commitBpm(n: number) {
    const clamped = Math.min(240, Math.max(40, n));
    setTempo.mutate(Number(clamped.toFixed(2)));
  }

  function halve() {
    if (state.tempo == null) return;
    const t = Math.max(40, state.tempo / 2);
    setTempo.mutate(Number(t.toFixed(2)));
  }
  function double() {
    if (state.tempo == null) return;
    const t = Math.min(240, state.tempo * 2);
    setTempo.mutate(Number(t.toFixed(2)));
  }

  return (
    <header className="h-16 shrink-0 flex items-center gap-3 px-4 border-b border-neutral-800 bg-neutral-950">
      <div className="flex items-center gap-2 pr-2 border-r border-neutral-800 h-10">
        <div className="w-6 h-6 rounded bg-amber-400" aria-hidden />
        <div className="text-neutral-200 font-semibold tracking-wide uppercase text-[11px]">
          Dance
        </div>
      </div>

      {/* Master BPM block — click the number for a slider with genre anchors,
          ×2/÷2 for quick jumps. */}
      <div className="relative flex items-baseline gap-1.5 px-2">
        <button
          type="button"
          onClick={() => setEditingBpm((s) => !s)}
          title="Click for BPM slider with genre anchors. Range 40–240."
          className={`font-mono text-3xl tabular-nums leading-none transition-colors cursor-pointer focus:outline-none ${
            editingBpm
              ? "text-amber-200"
              : "text-neutral-50 hover:text-amber-200 focus:text-amber-200"
          }`}
        >
          {state.tempo != null ? state.tempo.toFixed(1) : "--"}
        </button>
        <span className="text-neutral-500 text-[10px] uppercase tracking-wider">
          BPM
        </span>
        {editingBpm && state.tempo != null && (
          <BpmSlider
            value={state.tempo}
            onCommit={commitBpm}
            onClose={() => setEditingBpm(false)}
          />
        )}
        <div className="flex flex-col ml-1">
          <button
            type="button"
            onClick={double}
            className="text-[10px] leading-tight px-1.5 rounded bg-neutral-900 hover:bg-neutral-800 text-neutral-400"
            title="Double tempo (×2)"
          >
            ×2
          </button>
          <button
            type="button"
            onClick={halve}
            className="text-[10px] leading-tight px-1.5 rounded bg-neutral-900 hover:bg-neutral-800 text-neutral-400"
            title="Halve tempo (÷2)"
          >
            ÷2
          </button>
        </div>
      </div>

      {/* Transport */}
      <div className="flex items-center gap-1 px-1 border-l border-neutral-800 h-10">
        <button
          type="button"
          onClick={() => play.mutate()}
          className={`min-h-[40px] min-w-[60px] px-3 rounded-md font-semibold text-sm ${
            state.is_playing
              ? "bg-emerald-500 text-neutral-950"
              : "bg-neutral-800 text-neutral-200 hover:bg-neutral-700"
          }`}
          aria-label="Play"
        >
          {state.is_playing ? "Playing" : "Play"}
        </button>
        <button
          type="button"
          onClick={() => stop.mutate()}
          className="min-h-[40px] min-w-[56px] px-3 rounded-md bg-neutral-800 text-neutral-200 hover:bg-neutral-700 font-semibold text-sm"
          aria-label="Stop"
        >
          Stop
        </button>
        <PanicButton />
      </div>

      {/* Camelot key of the dominant playing scene (the anchor for compat). */}
      <KeyDisplay camelot={currentKey} />

      {/* Set-arc energy sparkline — ambient view of the trajectory so far. */}
      <EnergySparkline />

      {/* Live bridge heartbeat — red when AbletonOSC has gone stale. The
          deck-count and out-of-sync chips were removed: the SceneGrid mirror
          below already shows what's loaded, so the chip was redundant; and
          the "out-of-sync" warning surfaced stale localStorage state more
          confusingly than usefully. */}
      <HeartbeatDot alive={heartbeat.alive} />


      {/* Command bar trigger */}
      <button
        type="button"
        onClick={() => store.openCommandBar()}
        className="ml-auto h-10 px-3 rounded-md bg-neutral-900 border border-neutral-800 hover:border-neutral-700 text-sm text-neutral-400 flex items-center gap-2"
        title="Vibe search (⌘K)"
      >
        <span aria-hidden>✦</span>
        <span>Search…</span>
        <kbd className="font-mono text-[10px] text-neutral-500 border border-neutral-700 rounded px-1">
          ⌘K
        </kbd>
      </button>

      {/* View nav */}
      <nav className="flex items-center gap-1" role="tablist">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            role="tab"
            aria-selected={view === v.id}
            onClick={() => store.setView(v.id)}
            title={v.hint}
            className={`min-h-[40px] px-3 rounded-md font-semibold text-sm ${
              view === v.id
                ? "bg-neutral-100 text-neutral-950"
                : "text-neutral-300 hover:bg-neutral-800"
            }`}
          >
            {v.label}
          </button>
        ))}
      </nav>
    </header>
  );
}

function KeyDisplay({ camelot }: { camelot: string | null }) {
  return (
    <div
      className="flex items-baseline gap-1 px-2 border-l border-neutral-800 h-10 pt-1.5"
      title={
        camelot
          ? `Currently anchored at ${camelot} (Camelot). Compat math scores recs against this.`
          : "No scene currently playing"
      }
    >
      <span className="text-neutral-500 text-[10px] uppercase tracking-wider">
        Key
      </span>
      <span
        className={`font-mono text-2xl tabular-nums leading-none ${
          camelot ? "text-neutral-50" : "text-neutral-700"
        }`}
      >
        {camelot ?? "—"}
      </span>
    </div>
  );
}

function HeartbeatDot({ alive }: { alive: boolean }) {
  return (
    <div
      className="flex items-center gap-1.5 text-[11px]"
      title={
        alive
          ? "Ableton Live + AbletonOSC bridge responding"
          : "Ableton Live unreachable — check that Live is open with AbletonOSC enabled as a Control Surface"
      }
    >
      <span
        className={`w-2 h-2 rounded-full ${
          alive ? "bg-emerald-400" : "bg-rose-500 animate-pulse"
        }`}
      />
      <span className={alive ? "text-neutral-400" : "text-rose-300"}>
        {alive ? "Live" : "Live offline"}
      </span>
    </div>
  );
}

function PanicButton() {
  const [armed, setArmed] = useState(false);
  const stop = useMutation({ mutationFn: api.abletonStop });
  if (!armed) {
    return (
      <button
        type="button"
        onClick={() => setArmed(true)}
        className="min-h-[40px] px-3 rounded-md bg-neutral-900 text-neutral-500 hover:text-rose-300 hover:bg-rose-950/40 font-semibold text-sm"
        title="Hard-stop. Click once to arm; click again to fire."
      >
        ⏻
      </button>
    );
  }
  return (
    <button
      type="button"
      onClick={() => {
        stop.mutate();
        setArmed(false);
      }}
      onBlur={() => setArmed(false)}
      autoFocus
      className="min-h-[40px] px-3 rounded-md bg-rose-600 hover:bg-rose-500 text-white font-bold text-sm"
      title="Fires Stop — click to commit"
    >
      PANIC
    </button>
  );
}
