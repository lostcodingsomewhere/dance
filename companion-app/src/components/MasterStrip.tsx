import { useMutation } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import * as api from "../api";
import { useAbletonState } from "../hooks/useAbletonState";
import { useBridgeHeartbeat } from "../hooks/useBridgeHeartbeat";
import { useDeckMap, useResyncDecks } from "../hooks/useDeckMap";
import { store, useAppStore } from "../store";
import type { ViewName } from "../types";
import { BpmSlider } from "./BpmSlider";
import { DepsChip } from "./DepsChip";
import { EnergySparkline } from "./EnergySparkline";
import { SessionChip } from "./SessionChip";
import { VuMeter } from "./VuMeter";

const VIEWS: { id: ViewName; label: string; hint: string }[] = [
  { id: "booth", label: "Booth", hint: "Live performance — SceneGrid + recs" },
  { id: "set", label: "Set", hint: "Edit the active set (rail expanded)" },
  { id: "pipeline", label: "Pipeline", hint: "Ingest & processing status (library inventory)" },
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
  const resync = useResyncDecks();

  // Camelot key of the dominant playing cell — the harmonic anchor for compat
  // math. We scan playing_clips against the deck columns and pick the first
  // cell that's actually playing (vocals first — the human voice anchors
  // perceived key most strongly). Falls back to null when nothing is firing.
  //
  // Deck-pair: scans both A and B sides per source role; first playing cell
  // wins regardless of side.
  const currentKey = useMemo<string | null>(() => {
    const columns = deckMap.data?.columns;
    const cells = deckMap.data?.cells ?? [];
    const playing = state.playing_clips ?? {};
    if (!columns) return null;
    const orderedDeckKinds = [
      "vocals_a", "vocals_b",
      "bass_a", "bass_b",
      "drums_a", "drums_b",
      "other_a", "other_b",
      "mix",
    ];
    for (const dk of orderedDeckKinds) {
      const trackIdx = columns[dk];
      if (trackIdx == null) continue;
      const sIdx = playing[trackIdx];
      if (sIdx == null) continue;
      const cell = cells.find(
        (c) => c.scene_index === sIdx && c.kind === dk,
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

      {/* Transport — icons only. Three distinct actions:
            ▶/⏸  toggle play / pause Live's transport (clock)
            ■    stop transport (same OSC, but the visual "halt" state)
            🛑   panic — stop every firing clip but leave transport
                 running so the next fire syncs cleanly
            ●    session record toggle */}
      <div className="flex items-center gap-1 px-1 border-l border-neutral-800 h-10">
        <button
          type="button"
          onClick={() => play.mutate()}
          className={`min-h-[40px] w-10 rounded-md font-semibold text-base flex items-center justify-center ${
            state.is_playing
              ? "bg-emerald-500 text-neutral-950"
              : "bg-neutral-800 text-neutral-200 hover:bg-neutral-700"
          }`}
          aria-label={state.is_playing ? "Pause" : "Play"}
          title={state.is_playing ? "Pause transport" : "Play transport"}
        >
          {state.is_playing ? <IconPause /> : <IconPlay />}
        </button>
        <button
          type="button"
          onClick={() => stop.mutate()}
          className="min-h-[40px] w-10 rounded-md bg-neutral-800 text-neutral-200 hover:bg-neutral-700 flex items-center justify-center"
          aria-label="Stop transport"
          title="Stop transport (master clock halts)"
        >
          <IconStop />
        </button>
        <PanicButton />
        <RecordButton />
      </div>

      {/* Camelot key of the dominant playing scene (the anchor for compat). */}
      <KeyDisplay camelot={currentKey} />

      {/* Live combo VU — bouncing bar fed by deck-column output meters. */}
      <VuMeter />

      {/* Set-arc energy sparkline — ambient view of the trajectory so far. */}
      <EnergySparkline />

      {/* Live bridge heartbeat — red when AbletonOSC has gone stale. */}
      <HeartbeatDot alive={heartbeat.alive} />

      {/* Resync — non-destructive: scans Live's deck-column clips and
          adopts whatever's there into bridge state. Use when the
          SceneGrid looks empty but Live has clips loaded. */}
      <button
        type="button"
        onClick={() => resync.mutate()}
        disabled={resync.isPending}
        title={
          resync.data
            ? `Last resync: adopted ${resync.data.adopted}/${resync.data.scanned} clips from Live`
            : "Adopt clips already in Live into the SceneGrid (non-destructive)"
        }
        className="h-7 px-2 rounded text-[10px] uppercase tracking-widest text-neutral-500 hover:text-amber-300 hover:bg-neutral-900 border border-transparent hover:border-neutral-800 transition-colors disabled:opacity-50"
      >
        {resync.isPending
          ? "scanning…"
          : resync.data
          ? `↻ ${resync.data.adopted}/${resync.data.scanned}`
          : "↻ resync"}
      </button>


      <div className="ml-auto flex items-center gap-2">
        <DepsChip />
        <SessionChip />
      </div>

      {/* Command bar trigger */}
      <button
        type="button"
        onClick={() => store.openCommandBar()}
        className="h-10 px-3 rounded-md bg-neutral-900 border border-neutral-800 hover:border-neutral-700 text-sm text-neutral-400 flex items-center gap-2"
        title="Vibe search (⌘K)"
      >
        <span aria-hidden>✦</span>
        <span>Search…</span>
        <kbd className="font-mono text-[10px] text-neutral-500 border border-neutral-700 rounded px-1">
          ⌘K
        </kbd>
      </button>

      <ViewNav view={view} />
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

/** Panic = stop every firing clip but leave the master transport running,
 * so the next fired clip syncs cleanly. Distinct from the Stop button,
 * which halts the master clock. Two-click "arm + fire" gesture so it can't
 * be hit by accident during a set.
 */
function PanicButton() {
  const [armed, setArmed] = useState(false);
  const panic = useMutation({ mutationFn: api.abletonStopAllClips });
  if (!armed) {
    return (
      <button
        type="button"
        onClick={() => setArmed(true)}
        className="min-h-[40px] w-10 rounded-md bg-neutral-900 text-neutral-500 hover:text-rose-300 hover:bg-rose-950/40 flex items-center justify-center"
        aria-label="Panic — stop all clips"
        title="Panic: stop all firing clips, keep transport running. Click once to arm; click again to fire."
      >
        <IconPanic />
      </button>
    );
  }
  return (
    <button
      type="button"
      onClick={() => {
        panic.mutate();
        setArmed(false);
      }}
      onBlur={() => setArmed(false)}
      autoFocus
      className="min-h-[40px] px-3 rounded-md bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs flex items-center justify-center gap-1"
      title="Stop all clips — click to commit"
    >
      <IconPanic /> PANIC
    </button>
  );
}

/** Session record toggle. Live captures into armed tracks; we just
 * flip the global record_mode. Local elapsed-time ring ticks off when
 * we received the ok response — no separate /status endpoint needed
 * since record is binary and we own the source of truth here.
 */
function RecordButton() {
  const [recording, setRecording] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const toggle = useMutation({
    mutationFn: (on: boolean) => api.abletonRecord(on),
    onSuccess: (res) => {
      setRecording(res.recording);
      setStartedAt(res.recording ? Date.now() : null);
      setElapsed(0);
    },
  });
  useEffect(() => {
    if (!recording || startedAt == null) return;
    const id = window.setInterval(() => {
      setElapsed(Date.now() - startedAt);
    }, 250);
    return () => window.clearInterval(id);
  }, [recording, startedAt]);
  const mm = Math.floor(elapsed / 60_000).toString().padStart(2, "0");
  const ss = Math.floor((elapsed / 1000) % 60).toString().padStart(2, "0");
  return (
    <button
      type="button"
      onClick={() => toggle.mutate(!recording)}
      disabled={toggle.isPending}
      className={`min-h-[40px] rounded-md flex items-center justify-center gap-1.5 font-mono tabular-nums text-[11px] transition-colors disabled:opacity-50 ${
        recording
          ? "w-[72px] px-2 bg-rose-600 text-white"
          : "w-10 bg-neutral-900 text-neutral-500 hover:text-rose-300 hover:bg-rose-950/40"
      }`}
      aria-label={recording ? `Stop recording (${mm}:${ss})` : "Start recording"}
      title={
        recording
          ? `Recording — ${mm}:${ss}. Click to stop.`
          : "Start session record (Live captures into armed tracks)"
      }
    >
      {recording ? <IconRecordStop /> : <IconRecordDot />}
      {recording && <span>{mm}:{ss}</span>}
    </button>
  );
}

function IconPlay() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <polygon points="3,2 12,7 3,12" />
    </svg>
  );
}

function IconPause() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <rect x="3" y="2" width="3" height="10" rx="0.5" />
      <rect x="8" y="2" width="3" height="10" rx="0.5" />
    </svg>
  );
}

function IconStop() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <rect x="2.5" y="2.5" width="9" height="9" rx="1" />
    </svg>
  );
}

function IconPanic() {
  // Triangle with an exclamation — "stop everything firing right now."
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <path d="M7 1.2L13 12.5H1z" fillOpacity="0.15" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
      <rect x="6.4" y="5" width="1.2" height="4" rx="0.4" />
      <rect x="6.4" y="9.6" width="1.2" height="1.4" rx="0.4" />
    </svg>
  );
}

function IconRecordDot() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
      <circle cx="6" cy="6" r="4" />
    </svg>
  );
}

function IconRecordStop() {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
      <rect x="1" y="1" width="8" height="8" rx="0.5" />
    </svg>
  );
}

function IconBooth({ className }: { className?: string }) {
  return (
    <svg className={className} width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <rect x="0" y="0" width="6" height="6" rx="1.2"/>
      <rect x="8" y="0" width="6" height="6" rx="1.2"/>
      <rect x="0" y="8" width="6" height="6" rx="1.2"/>
      <rect x="8" y="8" width="6" height="6" rx="1.2"/>
    </svg>
  );
}

function IconSet({ className }: { className?: string }) {
  return (
    <svg className={className} width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <circle cx="1.5" cy="2.5" r="1.5"/>
      <rect x="4.5" y="1.5" width="9.5" height="2" rx="1"/>
      <circle cx="1.5" cy="7" r="1.5"/>
      <rect x="4.5" y="6" width="9.5" height="2" rx="1"/>
      <circle cx="1.5" cy="11.5" r="1.5"/>
      <rect x="4.5" y="10.5" width="7" height="2" rx="1"/>
    </svg>
  );
}

function IconPipeline({ className }: { className?: string }) {
  return (
    <svg className={className} width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <circle cx="2" cy="7" r="2"/>
      <rect x="5" y="6.2" width="2.5" height="1.6" rx="0.8"/>
      <circle cx="9.5" cy="7" r="2"/>
      <rect x="12" y="6.2" width="2" height="1.6" rx="0.8"/>
    </svg>
  );
}

function IconCourse({ className }: { className?: string }) {
  return (
    <svg className={className} width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
      <path d="M7 1L8.5 5h4L9.5 7.5 11 12 7 9.5 3 12l1.5-4.5L1.5 5h4z"/>
    </svg>
  );
}

// In Booth view the nav collapses: only the active Booth tab stays visible,
// and Set / Pipeline / Course move into a ⋯ dropdown so they don't clutter
// the live performance surface.
function ViewNav({ view }: { view: ViewName }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const isBooth = view === "booth";

  const tabCls = (active: boolean) =>
    `min-h-[40px] px-3 rounded-md font-semibold text-sm flex items-center gap-1.5 ${
      active ? "bg-neutral-100 text-neutral-950" : "text-neutral-300 hover:bg-neutral-800"
    }`;

  return (
    <nav className="relative flex items-center gap-1" role="tablist">
      {/* Booth tab — always visible */}
      <button
        role="tab"
        aria-selected={isBooth}
        onClick={() => store.setView("booth")}
        title="Live performance — SceneGrid + recs"
        className={tabCls(isBooth)}
      >
        <IconBooth />
        Booth
      </button>

      {/* Set + Pipeline tabs: visible when not in Booth */}
      {!isBooth && (
        <>
          <button
            role="tab"
            aria-selected={view === "set"}
            onClick={() => store.setView("set")}
            title="Edit the active set"
            className={tabCls(view === "set")}
          >
            <IconSet />
            Set
          </button>
          <button
            role="tab"
            aria-selected={view === "pipeline"}
            onClick={() => store.setView("pipeline")}
            title="Ingest & processing status"
            className={tabCls(view === "pipeline")}
          >
            <IconPipeline />
            Pipeline
          </button>
          <a
            href="/course/index.html"
            target="_blank"
            rel="noopener noreferrer"
            title="Open DJ course (progress saves automatically)"
            className={tabCls(false)}
          >
            <IconCourse className="text-emerald-400" />
            Course
          </a>
        </>
      )}

      {/* In Booth: ⋯ overflow menu with Set / Pipeline / Course */}
      {isBooth && (
        <div className="relative">
          <button
            type="button"
            onClick={() => setMenuOpen((o) => !o)}
            onBlur={(e) => {
              if (!e.currentTarget.parentElement?.contains(e.relatedTarget as Node)) {
                setMenuOpen(false);
              }
            }}
            title="Set, Pipeline, Course"
            className={`min-h-[40px] w-10 rounded-md font-semibold text-lg flex items-center justify-center transition-colors ${
              menuOpen
                ? "bg-neutral-800 text-neutral-100"
                : "text-neutral-500 hover:bg-neutral-800 hover:text-neutral-300"
            }`}
          >
            ···
          </button>

          {menuOpen && (
            <div className="absolute right-0 top-[calc(100%+4px)] z-50 bg-neutral-900 border border-neutral-700 rounded-lg shadow-xl overflow-hidden min-w-[148px]">
              <button
                type="button"
                onClick={() => { store.setView("set"); setMenuOpen(false); }}
                className="w-full px-4 py-2.5 text-left text-sm font-semibold text-neutral-300 hover:bg-neutral-800 hover:text-white flex items-center gap-2.5"
              >
                <IconSet className="text-neutral-500" /> Set
              </button>
              <button
                type="button"
                onClick={() => { store.setView("pipeline"); setMenuOpen(false); }}
                className="w-full px-4 py-2.5 text-left text-sm font-semibold text-neutral-300 hover:bg-neutral-800 hover:text-white flex items-center gap-2.5"
              >
                <IconPipeline className="text-neutral-500" /> Pipeline
              </button>
              <a
                href="/course/index.html"
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setMenuOpen(false)}
                className="w-full px-4 py-2.5 text-left text-sm font-semibold text-neutral-300 hover:bg-neutral-800 hover:text-white flex items-center gap-2.5"
              >
                <IconCourse className="text-emerald-500" /> Course
              </a>
            </div>
          )}
        </div>
      )}
    </nav>
  );
}
