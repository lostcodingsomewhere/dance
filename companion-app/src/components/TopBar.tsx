import { useMutation } from "@tanstack/react-query";
import * as api from "../api";
import { useAbletonState } from "../hooks/useAbletonState";
import { store, useAppStore } from "../store";
import type { ViewName } from "../types";

const VIEWS: { id: ViewName; label: string }[] = [
  { id: "now", label: "Now" },
  { id: "next", label: "Next" },
  { id: "library", label: "Library" },
  { id: "session", label: "Session" },
];

export function TopBar() {
  const state = useAbletonState();
  const view = useAppStore((s) => s.currentView);

  const play = useMutation({ mutationFn: api.abletonPlay });
  const stop = useMutation({ mutationFn: api.abletonStop });

  return (
    <header className="h-16 shrink-0 flex items-center gap-4 px-5 border-b border-neutral-800 bg-neutral-950">
      <div className="flex items-center gap-3">
        <div className="w-7 h-7 rounded bg-amber-400" aria-hidden />
        <div className="text-neutral-200 font-semibold tracking-wide uppercase text-xs">
          Dance
        </div>
      </div>

      <div className="flex items-baseline gap-2 ml-2">
        <span className="font-mono text-4xl text-neutral-50 tabular-nums">
          {state.tempo != null ? state.tempo.toFixed(1) : "--"}
        </span>
        <span className="text-neutral-500 text-xs uppercase tracking-wider">
          BPM
        </span>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => play.mutate()}
          className={`min-h-[44px] min-w-[64px] px-4 rounded-lg font-semibold ${
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
          className="min-h-[44px] min-w-[64px] px-4 rounded-lg bg-neutral-800 text-neutral-200 hover:bg-neutral-700 font-semibold"
          aria-label="Stop"
        >
          Stop
        </button>
      </div>

      <nav className="ml-auto flex items-center gap-1" role="tablist">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            role="tab"
            aria-selected={view === v.id}
            onClick={() => store.setView(v.id)}
            className={`min-h-[44px] px-4 rounded-lg font-semibold text-sm ${
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
