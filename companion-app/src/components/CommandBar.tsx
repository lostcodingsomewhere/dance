import { useMutation } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { pushTrackToLive, recommendByText, revealPath } from "../api";
import { store, useAppStore } from "../store";
import type { Recommendation } from "../types";
import { EnergyBar } from "./EnergyBar";
import { KeyBadge } from "./KeyBadge";

/**
 * Global vibe-search overlay. Cmd/Ctrl-K from anywhere; ESC to close. Each
 * result has the same one-tap "Load to Live" the Up Next rail has.
 */
export function CommandBar() {
  const open = useAppStore((s) => s.commandBarOpen);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [q, setQ] = useState("");

  const vibe = useMutation<Recommendation[], Error, string>({
    mutationFn: (s: string) => recommendByText(s, 12),
  });

  // Global hotkey listener — Cmd/Ctrl+K toggles.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (open) store.closeCommandBar();
        else store.openCommandBar();
      }
      if (open && e.key === "Escape") {
        store.closeCommandBar();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Focus on open / reset on close. We intentionally exclude `vibe` from deps
  // because calling vibe.reset() changes its identity, which would re-fire the
  // effect and infinite-loop. vibe.reset() is safe to call when there's
  // nothing to reset, so a stale closure here is harmless.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!open) {
      setQ("");
      vibe.reset();
      return;
    }
    const t = setTimeout(() => inputRef.current?.focus(), 30);
    return () => clearTimeout(t);
  }, [open]);

  if (!open) return null;

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const term = q.trim();
    if (term) vibe.mutate(term);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-24 px-4 bg-black/70 backdrop-blur-sm"
      onClick={() => store.closeCommandBar()}
      role="presentation"
    >
      <div
        className="w-full max-w-2xl rounded-xl border border-neutral-800 bg-neutral-950 shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Vibe search"
      >
        <form onSubmit={submit} className="flex items-center gap-2 px-4 py-3 border-b border-neutral-800">
          <span className="text-purple-400 text-xl" aria-hidden>✦</span>
          <input
            ref={inputRef}
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="What vibe? 'punchy techy with vocals', 'deep rolling bassline'…"
            className="flex-1 bg-transparent text-lg text-neutral-100 placeholder:text-neutral-600 outline-none"
          />
          <button
            type="submit"
            disabled={!q.trim() || vibe.isPending}
            className="h-9 px-3 rounded-md bg-purple-700 hover:bg-purple-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-white text-sm font-semibold"
          >
            {vibe.isPending ? "…" : "Search"}
          </button>
          <kbd className="font-mono text-[10px] text-neutral-500 border border-neutral-800 rounded px-1">
            ESC
          </kbd>
        </form>

        <div className="max-h-[60vh] overflow-y-auto p-2">
          {vibe.isError && (
            <div className="m-2 text-sm text-rose-300">
              {vibe.error.message}
            </div>
          )}
          {vibe.data?.length === 0 && (
            <div className="m-3 text-sm text-neutral-500">No matches.</div>
          )}
          {!vibe.data && !vibe.isPending && (
            <div className="m-3 text-sm text-neutral-500">
              Type a phrase and hit enter. Results are ranked by CLAP audio embedding
              similarity.
            </div>
          )}
          {vibe.data?.map((r) => (
            <ResultRow key={r.track_id} rec={r} />
          ))}
        </div>
      </div>
    </div>
  );
}

function ResultRow({ rec }: { rec: Recommendation }) {
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState<string | null>(null);

  async function loadToLive() {
    try {
      setLoading(true);
      const sceneIndex = store.peekNextScene();
      const r = await pushTrackToLive(rec.track_id, {
        includeStems: true,
        sceneIndex,
      });
      store.registerDeck({
        track_id: rec.track_id,
        scene_index: r.scene_index,
        stem_track_indices: Object.values(r.track_indices),
        loaded_at: Date.now(),
      });
      const fully = r.stems_loaded === 4;
      setDone(fully ? `scene ${r.scene_index + 1}` : `${r.stems_loaded}/4`);
      if (!fully && rec.file_path) {
        try {
          await revealPath(rec.file_path);
        } catch {
          // best-effort
        }
      }
      setTimeout(() => {
        store.closeCommandBar();
      }, 500);
    } catch (e) {
      setDone(`! ${(e as Error).message.slice(0, 40)}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex items-center gap-2 p-2 rounded-md hover:bg-neutral-900">
      <KeyBadge keyCamelot={rec.key_camelot ?? null} size="sm" />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-semibold text-neutral-100 truncate">
          {rec.title ?? `Track #${rec.track_id}`}
        </div>
        <div className="text-xs text-neutral-500 truncate">
          {rec.artist ?? "Unknown"}
          {rec.bpm != null && (
            <span className="ml-2 font-mono">{rec.bpm.toFixed(1)} BPM</span>
          )}
        </div>
      </div>
      <EnergyBar energy={rec.floor_energy ?? null} size="sm" />
      <span className="font-mono text-[10px] text-purple-300 w-12 text-right">
        ✦ {rec.score.toFixed(2)}
      </span>
      <button
        type="button"
        onClick={loadToLive}
        disabled={loading}
        className="min-h-[36px] px-3 rounded-md text-xs font-semibold bg-purple-700 hover:bg-purple-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-white"
      >
        {loading ? "…" : done ?? "Load"}
      </button>
      <button
        type="button"
        onClick={() => store.addToStack(rec.track_id)}
        className="min-h-[36px] px-2.5 rounded-md text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200"
      >
        + Stack
      </button>
    </div>
  );
}
