import { useState } from "react";
import { exportAls, revealPath } from "../api";
import { useTrack } from "../hooks/useTracks";
import { store, useAppStore } from "../store";
import { KeyBadge } from "./KeyBadge";

/**
 * The Stack — an ordered list of tracks the user is staging for an upcoming
 * set. Lives in localStorage. "Export all" runs the single-track .als exporter
 * over every entry and reveals the first one in Finder. A real batch export
 * (one .als with every track on its own scene) would need a backend
 * endpoint — skipped for now.
 */
export function Stack() {
  const stack = useAppStore((s) => s.stack);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function exportAll() {
    if (stack.length === 0) return;
    setRunning(true);
    setError(null);
    setProgress(`0 / ${stack.length}`);
    let firstPath: string | null = null;
    let failed = 0;
    for (let i = 0; i < stack.length; i++) {
      try {
        const r = await exportAls(stack[i]);
        if (i === 0) firstPath = r.out_path;
      } catch (e) {
        failed += 1;
        setError(`Track #${stack[i]}: ${(e as Error).message}`);
      }
      setProgress(`${i + 1} / ${stack.length}${failed > 0 ? ` · ${failed} failed` : ""}`);
    }
    if (firstPath) {
      try {
        await revealPath(firstPath);
      } catch {
        // best-effort reveal
      }
    }
    setRunning(false);
    setTimeout(() => setProgress(null), 4000);
  }

  return (
    <div className="bg-neutral-900/60 border border-neutral-800 rounded-lg p-3 flex flex-col gap-2 min-h-[160px]">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-xs uppercase tracking-widest text-neutral-500">
            Stack
          </div>
          <div className="text-sm text-neutral-300">
            {stack.length} {stack.length === 1 ? "track" : "tracks"} staged
          </div>
        </div>
        <div className="flex items-center gap-2">
          {progress && (
            <span className="text-xs text-neutral-400 font-mono">{progress}</span>
          )}
          <button
            type="button"
            onClick={exportAll}
            disabled={running || stack.length === 0}
            className="px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:bg-neutral-800 disabled:text-neutral-500 text-white text-xs font-semibold"
            title="Export every stacked track to its own .als and reveal the first in Finder"
          >
            {running ? "Exporting…" : "Export all .als"}
          </button>
          <button
            type="button"
            onClick={() => store.clearStack()}
            disabled={running || stack.length === 0}
            className="px-2 py-1.5 rounded-md bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 text-neutral-200 text-xs"
          >
            Clear
          </button>
        </div>
      </div>

      {error && (
        <div className="text-[11px] text-rose-300" title={error}>
          {error.slice(0, 120)}
        </div>
      )}

      {stack.length === 0 ? (
        <div className="text-xs text-neutral-600 italic">
          Use “+ Stack” on any track to stage it for an upcoming set.
        </div>
      ) : (
        <ol className="flex flex-col gap-1">
          {stack.map((id, idx) => (
            <StackRow key={id} id={id} index={idx} total={stack.length} />
          ))}
        </ol>
      )}
    </div>
  );
}

function StackRow({
  id,
  index,
  total,
}: {
  id: number;
  index: number;
  total: number;
}) {
  const track = useTrack(id);
  const t = track.data;
  const analysis = t?.analysis;
  return (
    <li className="flex items-center gap-2 px-2 py-1 rounded bg-neutral-900 border border-neutral-800/60">
      <span className="text-[11px] font-mono text-neutral-500 w-5 text-right">
        {index + 1}
      </span>
      <KeyBadge keyCamelot={analysis?.key_camelot ?? null} size="sm" />
      <div className="flex-1 min-w-0">
        <div className="text-sm text-neutral-100 truncate">
          {t?.title ?? `Track #${id}`}
        </div>
        <div className="text-[11px] text-neutral-500 truncate">
          {t?.artist ?? "—"}
          {analysis?.bpm != null && (
            <span className="ml-2 font-mono">{analysis.bpm.toFixed(1)} BPM</span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-0.5">
        <button
          type="button"
          onClick={() => store.moveInStack(index, index - 1)}
          disabled={index === 0}
          className="px-1.5 text-neutral-500 hover:text-neutral-200 disabled:opacity-20"
          title="Move up"
        >
          ↑
        </button>
        <button
          type="button"
          onClick={() => store.moveInStack(index, index + 1)}
          disabled={index === total - 1}
          className="px-1.5 text-neutral-500 hover:text-neutral-200 disabled:opacity-20"
          title="Move down"
        >
          ↓
        </button>
        <button
          type="button"
          onClick={() => store.removeFromStack(id)}
          className="px-1.5 text-neutral-500 hover:text-rose-400"
          title="Remove"
        >
          ×
        </button>
      </div>
    </li>
  );
}
