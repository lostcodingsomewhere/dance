import { useState } from "react";
import {
  copyToClipboard,
  exportAls,
  pushTrackToLive,
  revealPath,
} from "../api";
import { store } from "../store";

/**
 * "Get this into Ableton" affordances. AbletonOSC can prepare named/colored
 * empty audio tracks in Live, but it cannot load the sample file from disk
 * (Live's Python API doesn't expose that hook). So "Push to Live" creates
 * the slots, and Reveal opens Finder so the user can drag stems onto them
 * in one motion.
 */
export function LoadActions({
  path,
  trackId,
  label = "Track",
}: {
  path: string | null | undefined;
  trackId?: number;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pushing, setPushing] = useState(false);
  const [pushed, setPushed] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exported, setExported] = useState<string | null>(null);

  if (!path && trackId == null) return null;

  const copy = async () => {
    if (!path) return;
    try {
      setError(null);
      await copyToClipboard(path);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const reveal = async () => {
    if (!path) return;
    try {
      setError(null);
      await revealPath(path);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const doExport = async () => {
    if (trackId == null) return;
    try {
      setError(null);
      setExporting(true);
      const result = await exportAls(trackId);
      setExported("✓ Live Set exported");
      // Reveal in Finder so the user can double-click to open Live.
      try {
        await revealPath(result.out_path);
      } catch {
        // Best-effort — reveal failure shouldn't undo the export.
      }
      setTimeout(() => setExported(null), 2500);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setExporting(false);
    }
  };

  const push = async () => {
    if (trackId == null) return;
    try {
      setError(null);
      setPushing(true);
      const sceneIndex = store.peekNextScene();
      const result = await pushTrackToLive(trackId, {
        includeStems: true,
        sceneIndex,
      });
      // Track which scene this dance-track now occupies so the Booth can
      // auto-detect when its clips fire and stop double-loading the same
      // song into a new scene.
      store.registerDeck({
        track_id: trackId,
        scene_index: result.scene_index,
        stem_track_indices: Object.values(result.track_indices),
        loaded_at: Date.now(),
      });
      const fully = result.stems_loaded === 4;
      setPushed(
        fully ? `✓ scene ${result.scene_index + 1}` : `⚠ ${result.stems_loaded}/4`,
      );
      // Only fall back to Finder reveal if auto-load was incomplete — full
      // loads no longer need the user to drag anything.
      if (!fully && path) {
        try {
          await revealPath(path);
        } catch {
          // best-effort
        }
      }
      setTimeout(() => setPushed(null), 2500);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPushing(false);
    }
  };

  return (
    <div className="flex items-center gap-1">
      {trackId != null && (
        <button
          type="button"
          onClick={push}
          disabled={pushing}
          title="Create named audio tracks in Ableton Live and open Finder to drag the stems"
          className="h-10 px-3 rounded-md text-xs font-semibold bg-purple-700 hover:bg-purple-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-white active:scale-95"
        >
          {pushing ? "Pushing…" : pushed ?? "Push to Live"}
        </button>
      )}
      {trackId != null && (
        <button
          type="button"
          onClick={doExport}
          disabled={exporting}
          title="Generate an Ableton Live Set (.als) with the stems pre-loaded and reveal it in Finder"
          className="h-10 px-3 rounded-md text-xs font-semibold bg-emerald-700 hover:bg-emerald-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-white active:scale-95"
        >
          {exporting ? "Exporting…" : exported ?? "Export .als"}
        </button>
      )}
      {path && (
        <>
          <button
            type="button"
            onClick={copy}
            title={`Copy ${label.toLowerCase()} path`}
            className="h-10 px-3 rounded-md text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200 active:scale-95"
          >
            {copied ? "✓ Copied" : "Copy path"}
          </button>
          <button
            type="button"
            onClick={reveal}
            title={`Reveal ${label.toLowerCase()} in Finder`}
            className="h-10 px-3 rounded-md text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200 active:scale-95"
          >
            Reveal
          </button>
        </>
      )}
      {error && (
        <span className="text-xs text-red-400 ml-1" title={error}>!</span>
      )}
    </div>
  );
}
