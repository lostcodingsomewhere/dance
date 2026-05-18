import { useMemo, useState } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useCleanDecks, useDeckMap, useResetDecks } from "../hooks/useDeckMap";
import { useFireScene, useStopAllClips } from "../hooks/useTransport";
import { useAppStore } from "../store";
import type { DeckScene } from "../types";
import { KeyBadge } from "./KeyBadge";

/**
 * Vertical strip showing which dance Tracks are staged on which scenes of
 * Live's session view. The "▶" indicator on a row means at least one of
 * that scene's clips is currently playing (cross-referenced from
 * AbletonState.playing_clips and the deck-column track indices).
 *
 * Source of truth is the backend (``GET /ableton/decks``) — the FE's
 * localStorage ``loadedDecks`` is just a hint used by the next-scene
 * allocator before the first deck fetch lands.
 */
export function SceneMap() {
  const deckMap = useDeckMap();
  const ableton = useAbletonState();
  const reset = useResetDecks();
  const clean = useCleanDecks();
  const fireScene = useFireScene();
  const stopAll = useStopAllClips();
  const localDecks = useAppStore((s) => s.loadedDecks);
  const [cleanResult, setCleanResult] = useState<string | null>(null);

  async function onClean() {
    setCleanResult(null);
    if (
      !window.confirm(
        "Delete every \"Deck *\" track in Live and reset bridge + FE state? " +
          "Your other tracks are untouched.",
      )
    ) {
      return;
    }
    const r = await clean.mutateAsync();
    setCleanResult(
      r.warning
        ? `⚠ ${r.warning}`
        : `Deleted ${r.deleted} deck track${r.deleted === 1 ? "" : "s"} from Live`,
    );
    setTimeout(() => setCleanResult(null), 4000);
  }

  // The bridge's deck columns are an ordered set of ableton track indices
  // (mix/drums/bass/vocals/other). A scene is "playing" when ANY of those
  // tracks shows a playing clip whose scene index matches the row.
  const columns = deckMap.data?.columns ?? null;
  const playingClips = ableton.playing_clips ?? {};
  const playingScenes = useMemo(() => {
    if (!columns) return new Set<number>();
    const set = new Set<number>();
    for (const trackIdx of Object.values(columns)) {
      const scene = playingClips[trackIdx];
      if (scene != null) set.add(scene);
    }
    return set;
  }, [columns, playingClips]);

  const scenes: DeckScene[] = deckMap.data?.scenes ?? [];

  if (scenes.length === 0) {
    return (
      <div className="px-3 pt-3 pb-2 text-xs text-neutral-600">
        <div className="italic">No scenes loaded yet. Hit Load on a rec to stage it.</div>
        {Object.keys(localDecks).length > 0 && (
          <div className="mt-1 text-neutral-700">
            ({Object.keys(localDecks).length} stale locally — backend doesn't see them)
          </div>
        )}
        <button
          type="button"
          onClick={onClean}
          disabled={clean.isPending}
          className="mt-2 text-[10px] text-neutral-600 hover:text-rose-300"
          title="Delete every Deck * track in Live via OSC, then reset bridge + FE state"
        >
          {clean.isPending ? "cleaning…" : "clean Live decks ⌫"}
        </button>
        {cleanResult && (
          <div className="mt-1 text-[10px] text-emerald-400/80">{cleanResult}</div>
        )}
      </div>
    );
  }

  return (
    <div className="px-3 pt-3 pb-1">
      <div className="flex items-baseline justify-between mb-1">
        <div className="text-[10px] uppercase tracking-widest text-neutral-500">
          Loaded scenes
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => stopAll.mutate()}
            disabled={stopAll.isPending}
            className="text-[10px] text-neutral-600 hover:text-amber-300"
            title="Stop every playing clip (combo panic — transport keeps running)"
          >
            stop all
          </button>
          <button
            type="button"
            onClick={() => reset.mutate()}
            disabled={reset.isPending}
            className="text-[10px] text-neutral-600 hover:text-neutral-300"
            title="Forget all scene placements (does not delete tracks in Live)"
          >
            reset
          </button>
          <button
            type="button"
            onClick={onClean}
            disabled={clean.isPending}
            className="text-[10px] text-neutral-600 hover:text-rose-300"
            title="Delete every Deck * track in Live via OSC, then reset bridge + FE state"
          >
            {clean.isPending ? "cleaning…" : "clean ⌫"}
          </button>
        </div>
      </div>
      {cleanResult && (
        <div className="text-[10px] text-emerald-400/80 mb-1">{cleanResult}</div>
      )}
      <ol className="flex flex-col gap-1">
        {scenes.map((s) => (
          <SceneRow
            key={s.scene_index}
            scene={s}
            isPlaying={playingScenes.has(s.scene_index)}
            onFire={() => fireScene.mutate(s.scene_index)}
            firePending={fireScene.isPending}
          />
        ))}
      </ol>
    </div>
  );
}

function SceneRow({
  scene,
  isPlaying,
  onFire,
  firePending,
}: {
  scene: DeckScene;
  isPlaying: boolean;
  onFire: () => void;
  firePending: boolean;
}) {
  return (
    <li
      className={`flex items-center gap-2 px-2 py-1.5 rounded-md border transition-colors duration-100 ease-out group ${
        isPlaying
          ? "border-emerald-500/40 bg-emerald-500/10"
          : "border-neutral-800/60 bg-neutral-900/40 hover:border-neutral-700 hover:bg-neutral-900/70"
      }`}
    >
      <span
        className={`font-mono text-[11px] w-5 text-right shrink-0 ${
          isPlaying ? "text-emerald-300" : "text-neutral-500"
        }`}
      >
        {scene.scene_index + 1}
      </span>
      <button
        type="button"
        onClick={onFire}
        disabled={firePending}
        title={
          isPlaying
            ? `Re-fire scene ${scene.scene_index + 1} (anchor mode)`
            : `Fire scene ${scene.scene_index + 1} — play the full original combo`
        }
        className={`text-xs w-4 shrink-0 cursor-pointer transition-colors ${
          isPlaying
            ? "text-emerald-300 hover:text-emerald-200"
            : "text-neutral-700 hover:text-emerald-300"
        }`}
      >
        ▶
      </button>
      <KeyBadge keyCamelot={scene.key_camelot} size="sm" />
      <div className="flex-1 min-w-0">
        <div className="text-sm text-neutral-100 truncate">
          {scene.title ?? `Track #${scene.track_id}`}
        </div>
        <div className="text-[11px] text-neutral-500 truncate">
          {scene.artist ?? "—"}
          {scene.bpm != null && (
            <span className="font-mono ml-1.5">
              · {scene.bpm.toFixed(1)} BPM
            </span>
          )}
        </div>
      </div>
    </li>
  );
}
