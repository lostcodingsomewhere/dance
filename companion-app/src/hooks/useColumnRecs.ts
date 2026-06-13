import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import * as api from "../api";
import { useAbletonState } from "./useAbletonState";
import { useDeckMap } from "./useDeckMap";

/**
 * Re-scoring per-column rec stream. The "active combo" is the bag of cells
 * currently firing in Live, derived from:
 *
 *   AbletonState.playing_clips  : track_idx → scene_idx
 *   DeckMap.columns             : role → track_idx
 *   DeckMap.cells               : list of (scene_idx, kind, track_id)
 *
 * Cross-referencing these tells us "which source-track is producing each
 * playing role right now?". The result becomes ``exclude_track_ids`` so the
 * banner doesn't recommend a track that's already in the active combo.
 *
 * The backend's ColumnRecRequest takes ``combo_stem_ids`` (stem-file ids)
 * for embedding-based combo scoring. DeckCell now carries ``stem_file_id``
 * (types.ts), so we harvest the ids from the cells that are actually
 * FIRING in Live right now and feed them as the active combo. With real
 * stem ids flowing the backend's combo-embedding scoring fires; an empty
 * combo (nothing playing) still falls back to key + BPM against master
 * tempo. No banner changes are needed — the cards just re-score.
 */
export function useColumnRecs(column: string, opts: { k?: number } = {}) {
  const ableton = useAbletonState();
  const deckMap = useDeckMap();

  const combo = useMemo(() => {
    const columns = deckMap.data?.columns;
    const playing = ableton.playing_clips ?? {};
    if (!columns) return { stemIds: [] as number[], excludeTracks: [] as number[] };
    const cells = deckMap.data?.cells ?? [];
    const playingSceneIdxs = new Set<number>();
    for (const trackIdx of Object.values(columns)) {
      const s = playing[trackIdx];
      if (s != null) playingSceneIdxs.add(s);
    }
    const excludeTracks: number[] = [];
    const stemIds: number[] = [];
    for (const c of cells) {
      if (playingSceneIdxs.has(c.scene_index)) {
        excludeTracks.push(c.track_id);
        // Feed the embedding signal: collect stem-file ids of the cells
        // currently firing. Mix cells / not-yet-analyzed stems carry a
        // null stem_file_id — skip those.
        if (c.stem_file_id != null) stemIds.push(c.stem_file_id);
      }
    }
    return { stemIds, excludeTracks };
  }, [deckMap.data, ableton.playing_clips]);

  return useQuery({
    // Cache key includes the combo so it re-runs on combo change.
    queryKey: [
      "recommend",
      "by-column",
      column,
      combo.stemIds.slice().sort(),
      combo.excludeTracks.slice().sort(),
      Math.round(ableton.tempo ?? 0),
      opts.k ?? 5,
    ],
    queryFn: () =>
      api.recommendByColumn({
        column,
        combo_stem_ids: combo.stemIds,
        master_bpm: ableton.tempo ?? null,
        k: opts.k ?? 5,
        exclude_track_ids: combo.excludeTracks,
      }),
    // Refetch on combo change; otherwise stale-cache for 30s.
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
}
