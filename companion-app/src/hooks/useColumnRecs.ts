import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import * as api from "../api";
import { useAbletonState } from "./useAbletonState";
import { useDeckMap } from "./useDeckMap";

/**
 * Re-scoring per-column rec stream. The "active combo" is the bag of stem
 * files currently playing in Live, derived from:
 *
 *   AbletonState.playing_clips  : track_idx → scene_idx
 *   DeckMap.columns             : role → track_idx
 *   DeckMap.scenes              : list of (scene_idx, track_id) placements
 *
 * Combining the three tells us "which stem_file_id is firing in which column
 * right now?". We then ask the backend to rank candidates for the requested
 * column against that bag of active stems.
 *
 * NOTE: today the backend's deck map exposes track-level scene placements
 * (one ``DeckSceneOut`` per scene). We don't yet expose the per-cell
 * stem_file_id; for v1 the combo is the empty list (no embedding signal) and
 * the rec stream falls back to key + BPM scoring against master_bpm. When
 * the deck map is extended to carry per-cell stem_file_id (Phase 4+), this
 * hook upgrades to a real combo without any rec banner changes.
 */
export function useColumnRecs(column: string, opts: { k?: number } = {}) {
  const ableton = useAbletonState();
  const deckMap = useDeckMap();

  const combo = useMemo(() => {
    const columns = deckMap.data?.columns;
    const playing = ableton.playing_clips ?? {};
    if (!columns) return { stemIds: [] as number[], excludeTracks: [] as number[] };
    // For v1: no per-cell stem_file_id available. We still build an
    // exclude_track_ids list from currently-playing scenes so the banner
    // doesn't recommend a track that's already loaded.
    const excludeTracks: number[] = [];
    const scenes = deckMap.data?.scenes ?? [];
    const playingSceneIdxs = new Set<number>();
    for (const trackIdx of Object.values(columns)) {
      const s = playing[trackIdx];
      if (s != null) playingSceneIdxs.add(s);
    }
    for (const s of scenes) {
      if (playingSceneIdxs.has(s.scene_index)) {
        excludeTracks.push(s.track_id);
      }
    }
    return { stemIds: [], excludeTracks };
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
