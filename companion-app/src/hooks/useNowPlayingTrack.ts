import { useMemo } from "react";
import { useAbletonState } from "./useAbletonState";
import { useCurrentSession } from "./useSession";
import { useAppStore } from "../store";

/**
 * Figure out which dance Track is *actually playing in Ableton right now* by
 * cross-referencing two sources:
 *
 *   1. ``loadedDecks`` — set by us when the user hits "Load to Live". Maps a
 *      scene_index in Live to one of our track ids, plus the set of ableton
 *      track indices that host its stems.
 *   2. ``AbletonState.playing_clips`` — pushed by the OSC bridge: ``{
 *      ableton_track_idx -> scene_idx }``. A clip is playing iff a track is
 *      firing one of its scenes.
 *
 * For each loaded deck, we say "this deck is playing" if any of its
 * stem_track_indices currently has a playing clip whose scene equals
 * deck.scene_index. Most recently-loaded deck wins ties.
 *
 * Falls back to the most recent SessionPlay's track_id when nothing in the
 * loaded-decks map matches (covers the "you started a clip Live-side without
 * going through Load to Live" case during early adoption).
 */
export function useNowPlayingTrack(): {
  trackId: number | null;
  source: "ableton" | "session" | "none";
} {
  const ableton = useAbletonState();
  const loadedDecks = useAppStore((s) => s.loadedDecks);
  const session = useCurrentSession();

  return useMemo(() => {
    // 1) Ableton-derived.
    const decks = Object.values(loadedDecks).sort(
      (a, b) => b.loaded_at - a.loaded_at,
    );
    const playing = ableton.playing_clips ?? {};
    for (const deck of decks) {
      for (const tIdx of deck.stem_track_indices) {
        const playingScene = playing[tIdx];
        if (playingScene != null && playingScene === deck.scene_index) {
          return { trackId: deck.track_id, source: "ableton" };
        }
      }
    }
    // 2) Fall back to session.
    const lastPlay = session.data?.plays.at(-1);
    if (lastPlay) return { trackId: lastPlay.track_id, source: "session" };
    return { trackId: null, source: "none" };
  }, [ableton.playing_clips, loadedDecks, session.data]);
}
