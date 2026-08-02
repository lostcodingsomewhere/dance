import { beforeEach, describe, expect, it } from "vitest";
import { deckKey, nextSceneIndex, store } from "../src/store";
import { sideTrackIndices } from "../src/lib/roles";
import type { LoadedDeck } from "../src/types";

/**
 * Play logging depends on a chain that had a hole in it:
 *
 *   useAutoLog -> useNowPlayingTrack -> store.loadedDecks
 *
 * `useNowPlayingTrack` only recognises a playing clip if its deck is in
 * `loadedDecks`, and `registerDeck` was called from CueStrip, LoadActions and
 * CommandBar — but NOT from RoleColumn, where ⤒A/⤒B lives. ⤒A/⤒B is the
 * Booth's primary load button, so nothing fired from the plan grid could ever
 * be logged. The whole project has 6 logged plays; they came from ⌘K.
 */

const TRACK_INDICES = {
  drums_a: 0, drums_b: 1, bass_a: 2, bass_b: 3,
  vocals_a: 4, vocals_b: 5, other_a: 6, other_b: 7,
  mix_a: 8, mix_b: 9,
};

function deck(over: Partial<LoadedDeck> = {}): LoadedDeck {
  return {
    track_id: 1,
    scene_index: 0,
    side: "a",
    stem_track_indices: sideTrackIndices(TRACK_INDICES, "a"),
    loaded_at: 1,
    ...over,
  };
}

function decks(): Record<string, LoadedDeck> {
  const raw = window.localStorage.getItem("dance.companion.state.v2");
  return (raw ? JSON.parse(raw).loadedDecks : null) ?? {};
}

beforeEach(() => {
  window.localStorage.clear();
  store.clearDecks();
});

describe("side-scoped track indices", () => {
  it("includes only the loaded side", () => {
    expect(sideTrackIndices(TRACK_INDICES, "a").sort((x, y) => x - y)).toEqual([0, 2, 4, 6, 8]);
    expect(sideTrackIndices(TRACK_INDICES, "b").sort((x, y) => x - y)).toEqual([1, 3, 5, 7, 9]);
  });

  it("falls back to every index when the side is unknown", () => {
    // No worse than the old behaviour — never fewer indices than before.
    expect(sideTrackIndices(TRACK_INDICES, null)).toHaveLength(10);
  });

  it("keeps the two decks disjoint, so B's clip can't log A's track", () => {
    const a = new Set(sideTrackIndices(TRACK_INDICES, "a"));
    const b = sideTrackIndices(TRACK_INDICES, "b");
    expect(b.some((i) => a.has(i))).toBe(false);
  });
});

describe("deck registration", () => {
  it("keys by scene AND side, so a B load does not overwrite the A load", () => {
    store.registerDeck(deck({ track_id: 11, scene_index: 3, side: "a" }));
    store.registerDeck(deck({ track_id: 22, scene_index: 3, side: "b" }));
    const d = decks();
    expect(Object.keys(d).sort()).toEqual(["3:a", "3:b"]);
    expect(d[deckKey(3, "a")].track_id).toBe(11);
    expect(d[deckKey(3, "b")].track_id).toBe(22);
  });

  it("reloading the same side replaces it rather than accumulating", () => {
    store.registerDeck(deck({ track_id: 11, scene_index: 3, side: "a" }));
    store.registerDeck(deck({ track_id: 33, scene_index: 3, side: "a" }));
    expect(Object.keys(decks())).toEqual(["3:a"]);
    expect(decks()[deckKey(3, "a")].track_id).toBe(33);
  });

  it("unloads one side without disturbing the other", () => {
    store.registerDeck(deck({ track_id: 11, scene_index: 3, side: "a" }));
    store.registerDeck(deck({ track_id: 22, scene_index: 3, side: "b" }));
    store.unloadDeck(3, "a");
    expect(Object.keys(decks())).toEqual(["3:b"]);
  });

  it("unloads a whole scene when no side is given", () => {
    store.registerDeck(deck({ scene_index: 3, side: "a" }));
    store.registerDeck(deck({ scene_index: 3, side: "b" }));
    store.unloadDeck(3);
    expect(Object.keys(decks())).toEqual([]);
  });

  it("nextSceneIndex reads scene numbers off the values, not the keys", () => {
    store.registerDeck(deck({ scene_index: 4, side: "b" }));
    expect(nextSceneIndex(decks())).toBe(5);
  });
});
