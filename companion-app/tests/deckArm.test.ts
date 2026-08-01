import { beforeEach, describe, expect, it } from "vitest";
import { store } from "../src/store";

/**
 * The deck-arm regression.
 *
 * Before arming existed, a deck's ▶ fired an "anchor scene" heuristic that
 * resolves to either the scene ALREADY firing on that side or, failing that,
 * the LOWEST loaded scene. Both mean the same thing once you've loaded a
 * second track: ▶ replays the track you're already on, and the one you just
 * loaded is unreachable except by tapping its four stem cells one at a time.
 *
 * Verified against the real rig before the fix — with track 1 on scene 1 and
 * track 5 on scene 2 of Deck A, the play button read "fire scene 1".
 *
 * The rule these tests lock in: **the armed scene wins over the anchor**, and
 * the arm clears once Live confirms it fired.
 */

/** The resolution TwoDeckStrip's DeckHeader performs. Kept in sync with it. */
function fireTarget(
  armedSceneIdx: number | undefined,
  anchorSceneIdx: number | undefined,
): number | undefined {
  return armedSceneIdx ?? anchorSceneIdx;
}

/** Read the arm back. The store persists on every mutation, so localStorage
 *  is an honest observation point and doubles as the reload-safety check. */
function armed(): { a: number | null; b: number | null } {
  const raw = window.localStorage.getItem("dance.companion.state.v2");
  return (raw ? JSON.parse(raw).armed : null) ?? { a: null, b: null };
}

beforeEach(() => {
  window.localStorage.clear();
  store.clearArm("a");
  store.clearArm("b");
});

describe("deck arm", () => {
  it("armed scene wins over the anchor heuristic", () => {
    // Anchor says scene 0 (the oldest load). We armed scene 2.
    expect(fireTarget(2, 0)).toBe(2);
  });

  it("falls back to the anchor when nothing is armed", () => {
    expect(fireTarget(undefined, 0)).toBe(0);
  });

  it("stays undefined when the deck is empty", () => {
    expect(fireTarget(undefined, undefined)).toBeUndefined();
  });

  it("scene 0 arms correctly and is not treated as absent", () => {
    // Guards the ?? vs || trap: scene 0 is falsy but a real target.
    expect(fireTarget(0, 3)).toBe(0);
  });

  it("armDeck records the scene for that side only", () => {
    store.armDeck("a", 4);
    expect(armed().a).toBe(4);
    expect(armed().b).toBeNull();
  });

  it("armDeck ignores a null/undefined side rather than guessing", () => {
    // The backend reports which side it auto-picked; if it couldn't, we must
    // not arm the wrong deck — the old anchor behaviour is the safe fallback.
    store.armDeck(null, 5);
    store.armDeck(undefined, 6);
    expect(armed().a).toBeNull();
    expect(armed().b).toBeNull();
  });

  it("clearArm drops only the side that fired", () => {
    store.armDeck("a", 1);
    store.armDeck("b", 2);
    store.clearArm("a");
    expect(armed().a).toBeNull();
    expect(armed().b).toBe(2);
  });

  it("persists the arm so a mid-set reload does not silently re-break ▶", () => {
    store.armDeck("b", 7);
    const raw = window.localStorage.getItem("dance.companion.state.v2");
    expect(raw).toBeTruthy();
    expect(JSON.parse(raw as string).armed).toEqual({ a: null, b: 7 });
  });
});
