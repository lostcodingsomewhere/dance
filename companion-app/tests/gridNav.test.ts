import { describe, expect, it } from "vitest";
import {
  firstFocus,
  focusedTrackId,
  moveFocus,
  reconcileFocus,
  type GridShape,
} from "../src/lib/gridNav";

/**
 * Navigation rules for the plan grid's audition loop.
 *
 * These are the motions the hands learn, and they are shared verbatim by the
 * Set page and the Booth — only the commit verb differs. So a surprise here
 * is not a cosmetic bug: it is a habit that will misfire mid-set.
 */

/** drums: 2 queued + 3 recs · bass: recs only · vocals: empty · song: plan only */
const SHAPE: GridShape = {
  drums: { plan: [10, 11], recs: [20, 21, 22] },
  bass: { plan: [], recs: [30, 31] },
  vocals: { plan: [], recs: [] },
  other: { plan: [], recs: [40] },
  song: { plan: [50, 51, 52], recs: [] },
};

describe("firstFocus", () => {
  it("starts on recs — the plan is decided, the recs are what you came to judge", () => {
    expect(firstFocus(SHAPE)).toEqual({ role: "drums", zone: "recs", index: 0 });
  });

  it("skips past columns with nothing in them", () => {
    expect(firstFocus({ drums: { plan: [], recs: [] }, bass: { plan: [], recs: [9] } }))
      .toEqual({ role: "bass", zone: "recs", index: 0 });
  });

  it("is null when the whole grid is empty", () => {
    expect(firstFocus({ drums: { plan: [], recs: [] } })).toBeNull();
  });
});

describe("vertical movement", () => {
  it("reads a column as ONE list — ↓ off the plan continues into the recs", () => {
    // Last plan pick…
    const atPlanEnd = { role: "drums", zone: "plan", index: 1 } as const;
    // …↓ crosses the seam rather than dead-ending on it.
    expect(moveFocus(SHAPE, atPlanEnd, "ArrowDown")).toEqual({
      role: "drums",
      zone: "recs",
      index: 0,
    });
  });

  it("↑ crosses back from the first rec to the last plan pick", () => {
    expect(
      moveFocus(SHAPE, { role: "drums", zone: "recs", index: 0 }, "ArrowUp"),
    ).toEqual({ role: "drums", zone: "plan", index: 1 });
  });

  it("stops at the ends instead of wrapping", () => {
    const top = { role: "drums", zone: "plan", index: 0 } as const;
    expect(moveFocus(SHAPE, top, "ArrowUp")).toEqual(top);
    const bottom = { role: "drums", zone: "recs", index: 2 } as const;
    expect(moveFocus(SHAPE, bottom, "ArrowDown")).toEqual(bottom);
  });
});

describe("horizontal movement", () => {
  it("stays in the same zone — browsing recs must not drop into a plan", () => {
    // drums recs → bass. bass has 0 plan picks and 2 recs; a flattened-offset
    // scheme would land in the plan of a column that had one.
    expect(
      moveFocus(SHAPE, { role: "drums", zone: "recs", index: 1 }, "ArrowRight"),
    ).toEqual({ role: "bass", zone: "recs", index: 1 });
  });

  it("clamps the index into the shorter column", () => {
    expect(
      moveFocus(SHAPE, { role: "drums", zone: "recs", index: 2 }, "ArrowRight"),
    ).toEqual({ role: "bass", zone: "recs", index: 1 });
  });

  it("skips empty columns entirely", () => {
    // bass → (vocals is empty) → other
    expect(
      moveFocus(SHAPE, { role: "bass", zone: "recs", index: 0 }, "ArrowRight"),
    ).toEqual({ role: "other", zone: "recs", index: 0 });
  });

  it("falls to the column's other zone when the current one is empty", () => {
    // other(recs) → song, which has plan picks but no recs.
    expect(
      moveFocus(SHAPE, { role: "other", zone: "recs", index: 0 }, "ArrowRight"),
    ).toEqual({ role: "song", zone: "plan", index: 0 });
  });

  it("holds position at the last column with content", () => {
    const last = { role: "song", zone: "plan", index: 0 } as const;
    expect(moveFocus(SHAPE, last, "ArrowRight")).toEqual(last);
  });
});

describe("focus recovery", () => {
  it("adopts a first focus when there is none", () => {
    expect(moveFocus(SHAPE, null, "ArrowDown")).toEqual({
      role: "drums",
      zone: "recs",
      index: 0,
    });
  });

  it("restarts rather than trapping focus in a column that emptied", () => {
    const stranded = { role: "vocals", zone: "recs", index: 0 } as const;
    expect(moveFocus(SHAPE, stranded, "ArrowDown")).toEqual({
      role: "drums",
      zone: "recs",
      index: 0,
    });
  });
});

describe("reconcileFocus", () => {
  it("holds position so the NEXT candidate slides under the cursor", () => {
    // The point of the whole loop: commit a rec, the list shortens, and the
    // next one is already focused — triage continues without an arrow key.
    const focus = { role: "drums", zone: "recs", index: 1 } as const;
    const after: GridShape = { ...SHAPE, drums: { plan: [10, 11], recs: [20, 22] } };
    expect(reconcileFocus(after, focus)).toEqual(focus);
    expect(focusedTrackId(after, reconcileFocus(after, focus))).toBe(22);
  });

  it("steps back when the list ran out under the cursor", () => {
    const focus = { role: "bass", zone: "recs", index: 1 } as const;
    const after: GridShape = { ...SHAPE, bass: { plan: [], recs: [30] } };
    expect(reconcileFocus(after, focus)).toEqual({
      role: "bass",
      zone: "recs",
      index: 0,
    });
  });

  it("falls to the other zone when its own empties", () => {
    const focus = { role: "bass", zone: "recs", index: 0 } as const;
    const after: GridShape = { ...SHAPE, bass: { plan: [99], recs: [] } };
    expect(reconcileFocus(after, focus)).toEqual({
      role: "bass",
      zone: "plan",
      index: 0,
    });
  });

  it("leaves the column when it empties completely", () => {
    const focus = { role: "bass", zone: "recs", index: 0 } as const;
    const after: GridShape = { ...SHAPE, bass: { plan: [], recs: [] } };
    expect(reconcileFocus(after, focus)).toEqual({
      role: "drums",
      zone: "recs",
      index: 0,
    });
  });

  it("is a no-op on no focus", () => {
    expect(reconcileFocus(SHAPE, null)).toBeNull();
  });
});

describe("focusedTrackId", () => {
  it("resolves the id under the cursor", () => {
    expect(
      focusedTrackId(SHAPE, { role: "drums", zone: "recs", index: 2 }),
    ).toBe(22);
  });

  it("is null for a focus pointing past the end", () => {
    expect(
      focusedTrackId(SHAPE, { role: "bass", zone: "recs", index: 9 }),
    ).toBeNull();
  });
});
