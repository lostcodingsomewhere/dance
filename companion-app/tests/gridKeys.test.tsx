import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RoleColumnsGrid } from "../src/components/RoleColumnsGrid";
import { store } from "../src/store";

/**
 * The audition loop, driven end-to-end through real key events.
 *
 * The pure navigation rules are covered in gridNav.test.ts. What THESE tests
 * cover is the wiring the pure tests cannot see: that the columns publish
 * their shape, that keys reach the handler, that the commit verb differs by
 * mode, and that typing somewhere else doesn't move the cursor.
 */

const state = {
  plan: { set_id: 1, queues: {} as Record<string, unknown[]> },
  planRecs: [] as unknown[],
  liveRecs: [] as unknown[],
  hasContext: true,
  previewing: null as { trackId: number; column: string } | null,
};

const addToRole = vi.fn();
const removeFromRole = vi.fn();
const startPreview = vi.fn();
const stopPreview = vi.fn();
const loadToDeck = vi.fn();

vi.mock("../src/hooks/useSetPlan", () => ({
  useSetPlan: () => ({ data: state.plan }),
  usePlanMutations: () => ({ addToRole, removeFromRole, put: { isPending: false } }),
  usePlanRecs: () => ({ data: { recs: state.planRecs }, isLoading: false }),
  useAppendToPlan: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("../src/hooks/useColumnRecs", () => ({
  useColumnRecs: () => ({
    data: { recs: state.liveRecs },
    isLoading: false,
    hasContext: state.hasContext,
  }),
}));

vi.mock("../src/hooks/usePreview", () => ({
  useStartPreview: () => ({ mutate: startPreview, isPending: false }),
  useStopPreview: () => ({ mutate: stopPreview, isPending: false }),
  usePreviewState: () => state.previewing,
}));

vi.mock("../src/hooks/useLoadToDeck", () => ({
  useLoadToDeck: () => ({ mutate: loadToDeck, isPending: false }),
}));

function rec(over: Record<string, unknown>) {
  return {
    track_id: 1,
    stem_file_id: null,
    track_title: "Rec Track",
    track_artist: "Rec Artist",
    bpm: 128,
    key_camelot: "8A",
    floor_energy: 6,
    score: 0.82,
    score_breakdown: {},
    reasons: [],
    ...over,
  };
}

function renderGrid(mode: "plan" | "live", setId: number | null = 1) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <RoleColumnsGrid setId={setId} mode={mode} />
    </QueryClientProvider>,
  );
}

/** The focus ring is the only user-visible signal of where the cursor is. */
function focusedTitle(): string | null {
  const el = document.querySelector(".ring-amber-300\\/80");
  return el?.querySelector("span.truncate")?.textContent ?? null;
}

function press(key: string, init: KeyboardEventInit = {}) {
  act(() => {
    fireEvent.keyDown(window, { key, ...init });
  });
}

beforeEach(() => {
  state.plan = { set_id: 1, queues: {} };
  state.planRecs = [];
  state.liveRecs = [];
  state.hasContext = true;
  state.previewing = null;
  addToRole.mockReset();
  removeFromRole.mockReset();
  startPreview.mockReset();
  stopPreview.mockReset();
  loadToDeck.mockReset();
  store.setGridFocus(null);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("audition loop", () => {
  it("adopts a focus on the first arrow key, with no click needed", async () => {
    state.planRecs = [rec({ track_id: 1, track_title: "First" })];
    renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("First").length).toBeGreaterThan(0));

    expect(focusedTitle()).toBeNull(); // nothing focused yet
    press("ArrowDown");
    // Requiring a click first would reintroduce exactly the mouse trip this
    // whole loop exists to remove.
    await waitFor(() => expect(focusedTitle()).toBe("First"));
  });

  it("space auditions the focused card in headphones", async () => {
    state.planRecs = [rec({ track_id: 42, track_title: "Hear Me" })];
    renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("Hear Me").length).toBeGreaterThan(0));

    press("ArrowDown");
    press(" ");
    expect(startPreview).toHaveBeenCalledWith({ trackId: 42, column: "drums" });
  });

  it("space stops the audition already playing that card", async () => {
    state.planRecs = [rec({ track_id: 42, track_title: "Hear Me" })];
    state.previewing = { trackId: 42, column: "drums" };
    renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("Hear Me").length).toBeGreaterThan(0));

    press("ArrowDown");
    press(" ");
    expect(stopPreview).toHaveBeenCalled();
    expect(startPreview).not.toHaveBeenCalled();
  });

  it("does NOT audition merely by moving", async () => {
    // Every preview creates and fires a real clip in Live; auto-audition on
    // move would machine-gun the Cue track while arrowing through a list.
    state.planRecs = [
      rec({ track_id: 1, track_title: "A" }),
      rec({ track_id: 2, track_title: "B" }),
    ];
    renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("A").length).toBeGreaterThan(0));

    press("ArrowDown");
    press("ArrowDown");
    press("ArrowRight");
    expect(startPreview).not.toHaveBeenCalled();
  });
});

describe("the commit verb differs by mode", () => {
  it("plan mode: enter queues the pick", async () => {
    state.planRecs = [rec({ track_id: 7, track_title: "Queue Me" })];
    renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("Queue Me").length).toBeGreaterThan(0));

    press("ArrowDown");
    press("Enter");
    expect(addToRole).toHaveBeenCalledWith("drums", 7);
    expect(loadToDeck).not.toHaveBeenCalled();
  });

  it("live mode: enter loads Deck A, shift+enter Deck B", async () => {
    state.liveRecs = [rec({ track_id: 9, track_title: "Load Me" })];
    renderGrid("live");
    await waitFor(() => expect(screen.getAllByText("Load Me").length).toBeGreaterThan(0));

    press("ArrowDown");
    press("Enter");
    expect(loadToDeck).toHaveBeenCalledWith(
      expect.objectContaining({ trackId: 9, side: "a" }),
    );

    press("Enter", { shiftKey: true });
    expect(loadToDeck).toHaveBeenLastCalledWith(
      expect.objectContaining({ trackId: 9, side: "b" }),
    );
    expect(addToRole).not.toHaveBeenCalled();
  });
});

describe("the cursor survives the list changing under it", () => {
  it("stays in the recs zone when the list refreshes after a commit", async () => {
    // Found on the real rig, not in jsdom: committing bounced the cursor UP
    // into the plan card, so every single add needed an arrow key to get
    // back — the exact stutter this loop exists to remove.
    //
    // Two causes, both about a zone transiently reporting EMPTY:
    //   1. useEffect cleanup runs on every dependency change, not just
    //      unmount, so each new list was preceded by a published [].
    //   2. "loading" was published as "empty" rather than "unknown".
    // A mocked list that never changes cannot catch either; this one does.
    state.planRecs = [
      rec({ track_id: 1, track_title: "First" }),
      rec({ track_id: 2, track_title: "Second" }),
    ];
    const { rerender } = renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("First").length).toBeGreaterThan(0));

    press("ArrowDown");
    await waitFor(() => expect(focusedTitle()).toBe("First"));
    press("Enter");
    expect(addToRole).toHaveBeenCalledWith("drums", 1);

    // The backend re-scores: the committed track leaves the recs and appears
    // in the plan — exactly the shape change that used to relocate the cursor.
    state.plan = { set_id: 1, queues: { drums: [{ track_id: 1, title: "First" }] } };
    state.planRecs = [
      rec({ track_id: 2, track_title: "Second" }),
      rec({ track_id: 3, track_title: "Third" }),
    ];
    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <RoleColumnsGrid setId={1} mode="plan" />
      </QueryClientProvider>,
    );

    // Cursor holds its slot, so the next candidate is already under it and
    // the next Enter commits without an arrow key in between.
    await waitFor(() => expect(focusedTitle()).toBe("Second"));
    press("Enter");
    expect(addToRole).toHaveBeenLastCalledWith("drums", 2);
  });
});

describe("escape", () => {
  it("stops the audition first, and releases the cursor only after", async () => {
    state.planRecs = [rec({ track_id: 3, track_title: "Esc" })];
    const { rerender } = renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("Esc").length).toBeGreaterThan(0));

    press("ArrowDown");
    await waitFor(() => expect(focusedTitle()).toBe("Esc"));

    // Something is auditioning: esc silences it and KEEPS the cursor, so the
    // next arrow key carries on from where you were.
    state.previewing = { trackId: 3, column: "drums" };
    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <RoleColumnsGrid setId={1} mode="plan" />
      </QueryClientProvider>,
    );
    press("Escape");
    expect(stopPreview).toHaveBeenCalled();
    expect(focusedTitle()).toBe("Esc");

    // Nothing auditioning: esc now backs out of the grid.
    state.previewing = null;
    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <RoleColumnsGrid setId={1} mode="plan" />
      </QueryClientProvider>,
    );
    press("Escape");
    await waitFor(() => expect(focusedTitle()).toBeNull());
  });
});

describe("the keyboard yields when it should", () => {
  it("ignores arrows while typing in a field", async () => {
    state.planRecs = [
      rec({ track_id: 1, track_title: "A" }),
      rec({ track_id: 2, track_title: "B" }),
    ];
    renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("A").length).toBeGreaterThan(0));

    const input = document.createElement("input");
    document.body.appendChild(input);
    act(() => {
      fireEvent.keyDown(input, { key: "ArrowDown", bubbles: true });
    });
    // The cursor must not creep while the user edits a set name or searches.
    expect(focusedTitle()).toBeNull();
    input.remove();
  });

  it("ignores modified chords so browser shortcuts still work", async () => {
    state.planRecs = [rec({ track_id: 1, track_title: "A" })];
    renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("A").length).toBeGreaterThan(0));

    press("ArrowDown", { metaKey: true });
    expect(focusedTitle()).toBeNull();
  });
});

describe("mouse and keyboard compose", () => {
  it("clicking a card moves the cursor there", async () => {
    state.planRecs = [
      rec({ track_id: 1, track_title: "A" }),
      rec({ track_id: 2, track_title: "B" }),
    ];
    renderGrid("plan");
    await waitFor(() => expect(screen.getAllByText("B").length).toBeGreaterThan(0));

    const cardB = screen.getAllByText("B")[0].closest("div.rounded-md");
    act(() => {
      fireEvent.mouseDown(cardB!);
    });
    await waitFor(() => expect(focusedTitle()).toBe("B"));

    // …and the keyboard carries on from there rather than resetting.
    press("ArrowUp");
    await waitFor(() => expect(focusedTitle()).toBe("A"));
  });
});
