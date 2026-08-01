import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RoleColumnsGrid } from "../src/components/RoleColumnsGrid";

// Hoisted mock state shared by the mocked hooks.
const state = {
  plan: { set_id: 1, queues: {} as Record<string, unknown[]> },
  planRecs: [] as unknown[],
  liveRecs: [] as unknown[],
  // Whether the live recommender has anything to score against (something
  // playing / a tempo / trailing plays). False = the Booth's cold-open state.
  hasContext: true,
};

const addToRole = vi.fn();
const removeFromRole = vi.fn();

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
  useStartPreview: () => ({ mutate: vi.fn(), isPending: false }),
  useStopPreview: () => ({ mutate: vi.fn(), isPending: false }),
  usePreviewState: () => null,
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
    score_breakdown: { embedding: 0.9, bpm: 0.7 },
    reasons: ["close vibe"],
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

beforeEach(() => {
  state.plan = { set_id: 1, queues: {} };
  state.planRecs = [];
  state.liveRecs = [];
  state.hasContext = true;
  addToRole.mockReset();
  removeFromRole.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("RoleColumnsGrid", () => {
  it("renders all five role columns", () => {
    renderGrid("plan");
    for (const label of ["Drums", "Bass", "Vocals"]) {
      expect(screen.getAllByText(new RegExp(label, "i")).length).toBeGreaterThan(0);
    }
  });

  it("shows queued picks in the plan zone (Set/plan mode)", () => {
    state.plan = {
      set_id: 1,
      queues: { drums: [{ track_id: 7, title: "Queued Kick", artist: "Q", key_camelot: "8A", bpm: 124, floor_energy: 5 }] },
    };
    renderGrid("plan");
    expect(screen.getByText("Queued Kick")).toBeInTheDocument();
  });

  it("plan mode shows plan-scored recs and NO deck-load buttons", () => {
    state.planRecs = [rec({ track_id: 2, track_title: "Plan Rec" })];
    renderGrid("plan");
    // The same mocked rec surfaces in each role column.
    expect(screen.getAllByText("Plan Rec").length).toBeGreaterThan(0);
    // ⤒A/⤒B deck-load is live-only.
    expect(screen.queryByTitle(/Load → Deck A/i)).not.toBeInTheDocument();
  });

  it("live mode (Booth) shows deck-load buttons on rec cards", () => {
    state.liveRecs = [rec({ track_id: 3, track_title: "Live Rec" })];
    renderGrid("live", null);
    expect(screen.getAllByText("Live Rec").length).toBeGreaterThan(0);
    expect(screen.getAllByTitle(/Load → Deck A/i).length).toBeGreaterThan(0);
  });

  // Cold open — nothing playing. Previously every role column rendered the
  // same arbitrary tracks at score 0.00, because the scorer had no computable
  // feature to work with. That is the state the Booth OPENS in, so it was the
  // first thing you ever saw. Now the plan fills in, or we say nothing.
  it("cold Booth with an active set falls back to plan-scored recs", () => {
    state.hasContext = false;
    state.planRecs = [rec({ track_id: 9, track_title: "Plan Fallback" })];
    renderGrid("live", 1);
    expect(screen.getAllByText("Plan Fallback").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/from your plan/i).length).toBeGreaterThan(0);
  });

  it("cold Booth with no set says so instead of showing zero-score noise", () => {
    state.hasContext = false;
    state.liveRecs = [rec({ track_id: 4, track_title: "Should Not Show" })];
    renderGrid("live", null);
    expect(screen.queryByText("Should Not Show")).not.toBeInTheDocument();
    expect(screen.getAllByText(/nothing playing/i).length).toBe(5);
  });
});
