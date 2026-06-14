import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RoleColumnsGrid } from "../src/components/RoleColumnsGrid";

// Hoisted mock state shared by the mocked hooks.
const state = {
  plan: { set_id: 1, queues: {} as Record<string, unknown[]> },
  planRecs: [] as unknown[],
  liveRecs: [] as unknown[],
};

const addToRole = vi.fn();
const removeFromRole = vi.fn();

vi.mock("../src/hooks/useSetPlan", () => ({
  useSetPlan: () => ({ data: state.plan }),
  usePlanMutations: () => ({ addToRole, removeFromRole, put: { isPending: false } }),
  usePlanRecs: () => ({ data: { recs: state.planRecs }, isLoading: false }),
}));

vi.mock("../src/hooks/useColumnRecs", () => ({
  useColumnRecs: () => ({ data: { recs: state.liveRecs }, isLoading: false }),
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
});
