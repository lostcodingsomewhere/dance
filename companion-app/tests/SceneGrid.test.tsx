import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SceneGrid } from "../src/components/SceneGrid";

// Hoisted mock state — both data hooks and transport hooks read from these.
const state = {
  columns: null as Record<string, number> | null,
  cells: [] as Array<{
    scene_index: number;
    kind: string;
    track_id: number;
    title: string;
    artist: string;
    bpm: number;
    key_camelot: string;
    floor_energy: number;
  }>,
  playingClips: {} as Record<number, number>,
  tempo: 120,
};

const fireSceneMutate = vi.fn();
const fireCellMutate = vi.fn();

vi.mock("../src/hooks/useDeckMap", () => ({
  useDeckMap: () => ({ data: { columns: state.columns, cells: state.cells } }),
}));

vi.mock("../src/hooks/useAbletonState", () => ({
  useAbletonState: () => ({
    tempo: state.tempo,
    is_playing: true,
    beat: 0,
    playing_clips: state.playingClips,
    track_volumes: {},
  }),
}));

vi.mock("../src/hooks/useTransport", () => ({
  useFireScene: () => ({ mutate: fireSceneMutate, isPending: false }),
  useFireCell: () => ({ mutate: fireCellMutate, isPending: false }),
  useStopScene: () => ({ mutate: vi.fn(), isPending: false }),
  useStopCell: () => ({ mutate: vi.fn(), isPending: false }),
  useStopTrack: () => ({ mutate: vi.fn(), isPending: false }),
  useStopAllClips: () => ({ mutate: vi.fn(), isPending: false }),
  useSoloTrack: () => ({ mutate: vi.fn(), isPending: false }),
  useDeleteCell: () => ({ mutate: vi.fn(), isPending: false }),
}));

function renderGrid() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <SceneGrid />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  state.columns = null;
  state.cells = [];
  state.playingClips = {};
  state.tempo = 120;
  fireSceneMutate.mockReset();
  fireCellMutate.mockReset();
});

// Helper: build a full 4-stem row from one source track. ``side`` picks
// which deck pair gets the stems (A or B). Anchor-mode tests typically
// use 'a' (the conventional primary side).
function fillRow(
  sceneIndex: number,
  track: {
    track_id: number;
    title: string;
    artist: string;
    bpm: number;
    key_camelot: string;
    floor_energy: number;
  },
  side: "a" | "b" = "a",
) {
  for (const source of ["drums", "bass", "vocals", "other"]) {
    state.cells.push({
      scene_index: sceneIndex,
      kind: `${source}_${side}`,
      ...track,
    });
  }
}

// 9 deck columns (8 stem decks A/B per role + mix) — the post-deck-pair
// reality. Tests use this constant rather than re-typing it everywhere.
const DECK_COLUMNS: Record<string, number> = {
  drums_a: 0, drums_b: 1,
  bass_a: 2, bass_b: 3,
  vocals_a: 4, vocals_b: 5,
  other_a: 6, other_b: 7,
  mix: 8,
};

afterEach(() => {
  vi.clearAllMocks();
});

describe("SceneGrid", () => {
  it("shows the waiting-state hint when no deck columns are populated", () => {
    renderGrid();
    expect(screen.getByText(/Waiting for Ableton deck columns/i)).toBeInTheDocument();
  });

  it("renders 3 rows by default once deck columns exist, with an expand toggle", () => {
    state.columns = DECK_COLUMNS;
    renderGrid();
    // Default-collapsed: rows 1..3 visible, 4..8 hidden behind expand toggle.
    for (let i = 1; i <= 3; i++) {
      expect(screen.getByRole("button", { name: String(i) })).toBeInTheDocument();
    }
    expect(screen.queryByRole("button", { name: "4" })).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /show all 8 rows/i }),
    ).toBeInTheDocument();
  });

  it("expands to 8 rows when the toggle is clicked", () => {
    state.columns = DECK_COLUMNS;
    renderGrid();
    fireEvent.click(
      screen.getByRole("button", { name: /show all 8 rows/i }),
    );
    for (let i = 1; i <= 8; i++) {
      expect(screen.getByRole("button", { name: String(i) })).toBeInTheDocument();
    }
  });

  it("shows the loaded track title on each A-side half-cell of an anchor row + once in the SONG shadow", () => {
    state.columns = DECK_COLUMNS;
    fillRow(0, {
      track_id: 42,
      title: "Test Track",
      artist: "Tester",
      bpm: 128,
      key_camelot: "8A",
      floor_energy: 6,
    }, "a");
    renderGrid();
    const titles = screen.getAllByText(/Test Track/i);
    // 4 A-side half-cells (drums_a/bass_a/vocals_a/other_a) + 1 SONG
    // shadow cell that infers the anchor track = 5 total.
    expect(titles.length).toBe(5);
  });

  it("fires the whole scene when its row label is clicked", () => {
    state.columns = DECK_COLUMNS;
    fillRow(2, {
      track_id: 7,
      title: "Anchor Track",
      artist: "Tester",
      bpm: 124,
      key_camelot: "5A",
      floor_energy: 5,
    });
    renderGrid();
    // Scene 2 is the 3rd row, button text "3".
    fireEvent.click(screen.getByRole("button", { name: "3" }));
    expect(fireSceneMutate).toHaveBeenCalledWith(2);
  });

  it("fires only the matching half-cell when a loaded cell is clicked", () => {
    state.columns = DECK_COLUMNS;
    // Single-stem load: drums on A-side of scene 0.
    state.cells.push({
      scene_index: 0,
      kind: "drums_a",
      track_id: 11,
      title: "Cellable",
      artist: "Tester",
      bpm: 120,
      key_camelot: "1A",
      floor_energy: 4,
    });
    renderGrid();
    // The drums-A half-cell shows its source-track title; other half-
    // cells stay empty. drums_a → Live track index 0 per DECK_COLUMNS.
    const drumsCell = screen.getByTitle(/Drums A: Cellable/i);
    fireEvent.click(drumsCell);
    expect(fireCellMutate).toHaveBeenCalledWith({ track: 0, slot: 0 });
  });

  it("supports cells in the same row sourced from different tracks", () => {
    state.columns = DECK_COLUMNS;
    state.cells.push(
      {
        scene_index: 0,
        kind: "drums_a",
        track_id: 11,
        title: "Drum Donor",
        artist: "A",
        bpm: 128,
        key_camelot: "8A",
        floor_energy: 7,
      },
      {
        scene_index: 0,
        kind: "vocals_a",
        track_id: 22,
        title: "Vocal Donor",
        artist: "B",
        bpm: 128,
        key_camelot: "8A",
        floor_energy: 7,
      },
    );
    renderGrid();
    // Both half-cells render with their own source-track titles.
    expect(screen.getByTitle(/Drums A: Drum Donor/i)).toBeInTheDocument();
    expect(screen.getByTitle(/Vocals A: Vocal Donor/i)).toBeInTheDocument();
  });

  it("supports A and B sides of the same role from different tracks", () => {
    state.columns = DECK_COLUMNS;
    state.cells.push(
      {
        scene_index: 0,
        kind: "drums_a",
        track_id: 11,
        title: "Current Drums",
        artist: "A",
        bpm: 128,
        key_camelot: "8A",
        floor_energy: 7,
      },
      {
        scene_index: 0,
        kind: "drums_b",
        track_id: 22,
        title: "Incoming Drums",
        artist: "B",
        bpm: 128,
        key_camelot: "8A",
        floor_energy: 7,
      },
    );
    renderGrid();
    // The two sides of the drums column render independently — this is
    // the whole point of deck pairs.
    expect(screen.getByTitle(/Drums A: Current Drums/i)).toBeInTheDocument();
    expect(screen.getByTitle(/Drums B: Incoming Drums/i)).toBeInTheDocument();
  });

  it("highlights half-cells that are currently playing", () => {
    state.columns = DECK_COLUMNS;
    state.cells.push({
      scene_index: 0,
      kind: "drums_a",
      track_id: 33,
      title: "Now Playing",
      artist: "Tester",
      bpm: 130,
      key_camelot: "9A",
      floor_energy: 7,
    });
    // drums_a is at Live track 0 per DECK_COLUMNS.
    state.playingClips = { 0: 0 };
    renderGrid();
    // The Stop tooltip flags the playing half-cell.
    const stopBtn = screen.getByTitle(/Stop drums a/i);
    expect(stopBtn).toBeInTheDocument();
  });
});
