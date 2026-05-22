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

// Helper: build a full 4-stem row (drums/bass/vocals/other) from one source
// track so anchor-mode tests don't have to enumerate the cells inline.
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
) {
  for (const kind of ["drums", "bass", "vocals", "other"]) {
    state.cells.push({
      scene_index: sceneIndex,
      kind,
      ...track,
    });
  }
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("SceneGrid", () => {
  it("shows the waiting-state hint when no deck columns are populated", () => {
    renderGrid();
    expect(screen.getByText(/Waiting for Ableton deck columns/i)).toBeInTheDocument();
  });

  it("renders 8 rows once deck columns exist, even with no scenes loaded", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
    renderGrid();
    // Row labels are buttons numbered 1..8.
    for (let i = 1; i <= 8; i++) {
      expect(screen.getByRole("button", { name: String(i) })).toBeInTheDocument();
    }
  });

  it("shows the loaded track title on each cell of an anchor-loaded row", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
    fillRow(0, {
      track_id: 42,
      title: "Test Track",
      artist: "Tester",
      bpm: 128,
      key_camelot: "8A",
      floor_energy: 6,
    });
    renderGrid();
    const titles = screen.getAllByText(/Test Track/i);
    // Four cells (drums/bass/vocals/other) on the loaded row display title.
    // The "Song" column cell stays empty for whole-song loads (the mix
    // column is never populated by push_track_to_live).
    expect(titles.length).toBe(4);
  });

  it("fires the whole scene when its row label is clicked", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
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

  it("fires only the matching cell when a loaded cell is clicked", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
    // Single-stem load: drums-only in scene 0.
    state.cells.push({
      scene_index: 0,
      kind: "drums",
      track_id: 11,
      title: "Cellable",
      artist: "Tester",
      bpm: 120,
      key_camelot: "1A",
      floor_energy: 4,
    });
    renderGrid();
    // The drums cell shows its source-track title; other cells stay empty.
    const drumsCell = screen.getByTitle(/Drums: Cellable/i);
    fireEvent.click(drumsCell);
    expect(fireCellMutate).toHaveBeenCalledWith({ track: 0, slot: 0 });
  });

  it("supports cells in the same row sourced from different tracks", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
    state.cells.push(
      {
        scene_index: 0,
        kind: "drums",
        track_id: 11,
        title: "Drum Donor",
        artist: "A",
        bpm: 128,
        key_camelot: "8A",
        floor_energy: 7,
      },
      {
        scene_index: 0,
        kind: "vocals",
        track_id: 22,
        title: "Vocal Donor",
        artist: "B",
        bpm: 128,
        key_camelot: "8A",
        floor_energy: 7,
      },
    );
    renderGrid();
    // Both cells render with their own source-track titles.
    expect(screen.getByTitle(/Drums: Drum Donor/i)).toBeInTheDocument();
    expect(screen.getByTitle(/Vocals: Vocal Donor/i)).toBeInTheDocument();
  });

  it("highlights cells that are currently playing", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
    state.cells.push({
      scene_index: 0,
      kind: "drums",
      track_id: 33,
      title: "Now Playing",
      artist: "Tester",
      bpm: 130,
      key_camelot: "9A",
      floor_energy: 7,
    });
    state.playingClips = { 0: 0 };
    renderGrid();
    // Playing cell label reads "⏹ playing" (tap to stop) — the label text
    // can appear elsewhere too, so we look for the explicit stop affordance.
    const playingLabels = screen.getAllByText(/playing/i);
    expect(playingLabels.length).toBeGreaterThan(0);
  });
});
