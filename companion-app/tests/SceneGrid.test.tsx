import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SceneGrid } from "../src/components/SceneGrid";

// Hoisted mock state — both data hooks and transport hooks read from these.
const state = {
  columns: null as Record<string, number> | null,
  scenes: [] as Array<{
    scene_index: number;
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
  useDeckMap: () => ({ data: { columns: state.columns, scenes: state.scenes } }),
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
  state.scenes = [];
  state.playingClips = {};
  state.tempo = 120;
  fireSceneMutate.mockReset();
  fireCellMutate.mockReset();
});

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

  it("shows the loaded track title on each cell of a populated scene row", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
    state.scenes = [
      {
        scene_index: 0,
        track_id: 42,
        title: "Test Track",
        artist: "Tester",
        bpm: 128,
        key_camelot: "8A",
        floor_energy: 6,
      },
    ];
    renderGrid();
    const titles = screen.getAllByText(/Test Track/i);
    // Five cells (one per stem column) on the loaded row should each display
    // the track title.
    expect(titles.length).toBe(5);
  });

  it("fires the whole scene when its row label is clicked", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
    state.scenes = [
      {
        scene_index: 2,
        track_id: 7,
        title: "Anchor Track",
        artist: "Tester",
        bpm: 124,
        key_camelot: "5A",
        floor_energy: 5,
      },
    ];
    renderGrid();
    // Scene 2 is the 3rd row, button text "3".
    fireEvent.click(screen.getByRole("button", { name: "3" }));
    expect(fireSceneMutate).toHaveBeenCalledWith(2);
  });

  it("fires only the matching cell when a loaded cell is clicked", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
    state.scenes = [
      {
        scene_index: 0,
        track_id: 11,
        title: "Cellable",
        artist: "Tester",
        bpm: 120,
        key_camelot: "1A",
        floor_energy: 4,
      },
    ];
    renderGrid();
    // Find the drums cell on row 0 by its title attribute.
    const drumsCell = screen.getByTitle(/drums: Cellable/i);
    fireEvent.click(drumsCell);
    expect(fireCellMutate).toHaveBeenCalledWith({ track: 0, slot: 0 });
  });

  it("highlights cells that are currently playing", () => {
    state.columns = { drums: 0, bass: 1, vocals: 2, other: 3, mix: 4 };
    state.scenes = [
      {
        scene_index: 0,
        track_id: 33,
        title: "Now Playing",
        artist: "Tester",
        bpm: 130,
        key_camelot: "9A",
        floor_energy: 7,
      },
    ];
    // Drums track (idx 0) is playing scene 0; vocals (idx 2) is not.
    state.playingClips = { 0: 0 };
    renderGrid();
    expect(screen.getByText(/▶ playing/i)).toBeInTheDocument();
  });
});
