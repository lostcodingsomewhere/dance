import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BoothColumnHeaders } from "../src/components/BoothColumnHeaders";

/**
 * The APC40 mk2's eight channel strips address Live's FIRST EIGHT TRACKS.
 * The header used to print `i + 1` — the on-screen column order — which
 * asserts a hardware mapping the app cannot guarantee: deck columns are
 * appended wherever Live's track count happens to be, so anything preceding
 * them (the user's own tracks, a different Set) makes the badge confidently
 * name a strip that controls something else.
 */

const state = { columns: {} as Record<string, number> };

vi.mock("../src/hooks/useDeckMap", () => ({
  useDeckMap: () => ({ data: { columns: state.columns, cells: [] } }),
}));
vi.mock("../src/hooks/useAbletonState", () => ({
  useAbletonState: () => ({ soloed_kinds: [], playing_clips: {} }),
}));
vi.mock("../src/hooks/useTransport", () => ({
  useSoloTrack: () => ({ mutate: vi.fn() }),
}));

const ALIGNED = {
  drums_a: 0, drums_b: 1, bass_a: 2, bass_b: 3,
  vocals_a: 4, vocals_b: 5, other_a: 6, other_b: 7,
  mix_a: 8, mix_b: 9,
};
// Three of the user's own tracks first — the layout that used to lie.
const SHIFTED = Object.fromEntries(
  Object.entries(ALIGNED).map(([k, v]) => [k, v + 3]),
) as Record<string, number>;

function renderHeaders() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BoothColumnHeaders />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  state.columns = { ...ALIGNED };
});

describe("APC40 fader badges", () => {
  it("numbers the strips from the real Ableton track index", () => {
    renderHeaders();
    expect(screen.getByTestId("fader-badge-drums_a").textContent).toBe("1");
    expect(screen.getByTestId("fader-badge-other_b").textContent).toBe("8");
  });

  it("shows no number when the column is out of the APC40's reach", () => {
    // Deck columns at 3..12: only the first five are within strips 1-8.
    state.columns = SHIFTED;
    renderHeaders();
    // drums_a is Ableton track 3 -> strip 4, NOT strip 1.
    expect(screen.getByTestId("fader-badge-drums_a").textContent).toBe("4");
    // other_b is track 10 — past the eight strips entirely.
    expect(screen.getByTestId("fader-badge-other_b").textContent).toBe("—");
  });

  it("never claims strip 1 for a column that is not Ableton track 0", () => {
    state.columns = SHIFTED;
    renderHeaders();
    const ones = Object.keys(ALIGNED)
      .filter((k) => !k.startsWith("mix"))
      .map((k) => screen.queryByTestId(`fader-badge-${k}`)?.textContent)
      .filter((t) => t === "1");
    expect(ones).toHaveLength(0);
  });
});
