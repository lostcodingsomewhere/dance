import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import {
  BoothBackend,
  LIVE_TRACK_OF,
  renderApp,
  songOn,
  type Cell,
  type SeenRequest,
} from "./boothHarness";
import { API, SetsOnTheServer, rec } from "./setsFlowBackend";
import {
  LIBRARY,
  LOW_TIDE,
  NIGHT_DRIVE,
  SEVEN_PICKS,
  picksIn,
  picksOnScreen,
  recommended,
  recsIn,
  titlesOf,
} from "./setsFlowView";

const TEMPO_MATCH = { track_id: 21, title: "Tempo Match" };
const FITS_THE_PLAN = { track_id: 22, title: "Fits The Plan" };
const CLUB_MIX = { track_id: 23, title: "Night Drive (Club Mix)" };
const CLUB_MIX_OF_NIGHT_DRIVE = {
  ...rec(CLUB_MIX),
  version_of: NIGHT_DRIVE.track_id,
  version_of_title: NIGHT_DRIVE.title,
  version_label: "Club Mix",
};
const BY_COLUMN = `${API}/recommend/by-column`;
const DRUMS_A = String(LIVE_TRACK_OF.drums_a);

function aBoothWith(server: SetsOnTheServer, inLive: Cell[]) {
  return (request: SeenRequest): unknown => {
    if (request.path === `${API}/ableton/decks`) return { columns: LIVE_TRACK_OF, cells: inLive };
    if (request.path === BY_COLUMN) {
      const column = (request.body as { column: string }).column;
      return { column, combo_size: 0, recs: [rec(TEMPO_MATCH)] };
    }
    return server.answer(request);
  };
}

function theAppShowsLivesTempo(): Promise<HTMLElement> {
  return screen.findByRole("button", { name: "128.0" });
}

function askedOfLiveRecs(): { master_bpm: number; exclude_track_ids: number[] }[] {
  return backend
    .sent("POST", BY_COLUMN)
    .map((sent) => sent.body as { master_bpm: number; exclude_track_ids: number[] });
}

let backend: BoothBackend;

afterEach(() => {
  backend.restore();
});

describe("recs in the Booth while nothing plays", () => {
  it("are scored against the active set's plan, and follow the decks once one plays", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    const setId = server.has("Friday night", SEVEN_PICKS, true);
    server.recs = (role) =>
      role === "song" ? [CLUB_MIX_OF_NIGHT_DRIVE, rec(FITS_THE_PLAN)] : [rec(FITS_THE_PLAN)];
    const preparedSetInLive = [...songOn(0, "a", NIGHT_DRIVE), ...songOn(0, "b", LOW_TIDE)];
    backend = new BoothBackend(aBoothWith(server, preparedSetInLive));
    renderApp();
    await screen.findByRole("button", { name: "Page 1" });
    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));
    expect(screen.queryByRole("button", { name: /^Move .* (up|down)$/ })).not.toBeInTheDocument();

    backend.liveNowPlays({});
    await theAppShowsLivesTempo();

    await waitFor(() => expect(recsIn("song")).toEqual([CLUB_MIX.title, FITS_THE_PLAN.title]));
    expect(recsIn("drums")).toEqual([FITS_THE_PLAN.title]);
    expect(within(recommended("song")).getByText("from your plan")).toBeInTheDocument();
    expect(backend.sent("GET", `${API}/sets/${setId}/plan-recs`).map((sent) => sent.query)).toContain(
      "role=song&k=4",
    );
    expect(askedOfLiveRecs()).toEqual([]);
    expect(screen.queryByRole("group", { name: TEMPO_MATCH.title })).not.toBeInTheDocument();
    expect(
      within(recommended("song")).getByTitle("Load → Deck B, the deck beside its original"),
    ).toBeInTheDocument();

    backend.liveNowPlays({ [DRUMS_A]: 0 });

    await waitFor(() => expect(recsIn("song")).toEqual([TEMPO_MATCH.title]));
    expect(screen.queryByText("from your plan")).not.toBeInTheDocument();
    expect(askedOfLiveRecs().at(-1)).toMatchObject({ master_bpm: 128 });
    expect(askedOfLiveRecs().at(-1)?.exclude_track_ids).toContain(NIGHT_DRIVE.track_id);
  });

  it("stay ranked on tempo while the active set has nothing planned, and turn to the plan at its first pick", async () => {
    const server = new SetsOnTheServer([...LIBRARY, TEMPO_MATCH]);
    const setId = server.has("New set", {}, true);
    server.recs = () => [rec(FITS_THE_PLAN)];
    backend = new BoothBackend(aBoothWith(server, []));
    renderApp();
    await screen.findByRole("group", { name: "Song plan" });

    backend.liveNowPlays({});
    await theAppShowsLivesTempo();

    await waitFor(() => expect(recsIn("song")).toEqual([TEMPO_MATCH.title]));
    expect(screen.queryByText("from your plan")).not.toBeInTheDocument();
    expect(screen.queryByRole("group", { name: FITS_THE_PLAN.title })).not.toBeInTheDocument();
    expect(askedOfLiveRecs().at(-1)).toMatchObject({ master_bpm: 128, exclude_track_ids: [] });
    expect(backend.sent("GET", `${API}/sets/${setId}/plan-recs`)).toEqual([]);

    fireEvent.click(
      within(within(recommended("song")).getByRole("group", { name: TEMPO_MATCH.title })).getByTitle(
        "Add to plan",
      ),
    );

    await waitFor(() => expect(picksIn("song")).toEqual([TEMPO_MATCH.title]));
    await waitFor(() => expect(recsIn("song")).toEqual([FITS_THE_PLAN.title]));
    expect(within(recommended("song")).getByText("from your plan")).toBeInTheDocument();
    expect(recsIn("drums")).toEqual([FITS_THE_PLAN.title]);
  });
});
