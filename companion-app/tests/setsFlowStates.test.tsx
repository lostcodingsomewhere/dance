import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { BoothBackend } from "./boothHarness";
import { API, ROLES, SetsOnTheServer, rec } from "./setsFlowBackend";
import {
  HALF_LIGHT,
  LIBRARY,
  NIGHT_DRIVE,
  SEVEN_PICKS,
  SLOW_BURN,
  picksIn,
  picksOnScreen,
  planned,
  recommended,
  recsIn,
  renderSetView,
  titlesOf,
} from "./setsFlowView";

const FIRST_SONG_PROMPT =
  "Nothing planned yet. Press ⌘K and add a first song. Recommendations follow from what you plan.";
const CREATE_A_SET = "+ create empty set";

function theSameUnscoredTracksInEveryColumn() {
  return LIBRARY.map((song) => rec(song, 0));
}

let backend: BoothBackend;

afterEach(() => {
  backend.restore();
});

describe("a set with nothing planned", () => {
  it("asks for a first song instead of listing unscored recs", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    const setId = server.has("New set", {}, true);
    server.recs = theSameUnscoredTracksInEveryColumn;
    backend = new BoothBackend(server.answer);
    renderSetView();

    const prompt = await screen.findByText(FIRST_SONG_PROMPT);
    expect(prompt).toHaveClass("text-sm", "text-neutral-300");
    expect(screen.getAllByText("nothing queued")).toHaveLength(ROLES.length);
    expect(screen.queryByRole("group", { name: NIGHT_DRIVE.title })).not.toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "Song recs" })).not.toBeInTheDocument();
    expect(backend.sent("GET", `${API}/sets/${setId}/plan-recs`)).toEqual([]);
  });

  it("brings the recs back as soon as one pick is planned", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    const setId = server.has("One song", { song: [NIGHT_DRIVE.track_id] }, true);
    server.recs = () => [rec(HALF_LIGHT), rec(SLOW_BURN)];
    backend = new BoothBackend(server.answer);
    renderSetView();

    await waitFor(() => expect(recsIn("song")).toEqual([HALF_LIGHT.title, SLOW_BURN.title]));
    expect(screen.queryByText(FIRST_SONG_PROMPT)).not.toBeInTheDocument();
    const askedWithAPick = backend.sent("GET", `${API}/sets/${setId}/plan-recs`).length;
    expect(askedWithAPick).toBe(ROLES.length);

    fireEvent.click(
      within(within(planned("song")).getByRole("group", { name: NIGHT_DRIVE.title })).getByTitle(
        "Remove from plan",
      ),
    );

    expect(await screen.findByText(FIRST_SONG_PROMPT)).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: HALF_LIGHT.title })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => expect(recsIn("drums")).toEqual([HALF_LIGHT.title, SLOW_BURN.title]));
    expect(screen.queryByText(FIRST_SONG_PROMPT)).not.toBeInTheDocument();
    expect(picksIn("song")).toEqual([NIGHT_DRIVE.title]);
  });
});

describe("a request the Set view cannot get an answer to", () => {
  it("says the plan could not be loaded instead of showing empty queues, and loads it on Retry", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    const setId = server.has("Friday night", SEVEN_PICKS, true);
    server.unreachableFor = ({ path }) => path === `${API}/sets/${setId}/plan`;
    backend = new BoothBackend(server.answer);
    renderSetView();

    const failure = await screen.findByText("Could not reach the server. Your plan is not shown.");
    expect(failure.closest('[role="alert"]')).not.toBeNull();
    expect(screen.queryByText("nothing queued")).not.toBeInTheDocument();
    expect(screen.queryByText(FIRST_SONG_PROMPT)).not.toBeInTheDocument();

    server.unreachableFor = () => false;
    fireEvent.click(
      within(failure.closest('[role="alert"]') as HTMLElement).getByRole("button", { name: "Retry" }),
    );

    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));
    expect(screen.queryByText(/Could not reach the server/)).not.toBeInTheDocument();
  });

  it("says a column's recs could not be loaded instead of 'no candidates', and loads them on Retry", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    server.has("Friday night", SEVEN_PICKS, true);
    server.recs = () => [rec(SLOW_BURN)];
    server.refuses = ({ path, query }) => path.endsWith("/plan-recs") && query.includes("role=bass");
    backend = new BoothBackend(server.answer);
    renderSetView();

    await waitFor(() => expect(recsIn("drums")).toEqual([SLOW_BURN.title]));
    const failure = await within(recommended("bass")).findByRole("alert");
    expect(failure).toHaveTextContent("Could not reach the server.");
    expect(within(recommended("bass")).queryByText("no candidates")).not.toBeInTheDocument();
    expect(screen.getAllByRole("alert")).toHaveLength(1);

    server.refuses = () => false;
    fireEvent.click(within(failure).getByRole("button", { name: "Retry" }));

    await waitFor(() => expect(recsIn("bass")).toEqual([SLOW_BURN.title]));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("does not offer to create a set when it cannot tell which set is active", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    server.has("Friday night", SEVEN_PICKS, true);
    server.unreachableFor = ({ path }) => path === `${API}/sets/active`;
    backend = new BoothBackend(server.answer);
    renderSetView();

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not reach the server");
    expect(screen.queryByText("No active set yet")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: CREATE_A_SET })).not.toBeInTheDocument();

    server.unreachableFor = () => false;
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("the Set view with no active set", () => {
  it("lists the sets that exist with their picks, and opens the one you choose", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    const friday = server.has("Friday night", SEVEN_PICKS);
    server.has("Warm up", { song: [NIGHT_DRIVE.track_id] });
    server.has("Scratch");
    backend = new BoothBackend(server.answer);
    renderSetView();

    const sets = await screen.findByRole("list", { name: "Your sets" });
    expect(within(sets).getAllByRole("listitem").map((row) => row.textContent)).toEqual([
      "Friday night7 picksOpen",
      "Warm up1 pickOpen",
      "Scratch0 picksOpen",
    ]);
    const create = screen.getByRole("button", { name: CREATE_A_SET });
    expect(sets.compareDocumentPosition(create) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    fireEvent.click(within(sets).getByRole("button", { name: "Open Friday night" }));

    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));
    expect(backend.sent("POST", `${API}/sets/${friday}/activate`)).toHaveLength(1);
    expect(backend.sent("POST", `${API}/sets`)).toEqual([]);
  });

  it("offers the sets that are left after the active one is deleted", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    server.has("Friday night", SEVEN_PICKS, true);
    server.has("Warm up", { song: [NIGHT_DRIVE.track_id] });
    backend = new BoothBackend(server.answer);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderSetView();
    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));

    fireEvent.click(screen.getByRole("button", { name: "Friday night" }));
    await screen.findByText("Switch to");
    fireEvent.click(screen.getByRole("button", { name: "Delete current" }));

    const sets = await screen.findByRole("list", { name: "Your sets" });
    expect(within(sets).getAllByRole("listitem").map((row) => row.textContent)).toEqual([
      "Warm up1 pickOpen",
    ]);
    expect(screen.getByRole("button", { name: CREATE_A_SET })).toBeInTheDocument();

    fireEvent.click(within(sets).getByRole("button", { name: "Open Warm up" }));

    await waitFor(() => expect(picksIn("song")).toEqual([NIGHT_DRIVE.title]));
  });

  it("says so when the sets cannot be listed or the chosen one cannot be opened", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    const friday = server.has("Friday night", SEVEN_PICKS);
    server.unreachableFor = ({ method, path }) => method === "GET" && path === `${API}/sets`;
    backend = new BoothBackend(server.answer);
    renderSetView();

    const notListed = await screen.findByText(/your sets are not listed/);
    expect(notListed).toHaveTextContent("Could not reach the server, so your sets are not listed.");
    server.unreachableFor = ({ path }) => path === `${API}/sets/${friday}/activate`;
    fireEvent.click(within(notListed).getByRole("button", { name: "Retry" }));

    const sets = await screen.findByRole("list", { name: "Your sets" });
    fireEvent.click(within(sets).getByRole("button", { name: "Open Friday night" }));
    expect(await screen.findByText("Could not reach the server. Try again.")).toBeInTheDocument();

    server.unreachableFor = () => false;
    fireEvent.click(within(sets).getByRole("button", { name: "Open Friday night" }));
    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));
  });

  it("names the columns as the grid labels them", async () => {
    backend = new BoothBackend(new SetsOnTheServer(LIBRARY).answer);
    renderSetView();

    const explanation = await screen.findByText(/A set is a plan/);
    expect(explanation).toHaveTextContent("Drums, Bass, Vocals, Melody, Song");
    expect(explanation).not.toHaveTextContent(/other/i);
    expect(screen.queryByRole("list", { name: "Your sets" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: CREATE_A_SET })).toBeInTheDocument();
  });
});
