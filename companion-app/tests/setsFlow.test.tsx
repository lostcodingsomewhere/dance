import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { BoothBackend } from "./boothHarness";
import { API, SetsOnTheServer, holdAnswersTo, type Role } from "./setsFlowBackend";
import {
  HALF_LIGHT,
  LIBRARY,
  LOW_TIDE,
  NIGHT_DRIVE,
  SEVEN_PICKS,
  picksIn,
  picksOnScreen,
  planned,
  renderSetView,
  titlesOf,
} from "./setsFlowView";

function numberedPicksIn(role: Role): string[] {
  return within(planned(role))
    .getAllByRole("group")
    .map((card) => `${within(card).getByText(/^\d$/).textContent} ${card.getAttribute("aria-label")}`);
}

function pick(role: Role, title: string) {
  return within(within(planned(role)).getByRole("group", { name: title }));
}

function plansSaved(setId: number): unknown[] {
  return backend.sent("PUT", `${API}/sets/${setId}/plan`).map((sent) => sent.body);
}

function aSetWithSevenPicks() {
  const server = new SetsOnTheServer(LIBRARY);
  const setId = server.has("Friday night", SEVEN_PICKS, true);
  backend = new BoothBackend(server.answer);
  return { server, setId };
}

async function anyRequestHasGoneOut(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20));
  });
}

async function theSetViewWithSevenPicks() {
  const started = aSetWithSevenPicks();
  renderSetView();
  await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));
  return started;
}

let backend: BoothBackend;

afterEach(() => {
  backend.restore();
});

describe("duplicating a set from the set menu", () => {
  it("copies the whole plan and opens the copy only once its plan is written", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    const friday = server.has("Friday night", SEVEN_PICKS, true);
    backend = new BoothBackend(server.answer);
    renderSetView();
    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));

    fireEvent.click(screen.getByRole("button", { name: "Friday night" }));
    fireEvent.click(screen.getByRole("button", { name: "Duplicate as new" }));
    const copyId = friday + 1;
    const planOfTheCopy = `${API}/sets/${copyId}/plan`;
    const letThePlanBeWritten = holdAnswersTo("PUT", planOfTheCopy);
    const create = screen.getByRole("button", { name: "Create" });
    fireEvent.click(create);

    await waitFor(() => expect(backend.sent("PUT", planOfTheCopy)).toHaveLength(1));
    expect(create).toBeDisabled();
    expect(backend.sent("POST", `${API}/sets/${copyId}/activate`)).toEqual([]);
    expect(server.activeName()).toBe("Friday night");
    letThePlanBeWritten();

    await screen.findByRole("button", { name: "Friday night (copy)" });
    expect(backend.sent("PUT", planOfTheCopy).map((sent) => sent.body)).toEqual([
      { queues: SEVEN_PICKS },
    ]);
    expect(server.planOf(copyId)).toEqual(SEVEN_PICKS);
    expect(server.activeName()).toBe("Friday night (copy)");
    expect(backend.sent("POST", `${API}/sets/${copyId}/tracks`)).toEqual([]);
    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));
    expect(backend.sent("GET", planOfTheCopy).length).toBeGreaterThan(0);
  });

  it("says in the menu that the copy failed and stays on the set you were on", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    const friday = server.has("Friday night", SEVEN_PICKS, true);
    const copyId = friday + 1;
    server.refuses = ({ method, path }) => method === "PUT" && path === `${API}/sets/${copyId}/plan`;
    backend = new BoothBackend(server.answer);
    renderSetView();
    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));

    fireEvent.click(screen.getByRole("button", { name: "Friday night" }));
    fireEvent.click(screen.getByRole("button", { name: "Duplicate as new" }));
    const create = screen.getByRole("button", { name: "Create" });
    fireEvent.click(create);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not duplicate this set. Try again.",
    );
    expect(create).toBeEnabled();
    expect(server.activeName()).toBe("Friday night");
    expect(backend.sent("POST", `${API}/sets/${copyId}/activate`)).toEqual([]);
    expect(server.idOf("Friday night (copy)")).toBeUndefined();

    server.refuses = () => false;
    fireEvent.click(create);

    await screen.findByRole("button", { name: "Friday night (copy)" });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    const secondCopy = server.idOf("Friday night (copy)")!;
    expect(server.planOf(secondCopy)).toEqual(SEVEN_PICKS);
  });
});

describe("reordering picks in the Set view", () => {
  it("moves a pick up or down, saves the new order and renumbers the column", async () => {
    const { setId } = await theSetViewWithSevenPicks();
    const song = within(planned("song"));
    expect(numberedPicksIn("song")).toEqual(["1 Night Drive", "2 Low Tide", "3 Half Light"]);
    expect(song.queryByRole("button", { name: "Move Night Drive up" })).not.toBeInTheDocument();
    expect(song.queryByRole("button", { name: "Move Half Light down" })).not.toBeInTheDocument();
    expect(within(planned("bass")).queryByRole("button", { name: /^Move / })).not.toBeInTheDocument();

    fireEvent.click(song.getByRole("button", { name: "Move Half Light up" }));

    await waitFor(() =>
      expect(numberedPicksIn("song")).toEqual(["1 Night Drive", "2 Half Light", "3 Low Tide"]),
    );
    expect(plansSaved(setId)).toEqual([
      { queues: { ...SEVEN_PICKS, song: [NIGHT_DRIVE.track_id, HALF_LIGHT.track_id, LOW_TIDE.track_id] } },
    ]);

    fireEvent.click(song.getByRole("button", { name: "Move Night Drive down" }));

    await waitFor(() =>
      expect(numberedPicksIn("song")).toEqual(["1 Half Light", "2 Night Drive", "3 Low Tide"]),
    );
    expect(plansSaved(setId).at(-1)).toEqual({
      queues: { ...SEVEN_PICKS, song: [HALF_LIGHT.track_id, NIGHT_DRIVE.track_id, LOW_TIDE.track_id] },
    });
    expect(picksIn("drums")).toEqual(["Half Light", "Night Drive"]);
  });
});

describe("removing a pick in the Set view", () => {
  it("offers to undo, and undo puts the pick back where it was", async () => {
    const { server, setId } = await theSetViewWithSevenPicks();

    fireEvent.click(pick("song", "Low Tide").getByTitle("Remove from plan"));

    const song = within(planned("song"));
    expect(await song.findByText("Removed Low Tide")).toBeInTheDocument();
    expect(picksIn("song")).toEqual(["Night Drive", "Half Light"]);
    expect(server.planOf(setId).song).toEqual([NIGHT_DRIVE.track_id, HALF_LIGHT.track_id]);
    expect(within(planned("drums")).queryByText(/^Removed/)).not.toBeInTheDocument();

    fireEvent.click(song.getByRole("button", { name: "Undo" }));

    await waitFor(() => expect(picksIn("song")).toEqual(["Night Drive", "Low Tide", "Half Light"]));
    expect(server.planOf(setId)).toEqual(SEVEN_PICKS);
    expect(song.queryByText("Removed Low Tide")).not.toBeInTheDocument();
    expect(song.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument();
  });

  it("offers undo for the only pick of a column too", async () => {
    const { server, setId } = await theSetViewWithSevenPicks();

    fireEvent.click(pick("bass", "Low Tide").getByTitle("Remove from plan"));

    const bass = within(planned("bass"));
    expect(await bass.findByText("Removed Low Tide")).toBeInTheDocument();
    expect(bass.getByText("nothing queued")).toBeInTheDocument();

    fireEvent.click(bass.getByRole("button", { name: "Undo" }));

    await waitFor(() => expect(picksIn("bass")).toEqual(["Low Tide"]));
    expect(server.planOf(setId)).toEqual(SEVEN_PICKS);
  });

  it("stops offering undo at the next change to the plan", async () => {
    await theSetViewWithSevenPicks();

    fireEvent.click(pick("song", "Low Tide").getByTitle("Remove from plan"));
    expect(await within(planned("song")).findByText("Removed Low Tide")).toBeInTheDocument();

    fireEvent.click(within(planned("drums")).getByRole("button", { name: "Move Night Drive up" }));

    await waitFor(() => expect(picksIn("drums")).toEqual(["Night Drive", "Half Light"]));
    await waitFor(() => expect(screen.queryByText("Removed Low Tide")).not.toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument();
    expect(picksIn("song")).toEqual(["Night Drive", "Half Light"]);
  });

  it("stops offering undo after about eight seconds", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ["setTimeout", "clearTimeout"] });
    try {
      await theSetViewWithSevenPicks();
      fireEvent.click(pick("song", "Low Tide").getByTitle("Remove from plan"));
      expect(await within(planned("song")).findByText("Removed Low Tide")).toBeInTheDocument();

      act(() => {
        vi.advanceTimersByTime(6_000);
      });
      expect(screen.getByRole("button", { name: "Undo" })).toBeInTheDocument();

      act(() => {
        vi.advanceTimersByTime(3_000);
      });
      expect(screen.queryByText("Removed Low Tide")).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument();
      expect(picksIn("song")).toEqual(["Night Drive", "Half Light"]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("removes one pick when × is double-clicked", async () => {
    const { server, setId } = await theSetViewWithSevenPicks();

    fireEvent.click(pick("song", "Night Drive").getByTitle("Remove from plan"), { detail: 1 });
    await waitFor(() => expect(picksIn("song")).toEqual(["Low Tide", "Half Light"]));
    fireEvent.click(pick("song", "Low Tide").getByTitle("Remove from plan"), { detail: 2 });
    await anyRequestHasGoneOut();

    expect(plansSaved(setId)).toHaveLength(1);
    expect(server.planOf(setId).song).toEqual([LOW_TIDE.track_id, HALF_LIGHT.track_id]);
    await within(planned("song")).findByText("Removed Night Drive");
    expect(picksIn("song")).toEqual(["Low Tide", "Half Light"]);
  });

  it("ignores another × while the first removal is still being saved", async () => {
    const { server, setId } = aSetWithSevenPicks();
    renderSetView();
    await waitFor(() => expect(picksOnScreen()).toEqual(titlesOf(SEVEN_PICKS)));
    const letTheSaveLand = holdAnswersTo("PUT", `${API}/sets/${setId}/plan`);

    fireEvent.click(pick("song", "Night Drive").getByTitle("Remove from plan"));
    await waitFor(() => expect(plansSaved(setId)).toHaveLength(1));
    fireEvent.click(pick("song", "Night Drive").getByTitle("Remove from plan"));
    fireEvent.click(pick("song", "Low Tide").getByTitle("Remove from plan"));
    fireEvent.click(pick("drums", "Half Light").getByTitle("Remove from plan"));
    await anyRequestHasGoneOut();
    expect(plansSaved(setId)).toHaveLength(1);
    letTheSaveLand();

    await waitFor(() => expect(picksIn("song")).toEqual(["Low Tide", "Half Light"]));
    expect(server.planOf(setId)).toEqual({
      ...SEVEN_PICKS,
      song: [LOW_TIDE.track_id, HALF_LIGHT.track_id],
    });
    expect(picksOnScreen()).toEqual(titlesOf(server.planOf(setId)));
    expect(within(planned("song")).getByText("Removed Night Drive")).toBeInTheDocument();
  });
});
