import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { SetEditor } from "../src/views/SetEditor";
import { BoothBackend } from "./boothHarness";
import { API, ROLES, SetsOnTheServer, holdAnswersTo, type Plan, type Role } from "./setsFlowBackend";

const NIGHT_DRIVE = { track_id: 7, title: "Night Drive" };
const HALF_LIGHT = { track_id: 9, title: "Half Light" };
const LOW_TIDE = { track_id: 3, title: "Low Tide" };
const SLOW_BURN = { track_id: 5, title: "Slow Burn" };
const LIBRARY = [NIGHT_DRIVE, HALF_LIGHT, LOW_TIDE, SLOW_BURN];
const COLUMN_LABEL: Record<Role, string> = {
  drums: "Drums",
  bass: "Bass",
  vocals: "Vocals",
  other: "Melody",
  song: "Song",
};
const SEVEN_PICKS: Plan = {
  drums: [HALF_LIGHT.track_id, NIGHT_DRIVE.track_id],
  bass: [LOW_TIDE.track_id],
  vocals: [],
  other: [SLOW_BURN.track_id],
  song: [NIGHT_DRIVE.track_id, LOW_TIDE.track_id, HALF_LIGHT.track_id],
};

function titleOf(trackId: number): string {
  return LIBRARY.find((song) => song.track_id === trackId)!.title;
}

function renderSetView() {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <SetEditor />
    </QueryClientProvider>,
  );
}

function planned(role: Role): HTMLElement {
  return screen.getByRole("group", { name: `${COLUMN_LABEL[role]} plan` });
}

function picksIn(role: Role): string[] {
  return within(planned(role))
    .queryAllByRole("group")
    .map((card) => card.getAttribute("aria-label") ?? "");
}

function picksOnScreen(): Record<Role, string[]> {
  return Object.fromEntries(ROLES.map((role) => [role, picksIn(role)])) as Record<Role, string[]>;
}

function titlesOf(plan: Plan): Record<Role, string[]> {
  return Object.fromEntries(ROLES.map((role) => [role, plan[role].map(titleOf)])) as Record<
    Role,
    string[]
  >;
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
