import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { BoothMidSet, RECOMMENDED, SONG_COLUMN, press, recommendation } from "./failuresFlowBooth";

const WHAT_BROKE = "Cannot read properties of null (reading 'join')";

let booth: BoothMidSet;
let recommendationsAreMalformed: boolean;

function hushTheErrorReactLogsWhenAScreenBreaks(): void {
  vi.spyOn(console, "error").mockImplementation(() => {});
}

beforeEach(() => {
  booth = new BoothMidSet();
  recommendationsAreMalformed = true;
  booth.liveAnswers = ({ path, body }) => {
    if (!path.endsWith("/recommend/by-column") || !recommendationsAreMalformed) return undefined;
    const column = (body as { column: string }).column;
    const malformed = { ...recommendation(), reasons: null };
    return { column, combo_size: 0, recs: column === SONG_COLUMN ? [malformed] : [] };
  };
  hushTheErrorReactLogsWhenAScreenBreaks();
});

afterEach(() => {
  booth.close();
});

describe("a screen that breaks while it is drawn", () => {
  it("says so below the header, keeps the transport and Panic working, and comes back on Reload", async () => {
    booth.show();

    const broken = await screen.findByRole("alert");
    expect(broken).toHaveTextContent("Something broke on this screen");
    expect(broken).toHaveTextContent(WHAT_BROKE);
    expect(screen.queryByRole("group", { name: RECOMMENDED.title })).not.toBeInTheDocument();

    press("Stop transport");
    await waitFor(() => expect(booth.sent("POST", "/ableton/stop")).toHaveLength(1));
    press("Panic — stop all clips");
    press("PANIC");
    await waitFor(() => expect(booth.sent("POST", "/ableton/transport/stop-all")).toHaveLength(1));

    recommendationsAreMalformed = false;
    fireEvent.click(within(broken).getByRole("button", { name: "Reload" }));

    expect(await screen.findByRole("group", { name: RECOMMENDED.title })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop transport" })).toBeInTheDocument();
  });

  it("shows another screen when you leave the broken one", async () => {
    booth.show();
    await screen.findByRole("alert");

    fireEvent.click(screen.getByTitle("Set, Pipeline, Course"));
    press("Pipeline");

    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(screen.getByRole("tab", { name: "Pipeline" })).toHaveAttribute("aria-selected", "true");
  });
});
