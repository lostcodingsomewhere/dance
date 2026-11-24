import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderApp, type BoothBackend } from "./boothHarness";
import {
  CLUB_MIX,
  HALF_LIGHT,
  NIGHT_DRIVE,
  SessionLog,
  boothWith,
  minutesAgo,
  preparedSetInLive,
} from "./session-logBackend";

const UNTITLED = { track_id: 12, title: null };
const TEN_PLAYS: Array<[{ track_id: number; title: string | null }, number]> = [
  [NIGHT_DRIVE, 58],
  [CLUB_MIX, 51],
  [HALF_LIGHT, 44],
  [NIGHT_DRIVE, 37],
  [CLUB_MIX, 30],
  [HALF_LIGHT, 23],
  [UNTITLED, 16],
  [NIGHT_DRIVE, 9],
  [CLUB_MIX, 2],
  [HALF_LIGHT, 0],
];
const LAST_EIGHT_NEWEST_FIRST = [
  ["Half Light", "just now"],
  ["Night Drive (Club Mix)", "2 min ago"],
  ["Night Drive", "9 min ago"],
  ["Track #12", "16 min ago"],
  ["Half Light", "23 min ago"],
  ["Night Drive (Club Mix)", "30 min ago"],
  ["Night Drive", "37 min ago"],
  ["Half Light", "44 min ago"],
];

let backend: BoothBackend;
let log: SessionLog;

function recentPlays(): HTMLElement | null {
  return screen.queryByRole("list", { name: "Recent plays" });
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  log = new SessionLog();
  backend = boothWith(log, preparedSetInLive);
});

afterEach(() => {
  backend.restore();
  vi.useRealTimers();
});

describe("the session chip's play count", () => {
  it("opens the last eight plays, newest first, with how long ago each was", async () => {
    const session = log.open(
      minutesAgo(70),
      TEN_PLAYS.map(([song, ago]) => ({ track_id: song.track_id, played_at: minutesAgo(ago) })),
    );
    session.name = "Friday night";
    renderApp();

    const count = await screen.findByRole("button", { name: "10 plays" });
    expect(count).toHaveAttribute("aria-expanded", "false");
    expect(recentPlays()).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "end" })).toBeInTheDocument();

    fireEvent.click(count);
    expect(count).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Friday night")).toBeInTheDocument();
    const rows = within(recentPlays()!).getAllByRole("listitem");
    expect(rows).toHaveLength(LAST_EIGHT_NEWEST_FIRST.length);
    LAST_EIGHT_NEWEST_FIRST.forEach(([title, ago], row) => {
      expect(within(rows[row]).getByText(title)).toBeInTheDocument();
      expect(within(rows[row]).getByText(ago)).toBeInTheDocument();
    });

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(recentPlays()).not.toBeInTheDocument();
    expect(count).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(count);
    fireEvent.mouseDown(recentPlays()!);
    expect(recentPlays()).toBeInTheDocument();
    fireEvent.mouseDown(screen.getByRole("button", { name: "Page 1" }));
    expect(recentPlays()).not.toBeInTheDocument();

    fireEvent.click(count);
    expect(recentPlays()).toBeInTheDocument();
    fireEvent.click(count);
    expect(recentPlays()).not.toBeInTheDocument();
  });

  it("says so when nothing has played yet", async () => {
    log.open(minutesAgo(5));
    renderApp();

    fireEvent.click(await screen.findByRole("button", { name: "0 plays" }));

    expect(screen.getByText("No plays yet.")).toBeInTheDocument();
    expect(recentPlays()).not.toBeInTheDocument();
  });
});
