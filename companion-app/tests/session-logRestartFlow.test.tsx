import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { renderApp, type BoothBackend } from "./boothHarness";
import {
  CLUB_MIX,
  NIGHT_DRIVE,
  SessionLog,
  boothWith,
  clips,
  liveNowPlays,
  liveStops,
  minutesAgo,
  playsLogged,
  preparedSetInLive,
  sessionsStarted,
} from "./session-logBackend";

const NO_SESSION = "No active session — auto-starts on first clip fire";

let backend: BoothBackend;
let log: SessionLog;

async function boothShowsThePreparedSet(): Promise<void> {
  renderApp();
  await screen.findByRole("button", { name: "Page 1" });
}

function playsLoggedTo(sessionId: number): unknown[] {
  return backend.sent("POST", `/api/v1/sessions/${sessionId}/plays`).map((sent) => sent.body);
}

function ended(sessionId: number): number {
  return backend.sent("POST", `/api/v1/sessions/${sessionId}/end`).length;
}

beforeEach(() => {
  log = new SessionLog();
  backend = boothWith(log, preparedSetInLive);
});

afterEach(async () => {
  await liveStops();
  backend.restore();
});

describe("the session after you end it", () => {
  it("stays ended until the next clip fires, then a new one starts and logs", async () => {
    log.open(minutesAgo(30), [{ track_id: NIGHT_DRIVE.track_id, played_at: minutesAgo(2) }]);
    await boothShowsThePreparedSet();
    await liveNowPlays(backend, clips("a", 0));
    await screen.findByText("1 play");

    fireEvent.click(screen.getByRole("button", { name: "end" }));
    await screen.findByTitle(NO_SESSION);
    await liveNowPlays(backend, clips("a", 0));
    expect(sessionsStarted(backend)).toBe(0);

    await liveNowPlays(backend, { ...clips("a", 0), ...clips("b", 0, ["drums"]) });
    await waitFor(() => expect(sessionsStarted(backend)).toBe(1));
    await waitFor(() => expect(playsLoggedTo(2)).toEqual([{ track_id: NIGHT_DRIVE.track_id }]));
    expect(await screen.findByText("1 play")).toBeInTheDocument();
    expect(playsLoggedTo(1)).toEqual([]);
  });
});

describe("a session that could not start", () => {
  it("is tried again when the next clip fires, not before", async () => {
    log.startsToRefuse = 1;
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("a", 0));
    await waitFor(() => expect(sessionsStarted(backend)).toBe(1));
    await liveNowPlays(backend, clips("a", 0));
    expect(sessionsStarted(backend)).toBe(1);
    expect(screen.getByTitle(NO_SESSION)).toBeInTheDocument();

    await liveNowPlays(backend, clips("a", 1));
    await waitFor(() => expect(sessionsStarted(backend)).toBe(2));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));
  });

  it("is asked for once when more clips fire while it is starting", async () => {
    log.holdsStarts = true;
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("a", 0, ["drums"]));
    await waitFor(() => expect(sessionsStarted(backend)).toBe(1));
    await liveNowPlays(backend, clips("a", 0, ["drums", "bass"]));
    await liveNowPlays(backend, clips("a", 0));
    log.letStartsThrough();

    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));
    await liveNowPlays(backend, clips("b", 0));
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, CLUB_MIX.track_id]),
    );
    expect(sessionsStarted(backend)).toBe(1);
  });

  it("is asked for once when a clip fires before the new session shows up", async () => {
    await boothShowsThePreparedSet();
    await screen.findByTitle(NO_SESSION);
    log.holdsReads = true;

    await liveNowPlays(backend, clips("a", 0, ["drums"]));
    await waitFor(() => expect(sessionsStarted(backend)).toBe(1));
    await liveNowPlays(backend, clips("a", 0));
    log.letReadsThrough();

    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));
    expect(sessionsStarted(backend)).toBe(1);
  });
});

describe("a session left open from earlier", () => {
  it("is ended when its last play was hours ago, and the next play starts a new one", async () => {
    log.open(minutesAgo(6 * 60), [{ track_id: NIGHT_DRIVE.track_id, played_at: minutesAgo(5 * 60) }]);
    await boothShowsThePreparedSet();

    await waitFor(() => expect(ended(1)).toBe(1));
    await liveNowPlays(backend, clips("b", 0));
    await waitFor(() => expect(sessionsStarted(backend)).toBe(1));
    await waitFor(() => expect(playsLoggedTo(2)).toEqual([{ track_id: CLUB_MIX.track_id }]));
    expect(ended(1)).toBe(1);
  });

  it("is ended when it is over an hour old and has no plays", async () => {
    log.open(minutesAgo(90));
    await boothShowsThePreparedSet();

    await waitFor(() => expect(ended(1)).toBe(1));
  });

  it("is asked to end once when ending it fails, and goes on taking plays", async () => {
    log.open(minutesAgo(6 * 60), [{ track_id: NIGHT_DRIVE.track_id, played_at: minutesAgo(5 * 60) }]);
    log.refusesEnds = true;
    await boothShowsThePreparedSet();
    await waitFor(() => expect(ended(1)).toBe(1));

    await liveNowPlays(backend, clips("b", 0, ["drums"]));
    await liveNowPlays(backend, clips("b", 0));
    await waitFor(() => expect(playsLoggedTo(1)).toEqual([{ track_id: CLUB_MIX.track_id }]));
    expect(ended(1)).toBe(1);
    expect(sessionsStarted(backend)).toBe(0);
  });

  it("is kept when its last play was minutes ago", async () => {
    log.open(minutesAgo(50), [{ track_id: NIGHT_DRIVE.track_id, played_at: minutesAgo(3) }]);
    await boothShowsThePreparedSet();
    await screen.findByText("1 play");

    await liveNowPlays(backend, clips("b", 0));
    await waitFor(() => expect(playsLoggedTo(1)).toEqual([{ track_id: CLUB_MIX.track_id }]));
    expect(ended(1)).toBe(0);
    expect(sessionsStarted(backend)).toBe(0);
  });
});
