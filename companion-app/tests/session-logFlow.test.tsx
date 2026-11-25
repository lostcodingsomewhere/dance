import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderApp, type BoothBackend } from "./boothHarness";
import {
  CLUB_MIX,
  HALF_LIGHT,
  NIGHT_DRIVE,
  SessionLog,
  boothWith,
  clips,
  decksThisBrowserLoaded,
  journeysAsked,
  liveNowPlays,
  liveStops,
  playsLogged,
  preparedSetInLive,
  secondsPass,
  sessionsStarted,
} from "./session-logBackend";

const CUE_3 = 2;
const PAGE_2_CUE_1 = 5;
const STEMS = ["drums", "bass", "vocals", "other"];

let backend: BoothBackend;
let log: SessionLog;

async function boothShowsThePreparedSet(): Promise<void> {
  renderApp();
  await screen.findByRole("button", { name: "Page 1" });
}

beforeEach(() => {
  log = new SessionLog();
  backend = boothWith(log, preparedSetInLive);
});

afterEach(async () => {
  await liveStops();
  backend.restore();
  vi.useRealTimers();
});

describe("a prepared Set opened in Live from a file", () => {
  it("logs the song that plays though nothing was loaded from this browser", async () => {
    await boothShowsThePreparedSet();
    expect(decksThisBrowserLoaded()).toEqual({});

    await liveNowPlays(backend, clips("a", CUE_3));

    await waitFor(() => expect(sessionsStarted(backend)).toBe(1));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));
    expect(await screen.findByText("1 play")).toBeInTheDocument();
    expect(screen.getByTitle("Energy arc — 1 play so far")).toBeInTheDocument();
    await waitFor(() => expect(journeysAsked(backend)).toContainEqual([NIGHT_DRIVE.track_id]));
  });

  it("takes the song with the most stems playing when two play at once", async () => {
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, {
      ...clips("a", 0, ["drums"]),
      ...clips("b", 0, ["bass", "vocals", "other"]),
    });
    await waitFor(() => expect(playsLogged(backend)).toEqual([CLUB_MIX.track_id]));

    await liveNowPlays(backend, {
      ...clips("a", 0, ["drums", "bass", "vocals"]),
      ...clips("b", 0, ["other"]),
    });
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([CLUB_MIX.track_id, NIGHT_DRIVE.track_id]),
    );
  });

  it("counts stems, not the mix clip that plays along with them", async () => {
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("a", 0, [...STEMS, "mix"]));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));

    await liveNowPlays(backend, {
      ...clips("a", 0, [...STEMS, "mix"]),
      ...clips("b", 1, STEMS),
    });
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, CLUB_MIX.track_id]),
    );
  });

  it("logs a song that plays from its mix clip alone", async () => {
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("b", 0, ["mix"]));
    await waitFor(() => expect(playsLogged(backend)).toEqual([CLUB_MIX.track_id]));

    await liveNowPlays(backend, { ...clips("b", 0, ["mix"]), ...clips("a", 0, ["drums"]) });
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([CLUB_MIX.track_id, NIGHT_DRIVE.track_id]),
    );
  });

  it("takes the song that started later when both have as many stems playing", async () => {
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("b", 1, ["drums", "bass"]));
    await waitFor(() => expect(playsLogged(backend)).toEqual([CLUB_MIX.track_id]));

    await liveNowPlays(backend, {
      ...clips("b", 1, ["drums", "bass"]),
      ...clips("a", 6, ["vocals", "other"]),
    });
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([CLUB_MIX.track_id, HALF_LIGHT.track_id]),
    );
  });

  it("takes the song on the lower scene when it cannot tell which started later", async () => {
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, {
      ...clips("a", 6, ["drums", "bass"]),
      ...clips("b", 1, ["vocals", "other"]),
    });
    await waitFor(() => expect(playsLogged(backend)).toEqual([CLUB_MIX.track_id]));

    await liveNowPlays(backend, clips("a", CUE_3));
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([CLUB_MIX.track_id, NIGHT_DRIVE.track_id]),
    );
  });

  it("logs a song once however long it keeps playing", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("a", CUE_3));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));

    vi.setSystemTime(Date.now() + 4 * 60_000);
    await liveNowPlays(backend, clips("a", CUE_3 + 1));
    await liveNowPlays(backend, clips("b", 0));
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, CLUB_MIX.track_id]),
    );
  });

  it("logs each song once while their stems are swapped back and forth", async () => {
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("a", 0, STEMS));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));

    await liveNowPlays(backend, {
      ...clips("a", 0, ["vocals", "other"]),
      ...clips("b", 0, ["drums", "bass"]),
    });
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, CLUB_MIX.track_id]),
    );

    await liveNowPlays(backend, {
      ...clips("a", 0, ["bass", "vocals", "other"]),
      ...clips("b", 0, ["drums"]),
    });
    await liveNowPlays(backend, {
      ...clips("a", 0, ["other"]),
      ...clips("b", 0, ["drums", "bass", "vocals"]),
    });
    await liveNowPlays(backend, clips("a", PAGE_2_CUE_1, STEMS));
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([
        NIGHT_DRIVE.track_id,
        CLUB_MIX.track_id,
        HALF_LIGHT.track_id,
      ]),
    );
  });

  it("keeps a song's play open through a long blend in which it never stops", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("a", 0, STEMS));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));
    await liveNowPlays(backend, { ...clips("a", 0, STEMS), ...clips("b", 0, STEMS) });
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, CLUB_MIX.track_id]),
    );

    vi.setSystemTime(Date.now() + 3 * 60_000);
    await liveNowPlays(backend, {
      ...clips("a", 0, STEMS),
      ...clips("b", 0, ["drums", "vocals", "other"]),
    });
    await liveNowPlays(backend, {
      ...clips("a", 0, ["drums", "vocals", "other"]),
      ...clips("b", 0, STEMS),
    });
    await screen.findByText("2 plays");
    expect(log.current?.plays.at(-1)?.track_id).toBe(CLUB_MIX.track_id);

    await liveNowPlays(backend, clips("a", PAGE_2_CUE_1, STEMS));
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([
        NIGHT_DRIVE.track_id,
        CLUB_MIX.track_id,
        HALF_LIGHT.track_id,
      ]),
    );
  });

  it("logs a song again when it comes back later in the set", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("a", 0, STEMS));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));
    await liveNowPlays(backend, clips("b", 0, STEMS));
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, CLUB_MIX.track_id]),
    );

    vi.setSystemTime(Date.now() + 3 * 60_000);
    await liveNowPlays(backend, clips("a", 0, STEMS));
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([
        NIGHT_DRIVE.track_id,
        CLUB_MIX.track_id,
        NIGHT_DRIVE.track_id,
      ]),
    );
  });

  it("does not log the playing song again when you leave the Booth and come back", async () => {
    await boothShowsThePreparedSet();
    await liveNowPlays(backend, clips("a", CUE_3));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));
    await screen.findByText("1 play");

    fireEvent.click(screen.getByRole("tab", { name: "Set" }));
    await screen.findByText("No active set yet");
    fireEvent.click(screen.getByRole("tab", { name: "Booth" }));
    await screen.findByRole("button", { name: "Page 1, playing" });

    await liveNowPlays(backend, clips("b", 0));
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, CLUB_MIX.track_id]),
    );
  });
});

describe("a play the backend refuses", () => {
  it("is tried again a few seconds later while the song still leads, then left alone", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"], shouldAdvanceTime: true });
    log.playsToRefuse = 1;
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("a", CUE_3));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));
    expect(await screen.findByText("0 plays")).toBeInTheDocument();
    await liveNowPlays(backend, clips("a", CUE_3 + 1));
    expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]);

    await secondsPass(6);
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, NIGHT_DRIVE.track_id]),
    );
    expect(await screen.findByText("1 play")).toBeInTheDocument();
    expect(screen.getByTitle("Energy arc — 1 play so far")).toBeInTheDocument();

    await secondsPass(30);
    expect(log.current?.plays.map((play) => play.track_id)).toEqual([NIGHT_DRIVE.track_id]);
    expect(playsLogged(backend)).toHaveLength(2);
  });

  it("is not tried again for a song that no longer leads", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"], shouldAdvanceTime: true });
    log.playsToRefuse = 1;
    await boothShowsThePreparedSet();

    await liveNowPlays(backend, clips("a", CUE_3));
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));
    await liveNowPlays(backend, clips("b", 0));
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, CLUB_MIX.track_id]),
    );

    await secondsPass(30);
    expect(log.current?.plays.map((play) => play.track_id)).toEqual([CLUB_MIX.track_id]);
    expect(playsLogged(backend)).toHaveLength(2);
  });
});
