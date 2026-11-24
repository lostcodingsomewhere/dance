import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { SpotifyVersion } from "../src/types";
import { SetEditor } from "../src/views/SetEditor";
import { BoothBackend, type SeenRequest } from "./boothHarness";

const SET_ID = 1;
const ORIGINAL = { track_id: 7, title: "Night Drive" };
const DRUMS_PICK = { track_id: 9, title: "Half Light" };
const NEW_TRACK_ID = 31;

function onSpotify(
  version: Partial<SpotifyVersion> & Pick<SpotifyVersion, "spotify_id" | "title" | "label">,
): SpotifyVersion {
  return {
    artist: "Test Artist",
    album: "Night Drive",
    duration_ms: 210_000,
    image_url: null,
    explicit: false,
    library_track_id: null,
    library_state: null,
    download_failed: false,
    ...version,
  };
}

const CLUB_MIX = onSpotify({
  spotify_id: "sp-club",
  title: "Night Drive - Club Mix",
  label: "Club Mix",
  library_track_id: 8,
  library_state: "complete",
});
const REMIX = onSpotify({
  spotify_id: "sp-remix",
  title: "Night Drive - Tinlicker Remix",
  artist: "Test Artist, Tinlicker",
  label: "Tinlicker Remix",
});
const LONG_CUT = onSpotify({
  spotify_id: "sp-long",
  title: "Night Drive",
  label: "longer cut",
  duration_ms: 375_000,
  library_track_id: 12,
  library_state: "pending",
});

function planItem(song: { track_id: number; title: string }) {
  return { ...song, artist: "Test Artist", key_camelot: "8A", bpm: 128, floor_energy: 6 };
}

function aSetWithOneSongPlanned(spotifyAnswers: (request: SeenRequest) => unknown) {
  let remixIsOnItsWay = false;
  return (request: SeenRequest): unknown => {
    const { path, method } = request;
    const song = remixIsOnItsWay
      ? [planItem(ORIGINAL), planItem({ track_id: NEW_TRACK_ID, title: REMIX.title })]
      : [planItem(ORIGINAL)];
    const plan = {
      set_id: SET_ID,
      queues: { drums: [planItem(DRUMS_PICK)], bass: [], vocals: [], other: [], song },
    };
    if (path === "/api/v1/sets/active" || path === `/api/v1/sets/${SET_ID}`) {
      return {
        id: SET_ID,
        name: "Friday night",
        notes: null,
        is_active: true,
        created_at: "2026-11-01T00:00:00",
        updated_at: "2026-11-01T00:00:00",
        tracks: [],
      };
    }
    if (path === "/api/v1/sets") return [];
    if (path === `/api/v1/sets/${SET_ID}/plan`) return plan;
    if (path === `/api/v1/sets/${SET_ID}/plan/append`) {
      remixIsOnItsWay = true;
      return {
        ...plan,
        queues: {
          ...plan.queues,
          song: [planItem(ORIGINAL), planItem({ track_id: NEW_TRACK_ID, title: REMIX.title })],
        },
      };
    }
    if (path === `/api/v1/sets/${SET_ID}/plan-recs`) return { column: "mix", combo_size: 0, recs: [] };
    if (path === "/api/v1/pipeline/ingest/track" && method === "POST") {
      return { track_id: NEW_TRACK_ID, state: "pending", job_id: "job-1", already_existed: false };
    }
    if (path === "/api/v1/spotify/versions") {
      if (!remixIsOnItsWay) return spotifyAnswers(request);
      const queued = { ...REMIX, library_track_id: NEW_TRACK_ID, library_state: "pending" };
      return { track_id: ORIGINAL.track_id, query: "q", versions: [CLUB_MIX, queued, LONG_CUT] };
    }
    return undefined;
  };
}

function threeVersions(): unknown {
  return {
    track_id: ORIGINAL.track_id,
    query: 'track:"Night Drive" artist:"Test Artist"',
    versions: [CLUB_MIX, REMIX, LONG_CUT],
  };
}

function failure(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "content-type": "application/json" },
  });
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

function row(title: string): HTMLElement {
  return screen.getByRole("group", { name: `Version on Spotify: ${title}` });
}

let backend: BoothBackend;

afterEach(() => {
  backend.restore();
});

describe("finding a planned song's versions on Spotify", () => {
  it("lists them, says which you have, and gets one into the plan after its original", async () => {
    backend = new BoothBackend(aSetWithOneSongPlanned(threeVersions));
    renderSetView();

    const planned = await screen.findByRole("group", { name: ORIGINAL.title });
    const drumsPick = screen.getByRole("group", { name: DRUMS_PICK.title });
    expect(within(drumsPick).queryByRole("button", { name: "Versions" })).not.toBeInTheDocument();
    fireEvent.click(within(planned).getByRole("button", { name: "Versions" }));

    const dialog = await screen.findByRole("dialog", { name: "Versions · Night Drive" });
    await within(dialog).findByRole("group", { name: `Version on Spotify: ${REMIX.title}` });
    expect(backend.sent("GET", "/api/v1/spotify/versions").map((sent) => sent.query)).toEqual([
      "track_id=7",
    ]);
    expect(within(row(CLUB_MIX.title)).getByText("Club Mix · 3:30")).toBeInTheDocument();
    expect(within(row(CLUB_MIX.title)).getByText("in your library")).toBeInTheDocument();
    expect(within(row(CLUB_MIX.title)).queryByRole("button")).not.toBeInTheDocument();
    expect(within(row(LONG_CUT.title)).getByText("longer cut · 6:15")).toBeInTheDocument();
    expect(within(row(LONG_CUT.title)).getByText("on its way")).toBeInTheDocument();
    expect(within(row(REMIX.title)).getByText("Tinlicker Remix · 3:30")).toBeInTheDocument();
    expect(within(dialog).getByText(/track:"Night Drive" artist:"Test Artist"/)).toBeInTheDocument();

    fireEvent.click(within(row(REMIX.title)).getByRole("button", { name: "Get" }));

    await waitFor(() =>
      expect(backend.sent("POST", `/api/v1/sets/${SET_ID}/plan/append`).map((s) => s.body)).toEqual(
        [{ role: "song", track_id: NEW_TRACK_ID, after_track_id: ORIGINAL.track_id }],
      ),
    );
    expect(backend.sent("POST", "/api/v1/pipeline/ingest/track").map((s) => s.body)).toEqual([
      {
        spotify_id: "sp-remix",
        title: REMIX.title,
        artist: "Test Artist, Tinlicker",
        duration_ms: 210_000,
      },
    ]);
    await waitFor(() =>
      expect(within(row(REMIX.title)).getByText("on its way")).toBeInTheDocument(),
    );
    expect(within(row(REMIX.title)).queryByRole("button", { name: "Get" })).not.toBeInTheDocument();
    expect(await screen.findByRole("group", { name: REMIX.title })).toBeInTheDocument();
  });

  it("offers a version whose download failed again", async () => {
    const refused = onSpotify({
      spotify_id: "sp-remix",
      title: REMIX.title,
      label: "Tinlicker Remix",
      library_track_id: NEW_TRACK_ID,
      library_state: "error",
      download_failed: true,
    });
    const brokenLater = onSpotify({
      spotify_id: "sp-long",
      title: "Night Drive",
      label: "longer cut",
      library_track_id: 12,
      library_state: "error",
    });
    backend = new BoothBackend(
      aSetWithOneSongPlanned(() => ({ track_id: 7, query: "q", versions: [refused, brokenLater] })),
    );
    renderSetView();
    const planned = await screen.findByRole("group", { name: ORIGINAL.title });
    fireEvent.click(within(planned).getByRole("button", { name: "Versions" }));
    await screen.findByRole("group", { name: `Version on Spotify: ${REMIX.title}` });

    expect(within(row(REMIX.title)).getByText("download failed")).toBeInTheDocument();
    expect(within(row("Night Drive")).getByText("failed, see Pipeline")).toBeInTheDocument();
    expect(within(row("Night Drive")).queryByRole("button")).not.toBeInTheDocument();

    fireEvent.click(within(row(REMIX.title)).getByRole("button", { name: "Try again" }));

    await waitFor(() =>
      expect(backend.sent("POST", "/api/v1/pipeline/ingest/track").map((s) => s.body)).toEqual([
        { spotify_id: "sp-remix", title: REMIX.title, artist: "Test Artist", duration_ms: 210_000 },
      ]),
    );
  });

  it("says so when Spotify has no other version, and why when the search cannot run", async () => {
    let answer: () => unknown = () => ({ track_id: 7, query: "q", versions: [] });
    backend = new BoothBackend(aSetWithOneSongPlanned(() => answer()));
    renderSetView();
    const planned = await screen.findByRole("group", { name: ORIGINAL.title });

    fireEvent.click(within(planned).getByRole("button", { name: "Versions" }));
    const dialog = await screen.findByRole("dialog", { name: "Versions · Night Drive" });
    expect(
      await within(dialog).findByText("Spotify lists no other version of this song."),
    ).toBeInTheDocument();

    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));
    answer = () => failure(503, "Spotify search is not configured. Add DANCE_SPOTIFY_CLIENT_ID.");
    fireEvent.click(within(planned).getByRole("button", { name: "Versions" }));
    expect(
      await screen.findByText("Spotify search is not configured. Add DANCE_SPOTIFY_CLIENT_ID."),
    ).toBeInTheDocument();
  });
});
