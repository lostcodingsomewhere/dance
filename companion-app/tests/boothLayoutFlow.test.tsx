import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { store } from "../src/store";
import {
  BoothBackend,
  LIVE_TRACK_OF,
  PADS_PER_PAGE,
  grid,
  renderApp,
  songOn,
  type Cell,
  type SeenRequest,
} from "./boothHarness";
import { API, SetsOnTheServer, rec, type Role, type Song } from "./setsFlowBackend";
import {
  HALF_LIGHT,
  LIBRARY,
  LOW_TIDE,
  NIGHT_DRIVE,
  SEVEN_PICKS,
  SLOW_BURN,
  picksIn,
  planned,
  recommended,
  recsIn,
  renderSetView,
} from "./setsFlowView";

const TWENTY_SONGS: Song[] = Array.from({ length: 20 }, (_, i) => ({
  track_id: 101 + i,
  title: `Song ${i + 1}`,
}));
const TEMPO_MATCH: Song = { track_id: 21, title: "Tempo Match" };
const FITS_THE_PLAN: Song = { track_id: 22, title: "Fits The Plan" };
const CLUB_MIX: Song = { track_id: 23, title: "Night Drive (Club Mix)" };
const CLUB_MIX_OF_NIGHT_DRIVE = {
  ...rec(CLUB_MIX),
  version_of: NIGHT_DRIVE.track_id,
  version_of_title: NIGHT_DRIVE.title,
  version_label: "Club Mix",
};
const BY_COLUMN = `${API}/recommend/by-column`;
const LOAD = `${API}/ableton/load-track`;
const PLAN_BOX = "[data-plan-scroll-box]";
const PREVIEW = "Preview in headphones (cue)";
const LOAD_A = "Load → Deck A";
const LOAD_B = "Load → Deck B";
const SMALLEST_TEXT_PX = 12;
const WHERE_EACH_SONG_STANDS: Record<number, { state: string; download_failed: boolean }> = {
  [NIGHT_DRIVE.track_id]: { state: "complete", download_failed: false },
  [HALF_LIGHT.track_id]: { state: "pending", download_failed: false },
  [LOW_TIDE.track_id]: { state: "error", download_failed: true },
  [SLOW_BURN.track_id]: { state: "error", download_failed: false },
};

interface PlanAnswer {
  queues: Record<string, { track_id: number }[]>;
}

function isAPlan(answer: unknown): answer is PlanAnswer {
  return typeof answer === "object" && answer !== null && "queues" in answer;
}

function withStates(answer: unknown): unknown {
  if (!isAPlan(answer)) return answer;
  const queues = Object.fromEntries(
    Object.entries(answer.queues).map(([role, items]) => [
      role,
      items.map((item) => ({ ...item, ...WHERE_EACH_SONG_STANDS[item.track_id] })),
    ]),
  );
  return { ...answer, queues };
}

function aBoothWith(server: SetsOnTheServer, inLive: Cell[] = []) {
  return (request: SeenRequest): unknown => {
    if (request.path === `${API}/ableton/decks`) return { columns: LIVE_TRACK_OF, cells: inLive };
    if (request.path === BY_COLUMN) {
      const column = (request.body as { column: string }).column;
      return { column, combo_size: 0, recs: [rec(TEMPO_MATCH)] };
    }
    if (request.method === "POST" && request.path === LOAD) {
      return {
        ok: true,
        scene_index: PADS_PER_PAGE,
        page: 1,
        side: (request.body as { side: string }).side,
        track_indices: LIVE_TRACK_OF,
        stems_loaded: 4,
        message: null,
        warnings: [],
      };
    }
    return server.answer(request);
  };
}

function pick(role: Role, title: string): HTMLElement {
  return within(planned(role)).getByRole("group", { name: title });
}

function recCard(role: Role, title: string): HTMLElement {
  return within(recommended(role)).getByRole("group", { name: title });
}

function planBox(role: Role): HTMLElement {
  const boxes = planned(role).querySelectorAll<HTMLElement>(PLAN_BOX);
  expect(boxes).toHaveLength(1);
  return boxes[0];
}

function buttonTitlesIn(card: HTMLElement): (string | null)[] {
  return within(card)
    .getAllByRole("button")
    .map((button) => button.getAttribute("title"));
}

function underTheCursor(): string | null {
  const ringed = document.querySelector(".ring-amber-300\\/80");
  return ringed?.querySelector("span.truncate")?.textContent ?? null;
}

function press(key: string): void {
  fireEvent.keyDown(document.body, { key });
}

function loadsSent(): unknown[] {
  return backend.sent("POST", LOAD).map((sent) => sent.body);
}

function classesOf(roots: HTMLElement[]): string[] {
  return roots.flatMap((root) =>
    [root, ...root.querySelectorAll<HTMLElement>("*")].flatMap((element) =>
      Array.from(element.classList),
    ),
  );
}

function textSizesIn(roots: HTMLElement[]): number[] {
  return classesOf(roots)
    .map((name) => /^text-\[([\d.]+)px\]$/.exec(name))
    .filter((size) => size != null)
    .map((size) => Number(size[1]));
}

let backend: BoothBackend;

afterEach(() => {
  backend.restore();
  store.setGridFocus(null);
  store.clearArm("a");
  store.clearArm("b");
  store.clearDecks();
  store.setBoothPage(0);
});

describe("the plan in the Booth", () => {
  it("keeps a 20-song column inside one capped box, and the arrow keys still reach pick 20", async () => {
    const server = new SetsOnTheServer([...TWENTY_SONGS, FITS_THE_PLAN]);
    server.has("Long night", { song: TWENTY_SONGS.map((song) => song.track_id) }, true);
    server.recs = () => [rec(FITS_THE_PLAN)];
    backend = new BoothBackend(aBoothWith(server));
    renderApp();
    await waitFor(() => expect(picksIn("song")).toEqual(TWENTY_SONGS.map((song) => song.title)));

    const box = planBox("song");
    const rows = within(box).getAllByRole("group");
    expect(rows).toHaveLength(20);
    expect(box).toHaveClass(
      "overflow-y-auto",
      "max-h-[122px]",
      "[@media(max-height:1000px)]:max-h-[82px]",
    );
    for (const row of rows) expect(row).toHaveClass("h-9", "shrink-0");
    expect(within(planned("song")).getByText("◆ Plan · 20")).toBeInTheDocument();
    expect(planned("song").parentElement).toHaveClass("items-start");

    for (const emptyColumn of ["drums", "bass", "vocals", "other"] as const) {
      expect(within(planned(emptyColumn)).getByText("nothing queued")).toHaveClass("h-9");
      expect(planned(emptyColumn).querySelector(PLAN_BOX)).toBeNull();
    }

    fireEvent.mouseDown(pick("song", "Song 1"));
    expect(underTheCursor()).toBe("Song 1");
    for (let step = 0; step < 19; step++) press("ArrowDown");
    expect(underTheCursor()).toBe("Song 20");

    press("Enter");
    await waitFor(() =>
      expect(loadsSent()).toEqual([{ track_id: 120, include_stems: true, kinds: null, side: "a", page: 0 }]),
    );

    await waitFor(() => expect(recsIn("song")).toEqual([FITS_THE_PLAN.title]));
    press("ArrowDown");
    expect(underTheCursor()).toBe(FITS_THE_PLAN.title);
  });

  it("gives a planned row preview and both decks, and nothing to remove or reorder", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    server.has("Friday night", SEVEN_PICKS, true);
    backend = new BoothBackend(aBoothWith(server));
    renderApp();
    await waitFor(() => expect(picksIn("song")).toHaveLength(3));

    const row = pick("song", LOW_TIDE.title);
    expect(row.querySelector("span.truncate")).toHaveTextContent(LOW_TIDE.title);
    expect(row.querySelector("span.truncate")).toHaveClass("text-base", "font-semibold");
    expect(within(row).getByText("2")).toBeInTheDocument();
    expect(row).toHaveClass("h-9", "border-l-emerald-500/70");
    expect(buttonTitlesIn(row)).toEqual([PREVIEW, LOAD_A, LOAD_B]);
    for (const control of within(row).getAllByRole("button")) {
      expect(control).toHaveClass("w-11", "h-7");
    }

    expect(screen.queryByTitle("Remove from plan")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Move .* (up|down)$/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Versions" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Pads" })).not.toBeInTheDocument();

    fireEvent.click(within(pick("drums", HALF_LIGHT.title)).getByTitle(LOAD_B));
    await waitFor(() =>
      expect(loadsSent()).toEqual([
        { track_id: HALF_LIGHT.track_id, include_stems: true, kinds: ["drums"], side: "b", page: 0 },
      ]),
    );
  });

  it("says why a pick is not ready in place of its preview and deck buttons", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    server.has(
      "Friday night",
      { song: [NIGHT_DRIVE, HALF_LIGHT, LOW_TIDE, SLOW_BURN].map((song) => song.track_id) },
      true,
    );
    const booth = aBoothWith(server);
    backend = new BoothBackend((request) => withStates(booth(request)));
    renderApp();
    await waitFor(() => expect(picksIn("song")).toHaveLength(4));

    const wordsFor: [Song, string][] = [
      [HALF_LIGHT, "on its way"],
      [LOW_TIDE, "download failed"],
      [SLOW_BURN, "failed, see Pipeline"],
    ];
    for (const [song, words] of wordsFor) {
      const row = pick("song", song.title);
      expect(within(row).getByText(words)).toHaveClass("text-amber-300");
      expect(within(row).queryAllByRole("button")).toEqual([]);
      expect(row).toHaveClass("h-9");
    }
    expect(buttonTitlesIn(pick("song", NIGHT_DRIVE.title))).toEqual([PREVIEW, LOAD_A, LOAD_B]);
    expect(planBox("song").querySelectorAll('[role="group"]')).toHaveLength(4);
  });

  it("is not drawn at all while nothing is planned, and appears with the first pick", async () => {
    const server = new SetsOnTheServer([...LIBRARY, TEMPO_MATCH]);
    const setId = server.has("New set", {}, true);
    backend = new BoothBackend(aBoothWith(server));
    renderApp();
    await screen.findByRole("group", { name: "Song recs" });
    await waitFor(() => expect(backend.sent("GET", `${API}/sets/${setId}/plan`)).not.toEqual([]));

    backend.liveNowPlays({});
    await waitFor(() => expect(recsIn("song")).toEqual([TEMPO_MATCH.title]));

    expect(screen.queryByRole("group", { name: /plan$/ })).not.toBeInTheDocument();
    expect(screen.queryByText(/◆ Plan/)).not.toBeInTheDocument();
    expect(screen.queryByText("nothing queued")).not.toBeInTheDocument();
    expect(document.querySelector("[data-plan-seam]")).toBeNull();

    fireEvent.click(within(recCard("song", TEMPO_MATCH.title)).getByTitle("Add to plan"));

    await waitFor(() => expect(picksIn("song")).toEqual([TEMPO_MATCH.title]));
    expect(screen.getAllByText("nothing queued")).toHaveLength(4);
    expect(document.querySelector("[data-plan-seam]")).not.toBeNull();
  });
});

describe("a recommendation in the Booth", () => {
  it("shows score, title, one line of chips, queue and both decks, and loads onto Deck A", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    server.has("Friday night", SEVEN_PICKS, true);
    server.recs = (role) =>
      role === "song" ? [CLUB_MIX_OF_NIGHT_DRIVE, rec(FITS_THE_PLAN)] : [rec(FITS_THE_PLAN)];
    backend = new BoothBackend(aBoothWith(server, songOn(0, "a", NIGHT_DRIVE)));
    renderApp();
    await waitFor(() => expect(recsIn("song")).toEqual([CLUB_MIX.title, FITS_THE_PLAN.title]));

    const card = recCard("song", FITS_THE_PLAN.title);
    expect(card.querySelector("span.truncate")).toHaveTextContent(FITS_THE_PLAN.title);
    expect(card.querySelector("span.truncate")).toHaveClass("text-base", "font-semibold");
    expect(within(card).getByText("82")).toBeInTheDocument();
    expect(within(card).getByText("128 · 8A · E6 · Test Artist")).toHaveClass("text-[13px]");
    const chips = within(card).getByTitle("Vibe (CLAP embedding): 90").parentElement!;
    expect(chips).toHaveClass("h-[22px]", "flex-wrap", "overflow-hidden");
    expect(buttonTitlesIn(card)).toEqual([PREVIEW, "Add to plan", LOAD_A, LOAD_B]);
    for (const control of within(card).getAllByRole("button")) {
      expect(control).toHaveClass("w-11", "h-7");
    }
    expect(card.parentElement).toHaveClass("min-h-[396px]");

    expect(card).toHaveClass("h-24");
    fireEvent.mouseDown(card);
    expect(card).toHaveClass("h-24", "ring-2", "ring-amber-300/80");

    const version = recCard("song", CLUB_MIX.title);
    expect(version).toHaveClass("h-[120px]");
    expect(within(version).getByText("version")).toBeInTheDocument();
    expect(within(version).getByText("Club Mix of Night Drive")).toBeInTheDocument();
    expect(
      within(version).getByTitle("Load → Deck B, the deck beside its original"),
    ).toHaveClass("ring-2");
    expect(within(version).getByTitle(LOAD_A)).not.toHaveClass("ring-2");

    fireEvent.click(within(card).getByTitle(LOAD_A));
    await waitFor(() =>
      expect(loadsSent()).toEqual([
        { track_id: FITS_THE_PLAN.track_id, include_stems: true, kinds: null, side: "a", page: 0 },
      ]),
    );
  });
});

describe("the Set view", () => {
  it("still removes and reorders picks, and offers no decks", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    server.has("Friday night", SEVEN_PICKS, true);
    server.recs = () => [rec(FITS_THE_PLAN)];
    backend = new BoothBackend(server.answer);
    renderSetView();
    await waitFor(() => expect(recsIn("song")).toEqual([FITS_THE_PLAN.title]));

    expect(screen.getAllByTitle("Remove from plan")).toHaveLength(7);
    expect(screen.getByRole("button", { name: "Move Low Tide up" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Move Low Tide down" })).toBeInTheDocument();
    expect(within(pick("song", LOW_TIDE.title)).getByRole("button", { name: "Versions" })).toBeInTheDocument();
    expect(within(pick("song", LOW_TIDE.title)).getByRole("button", { name: "Pads" })).toBeInTheDocument();
    expect(screen.queryByTitle(LOAD_A)).not.toBeInTheDocument();
    expect(screen.queryByTitle(LOAD_B)).not.toBeInTheDocument();

    expect(document.querySelector(PLAN_BOX)).toBeNull();
    expect(planned("song").parentElement).toHaveClass("items-stretch");
    expect(within(planned("vocals")).getByText("nothing queued")).toHaveClass("border-dashed");
    expect(pick("song", LOW_TIDE.title)).not.toHaveClass("h-9");

    const card = recCard("song", FITS_THE_PLAN.title);
    expect(card).not.toHaveClass("h-24");
    expect(within(card).getByText("Test Artist")).toBeInTheDocument();
    expect(within(card).getByTitle("Vibe (CLAP embedding): 90").parentElement).not.toHaveClass(
      "overflow-hidden",
    );
  });
});

describe("the pads", () => {
  function twoSongsWithABorrowedBassAndAHandDroppedClip(): Cell[] {
    const cells = [...songOn(0, "a", NIGHT_DRIVE), ...songOn(0, "b", CLUB_MIX)];
    for (const cell of cells) {
      if (cell.kind === "bass_b" && (cell.scene_index === 2 || cell.scene_index === 3)) {
        Object.assign(cell, HALF_LIGHT);
      }
      if (cell.kind === "other_b" && cell.scene_index === 3) {
        Object.assign(cell, { cue: null, cue_label: null, cue_silent: false });
      }
    }
    return cells;
  }

  it("name a song once per run in a column, lead with the cue, and keep their tooltips", async () => {
    backend = new BoothBackend(({ path }) =>
      path === `${API}/ableton/decks`
        ? { columns: LIVE_TRACK_OF, cells: twoSongsWithABorrowedBassAndAHandDroppedClip() }
        : undefined,
    );
    renderApp();
    await screen.findByRole("button", { name: "Page 1" });

    const pads = within(grid());
    expect(pads.getAllByText(NIGHT_DRIVE.title)).toHaveLength(4);
    expect(pads.getAllByText(CLUB_MIX.title)).toHaveLength(6);
    expect(pads.getAllByText(HALF_LIGHT.title)).toHaveLength(1);
    expect(pads.getAllByText("3 drop")).toHaveLength(8);
    expect(pads.getAllByText("4 breakdown")).toHaveLength(7);

    const firstPad = pads.getByTitle("Drums A: Night Drive — Test Artist — cue 1, intro — tap to fire");
    expect(firstPad).toHaveClass("h-9");
    expect(within(firstPad).getByText("1 intro")).toHaveClass("text-base", "font-bold");
    expect(within(firstPad).getByText(NIGHT_DRIVE.title)).toHaveClass("text-[13px]");
    expect(within(firstPad).getByText(NIGHT_DRIVE.title)).toHaveAttribute(
      "title",
      "128.0 · 8A · E6 · 4:00",
    );

    const sameSongBelow = pads.getByTitle(
      "Drums A: Night Drive — Test Artist — cue 2, buildup — tap to fire",
    );
    expect(sameSongBelow).toHaveClass("h-9");
    expect(within(sameSongBelow).getByText("2 buildup")).toBeInTheDocument();
    expect(within(sameSongBelow).queryByText(NIGHT_DRIVE.title)).not.toBeInTheDocument();

    const borrowed = pads.getByTitle("Bass B: Half Light — Test Artist — cue 3, drop — tap to fire");
    expect(within(borrowed).getByText(HALF_LIGHT.title)).toBeInTheDocument();
    const borrowedAgain = pads.getByTitle(
      "Bass B: Half Light — Test Artist — cue 4, breakdown — tap to fire",
    );
    expect(within(borrowedAgain).queryByText(HALF_LIGHT.title)).not.toBeInTheDocument();
    const droppedInByHand = pads.getByTitle(
      "Melody B: Night Drive (Club Mix) — Test Artist — tap to fire",
    );
    expect(within(droppedInByHand).getByText(CLUB_MIX.title)).toBeInTheDocument();

    expect(pads.queryByText("Test Artist")).not.toBeInTheDocument();
    expect(pads.queryByText(/128\.0/)).not.toBeInTheDocument();
    expect(pads.queryByText(/4:00/)).not.toBeInTheDocument();

    const remove = within(firstPad.parentElement!).getByRole("button", {
      name: "Remove Drums clip from scene",
    });
    expect(remove).toHaveClass("h-4", "w-4");
    fireEvent.click(remove);
    expect(
      within(firstPad.parentElement!).getByRole("button", {
        name: "Press again to remove Drums clip from scene",
      }),
    ).toHaveClass("h-4");
  });
});

describe("text in the Booth", () => {
  it("is never under 12 px or dim grey in the plan grid, the pads and the column headers", async () => {
    const server = new SetsOnTheServer(LIBRARY);
    server.has(
      "Friday night",
      { ...SEVEN_PICKS, song: [NIGHT_DRIVE, HALF_LIGHT, LOW_TIDE].map((song) => song.track_id) },
      true,
    );
    server.recs = (role) =>
      role === "song" ? [CLUB_MIX_OF_NIGHT_DRIVE, rec(FITS_THE_PLAN)] : role === "bass" ? [] : [rec(FITS_THE_PLAN)];
    const booth = aBoothWith(server, songOn(0, "a", NIGHT_DRIVE));
    backend = new BoothBackend((request) => withStates(booth(request)));
    renderApp();
    await waitFor(() => expect(recsIn("song")).toEqual([CLUB_MIX.title, FITS_THE_PLAN.title]));
    await screen.findByRole("button", { name: "Page 1" });

    const planGrid = document.querySelector<HTMLElement>("[data-plan-grid]")!;
    const columnHeaders = screen.getByTestId("booth-column-headers");
    const pads = [
      ...within(grid()).getAllByTitle(/tap to fire$/),
      ...within(grid()).getAllByLabelText(/\(empty\)$/),
    ];
    expect(within(planGrid).getByText("nothing queued")).toHaveClass("text-neutral-500");
    expect(within(planGrid).getByText("no candidates")).toHaveClass("text-neutral-500");
    expect(within(planGrid).getAllByText("download failed")).toHaveLength(2);
    expect(within(planGrid).getByText("move").parentElement!.parentElement).toHaveClass(
      "text-xs",
      "text-neutral-400",
    );
    expect(within(columnHeaders).getAllByText("Drums")).toHaveLength(2);
    expect(pads).toHaveLength(8 * PADS_PER_PAGE);

    const read = [planGrid, columnHeaders, ...pads];
    expect(textSizesIn(read).filter((px) => px < SMALLEST_TEXT_PX)).toEqual([]);
    expect(classesOf(read).filter((name) => /^text-neutral-[67]00/.test(name))).toEqual([]);
  });
});
