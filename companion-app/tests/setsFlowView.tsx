import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";

import { SetEditor } from "../src/views/SetEditor";
import { ROLES, type Plan, type Role, type Song } from "./setsFlowBackend";

export const NIGHT_DRIVE: Song = { track_id: 7, title: "Night Drive" };
export const HALF_LIGHT: Song = { track_id: 9, title: "Half Light" };
export const LOW_TIDE: Song = { track_id: 3, title: "Low Tide" };
export const SLOW_BURN: Song = { track_id: 5, title: "Slow Burn" };
export const LIBRARY = [NIGHT_DRIVE, HALF_LIGHT, LOW_TIDE, SLOW_BURN];

export const SEVEN_PICKS: Plan = {
  drums: [HALF_LIGHT.track_id, NIGHT_DRIVE.track_id],
  bass: [LOW_TIDE.track_id],
  vocals: [],
  other: [SLOW_BURN.track_id],
  song: [NIGHT_DRIVE.track_id, LOW_TIDE.track_id, HALF_LIGHT.track_id],
};

export const COLUMN_LABEL: Record<Role, string> = {
  drums: "Drums",
  bass: "Bass",
  vocals: "Vocals",
  other: "Melody",
  song: "Song",
};

function titleOf(trackId: number): string {
  return LIBRARY.find((song) => song.track_id === trackId)!.title;
}

function cardTitles(band: HTMLElement): string[] {
  return within(band)
    .queryAllByRole("group")
    .map((card) => card.getAttribute("aria-label") ?? "");
}

function byRole<T>(read: (role: Role) => T): Record<Role, T> {
  return Object.fromEntries(ROLES.map((role) => [role, read(role)])) as Record<Role, T>;
}

export function renderSetView() {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <SetEditor />
    </QueryClientProvider>,
  );
}

export function planned(role: Role): HTMLElement {
  return screen.getByRole("group", { name: `${COLUMN_LABEL[role]} plan` });
}

export function recommended(role: Role): HTMLElement {
  return screen.getByRole("group", { name: `${COLUMN_LABEL[role]} recs` });
}

export function picksIn(role: Role): string[] {
  return cardTitles(planned(role));
}

export function recsIn(role: Role): string[] {
  return cardTitles(recommended(role));
}

export function picksOnScreen(): Record<Role, string[]> {
  return byRole(picksIn);
}

export function titlesOf(plan: Plan): Record<Role, string[]> {
  return byRole((role) => plan[role].map(titleOf));
}
