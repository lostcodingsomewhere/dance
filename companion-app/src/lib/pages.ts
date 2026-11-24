import type { DeckCell } from "../types";

export const PADS_PER_PAGE = 5;

export function pageOfScene(sceneIndex: number): number {
  return Math.floor(sceneIndex / PADS_PER_PAGE);
}

export function scenesOnPage(page: number): number[] {
  return Array.from({ length: PADS_PER_PAGE }, (_, row) => page * PADS_PER_PAGE + row);
}

export function pageCount(cells: readonly DeckCell[]): number {
  const lastScene = cells.reduce((last, cell) => Math.max(last, cell.scene_index), 0);
  return pageOfScene(lastScene) + 1;
}

export function pagesPlaying(
  columns: Record<string, number>,
  playingClips: Record<string, number>,
): Set<number> {
  const pages = new Set<number>();
  for (const trackIndex of Object.values(columns)) {
    const scene = playingClips[trackIndex];
    if (scene != null && scene >= 0) pages.add(pageOfScene(scene));
  }
  return pages;
}

export function songTitlesOnPage(cells: readonly DeckCell[], page: number): string {
  const titles: string[] = [];
  for (const side of ["a", "b"]) {
    const onSide = cells.find(
      (cell) =>
        pageOfScene(cell.scene_index) === page && cell.kind.endsWith(`_${side}`) && cell.title,
    );
    if (onSide) titles.push(`${side.toUpperCase()}: ${onSide.title}`);
  }
  return titles.join(" · ");
}

export function cueText(cell: DeckCell): string | null {
  if (cell.cue == null) return null;
  return cell.cue_label ? `${cell.cue} ${cell.cue_label}` : `${cell.cue}`;
}
