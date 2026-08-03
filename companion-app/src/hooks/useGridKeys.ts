import { useEffect } from "react";

import {
  focusedTrackId,
  moveFocus,
  reconcileFocus,
  type GridFocus,
  type NavKey,
} from "../lib/gridNav";
import { onShapeChange, readShape } from "../lib/gridShape";
import { store, useAppStore } from "../store";

const NAV_KEYS: NavKey[] = [
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
];

function isNavKey(key: string): key is NavKey {
  return (NAV_KEYS as string[]).includes(key);
}

/**
 * Typing somewhere? Then these aren't navigation keys.
 *
 * Without this, ⌘K search, the set-rename field and the BPM inputs would all
 * lose their arrow keys to the grid.
 */
function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  const tag = el.tagName.toLowerCase();
  return (
    tag === "input" ||
    tag === "textarea" ||
    tag === "select" ||
    el.isContentEditable
  );
}

export interface GridKeyActions {
  /** space — audition the focused card in headphones (toggles). */
  onAudition: (trackId: number, focus: GridFocus) => void;
  /** enter / shift+enter — the commit verb. Plan: queue. Booth: load a deck. */
  onCommit: (trackId: number, focus: GridFocus, shift: boolean) => void;
  /** escape — stop the audition; a second press releases the cursor. */
  onEscape: () => void;
}

/**
 * The audition loop: ↑↓←→ to move, space to hear, enter to commit, esc to back out.
 *
 * Bound once at the grid, on ``window``, because the loop has to work without
 * the user first clicking a card — needing a click to "activate" the keyboard
 * would reintroduce the mouse trip this exists to remove.
 *
 * Moving does NOT auto-audition. It's tempting (walk the list, hear each one)
 * but every preview creates and fires a real clip in Live, so arrowing through
 * twenty candidates would machine-gun the Cue track. Space is one keystroke;
 * making it explicit costs nothing and keeps the audio predictable.
 */
export function useGridKeys(actions: GridKeyActions, enabled = true): void {
  const focus = useAppStore((s) => s.gridFocus);
  const commandBarOpen = useAppStore((s) => s.commandBarOpen);

  // Keep the cursor pointing at something real as lists change beneath it —
  // committing a rec shortens its list, and the next candidate should slide
  // under the cursor rather than the cursor jumping away.
  useEffect(() => {
    return onShapeChange(() => {
      const next = reconcileFocus(readShape(), store.peekGridFocus());
      store.setGridFocus(next);
    });
  }, []);

  useEffect(() => {
    if (!enabled) return;
    function onKey(e: KeyboardEvent) {
      // ⌘K owns the keyboard while it's open, and modified chords belong to
      // the browser (⌘←, ⌥→ …) — shift is the exception, it selects the deck.
      if (commandBarOpen || isTyping(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const shape = readShape();

      if (isNavKey(e.key)) {
        e.preventDefault(); // don't scroll the page out from under the grid
        store.setGridFocus(moveFocus(shape, focus, e.key));
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        actions.onEscape();
        return;
      }
      // Everything below acts on the focused card.
      const current = focus ?? null;
      const trackId = focusedTrackId(shape, current);
      if (current == null || trackId == null) return;

      if (e.key === " " || e.key === "Spacebar") {
        e.preventDefault(); // space scrolls otherwise
        actions.onAudition(trackId, current);
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        actions.onCommit(trackId, current, e.shiftKey);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled, focus, commandBarOpen, actions]);
}
