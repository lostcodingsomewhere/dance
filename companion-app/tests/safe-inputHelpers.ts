import { act, fireEvent } from "@testing-library/react";

// jsdom sends no key to the focused control by itself, and never turns Enter or Space into a click.
export function press(key: string, init: KeyboardEventInit = {}): void {
  const focused = (document.activeElement ?? document.body) as HTMLElement;
  const leftToTheBrowser = fireEvent.keyDown(focused, { key, ...init });
  const activates = key === "Enter" || key === " ";
  if (leftToTheBrowser && activates && focused.matches("button")) fireEvent.click(focused);
}

export function tabTo(control: HTMLElement): void {
  press("Tab");
  control.focus();
}

const TAKES_FOCUS_ON_CLICK = "button, a[href], input, select, textarea, [tabindex]";

// A mouse press moves focus to the nearest thing that can take it, or off whatever had it.
export function clickWithTheMouse(target: HTMLElement): void {
  fireEvent.mouseDown(target);
  const taker = target.closest<HTMLElement>(TAKES_FOCUS_ON_CLICK);
  if (taker) taker.focus();
  else (document.activeElement as HTMLElement | null)?.blur();
  fireEvent.mouseUp(target);
  fireEvent.click(target);
}

// A browser takes focus off the page while its window is in the background, then hands it back.
export function switchToAnotherAppAndBack(): void {
  const focused = document.activeElement as HTMLElement;
  focused.blur();
  focused.focus();
}

export async function letRequestsGoOut(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

export function holdAnswersTo(path: string): () => void {
  let release = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const answer = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const response = await answer(input, init);
    if (String(input).split("?")[0] === path) await held;
    return response;
  }) as typeof fetch;
  return release;
}
