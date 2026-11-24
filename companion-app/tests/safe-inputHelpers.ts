import { act, fireEvent } from "@testing-library/react";

// jsdom sends no key to the focused control by itself, and never turns Enter or Space into a click.
export function press(key: string, init: KeyboardEventInit = {}): void {
  const focused = (document.activeElement ?? document.body) as HTMLElement;
  const leftToTheBrowser = fireEvent.keyDown(focused, { key, ...init });
  const activates = key === "Enter" || key === " ";
  if (leftToTheBrowser && activates && focused.matches("button")) fireEvent.click(focused);
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
