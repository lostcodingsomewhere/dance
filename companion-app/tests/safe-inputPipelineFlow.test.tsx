import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { store } from "../src/store";
import { BoothBackend, renderApp, type SeenRequest } from "./boothHarness";
import { press, tabTo } from "./safe-inputHelpers";

const ONE_SONG_CSV = "Track Name,Artist Name(s)\nNight Drive,Test Artist";

function anIdlePipeline({ path }: SeenRequest): unknown {
  if (path === "/api/v1/pipeline/status") {
    return {
      counts: {},
      total: 0,
      in_progress: false,
      errors: 0,
      complete: 0,
      weighted_progress: 0,
    };
  }
  if (path === "/api/v1/pipeline/recent") return [];
  return undefined;
}

// The tests load no stylesheet, so this stands in for the one rule that hides the file picker.
function hideWhatTheStylesheetHides(): HTMLStyleElement {
  const style = document.createElement("style");
  style.textContent = ".hidden { display: none; }";
  document.head.appendChild(style);
  return style;
}

let backend: BoothBackend;
let stylesheet: HTMLStyleElement;

beforeEach(() => {
  store.closeCommandBar();
  store.setView("pipeline");
  backend = new BoothBackend(anIdlePipeline);
  stylesheet = hideWhatTheStylesheetHides();
});

afterEach(() => {
  stylesheet.remove();
  store.setView("booth");
  backend.restore();
});

describe("the Ingest from CSV dialog in the Pipeline view", () => {
  it("keeps Tab among the controls that can take focus", async () => {
    renderApp();
    const ingest = await screen.findByRole("button", { name: "Ingest CSV…" });
    tabTo(ingest);
    press("Enter");

    const dialog = await screen.findByRole("dialog", { name: "Ingest from CSV" });
    const close = within(dialog).getByRole("button", { name: "Close" });
    const csv = within(dialog).getByRole("textbox");
    const preview = within(dialog).getByRole("button", { name: "Preview" });
    expect(close).toHaveFocus();
    expect(preview).toBeDisabled();

    const cancel = within(dialog).getByRole("button", { name: "Cancel" });
    cancel.focus();
    press("Tab");
    expect(close).toHaveFocus();
    press("Tab", { shiftKey: true });
    expect(cancel).toHaveFocus();

    fireEvent.change(csv, { target: { value: ONE_SONG_CSV } });
    expect(preview).toBeEnabled();
    cancel.focus();
    press("Tab");
    expect(close).toHaveFocus();

    press("Escape");
    expect(screen.getByRole("dialog", { name: "Ingest from CSV" })).toBeInTheDocument();
    fireEvent.click(cancel);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(ingest).toHaveFocus();
  });
});
