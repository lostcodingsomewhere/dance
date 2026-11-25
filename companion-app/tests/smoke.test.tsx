import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";

function renderApp() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <App />
    </QueryClientProvider>,
  );
}

const originalFetch = globalThis.fetch;

function mockFetch(impl: (url: string, init?: RequestInit) => Response) {
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    return impl(url, init);
  }) as unknown as typeof fetch;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const LIVE_TRACK_INDEX_BY_DECK_KIND = {
  drums_a: 0, bass_a: 1, vocals_a: 2, other_a: 3,
  drums_b: 4, bass_b: 5, vocals_b: 6, other_b: 7,
  mix_a: 8, mix_b: 9,
};

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("App smoke", () => {
  beforeEach(() => {
    mockFetch((url) => {
      if (url.includes("/sessions/current")) {
        return jsonResponse({ detail: "no active session" }, 404);
      }
      if (url.includes("/tracks")) {
        return jsonResponse([]);
      }
      return jsonResponse({});
    });
  });

  it("renders the top bar without crashing", async () => {
    renderApp();
    expect(screen.getByText(/BPM/)).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Booth" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Set" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Pipeline" })).toBeInTheDocument();
  });

  it("booth headers put deck A on faders 1 to 4 and deck B on 5 to 8", async () => {
    mockFetch((url) => {
      if (url.includes("/ableton/decks")) {
        return jsonResponse({ columns: LIVE_TRACK_INDEX_BY_DECK_KIND, cells: [] });
      }
      if (url.includes("/sessions/current")) {
        return jsonResponse({ detail: "no active session" }, 404);
      }
      if (url.includes("/tracks")) {
        return jsonResponse([]);
      }
      return jsonResponse({});
    });
    renderApp();
    const headers = await screen.findByTestId("booth-column-headers");
    const tooltips = Array.from(headers.children).map(
      (chip) => chip.getAttribute("title")?.split(" — ")[0],
    );
    expect(tooltips).toEqual([
      "Drums A · APC40 fader 1",
      "Bass A · APC40 fader 2",
      "Vocals A · APC40 fader 3",
      "Melody A · APC40 fader 4",
      "Drums B · APC40 fader 5",
      "Bass B · APC40 fader 6",
      "Vocals B · APC40 fader 7",
      "Melody B · APC40 fader 8",
    ]);
  });

  it("set editor renders the empty-state when no active set exists", async () => {
    mockFetch((url) => {
      if (url.includes("/sets/active")) {
        return jsonResponse({ detail: "no active set" }, 404);
      }
      if (url.includes("/sessions/current")) {
        return jsonResponse({ detail: "no active session" }, 404);
      }
      return jsonResponse([]);
    });
    renderApp();
    await act(async () => {
      screen.getByRole("tab", { name: "Set" }).click();
    });
    await waitFor(() => {
      expect(
        screen.getByText(/No active set yet/i),
      ).toBeInTheDocument();
    });
  });
});
