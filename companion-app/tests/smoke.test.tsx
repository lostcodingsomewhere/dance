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
    expect(screen.getByText(/Dance/i)).toBeInTheDocument();
    expect(screen.getByText(/BPM/)).toBeInTheDocument();
    // All four view tabs are visible.
    expect(screen.getByRole("tab", { name: "Now" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Next" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Library" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Session" })).toBeInTheDocument();
  });

  it("surfaces an API error visibly in the Library view", async () => {
    mockFetch((url) => {
      if (url.includes("/tracks")) {
        return new Response("boom", { status: 500 });
      }
      return jsonResponse({});
    });
    renderApp();
    // Switch to Library tab.
    await act(async () => {
      screen.getByRole("tab", { name: "Library" }).click();
    });
    await waitFor(() => {
      expect(screen.getByText(/Failed to load tracks/i)).toBeInTheDocument();
    });
  });
});
