import { StrictMode } from "react";
import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AbletonState, Job } from "../src/types";
import { EMPTY_ABLETON_STATE } from "../src/types";

/**
 * One WebSocket per channel for the whole app, and at most one React update
 * per animation frame.
 *
 * ``useAbletonState`` used to open its own socket per call — a dozen
 * components meant a dozen sockets carrying identical frames, each frame
 * re-rendering every one of them separately. These tests drive the real hooks
 * against a controllable WebSocket and a hand-cranked requestAnimationFrame.
 */

// ---------------------------------------------------------------------------
// Controllable WebSocket + animation frames
// ---------------------------------------------------------------------------

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  closed = false;

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }

  /** Client-side close — like a browser, fires onclose (once). */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.onclose?.({} as CloseEvent);
  }

  /** The server pushes one frame. */
  push(payload: unknown): void {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }

  pushRaw(data: string): void {
    this.onmessage?.({ data } as MessageEvent);
  }

  /** The server (or network) drops the connection. */
  drop(): void {
    this.close();
  }

  fail(): void {
    this.onerror?.({} as Event);
  }
}

let frameQueue = new Map<number, FrameRequestCallback>();
let nextFrameId = 1;

function runFrame(): void {
  const due = frameQueue;
  frameQueue = new Map();
  act(() => {
    for (const cb of due.values()) cb(performance.now());
  });
}

function sockets(path: string): MockWebSocket[] {
  return MockWebSocket.instances.filter((s) => new URL(s.url).pathname === path);
}

function openSockets(path: string): MockWebSocket[] {
  return sockets(path).filter((s) => !s.closed);
}

function setHidden(hidden: boolean): void {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => (hidden ? "hidden" : "visible"),
  });
  document.dispatchEvent(new Event("visibilitychange"));
}

function ableton(changes: Partial<AbletonState>): AbletonState {
  return { ...EMPTY_ABLETON_STATE, ...changes };
}

// A fresh module graph per test, so each test starts with no socket and an
// empty store — exactly like a page load.
async function loadHooks() {
  vi.resetModules();
  const { useAbletonState } = await import("../src/hooks/useAbletonState");
  const { usePipelineJobs } = await import("../src/hooks/usePipelineJobs");
  return { useAbletonState, usePipelineJobs };
}

beforeEach(() => {
  MockWebSocket.instances = [];
  frameQueue = new Map();
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  vi.stubGlobal("WebSocket", MockWebSocket);
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    const id = nextFrameId++;
    frameQueue.set(id, cb);
    return id;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    frameQueue.delete(id);
  });
  setHidden(false);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  // Drop the instance-level override so document falls back to jsdom's.
  delete (document as unknown as Record<string, unknown>).visibilityState;
});

// ---------------------------------------------------------------------------

describe("useAbletonState shares one socket", () => {
  it("N components → exactly one WebSocket, and every one gets the update", async () => {
    const { useAbletonState } = await loadHooks();
    const N = 12;
    function Tempo({ i }: { i: number }) {
      const s = useAbletonState();
      return <span data-testid={`t${i}`}>{String(s.tempo)}</span>;
    }
    render(
      <>
        {Array.from({ length: N }, (_, i) => (
          <Tempo key={i} i={i} />
        ))}
      </>,
    );

    expect(MockWebSocket.instances).toHaveLength(1);
    const ws = MockWebSocket.instances[0];
    expect(new URL(ws.url).pathname).toBe("/ws");
    expect(new URL(ws.url).protocol).toBe("ws:");

    act(() => ws.push(ableton({ tempo: 128 })));
    runFrame();
    for (let i = 0; i < N; i++) {
      expect(screen.getByTestId(`t${i}`)).toHaveTextContent("128");
    }
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it("a component mounted later starts from the current state, not empty", async () => {
    const { useAbletonState } = await loadHooks();
    function Tempo({ id }: { id: string }) {
      return <span data-testid={id}>{String(useAbletonState().tempo)}</span>;
    }
    const { rerender } = render(<Tempo id="first" />);
    act(() => MockWebSocket.instances[0].push(ableton({ tempo: 124 })));
    runFrame();
    rerender(
      <>
        <Tempo id="first" />
        <Tempo id="late" />
      </>,
    );
    expect(screen.getByTestId("late")).toHaveTextContent("124");
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it("StrictMode's mount→unmount→mount does not open a second socket", async () => {
    const { useAbletonState } = await loadHooks();
    function Tempo() {
      return <span>{String(useAbletonState().tempo)}</span>;
    }
    render(
      <StrictMode>
        <Tempo />
        <Tempo />
      </StrictMode>,
    );
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.instances[0].closed).toBe(false);
  });
});

describe("one React update per animation frame", () => {
  it("collapses a burst into a single render showing the newest state", async () => {
    const { useAbletonState } = await loadHooks();
    const renders = [0, 0, 0];
    function Probe({ i }: { i: number }) {
      const s = useAbletonState();
      renders[i]++;
      return <span data-testid={`p${i}`}>{String(s.beat)}</span>;
    }
    render(
      <>
        <Probe i={0} />
        <Probe i={1} />
        <Probe i={2} />
      </>,
    );
    const before = [...renders];
    const ws = MockWebSocket.instances[0];

    // 50 frames inside one display frame — e.g. clip positions + meters.
    act(() => {
      for (let b = 0; b < 50; b++) ws.push(ableton({ beat: b }));
    });
    // Nothing rendered yet, and only one frame was requested for the burst.
    expect(renders).toEqual(before);
    expect(frameQueue.size).toBe(1);

    runFrame();
    expect(renders).toEqual(before.map((n) => n + 1));
    for (let i = 0; i < 3; i++) {
      expect(screen.getByTestId(`p${i}`)).toHaveTextContent("49");
    }

    // The next burst gets its own frame — latest state always delivered.
    act(() => {
      ws.push(ableton({ beat: 50 }));
      ws.push(ableton({ beat: 51 }));
    });
    runFrame();
    expect(renders).toEqual(before.map((n) => n + 2));
    expect(screen.getByTestId("p0")).toHaveTextContent("51");

    // A frame with nothing new renders nothing.
    runFrame();
    expect(renders).toEqual(before.map((n) => n + 2));
  });

  it("ignores malformed frames without losing the good ones around them", async () => {
    const { useAbletonState } = await loadHooks();
    function Tempo() {
      return <span data-testid="t">{String(useAbletonState().tempo)}</span>;
    }
    render(<Tempo />);
    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.push(ableton({ tempo: 126 }));
      ws.pushRaw("{not json");
    });
    runFrame();
    expect(screen.getByTestId("t")).toHaveTextContent("126");
  });

  it("delivers immediately while the page is hidden (rAF is paused there)", async () => {
    const { useAbletonState } = await loadHooks();
    function Playing() {
      return <span data-testid="p">{String(useAbletonState().is_playing)}</span>;
    }
    render(<Playing />);
    const ws = MockWebSocket.instances[0];

    // A frame queued while visible, then the tab hides before it fires.
    act(() => ws.push(ableton({ is_playing: true })));
    expect(screen.getByTestId("p")).toHaveTextContent("null");
    act(() => setHidden(true));
    expect(screen.getByTestId("p")).toHaveTextContent("true");

    // While hidden, no frame is waited on at all.
    act(() => ws.push(ableton({ is_playing: false })));
    expect(screen.getByTestId("p")).toHaveTextContent("false");
    expect(frameQueue.size).toBe(0);
  });
});

describe("connection lifecycle", () => {
  it("reconnects 2 s after the server drops, and reads the new socket", async () => {
    const { useAbletonState } = await loadHooks();
    function Tempo() {
      return <span data-testid="t">{String(useAbletonState().tempo)}</span>;
    }
    render(
      <>
        <Tempo />
        <Tempo />
      </>,
    );
    const first = MockWebSocket.instances[0];
    act(() => first.push(ableton({ tempo: 120 })));
    runFrame();

    act(() => first.drop());
    expect(MockWebSocket.instances).toHaveLength(1);
    act(() => {
      vi.advanceTimersByTime(1999);
    });
    expect(MockWebSocket.instances).toHaveLength(1);
    act(() => {
      vi.advanceTimersByTime(1);
    });
    // Exactly one replacement for the whole app, not one per component.
    expect(MockWebSocket.instances).toHaveLength(2);
    const second = MockWebSocket.instances[1];

    // The last known state survives the gap.
    expect(screen.getAllByTestId("t")[0]).toHaveTextContent("120");
    act(() => second.push(ableton({ tempo: 132 })));
    runFrame();
    for (const el of screen.getAllByTestId("t")) expect(el).toHaveTextContent("132");

    // A late frame from the dead socket can't overwrite the live one.
    act(() => first.push(ableton({ tempo: 90 })));
    runFrame();
    expect(screen.getAllByTestId("t")[0]).toHaveTextContent("132");
  });

  it("an error closes the socket and the same backoff applies", async () => {
    const { useAbletonState } = await loadHooks();
    function Tempo() {
      return <span>{String(useAbletonState().tempo)}</span>;
    }
    render(<Tempo />);
    act(() => MockWebSocket.instances[0].fail());
    expect(MockWebSocket.instances[0].closed).toBe(true);
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(MockWebSocket.instances).toHaveLength(2);
    expect(MockWebSocket.instances[1].closed).toBe(false);
  });

  it("closes after the last subscriber leaves (with a linger) and never redials", async () => {
    const { useAbletonState } = await loadHooks();
    function Tempo() {
      return <span>{String(useAbletonState().tempo)}</span>;
    }
    const a = render(<Tempo />);
    const b = render(<Tempo />);
    const ws = MockWebSocket.instances[0];

    a.unmount();
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(ws.closed).toBe(false); // b still holds a reference

    b.unmount();
    act(() => {
      vi.advanceTimersByTime(999);
    });
    expect(ws.closed).toBe(false); // lingering
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(ws.closed).toBe(true);

    // Closed on purpose: no reconnect loop with nobody listening.
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(MockWebSocket.instances).toHaveLength(1);

    // A new subscriber dials a fresh socket.
    render(<Tempo />);
    expect(MockWebSocket.instances).toHaveLength(2);
    expect(openSockets("/ws")).toHaveLength(1);
  });

  it("a subscriber returning inside the linger keeps the same socket", async () => {
    const { useAbletonState } = await loadHooks();
    function Tempo() {
      return <span>{String(useAbletonState().tempo)}</span>;
    }
    const a = render(<Tempo />);
    a.unmount();
    act(() => {
      vi.advanceTimersByTime(500);
    });
    render(<Tempo />);
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.instances[0].closed).toBe(false);
  });

  it("does not dial a second socket when a subscriber arrives mid-backoff", async () => {
    const { useAbletonState } = await loadHooks();
    function Tempo() {
      return <span>{String(useAbletonState().tempo)}</span>;
    }
    render(<Tempo />);
    act(() => MockWebSocket.instances[0].drop());
    render(<Tempo />); // mounts during the 2 s wait
    expect(MockWebSocket.instances).toHaveLength(1);
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(MockWebSocket.instances).toHaveLength(2);
    expect(openSockets("/ws")).toHaveLength(1);
  });
});

describe("usePipelineJobs shares the same machinery", () => {
  function job(id: string, status: Job["status"]): Job {
    return { id, status } as unknown as Job;
  }

  it("one /ws/pipeline socket; a burst merges every job into one render", async () => {
    const { usePipelineJobs } = await loadHooks();
    let renders = 0;
    function Jobs() {
      const jobs = usePipelineJobs();
      renders++;
      return (
        <span data-testid="jobs">
          {Object.values(jobs)
            .map((j) => `${j.id}:${j.status}`)
            .sort()
            .join(",")}
        </span>
      );
    }
    render(
      <>
        <Jobs />
        <Jobs />
      </>,
    );
    expect(sockets("/ws/pipeline")).toHaveLength(1);
    const ws = sockets("/ws/pipeline")[0];
    const before = renders;

    act(() => {
      ws.push(job("a", "running"));
      ws.push(job("b", "queued"));
      ws.push(job("a", "done"));
    });
    runFrame();
    expect(renders).toBe(before + 2); // two components, one render each
    for (const el of screen.getAllByTestId("jobs")) {
      expect(el).toHaveTextContent("a:done,b:queued");
    }
  });

  it("starts fresh after the view is left (server replays on reconnect)", async () => {
    const { usePipelineJobs } = await loadHooks();
    function Jobs() {
      return <span data-testid="jobs">{Object.keys(usePipelineJobs()).join(",")}</span>;
    }
    const view = render(<Jobs />);
    act(() => sockets("/ws/pipeline")[0].push(job("old", "done")));
    runFrame();
    expect(screen.getByTestId("jobs")).toHaveTextContent("old");

    view.unmount();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    render(<Jobs />);
    expect(sockets("/ws/pipeline")).toHaveLength(2);
    expect(screen.getByTestId("jobs")).toHaveTextContent("");
  });
});

