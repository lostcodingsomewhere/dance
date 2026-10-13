// One WebSocket per channel per app, shared by every hook that reads it.
//
// Before this, each ``useAbletonState()`` call opened its own socket to /ws —
// a dozen components meant a dozen sockets, each delivering the same frame and
// each triggering its own React update. This module owns ONE socket per
// channel at module scope and hands React a ``useSyncExternalStore`` source
// (same pattern as ``src/store.ts``):
//
// - **Ref-counted.** The first subscriber opens the socket; when the last one
//   leaves, it closes after a short linger. The linger absorbs StrictMode's
//   mount→unmount→mount and view switches, where every subscriber unmounts and
//   remounts in the same commit — without it each would tear down and re-dial.
// - **Reconnects** exactly like the old per-hook code: on close/error, retry
//   after a fixed delay (2 s) for as long as anyone is subscribed.
// - **One React update per animation frame.** Every frame from the server is
//   folded into a pending state immediately (``reduce``), but listeners are
//   notified at most once per ``requestAnimationFrame``. A burst of N frames
//   inside one display frame becomes one render showing the newest state.
//   While the page is hidden rAF never fires, so updates are delivered
//   immediately instead — background tabs keep auto-logging plays and
//   clearing deck arms off the live state.

export interface SharedSocketOptions<T, M> {
  /** Path on the current origin, e.g. ``"/ws"``. */
  path: string;
  /** State before the first frame arrives (and after an idle reset). */
  initial: T;
  /** Fold one parsed server frame into the state. Must not mutate ``prev``. */
  reduce: (prev: T, message: M) => T;
  /** Delay before re-dialling after the socket closes. */
  reconnectDelayMs?: number;
  /** How long the socket outlives its last subscriber. */
  idleCloseMs?: number;
  /** Drop accumulated state when the socket closes for lack of subscribers. */
  resetOnIdle?: boolean;
}

export interface SharedSocket<T> {
  subscribe: (listener: () => void) => () => void;
  getSnapshot: () => T;
  getServerSnapshot: () => T;
}

export const DEFAULT_RECONNECT_DELAY_MS = 2000;
export const DEFAULT_IDLE_CLOSE_MS = 1000;

function pageHidden(): boolean {
  return typeof document !== "undefined" && document.visibilityState === "hidden";
}

export function createSharedSocket<T, M = unknown>(
  opts: SharedSocketOptions<T, M>,
): SharedSocket<T> {
  const reconnectDelayMs = opts.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS;
  const idleCloseMs = opts.idleCloseMs ?? DEFAULT_IDLE_CLOSE_MS;

  let state: T = opts.initial;
  // Folded-but-not-yet-published state. ``hasPending`` rather than a null
  // check so a reducer may legitimately produce null/undefined.
  let pending: T = opts.initial;
  let hasPending = false;

  const listeners = new Set<() => void>();
  let socket: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let idleTimer: ReturnType<typeof setTimeout> | null = null;
  let frameHandle: number | null = null;
  let frameIsTimeout = false;
  let visibilityHooked = false;

  function notify(): void {
    for (const l of Array.from(listeners)) l();
  }

  function cancelFrame(): void {
    if (frameHandle == null) return;
    if (frameIsTimeout) clearTimeout(frameHandle);
    else cancelAnimationFrame(frameHandle);
    frameHandle = null;
  }

  function flush(): void {
    cancelFrame();
    if (!hasPending) return;
    state = pending;
    hasPending = false;
    notify();
  }

  function scheduleFlush(): void {
    if (pageHidden()) {
      flush();
      return;
    }
    if (frameHandle != null) return; // already one queued for this frame
    if (typeof requestAnimationFrame === "function") {
      frameIsTimeout = false;
      frameHandle = requestAnimationFrame(() => {
        frameHandle = null;
        flush();
      });
    } else {
      frameIsTimeout = true;
      frameHandle = setTimeout(() => {
        frameHandle = null;
        flush();
      }, 16) as unknown as number;
    }
  }

  function onVisibilityChange(): void {
    // A frame queued just before the tab hid would sit until it's shown
    // again. Deliver it now.
    if (pageHidden()) flush();
  }

  function hookVisibility(): void {
    if (visibilityHooked || typeof document === "undefined") return;
    document.addEventListener("visibilitychange", onVisibilityChange);
    visibilityHooked = true;
  }

  function unhookVisibility(): void {
    if (!visibilityHooked) return;
    document.removeEventListener("visibilitychange", onVisibilityChange);
    visibilityHooked = false;
  }

  function scheduleReconnect(): void {
    if (listeners.size === 0 || reconnectTimer != null) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, reconnectDelayMs);
  }

  function connect(): void {
    if (socket != null || listeners.size === 0) return;
    if (typeof window === "undefined" || typeof WebSocket === "undefined") {
      return;
    }
    const url = new URL(opts.path, window.location.origin);
    url.protocol = url.protocol.replace("http", "ws");
    let ws: WebSocket;
    try {
      ws = new WebSocket(url.toString());
    } catch {
      scheduleReconnect();
      return;
    }
    socket = ws;
    hookVisibility();

    ws.onmessage = (e: MessageEvent) => {
      if (socket !== ws) return; // a frame from a socket we've moved on from
      let message: M;
      try {
        message = JSON.parse(e.data as string) as M;
      } catch {
        return; // ignore malformed frames
      }
      pending = opts.reduce(hasPending ? pending : state, message);
      hasPending = true;
      scheduleFlush();
    };
    ws.onclose = () => {
      if (socket !== ws) return; // closed on purpose (teardown)
      socket = null;
      scheduleReconnect();
    };
    ws.onerror = () => {
      // close() fires onclose, which schedules the reconnect.
      ws.close();
    };
  }

  function teardown(): void {
    if (reconnectTimer != null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    // Publish whatever arrived so a later subscriber starts from the newest
    // state (unless this channel resets on idle).
    flush();
    unhookVisibility();
    const ws = socket;
    socket = null; // before close(), so its onclose doesn't schedule a retry
    ws?.close();
    if (opts.resetOnIdle) {
      state = opts.initial;
      pending = opts.initial;
      hasPending = false;
    }
  }

  function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    if (idleTimer != null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
    // If a reconnect is already pending, let its timer dial.
    if (reconnectTimer == null) connect();
    return () => {
      if (!listeners.delete(listener)) return;
      if (listeners.size > 0 || idleTimer != null) return;
      idleTimer = setTimeout(() => {
        idleTimer = null;
        if (listeners.size === 0) teardown();
      }, idleCloseMs);
    };
  }

  return {
    subscribe,
    getSnapshot: () => state,
    getServerSnapshot: () => opts.initial,
  };
}
