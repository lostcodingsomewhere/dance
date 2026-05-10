import { useEffect, useRef, useState } from "react";
import type { AbletonState } from "../types";
import { EMPTY_ABLETON_STATE } from "../types";

/** Subscribes to /ws and returns the latest Ableton state pushed by the bridge. */
export function useAbletonState(): AbletonState {
  const [state, setState] = useState<AbletonState>(EMPTY_ABLETON_STATE);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (typeof window === "undefined" || typeof WebSocket === "undefined") {
      return;
    }
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (cancelled) return;
      const url = new URL("/ws", window.location.origin);
      url.protocol = url.protocol.replace("http", "ws");
      const ws = new WebSocket(url.toString());
      wsRef.current = ws;

      ws.onmessage = (e) => {
        try {
          const parsed = JSON.parse(e.data) as AbletonState;
          setState(parsed);
        } catch {
          // ignore malformed frames
        }
      };
      ws.onclose = () => {
        if (cancelled) return;
        // Reconnect with a small backoff
        reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => {
        // close() will fire and trigger reconnect.
        ws.close();
      };
    }

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, []);

  return state;
}
