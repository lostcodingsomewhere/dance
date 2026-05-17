import { useEffect, useRef, useState } from "react";
import type { Job } from "../types";

/** Subscribes to /ws/pipeline. Keeps a job-id → Job map updated in real time.
 *
 * Replaces polling for download jobs. On connect, the backend replays the
 * last few jobs so the UI can render immediately without a separate REST
 * round-trip. After that, each job mutation (item status flip, job-status
 * transition) arrives as a fresh full-job snapshot. */
export function usePipelineJobs(): Record<string, Job> {
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (typeof window === "undefined" || typeof WebSocket === "undefined") {
      return;
    }
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (cancelled) return;
      const url = new URL("/ws/pipeline", window.location.origin);
      url.protocol = url.protocol.replace("http", "ws");
      const ws = new WebSocket(url.toString());
      wsRef.current = ws;

      ws.onmessage = (e) => {
        try {
          const parsed = JSON.parse(e.data) as Job;
          setJobs((prev) => ({ ...prev, [parsed.id]: parsed }));
        } catch {
          // ignore malformed frames
        }
      };
      ws.onclose = () => {
        if (cancelled) return;
        reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => {
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

  return jobs;
}
