import { useSyncExternalStore } from "react";
import { createSharedSocket } from "../lib/sharedSocket";
import type { Job } from "../types";

const NO_JOBS: Record<string, Job> = {};

/** The one /ws/pipeline connection. Frames are per-job snapshots, so they
 * are merged into the map (not replaced) — coalescing a burst into one
 * render never loses another job's update. ``resetOnIdle`` keeps the old
 * per-mount semantics: leave the Pipeline view and the map is rebuilt from
 * the server's replay next time, rather than showing a stale one. */
const pipelineJobsSocket = createSharedSocket<Record<string, Job>, Job>({
  path: "/ws/pipeline",
  initial: NO_JOBS,
  reduce: (prev, job) => ({ ...prev, [job.id]: job }),
  resetOnIdle: true,
});

/** Subscribes to /ws/pipeline. Keeps a job-id → Job map updated in real time.
 *
 * Replaces polling for download jobs. On connect, the backend replays the
 * last few jobs so the UI can render immediately without a separate REST
 * round-trip. After that, each job mutation (item status flip, job-status
 * transition) arrives as a fresh full-job snapshot. */
export function usePipelineJobs(): Record<string, Job> {
  return useSyncExternalStore(
    pipelineJobsSocket.subscribe,
    pipelineJobsSocket.getSnapshot,
    pipelineJobsSocket.getServerSnapshot,
  );
}
