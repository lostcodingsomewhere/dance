import { useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";

export type BridgeHealth =
  | "checking"
  | "live"
  | "live_not_answering"
  | "server_out_of_date"
  | "server_down";

interface Heartbeat {
  serverIsOutOfDate: boolean;
  liveAnswered: boolean;
  liveAnsweredAt: number | null;
  serverAnsweredAt: number;
}

const HEARTBEAT_KEY = ["bridge", "heartbeat"];
const POLL_MS = 2_000;
const GIVE_UP_AFTER_MS = 4_000;
// Live goes quiet for a few seconds while it makes clips, so one missed answer is not shown.
const HELD_FOR_MS = 5_000;

type LiveCheck = "answered" | "silent" | "server_cannot_say";

async function checkLive(): Promise<LiveCheck> {
  const givingUp = new AbortController();
  const timer = setTimeout(() => givingUp.abort(), GIVE_UP_AFTER_MS);
  try {
    const answer = await api.abletonHealth(givingUp.signal);
    return answer?.live_reachable === true ? "answered" : "silent";
  } catch (error) {
    // A server started before the health route was added answers 404 to it.
    if (error instanceof api.ApiError && error.status === 404) return "server_cannot_say";
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function stillHeld(at: number | null | undefined, asOf: number): boolean {
  return at != null && asOf - at < HELD_FOR_MS;
}

export function useBridgeHeartbeat(): BridgeHealth {
  const queryClient = useQueryClient();
  const beat = useQuery({
    queryKey: HEARTBEAT_KEY,
    queryFn: async (): Promise<Heartbeat> => {
      const check = await checkLive();
      const liveAnswered = check === "answered";
      const now = Date.now();
      const before = queryClient.getQueryData<Heartbeat>(HEARTBEAT_KEY);
      return {
        serverIsOutOfDate: check === "server_cannot_say",
        liveAnswered,
        liveAnsweredAt: liveAnswered ? now : (before?.liveAnsweredAt ?? null),
        serverAnsweredAt: now,
      };
    },
    refetchInterval: POLL_MS,
    staleTime: 1500,
    retry: 0,
    // The server is on this computer, so a browser that is offline can still reach it.
    networkMode: "always",
  });

  if (beat.status === "pending") return "checking";
  if (beat.status === "success" && beat.data.serverIsOutOfDate) return "server_out_of_date";
  // Reading the time of the last failed check redraws the light on every failure, not only the first.
  const lastCheckedAt =
    beat.status === "error" ? beat.errorUpdatedAt : beat.data.serverAnsweredAt;
  if (stillHeld(beat.data?.liveAnsweredAt, lastCheckedAt)) return "live";
  if (beat.status === "success" || stillHeld(beat.data?.serverAnsweredAt, lastCheckedAt)) {
    return "live_not_answering";
  }
  return "server_down";
}
