import { useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";

export type BridgeHealth = "checking" | "live" | "live_not_answering" | "server_down";

interface Heartbeat {
  liveAnswered: boolean;
  liveAnsweredAt: number | null;
  serverAnsweredAt: number;
}

const HEARTBEAT_KEY = ["bridge", "heartbeat"];
const POLL_MS = 2_000;
const GIVE_UP_AFTER_MS = 4_000;
// Live goes quiet for a few seconds while it makes clips, so one missed answer is not shown.
const HELD_FOR_MS = 5_000;

async function liveAnswers(): Promise<boolean> {
  const givingUp = new AbortController();
  const timer = setTimeout(() => givingUp.abort(), GIVE_UP_AFTER_MS);
  try {
    const answer = await api.abletonHealth(givingUp.signal);
    return answer?.live_reachable === true;
  } finally {
    clearTimeout(timer);
  }
}

function stillHeld(at: number | null | undefined): boolean {
  return at != null && Date.now() - at < HELD_FOR_MS;
}

export function useBridgeHeartbeat(): BridgeHealth {
  const queryClient = useQueryClient();
  const beat = useQuery({
    queryKey: HEARTBEAT_KEY,
    queryFn: async (): Promise<Heartbeat> => {
      const liveAnswered = await liveAnswers();
      const now = Date.now();
      const before = queryClient.getQueryData<Heartbeat>(HEARTBEAT_KEY);
      return {
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
  if (beat.status === "success" && beat.data.liveAnswered) return "live";
  if (stillHeld(beat.data?.liveAnsweredAt)) return "live";
  if (beat.status === "success" || stillHeld(beat.data?.serverAnsweredAt)) {
    return "live_not_answering";
  }
  return "server_down";
}
