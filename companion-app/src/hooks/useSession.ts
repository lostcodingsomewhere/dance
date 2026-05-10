import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import * as api from "../api";
import { store } from "../store";
import type { AddPlayBody } from "../types";

export function useCurrentSession() {
  return useQuery({
    queryKey: ["session", "current"],
    queryFn: api.currentSession,
    staleTime: 5_000,
  });
}

export function useCreateSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name?: string) => api.createSession(name),
    onSuccess: (session) => {
      store.setSessionId(session.id);
      qc.invalidateQueries({ queryKey: ["session"] });
    },
  });
}

export function useAddPlay(sessionId: number | null | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: AddPlayBody) => {
      if (sessionId == null) {
        throw new Error("No active session — start one first.");
      }
      return api.addPlay(sessionId, body);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["session"] });
    },
  });
}

export function useEndSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: number) => api.endSession(sessionId),
    onSuccess: () => {
      store.setSessionId(null);
      qc.invalidateQueries({ queryKey: ["session"] });
    },
  });
}
