import { useMutation } from "@tanstack/react-query";
import * as api from "../api";

export function useExportSetToLive() {
  return useMutation({ mutationFn: (setId: number) => api.exportSetAls(setId) });
}

export function useOpenInLive() {
  return useMutation({ mutationFn: (path: string) => api.openInLive(path) });
}

export function useRevealInFinder() {
  return useMutation({ mutationFn: (path: string) => api.revealPath(path) });
}
