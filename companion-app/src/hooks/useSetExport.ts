import { useMutation } from "@tanstack/react-query";
import * as api from "../api";

export interface SetSentToLive {
  exported: api.SetAlsExportResult;
  openedIn: string | null;
  openFailure: Error | null;
}

async function writeSetAndOpenIt(setId: number): Promise<SetSentToLive> {
  const exported = await api.writeSetAsNewLiveSet(setId);
  try {
    const opened = await api.openInLive(exported.out_path);
    return { exported, openedIn: opened.app, openFailure: null };
  } catch (error) {
    const openFailure = error instanceof Error ? error : new Error(String(error));
    return { exported, openedIn: null, openFailure };
  }
}

export function useOpenSetInLive() {
  return useMutation({ mutationFn: writeSetAndOpenIt, meta: { showsOwnFailure: true } });
}

export function useRevealInFinder() {
  return useMutation({
    mutationFn: (path: string) => api.revealPath(path),
    meta: { attempt: "show the file in Finder" },
  });
}
